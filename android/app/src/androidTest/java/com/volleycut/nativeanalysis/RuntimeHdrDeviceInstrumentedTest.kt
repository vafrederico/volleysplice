package com.volleycut.nativeanalysis
import com.volleycut.video.AnalysisVideoDecoder

import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.util.Log
import androidx.annotation.OptIn
import androidx.media3.common.MediaItem
import androidx.media3.common.util.UnstableApi
import androidx.media3.transformer.*
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import java.util.function.BooleanSupplier

/** Opt-in physical-device checks. Private sources are passed only as runtime arguments. */
@OptIn(UnstableApi::class)
@RunWith(AndroidJUnit4::class)
class RuntimeHdrDeviceInstrumentedTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext
    private val args = InstrumentationRegistry.getArguments()
    private val outputs: File get() = checkNotNull(context.getExternalFilesDir("hdr-validation"))
    private val durationMs: Long get() = args.getString("sampleDurationMs", "60000")!!.toLong()

    @Test fun runtimeHdrAnalysis() = analyze("hdr")
    @Test fun runtimeSdrAnalysis() = analyze("sdr")
    @Test fun runtimeHdrExportsWithScores() = export("hdr", true)
    @Test fun runtimeHdrExportsWithoutScores() = export("hdr", false)
    @Test fun runtimeSdrExportsWithScores() = export("sdr", true)
    @Test fun runtimeSdrExportsWithoutScores() = export("sdr", false)

    private fun source(kind: String): Uri {
        val path = args.getString("${kind}Input")
        assumeTrue("Supply a runtime $kind input", !path.isNullOrBlank())
        val file = File(checkNotNull(path))
        assertTrue("Runtime input is missing or unreadable", file.isFile && file.canRead())
        return Uri.fromFile(file)
    }

    private fun analyze(kind: String) {
        assertTrue(OpenCVLoader.initLocal())
        val uri = source(kind)
        val engine = AnalysisEngine(context)
        val media = engine.probe(uri)
        val seconds = minOf(durationMs / 1000.0, media.durationSeconds())
        val extractor = MediaExtractor()
        val frameRate = try {
            extractor.setDataSource(context, uri, null)
            extractor.getTrackFormat(NativeVideoDecoder.findTrack(extractor, "video/"))
                .getInteger(MediaFormat.KEY_FRAME_RATE, 30)
        } finally { extractor.release() }
        // Raw decoding uses a source-frame cap independently of the model window.
        // Limit both when the runtime source is a full recording already on the phone.
        val frameLimit = if (media.durationSeconds() <= seconds + 1e-6) Int.MAX_VALUE
            else kotlin.math.ceil(seconds * frameRate).toInt() + 1
        val started = System.nanoTime()
        val progress = AnalysisTypes.ProgressListener { stage, fraction, _ ->
            if (fraction >= 1) Log.i(TAG, "$kind analysis stage=$stage complete")
        }
        val roi = AnalysisTypes.Roi(0.0, 0.0, 1.0, 1.0, "Full frame")
        val cancelled = AtomicBoolean(false)
        var probabilities = floatArrayOf()
        val result = checkNotNull(DistilledRallyModels.open(context, RallyModels.RECALL,
            media, roi, cancelled::get)).use { neural ->
            val result = AnalysisEngine(context, neural).analyze(uri, frameLimit,
                AnalysisTypes.VideoDecoderOptions.defaults(), NativeFeatureCache.Mode.REFRESH,
                AnalysisTypes.AnalysisWindow(0.0, seconds), AnalysisTypes.AnalysisStages.all(),
                AnalysisTypes.AudioDecoderMode.AUTO, cancelled, progress, false, false)
            probabilities = checkNotNull(neural.scores()).probabilities.copyOf()
            assertTrue(probabilities.isNotEmpty() && probabilities.all { it.isFinite() })
            if (frameLimit == Int.MAX_VALUE) assertTrue(neural.report.getBoolean("sharedDecoding"))
            result
        }
        // A cache hit skips the shared video decoder and uses the independent
        // neural decoder. Both paths must produce the same model probabilities.
        checkNotNull(DistilledRallyModels.open(context, RallyModels.RECALL,
            media, roi, cancelled::get)).use { neural ->
            val cached = AnalysisEngine(context, neural).analyze(uri, frameLimit,
                AnalysisTypes.VideoDecoderOptions.defaults(), NativeFeatureCache.Mode.USE,
                AnalysisTypes.AnalysisWindow(0.0, seconds), AnalysisTypes.AnalysisStages.all(),
                AnalysisTypes.AudioDecoderMode.AUTO, cancelled, progress, false, false)
            assertFalse(neural.report.optBoolean("sharedDecoding", false))
            assertEquals(result.ranges(), cached.ranges())
            assertArrayEquals(probabilities, checkNotNull(neural.scores()).probabilities, 1e-5f)
        }
        assertTrue("Analysis ended early", result.sampleRows() >= (seconds * 4).toInt() - 2)
        assertTrue(result.decodedSourceFrames() > 0)
        // Exercise the independent score-feature decoder, including its end-of-window sample.
        val times = doubleArrayOf(0.0, seconds / 2, seconds - 0.25)
        val specialist = SpecialistFrameDecoder(context).decode(uri, media,
            AnalysisTypes.Roi(0.0, 0.0, 1.0, 1.0, "Full frame"), times, times,
            progress, BooleanSupplier { false })
        assertEquals(times.size, specialist.servingGray().size)
        assertEquals(times.size, specialist.sideSwitchBgr().size)
        // The synchronous control must select the same source frames as the production path.
        val control = NativeVideoDecoder(context).decodeSynchronous(uri, media,
            AnalysisTypes.Roi(0.0, 0.0, 1.0, 1.0, "Full frame"),
            doubleArrayOf(0.0, 0.25, 0.5, 0.75), frameRate + 1,
            AnalysisTypes.VideoDecoderOptions.defaults(), progress, BooleanSupplier { false })
        assertTrue(control.values().all { it.isFinite() })
        receipt("$kind-analysis", JSONObject().apply {
            put("sampleRows", result.sampleRows())
            put("decoder", result.videoDecoder())
            put("hardwareDecoder", result.hardwareVideoDecoder())
            put("analyzedDurationSeconds", result.analyzedDurationSeconds())
            put("specialistSamples", specialist.servingGray().size)
            put("modelId", RallyModels.RECALL)
            put("neuralProbabilityRows", probabilities.size)
            put("cachedIndependentPredictionsMatch", true)
            put("wallSeconds", (System.nanoTime() - started) / 1e9)
        })
    }

    private fun export(kind: String, scores: Boolean) {
        val uri = source(kind)
        val sourcePolicy = ExportColorPolicy.read(context, uri)
        assertEquals(kind == "hdr", sourcePolicy.isHdr)
        val targetPolicy = sourcePolicy
        if (targetPolicy.isHdr) targetPolicy.requireHdrEncoder()
        val output = outputs.resolve("$kind-${if (scores) "scores" else "control"}.mp4")
        output.delete()
        val item = MediaItem.Builder().setUri(uri).setClippingConfiguration(
            MediaItem.ClippingConfiguration.Builder().setEndPositionMs(durationMs).build()).build()
        val snapshot = ScoreExportSnapshot(render = scores, renderPointTimeline = false,
            scoreTracking = ScoreTracking(team1Name = "Falcons", team2Name = "Wolves"),
            ignoredIntervals = emptyList(), excludedRallyIds = emptySet(), rallyRanges = emptyList())
        val edited = EditedMediaItem.Builder(item).setDurationUs(durationMs * 1000).apply {
            if (scores) setEffects(Effects(emptyList(), listOf(ScoreOverlayEffect(
                ScoreCanvasOverlay(snapshot, 0, 0), targetPolicy.color.colorTransfer))))
        }.build()
        val composition = Composition.Builder(EditedMediaItemSequence.withAudioAndVideoFrom(listOf(edited)))
            .setHdrMode(Composition.HDR_MODE_KEEP_HDR).build()
        val done = CountDownLatch(1)
        val failure = AtomicReference<ExportException?>()
        var exportedResult: ExportResult? = null
        var transformer: Transformer? = null
        val started = System.nanoTime()
        instrumentation.runOnMainSync {
            transformer = Transformer.Builder(context)
                .setVideoMimeType(targetPolicy.outputMimeType)
                .setMuxerFactory(targetPolicy.preservingMuxer(InAppMp4Muxer.Factory()))
                .addListener(object : Transformer.Listener {
                    override fun onCompleted(composition: Composition, exportResult: ExportResult) {
                        exportedResult = exportResult
                        done.countDown()
                    }
                    override fun onError(composition: Composition, exportResult: ExportResult,
                        exportException: ExportException) {
                        failure.set(exportException)
                        done.countDown()
                    }
                }).build().also { it.start(composition, output.absolutePath) }
        }
        try {
            assertTrue("Runtime export timed out", done.await(300, TimeUnit.SECONDS))
            assertNull(failure.get()?.stackTraceToString(), failure.get())
            assertTrue(output.length() > 0)
            val color = ExportColorPolicy.read(context, Uri.fromFile(output)).color
            assertTrue("Encoded color characteristics differ", targetPolicy.matches(color))
            val extractor = MediaExtractor()
            try {
                extractor.setDataSource(output.absolutePath)
                val track = NativeVideoDecoder.findTrack(extractor, "video/")
                val format = extractor.getTrackFormat(track)
                val actualMime = format.getString(MediaFormat.KEY_MIME)
                // With no pixel changes, Transformer may transmux the original
                // Dolby Vision bitstream and retain its valid dynamic metadata.
                if (targetPolicy.isHdr && !scores &&
                    actualMime == MediaFormat.MIMETYPE_VIDEO_DOLBY_VISION) {
                    assertTrue(AnalysisVideoDecoder.usesHlgBaseLayer(actualMime,
                        format.getInteger(MediaFormat.KEY_PROFILE, 0),
                        format.getInteger(MediaFormat.KEY_COLOR_STANDARD, 0),
                        format.getInteger(MediaFormat.KEY_COLOR_TRANSFER, 0)))
                } else assertEquals(targetPolicy.outputMimeType, actualMime)
                assertTrue("Encoded video ended early", format.getLong(MediaFormat.KEY_DURATION)
                    >= durationMs * 1000 - 100_000)
                receipt("$kind-${if (scores) "scores" else "control"}", JSONObject().apply {
                    put("bytes", output.length())
                    put("durationUs", format.getLong(MediaFormat.KEY_DURATION))
                    put("colorSpace", color.colorSpace)
                    put("colorTransfer", color.colorTransfer)
                    put("colorRange", color.colorRange)
                    put("mime", format.getString(MediaFormat.KEY_MIME))
                    put("encoder", exportedResult?.videoEncoderName)
                    put("wallSeconds", (System.nanoTime() - started) / 1e9)
                })
            } finally { extractor.release() }
        } finally {
            instrumentation.runOnMainSync { transformer?.cancel() }
        }
    }

    private fun receipt(name: String, value: JSONObject) {
        value.put("sourceIndex", args.getString("${name.substringBefore('-')}SourceIndex", "runtime-input"))
        value.put("variant", "preserve-hdr")
        outputs.resolve("$name.json").writeText(value.toString(2))
        Log.i(TAG, "$name $value")
    }

    companion object { private const val TAG = "RuntimeHdrDeviceTest" }
}
