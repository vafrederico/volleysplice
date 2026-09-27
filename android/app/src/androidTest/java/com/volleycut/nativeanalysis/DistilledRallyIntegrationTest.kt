package com.volleycut.nativeanalysis

import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMuxer
import android.net.Uri
import android.util.Base64
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import org.junit.Assume.assumeTrue
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import java.io.File
import java.nio.ByteBuffer
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicBoolean
import org.json.JSONObject
import org.json.JSONArray

/** Loads the actual production APK bundles, not benchmark-side model injection. */
@RunWith(AndroidJUnit4::class)
class DistilledRallyIntegrationTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext
    private val progress = AnalysisTypes.ProgressListener { _, _, _ -> }
    private val roi = AnalysisTypes.Roi(0.0, 0.0, 1.0, 1.0, "Full frame")

    private data class Output(val result: AnalysisTypes.AnalysisResult, val tokens: String, val metadata: String,
                              val scores: NeuralRallyScores)

    private fun run(file: File, model: String, shared: Boolean, window: AnalysisTypes.AnalysisWindow,
                    cacheMode: NativeFeatureCache.Mode = NativeFeatureCache.Mode.BYPASS): Output {
        val uri = Uri.fromFile(file)
        val media = AnalysisEngine(context).probe(uri)
        val cancelled = AtomicBoolean(false)
        return requireNotNull(DistilledRallyModels.open(context, model, media, roi, cancelled::get, shared)).use { neural ->
            val result = AnalysisEngine(context, neural).analyze(uri, true, Int.MAX_VALUE,
                AnalysisTypes.VideoDecoderOptions.defaults(), cacheMode, window, cancelled, progress, false, false)
            assertNull("Ensemble cleanup must not suppress neural rallies", result.suppression())
            assertTrue(result.ranges().all { it.agreement() == "neural" && it.start() >= window.start() && it.end() <= window.end() })
            val tokenFile = context.cacheDir.resolve("neural-analysis").listFiles().orEmpty()
                .flatMap { it.listFiles().orEmpty().toList() }.single { it.name.endsWith("-tokens.f32") }
            val hash = MessageDigest.getInstance("SHA-256").digest(tokenFile.readBytes()).joinToString("") { "%02x".format(it) }
            val video = neural.report.getJSONObject("video")
            val metadata = listOf("sampleTimestamps", "quality").joinToString("|") { video.getJSONArray(it).toString() }
            if (shared && cacheMode == NativeFeatureCache.Mode.BYPASS) assertTrue(neural.report.getBoolean("sharedDecoding"))
            val scores = requireNotNull(neural.scores())
            assertEquals(model, scores.modelId)
            assertArrayEquals(result.productionServeOutputs().allLabelsV2().times(), scores.timestamps, 0.0)
            Output(result, hash, metadata, scores)
        }.also {
            assertTrue("Temporary embeddings must be removed", context.cacheDir.resolve("neural-analysis").listFiles().orEmpty().isEmpty())
        }
    }

    @Test fun bothPackagedVariantsMatchIndependentDecodingForRotatedFractionalWindow() {
        assertTrue(OpenCVLoader.initLocal())
        val fixture = fixture(90)
        try {
            val window = AnalysisTypes.AnalysisWindow(3.2, 5.5)
            for (model in listOf(RallyModels.RECALL, RallyModels.F1)) {
                val shared = run(fixture, model, true, window)
                val independent = run(fixture, model, false, window)
                assertEquals(model, independent.tokens, shared.tokens)
                assertEquals(independent.metadata, shared.metadata)
                assertEquals(independent.result.ranges(), shared.result.ranges())
                assertArrayEquals(independent.scores.probabilities, shared.scores.probabilities, 1e-6f)
                val source = NativeProjectStore.source(context, Uri.fromFile(fixture), fixture.name)
                val project = NativeProjectStore.newQueued(source, shared.result.media(), roi, window,
                    false, false, model)
                NativeProjectStore.save(context, project)
                try {
                    val saved = requireNotNull(NativeProjectStore.complete(context, project.id, shared.result, shared.scores))
                    assertEquals(model, saved.modelId)
                    assertEquals(model, requireNotNull(saved.editorSeed()).rallyModelId)
                    val restored = requireNotNull(NativeProjectStore.get(context, project.id)?.neuralScores)
                    assertArrayEquals(shared.scores.probabilities, restored.probabilities, 0f)
                } finally { NativeProjectStore.delete(context, project) }
            }
        } finally { fixture.delete() }
    }

    @Test fun cachedAvRunsIndependentEncoderWithoutChangingPredictionsOrVariant() {
        assertTrue(OpenCVLoader.initLocal())
        val fixture = fixture(0)
        try {
            val window = AnalysisTypes.AnalysisWindow(0.0, 2.0)
            val first = run(fixture, RallyModels.RECALL, true, window, NativeFeatureCache.Mode.USE)
            val cached = run(fixture, RallyModels.RECALL, true, window, NativeFeatureCache.Mode.USE)
            assertEquals(first.tokens, cached.tokens)
            assertEquals(first.metadata, cached.metadata)
            assertEquals(first.result.ranges(), cached.result.ranges())
        } finally { fixture.delete() }
    }

    @Test fun cancellationBeforeExtractionLeavesNoTemporaryTokens() {
        val fixture = fixture(0)
        try {
            val media = AnalysisEngine(context).probe(Uri.fromFile(fixture))
            try {
                DistilledRallyModels.open(context, RallyModels.RECALL, media, roi, { true })
                fail("Cancellation must abort before neural decoding")
            } catch (expected: java.io.IOException) {
                assertTrue(expected.message.orEmpty().contains("cancelled"))
            }
        } finally { fixture.delete() }
    }

    /** Optional external fixture: runtime values come only from test arguments. */
    @Test fun indexedVideoThroughPackagedProductionRuntime() {
        val args = InstrumentationRegistry.getArguments()
        assumeTrue("An indexed video URI was not supplied", args.containsKey("neuralVideoUri"))
        assertTrue(OpenCVLoader.initLocal())
        val uri = Uri.parse(requireNotNull(args.getString("neuralVideoUri")))
        val media = AnalysisEngine(context).probe(uri)
        val end = args.getString("neuralVideoSeconds")?.toDouble() ?: media.durationSeconds()
        val model = args.getString("neuralRallyModel") ?: RallyModels.DEFAULT
        require(RallyModels.isNeural(model))
        val cancelled = AtomicBoolean(false)
        val json = requireNotNull(DistilledRallyModels.open(context, model, media, roi, cancelled::get)).use { neural ->
            val result = AnalysisEngine(context, neural).analyze(uri, true, Int.MAX_VALUE,
                AnalysisTypes.VideoDecoderOptions.defaults(), NativeFeatureCache.Mode.BYPASS,
                AnalysisTypes.AnalysisWindow(0.0, end), cancelled, progress, true, true)
            JSONObject().apply {
                put("modelId", model)
                val tokenFile = context.cacheDir.resolve("neural-analysis").listFiles().orEmpty()
                    .flatMap { it.listFiles().orEmpty().toList() }.single { it.name.endsWith("-tokens.f32") }
                put("embeddingSha256", MessageDigest.getInstance("SHA-256").digest(tokenFile.readBytes())
                    .joinToString("") { "%02x".format(it) })
                put("durationSeconds", end)
                put("ranges", JSONArray().apply {
                    result.ranges().forEach { range -> put(JSONObject().apply {
                        put("start", range.start()); put("end", range.end())
                        put("confidence", range.confidence().toDouble()); put("agreement", range.agreement())
                    }) }
                })
                put("neural", neural.report)
                put("timingsMilliseconds", JSONObject(result.stageMilliseconds()))
                put("servingSideReady", result.servingSide() != null)
                put("sideSwitchReady", result.sideSwitch() != null)
                put("suppressionApplied", result.suppression() != null)
            }
        }
        val output = context.filesDir.resolve("integration-validation")
        output.mkdirs()
        output.resolve("neural-production-result.json").writeText(json.toString())
        assertFalse(json.getBoolean("suppressionApplied"))
        assertTrue(json.getJSONObject("neural").getBoolean("sharedDecoding"))
        assertTrue(json.getBoolean("servingSideReady"))
        assertTrue(json.getBoolean("sideSwitchReady"))
    }

    /** Loop the checked-in synthetic overlay fixture and mark display rotation. */
    private fun fixture(rotation: Int): File {
        val encoded = instrumentation.context.assets.open("overlay-fixture.mp4.b64").bufferedReader().use { it.readText() }
        val source = File.createTempFile("neural-source-", ".mp4", context.cacheDir)
        val target = File.createTempFile("neural-integration-", ".mp4", context.cacheDir)
        source.writeBytes(Base64.decode(encoded.trim(), Base64.DEFAULT))
        val extractor = MediaExtractor()
        val muxer = MediaMuxer(target.path, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        try {
            extractor.setDataSource(source.path)
            val track = (0 until extractor.trackCount).first { extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME).orEmpty().startsWith("video/") }
            val format = extractor.getTrackFormat(track)
            extractor.selectTrack(track)
            val outTrack = muxer.addTrack(format)
            muxer.setOrientationHint(rotation)
            muxer.start()
            val duration = format.getLong(MediaFormat.KEY_DURATION)
            val buffer = ByteBuffer.allocate(2 * 1024 * 1024)
            val info = MediaCodec.BufferInfo()
            val loops = ((8_000_000L + duration - 1) / duration).toInt()
            repeat(loops) { loop ->
                extractor.seekTo(0, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)
                while (extractor.sampleTime >= 0) {
                    buffer.clear()
                    val size = extractor.readSampleData(buffer, 0)
                    if (size < 0) break
                    info.set(0, size, extractor.sampleTime + loop * duration, extractor.sampleFlags)
                    muxer.writeSampleData(outTrack, buffer, info)
                    if (!extractor.advance()) break
                }
            }
            muxer.stop()
        } finally {
            muxer.release()
            extractor.release()
            source.delete()
        }
        return target
    }
}
