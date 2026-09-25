package com.volleycut.nativeanalysis

import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.function.BooleanSupplier
import kotlin.math.abs
import kotlin.math.ceil

/** Opt-in, actual app decoder parity check; media identity is supplied only at runtime. */
@RunWith(AndroidJUnit4::class)
class AudioTimelineInstrumentedTest {
    @Test
    fun compareBatchedAndSingleUnitAudio() {
        val args = InstrumentationRegistry.getArguments()
        assumeTrue("Supply audio_source_uri to run the device audio check", args.containsKey("audio_source_uri"))
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val duration = requireNotNull(args.getString("audio_duration_seconds")).toDouble()
        val uri = Uri.parse(requireNotNull(args.getString("audio_source_uri")))
        requireNotNull(context.contentResolver.openAssetFileDescriptor(uri, "r")).use { source ->
            check(source.length > 0) { "Audio validation source is empty" }
        }
        val times = DoubleArray(ceil(duration * 4).toInt()) { it / 4.0 }
        val output = File(context.filesDir, "audio-timeline-validation").apply { mkdirs() }
        val summaries = JSONArray()
        val results = listOf(
            AnalysisTypes.AudioDecoderMode.SINGLE_ACCESS_UNIT,
            AnalysisTypes.AudioDecoderMode.AUTO,
        ).map { mode ->
            val started = System.nanoTime()
            val result = NativeAudioDecoder(context).decode(
                uri, AnalysisTypes.AnalysisWindow(0.0, duration), times, mode,
                AnalysisTypes.ProgressListener { _, _, _ -> }, BooleanSupplier { false },
            )
            val buffer = ByteBuffer.allocate(result.features().size * 4).order(ByteOrder.LITTLE_ENDIAN)
            buffer.asFloatBuffer().put(result.features())
            File(output, "${mode.wireName()}.f32").writeBytes(buffer.array())
            summaries.put(JSONObject().apply {
                put("requestedMode", mode.wireName())
                put("actualMode", result.decoderMode())
                put("elapsedSeconds", (System.nanoTime() - started) / 1e9)
                put("decodedPcmFrames", result.decodedPcmFrames())
                put("resampledOutputSamples", result.resampledOutputSamples())
                put("audioFeatureFrames", result.audioFeatureFrames())
                put("featureSha256", result.featureSha256())
                put("rows", times.size)
                put("columns", FeatureSchema.AUDIO.size)
            })
            File(output, "summary.json").writeText(JSONObject().apply {
                put("durationSeconds", duration)
                put("audioExtractorVersion", NativeFeatureCache.AUDIO_EXTRACTOR_VERSION)
                put("decoders", summaries)
            }.toString(2))
            assertTrue("Audio must reach the end of the source", abs(result.resampledOutputSamples() / 16000.0 - duration) < 0.1)
            result
        }
        assertEquals(results[0].resampledOutputSamples(), results[1].resampledOutputSamples())
        var maxDifference = 0.0
        results[0].features().indices.forEach { index ->
            maxDifference = maxOf(maxDifference, abs(results[0].features()[index] - results[1].features()[index]).toDouble())
        }
        assertTrue("Batched/single feature maximum difference: $maxDifference", maxDifference < 1e-5)
    }
}
