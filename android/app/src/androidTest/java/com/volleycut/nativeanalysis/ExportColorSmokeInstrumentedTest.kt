package com.volleycut.nativeanalysis

import android.net.Uri
import android.util.Log
import androidx.annotation.OptIn
import androidx.media3.common.MediaItem
import androidx.media3.common.util.UnstableApi
import androidx.media3.transformer.Composition
import androidx.media3.transformer.EditedMediaItem
import androidx.media3.transformer.EditedMediaItemSequence
import androidx.media3.transformer.Effects
import androidx.media3.transformer.ExportException
import androidx.media3.transformer.ExportResult
import androidx.media3.transformer.InAppMp4Muxer
import androidx.media3.transformer.Transformer
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/** Optional private media stays outside the APK and is supplied at test runtime. */
@OptIn(UnstableApi::class)
@RunWith(AndroidJUnit4::class)
class ExportColorSmokeInstrumentedTest {
    @Test fun runtimeSdrSampleExportsWithScores() = exportRuntimeSample("sdr", false)
    @Test fun runtimeHdrSampleExportsOrReportsUnsupportedEncoder() = exportRuntimeSample("hdr", true)

    private fun exportRuntimeSample(kind: String, hdr: Boolean) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val input = context.cacheDir.resolve("$kind-export-input.mp4")
        assumeTrue("Runtime media fixture was not supplied", input.isFile)
        val output = context.cacheDir.resolve("$kind-export-output.mp4")
        output.delete()
        val policy = ExportColorPolicy.read(context, Uri.fromFile(input))
        assertEquals(hdr, policy.isHdr)
        if (hdr) {
            val unsupported = runCatching { policy.requireHdrEncoder() }.exceptionOrNull()
            if (unsupported != null) {
                assertTrue(unsupported.message?.contains("original HDR color space") == true)
                Log.w("HdrScoreTest", "HDR ENCODE UNAVAILABLE: ${unsupported.message}")
                return
            }
        }
        val snapshot = ScoreExportSnapshot(render = true,
            scoreTracking = ScoreTracking(team1Name = "Falcons", team2Name = "Wolves"),
            ignoredIntervals = emptyList(), excludedRallyIds = emptySet(), rallyRanges = emptyList())
        val item = MediaItem.Builder().setUri(Uri.fromFile(input)).setClippingConfiguration(
            MediaItem.ClippingConfiguration.Builder().setEndPositionMs(1_000).build()).build()
        val edited = EditedMediaItem.Builder(item).setDurationUs(1_000_000)
            .setEffects(Effects(emptyList(), listOf(ScoreOverlayEffect(
                ScoreCanvasOverlay(snapshot, 0, 0), policy.color.colorTransfer)))).build()
        val composition = Composition.Builder(EditedMediaItemSequence.withAudioAndVideoFrom(listOf(edited)))
            .setHdrMode(Composition.HDR_MODE_KEEP_HDR).build()
        val completed = CountDownLatch(1)
        val failure = AtomicReference<ExportException?>(null)
        var transformer: Transformer? = null
        instrumentation.runOnMainSync {
            transformer = Transformer.Builder(context).withInstrumentedDecoder(context)
                .setVideoMimeType(policy.outputMimeType)
                .setMuxerFactory(policy.preservingMuxer(InAppMp4Muxer.Factory()))
                .addListener(object : Transformer.Listener {
                    override fun onCompleted(composition: Composition, exportResult: ExportResult) {
                        completed.countDown()
                    }
                    override fun onError(composition: Composition, exportResult: ExportResult,
                        exportException: ExportException) {
                        failure.set(exportException)
                        completed.countDown()
                    }
                }).build().also { it.start(composition, output.absolutePath) }
        }
        try {
            assertTrue("Runtime export timed out", completed.await(90, TimeUnit.SECONDS))
            assertNull(failure.get()?.stackTraceToString(), failure.get())
            assertTrue(output.length() > 0)
            val exported = ExportColorPolicy.read(context, Uri.fromFile(output))
            assertTrue("Output transfer, primaries or range changed", policy.matches(exported.color))
            Log.i("HdrScoreTest", "$kind export preserved ${exported.color}")
        } finally {
            instrumentation.runOnMainSync { transformer?.cancel() }
        }
    }
}
