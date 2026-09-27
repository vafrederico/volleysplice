package com.volleycut.nativeanalysis

import android.net.Uri
import android.util.Base64
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.util.function.BooleanSupplier
import android.media.Image
import com.volleycut.video.NearestFrameSelection
import com.volleycut.video.YuvColorConversion
import java.io.IOException

/** Exercises the actual app decoder, including duplicate and terminal nearest-frame targets. */
@RunWith(AndroidJUnit4::class)
class VisualTimelineInstrumentedTest {
    @Test
    fun asynchronousAndSynchronousVisualFeaturesMatchAtRepeatedAndTerminalTargets() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val encoded = instrumentation.context.assets.open("overlay-fixture.mp4.b64")
            .bufferedReader().use { it.readText() }
        val input = context.cacheDir.resolve("visual-timeline-fixture.mp4")
        input.writeBytes(Base64.decode(encoded.trim(), Base64.DEFAULT))
        try {
            val uri = Uri.fromFile(input)
            val media = AnalysisEngine(context).probe(uri)
            assertTrue(media.durationSeconds() > 0.01)
            // Repeated targets use the same selected source image. The terminal
            // target lies after the last presentation frame but inside the video.
            val quarter = media.durationSeconds() / 4.0
            val times = doubleArrayOf(0.0, quarter, quarter, quarter * 2, media.durationSeconds() - 0.001)
            val roi = AnalysisTypes.Roi(0.0, 0.0, 1.0, 1.0, "Full frame")
            val options = AnalysisTypes.VideoDecoderOptions.defaults()
            val cache = NativeFeatureCache.open(context, uri, input.name, media, roi,
                Int.MAX_VALUE, times.size, null, NativeFeatureCache.Mode.BYPASS)
            val decoder = NativeVideoDecoder(context)
            val progress = AnalysisTypes.ProgressListener { _, _, _ -> }
            val cancelled = BooleanSupplier { false }
            val sync = decoder.decodeSynchronous(uri, media, roi, times, Int.MAX_VALUE,
                options, progress, cancelled)
            val async = decoder.decode(uri, media, roi, times, Int.MAX_VALUE, options,
                NativeFeatureCache.LoadedVisual.empty(),
                cache.newVisualWriter(NativeFeatureCache.LoadedVisual.empty()), progress, cancelled)
            assertArrayEquals(times, sync.analysisTimes(), 0.0)
            assertArrayEquals(times, async.analysisTimes(), 0.0)
            assertEquals(times.size * FeatureSchema.FRAME.size, async.values().size)
            assertTrue(async.values().all { it.isFinite() })
            assertArrayEquals(sync.values(), async.values(), 1e-5f)
            val seen = mutableSetOf<Long>()
            var requested = emptySet<Long>()
            var drained = false
            val consumer = object : SharedVideoFrameConsumer {
                override fun plan(source: LongArray, decoder: String, hardware: Boolean): Set<Long> {
                    val av = NearestFrameSelection.select(source, times, media.durationSeconds()).toSet()
                    val extra = source.first { it !in av }
                    requested = setOf(extra, source.first(), source.last())
                    return requested
                }
                override fun wants(pts: Long) = pts in requested
                override fun accept(image: Image, pts: Long, color: YuvColorConversion.Profile) {
                    assertTrue(image.planes.isNotEmpty())
                    assertTrue(image.planes[0].buffer.remaining() > 0)
                    assertTrue(seen.add(pts))
                }
                override fun finish() { assertEquals(requested, seen); drained = true }
                override fun close() {}
            }
            val shared = decoder.decode(uri, media, roi, times, Int.MAX_VALUE, options,
                NativeFeatureCache.LoadedVisual.empty(),
                cache.newVisualWriter(NativeFeatureCache.LoadedVisual.empty()), progress, cancelled, consumer)
            assertTrue(drained)
            assertArrayEquals(async.values(), shared.values(), 0f)
            assertArrayEquals(async.analysisTimes(), shared.analysisTimes(), 0.0)
            // A failed secondary consumer must tear down the actual app decoder.
            val failed = object : SharedVideoFrameConsumer {
                override fun plan(source: LongArray, decoder: String, hardware: Boolean) = setOf(source.first())
                override fun wants(pts: Long) = true
                override fun accept(image: Image, pts: Long, color: YuvColorConversion.Profile) {
                    throw IOException("Injected shared consumer failure")
                }
                override fun finish() { throw AssertionError("Failed consumer must not finish") }
                override fun close() {}
            }
            try {
                decoder.decode(uri, media, roi, times, Int.MAX_VALUE, options,
                    NativeFeatureCache.LoadedVisual.empty(),
                    cache.newVisualWriter(NativeFeatureCache.LoadedVisual.empty()), progress, cancelled, failed)
                throw AssertionError("Shared failure must propagate")
            } catch (expected: IOException) {
                assertTrue(expected.message.orEmpty().contains("Injected shared consumer failure"))
            }
        } finally {
            input.delete()
        }
    }
}
