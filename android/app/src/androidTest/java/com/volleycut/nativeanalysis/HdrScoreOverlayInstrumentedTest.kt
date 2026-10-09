package com.volleycut.nativeanalysis

import android.opengl.EGL14
import android.opengl.GLES20
import android.util.Log
import androidx.annotation.OptIn
import androidx.media3.common.C
import androidx.media3.common.ColorInfo
import androidx.media3.common.MimeTypes
import androidx.media3.common.util.GlUtil
import androidx.media3.common.util.UnstableApi
import androidx.media3.transformer.EncoderUtil
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.nio.ByteBuffer
import java.nio.ByteOrder

@OptIn(UnstableApi::class)
@RunWith(AndroidJUnit4::class)
class HdrScoreOverlayInstrumentedTest {
    @Test
    fun hdrGpuCompositesHlgAndPqWithReferenceWhiteAndSourcePixelsPreserved() {
        for (hlg in listOf(false, true)) renderAndCheck(hlg, faded = false)
    }

    @Test
    fun hdrGpuFadesInDisplayLightWithoutDarkeningPremultipliedEdgesTwice() {
        for (hlg in listOf(false, true)) renderAndCheck(hlg, faded = true)
    }

    @Test
    fun sdrGpuFadesCanvasColorsWithoutApplyingAlphaTwice() {
        renderAndCheck(hlg = false, faded = true, hdr = false)
    }

    @Test
    fun unsupportedHdrEncoderFailsExplicitlyInsteadOfSelectingSdr() {
        for (transfer in listOf(C.COLOR_TRANSFER_HLG, C.COLOR_TRANSFER_ST2084)) {
            val policy = ExportColorPolicy(ColorInfo.Builder()
                .setColorSpace(C.COLOR_SPACE_BT2020).setColorRange(C.COLOR_RANGE_LIMITED)
                .setColorTransfer(transfer).build())
            val encoders = EncoderUtil.getSupportedEncodersForHdrEditing(MimeTypes.VIDEO_H265, policy.color)
            Log.i("HdrScoreTest", "transfer=$transfer HDR encoders=${encoders.map { it.name }}")
            if (encoders.isEmpty()) {
                val failure = runCatching { policy.requireHdrEncoder() }.exceptionOrNull()
                assertTrue(failure is IllegalStateException)
                assertTrue(failure?.message?.contains("original HDR color space") == true)
            } else policy.requireHdrEncoder()
            assertTrue(!policy.matches(ColorInfo.SDR_BT709_LIMITED))
        }
    }

    private fun renderAndCheck(hlg: Boolean, faded: Boolean, hdr: Boolean = true) {
        val display = GlUtil.getDefaultEglDisplay()
        val context = GlUtil.createEglContext(EGL14.EGL_NO_CONTEXT, display, 3,
            GlUtil.EGL_CONFIG_ATTRIBUTES_RGBA_8888)
        val surface = GlUtil.createFocusedPlaceholderEglSurface(context, display)
        val input = GlUtil.createTexture(WIDTH, HEIGHT, true)
        val inputFbo = GlUtil.createFboForTexture(input)
        val output = GlUtil.createTexture(WIDTH, HEIGHT, true)
        val outputFbo = GlUtil.createFboForTexture(output)
        val shader = ScoreShaderProgram(ScoreCanvasOverlay(snapshot(faded), 0, 0), hlg, hdr)
        try {
            GlUtil.focusFramebufferUsingCurrentContext(inputFbo, WIDTH, HEIGHT)
            GLES20.glClearColor(0.08f, 0.08f, 0.08f, 1f)
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
            shader.configure(WIDTH, HEIGHT)
            GlUtil.focusFramebufferUsingCurrentContext(outputFbo, WIDTH, HEIGHT)
            shader.drawFrame(input, if (faded) 4_125_000 else 0)
            val untouched = pixel(300, 100)
            untouched.take(3).forEach { assertEquals("Transparent source pixel", .08f, it, .0001f) }
            // CPU reference uses BT.2100 1000-nit HLG system gamma 1.2 and 203-nit graphics.
            val alpha = if (faded) 128.0 / 255.0 else 1.0
            if (!hdr) {
                fun expectedSdr(rgb: IntArray) = rgb.map { it / 255.0 * alpha + .08 * (1 - alpha) }.toDoubleArray()
                assertRgb(expectedSdr(intArrayOf(217, 52, 43)), pixel(10, 5), tolerance = .004)
                assertRgb(expectedSdr(intArrayOf(255, 255, 255)), pixel(90, 5), tolerance = .004)
                assertRgb(expectedSdr(intArrayOf(35, 103, 201)), pixel(138, 5), tolerance = .004)
                return
            }
            val backgroundDisplay = if (hlg) Math.pow(.08, 1.2) else .08
            fun expected(displayRgb: DoubleArray): DoubleArray {
                val mixed = displayRgb.map { it * alpha + backgroundDisplay * (1 - alpha) }.toDoubleArray()
                if (!hlg) return mixed
                val y = .2627 * mixed[0] + .6780 * mixed[1] + .0593 * mixed[2]
                return mixed.map { it * Math.pow(y, -1.0 / 6.0) }.toDoubleArray()
            }
            assertRgb(expected(doubleArrayOf(.090881426, .016198536, .007314390)), pixel(10, 5))
            assertRgb(expected(doubleArrayOf(.203, .203, .203)), pixel(90, 5))
            assertRgb(expected(doubleArrayOf(.016342549, .026901041, .108668079)), pixel(138, 5))
        } finally {
            shader.release()
            GlUtil.deleteTexture(input)
            GlUtil.deleteTexture(output)
            GlUtil.deleteFbo(inputFbo)
            GlUtil.deleteFbo(outputFbo)
            GlUtil.destroyEglSurface(display, surface)
            GlUtil.destroyEglContext(display, context)
        }
    }

    private fun snapshot(faded: Boolean) = ScoreExportSnapshot(
        render = true, fadeScoreOverlay = faded, renderPointTimeline = false,
        scoreTracking = ScoreTracking(team1Name = "Falcons", team2Name = "Wolves",
            serveMarkers = if (faded) listOf(
                ServeMarker("S1", 1_000, ServingSide.NEAR, ServeMarkerOrigin.MANUAL),
                ServeMarker("S2", 5_000, ServingSide.NEAR, ServeMarkerOrigin.MANUAL),
            ) else emptyList()),
        ignoredIntervals = emptyList(), excludedRallyIds = emptySet(),
        rallyRanges = if (faded) listOf(ScoreRallyRange(5_000, 7_000, 4_000, 8_000)) else emptyList(),
    )

    private fun pixel(x: Int, yFromTop: Int): FloatArray {
        val values = ByteBuffer.allocateDirect(16).order(ByteOrder.nativeOrder()).asFloatBuffer()
        GLES20.glReadPixels(x, HEIGHT - 1 - yFromTop, 1, 1, GLES20.GL_RGBA, GLES20.GL_FLOAT, values)
        GlUtil.checkGlError()
        return FloatArray(4) { values.get(it) }
    }

    private fun assertRgb(expected: DoubleArray, actual: FloatArray, tolerance: Double = .0015) {
        expected.forEachIndexed { index, value ->
            assertEquals("channel $index", value, actual[index].toDouble(), tolerance)
        }
    }

    private companion object { const val WIDTH = 320; const val HEIGHT = 240 }
}
