package com.volleycut.nativeanalysis

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

/** Uses the actual packaged OpenCV runtime; empty Hough Mats cannot be read as integer arrays. */
@RunWith(AndroidJUnit4::class)
class EmptyCourtGeometryInstrumentedTest {
    @Test fun uniformFramesUseTheNoEvidenceFallback() {
        for (shade in listOf(0, 80, 255)) {
            val frame = ByteArray(SIDE_SWITCH_WIDTH * SIDE_SWITCH_HEIGHT * 3) { shade.toByte() }
            val geometry = SideSwitchFeatureExtractor.estimateCourtGeometry(arrayOf(frame, frame, frame))
            assertEquals(.5, geometry.netYRatio(), 0.0)
            assertEquals(0.0, geometry.confidence(), 0.0)
        }
    }

    @Test fun blankFramesDoNotDiscardValidLineEvidence() {
        val blank = ByteArray(SIDE_SWITCH_WIDTH * SIDE_SWITCH_HEIGHT * 3)
        val line = blank.copyOf()
        val y = SIDE_SWITCH_HEIGHT / 2
        for (row in y - 1..y + 1) for (x in 0 until SIDE_SWITCH_WIDTH) for (channel in 0..2)
            line[(row * SIDE_SWITCH_WIDTH + x) * 3 + channel] = 255.toByte()
        val onlyLine = SideSwitchFeatureExtractor.estimateCourtGeometry(arrayOf(line))
        val mixed = SideSwitchFeatureExtractor.estimateCourtGeometry(arrayOf(blank, line, blank))
        assertTrue(onlyLine.confidence() > 0.0)
        assertEquals(.5, onlyLine.netYRatio(), .03)
        assertEquals(onlyLine.netYRatio(), mixed.netYRatio(), 0.0)
        assertEquals(onlyLine.confidence() / 3, mixed.confidence(), 1e-9)
    }
}
