package com.volleycut.nativeanalysis

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class AnalysisWindowTest {
    @Test
    fun normalizesBoundsToSourceDuration() {
        assertEquals(
            AnalysisTypes.AnalysisWindow(0.0, 12.0),
            AnalysisTypes.AnalysisWindow.normalize(
                AnalysisTypes.AnalysisWindow(-5.0, 20.0),
                12.0,
            ),
        )
        assertEquals(
            AnalysisTypes.AnalysisWindow(9.0, 9.0),
            AnalysisTypes.AnalysisWindow.normalize(
                AnalysisTypes.AnalysisWindow(9.0, 4.0),
                12.0,
            ),
        )
    }

    @Test
    fun analysisTimesContainOnlyGloballyAlignedGameWindowSamples() {
        assertArrayEquals(
            doubleArrayOf(1.25, 1.5, 1.75, 2.0),
            AnalysisEngine.analysisTimes(10.0, 1.1, 2.2),
            1e-9,
        )
    }
}
