package com.volleycut.nativeanalysis

import java.io.IOException
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class VideoPreparationProgressTest {
    @Test fun reportsTimestampCoverageWithoutRegressingForReorderedFrames() {
        var now = 0L
        val fractions = mutableListOf<Double>()
        val reporter = VideoPreparationProgress(100.0, Int.MAX_VALUE, listener(fractions), { false }, { now })
        reporter.sample(10_000_000)
        assertEquals(listOf(0.0), fractions)
        now += 250_000_000
        reporter.sample(40_000_000)
        now += 250_000_000
        reporter.sample(30_000_000)
        assertEquals(listOf(0.0, .4, .4), fractions)
        now += 250_000_000
        reporter.sample(100_000_000)
        assertEquals(.99, fractions.last(), 0.0)
        reporter.complete()
        assertEquals(1.0, fractions.last(), 0.0)
    }

    @Test fun frameLimitedScansReportTheirOwnInventoryProgressAndCanCancel() {
        var now = 0L
        var cancelled = false
        val fractions = mutableListOf<Double>()
        val reporter = VideoPreparationProgress(100.0, 4, listener(fractions), { cancelled }, { now })
        now += 250_000_000
        reporter.sample(0)
        assertEquals(.25, fractions.last(), 0.0)
        cancelled = true
        assertThrows(IOException::class.java) { reporter.sample(100_000) }
        assertThrows(IOException::class.java) { reporter.complete() }
        assertEquals(.25, fractions.last(), 0.0)
    }

    private fun listener(fractions: MutableList<Double>) = object : AnalysisTypes.ProgressListener {
        override fun onProgress(stage: String, fraction: Double, detail: String) {
            assertEquals("video-indexing", stage)
            assertTrue(detail.isNotBlank())
            fractions += fraction
        }
        override fun onPerformance(stats: AnalysisTypes.PerformanceStats) = Unit
    }
}
