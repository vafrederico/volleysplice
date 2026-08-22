package com.volleycut.nativeanalysis

import org.junit.Assert.assertEquals
import org.junit.Test

class ProjectAnalysisServiceTest {
    @Test
    fun projectNotificationShowsOnlyFilenameAndSpeedMeasurements() {
        val stats = AnalysisTypes.PerformanceStats(
            100,
            200,
            300,
            25.0,
            10.0,
            12.5,
            1.25,
            65.0,
            1L,
            2L,
        )

        assertEquals(
            "match.mp4 · 12.5 fps · 1.25x realtime · ETA 1m 5s",
            projectInferenceNotificationDetail("match.mp4", stats),
        )
        assertEquals("match.mp4", projectInferenceNotificationDetail("match.mp4", null))
    }

    @Test
    fun projectCreationProgressAdvancesAcrossPipelineStages() {
        assertEquals(0.22, projectCreationOverallProgress("video", 0.5, true), 0.0001)
        assertEquals(0.42, projectCreationOverallProgress("audio", 0.0, true), 0.0001)
        assertEquals(0.70, projectCreationOverallProgress("serving-side-frames", 0.0, true), 0.01)
        assertEquals(0.898, projectCreationOverallProgress("serving-side-frames", 0.8, true), 0.0001)
        assertEquals(1.0, projectCreationOverallProgress("complete", 1.0, true), 0.0001)
    }

    @Test
    fun projectCreationWithoutServingSideUsesTheFullProgressRange() {
        assertEquals(0.65, projectCreationOverallProgress("audio", 0.0, false), 0.0001)
        assertEquals(0.99, projectCreationOverallProgress("inference", 1.0, false), 0.0001)
        assertEquals(0.99, projectCreationOverallProgress("serving-side", 1.0, false), 0.0001)
    }

    @Test
    fun servingSideProgressDoesNotResetBetweenFrameSamplingAndFeatureExtraction() {
        assertEquals(0.42, servingSideOverallProgress("serving-side-frames", 0.5), 0.0001)
        assertEquals(0.82, servingSideOverallProgress("serving-side-frames", 1.0), 0.0001)
        assertEquals(0.82, servingSideOverallProgress("serving-side-features", 0.0), 0.0001)
        assertEquals(0.905, servingSideOverallProgress("serving-side-features", 0.5), 0.0001)
        assertEquals(1.0, servingSideOverallProgress("serving-side", 1.0), 0.0001)
    }
}
