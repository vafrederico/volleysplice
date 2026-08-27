package com.volleycut.nativeanalysis

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
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
        assertEquals(
            "match.mp4 · 50% · 12.5 fps · 1.25x realtime · ETA 1m 5s",
            projectInferenceNotificationDetail("match.mp4", stats, 50),
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
        assertEquals(0.65, servingSideOverallProgress("serving-side-features", 0.0), 0.0001)
        assertEquals(0.71, servingSideOverallProgress("serving-side-features", 0.5), 0.0001)
        assertEquals(0.77, servingSideOverallProgress("serving-side", 1.0), 0.0001)
        assertEquals(0.88, servingSideOverallProgress("side-switch-features", 0.5), 0.0001)
        assertEquals(1.0, servingSideOverallProgress("side-switch", 1.0), 0.0001)
    }

    @Test
    fun notificationProgressTracksEachEnabledStepIndependently() {
        val video = projectCreationNotificationStage("video", 0.5, true)
        assertEquals("Video analysis (1/3)", video.title)
        assertEquals(50, video.progressPercent)

        val audio = projectCreationNotificationStage("audio", 0.5, true)
        assertEquals("Audio analysis (2/3)", audio.title)
        assertEquals(40, audio.progressPercent)
        assertEquals(85, projectCreationNotificationStage("normalizing", 0.5, true).progressPercent)
        assertEquals(95, projectCreationNotificationStage("inference", 0.5, true).progressPercent)

        val serving = projectCreationNotificationStage("serving-side-features", 0.5, true)
        assertEquals("Score tracking analysis (3/3)", serving.title)
        assertEquals(71, serving.progressPercent)

        val withoutServing = projectCreationNotificationStage("audio", 1.0, false)
        assertEquals("Audio analysis (2/2)", withoutServing.title)
        assertEquals(80, withoutServing.progressPercent)
    }

    @Test
    fun laterStagesReportTheirOwnProgressSpeedAndEta() {
        var now = 0L
        val tracker = NotificationStageSpeedTracker { now }

        assertEquals(
            "match.mp4 · 0% · measuring speed",
            tracker.detail("match.mp4", "2/3", 0.0),
        )
        now = 2_000_000_000L
        assertEquals(
            "match.mp4 · 40% · 20.0%/s · ETA 3s",
            tracker.detail("match.mp4", "2/3", 0.4),
        )
        assertEquals(
            "match.mp4 · 1% · measuring speed",
            tracker.detail("match.mp4", "3/3", 0.01),
        )
        now = 4_000_000_000L
        assertEquals(
            "match.mp4 · 42% · 20.5%/s · ETA 3s",
            tracker.detail("match.mp4", "3/3", 0.42),
        )
    }

    @Test
    fun callbackUpdatesAreRateLimitedButStagesAndCompletionRemainImmediate() {
        var now = 0L
        val throttle = CallbackEmissionThrottle(500_000_000L) { now }

        assertTrue(throttle.shouldEmit("specialist-frames", 0.0))
        now = 100_000_000L
        assertFalse(throttle.shouldEmit("specialist-frames", 0.1))
        now = 200_000_000L
        assertTrue(throttle.shouldEmit("serving-side-features", 0.0))
        now = 300_000_000L
        assertTrue(throttle.shouldEmit("serving-side-features", 1.0))
        now = 350_000_000L
        assertFalse(throttle.shouldEmit("serving-side-features", 1.0))
        now = 900_000_000L
        assertTrue(throttle.shouldEmit("serving-side-features", 1.0))
    }

    @Test
    fun liveInferenceMeasurementsFollowTheProductionStepOrder() {
        var now = 0L
        val tracker = InferenceProgressTracker(
            includeCore = true,
            includeServingSide = true,
            includeSideSwitch = true,
            nanoTime = { now },
        )

        tracker.update("video", 0.25, "Generating visual features")
        now = 2_000_000_000L
        val audio = tracker.update("audio", 0.5, "Generating audio features")
        assertEquals(listOf("video", "audio", "rally", "serving-side", "side-switch"), audio.map { it.id })
        assertEquals(
            listOf(
                "Scanning video",
                "Listening for play",
                "Finding rallies",
                "Finding serve markers",
                "Finding team switches",
            ),
            audio.map { it.label },
        )
        assertEquals(InferenceStepStatus.COMPLETE, audio[0].status)
        assertEquals(2_000.0, audio[0].elapsedMilliseconds, 0.001)
        assertEquals(InferenceStepStatus.RUNNING, audio[1].status)
        assertEquals(0.5, audio[1].fraction, 0.001)

        now = 3_000_000_000L
        val scoreFrames = tracker.update("specialist-frames", 0.5, "Shared score frames")
        assertEquals(InferenceStepStatus.COMPLETE, scoreFrames[1].status)
        assertEquals(InferenceStepStatus.COMPLETE, scoreFrames[2].status)
        assertEquals(InferenceStepStatus.RUNNING, scoreFrames[3].status)
        assertEquals(0.365, scoreFrames[3].fraction, 0.001)

        val decoded = InferenceStepMeasurementsJson.decode(
            InferenceStepMeasurementsJson.encode(scoreFrames),
        )
        assertEquals(scoreFrames, decoded)
    }

    @Test
    fun scoreOnlyMeasurementsOmitDisabledTeamSwitchStep() {
        val tracker = InferenceProgressTracker(
            includeCore = false,
            includeServingSide = true,
            includeSideSwitch = false,
        )

        val steps = tracker.update("score-specialists", 0.0, "Loading serving-side model")
        assertEquals(listOf("serving-side"), steps.map { it.id })
        assertEquals(InferenceStepStatus.RUNNING, steps.single().status)
    }
}
