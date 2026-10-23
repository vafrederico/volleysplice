package com.volleycut.nativeanalysis

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class NeuralServingProgressTest {
    @Test fun servingSequenceIsMonotonicWithMissingOrCachedNeuralContext() {
        for (extractContext in listOf(true, false)) {
            val events = mutableListOf<Pair<String, Double>>()
            val tracker = InferenceProgressTracker(false, true, true)
            val listener = AnalysisTypes.ProgressListener { stage, fraction, detail ->
                events += stage to fraction
                tracker.update(stage, fraction, detail)
            }
            val progress = NeuralServingProgress(listener)
            listener.onProgress("score-specialists", 0.0, "load")
            for (fraction in listOf(0.0, 0.5, 1.0)) listener.onProgress("specialist-frames", fraction, "decode")
            for (fraction in listOf(0.0, 0.5, 1.0)) progress.specialist.onProgress("serving-side-features", fraction, "fixed-flight")
            progress.specialist.onProgress("serving-side", 1.0, "preliminary verdicts")
            assertEquals(InferenceStepStatus.RUNNING, tracker.snapshot().first().status)
            if (extractContext) {
                for (fraction in listOf(0.0, 0.25, 1.0)) progress.neural.onProgress("serving-side-features", fraction, "context embeddings")
            }
            for (fraction in listOf(0.2, 0.6, 1.0)) progress.neural.onProgress("serving-side", fraction, "neural verdicts")
            for (fraction in listOf(0.0, 0.5, 1.0)) listener.onProgress("side-switch-features", fraction, "switches")
            listener.onProgress("side-switch", 1.0, "done")

            val overall = events.map { (stage, fraction) -> servingSideOverallProgress(stage, fraction) }
            val creation = events.map { (stage, fraction) -> projectCreationOverallProgress(stage, fraction, true) }
            val notification = events.map { (stage, fraction) -> projectCreationNotificationStage(stage, fraction, true).progress }
            for (sequence in listOf(overall, creation, notification)) {
                assertTrue("Context extraction=$extractContext: $sequence", sequence.zipWithNext().all { (a, b) -> b >= a })
                assertEquals(1.0, sequence.last(), 0.00001)
            }
            assertEquals(0.71, servingSideOverallProgress(events[7].first, events[7].second), 0.00001)
            assertEquals(1, events.count { it.first == "serving-side" && it.second == 1.0 })
        }
    }

    @Test fun disabledSwitchesAndEmptyServingCandidatesCompleteWithoutRewinding() {
        val fractions = mutableListOf<Double>()
        val listener = AnalysisTypes.ProgressListener { stage, fraction, _ -> fractions += servingSideOverallProgress(stage, fraction) }
        listener.onProgress("specialist-frames", 1.0, "decoded")
        // No candidates means both serving evaluators can return without progress callbacks.
        listener.onProgress("side-switch", 1.0, "disabled")
        assertTrue(fractions.zipWithNext().all { (a, b) -> b >= a })
        assertEquals(1.0, fractions.last(), 0.0)
    }
}
