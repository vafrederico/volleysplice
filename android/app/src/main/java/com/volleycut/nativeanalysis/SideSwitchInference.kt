package com.volleycut.nativeanalysis

import android.content.Context

internal object SideSwitchInference {
    internal data class FramePlan(
        val runtime: SideSwitchRuntime,
        val candidates: List<SideSwitchCandidateProposal>,
        val requestedTimes: DoubleArray,
    )

    fun framePlan(
        context: Context,
        input: SideSwitchAnalysisInput,
        duration: Double,
    ): FramePlan {
        val runtime = SideSwitchRuntimeParser.load(context)
        val candidates = SideSwitchModelRunner.generateCandidates(input, runtime)
        return FramePlan(
            runtime,
            candidates,
            if (candidates.isEmpty()) doubleArrayOf()
            else SideSwitchModelRunner.requestedTimestamps(input, candidates, duration),
        )
    }

    fun evaluate(
        duration: Double,
        input: SideSwitchAnalysisInput,
        plan: FramePlan,
        sampled: Map<Double, ByteArray>,
        progress: AnalysisTypes.ProgressListener,
        cancelled: () -> Boolean,
    ): SideSwitchOutput {
        if (plan.candidates.isEmpty()) return SideSwitchModelRunner.emptyOutput()
        fun frame(time: Double): ByteArray {
            val clamped = time.coerceIn(0.0, (duration - 0.01).coerceAtLeast(0.0))
            return sampled[clamped] ?: error("Missing team-switch frame at $clamped seconds")
        }
        val calibration = SideSwitchModelRunner.calibrationTimes(input.intervals)
            .map(::frame).toTypedArray()
        val geometry = SideSwitchFeatureExtractor.estimateCourtGeometry(calibration)
        val features = DoubleArray(plan.candidates.size * SIDE_SWITCH_FEATURE_COLUMNS)
        plan.candidates.forEachIndexed { row, candidate ->
            if (cancelled()) error("Analysis cancelled")
            val times = SideSwitchModelRunner.candidateSampleTimes(candidate)
            val before = Array(7) { frame(times[it]) }
            val after = Array(7) { frame(times[it + 7]) }
            val visual = SideSwitchFeatureExtractor.extract(before, after, geometry)
            val state = SideSwitchModelRunner.stateFeatures(candidate, input)
            val offset = row * SIDE_SWITCH_FEATURE_COLUMNS
            visual.copyInto(features, offset)
            state.copyInto(features, offset + visual.size)
            features[offset + visual.size + state.size] =
                if (candidate.kind == SideSwitchCandidateKind.INTERNAL_DEAD_STATE_PEAK) 1.0 else 0.0
            features[offset + visual.size + state.size + 1] = candidate.generatorScore
            progress.onProgress(
                "side-switch-features",
                (row + 1).toDouble() / plan.candidates.size,
                "Comparing team sides · ${row + 1}/${plan.candidates.size} candidates",
            )
        }
        val probabilities = SideSwitchModelRunner.predict(plan.runtime, features)
        val selected = SideSwitchModelRunner.decode(plan.runtime, plan.candidates, probabilities)
        val predictions = selected.map { index ->
            val candidate = plan.candidates[index]
            SideSwitchPrediction(
                id = candidate.id,
                timestamp = candidate.transitionTime,
                probability = probabilities[index],
                kind = candidate.kind,
                sourceRangeIds = candidate.sourceRangeIds,
            )
        }
        progress.onProgress(
            "side-switch",
            1.0,
            "Team-side switch markers ready · ${predictions.size} predicted",
        )
        return SideSwitchOutput(
            rows = plan.candidates.size,
            features = features,
            candidates = predictions,
        )
    }
}
