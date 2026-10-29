package com.volleycut.nativeanalysis

import android.content.Context

/** Restores missing rally/dead traces for pre-team-switch projects from the retained context cache. */
internal object ProductionStateLoader {
    fun loadIfMissing(
        context: Context,
        project: NativeProject,
    ): AnalysisTypes.ProductionStateOutputs {
        runCatching {
            SideSwitchModelRunner.validateStateOutputs(project.productionStateOutputs)
        }.onSuccess { return project.productionStateOutputs }

        val retained = requireNotNull(ProjectFeatureCache.loadVisual(context, project)) {
            "Team-switch inference needs the retained feature cache; rerun project analysis"
        }
        val cache = retained.cache
        val visual = retained.visual
        val contextual = cache.loadContext(visual.rows())
            ?: error("Team-switch inference needs contextual features; rerun project analysis")
        val all = ModelRunner(context, FeatureSchema.ALL_LABELS_V2_MODEL_ID)
            .runProfiled(visual.times(), contextual, visual.analyzedDurationSeconds())
        val previous = ModelRunner(context, FeatureSchema.PREVIOUS_PRODUCTION_MODEL_ID)
            .runProfiled(visual.times(), contextual, visual.analyzedDurationSeconds())
        return AnalysisTypes.ProductionStateOutputs(
            AnalysisTypes.ProductionStateOutput(
                FeatureSchema.ALL_LABELS_V2_MODEL_ID,
                visual.times().clone(),
                all.rallyProbabilities().clone(),
                all.deadStateProbabilities().clone(),
            ),
            AnalysisTypes.ProductionStateOutput(
                FeatureSchema.PREVIOUS_PRODUCTION_MODEL_ID,
                visual.times().clone(),
                previous.rallyProbabilities().clone(),
                previous.deadStateProbabilities().clone(),
            ),
        ).also(SideSwitchModelRunner::validateStateOutputs)
    }
}
