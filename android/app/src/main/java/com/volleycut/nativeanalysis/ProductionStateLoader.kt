package com.volleycut.nativeanalysis

import android.content.Context
import android.net.Uri

/** Restores missing rally/dead traces for pre-team-switch projects from the retained context cache. */
internal object ProductionStateLoader {
    fun loadIfMissing(
        context: Context,
        project: NativeProject,
    ): AnalysisTypes.ProductionStateOutputs {
        runCatching {
            SideSwitchModelRunner.validateStateOutputs(project.productionStateOutputs)
        }.onSuccess { return project.productionStateOutputs }

        val requestedTimes = AnalysisEngine.analysisTimes(
            project.media.durationSeconds(),
            project.analysisWindow.start(),
            project.analysisWindow.end(),
        )
        val cacheSource = project.featureCacheSource ?: project.source
        val cache = NativeFeatureCache.openWithSourceMetadata(
            context,
            Uri.parse(cacheSource.uri),
            cacheSource.name,
            cacheSource.size,
            cacheSource.lastModified,
            project.media,
            project.roi,
            FeatureSchema.FULL_SOURCE_FRAME_LIMIT,
            requestedTimes.size,
            project.analysisWindow,
        )
        val visual = cache.loadVisual()
        require(visual.complete() && visual.rows() > 0 && cache.timesMatch(visual, requestedTimes)) {
            "Team-switch inference needs the retained feature cache; rerun project analysis"
        }
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
