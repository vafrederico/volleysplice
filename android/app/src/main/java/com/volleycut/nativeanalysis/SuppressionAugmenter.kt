package com.volleycut.nativeanalysis

import android.content.Context

/** Adds suppression analysis to a ready legacy project using only its contextual feature cache. */
internal object SuppressionAugmenter {
    fun augment(context: Context, project: NativeProject): NativeProject {
        if (RallyModels.isNeural(project.modelId)) return project
        project.suppression?.let { return project }
        val retained = requireNotNull(ProjectFeatureCache.loadVisual(context, project)) {
            "Suppression needs the saved feature cache for this recording"
        }
        val cache = retained.cache
        val visual = retained.visual
        val contextual = cache.loadContext(visual.rows())
            ?: error("Suppression needs the saved contextual feature cache")
        val duration = minOf(project.analysisWindow.end(), visual.analyzedDurationSeconds())
        val allLabels = clip(
            ModelRunner(context, FeatureSchema.ALL_LABELS_V2_MODEL_ID)
                .run(visual.times(), contextual, duration),
            project.analysisWindow.start(),
            duration,
        )
        val previous = clip(
            ModelRunner(context, FeatureSchema.PREVIOUS_PRODUCTION_MODEL_ID)
                .run(visual.times(), contextual, duration),
            project.analysisWindow.start(),
            duration,
        )
        val suppressionRun = SuppressionModelRunner(context).run(
            visual.times(), contextual, duration,
        )
        val updated = project.copy(
            productionComponents = AnalysisTypes.ProductionComponents(allLabels, previous),
            suppression = SuppressionPolicyEngine.build(
                allLabels,
                previous,
                suppressionRun.probabilities(),
                clip(suppressionRun.decodedIntervals(), project.analysisWindow.start(), duration),
                duration,
            ),
            updatedAtMs = System.currentTimeMillis(),
        )
        NativeProjectStore.save(context, updated)
        updated.editorSeed()?.let { EditorProjectStore.save(context, it) }
        return updated
    }

    private fun clip(
        intervals: List<AnalysisTypes.Interval>,
        start: Double,
        end: Double,
    ) = intervals.mapNotNull { interval ->
        val clippedStart = maxOf(start, interval.start())
        val clippedEnd = minOf(end, interval.end())
        if (clippedEnd <= clippedStart) null else AnalysisTypes.Interval(
            clippedStart, clippedEnd, interval.confidence(), interval.agreement(),
        )
    }
}
