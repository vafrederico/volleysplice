package com.volleycut.nativeanalysis

import android.content.Context
import android.net.Uri

/** Adds suppression analysis to a ready legacy project using only its contextual feature cache. */
internal object SuppressionAugmenter {
    fun augment(context: Context, project: NativeProject): NativeProject {
        project.suppression?.let { return project }
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
            "Suppression needs the saved feature cache for this recording"
        }
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
