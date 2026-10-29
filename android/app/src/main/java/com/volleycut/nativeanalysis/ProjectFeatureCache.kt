package com.volleycut.nativeanalysis

import android.content.Context
import android.net.Uri

/** Opens retained features using the original cache identity after a source is re-linked. */
internal object ProjectFeatureCache {
    data class Visual(val cache: NativeFeatureCache, val visual: NativeFeatureCache.LoadedVisual)

    fun loadVisual(context: Context, project: NativeProject): Visual? {
        val requestedTimes = AnalysisEngine.analysisTimes(
            project.media.durationSeconds(),
            project.analysisWindow.start(),
            project.analysisWindow.end(),
        )
        val source = project.featureCacheSource ?: project.source
        val cache = NativeFeatureCache.openWithSourceMetadata(
            context,
            Uri.parse(source.uri),
            source.name,
            source.size,
            source.lastModified,
            project.media,
            project.roi,
            FeatureSchema.FULL_SOURCE_FRAME_LIMIT,
            requestedTimes.size,
            project.analysisWindow,
        )
        val visual = cache.loadVisual()
        return if (visual.complete() && visual.rows() > 0 && cache.timesMatch(visual, requestedTimes)) {
            Visual(cache, visual)
        } else null
    }
}
