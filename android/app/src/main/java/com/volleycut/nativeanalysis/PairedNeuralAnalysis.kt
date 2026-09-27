package com.volleycut.nativeanalysis

import android.content.Context
import android.net.Uri
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicBoolean

/** Experimental normal-app entry point. Keeps variant-specific cuts and score results separate. */
internal object PairedNeuralAnalysis {
    data class Output(
        val primary: AnalysisTypes.AnalysisResult,
        val alternate: AnalysisTypes.AnalysisResult,
        val alternateModelId: String,
        val report: JSONObject,
    )

    fun run(context: Context, uri: Uri, primaryModelId: String, media: AnalysisTypes.MediaInfo,
            roi: AnalysisTypes.Roi, window: AnalysisTypes.AnalysisWindow, cacheMode: NativeFeatureCache.Mode,
            cancelled: AtomicBoolean, progress: AnalysisTypes.ProgressListener,
            includeServingSide: Boolean, includeSideSwitch: Boolean,
            inspect: ((PairedNeuralPipeline) -> Unit)? = null): Output {
        val started = System.nanoTime()
        val scoringStages = setOf("score-specialists", "specialist-frames", "serving-side-frames",
            "serving-side-features", "serving-side", "side-switch-features", "side-switch")
        val primaryProgress = object : AnalysisTypes.ProgressListener {
            override fun onProgress(stage: String, fraction: Double, detail: String) {
                if (stage == "complete") return
                if (stage in scoringStages) progress.onProgress("paired-scores", servingSideOverallProgress(stage, fraction) * .5,
                    "${RallyModels.shortLabel(primaryModelId)}: $detail")
                else progress.onProgress(stage, fraction, detail)
            }
            override fun onPerformance(stats: AnalysisTypes.PerformanceStats) = progress.onPerformance(stats)
        }
        PairedNeuralPipeline(context, primaryModelId, media, roi, cancelled::get).use { pair ->
            val loaded = System.nanoTime()
            val primary = AnalysisEngine(context, pair).analyze(uri, true, Int.MAX_VALUE,
                AnalysisTypes.VideoDecoderOptions.defaults(), cacheMode, window, cancelled,
                primaryProgress, includeServingSide, includeSideSwitch)
            val primaryReady = System.nanoTime()
            val companionCpuStart = android.os.Debug.threadCpuTimeNanos()
            val alternateRanges = pair.alternateRanges.mapNotNull {
                val start = maxOf(window.start(), it.start())
                val end = minOf(window.end(), it.end())
                if (end > start) AnalysisTypes.Interval(start, end, it.confidence(), "neural") else null
            }
            var serving: ServingSideOutput? = null
            var switching: SideSwitchOutput? = null
            var servingError: String? = null
            val profile = pair.alternateProfile.toMutableMap()
            if (includeServingSide) {
                progress.onProgress("paired-scores", 0.5, "Preparing ${RallyModels.shortLabel(pair.alternateId)} score results")
                try {
                    val score = ScoreSpecialistInference.run(context, uri, media, roi, alternateRanges,
                        primary.productionServeOutputs(), primary.productionStateOutputs(), primary.productionComponents(),
                        AnalysisTypes.ProgressListener { stage, fraction, detail ->
                            progress.onProgress("paired-scores", .5 + servingSideOverallProgress(stage, fraction) * .5,
                                "${RallyModels.shortLabel(pair.alternateId)}: $detail")
                        }, cancelled::get, includeSideSwitch)
                    serving = score.servingSide
                    switching = score.sideSwitch
                    profile.putAll(score.profile.measurementProfile())
                } catch (error: Exception) {
                    if (cancelled.get()) throw java.io.IOException("Analysis cancelled", error)
                    servingError = error.message ?: "Alternate score analysis failed"
                }
            }
            if (cancelled.get()) throw java.io.IOException("Analysis cancelled")
            val scoresEnd = System.nanoTime()
            val companionCpuMs = (android.os.Debug.threadCpuTimeNanos() - companionCpuStart) / 1e6
            val scoresMs = (scoresEnd - primaryReady) / 1_000_000
            val alternate = primary.withVariantPredictions(alternateRanges, serving, servingError, switching,
                servingError.takeIf { includeSideSwitch }, pair.alternateHeadMilliseconds, scoresMs, profile)
            val report = pair.report().apply {
                put("mode", "paired-distilled-analysis-v1")
                put("bundleLoadMs", (loaded - started) / 1e6)
                put("primaryReadyMs", (primaryReady - started) / 1e6)
                put("alternateScoreMs", scoresMs)
                put("allReadyMs", (scoresEnd - started) / 1e6)
                put("primaryStageMilliseconds", JSONObject(primary.stageMilliseconds()))
                put("primaryProfileMilliseconds", JSONObject(primary.profileMilliseconds()))
                put("alternateScoreProfileMilliseconds", JSONObject(profile.filterKeys { !it.startsWith("neural/") }))
                put("timingScope", "Primary readiness includes both encoders and temporal heads; alternate scoring follows. Nested profile counters are not additive.")
            }
            // Diagnostics run after the readiness timestamp, while temporary tokens still exist.
            inspect?.invoke(pair)
            progress.onProgress("complete", 1.0, "Recall and F1 results are ready")
            val allReady = primary.withPairedCompletion(scoresMs, (loaded - started) / 1_000_000,
                (scoresEnd - started) / 1_000_000, companionCpuMs)
            return Output(allReady, alternate, pair.alternateId, report)
        }
    }
}
