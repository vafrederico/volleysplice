package com.volleycut.nativeanalysis

import android.content.Context
import android.net.Uri

internal object ServingSideInference {
    internal fun clampedTimestamp(anchor: Double, offset: Double, duration: Double): Double =
        (anchor + offset).coerceIn(0.0, (duration - 0.01).coerceAtLeast(0.0))

    internal fun requestedTimestamps(
        candidates: List<ServingSideModelRunner.CandidateInterval>,
        duration: Double,
    ): DoubleArray = candidates.flatMap { candidate ->
        (COURT_FLOW_OFFSETS_SECONDS.asList() + FLIGHT_OFFSETS_SECONDS.asList()).map { offset ->
            clampedTimestamp(candidate.start, offset, duration)
        }
    }.distinct().sorted().toDoubleArray()

    fun run(
        context: Context,
        uri: Uri,
        media: AnalysisTypes.MediaInfo,
        roi: AnalysisTypes.Roi,
        ranges: List<AnalysisTypes.Interval>,
        serveOutputs: AnalysisTypes.ProductionServeOutputs,
        progress: AnalysisTypes.ProgressListener,
        cancelled: () -> Boolean,
    ): ServingSideOutput {
        val seeds = ranges.map {
            SeedRange(secondsToMs(it.start()), secondsToMs(it.end()), it.confidence(), it.agreement())
        }
        val candidates = ServingSideModelRunner.candidates(seeds)
        if (candidates.isEmpty()) return ServingSideOutput(
            rows = 0,
            rawFeatures = doubleArrayOf(),
            candidates = emptyList(),
        )
        validateServeOutputs(serveOutputs)
        val duration = media.durationSeconds()
        val requested = requestedTimestamps(candidates, duration)
        progress.onProgress("serving-side", 0.0, "Sampling serving-side windows")
        val sampled = ServingSideFrameDecoder(context).decode(
            uri, media, roi, requested, progress, cancelled,
        )
        val runtime = context.assets.open(SERVING_SIDE_RUNTIME_ASSET).use(ServingSideRuntimeParser::read)
        val raw = DoubleArray(candidates.size * SERVING_SIDE_FEATURE_COLUMNS)
        candidates.forEachIndexed { row, candidate ->
            if (cancelled()) error("Analysis cancelled")
            fun frames(offsets: DoubleArray) = offsets.map { offset ->
                val time = clampedTimestamp(candidate.start, offset, duration)
                sampled[time] ?: error("Missing serving-side frame at $time seconds")
            }.toTypedArray()
            val values = ServingSideFeatureExtractor.extract(
                frames(COURT_FLOW_OFFSETS_SECONDS), frames(FLIGHT_OFFSETS_SECONDS),
            )
            values.copyInto(raw, row * SERVING_SIDE_FEATURE_COLUMNS)
            progress.onProgress(
                "serving-side-features",
                (row + 1).toDouble() / candidates.size,
                "Measuring serving side · ${row + 1}/${candidates.size} rallies",
            )
        }
        val ranked = ServingSideModelRunner.tiedPercentileRanks(
            raw, candidates.size, SERVING_SIDE_FEATURE_COLUMNS,
        )
        val verdicts = candidates.mapIndexed { row, candidate ->
            ServingSideModelRunner.verdict(
                candidate,
                ranked.copyOfRange(
                    row * SERVING_SIDE_FEATURE_COLUMNS,
                    (row + 1) * SERVING_SIDE_FEATURE_COLUMNS,
                ),
                serveOutputs.allLabelsV2(),
                serveOutputs.previousProduction(),
                runtime,
            )
        }
        progress.onProgress("serving-side", 1.0, "Serving-side verdicts ready · ${verdicts.size} rallies")
        return ServingSideOutput(
            rows = candidates.size,
            rawFeatures = raw,
            candidates = verdicts,
        )
    }

    private fun validateServeOutputs(outputs: AnalysisTypes.ProductionServeOutputs) {
        listOf(outputs.allLabelsV2(), outputs.previousProduction()).forEach { output ->
            require(output.times().isNotEmpty() && output.times().size == output.probabilities().size) {
                "Serving-side inference requires both production serve-head outputs"
            }
        }
    }
}
