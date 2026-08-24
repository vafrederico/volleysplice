package com.volleycut.nativeanalysis

import android.content.Context
import org.json.JSONObject
import java.io.InputStream
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToLong

internal const val SIDE_SWITCH_RUNTIME_ASSET = "side-switch-c2570481c30d.json"
internal const val SIDE_SWITCH_MODEL_ID = "side-switch-hard-negative-mining-v1/union34-top2-x2"
internal const val SIDE_SWITCH_MODEL_FINGERPRINT =
    "sha256:c2570481c30dec62f56ac3284cd4028763ffd8e72a9ad07e9908fec10df387e3"
internal const val SIDE_SWITCH_FEATURE_VERSION = "SIDE-SWITCH-UNION34-V1"
internal const val SIDE_SWITCH_CANDIDATE_CONTRACT = "range-boundaries-dead-peaks-v1"
internal const val SIDE_SWITCH_WIDTH = 256
internal const val SIDE_SWITCH_HEIGHT = 144
internal const val SIDE_SWITCH_FEATURE_COLUMNS = 34

internal val SIDE_SWITCH_FEATURE_NAMES = listOf(
    "v4BroadSameAssignmentCost",
    "v4TightSameAssignmentCost",
    "v4MeanSwapMargin",
    "v4GlobalAppearanceChange",
    "v4MaximumCameraShift",
    "v4MinimumAlignmentResponse",
    "playerSameAssignmentCost",
    "playerSwappedAssignmentCost",
    "playerSwapMargin",
    "playerOrientationFlipEvidence",
    "minimumPlayerSideSeparation",
    "playerSideSeparationChange",
    "beforePlayerPaletteInstability",
    "afterPlayerPaletteInstability",
    "playerGlobalAppearanceChange",
    "minimumProposalCoverage",
    "proposalCoverageChange",
    "minimumProposalCount",
    "proposalCountChange",
    "minimumNearSupport",
    "minimumFarSupport",
    "sideSupportImbalanceChange",
    "productionBeforeSupportCount",
    "productionAfterSupportCount",
    "productionMinimumAdjacentSupportCount",
    "productionMinimumAdjacentRallyPeak",
    "productionGapLiveFraction",
    "productionGapMeanRallyScore",
    "productionGapPeakRallyScore",
    "productionGapMeanDeadStateScore",
    "productionGapPeakDeadStateScore",
    "productionGapDurationSeconds",
    "candidateIsInternalDeadStatePeak",
    "candidateGeneratorScore",
)

internal enum class SideSwitchCandidateKind(val wireName: String) {
    ADJACENT_RALLY_BOUNDARY("adjacent-rally-boundary"),
    INTERNAL_DEAD_STATE_PEAK("internal-dead-state-peak");

    companion object {
        fun fromWireName(value: String) = entries.firstOrNull { it.wireName == value }
    }
}

internal data class SideSwitchPrediction(
    val id: String,
    val timestamp: Double,
    val probability: Double,
    val kind: SideSwitchCandidateKind,
    val sourceRangeIds: List<String>,
)

internal data class SideSwitchOutput(
    val modelId: String = SIDE_SWITCH_MODEL_ID,
    val modelFingerprint: String = SIDE_SWITCH_MODEL_FINGERPRINT,
    val featureVersion: String = SIDE_SWITCH_FEATURE_VERSION,
    val candidateContract: String = SIDE_SWITCH_CANDIDATE_CONTRACT,
    val rows: Int,
    val columns: Int = SIDE_SWITCH_FEATURE_COLUMNS,
    val features: DoubleArray,
    val candidates: List<SideSwitchPrediction>,
) {
    override fun equals(other: Any?): Boolean = other is SideSwitchOutput &&
        modelId == other.modelId && modelFingerprint == other.modelFingerprint &&
        featureVersion == other.featureVersion && candidateContract == other.candidateContract &&
        rows == other.rows && columns == other.columns && features.contentEquals(other.features) &&
        candidates == other.candidates

    override fun hashCode(): Int {
        var result = modelId.hashCode()
        result = 31 * result + modelFingerprint.hashCode()
        result = 31 * result + featureVersion.hashCode()
        result = 31 * result + candidateContract.hashCode()
        result = 31 * result + rows
        result = 31 * result + columns
        result = 31 * result + features.contentHashCode()
        return 31 * result + candidates.hashCode()
    }
}

internal data class SideSwitchCandidateProposal(
    val id: String,
    val kind: SideSwitchCandidateKind,
    val gapStart: Double,
    val gapEnd: Double,
    val transitionTime: Double,
    val generatorScore: Double,
    val sourceRangeIds: List<String>,
    val beforeStart: Double,
    val beforeEnd: Double,
    val afterStart: Double,
    val afterEnd: Double,
)

internal data class SideSwitchAnalysisInput(
    val intervals: List<AnalysisTypes.Interval>,
    val productionComponents: AnalysisTypes.ProductionComponents,
    val productionStateOutputs: AnalysisTypes.ProductionStateOutputs,
)

internal data class SideSwitchRuntime(
    val impute: DoubleArray,
    val mean: DoubleArray,
    val scale: DoubleArray,
    val weights: DoubleArray,
    val bias: Double,
    val threshold: Double,
    val internalPeakThreshold: Double,
    val internalPeakMinimumSeparationSeconds: Double,
    val internalPeakRangeEdgeExclusionSeconds: Double,
    val internalPeakProposalHalfWidthSeconds: Double,
    val minimumCandidateIndexSeparation: Int,
    val minimumTimeSeparationSeconds: Double,
    val freePredictionsPerRecording: Int,
    val countPenaltyLogitPerExcessPrediction: Double,
)

internal object SideSwitchRuntimeParser {
    fun read(input: InputStream): SideSwitchRuntime = parse(JSONObject(input.bufferedReader().readText()))

    fun load(context: Context): SideSwitchRuntime =
        context.assets.open(SIDE_SWITCH_RUNTIME_ASSET).use(::read)

    fun parse(json: JSONObject): SideSwitchRuntime {
        require(json.optInt("schemaVersion") == 1)
        require(json.optString("modelId") == SIDE_SWITCH_MODEL_ID)
        require(json.optString("fingerprint") == SIDE_SWITCH_MODEL_FINGERPRINT)
        require(json.optString("featureVersion") == SIDE_SWITCH_FEATURE_VERSION)
        val classifier = json.getJSONObject("classifier")
        val names = classifier.getJSONArray("featureNames")
        require(names.length() == SIDE_SWITCH_FEATURE_COLUMNS)
        SIDE_SWITCH_FEATURE_NAMES.forEachIndexed { index, expected ->
            require(names.getString(index) == expected) { "Side-switch feature signature differs at $index" }
        }
        fun vector(name: String) = classifier.getJSONArray(name).let { source ->
            require(source.length() == SIDE_SWITCH_FEATURE_COLUMNS)
            DoubleArray(source.length()) { source.getDouble(it) }.also { values ->
                require(values.all(Double::isFinite))
            }
        }
        val generator = json.getJSONObject("candidateGenerator")
        val decoder = json.getJSONObject("decoder")
        return SideSwitchRuntime(
            impute = vector("impute"),
            mean = vector("mean"),
            scale = vector("scale").also { require(it.all { value -> value > 0 }) },
            weights = vector("weights"),
            bias = classifier.getDouble("bias").also { require(it.isFinite()) },
            threshold = classifier.getDouble("threshold").also { require(it in 0.0..1.0) },
            internalPeakThreshold = generator.getDouble("internalPeakThreshold"),
            internalPeakMinimumSeparationSeconds =
                generator.getDouble("internalPeakMinimumSeparationSeconds"),
            internalPeakRangeEdgeExclusionSeconds =
                generator.getDouble("internalPeakRangeEdgeExclusionSeconds"),
            internalPeakProposalHalfWidthSeconds =
                generator.getDouble("internalPeakProposalHalfWidthSeconds"),
            minimumCandidateIndexSeparation = decoder.getInt("minimumCandidateIndexSeparation"),
            minimumTimeSeparationSeconds = decoder.getDouble("minimumTimeSeparationSeconds"),
            freePredictionsPerRecording = decoder.getInt("freePredictionsPerRecording"),
            countPenaltyLogitPerExcessPrediction =
                decoder.getDouble("countPenaltyLogitPerExcessPrediction"),
        )
    }
}

internal object SideSwitchModelRunner {
    private data class NamedRange(
        val id: String,
        val start: Double,
        val end: Double,
        val interval: AnalysisTypes.Interval,
    )

    private fun orderedRanges(source: List<AnalysisTypes.Interval>): List<NamedRange> = source
        .filter { it.end() > it.start() && it.start().isFinite() && it.end().isFinite() }
        .sortedWith(compareBy<AnalysisTypes.Interval> { it.start() }.thenBy { it.end() })
        .mapIndexed { index, interval ->
            NamedRange("R${(index + 1).toString().padStart(3, '0')}", interval.start(), interval.end(), interval)
        }

    internal fun sampleTimes(start: Double, end: Double): DoubleArray {
        val duration = end - start
        var first = start + min(0.2, duration * 0.08)
        var last = end - min(0.15, duration * 0.08)
        if (last <= first) {
            first = start + duration * 0.2
            last = start + duration * 0.8
        }
        return DoubleArray(7) { first + (last - first) * it / 6.0 }
    }

    fun calibrationTimes(intervals: List<AnalysisTypes.Interval>): DoubleArray = orderedRanges(intervals)
        .take(7).flatMap { sampleTimes(it.start, it.end).asList() }.toDoubleArray()

    fun generateCandidates(
        input: SideSwitchAnalysisInput,
        runtime: SideSwitchRuntime,
    ): List<SideSwitchCandidateProposal> {
        val ranges = orderedRanges(input.intervals)
        val state = input.productionStateOutputs.allLabelsV2()
        validateStateOutputs(input.productionStateOutputs)
        val times = state.times()
        val dead = state.deadStateProbabilities()
        val candidates = mutableListOf<SideSwitchCandidateProposal>()
        for (index in 0 until ranges.lastIndex) {
            val before = ranges[index]
            val after = ranges[index + 1]
            require(after.start >= before.end) { "Side-switch ranges overlap after production merge" }
            candidates += SideSwitchCandidateProposal(
                id = "switch:boundary:${before.id}:${after.id}",
                kind = SideSwitchCandidateKind.ADJACENT_RALLY_BOUNDARY,
                gapStart = before.end,
                gapEnd = after.start,
                transitionTime = (before.end + after.start) / 2,
                generatorScore = 0.0,
                sourceRangeIds = listOf(before.id, after.id),
                beforeStart = before.start,
                beforeEnd = before.end,
                afterStart = after.start,
                afterEnd = after.end,
            )
        }
        ranges.forEach { range ->
            val eligible = times.indices.filter { index ->
                times[index] >= range.start + runtime.internalPeakRangeEdgeExclusionSeconds &&
                    times[index] <= range.end - runtime.internalPeakRangeEdgeExclusionSeconds &&
                    dead[index] >= runtime.internalPeakThreshold
            }.sortedWith(compareByDescending<Int> { dead[it] }.thenBy { times[it] }.thenBy { it })
            val selected = mutableListOf<Int>()
            eligible.forEach { candidate ->
                if (selected.all { other ->
                        abs(times[candidate] - times[other]) >=
                            runtime.internalPeakMinimumSeparationSeconds
                    }) selected += candidate
            }
            selected.sortedBy { times[it] }.forEach { index ->
                val time = times[index]
                candidates += SideSwitchCandidateProposal(
                    id = "switch:internal-dead-peak:${range.id}:${(time * 1000).roundToLong()}",
                    kind = SideSwitchCandidateKind.INTERNAL_DEAD_STATE_PEAK,
                    gapStart = max(range.start, time - runtime.internalPeakProposalHalfWidthSeconds),
                    gapEnd = min(range.end, time + runtime.internalPeakProposalHalfWidthSeconds),
                    transitionTime = time,
                    generatorScore = dead[index].toDouble(),
                    sourceRangeIds = listOf(range.id),
                    beforeStart = time - 4,
                    beforeEnd = time - 1,
                    afterStart = time + 1,
                    afterEnd = time + 4,
                )
            }
        }
        return candidates.sortedWith(
            compareBy<SideSwitchCandidateProposal> { it.transitionTime }
                .thenBy { it.kind.wireName }.thenBy { it.id },
        )
    }

    fun candidateSampleTimes(candidate: SideSwitchCandidateProposal): DoubleArray =
        sampleTimes(candidate.beforeStart, candidate.beforeEnd) +
            sampleTimes(candidate.afterStart, candidate.afterEnd)

    fun requestedTimestamps(
        input: SideSwitchAnalysisInput,
        candidates: List<SideSwitchCandidateProposal>,
        duration: Double,
    ): DoubleArray = (calibrationTimes(input.intervals).asList() +
        candidates.flatMap { candidateSampleTimes(it).asList() })
        .map { it.coerceIn(0.0, (duration - 0.01).coerceAtLeast(0.0)) }
        .distinct().sorted().toDoubleArray()

    fun stateFeatures(
        candidate: SideSwitchCandidateProposal,
        input: SideSwitchAnalysisInput,
    ): DoubleArray {
        val ranges = orderedRanges(input.intervals).associateBy { it.id }
        val beforeRange = checkNotNull(ranges[candidate.sourceRangeIds.first()])
        val afterRange = checkNotNull(ranges[candidate.sourceRangeIds.last()])
        val before = evidence(beforeRange, input)
        val after = evidence(afterRange, input)
        val all = input.productionStateOutputs.allLabelsV2()
        val previous = input.productionStateOutputs.previousProduction()
        val times = all.times()
        val rally = FloatArray(times.size) { max(all.rallyProbabilities()[it], previous.rallyProbabilities()[it]) }
        val dead = FloatArray(times.size) { max(all.deadStateProbabilities()[it], previous.deadStateProbabilities()[it]) }
        val rallyWindow = valuesInWindow(times, rally, candidate.gapStart, candidate.gapEnd)
        val deadWindow = valuesInWindow(times, dead, candidate.gapStart, candidate.gapEnd)
        val sourceRanges = input.productionComponents.allLabelsV2() +
            input.productionComponents.previousProduction()
        return doubleArrayOf(
            before.first.toDouble(),
            after.first.toDouble(),
            min(before.first, after.first).toDouble(),
            min(before.second, after.second),
            unionDuration(candidate.gapStart, candidate.gapEnd, sourceRanges) /
                max(candidate.gapEnd - candidate.gapStart, 1e-6),
            rallyWindow.average(),
            rallyWindow.maxOrNull() ?: 0.0,
            deadWindow.average(),
            deadWindow.maxOrNull() ?: 0.0,
            candidate.gapEnd - candidate.gapStart,
        )
    }

    fun predict(runtime: SideSwitchRuntime, features: DoubleArray): DoubleArray {
        require(features.size % SIDE_SWITCH_FEATURE_COLUMNS == 0)
        return DoubleArray(features.size / SIDE_SWITCH_FEATURE_COLUMNS) { row ->
            var score = runtime.bias
            repeat(SIDE_SWITCH_FEATURE_COLUMNS) { column ->
                val raw = features[row * SIDE_SWITCH_FEATURE_COLUMNS + column]
                val value = if (raw.isFinite()) raw else runtime.impute[column]
                score += ((value - runtime.mean[column]) / runtime.scale[column]) *
                    runtime.weights[column]
            }
            1.0 / (1.0 + exp(-score.coerceIn(-30.0, 30.0)))
        }
    }

    fun decode(
        runtime: SideSwitchRuntime,
        candidates: List<SideSwitchCandidateProposal>,
        probabilities: DoubleArray,
    ): List<Int> {
        require(candidates.size == probabilities.size)
        val chronological = candidates.indices.sortedWith(
            compareBy<Int> { candidates[it].transitionTime }.thenBy { candidates[it].id },
        )
        val ordinal = chronological.withIndex().associate { (order, index) -> index to order }
        val ranked = candidates.indices.sortedWith(
            compareByDescending<Int> { probabilities[it] }
                .thenBy { candidates[it].transitionTime }.thenBy { candidates[it].id },
        )
        val selected = mutableListOf<Int>()
        val thresholdLogit = logit(runtime.threshold)
        ranked.forEach { index ->
            val score = probabilities[index]
            require(score.isFinite() && score in 0.0..1.0)
            if (selected.any { other ->
                    (runtime.minimumCandidateIndexSeparation > 0 &&
                        abs(checkNotNull(ordinal[index]) - checkNotNull(ordinal[other])) <
                        runtime.minimumCandidateIndexSeparation) ||
                        (runtime.minimumTimeSeparationSeconds > 0 &&
                            abs(candidates[index].transitionTime - candidates[other].transitionTime) <
                            runtime.minimumTimeSeparationSeconds)
                }) return@forEach
            val excess = max(0, selected.size + 1 - runtime.freePredictionsPerRecording)
            val margin = logit(score) - thresholdLogit -
                runtime.countPenaltyLogitPerExcessPrediction * excess
            if (margin >= -1e-12) selected += index
        }
        return selected.sortedWith(
            compareBy<Int> { candidates[it].transitionTime }.thenBy { candidates[it].id },
        )
    }

    fun emptyOutput() = SideSwitchOutput(rows = 0, features = doubleArrayOf(), candidates = emptyList())

    fun validateStateOutputs(outputs: AnalysisTypes.ProductionStateOutputs) {
        val all = outputs.allLabelsV2()
        val previous = outputs.previousProduction()
        listOf(all, previous).forEach { output ->
            require(output.times().isNotEmpty()) { "Team-switch inference requires production state traces" }
            require(output.rallyProbabilities().size == output.times().size)
            require(output.deadStateProbabilities().size == output.times().size)
            require(output.times().indices.all { index ->
                output.times()[index].isFinite() &&
                    (index == 0 || output.times()[index] > output.times()[index - 1]) &&
                    output.rallyProbabilities()[index] in 0f..1f &&
                    output.deadStateProbabilities()[index] in 0f..1f
            })
        }
        require(all.times().contentEquals(previous.times())) {
            "Production state traces use different timestamps"
        }
    }

    private fun evidence(
        range: NamedRange,
        input: SideSwitchAnalysisInput,
    ): Pair<Int, Double> {
        val sources = listOf(
            input.productionComponents.allLabelsV2() to input.productionStateOutputs.allLabelsV2(),
            input.productionComponents.previousProduction() to input.productionStateOutputs.previousProduction(),
        ).filter { (ranges, _) -> ranges.any { overlap(range.start, range.end, it.start(), it.end()) > 0 } }
        val peaks = sources.map { (_, output) ->
            valuesInWindow(
                output.times(), output.rallyProbabilities(), range.start, range.end,
            ).maxOrNull() ?: 0.0
        }
        return sources.size to (peaks.minOrNull() ?: 0.0)
    }

    private fun valuesInWindow(
        times: DoubleArray,
        values: FloatArray,
        start: Double,
        end: Double,
    ): DoubleArray {
        val selected = times.indices.filter { times[it] >= start && times[it] < end }
            .map { values[it].toDouble() }
        if (selected.isNotEmpty()) return selected.toDoubleArray()
        val center = (start + end) / 2
        val nearest = times.indices.minByOrNull { abs(times[it] - center) } ?: 0
        return doubleArrayOf(values[nearest].toDouble())
    }

    private fun unionDuration(
        start: Double,
        end: Double,
        ranges: List<AnalysisTypes.Interval>,
    ): Double {
        val clipped = ranges.mapNotNull {
            val left = max(start, it.start())
            val right = min(end, it.end())
            if (right > left) left to right else null
        }.sortedWith(compareBy<Pair<Double, Double>> { it.first }.thenBy { it.second })
        if (clipped.isEmpty()) return 0.0
        var activeStart = clipped.first().first
        var activeEnd = clipped.first().second
        var total = 0.0
        clipped.drop(1).forEach { (left, right) ->
            if (left <= activeEnd) activeEnd = max(activeEnd, right) else {
                total += activeEnd - activeStart
                activeStart = left
                activeEnd = right
            }
        }
        return total + activeEnd - activeStart
    }

    private fun overlap(start: Double, end: Double, otherStart: Double, otherEnd: Double) =
        max(0.0, min(end, otherEnd) - max(start, otherStart))

    private fun logit(value: Double): Double {
        val clipped = value.coerceIn(1e-9, 1 - 1e-9)
        return ln(clipped / (1 - clipped))
    }
}
