package com.volleycut.nativeanalysis

import org.json.JSONArray
import org.json.JSONObject
import java.io.InputStream
import java.security.MessageDigest
import kotlin.math.abs
import kotlin.math.exp

internal const val SERVING_SIDE_MODEL_ID = "serving-side-fixed-flight-v3"
internal const val SERVING_SIDE_MODEL_FINGERPRINT =
    "85bc3325fbd43abba6ba3726ac091dc6a3a1eafc68d63d979cb2a7450eb49e06"
internal const val SERVING_SIDE_FEATURE_VERSION = "SERVSIDE237-FLIGHT"
internal const val SERVING_SIDE_ANCHOR_CONTRACT = "merged-production-interval-start-v1"
internal const val SERVING_SIDE_RUNTIME_ASSET = "serving-side-85bc3325fbd4.json"
internal const val SERVING_SIDE_RUNTIME_SHA256 =
    "14f18bf0b0f326ccd7ef4b3d614a96a53dd9675df61813fd375677489d0e5a7c"
internal const val SERVING_SIDE_WIDTH = 192
internal const val SERVING_SIDE_HEIGHT = 108
internal const val SERVING_SIDE_FEATURE_COLUMNS = 237
internal const val SERVING_SIDE_SIDE_THRESHOLD = 0.4783744762021848
internal const val SERVING_SIDE_FAR_REVIEW_THRESHOLD = 0.3121748736511044
internal const val SERVING_SIDE_NEAR_REVIEW_THRESHOLD = 0.5028396703865513
internal const val SERVING_SIDE_SERVE_HEAD_THRESHOLD = 0.85
internal const val SERVING_SIDE_SERVE_HEAD_WINDOW_SECONDS = 1.0

internal val COURT_FLOW_OFFSETS_SECONDS = doubleArrayOf(
    -1.25, -0.75, -0.35, -0.10, 0.10, 0.30, 0.55, 0.85,
)
internal val FLIGHT_OFFSETS_SECONDS = doubleArrayOf(
    -0.15, 0.05, 0.20, 0.35, 0.55, 0.80, 1.10, 1.40, 1.75,
)

internal enum class ServingSide(val wireName: String) {
    NEAR("near"), FAR("far"), REVIEW("review");

    companion object {
        fun fromWireName(value: String) = entries.firstOrNull { it.wireName == value }
    }
}

internal enum class ServingSideVerdict(val wireName: String) {
    NEAR("near"), FAR("far"), REVIEW("review"), NOT_SERVE("not-serve");

    companion object {
        fun fromWireName(value: String) = entries.firstOrNull { it.wireName == value }
    }
}

internal enum class ServingSideDecisionSource(val wireName: String) {
    SERVE_HEAD("serve-head"), PRODUCTION_RALLY_RECOVERY("production-rally-recovery"), NONE("none");

    companion object {
        fun fromWireName(value: String) = entries.firstOrNull { it.wireName == value }
    }
}

internal enum class ServingSideReviewReason(val wireName: String) {
    SIDE_SCORE("side-score"), PRODUCTION_RALLY_RECOVERY("production-rally-recovery");

    companion object {
        fun fromWireName(value: String) = entries.firstOrNull { it.wireName == value }
    }
}

internal data class ServingSideHeadEvidence(
    val modelId: String,
    val threshold: Double,
    val peakProbability: Double,
    val peakTime: Double,
    val crossesThreshold: Boolean,
    val nearestDetection: AnalysisTypes.Serve?,
)

internal data class ServingSideCandidate(
    val id: String,
    val anchor: Double,
    val intervalStart: Double,
    val intervalEnd: Double,
    val agreement: String?,
    val nearProbability: Double,
    val side: ServingSide,
    val verdict: ServingSideVerdict,
    val serveDecisionSource: ServingSideDecisionSource,
    val reviewReasons: List<ServingSideReviewReason>,
    val allLabelsV2Evidence: ServingSideHeadEvidence,
    val previousProductionEvidence: ServingSideHeadEvidence,
)

internal data class ServingSideOutput(
    val modelId: String = SERVING_SIDE_MODEL_ID,
    val modelFingerprint: String = SERVING_SIDE_MODEL_FINGERPRINT,
    val featureVersion: String = SERVING_SIDE_FEATURE_VERSION,
    val anchorContract: String = SERVING_SIDE_ANCHOR_CONTRACT,
    val rows: Int,
    val columns: Int = SERVING_SIDE_FEATURE_COLUMNS,
    val rawFeatures: DoubleArray,
    val candidates: List<ServingSideCandidate>,
) {
    fun isReusableFor(ranges: List<SeedRange>): Boolean {
        if (modelId != SERVING_SIDE_MODEL_ID ||
            modelFingerprint != SERVING_SIDE_MODEL_FINGERPRINT ||
            featureVersion != SERVING_SIDE_FEATURE_VERSION ||
            anchorContract != SERVING_SIDE_ANCHOR_CONTRACT ||
            columns != SERVING_SIDE_FEATURE_COLUMNS || rows != candidates.size ||
            rawFeatures.size != rows * columns
        ) return false
        val expected = ServingSideModelRunner.candidates(ranges)
        return expected.size == candidates.size && expected.indices.all { index ->
            val left = expected[index]
            val right = candidates[index]
            left.id == right.id && left.start == right.anchor && left.start == right.intervalStart &&
                left.end == right.intervalEnd && left.agreement == right.agreement
        }
    }
}

internal data class ServingSideRuntime(
    val featureNames: List<String>,
    val impute: DoubleArray,
    val mean: DoubleArray,
    val scale: DoubleArray,
    val weights: DoubleArray,
    val bias: Double,
    val l2: Double,
    val threshold: Double,
)

internal object ServingSideRuntimeParser {
    private const val KIND = "volleycut-serving-side-fixed-flight-runtime-v1"

    fun read(input: InputStream): ServingSideRuntime {
        val bytes = input.readBytes()
        // Git's Windows checkout can expose the one-line JSON with CRLF even though the
        // repository and packaged asset are pinned to LF. Hash the canonical asset bytes.
        val canonicalBytes = bytes.toString(Charsets.UTF_8).replace("\r\n", "\n")
            .toByteArray(Charsets.UTF_8)
        val actualHash = MessageDigest.getInstance("SHA-256").digest(canonicalBytes)
            .joinToString("") { "%02x".format(it.toInt() and 0xff) }
        require(actualHash == SERVING_SIDE_RUNTIME_SHA256) {
            "Frozen serving-side asset hash mismatch: expected $SERVING_SIDE_RUNTIME_SHA256, got $actualHash"
        }
        return parse(JSONObject(canonicalBytes.toString(Charsets.UTF_8)))
    }

    fun parse(payload: JSONObject): ServingSideRuntime {
        require(payload.optInt("schemaVersion", -1) == 1 && payload.optString("kind") == KIND &&
            payload.optString("modelId") == SERVING_SIDE_MODEL_ID &&
            payload.optString("fingerprint") == SERVING_SIDE_MODEL_FINGERPRINT &&
            payload.optString("featureVersion") == SERVING_SIDE_FEATURE_VERSION
        ) { "Serving-side runtime identity does not match the frozen model" }
        require(payload.requireNumbers("courtFlowOffsetsSeconds", 8).contentEquals(COURT_FLOW_OFFSETS_SECONDS) &&
            payload.requireNumbers("flightOffsetsSeconds", 9).contentEquals(FLIGHT_OFFSETS_SECONDS)
        ) { "Serving-side runtime sampling offsets changed" }
        val resize = payload.getJSONObject("resize")
        val grid = payload.getJSONObject("flightGrid")
        require(resize.optInt("width") == SERVING_SIDE_WIDTH &&
            resize.optInt("height") == SERVING_SIDE_HEIGHT &&
            grid.optInt("rows") == 4 && grid.optInt("columns") == 6
        ) { "Serving-side runtime image geometry changed" }
        val expectedNames = ServingSideFeatureNames.all()
        require(payload.getJSONArray("featureNames").strings() == expectedNames) {
            "Serving-side runtime feature signature changed"
        }
        val model = payload.getJSONObject("model")
        require(model.optString("family") == "class-balanced-logistic") {
            "Serving-side runtime must use class-balanced logistic regression"
        }
        val impute = model.requireNumbers("impute", expectedNames.size)
        val mean = model.requireNumbers("mean", expectedNames.size)
        val scale = model.requireNumbers("scale", expectedNames.size)
        val weights = model.requireNumbers("weights", expectedNames.size)
        require(scale.all { it > 0 }) { "Serving-side scales must be positive" }
        val bias = model.requireFinite("bias")
        val l2 = model.requireFinite("l2")
        val threshold = model.requireFinite("threshold")
        val review = payload.getJSONObject("reviewBand")
        val gate = payload.getJSONObject("gate")
        require(threshold == SERVING_SIDE_SIDE_THRESHOLD &&
            payload.requireFinite("sideThreshold") == SERVING_SIDE_SIDE_THRESHOLD &&
            review.requireFinite("farUpperExclusive") == SERVING_SIDE_FAR_REVIEW_THRESHOLD &&
            review.requireFinite("nearLowerInclusive") == SERVING_SIDE_NEAR_REVIEW_THRESHOLD &&
            gate.requireFinite("serveHeadThreshold") == SERVING_SIDE_SERVE_HEAD_THRESHOLD &&
            gate.requireFinite("serveHeadWindowSeconds") == SERVING_SIDE_SERVE_HEAD_WINDOW_SECONDS &&
            gate.optString("rallyRecoveryAgreement") == "both-models" &&
            gate.optBoolean("rallyRecoveryRequiresReview")
        ) { "Serving-side decision policy is malformed" }
        return ServingSideRuntime(expectedNames, impute, mean, scale, weights, bias, l2, threshold)
    }

    private fun JSONObject.requireFinite(name: String): Double = getDouble(name).also {
        require(it.isFinite()) { "$name must be finite" }
    }

    private fun JSONObject.requireNumbers(name: String, count: Int): DoubleArray {
        val array = getJSONArray(name)
        require(array.length() == count) { "$name must contain $count values" }
        return DoubleArray(count) { index -> array.getDouble(index).also {
            require(it.isFinite()) { "$name[$index] must be finite" }
        } }
    }

    private fun JSONArray.strings() = List(length()) { getString(it) }
}

internal object ServingSideFeatureNames {
    private val phases = listOf("pre", "contact", "post")
    private val zones = listOf("near", "far")
    internal val courtStatistics = listOf(
        "flowMean", "flowP90", "activeFraction", "largestComponentFraction",
        "componentCountDensity", "centroidX", "centroidY", "flowX", "flowY",
    )
    internal val flightGlobalStatistics = listOf(
        "energyMean", "activeFraction", "centroidX", "centroidY", "spreadX", "spreadY",
        "entropy", "largestComponentFraction", "flowX", "flowY", "divergence",
        "bottomMinusTop", "smallComponentEnergyFraction", "smallComponentCentroidY",
        "smallComponentFlowY",
    )
    internal val flightTrajectoryStatistics = listOf(
        "centroidY", "spreadY", "entropy", "flowY", "divergence", "bottomMinusTop",
        "smallComponentEnergyFraction", "smallComponentCentroidY", "smallComponentFlowY",
    )

    fun courtFlow(): List<String> = buildList {
        phases.forEach { phase ->
            zones.forEach { zone -> courtStatistics.forEach { add("$phase:$zone:$it") } }
            courtStatistics.take(5).forEach { add("$phase:nearMinusFar:$it") }
        }
        zones.forEach { zone ->
            add("contactMinusPre:$zone:flowMean")
            add("contactMinusPre:$zone:activeFraction")
            add("contactMinusPre:$zone:largestComponentFraction")
            add("postMinusContact:$zone:flowMean")
            add("postMinusContact:$zone:activeFraction")
        }
        add("contactDelta:nearMinusFar:flowMean")
        add("contactDelta:nearMinusFar:activeFraction")
        add("contactDelta:nearMinusFar:largestComponentFraction")
    }

    fun flight(rows: Int = 4, columns: Int = 6): List<String> {
        require(rows >= 2 && columns >= 2)
        return buildList {
            listOf("launch", "early", "late").forEach { phase ->
                repeat(rows) { row -> repeat(columns) { column ->
                    add("$phase:grid:r$row:c$column:energy")
                } }
                repeat(rows) { row -> add("$phase:row:r$row:flowY") }
                flightGlobalStatistics.forEach { add("$phase:$it") }
            }
            listOf("launchToEarly", "earlyToLate").forEach { transition ->
                flightTrajectoryStatistics.forEach { add("trajectory:$transition:$it") }
                repeat(rows) { row -> add("trajectory:$transition:row:r$row:energy") }
            }
        }
    }

    fun all(): List<String> = courtFlow().map { "v2:$it" } + flight().map { "flight:$it" }
}

internal object ServingSideModelRunner {
    internal data class CandidateInterval(
        val id: String,
        val start: Double,
        val end: Double,
        val agreement: String?,
    )

    fun candidates(ranges: List<SeedRange>): List<CandidateInterval> {
        val ordered = ranges.mapIndexedNotNull { index, range ->
            if (range.startMs < 0 || range.endMs <= range.startMs) null else CandidateInterval(
                "R${(index + 1).toString().padStart(3, '0')}",
                range.startMs / 1_000.0,
                range.endMs / 1_000.0,
                range.agreement,
            )
        }.sortedWith(compareBy<CandidateInterval> { it.start }.thenBy { it.end }.thenBy { it.id })
        require(ordered.map { it.id }.all(String::isNotBlank) && ordered.map { it.id }.distinct().size == ordered.size) {
            "Serving-side candidates need stable unique IDs"
        }
        return ordered
    }

    fun tiedPercentileRanks(values: DoubleArray, rows: Int, columns: Int): DoubleArray {
        require(rows >= 0 && columns >= 0 && values.size == rows * columns) {
            "Serving-side raw feature dimensions do not match"
        }
        if (rows == 0) return DoubleArray(0)
        if (rows == 1) return DoubleArray(columns) { 0.5 }
        val result = DoubleArray(values.size)
        repeat(columns) { column ->
            val order = (0 until rows).sortedWith(compareBy<Int> { values[it * columns + column] }.thenBy { it })
            var start = 0
            while (start < rows) {
                val value = values[order[start] * columns + column]
                require(value.isFinite()) { "Serving-side raw features must be finite" }
                var end = start + 1
                while (end < rows && values[order[end] * columns + column] == value) end++
                val rank = ((start + end - 1) / 2.0) / (rows - 1)
                for (position in start until end) result[order[position] * columns + column] = rank
                start = end
            }
        }
        return result
    }

    fun nearProbability(rankedFeatures: DoubleArray, runtime: ServingSideRuntime): Double {
        require(rankedFeatures.size == runtime.featureNames.size)
        var logit = runtime.bias
        rankedFeatures.indices.forEach { index ->
            val filled = rankedFeatures[index].takeIf(Double::isFinite) ?: runtime.impute[index]
            logit += (filled - runtime.mean[index]) / runtime.scale[index] * runtime.weights[index]
        }
        return 1.0 / (1.0 + exp(-logit.coerceIn(-30.0, 30.0)))
    }

    fun serveHeadEvidence(
        output: AnalysisTypes.ProductionServeOutput,
        anchor: Double,
        threshold: Double = SERVING_SIDE_SERVE_HEAD_THRESHOLD,
        windowSeconds: Double = SERVING_SIDE_SERVE_HEAD_WINDOW_SECONDS,
    ): ServingSideHeadEvidence {
        val times = output.times()
        val probabilities = output.probabilities()
        require(anchor.isFinite() && anchor >= 0 && windowSeconds.isFinite() && windowSeconds >= 0 &&
            times.isNotEmpty() && probabilities.size == times.size
        ) { "Serving-side serve evidence is not aligned" }
        var nearestIndex = 0
        var nearestDistance = Double.POSITIVE_INFINITY
        val selected = mutableListOf<Int>()
        times.indices.forEach { index ->
            val time = times[index]
            val probability = probabilities[index].toDouble()
            require(time.isFinite() && probability.isFinite() && probability in 0.0..1.0)
            val distance = abs(time - anchor)
            if (distance < nearestDistance) {
                nearestDistance = distance
                nearestIndex = index
            }
            if (distance <= windowSeconds + 1e-9) selected += index
        }
        if (selected.isEmpty()) selected += nearestIndex
        var peak = selected.first()
        selected.drop(1).forEach { index ->
            if (probabilities[index] > probabilities[peak]) peak = index
        }
        val detection = output.detections().minByOrNull { abs(it.time() - anchor) }
        return ServingSideHeadEvidence(
            output.modelId(), threshold, probabilities[peak].toDouble(), times[peak],
            probabilities[peak] >= threshold, detection,
        )
    }

    fun verdict(
        candidate: CandidateInterval,
        ranked: DoubleArray,
        allLabelsV2: AnalysisTypes.ProductionServeOutput,
        previousProduction: AnalysisTypes.ProductionServeOutput,
        runtime: ServingSideRuntime,
    ): ServingSideCandidate {
        val allEvidence = serveHeadEvidence(allLabelsV2, candidate.start)
        val previousEvidence = serveHeadEvidence(previousProduction, candidate.start)
        val probability = nearProbability(ranked, runtime)
        val side = if (probability >= SERVING_SIDE_SIDE_THRESHOLD) ServingSide.NEAR else ServingSide.FAR
        val reasons = mutableListOf<ServingSideReviewReason>()
        val decisionSource: ServingSideDecisionSource
        val isServe: Boolean
        when {
            allEvidence.crossesThreshold || previousEvidence.crossesThreshold -> {
                decisionSource = ServingSideDecisionSource.SERVE_HEAD
                isServe = true
            }
            candidate.agreement == ProductionEnsemble.BOTH_MODELS -> {
                decisionSource = ServingSideDecisionSource.PRODUCTION_RALLY_RECOVERY
                reasons += ServingSideReviewReason.PRODUCTION_RALLY_RECOVERY
                isServe = true
            }
            else -> {
                decisionSource = ServingSideDecisionSource.NONE
                isServe = false
            }
        }
        if (probability >= SERVING_SIDE_FAR_REVIEW_THRESHOLD &&
            probability < SERVING_SIDE_NEAR_REVIEW_THRESHOLD
        ) reasons.add(0, ServingSideReviewReason.SIDE_SCORE)
        val verdict = when {
            !isServe -> ServingSideVerdict.NOT_SERVE
            reasons.isNotEmpty() -> ServingSideVerdict.REVIEW
            side == ServingSide.NEAR -> ServingSideVerdict.NEAR
            else -> ServingSideVerdict.FAR
        }
        return ServingSideCandidate(
            candidate.id, candidate.start, candidate.start, candidate.end, candidate.agreement,
            probability, side, verdict, decisionSource, reasons.toList(), allEvidence, previousEvidence,
        )
    }
}
