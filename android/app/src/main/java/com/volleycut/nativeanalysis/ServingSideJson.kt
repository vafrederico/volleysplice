package com.volleycut.nativeanalysis

import org.json.JSONArray
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Base64

internal object ServingSideJson {
    fun encodeServeOutputs(value: AnalysisTypes.ProductionServeOutputs) = JSONObject().apply {
        put("allLabelsV2", encodeServeOutput(value.allLabelsV2()))
        put("previousProduction", encodeServeOutput(value.previousProduction()))
    }

    fun decodeServeOutputs(json: JSONObject) = AnalysisTypes.ProductionServeOutputs(
        decodeServeOutput(json.getJSONObject("allLabelsV2")),
        decodeServeOutput(json.getJSONObject("previousProduction")),
    ).also {
        require(it.allLabelsV2().modelId() == FeatureSchema.ALL_LABELS_V2_MODEL_ID)
        require(it.previousProduction().modelId() == FeatureSchema.PREVIOUS_PRODUCTION_MODEL_ID)
        require(it.allLabelsV2().times().contentEquals(it.previousProduction().times()))
    }

    fun encodeOutput(value: ServingSideOutput) = JSONObject().apply {
        put("modelId", value.modelId)
        put("modelFingerprint", value.modelFingerprint)
        put("featureVersion", value.featureVersion)
        put("anchorContract", value.anchorContract)
        put("rows", value.rows)
        put("columns", value.columns)
        put("rawFeatures", encodeDoubles(value.rawFeatures))
        put("candidates", JSONArray().apply {
            value.candidates.forEach { candidate -> put(JSONObject().apply {
                put("id", candidate.id)
                put("anchor", candidate.anchor)
                put("intervalStart", candidate.intervalStart)
                put("intervalEnd", candidate.intervalEnd)
                put("agreement", candidate.agreement ?: JSONObject.NULL)
                put("nearProbability", candidate.nearProbability)
                put("side", candidate.side.wireName)
                put("verdict", candidate.verdict.wireName)
                put("serveDecisionSource", candidate.serveDecisionSource.wireName)
                put("reviewReasons", JSONArray(candidate.reviewReasons.map { it.wireName }))
                put("allLabelsV2Evidence", encodeEvidence(candidate.allLabelsV2Evidence))
                put("previousProductionEvidence", encodeEvidence(candidate.previousProductionEvidence))
            }) }
        })
    }

    fun decodeOutput(json: JSONObject): ServingSideOutput? = runCatching {
        val rows = json.getInt("rows")
        val columns = json.getInt("columns")
        val candidatesJson = json.getJSONArray("candidates")
        ServingSideOutput(
            modelId = json.getString("modelId"),
            modelFingerprint = json.getString("modelFingerprint"),
            featureVersion = json.getString("featureVersion"),
            anchorContract = json.getString("anchorContract"),
            rows = rows,
            columns = columns,
            rawFeatures = decodeDoubles(json.getString("rawFeatures")),
            candidates = buildList {
                repeat(candidatesJson.length()) { index ->
                    val item = candidatesJson.getJSONObject(index)
                    val reasons = item.getJSONArray("reviewReasons")
                    add(ServingSideCandidate(
                        item.getString("id"), item.getDouble("anchor"),
                        item.getDouble("intervalStart"), item.getDouble("intervalEnd"),
                        if (item.isNull("agreement")) null else item.getString("agreement"),
                        item.getDouble("nearProbability"),
                        ServingSide.fromWireName(item.getString("side")) ?: error("Invalid side"),
                        ServingSideVerdict.fromWireName(item.getString("verdict")) ?: error("Invalid verdict"),
                        ServingSideDecisionSource.fromWireName(item.getString("serveDecisionSource"))
                            ?: error("Invalid source"),
                        List(reasons.length()) { reason ->
                            ServingSideReviewReason.fromWireName(reasons.getString(reason))
                                ?: error("Invalid review reason")
                        },
                        decodeEvidence(item.getJSONObject("allLabelsV2Evidence")),
                        decodeEvidence(item.getJSONObject("previousProductionEvidence")),
                    ))
                }
            },
        ).takeIf { output ->
            output.modelId == SERVING_SIDE_MODEL_ID &&
                output.modelFingerprint == SERVING_SIDE_MODEL_FINGERPRINT &&
                output.featureVersion == SERVING_SIDE_FEATURE_VERSION &&
                output.anchorContract == SERVING_SIDE_ANCHOR_CONTRACT &&
                output.columns == SERVING_SIDE_FEATURE_COLUMNS &&
                output.rows == output.candidates.size &&
                output.rawFeatures.size == output.rows * output.columns &&
                output.rawFeatures.all(Double::isFinite) &&
                output.candidates.all { candidate ->
                    candidate.id.isNotBlank() && candidate.anchor.isFinite() &&
                        candidate.intervalStart.isFinite() && candidate.intervalEnd.isFinite() &&
                        candidate.anchor == candidate.intervalStart &&
                        candidate.intervalEnd > candidate.intervalStart &&
                        candidate.nearProbability.isFinite() && candidate.nearProbability in 0.0..1.0 &&
                        candidate.side != ServingSide.REVIEW &&
                        (candidate.agreement == null || ProductionEnsemble.isValidAgreement(candidate.agreement)) &&
                        candidate.allLabelsV2Evidence.modelId == FeatureSchema.ALL_LABELS_V2_MODEL_ID &&
                        candidate.previousProductionEvidence.modelId == FeatureSchema.PREVIOUS_PRODUCTION_MODEL_ID &&
                        validEvidence(candidate.allLabelsV2Evidence) &&
                        validEvidence(candidate.previousProductionEvidence)
                }
        }
    }.getOrNull()

    private fun encodeServeOutput(value: AnalysisTypes.ProductionServeOutput) = JSONObject().apply {
        put("modelId", value.modelId())
        put("times", encodeDoubles(value.times()))
        put("probabilities", encodeFloats(value.probabilities()))
        put("detections", JSONArray().apply {
            value.detections().forEach { put(JSONObject().apply {
                put("time", it.time())
                put("confidence", it.confidence().toDouble())
            }) }
        })
    }

    private fun decodeServeOutput(json: JSONObject): AnalysisTypes.ProductionServeOutput {
        val detections = json.optJSONArray("detections") ?: JSONArray()
        return AnalysisTypes.ProductionServeOutput(
            json.getString("modelId"),
            decodeDoubles(json.getString("times")),
            decodeFloats(json.getString("probabilities")),
            List(detections.length()) { index -> detections.getJSONObject(index).let {
                AnalysisTypes.Serve(it.getDouble("time"), it.getDouble("confidence").toFloat())
            } },
        ).also {
            require(it.times().size == it.probabilities().size)
            require(it.times().all(Double::isFinite) &&
                it.times().indices.all { index -> index == 0 || it.times()[index] > it.times()[index - 1] })
            require(it.probabilities().all { probability -> probability.isFinite() && probability in 0f..1f })
            require(it.detections().all { detection ->
                detection.time().isFinite() && detection.time() >= 0 &&
                    detection.confidence().isFinite() && detection.confidence() in 0f..1f
            })
        }
    }

    private fun encodeEvidence(value: ServingSideHeadEvidence) = JSONObject().apply {
        put("modelId", value.modelId)
        put("threshold", value.threshold)
        put("peakProbability", value.peakProbability)
        put("peakTime", value.peakTime)
        put("crossesThreshold", value.crossesThreshold)
        put("nearestDetection", value.nearestDetection?.let { JSONObject().apply {
            put("time", it.time())
            put("confidence", it.confidence().toDouble())
        }} ?: JSONObject.NULL)
    }

    private fun decodeEvidence(json: JSONObject): ServingSideHeadEvidence {
        val detection = json.optJSONObject("nearestDetection")?.let {
            AnalysisTypes.Serve(it.getDouble("time"), it.getDouble("confidence").toFloat())
        }
        return ServingSideHeadEvidence(
            json.getString("modelId"), json.getDouble("threshold"),
            json.getDouble("peakProbability"), json.getDouble("peakTime"),
            json.getBoolean("crossesThreshold"), detection,
        )
    }

    private fun validEvidence(value: ServingSideHeadEvidence): Boolean =
        value.modelId.isNotBlank() && value.threshold.isFinite() && value.threshold in 0.0..1.0 &&
            value.peakProbability.isFinite() && value.peakProbability in 0.0..1.0 &&
            value.peakTime.isFinite() && value.peakTime >= 0 &&
            (value.nearestDetection == null ||
                (value.nearestDetection.time().isFinite() && value.nearestDetection.time() >= 0 &&
                    value.nearestDetection.confidence().isFinite() &&
                    value.nearestDetection.confidence() in 0f..1f))

    private fun encodeDoubles(values: DoubleArray): String = Base64.getEncoder().encodeToString(
        ByteBuffer.allocate(values.size * Double.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)
            .also { it.asDoubleBuffer().put(values) }.array(),
    )

    private fun decodeDoubles(value: String): DoubleArray {
        val bytes = Base64.getDecoder().decode(value)
        require(bytes.size % Double.SIZE_BYTES == 0)
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer()
        return DoubleArray(buffer.remaining()).also(buffer::get)
    }

    private fun encodeFloats(values: FloatArray): String = Base64.getEncoder().encodeToString(
        ByteBuffer.allocate(values.size * Float.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)
            .also { it.asFloatBuffer().put(values) }.array(),
    )

    private fun decodeFloats(value: String): FloatArray {
        val bytes = Base64.getDecoder().decode(value)
        require(bytes.size % Float.SIZE_BYTES == 0)
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(buffer.remaining()).also(buffer::get)
    }
}
