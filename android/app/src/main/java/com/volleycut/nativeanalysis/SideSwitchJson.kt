package com.volleycut.nativeanalysis

import org.json.JSONArray
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Base64

internal object SideSwitchJson {
    fun encodeStateOutputs(value: AnalysisTypes.ProductionStateOutputs) = JSONObject().apply {
        put("allLabelsV2", encodeStateOutput(value.allLabelsV2()))
        put("previousProduction", encodeStateOutput(value.previousProduction()))
    }

    fun decodeStateOutputs(json: JSONObject): AnalysisTypes.ProductionStateOutputs =
        AnalysisTypes.ProductionStateOutputs(
            decodeStateOutput(json.getJSONObject("allLabelsV2")),
            decodeStateOutput(json.getJSONObject("previousProduction")),
        ).also {
            require(it.allLabelsV2().modelId() == FeatureSchema.ALL_LABELS_V2_MODEL_ID)
            require(it.previousProduction().modelId() == FeatureSchema.PREVIOUS_PRODUCTION_MODEL_ID)
            if (it.allLabelsV2().times().isEmpty() && it.previousProduction().times().isEmpty()) {
                require(it.allLabelsV2().rallyProbabilities().isEmpty())
                require(it.allLabelsV2().deadStateProbabilities().isEmpty())
                require(it.previousProduction().rallyProbabilities().isEmpty())
                require(it.previousProduction().deadStateProbabilities().isEmpty())
            } else SideSwitchModelRunner.validateStateOutputs(it)
        }

    fun encodeOutput(value: SideSwitchOutput) = JSONObject().apply {
        put("modelId", value.modelId)
        put("modelFingerprint", value.modelFingerprint)
        put("featureVersion", value.featureVersion)
        put("candidateContract", value.candidateContract)
        put("rows", value.rows)
        put("columns", value.columns)
        put("features", encodeDoubles(value.features))
        put("candidates", JSONArray().apply {
            value.candidates.forEach { candidate -> put(JSONObject().apply {
                put("id", candidate.id)
                put("timestamp", candidate.timestamp)
                put("probability", candidate.probability)
                put("kind", candidate.kind.wireName)
                put("sourceRangeIds", JSONArray(candidate.sourceRangeIds))
            }) }
        })
    }

    fun decodeOutput(json: JSONObject): SideSwitchOutput? = runCatching {
        val candidatesJson = json.getJSONArray("candidates")
        SideSwitchOutput(
            modelId = json.getString("modelId"),
            modelFingerprint = json.getString("modelFingerprint"),
            featureVersion = json.getString("featureVersion"),
            candidateContract = json.getString("candidateContract"),
            rows = json.getInt("rows"),
            columns = json.getInt("columns"),
            features = decodeDoubles(json.getString("features")),
            candidates = List(candidatesJson.length()) { index ->
                val item = candidatesJson.getJSONObject(index)
                val ids = item.getJSONArray("sourceRangeIds")
                SideSwitchPrediction(
                    item.getString("id"),
                    item.getDouble("timestamp"),
                    item.getDouble("probability"),
                    SideSwitchCandidateKind.fromWireName(item.getString("kind"))
                        ?: error("Invalid team-switch candidate kind"),
                    List(ids.length()) { ids.getString(it) },
                )
            },
        ).takeIf(::validOutput)
    }.getOrNull()

    fun validOutput(value: SideSwitchOutput): Boolean =
        value.modelId == SIDE_SWITCH_MODEL_ID &&
            value.modelFingerprint == SIDE_SWITCH_MODEL_FINGERPRINT &&
            value.featureVersion == SIDE_SWITCH_FEATURE_VERSION &&
            value.candidateContract == SIDE_SWITCH_CANDIDATE_CONTRACT &&
            value.columns == SIDE_SWITCH_FEATURE_COLUMNS &&
            value.rows >= 0 && value.features.size == value.rows * value.columns &&
            value.features.all(Double::isFinite) &&
            value.candidates.size <= value.rows &&
            value.candidates.map { it.id }.distinct().size == value.candidates.size &&
            value.candidates.all {
                it.id.isNotBlank() && it.timestamp.isFinite() && it.timestamp >= 0 &&
                    it.probability.isFinite() && it.probability in 0.0..1.0 &&
                    it.sourceRangeIds.isNotEmpty() && it.sourceRangeIds.all(String::isNotBlank)
            }

    private fun encodeStateOutput(value: AnalysisTypes.ProductionStateOutput) = JSONObject().apply {
        put("modelId", value.modelId())
        put("times", encodeDoubles(value.times()))
        put("rallyProbabilities", encodeFloats(value.rallyProbabilities()))
        put("deadStateProbabilities", encodeFloats(value.deadStateProbabilities()))
    }

    private fun decodeStateOutput(json: JSONObject) = AnalysisTypes.ProductionStateOutput(
        json.getString("modelId"),
        decodeDoubles(json.getString("times")),
        decodeFloats(json.getString("rallyProbabilities")),
        decodeFloats(json.getString("deadStateProbabilities")),
    )

    private fun encodeDoubles(values: DoubleArray): String = Base64.getEncoder().encodeToString(
        ByteBuffer.allocate(values.size * Double.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)
            .also { it.asDoubleBuffer().put(values) }.array(),
    )

    private fun decodeDoubles(value: String): DoubleArray {
        val bytes = Base64.getDecoder().decode(value)
        require(bytes.size % Double.SIZE_BYTES == 0)
        val buffer = ByteBuffer.wrap(bytes)
            .order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer()
        return DoubleArray(buffer.remaining()).also(buffer::get)
    }

    private fun encodeFloats(values: FloatArray): String = Base64.getEncoder().encodeToString(
        ByteBuffer.allocate(values.size * Float.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)
            .also { it.asFloatBuffer().put(values) }.array(),
    )

    private fun decodeFloats(value: String): FloatArray {
        val bytes = Base64.getDecoder().decode(value)
        require(bytes.size % Float.SIZE_BYTES == 0)
        val buffer = ByteBuffer.wrap(bytes)
            .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(buffer.remaining()).also(buffer::get)
    }
}
