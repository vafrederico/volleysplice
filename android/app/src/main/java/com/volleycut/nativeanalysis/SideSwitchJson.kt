package com.volleycut.nativeanalysis

import org.json.JSONArray
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Base64

internal object SideSwitchJson {
    // Match the portable feedback numeric limit before decoding or allocating a matrix.
    private const val MAX_NUMERIC_VALUES = 25_000_000

    /** Portable evidence may belong to an archived model; its data contract must still match. */
    fun decodeFeedback(json: JSONObject, duration: Double): SideSwitchOutput {
        val features = json.getJSONObject("features")
        val rows = integer(features.get("rows"), "sideSwitch.features.rows")
        val columns = integer(features.get("columns"), "sideSwitch.features.columns")
        require(columns == SIDE_SWITCH_FEATURE_COLUMNS && rows.toLong() * columns <= MAX_NUMERIC_VALUES) {
            "sideSwitch.features has unsupported dimensions"
        }
        val values = features.getJSONObject("values")
        val shape = values.getJSONArray("shape")
        require(values.get("encoding") == "base64" && values.get("dataType") == "float64" &&
            values.get("byteOrder") == "little-endian" && shape.length() == 2 &&
            integer(shape.get(0), "sideSwitch.features.values.shape[0]") == rows &&
            integer(shape.get(1), "sideSwitch.features.values.shape[1]") == columns
        ) { "sideSwitch.features.values must be a matching little-endian float64 matrix" }
        val raw = values.get("data")
        require(raw is String) { "sideSwitch.features.values.data must be base64 text" }
        val compact = JSONObject()
            .put("modelId", json.get("modelId")).put("modelFingerprint", json.get("modelFingerprint"))
            .put("featureVersion", json.get("featureVersion")).put("candidateContract", json.get("candidateContract"))
            .put("candidates", json.getJSONArray("candidates"))
            .put("rows", rows).put("columns", columns).put("features", raw)
        val output = requireNotNull(decodeOutput(compact)) { "sideSwitch contains invalid output" }
        require(output.candidates.all { it.timestamp <= duration }) {
            "sideSwitch candidate timestamp exceeds source duration"
        }
        return output
    }

    fun decodeWarnings(json: JSONObject): List<String> {
        if (!json.has("warnings")) return emptyList()
        val warnings = json.getJSONArray("warnings")
        return List(warnings.length()) { index ->
            val value = warnings.get(index)
            require(value is String) { "Invalid feedback warning" }
            value
        }
    }

    private fun integer(value: Any, field: String): Int {
        require(value is Number && value.toDouble().isFinite() &&
            value.toDouble() in 0.0..Int.MAX_VALUE.toDouble() &&
            value.toDouble() == value.toInt().toDouble()
        ) { "$field must be a non-negative integer" }
        return value.toInt()
    }

    private fun text(json: JSONObject, field: String): String = json.get(field).let {
        require(it is String && it.isNotBlank()) { "sideSwitch.$field must be non-empty text" }
        it
    }

    private fun number(json: JSONObject, field: String): Double = json.get(field).let {
        require(it is Number && it.toDouble().isFinite()) { "sideSwitch.$field must be finite" }
        it.toDouble()
    }

    private fun canonicalBytes(value: String, expectedBytes: Int): ByteArray {
        require(value.length.toLong() == ((expectedBytes.toLong() + 2) / 3) * 4) {
            "sideSwitch.features data length does not match its shape"
        }
        val bytes = Base64.getDecoder().decode(value)
        require(bytes.size == expectedBytes && Base64.getEncoder().encodeToString(bytes) == value) {
            "sideSwitch.features must use canonical base64"
        }
        return bytes
    }

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
        val rows = integer(json.get("rows"), "rows")
        val columns = integer(json.get("columns"), "columns")
        require(columns == SIDE_SWITCH_FEATURE_COLUMNS && rows.toLong() * columns <= MAX_NUMERIC_VALUES &&
            candidatesJson.length() <= rows)
        val raw = json.get("features")
        require(raw is String)
        val bytes = canonicalBytes(raw, rows * columns * Double.SIZE_BYTES)
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer()
        SideSwitchOutput(
            modelId = text(json, "modelId"),
            modelFingerprint = text(json, "modelFingerprint"),
            featureVersion = text(json, "featureVersion"),
            candidateContract = text(json, "candidateContract"),
            rows = rows,
            columns = columns,
            features = DoubleArray(buffer.remaining()).also(buffer::get),
            candidates = List(candidatesJson.length()) { index ->
                val item = candidatesJson.getJSONObject(index)
                val ids = item.getJSONArray("sourceRangeIds")
                SideSwitchPrediction(
                    text(item, "id"),
                    number(item, "timestamp"),
                    number(item, "probability"),
                    SideSwitchCandidateKind.fromWireName(text(item, "kind"))
                        ?: error("Invalid team-switch candidate kind"),
                    List(ids.length()) { idIndex ->
                        ids.get(idIndex).also { require(it is String && it.isNotBlank()) } as String
                    },
                )
            },
        ).takeIf(::validOutput)
    }.getOrNull()

    fun validOutput(value: SideSwitchOutput): Boolean =
        value.modelId.isNotBlank() && value.modelFingerprint.isNotBlank() &&
            value.featureVersion == SIDE_SWITCH_FEATURE_VERSION &&
            value.candidateContract == SIDE_SWITCH_CANDIDATE_CONTRACT &&
            value.columns == SIDE_SWITCH_FEATURE_COLUMNS &&
            value.rows >= 0 && value.rows.toLong() * value.columns <= MAX_NUMERIC_VALUES &&
            value.features.size.toLong() == value.rows.toLong() * value.columns &&
            value.features.all(Double::isFinite) &&
            value.candidates.size <= value.rows &&
            value.candidates.map { it.id }.distinct().size == value.candidates.size &&
            value.candidates.all {
                it.id.isNotBlank() && it.timestamp.isFinite() && it.timestamp >= 0 &&
                    it.probability.isFinite() && it.probability in 0.0..1.0 &&
                    it.sourceRangeIds.all(String::isNotBlank)
            }

    fun isCurrentOutput(value: SideSwitchOutput): Boolean = validOutput(value) &&
        value.modelId == SIDE_SWITCH_MODEL_ID && value.modelFingerprint == SIDE_SWITCH_MODEL_FINGERPRINT

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
