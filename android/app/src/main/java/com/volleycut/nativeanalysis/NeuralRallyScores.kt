package com.volleycut.nativeanalysis

import org.json.JSONArray
import org.json.JSONObject

/** Small, source-timestamped classifier output; contains no image embeddings. */
internal data class NeuralRallyScores(
    val modelId: String,
    val timestamps: DoubleArray,
    val probabilities: FloatArray,
) {
    init {
        require(RallyModels.isNeural(modelId)) { "Unknown neural score model" }
        require(timestamps.isNotEmpty() && timestamps.all { it.isFinite() && it >= 0 } &&
            timestamps.indices.drop(1).all { timestamps[it] > timestamps[it - 1] }) {
            "Invalid neural score timeline"
        }
        require(probabilities.size == timestamps.size * 4 && probabilities.all { it in 0f..1f }) {
            "Invalid neural score probabilities"
        }
    }

    fun encode() = JSONObject().apply {
        put("modelId", modelId)
        put("heads", JSONArray(HEADS))
        put("timestamps", ModelFeedbackExporter.encode(timestamps, intArrayOf(timestamps.size)))
        put("probabilities", ModelFeedbackExporter.encode(probabilities, intArrayOf(timestamps.size, 4)))
    }

    companion object {
        private val HEADS = listOf("live", "serve", "end", "keep")

        fun decode(json: JSONObject, expectedModelId: String, start: Double, end: Double): NeuralRallyScores {
            require(json.getString("modelId") == expectedModelId) { "Neural score model mismatch" }
            val heads = json.getJSONArray("heads")
            require(heads.length() == HEADS.size && HEADS.indices.all { heads.getString(it) == HEADS[it] }) {
                "Unknown neural score heads"
            }
            val timesJson = json.getJSONObject("timestamps")
            val shape = timesJson.getJSONArray("shape")
            require(shape.length() == 1 && timesJson.getString("dataType") == "float64")
            val times = ModelFeedbackImporter.decodeNumeric(timesJson, intArrayOf(shape.getInt(0)))
            require(times.all { it >= start && it < end }) { "Neural scores are outside the game window" }
            val probabilitiesJson = json.getJSONObject("probabilities")
            require(probabilitiesJson.getString("dataType") == "float32")
            val probabilities = ModelFeedbackImporter.decodeNumeric(probabilitiesJson, intArrayOf(times.size, 4))
            return NeuralRallyScores(expectedModelId, times, FloatArray(probabilities.size) { probabilities[it].toFloat() })
        }
    }
}
