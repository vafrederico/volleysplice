package com.volleycut.nativeanalysis

import org.json.JSONObject
import java.nio.FloatBuffer
import java.security.MessageDigest
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.exp

internal object NeuralServingContract {
    const val MANIFEST_SHA256 = "ffc0aaf5a46344e51702bf4e4a8c4fbfa02cc8c2240c23e8ea94b3929122aada"
    data class Bundle(val id: String, val fingerprint: String, val directory: String,
                      val encoderSha256: String, val encoderWeightsSha256: String,
                      val sha256: String, val size: Long, val far: DoubleArray, val near: DoubleArray)
    data class Decision(val nearProbability: Double, val side: ServingSide, val accepted: Boolean)
    fun identity(modelId: String): Pair<String, String> = when (modelId) {
        RallyModels.RECALL -> "serving-side-centered-recall-v1" to "ff535f821460f2565e9db4df1ed3b36b7f1f3b0790ee87f05ff9aae5adc6ee76"
        RallyModels.F1 -> "serving-side-centered-f1-v1" to "14472e860198753ebfedd185a315ba7c68f36d124d741481b9fe5146ead0f7d1"
        else -> SERVING_SIDE_MODEL_ID to SERVING_SIDE_MODEL_FINGERPRINT
    }
    fun bundle(bytes: ByteArray, modelId: String, rally: JSONObject): Bundle {
        val canonical = bytes.toString(Charsets.UTF_8).replace("\r\n", "\n").toByteArray()
        require(hash(canonical) == MANIFEST_SHA256) { "The serving-side catalog differs from the frozen production models" }
        val value = JSONObject(canonical.toString(Charsets.UTF_8)).getJSONObject("variants").getJSONObject(RallyModels.variant(modelId))
        require(value.getString("rallyModelId") == modelId && rally.getString("id") == modelId &&
            value.getString("encoderSha256") == rally.getJSONObject("files").getJSONObject("encoder").getString("sha256") &&
            value.getString("encoderWeightsSha256") == rally.getString("encoderWeightsSha256")) { "The serving head and rally encoder do not match" }
        val file = value.getJSONObject("file"); val seeds = value.getJSONArray("seeds")
        return Bundle(value.getString("id"), value.getString("fingerprint"), value.getString("directory"),
            value.getString("encoderSha256"), value.getString("encoderWeightsSha256"), file.getString("sha256"), file.getLong("sizeBytes"),
            DoubleArray(3) { seeds.getJSONObject(it).getDouble("farCutoff") }, DoubleArray(3) { seeds.getJSONObject(it).getDouble("nearCutoff") })
    }
    fun hash(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it.toInt() and 255) }
    fun decision(logits: FloatArray, bundle: Bundle): Decision {
        require(logits.size == 6 && logits.all(Float::isFinite)) { "Invalid serving-side scores" }
        val routes = DoubleArray(3); val decisions = arrayOfNulls<ServingSide>(3)
        repeat(3) { seed ->
            fun sigmoid(x: Float) = 1 / (1 + exp(-x.toDouble().coerceIn(-35.0, 35.0)))
            val far = sigmoid(logits[seed * 2]); val near = sigmoid(logits[seed * 2 + 1]); val mass = far + near
            require(mass > 0 && mass <= 1 + 1e-12) { "Serving-side evidence exceeds its available mass" }
            val route = near / mass; val side = if (route >= .5) ServingSide.NEAR else ServingSide.FAR
            val confidence = mass.coerceAtMost(1.0) * if (side == ServingSide.NEAR) route else 1 - route
            val cutoff = if (side == ServingSide.NEAR) bundle.near[seed] else bundle.far[seed]
            routes[seed] = route; decisions[seed] = if (confidence > 0 && confidence >= cutoff) side else null
        }
        val near = routes.sum() / 3
        return Decision(near, if (near >= .5) ServingSide.NEAR else ServingSide.FAR,
            decisions[0] != null && decisions.all { it == decisions[0] })
    }
    fun apply(candidate: ServingSideCandidate, decision: Decision): ServingSideCandidate {
        val reasons = candidate.reviewReasons.filter { it != ServingSideReviewReason.SIDE_SCORE }.toMutableList()
        if (!decision.accepted) reasons.add(0, ServingSideReviewReason.SIDE_SCORE)
        return candidate.copy(nearProbability = decision.nearProbability, side = decision.side, reviewReasons = reasons,
            verdict = when {
                candidate.verdict == ServingSideVerdict.NOT_SERVE -> ServingSideVerdict.NOT_SERVE
                reasons.isNotEmpty() -> ServingSideVerdict.REVIEW
                decision.side == ServingSide.NEAR -> ServingSideVerdict.NEAR
                else -> ServingSideVerdict.FAR
            })
    }
    fun features(raw: DoubleArray, rows: Int): FloatArray {
        require(rows >= 0 && raw.size == rows * 237 && raw.all(Double::isFinite))
        val ranks = ServingSideModelRunner.tiedPercentileRanks(raw, rows, 237)
        val names = ServingSideFeatureNames.all()
        val quality = listOf("launch", "early", "late").flatMap { phase ->
            listOf("energyMean", "activeFraction", "smallComponentEnergyFraction", "flowY").map { names.indexOf("flight:$phase:$it") }
        }
        return FloatArray(rows * 249) { index ->
            val row = index / 249; val column = index % 249
            if (column < 237) ranks[row * 237 + column].toFloat() else {
                val value = raw[row * 237 + quality[column - 237]]
                (if ((column - 237) % 4 == 3) abs(value) else value).toFloat()
            }
        }.also { require(it.all(Float::isFinite)) }
    }
    fun embeddingTimes(anchors: DoubleArray, duration: Double): DoubleArray {
        require(duration.isFinite() && duration > 0 && anchors.all { it.isFinite() && it >= 0 && it < duration })
        val ticks = sortedSetOf<Double>()
        anchors.forEach { anchor -> repeat(13) { frame ->
            val requested = anchor - 3 + frame * .5
            if (requested >= 0 && requested < duration) {
                val tick = ceil(requested * 2 - .5) / 2
                if (tick >= 0 && tick < duration) ticks.add(tick)
            }
        } }
        return ticks.toDoubleArray()
    }
    fun window(times: DoubleArray, tokens: FloatBuffer, anchor: Double, duration: Double): FloatArray {
        require(anchor.isFinite() && anchor >= 0 && anchor < duration && times.isNotEmpty() && tokens.limit() == times.size * 3840 &&
            times.all(Double::isFinite) && times.asList().zipWithNext().all { it.first < it.second })
        val values = FloatArray(13 * 3843)
        repeat(13) { frame ->
            val offset = -3 + frame * .5; val requested = anchor + offset; val destination = frame * 3843
            values[destination + 3841] = (offset / 3).toFloat()
            if (requested >= 0 && requested < duration) {
                val found = times.binarySearch(requested); val later = (if (found >= 0) found else -found - 1).coerceAtMost(times.lastIndex)
                val earlier = (later - 1).coerceAtLeast(0)
                val chosen = if (abs(times[earlier] - requested) <= abs(times[later] - requested)) earlier else later
                if (abs(times[chosen] - requested) <= .250001) {
                    repeat(3840) { values[destination + it] = tokens.get(chosen * 3840 + it) }
                    values[destination + 3840] = 1f; values[destination + 3842] = ((times[chosen] - requested) / .25).toFloat()
                }
            }
        }
        require(values[6 * 3843 + 3840] == 1f) { "The serving anchor has no qualified image sample" }
        return values
    }
}
