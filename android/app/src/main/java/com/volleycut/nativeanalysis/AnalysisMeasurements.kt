package com.volleycut.nativeanalysis

import org.json.JSONArray
import org.json.JSONObject

internal enum class AnalysisRunKind(val wireName: String, val label: String) {
    PROJECT("project", "Full project inference"),
    SCORE_SPECIALISTS("score-specialists", "Score-tracking inference");

    companion object {
        fun fromWireName(value: String) = entries.firstOrNull { it.wireName == value }
    }
}

/** Durable timing data saved with the native project rather than left only in logcat. */
internal data class AnalysisRunMeasurements(
    val kind: AnalysisRunKind,
    val completedAtMs: Long,
    val succeeded: Boolean,
    val error: String? = null,
    val totalMilliseconds: Double,
    val stageMilliseconds: Map<String, Double>,
    val profileMilliseconds: Map<String, Double> = emptyMap(),
    val counters: Map<String, Long> = emptyMap(),
) {
    companion object {
        fun fromResult(result: AnalysisTypes.AnalysisResult) = AnalysisRunMeasurements(
            kind = AnalysisRunKind.PROJECT,
            completedAtMs = System.currentTimeMillis(),
            succeeded = true,
            totalMilliseconds = result.totalMilliseconds().toDouble(),
            stageMilliseconds = LinkedHashMap<String, Double>().apply {
                result.stageMilliseconds().forEach { (key, value) -> put(key, value.toDouble()) }
            },
            profileMilliseconds = LinkedHashMap(result.profileMilliseconds()),
            counters = linkedMapOf(
                "sample_rows" to result.sampleRows().toLong(),
                "decoded_video_frames" to result.decodedSourceFrames().toLong(),
                "decoder_output_frames" to result.decoderOutputFrames().toLong(),
                "decoded_audio_frames" to result.decodedAudioFrames(),
                "audio_feature_frames" to result.audioFeatureFrames().toLong(),
                "cache_bytes" to result.featureCache().bytes(),
            ),
        )

        fun fromScoreProfile(profile: ScoreSpecialistInference.Profile) = AnalysisRunMeasurements(
            kind = AnalysisRunKind.SCORE_SPECIALISTS,
            completedAtMs = System.currentTimeMillis(),
            succeeded = true,
            totalMilliseconds = profile.totalMilliseconds,
            stageMilliseconds = linkedMapOf(
                "score_planning" to profile.planningMilliseconds,
                "score_frame_decode" to profile.sharedDecode.wallMilliseconds(),
                "serving_side" to profile.servingEvaluationMilliseconds,
                "side_switch" to profile.sideSwitchEvaluationMilliseconds,
            ).filterValues { it > 0.0 },
            profileMilliseconds = profile.measurementProfile(),
            counters = profile.measurementCounters(),
        )
    }
}

internal fun replaceAnalysisMeasurement(
    current: List<AnalysisRunMeasurements>,
    measurement: AnalysisRunMeasurements,
): List<AnalysisRunMeasurements> = (current.filterNot { it.kind == measurement.kind } + measurement)
    .sortedBy { it.kind.ordinal }

internal object AnalysisMeasurementsJson {
    fun encode(runs: List<AnalysisRunMeasurements>) = JSONArray().apply {
        runs.forEach { run ->
            put(JSONObject().apply {
                put("kind", run.kind.wireName)
                put("completedAtMs", run.completedAtMs)
                put("succeeded", run.succeeded)
                put("error", run.error ?: JSONObject.NULL)
                put("totalMilliseconds", run.totalMilliseconds)
                put("stageMilliseconds", encodeDoubles(run.stageMilliseconds))
                put("profileMilliseconds", encodeDoubles(run.profileMilliseconds))
                put("counters", JSONObject().apply {
                    run.counters.forEach { (key, value) -> put(key, value) }
                })
            })
        }
    }

    fun decode(array: JSONArray?): List<AnalysisRunMeasurements> = buildList {
        if (array == null) return@buildList
        for (index in 0 until array.length()) {
            val json = array.optJSONObject(index) ?: continue
            val kind = AnalysisRunKind.fromWireName(json.optString("kind")) ?: continue
            val total = json.optDouble("totalMilliseconds", Double.NaN)
            if (!total.isFinite() || total < 0.0) continue
            add(AnalysisRunMeasurements(
                kind = kind,
                completedAtMs = json.optLong("completedAtMs", 0L),
                succeeded = json.optBoolean("succeeded", true),
                error = if (json.isNull("error")) null else json.optString("error"),
                totalMilliseconds = total,
                stageMilliseconds = decodeDoubles(json.optJSONObject("stageMilliseconds")),
                profileMilliseconds = decodeDoubles(json.optJSONObject("profileMilliseconds")),
                counters = decodeLongs(json.optJSONObject("counters")),
            ))
        }
    }

    private fun encodeDoubles(values: Map<String, Double>) = JSONObject().apply {
        values.forEach { (key, value) -> if (value.isFinite() && value >= 0.0) put(key, value) }
    }

    private fun decodeDoubles(json: JSONObject?): Map<String, Double> = linkedMapOf<String, Double>().apply {
        if (json == null) return@apply
        json.keys().forEach { key ->
            json.optDouble(key, Double.NaN).takeIf { it.isFinite() && it >= 0.0 }
                ?.let { put(key, it) }
        }
    }

    private fun decodeLongs(json: JSONObject?): Map<String, Long> = linkedMapOf<String, Long>().apply {
        if (json == null) return@apply
        json.keys().forEach { key ->
            json.optLong(key, Long.MIN_VALUE).takeIf { it >= 0L }?.let { put(key, it) }
        }
    }
}

internal enum class InferenceStepStatus(val wireName: String) {
    PENDING("pending"),
    RUNNING("running"),
    COMPLETE("complete"),
    ERROR("error");

    companion object {
        fun fromWireName(value: String) = entries.firstOrNull { it.wireName == value }
    }
}

internal data class InferenceStepMeasurement(
    val id: String,
    val label: String,
    val status: InferenceStepStatus,
    val fraction: Double,
    val detail: String,
    val elapsedMilliseconds: Double,
)

/** Maintains the same user-facing inference steps as the production web progress panel. */
internal class InferenceProgressTracker(
    includeCore: Boolean,
    includeServingSide: Boolean,
    includeSideSwitch: Boolean,
    private val nanoTime: () -> Long = System::nanoTime,
) {
    private data class Step(
        val id: String,
        val label: String,
        var status: InferenceStepStatus = InferenceStepStatus.PENDING,
        var fraction: Double = 0.0,
        var detail: String = "Waiting for the previous step",
        var startedNanos: Long? = null,
        var finishedNanos: Long? = null,
    )

    private val steps = buildList {
        if (includeCore) {
            add(Step("video", "Scanning video"))
            add(Step("audio", "Listening for play"))
            add(Step("rally", "Finding rallies"))
        }
        if (includeServingSide) add(Step("serving-side", "Finding serve markers"))
        if (includeSideSwitch) add(Step("side-switch", "Finding team switches"))
    }

    fun update(stage: String, fraction: Double, detail: String): List<InferenceStepMeasurement> {
        val now = nanoTime()
        if (stage == "complete") {
            steps.forEach { complete(it, now) }
            return snapshot(now)
        }
        val target = target(stage, fraction.coerceIn(0.0, 1.0)) ?: return snapshot(now)
        val index = steps.indexOfFirst { it.id == target.first }
        if (index < 0) return snapshot(now)
        for (earlier in 0 until index) complete(steps[earlier], now)
        val step = steps[index]
        if (step.status == InferenceStepStatus.PENDING) {
            step.status = InferenceStepStatus.RUNNING
            step.startedNanos = now
        }
        if (step.status != InferenceStepStatus.ERROR) {
            step.fraction = maxOf(step.fraction, target.second).coerceIn(0.0, 1.0)
            step.detail = detail
            if (step.fraction >= 1.0) complete(step, now)
        }
        return snapshot(now)
    }

    fun markError(id: String, detail: String): List<InferenceStepMeasurement> {
        val now = nanoTime()
        steps.firstOrNull { it.id == id }?.let { step ->
            if (step.startedNanos == null) step.startedNanos = now
            step.finishedNanos = now
            step.status = InferenceStepStatus.ERROR
            step.detail = detail
        }
        return snapshot(now)
    }

    fun snapshot(): List<InferenceStepMeasurement> = snapshot(nanoTime())

    fun failedRun(kind: AnalysisRunKind, error: String): AnalysisRunMeasurements {
        val now = nanoTime()
        val snapshot = snapshot(now)
        val started = steps.mapNotNull { it.startedNanos }.minOrNull() ?: now
        return AnalysisRunMeasurements(
            kind = kind,
            completedAtMs = System.currentTimeMillis(),
            succeeded = false,
            error = error,
            totalMilliseconds = (now - started).coerceAtLeast(0L) / 1_000_000.0,
            stageMilliseconds = snapshot.associateTo(linkedMapOf()) {
                it.id to it.elapsedMilliseconds
            },
        )
    }

    private fun target(stage: String, fraction: Double): Pair<String, Double>? = when (stage) {
        "opening" -> "video" to 0.0
        "video" -> "video" to fraction
        "audio" -> "audio" to fraction
        "normalizing" -> "rally" to (0.05 + 0.25 * fraction)
        "inference" -> "rally" to (0.30 + 0.70 * fraction)
        "score-specialists" -> "serving-side" to if (fraction >= 1.0) 1.0 else 0.02
        "specialist-frames", "serving-side-frames" ->
            "serving-side" to (0.05 + 0.63 * fraction)
        "serving-side-features" -> "serving-side" to (0.68 + 0.30 * fraction)
        "serving-side" -> "serving-side" to if (fraction >= 1.0) 1.0 else 0.68
        "side-switch-features" -> "side-switch" to (0.68 + 0.30 * fraction)
        "side-switch" -> "side-switch" to if (fraction >= 1.0) 1.0 else 0.98
        else -> null
    }

    private fun complete(step: Step, now: Long) {
        if (step.status == InferenceStepStatus.COMPLETE || step.status == InferenceStepStatus.ERROR) return
        if (step.startedNanos == null) step.startedNanos = now
        step.finishedNanos = now
        step.status = InferenceStepStatus.COMPLETE
        step.fraction = 1.0
        if (step.detail == "Waiting for the previous step") step.detail = "${step.label} complete"
    }

    private fun snapshot(now: Long) = steps.map { step ->
        val started = step.startedNanos
        val end = step.finishedNanos ?: now
        InferenceStepMeasurement(
            id = step.id,
            label = step.label,
            status = step.status,
            fraction = step.fraction,
            detail = step.detail,
            elapsedMilliseconds = if (started == null) 0.0
                else (end - started).coerceAtLeast(0L) / 1_000_000.0,
        )
    }
}

internal object InferenceStepMeasurementsJson {
    fun encode(steps: List<InferenceStepMeasurement>): String = JSONArray().apply {
        steps.forEach { step -> put(JSONObject().apply {
            put("id", step.id)
            put("label", step.label)
            put("status", step.status.wireName)
            put("fraction", step.fraction)
            put("detail", step.detail)
            put("elapsedMilliseconds", step.elapsedMilliseconds)
        }) }
    }.toString()

    fun decode(value: String?): List<InferenceStepMeasurement> = runCatching {
        val array = JSONArray(value ?: return emptyList())
        buildList {
            for (index in 0 until array.length()) {
                val json = array.optJSONObject(index) ?: continue
                val status = InferenceStepStatus.fromWireName(json.optString("status")) ?: continue
                add(InferenceStepMeasurement(
                    id = json.optString("id"),
                    label = json.optString("label"),
                    status = status,
                    fraction = json.optDouble("fraction", 0.0).coerceIn(0.0, 1.0),
                    detail = json.optString("detail"),
                    elapsedMilliseconds = json.optDouble("elapsedMilliseconds", 0.0).coerceAtLeast(0.0),
                ))
            }
        }
    }.getOrDefault(emptyList())
}
