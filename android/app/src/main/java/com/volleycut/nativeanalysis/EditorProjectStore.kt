package com.volleycut.nativeanalysis

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption

/** Stores only enough metadata to reopen the most recent editor after process death. */
internal object EditorProjectStore {
    private const val VERSION = 6
    private const val TAG = "VolleySpliceEditor"
    private const val FILE_NAME = "latest-editor-project.json"

    @Synchronized
    fun save(context: Context, seed: EditorSeed) {
        runCatching {
            val directory = File(context.filesDir, "editor-drafts")
            directory.mkdirs()
            val target = File(directory, FILE_NAME)
            val temporary = File(directory, "$FILE_NAME.tmp")
            temporary.writeText(encode(seed).toString())
            try {
                Files.move(
                    temporary.toPath(), target.toPath(),
                    StandardCopyOption.REPLACE_EXISTING,
                    StandardCopyOption.ATOMIC_MOVE,
                )
            } catch (_: Exception) {
                Files.move(temporary.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING)
            }
        }.onFailure { Log.e(TAG, "Could not save editor project", it) }
    }

    @Synchronized
    fun load(context: Context): EditorSeed? = runCatching {
        val target = File(File(context.filesDir, "editor-drafts"), FILE_NAME)
        if (!target.isFile) return null
        decode(JSONObject(target.readText()))
    }.onFailure { Log.w(TAG, "Could not restore editor project", it) }.getOrNull()

    internal fun encode(seed: EditorSeed) = JSONObject().apply {
        put("version", VERSION)
        put("sourceUri", seed.sourceUri)
        put("displayName", seed.displayName)
        put("durationMs", seed.durationMs)
        put("width", seed.width)
        put("height", seed.height)
        put("rotation", seed.rotation)
        put("gameStartMs", seed.gameStartMs)
        put("gameEndMs", seed.gameEndMs)
        put("analysisRoi", seed.analysisRoi?.let { roi -> JSONObject().apply {
            put("x", roi.x())
            put("y", roi.y())
            put("width", roi.width())
            put("height", roi.height())
            put("label", roi.label())
        } } ?: JSONObject.NULL)
        put("ranges", JSONArray().apply {
            seed.ranges.forEach { range ->
                put(JSONObject().apply {
                    put("startMs", range.startMs)
                    put("endMs", range.endMs)
                    put("confidence", range.confidence.toDouble())
                    put("agreement", range.agreement ?: JSONObject.NULL)
                })
            }
        })
        put("productionServeOutputs", ServingSideJson.encodeServeOutputs(seed.productionServeOutputs))
        put("productionStateOutputs", SideSwitchJson.encodeStateOutputs(seed.productionStateOutputs))
        put("servingSide", seed.servingSide?.let(ServingSideJson::encodeOutput) ?: JSONObject.NULL)
        put("servingSideError", seed.servingSideError ?: JSONObject.NULL)
        put("sideSwitch", seed.sideSwitch?.let(SideSwitchJson::encodeOutput) ?: JSONObject.NULL)
        put("sideSwitchError", seed.sideSwitchError ?: JSONObject.NULL)
        put("sideSwitchEnabled", seed.sideSwitchEnabled)
        put("scoreTrackingInitiallyEnabled", seed.scoreTrackingInitiallyEnabled)
    }

    internal fun decode(json: JSONObject): EditorSeed? {
        if (json.optInt("version") !in 1..VERSION) return null
        val durationMs = json.getLong("durationMs")
        val rangesJson = json.getJSONArray("ranges")
        val ranges = buildList {
            for (index in 0 until rangesJson.length()) {
                val range = rangesJson.getJSONObject(index)
                add(SeedRange(
                    startMs = range.getLong("startMs"),
                    endMs = range.getLong("endMs"),
                    confidence = range.getDouble("confidence").toFloat(),
                    agreement = if (range.isNull("agreement")) null else range.optString("agreement"),
                ))
            }
        }
        return EditorSeed(
            sourceUri = json.getString("sourceUri"),
            displayName = json.getString("displayName"),
            durationMs = durationMs,
            width = json.optInt("width"),
            height = json.optInt("height"),
            rotation = json.optInt("rotation"),
            ranges = ranges,
            gameStartMs = json.optLong("gameStartMs", 0),
            gameEndMs = json.optLong("gameEndMs", durationMs),
            productionServeOutputs = json.optJSONObject("productionServeOutputs")?.let {
                ServingSideJson.decodeServeOutputs(it)
            } ?: AnalysisTypes.ProductionServeOutputs.empty(),
            productionStateOutputs = json.optJSONObject("productionStateOutputs")?.let {
                SideSwitchJson.decodeStateOutputs(it)
            } ?: AnalysisTypes.ProductionStateOutputs.empty(),
            servingSide = json.optJSONObject("servingSide")?.let(ServingSideJson::decodeOutput),
            servingSideError = if (json.isNull("servingSideError")) null
                else json.optString("servingSideError").takeIf(String::isNotBlank),
            sideSwitch = json.optJSONObject("sideSwitch")?.let(SideSwitchJson::decodeOutput),
            sideSwitchError = if (json.isNull("sideSwitchError")) null
                else json.optString("sideSwitchError").takeIf(String::isNotBlank),
            sideSwitchEnabled = if (json.has("sideSwitchEnabled")) {
                json.getBoolean("sideSwitchEnabled")
            } else json.optJSONObject("sideSwitch") != null,
            scoreTrackingInitiallyEnabled = json.optBoolean("scoreTrackingInitiallyEnabled", false),
            analysisRoi = json.optJSONObject("analysisRoi")?.let { roi ->
                runCatching { AnalysisTypes.Roi(
                    roi.getDouble("x"), roi.getDouble("y"),
                    roi.getDouble("width"), roi.getDouble("height"), roi.optString("label"),
                ) }.getOrNull()
            }?.takeIf { roi ->
                listOf(roi.x(), roi.y(), roi.width(), roi.height()).all(Double::isFinite) &&
                    roi.x() >= 0 && roi.y() >= 0 && roi.width() > 0 && roi.height() > 0 &&
                    roi.x() + roi.width() <= 1.000001 && roi.y() + roi.height() <= 1.000001
            },
        ).takeIf { seed ->
            seed.durationMs > 0 && seed.sourceUri.isNotBlank() &&
                seed.gameStartMs >= 0 && seed.gameEndMs <= seed.durationMs &&
                seed.gameEndMs - seed.gameStartMs >= 1_000 && seed.ranges.all {
                it.startMs >= seed.gameStartMs && it.endMs > it.startMs &&
                    it.endMs <= seed.gameEndMs &&
                    it.confidence in 0f..1f &&
                    (it.agreement == null || ProductionEnsemble.isValidAgreement(it.agreement))
            }
        }
    }

    /**
     * Removes the legacy last-editor recovery record when it points at a project being deleted.
     * Native project records superseded this file, but older launch and benchmark flows still use
     * it as a fallback. Leaving it behind causes the deleted last project to be recreated on the
     * next launch.
     */
    @Synchronized
    fun clearIfMatches(context: Context, seed: EditorSeed) {
        if (load(context)?.sourceRevision != seed.sourceRevision) return
        val directory = File(context.filesDir, "editor-drafts")
        listOf(
            File(directory, FILE_NAME),
            File(directory, "$FILE_NAME.tmp"),
        ).forEach { target ->
            if (target.exists() && !target.delete()) {
                Log.w(TAG, "Could not delete ${target.name}")
            }
        }
    }
}
