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
    private const val VERSION = 1
    private const val TAG = "VolleyCutEditor"
    private const val FILE_NAME = "latest-editor-project.json"

    fun save(context: Context, seed: EditorSeed) {
        runCatching {
            val directory = File(context.filesDir, "editor-drafts")
            directory.mkdirs()
            val target = File(directory, FILE_NAME)
            val temporary = File(directory, "$FILE_NAME.tmp")
            temporary.writeText(JSONObject().apply {
                put("version", VERSION)
                put("sourceUri", seed.sourceUri)
                put("displayName", seed.displayName)
                put("durationMs", seed.durationMs)
                put("width", seed.width)
                put("height", seed.height)
                put("rotation", seed.rotation)
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
            }.toString())
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

    fun load(context: Context): EditorSeed? = runCatching {
        val target = File(File(context.filesDir, "editor-drafts"), FILE_NAME)
        if (!target.isFile) return null
        val json = JSONObject(target.readText())
        if (json.optInt("version") != VERSION) return null
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
        EditorSeed(
            sourceUri = json.getString("sourceUri"),
            displayName = json.getString("displayName"),
            durationMs = durationMs,
            width = json.optInt("width"),
            height = json.optInt("height"),
            rotation = json.optInt("rotation"),
            ranges = ranges,
        ).takeIf { seed ->
            seed.durationMs > 0 && seed.sourceUri.isNotBlank() && seed.ranges.all {
                it.startMs >= 0 && it.endMs > it.startMs && it.endMs <= seed.durationMs &&
                    it.confidence in 0f..1f &&
                    (it.agreement == null || ProductionEnsemble.isValidAgreement(it.agreement))
            }
        }
    }.onFailure { Log.w(TAG, "Could not restore editor project", it) }.getOrNull()
}
