package com.volleycut.nativeanalysis

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.UUID

internal enum class EditorExportKind { VIDEO, CHAPTERS, PROJECT }

/** An immutable request captured before launching the destination picker. */
internal sealed interface PendingEditorExport {
    val kind: EditorExportKind
    val projectId: String
    val filename: String

    data class Video(
        val jobId: String,
        override val projectId: String,
        override val filename: String,
        val sourceUri: String,
        val sourceName: String,
        val durationMs: Long,
        val startsMs: List<Long>,
        val endsMs: List<Long>,
        val scoreSnapshot: String,
    ) : PendingEditorExport {
        override val kind = EditorExportKind.VIDEO
        init {
            require(jobId.isNotEmpty() && durationMs > 0 && sourceUri.isNotEmpty())
            require(startsMs.isNotEmpty() && startsMs.size == endsMs.size)
            require(startsMs.indices.all { startsMs[it] >= 0 && endsMs[it] > startsMs[it] && endsMs[it] <= durationMs })
            require((1 until startsMs.size).all { startsMs[it] >= endsMs[it - 1] })
            require(ScoreExportSnapshotJson.decode(scoreSnapshot, durationMs) != null)
        }
    }

    data class Text(
        override val kind: EditorExportKind,
        override val projectId: String,
        override val filename: String,
        val text: String,
    ) : PendingEditorExport {
        init { require(kind != EditorExportKind.VIDEO) }
    }
}

/** Large payloads stay in private files; saved activity state contains only IDs. */
internal class PendingEditorExportStore(filesDir: File) {
    private val directory = File(filesDir, "pending-editor-exports")

    fun save(request: PendingEditorExport): String {
        check(directory.isDirectory || directory.mkdirs()) { "Could not prepare the export request" }
        val id = UUID.randomUUID().toString()
        val target = file(id)
        val temporary = File(directory, "$id.tmp")
        try {
            temporary.writeText(encode(request).toString())
            try {
                Files.move(temporary.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE)
            } catch (_: Exception) {
                Files.move(temporary.toPath(), target.toPath())
            }
        } finally { temporary.delete() }
        return id
    }

    fun load(id: String): PendingEditorExport = try {
        decode(JSONObject(file(id).readText()))
    } catch (failure: Exception) {
        throw IOException("The export request could not be restored. No export was started. Save it again from the review.", failure)
    }

    fun remove(id: String): Boolean = file(id).let { !it.exists() || it.delete() }

    private fun file(id: String): File {
        require(id.matches(Regex("[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")))
        return File(directory, "$id.json")
    }

    private fun encode(request: PendingEditorExport) = JSONObject().apply {
        put("version", 1)
        put("kind", request.kind.name)
        put("projectId", request.projectId)
        put("filename", request.filename)
        when (request) {
            is PendingEditorExport.Video -> {
                put("jobId", request.jobId)
                put("sourceUri", request.sourceUri)
                put("sourceName", request.sourceName)
                put("durationMs", request.durationMs)
                put("startsMs", JSONArray(request.startsMs))
                put("endsMs", JSONArray(request.endsMs))
                put("scoreSnapshot", request.scoreSnapshot)
            }
            is PendingEditorExport.Text -> put("text", request.text)
        }
    }

    private fun decode(json: JSONObject): PendingEditorExport {
        require(json.getInt("version") == 1)
        val kind = EditorExportKind.valueOf(json.getString("kind"))
        val projectId = json.getString("projectId").also { require(it.isNotEmpty()) }
        val filename = json.getString("filename").also { require(it.isNotEmpty()) }
        return if (kind == EditorExportKind.VIDEO) {
            fun longs(name: String): List<Long> = json.getJSONArray(name).let { array ->
                List(array.length()) { array.getLong(it) }
            }
            PendingEditorExport.Video(json.getString("jobId"), projectId, filename, json.getString("sourceUri"), json.getString("sourceName"),
                json.getLong("durationMs"), longs("startsMs"), longs("endsMs"), json.getString("scoreSnapshot"))
        } else PendingEditorExport.Text(kind, projectId, filename, json.getString("text"))
    }
}
