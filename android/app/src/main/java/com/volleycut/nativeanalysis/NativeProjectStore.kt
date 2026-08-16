package com.volleycut.nativeanalysis

import android.content.Context
import android.database.Cursor
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption

internal enum class ProjectStatus(val wireName: String) {
    QUEUED("queued"),
    ANALYZING("analyzing"),
    READY("ready"),
    ERROR("error");

    companion object {
        fun fromWireName(value: String) = entries.firstOrNull { it.wireName == value }
    }
}

internal data class ProjectSource(
    val uri: String,
    val name: String,
    val size: Long,
    val lastModified: Long,
    val mimeType: String,
)

internal data class NativeProject(
    val id: String,
    val source: ProjectSource,
    /** Original source identity retained only while a re-linked project still uses its old cache. */
    val featureCacheSource: ProjectSource? = null,
    val media: AnalysisTypes.MediaInfo,
    val analysisWindow: AnalysisTypes.AnalysisWindow = AnalysisTypes.AnalysisWindow.full(media.durationSeconds()),
    val roi: AnalysisTypes.Roi,
    val status: ProjectStatus,
    val ranges: List<SeedRange> = emptyList(),
    val modelId: String = FeatureSchema.MODEL_ID,
    val cacheMode: String = NativeFeatureCache.Mode.USE.wireName(),
    val error: String? = null,
    val createdAtMs: Long,
    val updatedAtMs: Long,
) {
    fun editorSeed(): EditorSeed? = if (status == ProjectStatus.READY) {
        EditorSeed(
            sourceUri = source.uri,
            displayName = source.name,
            durationMs = (media.durationSeconds() * 1_000.0).toLong(),
            width = media.width(),
            height = media.height(),
            rotation = media.rotation(),
            ranges = ranges,
            gameStartMs = secondsToMs(analysisWindow.start()),
            gameEndMs = secondsToMs(analysisWindow.end()),
        )
    } else null
}

/** Atomic, process-safe-enough project records. Analysis itself is serialized by the service. */
internal object NativeProjectStore {
    private const val VERSION = 1
    private const val TAG = "VolleyCutProjects"
    private const val DIRECTORY = "native-projects"
    private const val PREFERENCES = "native-project-selection"
    private const val SELECTED_ID = "selectedProjectId"

    @Synchronized
    fun list(context: Context): List<NativeProject> {
        val directory = directory(context)
        return directory.listFiles { file -> file.extension == "json" }
            .orEmpty()
            .mapNotNull { file ->
                runCatching { decode(JSONObject(file.readText()))?.let(::normalizeStored) }
                    .onFailure { Log.w(TAG, "Could not read ${file.name}", it) }
                    .getOrNull()
            }
            .sortedByDescending { it.updatedAtMs }
    }

    @Synchronized
    fun get(context: Context, id: String): NativeProject? = runCatching {
        val target = projectFile(context, id)
        if (!target.isFile) null else decode(JSONObject(target.readText()))?.let(::normalizeStored)
    }.onFailure { Log.w(TAG, "Could not read project $id", it) }.getOrNull()

    @Synchronized
    fun save(context: Context, project: NativeProject) {
        val directory = directory(context)
        directory.mkdirs()
        val target = projectFile(context, project.id)
        val temporary = File(directory, "${target.name}.tmp")
        temporary.writeText(encode(project).toString())
        try {
            Files.move(
                temporary.toPath(), target.toPath(),
                StandardCopyOption.REPLACE_EXISTING,
                StandardCopyOption.ATOMIC_MOVE,
            )
        } catch (_: Exception) {
            Files.move(temporary.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING)
        }
    }

    @Synchronized
    fun updateStatus(
        context: Context,
        id: String,
        status: ProjectStatus,
        error: String? = null,
    ): NativeProject? {
        val current = get(context, id) ?: return null
        return current.copy(
            status = status,
            error = error,
            updatedAtMs = System.currentTimeMillis(),
        ).also { save(context, it) }
    }

    @Synchronized
    fun complete(context: Context, id: String, result: AnalysisTypes.AnalysisResult): NativeProject? {
        val current = get(context, id) ?: return null
        return current.copy(
            source = current.source.copy(name = result.displayName()),
            featureCacheSource = null,
            media = result.media(),
            roi = result.roi(),
            status = ProjectStatus.READY,
            ranges = result.ranges().map {
                SeedRange(
                    startMs = (it.start() * 1_000.0).toLong(),
                    endMs = (it.end() * 1_000.0).toLong(),
                    confidence = it.confidence(),
                    agreement = it.agreement(),
                )
            },
            modelId = FeatureSchema.MODEL_ID,
            cacheMode = NativeFeatureCache.Mode.USE.wireName(),
            error = null,
            updatedAtMs = System.currentTimeMillis(),
        ).also {
            save(context, it)
            EditorProjectStore.save(context, checkNotNull(it.editorSeed()))
        }
    }

    @Synchronized
    fun delete(context: Context, project: NativeProject) {
        // Delete authoritative and recovery metadata before optional artifact cleanup. A missing
        // source or a cache cleanup failure must never leave (or recreate) a visible project.
        val target = projectFile(context, project.id)
        if (target.exists() && !target.delete()) Log.w(TAG, "Could not delete ${target.name}")
        if (selectedId(context) == project.id) setSelectedId(context, null)
        project.editorSeed()?.let { seed ->
            EditorProjectStore.clearIfMatches(context, seed)
            runCatching { EditorDraftStore(context, seed).clear() }
                .onFailure { Log.w(TAG, "Could not delete editor draft for ${project.id}", it) }
        }
        val cacheSource = project.featureCacheSource ?: project.source
        runCatching {
            NativeFeatureCache.clearEntryWithSourceMetadata(
                context,
                Uri.parse(cacheSource.uri),
                cacheSource.name,
                cacheSource.size,
                cacheSource.lastModified,
                project.media,
                project.roi,
                FeatureSchema.FULL_SOURCE_FRAME_LIMIT,
                project.analysisWindow,
            )
        }.onFailure { Log.w(TAG, "Could not delete feature cache for ${project.id}", it) }
    }

    fun sourceAvailable(context: Context, project: NativeProject): Boolean = runCatching {
        context.contentResolver.openFileDescriptor(Uri.parse(project.source.uri), "r")?.use {
            it.fileDescriptor.valid()
        } == true
    }.getOrDefault(false)

    internal fun replacementMismatch(
        project: NativeProject,
        replacement: ProjectSource,
        replacementMedia: AnalysisTypes.MediaInfo,
    ): String? = when {
        project.source.size >= 0 && replacement.size >= 0 &&
            project.source.size != replacement.size ->
            "That file has a different byte size from the original recording."
        kotlin.math.abs(project.media.durationSeconds() - replacementMedia.durationSeconds()) > .05 ->
            "That file has a different duration from the original recording."
        project.media.width() != replacementMedia.width() ||
            project.media.height() != replacementMedia.height() ||
            project.media.rotation() != replacementMedia.rotation() ->
            "That file has different video dimensions or rotation from the original recording."
        project.media.videoMime() != replacementMedia.videoMime() ->
            "That file uses a different video codec from the original recording."
        project.media.audioMime() != replacementMedia.audioMime() ->
            "That file has a different audio track from the original recording."
        else -> null
    }

    @Synchronized
    fun relink(
        context: Context,
        project: NativeProject,
        replacement: ProjectSource,
        replacementMedia: AnalysisTypes.MediaInfo,
    ): NativeProject {
        replacementMismatch(project, replacement, replacementMedia)?.let {
            throw IllegalArgumentException(it)
        }
        val oldSeed = project.editorSeed()
        val updated = project.copy(
            source = replacement,
            featureCacheSource = project.featureCacheSource ?: project.source,
            updatedAtMs = System.currentTimeMillis(),
        )
        val newSeed = updated.editorSeed()
        if (oldSeed != null && newSeed != null) {
            EditorDraftStore.relink(context, oldSeed, newSeed)
            EditorProjectStore.save(context, newSeed)
        }
        save(context, updated)
        return updated
    }

    fun selectedId(context: Context): String? =
        context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
            .getString(SELECTED_ID, null)

    fun setSelectedId(context: Context, id: String?) {
        context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
            .edit()
            .apply { if (id == null) remove(SELECTED_ID) else putString(SELECTED_ID, id) }
            .apply()
    }

    fun source(context: Context, uri: Uri, fallbackName: String): ProjectSource {
        var name = fallbackName
        var size = -1L
        var modified = -1L
        runCatching {
            context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                if (cursor.moveToFirst()) {
                    name = cursor.string(OpenableColumns.DISPLAY_NAME) ?: fallbackName
                    size = cursor.long(OpenableColumns.SIZE) ?: -1L
                    modified = cursor.long("last_modified")
                        ?: cursor.long("date_modified")
                        ?: -1L
                }
            }
        }
        return ProjectSource(
            uri = uri.toString(),
            name = name,
            size = size,
            lastModified = modified,
            mimeType = context.contentResolver.getType(uri).orEmpty(),
        )
    }

    fun newQueued(
        source: ProjectSource,
        media: AnalysisTypes.MediaInfo,
        roi: AnalysisTypes.Roi,
        requestedWindow: AnalysisTypes.AnalysisWindow = AnalysisTypes.AnalysisWindow.full(media.durationSeconds()),
    ): NativeProject {
        val now = System.currentTimeMillis()
        val analysisWindow = AnalysisTypes.AnalysisWindow.normalize(requestedWindow, media.durationSeconds())
        return NativeProject(
            id = projectId(source, media.durationSeconds(), analysisWindow),
            source = source,
            media = media,
            analysisWindow = analysisWindow,
            roi = roi,
            status = ProjectStatus.QUEUED,
            createdAtMs = now,
            updatedAtMs = now,
        )
    }

    fun findMatching(context: Context, candidate: NativeProject): NativeProject? =
        get(context, candidate.id) ?: list(context).firstOrNull { existing ->
            sameWindow(existing.analysisWindow, candidate.analysisWindow) && (
                existing.source.uri == candidate.source.uri || (
                existing.source.name == candidate.source.name &&
                    existing.source.size >= 0 && existing.source.size == candidate.source.size &&
                    sameModifiedSecond(existing.source.lastModified, candidate.source.lastModified) &&
                    kotlin.math.abs(
                        existing.media.durationSeconds() - candidate.media.durationSeconds()
                    ) < .001
                ))
        }

    fun fromResult(context: Context, result: AnalysisTypes.AnalysisResult): NativeProject {
        val source = source(context, result.source(), result.displayName())
        val now = System.currentTimeMillis()
        return NativeProject(
            id = projectId(source, result.media().durationSeconds()),
            source = source,
            media = result.media(),
            roi = result.roi(),
            status = ProjectStatus.READY,
            ranges = result.ranges().map {
                SeedRange(
                    (it.start() * 1_000).toLong(),
                    (it.end() * 1_000).toLong(),
                    it.confidence(),
                    it.agreement(),
                )
            },
            createdAtMs = now,
            updatedAtMs = now,
        )
    }

    fun fromSeed(context: Context, seed: EditorSeed): NativeProject {
        val uri = Uri.parse(seed.sourceUri)
        val source = source(context, uri, seed.displayName)
        val media = runCatching { AnalysisEngine(context).probe(uri) }.getOrElse {
            AnalysisTypes.MediaInfo(
                seed.durationMs / 1_000.0,
                seed.width,
                seed.height,
                seed.rotation,
                "video/unknown",
                "audio/unknown",
            )
        }
        val now = System.currentTimeMillis()
        val analysisWindow = AnalysisTypes.AnalysisWindow.normalize(
            AnalysisTypes.AnalysisWindow(seed.gameStartMs / 1_000.0, seed.gameEndMs / 1_000.0),
            media.durationSeconds(),
        )
        return NativeProject(
            id = projectId(source, media.durationSeconds(), analysisWindow),
            source = source,
            media = media,
            analysisWindow = analysisWindow,
            roi = AnalysisEngine.inferRoi(seed.displayName),
            status = ProjectStatus.READY,
            ranges = seed.ranges,
            createdAtMs = now,
            updatedAtMs = now,
        )
    }

    fun repairLegacyMetadata(context: Context, project: NativeProject): NativeProject {
        if (project.media.videoMime() != "video/unknown") return project
        return runCatching {
            project.copy(
                media = AnalysisEngine(context).probe(Uri.parse(project.source.uri)),
                roi = AnalysisEngine.inferRoi(project.source.name),
                updatedAtMs = System.currentTimeMillis(),
            ).also { save(context, it) }
        }.getOrDefault(project)
    }

    internal fun projectId(
        source: ProjectSource,
        durationSeconds: Double,
        requestedWindow: AnalysisTypes.AnalysisWindow = AnalysisTypes.AnalysisWindow.full(durationSeconds),
    ): String {
        val analysisWindow = AnalysisTypes.AnalysisWindow.normalize(requestedWindow, durationSeconds)
        val identity = listOf(
            source.name,
            source.size.toString(),
            normalizedModified(source.lastModified).toString(),
            durationSeconds.toString(),
        ).joinToString("\u0000") + if (analysisWindow.isFull(durationSeconds)) "" else {
            "\u0000${analysisWindow.start()}\u0000${analysisWindow.end()}"
        }
        var hash = 0x811c9dc5u
        identity.forEach { character ->
            hash = (hash xor character.code.toUInt()) * 0x01000193u
        }
        return "project-${hash.toString(36)}"
    }

    internal fun encode(project: NativeProject) = JSONObject().apply {
        put("version", VERSION)
        put("id", project.id)
        put("source", JSONObject().apply {
            put("uri", project.source.uri)
            put("name", project.source.name)
            put("size", project.source.size)
            put("lastModified", project.source.lastModified)
            put("mimeType", project.source.mimeType)
        })
        put("featureCacheSource", project.featureCacheSource?.let(::encodeSource) ?: JSONObject.NULL)
        put("media", JSONObject().apply {
            put("durationSeconds", project.media.durationSeconds())
            put("width", project.media.width())
            put("height", project.media.height())
            put("rotation", project.media.rotation())
            put("videoMime", project.media.videoMime())
            put("audioMime", project.media.audioMime())
        })
        put("analysisWindow", JSONObject().apply {
            put("start", project.analysisWindow.start())
            put("end", project.analysisWindow.end())
        })
        put("roi", JSONObject().apply {
            put("x", project.roi.x())
            put("y", project.roi.y())
            put("width", project.roi.width())
            put("height", project.roi.height())
            put("label", project.roi.label())
        })
        put("status", project.status.wireName)
        put("modelId", project.modelId)
        put("cacheMode", project.cacheMode)
        put("error", project.error ?: JSONObject.NULL)
        put("createdAtMs", project.createdAtMs)
        put("updatedAtMs", project.updatedAtMs)
        put("ranges", JSONArray().apply {
            project.ranges.forEach { range -> put(JSONObject().apply {
                put("startMs", range.startMs)
                put("endMs", range.endMs)
                put("confidence", range.confidence.toDouble())
                put("agreement", range.agreement ?: JSONObject.NULL)
            }) }
        })
    }

    internal fun decode(json: JSONObject): NativeProject? {
        if (json.optInt("version") != VERSION) return null
        val sourceJson = json.getJSONObject("source")
        val featureCacheSource = json.optJSONObject("featureCacheSource")?.let(::decodeSource)
        val mediaJson = json.getJSONObject("media")
        val roiJson = json.getJSONObject("roi")
        val rangesJson = json.optJSONArray("ranges") ?: JSONArray()
        val requestedWindow = json.optJSONObject("analysisWindow")?.let {
            AnalysisTypes.AnalysisWindow(it.optDouble("start"), it.optDouble("end"))
        }
        val analysisWindow = AnalysisTypes.AnalysisWindow.normalize(
            requestedWindow,
            mediaJson.getDouble("durationSeconds"),
        )
        val project = NativeProject(
            id = json.getString("id"),
            source = ProjectSource(
                uri = sourceJson.getString("uri"),
                name = sourceJson.getString("name"),
                size = sourceJson.optLong("size", -1),
                lastModified = sourceJson.optLong("lastModified", -1),
                mimeType = sourceJson.optString("mimeType"),
            ),
            featureCacheSource = featureCacheSource,
            media = AnalysisTypes.MediaInfo(
                mediaJson.getDouble("durationSeconds"),
                mediaJson.optInt("width"),
                mediaJson.optInt("height"),
                mediaJson.optInt("rotation"),
                mediaJson.optString("videoMime"),
                mediaJson.optString("audioMime"),
            ),
            analysisWindow = analysisWindow,
            roi = AnalysisTypes.Roi(
                roiJson.getDouble("x"),
                roiJson.getDouble("y"),
                roiJson.getDouble("width"),
                roiJson.getDouble("height"),
                roiJson.optString("label"),
            ),
            status = ProjectStatus.fromWireName(json.getString("status")) ?: return null,
            ranges = buildList {
                for (index in 0 until rangesJson.length()) {
                    val range = rangesJson.getJSONObject(index)
                    add(SeedRange(
                        range.getLong("startMs"),
                        range.getLong("endMs"),
                        range.getDouble("confidence").toFloat(),
                        if (range.isNull("agreement")) null else range.optString("agreement"),
                    ))
                }
            },
            modelId = json.optString("modelId"),
            cacheMode = json.optString("cacheMode", NativeFeatureCache.Mode.USE.wireName()),
            error = if (json.isNull("error")) null else json.optString("error"),
            createdAtMs = json.getLong("createdAtMs"),
            updatedAtMs = json.getLong("updatedAtMs"),
        )
        return project.takeIf {
            it.id.isNotBlank() && it.source.uri.isNotBlank() &&
                (it.featureCacheSource == null || it.featureCacheSource.uri.isNotBlank()) &&
                it.media.durationSeconds() > 0 &&
                it.analysisWindow.end() - it.analysisWindow.start() >=
                    AnalysisTypes.MIN_ANALYSIS_WINDOW_SECONDS &&
                it.ranges.all { range ->
                    range.startMs >= 0 && range.endMs > range.startMs &&
                        range.endMs <= (it.media.durationSeconds() * 1_000.0).toLong() &&
                        range.confidence in 0f..1f &&
                        (range.agreement == null || ProductionEnsemble.isValidAgreement(range.agreement))
                }
        }
    }

    internal fun normalizeStored(project: NativeProject): NativeProject {
        val analysisWindow = AnalysisTypes.AnalysisWindow.normalize(
            project.analysisWindow,
            project.media.durationSeconds(),
        )
        val normalized = if (analysisWindow == project.analysisWindow) project
            else project.copy(analysisWindow = analysisWindow)
        val staleInference = normalized.modelId != FeatureSchema.MODEL_ID ||
            (normalized.status == ProjectStatus.READY && normalized.ranges.any {
                !ProductionEnsemble.isValidAgreement(it.agreement)
            })
        if (!staleInference) return normalized
        return normalized.copy(
            status = ProjectStatus.QUEUED,
            ranges = emptyList(),
            modelId = FeatureSchema.MODEL_ID,
            error = "Production model ensemble changed; cached features will be reused.",
            updatedAtMs = System.currentTimeMillis(),
        )
    }

    private fun directory(context: Context) = File(context.filesDir, DIRECTORY)

    private fun normalizedModified(value: Long): Long =
        if (value > 10_000_000_000L) value / 1_000 else value

    private fun sameModifiedSecond(left: Long, right: Long): Boolean =
        left < 0 || right < 0 || normalizedModified(left) == normalizedModified(right)

    private fun sameWindow(
        left: AnalysisTypes.AnalysisWindow,
        right: AnalysisTypes.AnalysisWindow,
    ): Boolean = kotlin.math.abs(left.start() - right.start()) < 1e-9 &&
        kotlin.math.abs(left.end() - right.end()) < 1e-9

    private fun projectFile(context: Context, id: String) =
        File(directory(context), "${id.replace(Regex("[^A-Za-z0-9._-]"), "_")}.json")

    private fun encodeSource(source: ProjectSource) = JSONObject().apply {
        put("uri", source.uri)
        put("name", source.name)
        put("size", source.size)
        put("lastModified", source.lastModified)
        put("mimeType", source.mimeType)
    }

    private fun decodeSource(json: JSONObject) = ProjectSource(
        uri = json.getString("uri"),
        name = json.getString("name"),
        size = json.optLong("size", -1),
        lastModified = json.optLong("lastModified", -1),
        mimeType = json.optString("mimeType"),
    )

    private fun Cursor.string(column: String): String? {
        val index = getColumnIndex(column)
        return if (index >= 0 && !isNull(index)) getString(index) else null
    }

    private fun Cursor.long(column: String): Long? {
        val index = getColumnIndex(column)
        return if (index >= 0 && !isNull(index)) getLong(index) else null
    }
}
