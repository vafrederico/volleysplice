package com.volleycut.nativeanalysis

import android.net.Uri
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** The activity registry restores its launcher; this restores the matching request. */
internal class PendingEditorExports(private val state: SavedStateHandle) : ViewModel() {
    data class Ready(val kind: EditorExportKind, val id: String, val filename: String, val projectId: String)
    data class Outcome(val sequence: Long, val kind: EditorExportKind, val projectId: String, val message: String, val jobId: String? = null, val failed: Boolean = false)

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private var sequence = 0L
    val preparing = mutableStateOf(false)
    val activeWrites = mutableStateOf(0)
    val ready = mutableStateOf(state.get<ArrayList<String>>(READY)?.let { fields ->
        runCatching { Ready(EditorExportKind.valueOf(fields[0]), fields[1], fields[2], fields[3]) }.getOrNull()
    })
    val outcome = mutableStateOf<Outcome?>(null)
    val busy: Boolean get() = preparing.value || activeWrites.value > 0 || ready.value != null

    fun prepare(kind: EditorExportKind, projectId: String, store: PendingEditorExportStore, build: () -> PendingEditorExport) {
        if (busy) return
        preparing.value = true
        scope.launch {
            try {
                val prepared = withContext(Dispatchers.IO) {
                    val request = build()
                    require(request.kind == kind && request.projectId == projectId)
                    Ready(kind, store.save(request), request.filename, request.projectId)
                }
                state[key(kind)] = prepared.id
                state[READY] = arrayListOf(kind.name, prepared.id, prepared.filename, prepared.projectId)
                ready.value = prepared
            } catch (failure: Exception) {
                report(kind, projectId, failure.message ?: "Could not prepare the export. Try again.", failed = true)
            } finally { preparing.value = false }
        }
    }

    fun launched(request: Ready) {
        if (ready.value == request) {
            ready.value = null
            state.remove<ArrayList<String>>(READY)
        }
    }

    fun launchFailed(request: Ready, store: PendingEditorExportStore, failure: Exception) {
        launched(request)
        if (state.get<String>(key(request.kind)) == request.id) state.remove<String>(key(request.kind))
        scope.launch(Dispatchers.IO) { runCatching { store.remove(request.id) } }
        report(request.kind, request.projectId, failure.message ?: "Could not open the save dialog. Try again.", failed = true)
    }

    fun consume(
        kind: EditorExportKind,
        currentProjectId: String,
        uri: Uri?,
        store: PendingEditorExportStore,
        write: (PendingEditorExport, Uri) -> Unit,
    ) {
        val id = state.remove<String>(key(kind))
        if (id == null) {
            if (uri != null) report(kind, currentProjectId, "The destination returned without an export request. Save again from the review.", failed = true)
            return
        }
        activeWrites.value++
        scope.launch {
            var requestedProject = currentProjectId
            try {
                if (uri != null) {
                    val request = withContext(Dispatchers.IO) {
                        store.load(id).also { require(it.kind == kind) }
                    }
                    requestedProject = request.projectId
                    withContext(Dispatchers.IO) { write(request, uri) }
                    report(kind, requestedProject, when (kind) {
                        EditorExportKind.VIDEO -> "Added to export queue"
                        EditorExportKind.CHAPTERS -> "Saved the YouTube chapters text file."
                        EditorExportKind.PROJECT -> "Saved the detailed project file"
                    }, (request as? PendingEditorExport.Video)?.jobId)
                }
            } catch (failure: Exception) {
                report(kind, requestedProject, failure.message ?: "Could not save the export. Try again.", failed = true)
            } finally {
                withContext(Dispatchers.IO) { runCatching { store.remove(id) } }
                activeWrites.value--
            }
        }
    }

    fun acknowledge(result: Outcome) { if (outcome.value == result) outcome.value = null }

    private fun report(kind: EditorExportKind, projectId: String, text: String, jobId: String? = null, failed: Boolean = false) {
        outcome.value = Outcome(++sequence, kind, projectId, text, jobId, failed)
    }

    private fun key(kind: EditorExportKind) = "pending-editor-export-${kind.name}"

    companion object { private const val READY = "prepared-editor-export" }
}
