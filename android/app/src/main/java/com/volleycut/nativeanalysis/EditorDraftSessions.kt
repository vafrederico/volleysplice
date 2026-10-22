package com.volleycut.nativeanalysis

import android.os.Handler
import android.os.Looper
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.ViewModel
import java.util.concurrent.CompletableFuture
import java.util.concurrent.Executors

/** Retains in-memory edits during recreation and writes after leaving an editor. */
internal class EditorDraftSessions : ViewModel() {
    private val sessions = mutableMapOf<String, EditorDraftSession>()

    fun open(store: EditorDraftStore, initial: EditorDraft): EditorDraftSession {
        return sessions[initial.sourceRevision]?.also { session ->
            // A setup/import action can update a saved review while it is closed.
            // Pending or failed writes always keep the newer in-memory review.
            if (!session.queue.hasUnsavedChanges) {
                (store.load() as? EditorDraftLoadResult.Restored)?.draft?.let { session.draft.value = it }
            }
        } ?: EditorDraftSession(store, initial).also { sessions[initial.sourceRevision] = it }
    }

    fun close(sourceRevision: String): CompletableFuture<Unit> =
        sessions.remove(sourceRevision)?.queue?.close() ?: CompletableFuture.completedFuture(Unit)

    fun save(sourceRevision: String): CompletableFuture<Throwable?> =
        sessions[sourceRevision]?.let { it.queue.request(it.draft.value) }
            ?: CompletableFuture.completedFuture(null)
}

internal class EditorDraftSession(store: EditorDraftStore, initial: EditorDraft) {
    val draft = mutableStateOf(initial)
    val failure = mutableStateOf<Throwable?>(null)
    val queue: EditorDraftSaveQueue

    init {
        val handler = Handler(Looper.getMainLooper())
        queue = EditorDraftSaveQueue(writer, store::save) { sequence, _, error ->
            handler.post {
                if (queue.isLatest(sequence)) failure.value = error
            }
        }
        queue.request(initial)
    }

    fun update(next: EditorDraft) {
        draft.value = next
        queue.request(next)
    }

    fun retry() { queue.request(draft.value) }

    companion object {
        // Never cancelled by composition disposal or ViewModel clearing: queued
        // writes must settle even when the activity finishes. Each session also
        // coalesces edits while its previous write is in progress.
        private val writer = Executors.newSingleThreadExecutor { task ->
            Thread(task, "editor-draft-writer").apply { isDaemon = true }
        }
    }
}
