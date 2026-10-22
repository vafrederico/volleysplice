package com.volleycut.nativeanalysis

import android.os.Handler
import android.os.Looper
import android.os.Bundle
import android.os.ResultReceiver
import android.content.Context
import android.content.Intent
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.ViewModel
import java.lang.ref.WeakReference
import java.util.concurrent.CompletableFuture
import java.util.concurrent.Executor
import java.util.concurrent.Executors
import java.util.concurrent.CancellationException
import java.io.IOException

/** Activity facade; unfinished reviews belong to the process, even after finish(). */
internal class EditorDraftSessions : ViewModel() {
    fun close(store: EditorDraftStore): CompletableFuture<Unit> = registry.close(store)
    fun save(store: EditorDraftStore): CompletableFuture<Throwable?> = registry.save(store)
    fun resetUnreadable(store: EditorDraftStore): EditorDraftLoadResult = registry.resetUnreadable(store)

    fun openProject(context: Context, project: NativeProject, store: EditorDraftStore, initial: EditorDraft,
                    prepare: (EditorDraft) -> EditorDraft): EditorDraftSession? =
        registry.open(store, initial, available = {
            !isDeleting(context, project.id) && NativeProjectStore.exists(context, project.id)
        }, prepare = prepare)

    /** Own the complete operation independently of the confirming Activity. */
    fun deleteProject(context: Context, project: NativeProject): CompletableFuture<Throwable?> {
        val app = context.applicationContext
        val store = project.editorSeed()?.let { EditorDraftStore(app, it) }
        val deletion = registry.beginDeletion(NativeProjectStore.identity(app, project.id), store)
        if (!deletion.started.compareAndSet(false, true)) return deletion.result
        deletionWorker.execute {
            deletion.settled.get()
            fun complete(failure: Throwable?) {
                deletionWorker.execute {
                    var error = failure
                    try {
                        if (error == null) check(!NativeProjectStore.exists(app, project.id)) { "Project metadata could not be deleted" }
                        if (error == null && store != null) {
                            store.clear()
                            check(store.load() == EditorDraftLoadResult.Missing) { "Saved edits could not be deleted" }
                        }
                    } catch (problem: Exception) { error = problem }
                    if (error != null && !NativeProjectStore.exists(app, project.id)) {
                        runCatching { NativeProjectStore.save(app, project) }.onFailure { error = it }
                    }
                    registry.finishDeletion(deletion, error)
                    app.sendBroadcast(Intent(ProjectAnalysisService.ACTION_UPDATE).setPackage(app.packageName)
                        .putExtra(ProjectAnalysisService.EXTRA_PROJECT_ID, project.id)
                        .putExtra(ProjectAnalysisService.EXTRA_STAGE, if (error == null) "deleted" else "delete-failed")
                        .putExtra(ProjectAnalysisService.EXTRA_DETAIL, error?.message ?: "Project deleted"))
                }
            }
            try {
                ProjectAnalysisService.delete(app, project.id, object : ResultReceiver(null) {
                    override fun onReceiveResult(resultCode: Int, resultData: Bundle?) {
                        complete(if (resultCode == 0) null else IOException(
                            "Project could not be deleted. Your edits were preserved; please retry.",
                        ))
                    }
                })
            } catch (error: Exception) { complete(error) }
        }
        return deletion.result
    }

    companion object {
        fun isDeleting(context: Context, projectId: String): Boolean =
            registry.isDeleting(NativeProjectStore.identity(context, projectId))

        fun isDeleting(context: Context, seed: EditorSeed): Boolean =
            registry.isDeletingDraft(EditorDraftStore(context, seed).sessionKey)

        private val deletionWorker = Executors.newSingleThreadExecutor { task ->
            Thread(task, "editor-project-deletion").apply { isDaemon = true }
        }
        private val writer = Executors.newSingleThreadExecutor { task ->
            Thread(task, "editor-draft-writer").apply { isDaemon = true }
        }
        private val registry = EditorDraftSessionRegistry(writer) { action ->
            Handler(Looper.getMainLooper()).post(action)
        }
    }
}

/**
 * One queue per app-storage/review identity. Dirty/failed snapshots are held
 * strongly; clean sessions are weak and leave with their activity/composition.
 * Open checks pending memory and reads disk under the same lock, so an earlier
 * disk snapshot can never become a second queue's initial write.
 * Disk-reading methods must be called on IO, never the UI thread.
 */
internal class EditorDraftSessionRegistry(
    private val writer: Executor,
    private val dispatch: (() -> Unit) -> Unit,
) {
    private class Entry(session: EditorDraftSession) {
        val reference = WeakReference(session)
        var dirty: EditorDraftSession? = session
        fun get() = dirty ?: reference.get()
    }
    private val lock = Any()
    private val entries = mutableMapOf<String, Entry>()
    private val closing = mutableMapOf<String, CompletableFuture<Unit>>()
    private val deletions = mutableMapOf<String, Deletion>()

    internal class Deletion(
        val projectKey: String,
        val store: EditorDraftStore?,
        val previous: EditorDraftSession?,
        val settled: CompletableFuture<Unit>,
    ) {
        val started = java.util.concurrent.atomic.AtomicBoolean()
        val result = CompletableFuture<Throwable?>()
    }

    fun isDeleting(projectKey: String): Boolean = synchronized(lock) { projectKey in deletions }
    fun isDeletingDraft(key: String): Boolean = synchronized(lock) { deletions.values.any { it.store?.sessionKey == key } }

    fun beginDeletion(projectKey: String, store: EditorDraftStore?): Deletion = synchronized(lock) {
        deletions[projectKey] ?: run {
            val previous = store?.let { entries.remove(it.sessionKey)?.get() }
            val settled = previous?.queue?.close()
                ?: store?.let { closing[it.sessionKey] } ?: CompletableFuture.completedFuture(Unit)
            Deletion(projectKey, store, previous, settled).also { deletions[projectKey] = it }
        }
    }

    fun finishDeletion(deletion: Deletion, error: Throwable?) {
        synchronized(lock) {
            if (deletions[deletion.projectKey] !== deletion) return
            check(deletion.settled.isDone)
            if (error != null && deletion.previous != null && deletion.store != null) {
                // Closing cancelled pending work, so an unsuccessful deletion must
                // restore the newest snapshot, including its existing save error.
                val previous = deletion.previous
                val replacement = EditorDraftSession(writer, deletion.store, previous.draft.value, dispatch, lock, ::settled, ::request)
                replacement.failure.value = previous.failure.value
                entries[deletion.store.sessionKey] = Entry(replacement)
                replacement.retry()
            }
            deletions.remove(deletion.projectKey)
        }
        deletion.result.complete(error)
    }

    private fun prune() {
        entries.entries.removeAll { it.value.get() == null }
        closing.entries.removeAll { it.value.isDone }
    }

    fun open(store: EditorDraftStore, initial: EditorDraft, available: () -> Boolean = { true },
             prepare: (EditorDraft) -> EditorDraft = { it }): EditorDraftSession? {
        while (true) {
            val barrier = synchronized(lock) {
                prune()
                if (isDeletingDraft(store.sessionKey) || !available()) return null
                closing[store.sessionKey]?.let { return@synchronized it }
                val existing = entries[store.sessionKey]?.get()
                if (existing != null && existing.queue.hasUnsavedChanges) return existing
                val loaded = store.load()
                if (loaded == EditorDraftLoadResult.Unreadable) return null
                val restored = (loaded as? EditorDraftLoadResult.Restored)?.draft ?: initial
                if (existing != null) {
                    // Setup/import can change metadata while the review is closed.
                    existing.draft.value = restored
                    existing.request(restored)
                    return existing
                }
                val session = EditorDraftSession(writer, store, prepare(restored), dispatch, lock, ::settled, ::request)
                entries[store.sessionKey] = Entry(session)
                session.request(session.draft.value)
                return session
            }
            barrier.get()
        }
    }

    private fun settled(key: String, session: EditorDraftSession) = synchronized(lock) {
        entries[key]?.takeIf { it.get() === session }?.let {
            it.dirty = session.takeIf { session.queue.hasUnsavedChanges }
        }
        prune()
    }

    fun save(store: EditorDraftStore): CompletableFuture<Throwable?> = synchronized(lock) {
        if (isDeletingDraft(store.sessionKey)) return@synchronized CompletableFuture.completedFuture(
            CancellationException("Project is being deleted"),
        )
        entries[store.sessionKey]?.get()?.let { it.request(it.draft.value) }
            ?: CompletableFuture.completedFuture(null)
    }

    // Pin the snapshot before publishing it to the writer, including after a
    // previously clean session is edited again.
    private fun request(session: EditorDraftSession, draft: EditorDraft): CompletableFuture<Throwable?> =
        synchronized(lock) {
            entries[session.key]?.takeIf { it.get() === session }?.dirty = session
            session.queue.request(draft)
        }

    fun close(store: EditorDraftStore): CompletableFuture<Unit> = synchronized(lock) {
        prune()
        deletions.values.firstOrNull { it.store?.sessionKey == store.sessionKey }?.let { return@synchronized it.settled }
        closing[store.sessionKey] ?: run {
            val barrier = entries.remove(store.sessionKey)?.get()?.queue?.close()
                ?: CompletableFuture.completedFuture(Unit)
            closing[store.sessionKey] = barrier
            barrier
        }
    }

    /** Explicit discard only. Fence opens until the last old write AND reset settle. */
    fun resetUnreadable(store: EditorDraftStore): EditorDraftLoadResult {
        val reset = CompletableFuture<Unit>()
        val prior = synchronized(lock) {
            check(!isDeletingDraft(store.sessionKey)) { "Project is being deleted" }
            prune()
            val active = closing[store.sessionKey]
                ?: entries.remove(store.sessionKey)?.get()?.queue?.close()
                ?: CompletableFuture.completedFuture(Unit)
            closing[store.sessionKey] = reset
            active
        }
        return try {
            prior.get()
            store.resetUnreadable()
        } finally {
            synchronized(lock) { closing.remove(store.sessionKey, reset) }
            reset.complete(Unit)
        }
    }
}

internal class EditorDraftSession(
    writer: Executor,
    store: EditorDraftStore,
    initial: EditorDraft,
    dispatch: (() -> Unit) -> Unit,
    private val lock: Any,
    settled: (String, EditorDraftSession) -> Unit,
    private val enqueue: (EditorDraftSession, EditorDraft) -> CompletableFuture<Throwable?>,
) {
    val key = store.sessionKey
    val draft = mutableStateOf(initial)
    val failure = mutableStateOf<Throwable?>(null)
    val queue: EditorDraftSaveQueue

    init {
        queue = EditorDraftSaveQueue(writer, store::save) { sequence, _, error ->
            settled(key, this)
            dispatch { if (queue.isLatest(sequence)) failure.value = error }
        }
    }

    fun request(next: EditorDraft) = enqueue(this, next)

    fun update(next: EditorDraft) = synchronized(lock) {
        draft.value = next
        request(next)
    }

    fun retry() = synchronized(lock) { request(draft.value) }
}
