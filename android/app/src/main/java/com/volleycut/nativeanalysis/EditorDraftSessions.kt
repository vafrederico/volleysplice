package com.volleycut.nativeanalysis

import android.os.Handler
import android.os.Looper
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.ViewModel
import java.lang.ref.WeakReference
import java.util.concurrent.CompletableFuture
import java.util.concurrent.Executor
import java.util.concurrent.Executors

/** Activity facade; unfinished reviews belong to the process, even after finish(). */
internal class EditorDraftSessions : ViewModel() {
    fun open(store: EditorDraftStore, initial: EditorDraft, prepare: (EditorDraft) -> EditorDraft = { it }): EditorDraftSession? =
        registry.open(store, initial, prepare)
    fun close(store: EditorDraftStore): CompletableFuture<Unit> = registry.close(store)
    fun save(store: EditorDraftStore): CompletableFuture<Throwable?> = registry.save(store)
    fun resetUnreadable(store: EditorDraftStore): EditorDraftLoadResult = registry.resetUnreadable(store)

    companion object {
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

    private fun prune() {
        entries.entries.removeAll { it.value.get() == null }
        closing.entries.removeAll { it.value.isDone }
    }

    fun open(store: EditorDraftStore, initial: EditorDraft, prepare: (EditorDraft) -> EditorDraft = { it }): EditorDraftSession? {
        while (true) {
            val barrier = synchronized(lock) {
                prune()
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
