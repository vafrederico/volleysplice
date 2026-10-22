package com.volleycut.nativeanalysis

import java.util.concurrent.CancellationException
import java.util.concurrent.CompletableFuture
import java.util.concurrent.Executor

/** Serial writes owned by a review session, independent of its composition. */
internal class EditorDraftSaveQueue(
    private val executor: Executor,
    private val save: (EditorDraft) -> Unit,
    private val settled: (Long, EditorDraft, Throwable?) -> Unit = { _, _, _ -> },
) {
    private data class Request(val sequence: Long, val draft: EditorDraft)
    private val lock = Any()
    private var pending: Request? = null
    private val waiters = mutableListOf<Pair<Long, CompletableFuture<Throwable?>>>()
    private val closing = CompletableFuture<Unit>()
    private var sequence = 0L
    private var savedSequence = -1L
    private var running = false
    private var closed = false

    val hasUnsavedChanges: Boolean get() = synchronized(lock) { savedSequence != sequence }
    fun isLatest(candidate: Long): Boolean = synchronized(lock) { !closed && sequence == candidate }

    fun request(draft: EditorDraft): CompletableFuture<Throwable?> {
        val result = CompletableFuture<Throwable?>()
        synchronized(lock) {
            if (closed) return result.apply { complete(CancellationException("Review session closed")) }
            pending = Request(++sequence, draft)
            waiters += sequence to result
            if (!running) {
                running = true
                executor.execute(::drain)
            }
        }
        return result
    }

    /** Stop accepting edits and settle any active write before deleting a project. */
    fun close(): CompletableFuture<Unit> = synchronized(lock) {
        closed = true
        pending = null
        waiters.forEach { it.second.complete(CancellationException("Review session closed")) }
        waiters.clear()
        if (!running) closing.complete(Unit)
        closing
    }

    private fun drain() {
        while (true) {
            val request = synchronized(lock) {
                val next = pending
                pending = null
                if (next == null) {
                    running = false
                    if (closed) closing.complete(Unit)
                }
                next
            } ?: return
            val error = try {
                save(request.draft)
                null
            } catch (failure: Exception) {
                failure
            }
            val completed = synchronized(lock) {
                if (error == null) savedSequence = request.sequence
                waiters.filter { it.first <= request.sequence }.also { waiters.removeAll(it.toSet()) }
            }
            completed.forEach { it.second.complete(error) }
            settled(request.sequence, request.draft, error)
        }
    }
}
