package com.volleycut.nativeanalysis

import org.junit.Assert.*
import org.junit.Test
import java.io.IOException
import java.util.concurrent.CancellationException
import java.util.concurrent.CompletableFuture
import java.util.concurrent.Executor

class EditorDraftSaveQueueTest {
    private val initial = EditorMath.newDraft(EditorSeed(
        "content://synthetic/queued-review", "synthetic.mp4", 12_000, 320, 240, 0,
        listOf(SeedRange(3_000, 6_000, .9f, ProductionEnsemble.BOTH_MODELS)),
    ))
    private class ManualExecutor : Executor {
        val tasks = ArrayDeque<Runnable>()
        override fun execute(command: Runnable) { tasks.addLast(command) }
        fun runAll() { while (tasks.isNotEmpty()) tasks.removeFirst().run() }
    }

    @Test fun latestEditSurvivesDisposalBeforeWorkerStarts() {
        val executor = ManualExecutor()
        val saved = mutableListOf<EditorDraft>()
        val queue = EditorDraftSaveQueue(executor, saved::add)
        val first = queue.request(initial)
        val latest = initial.copy(cuts = initial.cuts.map { it.copy(included = false) }, playbackRate = 2f)
        val second = queue.request(latest)
        // No composition scope or activity is required to finish queued work.
        assertFalse(second.isDone)
        executor.runAll()
        assertEquals(listOf(latest), saved)
        assertNull(first.get())
        assertNull(second.get())
        assertFalse(queue.hasUnsavedChanges)
    }

    @Test fun editsDuringBlockedWriteAreCoalescedAndNeverOverwrittenByOldResult() {
        val executor = ManualExecutor()
        val saved = mutableListOf<Long>()
        lateinit var queue: EditorDraftSaveQueue
        lateinit var middle: CompletableFuture<Throwable?>
        lateinit var latest: CompletableFuture<Throwable?>
        queue = EditorDraftSaveQueue(executor, { draft ->
            saved += draft.updatedAtMs
            if (draft.updatedAtMs == 1L) {
                middle = queue.request(initial.copy(updatedAtMs = 2))
                latest = queue.request(initial.copy(updatedAtMs = 3))
                assertTrue(queue.hasUnsavedChanges)
            }
        })
        queue.request(initial.copy(updatedAtMs = 1))
        executor.runAll()
        assertEquals(listOf(1L, 3L), saved)
        assertNull(middle.get())
        assertNull(latest.get())
        assertFalse(queue.hasUnsavedChanges)
    }

    @Test fun failureKeepsReviewUnsavedUntilExplicitRetrySucceeds() {
        val executor = ManualExecutor()
        val error = IOException("synthetic full storage")
        var failing = true
        val queue = EditorDraftSaveQueue(executor, { if (failing) throw error })
        val first = queue.request(initial)
        executor.runAll()
        assertSame(error, first.get())
        assertTrue(queue.hasUnsavedChanges)
        failing = false
        val retry = queue.request(initial)
        executor.runAll()
        assertNull(retry.get())
        assertFalse(queue.hasUnsavedChanges)
    }

    @Test fun staleFailureCannotReplaceLatestSuccessfulSaveState() {
        val executor = ManualExecutor()
        val accepted = mutableListOf<Throwable?>()
        lateinit var queue: EditorDraftSaveQueue
        queue = EditorDraftSaveQueue(executor, { draft ->
            if (draft.updatedAtMs == 1L) {
                queue.request(initial.copy(updatedAtMs = 2))
                throw IOException("superseded write failed")
            }
        }, { sequence, _, error -> if (queue.isLatest(sequence)) accepted += error })
        val old = queue.request(initial.copy(updatedAtMs = 1))
        executor.runAll()
        assertTrue(old.get() is IOException)
        assertEquals(listOf<Throwable?>(null), accepted)
        assertFalse(queue.hasUnsavedChanges)
    }

    @Test fun closingBeforeWorkerStartsPreventsDeletedReviewFromBeingRecreated() {
        val executor = ManualExecutor()
        val queue = EditorDraftSaveQueue(executor, { fail("Closed review must not write") })
        val edit = queue.request(initial)
        val closed = queue.close()
        assertTrue(edit.get() is CancellationException)
        assertFalse(closed.isDone)
        executor.runAll()
        assertTrue(closed.isDone)
        assertTrue(queue.request(initial).get() is CancellationException)
    }

    @Test fun deletionWaitsForInFlightWriteAndDropsPendingOlderReview() {
        val executor = ManualExecutor()
        val saved = mutableListOf<Long>()
        lateinit var queue: EditorDraftSaveQueue
        lateinit var closed: CompletableFuture<Unit>
        queue = EditorDraftSaveQueue(executor, { draft ->
            queue.request(initial.copy(updatedAtMs = 2))
            closed = queue.close()
            assertFalse(closed.isDone)
            saved += draft.updatedAtMs
        })
        queue.request(initial.copy(updatedAtMs = 1))
        executor.runAll()
        assertTrue(closed.isDone)
        assertEquals(listOf(1L), saved)
    }

    @Test fun separateProjectsKeepIndependentPendingDraftsAndFailures() {
        val executor = ManualExecutor()
        val saved = mutableListOf<EditorDraft>()
        val first = EditorDraftSaveQueue(executor, { throw IOException("first project unavailable") })
        val second = EditorDraftSaveQueue(executor, saved::add)
        val failed = first.request(initial)
        val other = initial.copy(sourceRevision = "another-synthetic-review", playbackRate = 1.5f)
        val success = second.request(other)
        executor.runAll()
        assertTrue(failed.get() is IOException)
        assertNull(success.get())
        assertEquals(listOf(other), saved)
        assertTrue(first.hasUnsavedChanges)
        assertFalse(second.hasUnsavedChanges)
    }
}
