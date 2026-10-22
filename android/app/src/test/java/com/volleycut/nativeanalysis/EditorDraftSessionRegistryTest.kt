package com.volleycut.nativeanalysis

import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.io.IOException
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.Executor
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit.SECONDS
import java.util.concurrent.atomic.AtomicReference

class EditorDraftSessionRegistryTest {
    @get:Rule val temporary = TemporaryFolder()
    private val seed = EditorSeed(
        "content://synthetic/session-review", "synthetic.mp4", 12_000, 320, 240, 0,
        listOf(SeedRange(3_000, 6_000, .9f, ProductionEnsemble.BOTH_MODELS)),
    )
    private val initial = EditorMath.newDraft(seed)
    private val removed = EditorMath.reconcileTouchedCuts(
        initial.copy(cuts = initial.cuts.map { it.copy(included = false) }), seed,
    )
    private fun store(root: File = temporary.root) = EditorDraftStore(root, seed)
    private val target get() = File(temporary.root, "editor-drafts/${seed.sourceRevision}.json")
    private class ManualExecutor : Executor {
        val tasks = ConcurrentLinkedQueue<Runnable>()
        override fun execute(command: Runnable) { tasks.add(command) }
        fun runAll() { while (true) (tasks.poll() ?: return).run() }
    }

    @Test fun reopenBeforeOldWriterStartsUsesPendingRemoveInsteadOfStaleDisk() {
        val executor = ManualExecutor()
        val registry = EditorDraftSessionRegistry(executor) { it() }
        store().save(initial)
        val old = checkNotNull(registry.open(store(), initial))
        old.update(removed)
        // A new Activity/ViewModel has a fresh store and only stale disk input.
        val reopened = checkNotNull(registry.open(store(), initial))
        assertSame(old, reopened)
        assertEquals(removed, reopened.draft.value)
        assertEquals(1, executor.tasks.size)
        executor.runAll()
        assertEquals(removed, (store().load() as EditorDraftLoadResult.Restored).draft)
    }

    @Test fun openAfterWriterCompletionReadsCurrentDiskRatherThanEarlierCallerSnapshot() {
        val executor = ManualExecutor()
        val registry = EditorDraftSessionRegistry(executor) { it() }
        store().save(initial)
        val staleDisk = (store().load() as EditorDraftLoadResult.Restored).draft
        checkNotNull(registry.open(store(), staleDisk)).update(removed)
        executor.runAll()
        val reopened = checkNotNull(registry.open(store(), staleDisk))
        assertEquals(removed, reopened.draft.value)
        executor.runAll()
        assertEquals(removed, (store().load() as EditorDraftLoadResult.Restored).draft)
    }

    @Test fun failedWriteSurvivesReopenAndOnlyExplicitRetryClearsFailure() {
        val executor = ManualExecutor()
        val registry = EditorDraftSessionRegistry(executor) { it() }
        val session = checkNotNull(registry.open(store(), initial))
        executor.runAll()
        val blocker = File(target.parentFile, "${target.name}.tmp").apply { mkdirs() }
        session.update(removed)
        executor.runAll()
        val error = session.failure.value
        assertTrue(error is EditorDraftSaveException)
        val reopened = checkNotNull(registry.open(store(), initial))
        assertSame(session, reopened)
        assertEquals(removed, reopened.draft.value)
        assertSame(error, reopened.failure.value)
        assertTrue(blocker.delete())
        registry.open(store(), initial)
        assertSame(error, reopened.failure.value)
        reopened.retry()
        executor.runAll()
        assertNull(reopened.failure.value)
        assertEquals(removed, (store().load() as EditorDraftLoadResult.Restored).draft)
    }

    @Test fun confirmedUnreadableResetReplacesFailedSessionAndIgnoresLateFailureCallback() {
        val executor = ManualExecutor()
        val callbacks = ArrayDeque<() -> Unit>()
        val registry = EditorDraftSessionRegistry(executor) { callbacks.add(it) }
        val old = checkNotNull(registry.open(store(), initial))
        executor.runAll()
        while (callbacks.isNotEmpty()) callbacks.removeFirst()()
        val original = "{broken saved review".toByteArray()
        target.writeBytes(original)
        old.update(removed)
        executor.runAll()
        assertArrayEquals(original, target.readBytes())
        assertSame(old, registry.open(store(), initial))
        assertEquals(EditorDraftLoadResult.Missing, registry.resetUnreadable(store()))
        val fresh = checkNotNull(registry.open(store(), initial))
        assertNotSame(old, fresh)
        while (callbacks.isNotEmpty()) callbacks.removeFirst()()
        assertNull(fresh.failure.value)
        fresh.update(removed.copy(playbackRate = 2f))
        executor.runAll()
        while (callbacks.isNotEmpty()) callbacks.removeFirst()()
        assertNull(fresh.failure.value)
        assertEquals(fresh.draft.value, (store().load() as EditorDraftLoadResult.Restored).draft)
    }

    @Test fun resetWaitsForBlockedWriterAndFencesConcurrentOpen() {
        val executor = ManualExecutor()
        val registry = EditorDraftSessionRegistry(executor) { it() }
        val old = checkNotNull(registry.open(store(), initial))
        old.update(removed)
        target.parentFile!!.mkdirs()
        target.writeText("{broken saved review")
        val resetting = CompletableFuture.supplyAsync { registry.resetUnreadable(store()) }
        // close() observes the reset fence after the old queue has been closed.
        val deadline = System.nanoTime() + SECONDS.toNanos(5)
        while (old.queue.isLatest(2) && System.nanoTime() < deadline) Thread.yield()
        assertFalse(old.queue.isLatest(2))
        val opening = CompletableFuture.supplyAsync { registry.open(store(), initial) }
        assertFalse(resetting.isDone)
        assertFalse(opening.isDone)
        executor.runAll()
        assertEquals(EditorDraftLoadResult.Missing, resetting.get(5, SECONDS))
        val fresh = checkNotNull(opening.get(5, SECONDS))
        assertNotSame(old, fresh)
        executor.runAll()
        assertEquals(initial, (store().load() as EditorDraftLoadResult.Restored).draft)
    }

    @Test fun distinctStorageAndProjectsDoNotSharePendingEditsOrFailures() {
        val executor = ManualExecutor()
        val registry = EditorDraftSessionRegistry(executor) { it() }
        val first = checkNotNull(registry.open(store(), initial))
        first.update(removed)
        val otherStorage = store(temporary.newFolder())
        val second = checkNotNull(registry.open(otherStorage, initial))
        val otherSeed = seed.copy(draftNamespace = "independent-synthetic-project")
        val otherStore = EditorDraftStore(temporary.root, otherSeed)
        val third = checkNotNull(registry.open(otherStore, EditorMath.newDraft(otherSeed)))
        target.parentFile!!.mkdirs()
        target.writeText("{unreadable first review")
        executor.runAll()
        assertTrue(first.failure.value is UnreadableEditorDraftException)
        assertNull(second.failure.value)
        assertNull(third.failure.value)
        assertTrue((otherStorage.load() as EditorDraftLoadResult.Restored).draft.cuts.first().included)
        assertTrue((otherStore.load() as EditorDraftLoadResult.Restored).draft.cuts.first().included)
    }

    @Test fun confirmedResetWaitsForAnActualInFlightStoreWriteBeforeDeletingBytes() {
        val thread = AtomicReference<Thread>()
        val executor = Executors.newSingleThreadExecutor { task -> Thread(task).also(thread::set) }
        val registry = EditorDraftSessionRegistry(executor) { it() }
        val storage = store()
        try {
            val old = checkNotNull(registry.open(storage, initial))
            executor.submit {}.get(5, SECONDS)
            lateinit var resetting: CompletableFuture<EditorDraftLoadResult>
            synchronized(storage) {
                old.update(removed)
                val deadline = System.nanoTime() + SECONDS.toNanos(5)
                // save() is synchronized on this very store. The worker has
                // consumed the request and is now inside the in-flight write.
                while (thread.get().state != Thread.State.BLOCKED && System.nanoTime() < deadline) Thread.yield()
                assertEquals(Thread.State.BLOCKED, thread.get().state)
                target.writeText("{unreadable while writer is delayed")
                resetting = CompletableFuture.supplyAsync { registry.resetUnreadable(storage) }
                while (old.queue.isLatest(2) && System.nanoTime() < deadline) Thread.yield()
                assertFalse(old.queue.isLatest(2))
                assertFalse(resetting.isDone)
            }
            assertEquals(EditorDraftLoadResult.Missing, resetting.get(5, SECONDS))
            val fresh = checkNotNull(registry.open(store(), initial))
            fresh.retry().get(5, SECONDS)
            executor.submit {}.get(5, SECONDS)
            assertNotSame(old, fresh)
            assertNull(fresh.failure.value)
            assertEquals(initial, (store().load() as EditorDraftLoadResult.Restored).draft)
        } finally {
            executor.shutdownNow()
            assertTrue(executor.awaitTermination(5, SECONDS))
        }
    }

    @Test fun cleanReopenRefreshesImportedMetadataAndDeletionDropsQueuedSnapshot() {
        val executor = ManualExecutor()
        val registry = EditorDraftSessionRegistry(executor) { it() }
        val old = checkNotNull(registry.open(store(), initial))
        executor.runAll()
        val imported = EditorMath.reconcileTouchedCuts(initial.copy(playbackRate = 2f, beforePaddingMs = 0), seed)
        store().save(imported)
        assertEquals(imported, checkNotNull(registry.open(store(), initial)).draft.value)
        old.update(removed)
        val closed = registry.close(store())
        assertFalse(closed.isDone)
        executor.runAll()
        closed.get(5, SECONDS)
        store().clear()
        old.update(removed)
        executor.runAll()
        assertEquals(EditorDraftLoadResult.Missing, store().load())
        assertNotSame(old, registry.open(store(), initial))
        executor.runAll()
    }

    @Test fun deletionRejectsReopensUntilServiceCompletesAndStaleOpensAfterMetadataRemoval() {
        val executor = ManualExecutor()
        val registry = EditorDraftSessionRegistry(executor) { it() }
        val old = checkNotNull(registry.open(store(), initial))
        old.update(removed)
        var projectExists = true
        val deletion = registry.beginDeletion("synthetic-project", store())
        assertTrue(registry.isDeleting("synthetic-project"))
        assertFalse(deletion.settled.isDone)
        assertTrue(registry.save(store()).get() is java.util.concurrent.CancellationException)
        assertNull(registry.open(store(), initial, available = { projectExists }))
        executor.runAll()
        assertTrue(deletion.settled.isDone)
        // Closing the old writer does not finish the service's transaction.
        assertFalse(deletion.result.isDone)
        assertNull(registry.open(store(), initial, available = { projectExists }))
        projectExists = false
        store().clear()
        registry.finishDeletion(deletion, null)
        assertNull(deletion.result.get(5, SECONDS))
        assertFalse(registry.isDeleting("synthetic-project"))
        assertNull(registry.open(store(), initial, available = { projectExists }))
        old.update(removed)
        executor.runAll()
        assertEquals(EditorDraftLoadResult.Missing, store().load())
        // A deliberate new project with the same source may be created later.
        projectExists = true
        assertNotSame(old, registry.open(store(), initial, available = { projectExists }))
        executor.runAll()
    }

    @Test fun failedDeletionRestoresNewestFailedReviewAndExplicitRetryStillWorks() {
        val executor = ManualExecutor()
        val callbacks = ArrayDeque<() -> Unit>()
        val registry = EditorDraftSessionRegistry(executor) { callbacks.addLast(it) }
        val old = checkNotNull(registry.open(store(), initial))
        executor.runAll()
        while (callbacks.isNotEmpty()) callbacks.removeFirst()()
        val blocker = File(target.parentFile, "${target.name}.tmp").apply { mkdirs() }
        old.update(removed)
        executor.runAll()
        while (callbacks.isNotEmpty()) callbacks.removeFirst()()
        val previousFailure = old.failure.value
        assertTrue(previousFailure is EditorDraftSaveException)
        val deleting = registry.beginDeletion("synthetic-project", store())
        assertSame(deleting, registry.beginDeletion("synthetic-project", store()))
        val error = IOException("synthetic deletion refused")
        registry.finishDeletion(deleting, error)
        assertSame(error, deleting.result.get(5, SECONDS))
        val restored = checkNotNull(registry.open(store(), initial))
        assertNotSame(old, restored)
        assertEquals(removed, restored.draft.value)
        assertSame(previousFailure, restored.failure.value)
        executor.runAll()
        while (callbacks.isNotEmpty()) callbacks.removeFirst()()
        assertTrue(restored.failure.value is EditorDraftSaveException)
        assertTrue(blocker.delete())
        restored.retry()
        executor.runAll()
        while (callbacks.isNotEmpty()) callbacks.removeFirst()()
        assertNull(restored.failure.value)
        assertEquals(removed, (store().load() as EditorDraftLoadResult.Restored).draft)
    }

    @Test fun deletingOneProjectDoesNotFenceAnotherStorageOrReviewOrDropNoDraftDeletion() {
        val executor = ManualExecutor()
        val registry = EditorDraftSessionRegistry(executor) { it() }
        val deletion = registry.beginDeletion("synthetic-project", store())
        assertTrue(deletion.settled.isDone)
        assertNull(registry.open(store(), initial))
        val otherStore = store(temporary.newFolder())
        assertNotNull(registry.open(otherStore, initial))
        val otherSeed = seed.copy(draftNamespace = "independent-synthetic-review")
        assertNotNull(registry.open(EditorDraftStore(temporary.root, otherSeed), EditorMath.newDraft(otherSeed)))
        val noDraft = registry.beginDeletion("synthetic-queued-project", null)
        assertTrue(registry.isDeleting("synthetic-queued-project"))
        executor.runAll()
        assertNull(registry.open(store(), initial))
        registry.finishDeletion(noDraft, null)
        registry.finishDeletion(deletion, null)
        assertFalse(registry.isDeleting("synthetic-queued-project"))
    }
}
