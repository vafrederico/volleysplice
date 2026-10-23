package com.volleycut.nativeanalysis

import org.junit.Assert.*
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.io.OutputStream
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

class ExportFilePublicationTest {
    private class Harness : AutoCloseable {
        val worker = Executors.newSingleThreadExecutor()
        val callbacks = LinkedBlockingQueue<Runnable>()
        val mainThread = Thread.currentThread()
        var current = true
        var success: String? = null
        var failure: Throwable? = null
        var cancelled = false
        var temporaryRemoved = false
        var destinationRemoved = false
        var settled = false
        val progress = mutableListOf<Long>()
        lateinit var task: ExportFilePublication

        fun start(data: ByteArray, output: () -> OutputStream, input: () -> ByteArrayInputStream = { ByteArrayInputStream(data) }) {
            task = ExportFilePublication(worker, { callbacks.add(it) }, input, output, data.size.toLong(),
                { assertNotSame(mainThread, Thread.currentThread()); "finished.mp4" },
                { assertNotSame(mainThread, Thread.currentThread()); temporaryRemoved = true },
                { assertNotSame(mainThread, Thread.currentThread()); destinationRemoved = true },
                { current }, { bytes, _ -> assertSame(mainThread, Thread.currentThread()); progress += bytes },
                { name -> assertSame(mainThread, Thread.currentThread()); success = name },
                { error, wasCancelled -> assertSame(mainThread, Thread.currentThread()); failure = error; cancelled = wasCancelled },
                { settled = true })
            task.start()
        }

        fun drainUntil(done: () -> Boolean) {
            repeat(20) {
                if (done()) return
                callbacks.poll(200, TimeUnit.MILLISECONDS)?.run()
            }
            assertTrue("Publication callback did not settle", done())
        }

        fun awaitWorker() { worker.submit {}.get(3, TimeUnit.SECONDS) }
        override fun close() { if (::task.isInitialized) task.cancel(); worker.shutdownNow(); worker.awaitTermination(3, TimeUnit.SECONDS) }
    }

    @Test fun slowCopyIsOffMainAndSuccessWaitsForStreamCloseAndCurrentJobAcceptance() {
        Harness().use { h ->
            val entered = CountDownLatch(1)
            val release = CountDownLatch(1)
            val data = ByteArray(2 * 1_024 * 1_024 + 7) { (it % 251).toByte() }
            val bytes = ByteArrayOutputStream()
            val closed = AtomicBoolean(false)
            h.start(data, { object : OutputStream() {
                override fun write(value: Int) { bytes.write(value) }
                override fun write(value: ByteArray, offset: Int, count: Int) {
                    assertNotSame(h.mainThread, Thread.currentThread())
                    entered.countDown(); assertTrue(release.await(3, TimeUnit.SECONDS)); bytes.write(value, offset, count)
                }
                override fun close() { closed.set(true) }
            } })
            assertTrue(entered.await(3, TimeUnit.SECONDS))
            assertNull(h.success)
            // The UI thread can continue processing work while provider output blocks.
            var uiWorked = false
            Runnable { uiWorked = true }.run()
            assertTrue(uiWorked)
            release.countDown()
            h.drainUntil { h.success != null }
            assertTrue(closed.get())
            assertArrayEquals(data, bytes.toByteArray())
            h.awaitWorker()
            assertTrue(h.temporaryRemoved && h.settled)
            assertFalse(h.destinationRemoved)
            assertEquals(data.size.toLong(), h.progress.last())
            assertTrue(h.progress.zipWithNext().all { (a, b) -> b >= a })
        }
    }

    @Test fun cancellingBlockedProviderClosesStreamsAndCleansOnWorkerBeforeReportingCancellation() {
        Harness().use { h ->
            val entered = CountDownLatch(1)
            val closed = AtomicBoolean(false)
            h.start(byteArrayOf(1), { object : OutputStream() {
                override fun write(value: Int) { error("bulk writes expected") }
                override fun write(value: ByteArray, offset: Int, count: Int) { entered.countDown(); CountDownLatch(1).await() }
                override fun close() { closed.set(true) }
            } })
            assertTrue(entered.await(3, TimeUnit.SECONDS))
            h.task.cancel()
            h.drainUntil { h.failure != null }
            assertTrue(h.cancelled && closed.get() && h.temporaryRemoved && h.destinationRemoved && h.settled)
            assertNull(h.success)
        }
    }

    @Test fun providerFailureAndInputFailureCannotReportSuccess() {
        for (failInput in listOf(false, true)) Harness().use { h ->
            h.start(byteArrayOf(1), { if (!failInput) throw IOException("provider unavailable") else ByteArrayOutputStream() },
                { if (failInput) throw IOException("source unavailable") else ByteArrayInputStream(byteArrayOf(1)) })
            h.drainUntil { h.failure != null }
            assertTrue(h.temporaryRemoved && h.destinationRemoved && h.settled)
            assertFalse(h.cancelled)
            assertNull(h.success)
        }
    }

    @Test fun lateCallbackForAbandonedJobDoesNotCompleteNewJobOrSkipCleanup() {
        Harness().use { h ->
            val closed = CountDownLatch(1)
            h.start(byteArrayOf(1), { object : ByteArrayOutputStream() { override fun close() { closed.countDown() } } })
            assertTrue(closed.await(3, TimeUnit.SECONDS))
            h.current = false
            h.task.cancel()
            h.awaitWorker()
            while (true) (h.callbacks.poll() ?: break).run()
            assertNull(h.success)
            assertNull(h.failure)
            assertTrue(h.temporaryRemoved && h.destinationRemoved && h.settled)
        }
    }

    @Test fun destinationOwnershipSurvivesRestartAndStaleReleaseCannotStealNewLease() {
        val key = "content://synthetic/publication-test"
        val old = Any(); val next = Any()
        try {
            assertTrue(ExportPublicationDestinations.claim(key, old))
            assertTrue(ExportPublicationDestinations.isInUse(key))
            assertFalse(ExportPublicationDestinations.claim(key, next))
            ExportPublicationDestinations.release(key, next)
            assertTrue(ExportPublicationDestinations.isInUse(key))
            ExportPublicationDestinations.release(key, old)
            assertTrue(ExportPublicationDestinations.claim(key, next))
            ExportPublicationDestinations.release(key, old)
            assertTrue(ExportPublicationDestinations.isInUse(key))
        } finally { ExportPublicationDestinations.release(key, old); ExportPublicationDestinations.release(key, next) }
    }
}
