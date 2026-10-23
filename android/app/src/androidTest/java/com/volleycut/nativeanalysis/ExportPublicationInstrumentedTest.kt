package com.volleycut.nativeanalysis

import android.os.Handler
import android.os.Looper
import android.os.ParcelFileDescriptor
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.ByteArrayInputStream
import java.io.FilterOutputStream
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

@RunWith(AndroidJUnit4::class)
class ExportPublicationInstrumentedTest {
    @Test fun nonseekablePipePublishesOffMainAndKeepsUiResponsive() = exercisePipe(false)
    @Test fun cancellingBlockedPipeKeepsUiResponsiveAndCleansBeforeCompletion() = exercisePipe(true)

    private fun exercisePipe(cancel: Boolean) {
        val handler = Handler(Looper.getMainLooper())
        val executor = Executors.newSingleThreadExecutor()
        val pipe = ParcelFileDescriptor.createPipe()
        val data = ByteArray(4 * 1_024 * 1_024 + 5) { (it % 251).toByte() }
        val opening = CountDownLatch(1)
        val terminal = CountDownLatch(1)
        val settled = CountDownLatch(1)
        val cleaned = AtomicBoolean(false)
        val error = AtomicReference<Throwable?>()
        val succeeded = AtomicBoolean(false)
        val wasCancelled = AtomicBoolean(false)
        val received = AtomicReference<ByteArray?>()
        val task = ExportFilePublication(executor, { handler.post(it) },
            { ByteArrayInputStream(data) },
            { assertNotEquals(Looper.getMainLooper(), Looper.myLooper())
                object : FilterOutputStream(ParcelFileDescriptor.AutoCloseOutputStream(pipe[1])) {
                    override fun write(value: ByteArray, offset: Int, count: Int) { opening.countDown(); out.write(value, offset, count) }
                }
            },
            data.size.toLong(), { "synthetic.mp4" }, {},
            { assertNotEquals(Looper.getMainLooper(), Looper.myLooper()); cleaned.set(true) },
            { true }, { _, _ -> assertEquals(Looper.getMainLooper(), Looper.myLooper()) },
            { succeeded.set(true); terminal.countDown() },
            { failure, cancelled -> error.set(failure); wasCancelled.set(cancelled); terminal.countDown() },
            { settled.countDown() })
        var reader: Thread? = null
        try {
            task.start()
            assertTrue(opening.await(5, TimeUnit.SECONDS))
            // A pipe with no reader fills and blocks the worker. Main still
            // handles a UI callback and (for the second case) cancellation.
            val responsive = CountDownLatch(1)
            handler.post { if (cancel) task.cancel(); responsive.countDown() }
            assertTrue("Main thread blocked by publication", responsive.await(2, TimeUnit.SECONDS))
            reader = Thread {
                runCatching { ParcelFileDescriptor.AutoCloseInputStream(pipe[0]).use { received.set(it.readBytes()) } }
                    .onFailure { error.compareAndSet(null, it) }
            }.also { it.start() }
            assertTrue(terminal.await(10, TimeUnit.SECONDS))
            assertTrue(settled.await(5, TimeUnit.SECONDS))
            reader.join(5_000)
            if (cancel) {
                assertFalse(succeeded.get())
                assertTrue(wasCancelled.get() && cleaned.get())
            } else {
                assertNull(error.get())
                assertTrue(succeeded.get())
                assertFalse(cleaned.get())
                assertArrayEquals(data, received.get())
            }
        } finally {
            task.cancel()
            runCatching { pipe[0].close() }; runCatching { pipe[1].close() }
            executor.shutdownNow()
            executor.awaitTermination(5, TimeUnit.SECONDS)
            reader?.join(2_000)
        }
    }
}
