package com.volleycut.nativeanalysis

import java.io.InputStream
import java.io.OutputStream
import java.util.concurrent.CancellationException
import java.util.concurrent.CompletableFuture
import java.util.concurrent.Executor
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.ConcurrentHashMap

/** A service restart must not reopen a destination still owned by a cancelled worker. */
internal object ExportPublicationDestinations {
    private val owners = ConcurrentHashMap<String, Any>()
    fun isInUse(destination: String) = owners.containsKey(destination)
    fun claim(destination: String, owner: Any) = owners.putIfAbsent(destination, owner) == null
    fun release(destination: String, owner: Any) { owners.remove(destination, owner) }
}

/** Owns streams and cleanup on a worker until the current main-thread job accepts success. */
internal class ExportFilePublication(
    private val worker: Executor,
    private val dispatch: (Runnable) -> Unit,
    private val openInput: () -> InputStream,
    private val openOutput: () -> OutputStream,
    private val totalBytes: Long,
    private val displayName: () -> String,
    private val removeTemporary: () -> Unit,
    private val removeIncomplete: () -> Unit,
    private val isCurrent: () -> Boolean,
    private val onProgress: (Long, Long) -> Unit,
    private val onSuccess: (String) -> Unit,
    private val onFailure: (Throwable, Boolean) -> Unit,
    private val onSettled: () -> Unit = {},
) {
    // Main callbacks and cancellation compete to commit this particular job once.
    private val state = AtomicInteger(ACTIVE)
    private val accepted = CompletableFuture<Boolean>()
    @Volatile private var copyingThread: Thread? = null

    fun start() = worker.execute {
        copyingThread = Thread.currentThread()
        val result = runCatching {
            checkActive()
            require(totalBytes > 0) { "Temporary export is empty" }
            openOutput().use { output -> openInput().use { input ->
                val buffer = ByteArray(1_024 * 1_024)
                var copied = 0L
                var reportedAt = 0L
                while (true) {
                    checkActive()
                    val count = input.read(buffer)
                    if (count < 0) break
                    if (count == 0) continue
                    checkActive()
                    output.write(buffer, 0, count)
                    copied += count
                    val now = System.nanoTime()
                    if (copied >= totalBytes || now - reportedAt >= 500_000_000L) {
                        reportedAt = now
                        val bytes = copied
                        dispatch(Runnable { if (state.get() == ACTIVE && isCurrent()) onProgress(bytes, totalBytes) })
                    }
                }
                require(copied == totalBytes) { "Temporary export size changed during publication" }
            } }
            checkActive()
            val name = displayName()
            checkActive()
            dispatch(Runnable {
                if (isCurrent() && state.compareAndSet(ACTIVE, COMMITTED)) {
                    accepted.complete(true)
                    onSuccess(name)
                } else accepted.complete(false)
            })
            if (!accepted.get()) throw CancellationException("Export publication cancelled")
        }
        // Providers may respond to cancellation with InterruptedException. Clear
        // that flag before cleanup; no provider work is performed by the UI.
        Thread.interrupted()
        runCatching(removeTemporary)
        if (state.get() != COMMITTED) {
            runCatching(removeIncomplete)
            runCatching(onSettled)
            val error = result.exceptionOrNull() ?: CancellationException("Export publication cancelled")
            val cancelled = state.get() == CANCELLED
            dispatch(Runnable { if (isCurrent()) onFailure(error, cancelled) })
        } else runCatching(onSettled)
        copyingThread = null
    }

    fun cancel() {
        if (state.compareAndSet(ACTIVE, CANCELLED)) {
            accepted.complete(false)
            copyingThread?.interrupt()
        }
    }

    private fun checkActive() {
        if (state.get() != ACTIVE || Thread.currentThread().isInterrupted) {
            throw CancellationException("Export publication cancelled")
        }
    }

    private companion object { const val ACTIVE = 0; const val CANCELLED = 1; const val COMMITTED = 2 }
}
