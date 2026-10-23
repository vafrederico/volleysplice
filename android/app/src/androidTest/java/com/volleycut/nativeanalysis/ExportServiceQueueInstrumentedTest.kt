package com.volleycut.nativeanalysis

import android.content.BroadcastReceiver
import android.content.ContentProvider
import android.content.ContentResolver
import android.content.ContentValues
import android.content.Context
import android.content.ContextWrapper
import android.content.Intent
import android.content.IntentFilter
import android.database.MatrixCursor
import android.net.Uri
import android.os.Looper
import android.os.ParcelFileDescriptor
import android.os.PowerManager
import android.os.SystemClock
import android.provider.OpenableColumns
import androidx.activity.ComponentActivity
import androidx.media3.transformer.ExportResult
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.UUID
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutorService
import java.util.concurrent.TimeUnit

@RunWith(AndroidJUnit4::class)
class ExportServiceQueueInstrumentedTest {
    @Test fun cancellingQueuedJobsDoesNotTouchBlockedPublicationOrExistingFile() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val scenario = ActivityScenario.launch(ComponentActivity::class.java)
        val directory = File(context.cacheDir, "export-queue-test-${UUID.randomUUID()}").apply { mkdirs() }
        val provider = RecordingDestinationProvider(directory)
        val target = Uri.parse("content://export-queue-test/active.mp4")
        val existingTarget = Uri.parse("content://export-queue-test/existing.mp4")
        val bytes = ByteArray(2 * 1_024 * 1_024 + 5) { (it % 251).toByte() }
        val existingBytes = "pre-existing document".toByteArray()
        provider.file(existingTarget).writeBytes(existingBytes)
        val temporary = File(directory, "encoded.mp4").apply { writeBytes(bytes) }
        val activeId = "active-${UUID.randomUUID()}"
        val queuedId = "queued-${UUID.randomUUID()}"
        val existingId = "existing-${UUID.randomUUID()}"
        val completed = CountDownLatch(1)
        val broadcasts = CopyOnWriteArrayList<Intent>()
        val receiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context, intent: Intent) {
                broadcasts.add(intent)
                if (intent.getStringExtra(ExportService.EXTRA_JOB_ID) == activeId &&
                    intent.getStringExtra(ExportService.EXTRA_STATUS) == "complete"
                ) completed.countDown()
            }
        }
        context.registerReceiver(receiver, IntentFilter(ExportService.ACTION_PROGRESS), Context.RECEIVER_NOT_EXPORTED)
        var service: ExportService? = null
        try {
            scenario.onActivity { it.startService(Intent(it, ExportService::class.java)) }
            service = awaitRunningService()
            instrumentation.runOnMainSync {
                val running = checkNotNull(service)
                // Use the framework-created service and its real queue/callbacks.
                // Only substitute the document provider and an already encoded job;
                // no codec run or production test hook is needed for this race.
                val base = running.baseContext
                val resolver = ContentResolver.wrap(provider)
                ContextWrapper::class.java.getDeclaredField("mBase").apply { isAccessible = true }
                    .set(running, object : ContextWrapper(base) {
                        override fun getContentResolver() = resolver
                    })
                running.setField("currentJob", running.call("exportJob", request(activeId, target)))
                running.setField("destination", target)
                running.setField("temporaryFile", temporary)
                running.setField("sourceName", "synthetic.mp4")
                running.setField("outputWriteMode", "temporary_then_copy")
                running.setField("startedAtMs", SystemClock.elapsedRealtime())
                running.setField("outputDurationMs", 1_000L)
                running.setField("intervalCount", 1)
                running.call("startForegroundCompat", running.call("notification", 90, "Saving encoded file", true))
                running.setField("foreground", true)
                (running.field("wakeLock") as PowerManager.WakeLock).acquire(30_000)
                running.call("publishTemporaryOutput", activeId, ExportResult.Builder().build())
            }
            assertTrue("Publication did not reach the blocked provider", provider.nameRequested.await(5, TimeUnit.SECONDS))
            val running = checkNotNull(service)
            instrumentation.runOnMainSync {
                assertTrue(ExportPublicationDestinations.isInUse(target.toString()))
                running.onStartCommand(request(queuedId, target), 0, 2)
                running.onStartCommand(request(existingId, existingTarget), 0, 3)
                assertEquals(3, ExportService.pendingCount())
                assertEquals("queued", ExportService.statusForProject(queuedId)?.status)

                // Exercise ACTION_CANCEL through the service, not the publisher helper.
                running.onStartCommand(cancel(queuedId), 0, 4)
                assertEquals("cancelled", ExportService.statusForProject(queuedId)?.status)
                assertEquals(2, ExportService.pendingCount())
                assertEquals("Queued behind 1 export", ExportService.statusForProject(existingId)?.detail)
                running.onStartCommand(cancel(existingId), 0, 5)
                assertEquals("cancelled", ExportService.statusForProject(existingId)?.status)
                assertEquals(1, ExportService.pendingCount())
                assertEquals("running", ExportService.statusForProject(activeId)?.status)
                assertEquals(99, ExportService.statusForProject(activeId)?.progress)
                assertNotNull(running.field("publication"))
                assertEquals(true, running.field("foreground"))
                assertTrue((running.field("wakeLock") as PowerManager.WakeLock).isHeld)
                assertTrue(ExportPublicationDestinations.isInUse(target.toString()))
            }
            assertTrue("Queued cancellation requested destination deletion", provider.deletes.isEmpty())
            assertEquals("Queued cancellation reopened/truncated a destination", listOf(target to "w"), provider.opens.toList())
            assertArrayEquals(existingBytes, provider.file(existingTarget).readBytes())
            assertEquals(1L, completed.count)

            provider.allowName.countDown()
            assertTrue("Active publication did not complete", completed.await(5, TimeUnit.SECONDS))
            val worker = running.field("publicationWorker") as ExecutorService
            worker.shutdown()
            assertTrue("Publication worker did not settle", worker.awaitTermination(5, TimeUnit.SECONDS))
            instrumentation.runOnMainSync {
                assertEquals("complete", ExportService.statusForProject(activeId)?.status)
                assertEquals(100, ExportService.statusForProject(activeId)?.progress)
                assertEquals(0, ExportService.pendingCount())
                assertNull(running.field("currentJob"))
                assertEquals(false, running.field("foreground"))
                assertFalse((running.field("wakeLock") as PowerManager.WakeLock).isHeld)
            }
            assertFalse(ExportPublicationDestinations.isInUse(target.toString()))
            assertFalse(temporary.exists())
            assertArrayEquals(bytes, provider.file(target).readBytes())
            assertArrayEquals(existingBytes, provider.file(existingTarget).readBytes())
            assertTrue(provider.deletes.isEmpty())
            assertEquals(listOf(target to "w"), provider.opens.toList())
            for ((id, pending) in listOf(queuedId to 2, existingId to 1)) {
                val cancelled = broadcasts.single {
                    it.getStringExtra(ExportService.EXTRA_JOB_ID) == id &&
                        it.getStringExtra(ExportService.EXTRA_STATUS) == "cancelled"
                }
                assertEquals(0, cancelled.getIntExtra(ExportService.EXTRA_PROGRESS, -1))
                assertEquals(pending, cancelled.getIntExtra(ExportService.EXTRA_QUEUE_COUNT, -1))
            }
        } finally {
            provider.allowName.countDown()
            context.stopService(Intent(context, ExportService::class.java))
            instrumentation.waitForIdleSync()
            (service?.field("publicationWorker") as? ExecutorService)?.awaitTermination(5, TimeUnit.SECONDS)
            context.unregisterReceiver(receiver)
            scenario.close()
            directory.deleteRecursively()
        }
    }

    private fun request(id: String, destination: Uri) = Intent().apply {
        putExtra(ExportService.EXTRA_JOB_ID, id)
        putExtra(ExportService.EXTRA_PROJECT_ID, id)
        putExtra(ExportService.EXTRA_SOURCE_URI, "content://export-queue-test/source.mp4")
        putExtra(ExportService.EXTRA_SOURCE_NAME, "synthetic.mp4")
        putExtra(ExportService.EXTRA_SOURCE_DURATION_MS, 1_000L)
        putExtra(ExportService.EXTRA_DESTINATION_URI, destination.toString())
        putExtra(ExportService.EXTRA_INTERVAL_STARTS, longArrayOf(0))
        putExtra(ExportService.EXTRA_INTERVAL_ENDS, longArrayOf(1_000))
    }

    private fun cancel(id: String) = Intent(ExportService.ACTION_CANCEL).putExtra(ExportService.EXTRA_JOB_ID, id)

    private fun awaitRunningService(): ExportService {
        check(Looper.myLooper() != Looper.getMainLooper()) { "Service registration must be awaited off main" }
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val activityThread = Class.forName("android.app.ActivityThread")
        val currentThread = activityThread.getDeclaredMethod("currentActivityThread")
        val servicesField = activityThread.getDeclaredField("mServices").apply { isAccessible = true }
        val deadline = SystemClock.elapsedRealtime() + 10_000
        var matches = emptyList<ExportService>()
        var registered = emptyList<String>()
        do {
            // Framework creation is asynchronous even after waitForIdleSync.
            // Inspect its real instances on main, then leave main free to create them.
            instrumentation.runOnMainSync {
                val thread = currentThread.invoke(null)
                val services = servicesField.get(thread) as Map<*, *>
                matches = services.values.filterIsInstance<ExportService>()
                registered = services.values.map { it?.javaClass?.name ?: "null" }
            }
            if (matches.size == 1) return matches.single()
            if (matches.size > 1) {
                throw AssertionError("Expected one framework-registered ExportService, found ${matches.size}; " +
                    "registered service classes=$registered")
            }
            SystemClock.sleep(25)
        } while (SystemClock.elapsedRealtime() < deadline)
        throw AssertionError("Timed out after 10s awaiting one framework-registered ExportService; " +
            "matches=${matches.size}, registered service classes=$registered")
    }

    private fun ExportService.field(name: String) = ExportService::class.java.getDeclaredField(name)
        .apply { isAccessible = true }.get(this)

    private fun ExportService.setField(name: String, value: Any?) = ExportService::class.java.getDeclaredField(name)
        .apply { isAccessible = true }.set(this, value)

    private fun ExportService.call(name: String, vararg args: Any?) = ExportService::class.java.declaredMethods
        .single { it.name == name }.apply { isAccessible = true }.invoke(this, *args)

    /** In-process provider exists only in androidTest; no app manifest/debug hook. */
    private class RecordingDestinationProvider(private val directory: File) : ContentProvider() {
        val nameRequested = CountDownLatch(1)
        val allowName = CountDownLatch(1)
        val deletes = CopyOnWriteArrayList<Uri>()
        val opens = CopyOnWriteArrayList<Pair<Uri, String>>()
        fun file(uri: Uri) = File(directory, checkNotNull(uri.lastPathSegment))
        override fun onCreate() = true
        override fun getType(uri: Uri) = "video/mp4"
        override fun openFile(uri: Uri, mode: String): ParcelFileDescriptor {
            opens.add(uri to mode)
            return ParcelFileDescriptor.open(file(uri), ParcelFileDescriptor.parseMode(mode))
        }
        override fun query(uri: Uri, projection: Array<out String>?, selection: String?,
            selectionArgs: Array<out String>?, sortOrder: String?): MatrixCursor {
            nameRequested.countDown()
            check(allowName.await(10, TimeUnit.SECONDS)) { "Provider was not released by the test" }
            return MatrixCursor(arrayOf(OpenableColumns.DISPLAY_NAME)).apply { addRow(arrayOf(uri.lastPathSegment)) }
        }
        override fun delete(uri: Uri, selection: String?, selectionArgs: Array<out String>?): Int {
            deletes.add(uri)
            // A declined delete forces the old bug through its truncation fallback.
            return 0
        }
        override fun insert(uri: Uri, values: ContentValues?): Uri? = error("Not supported")
        override fun update(uri: Uri, values: ContentValues?, selection: String?, selectionArgs: Array<out String>?): Int =
            error("Not supported")
    }
}
