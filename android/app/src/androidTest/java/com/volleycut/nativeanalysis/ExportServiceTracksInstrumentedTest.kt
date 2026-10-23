package com.volleycut.nativeanalysis

import android.content.BroadcastReceiver
import android.content.ContentProvider
import android.content.ContentResolver
import android.content.ContentValues
import android.content.Context
import android.content.ContextWrapper
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Color
import android.database.MatrixCursor
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Bundle
import android.os.Looper
import android.os.ParcelFileDescriptor
import android.util.Base64
import androidx.activity.ComponentActivity
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.FileNotFoundException
import java.nio.ByteBuffer
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.CopyOnWriteArrayList

/** Uses the stable service Intent API so this test APK also runs against the pre-fix app. */
@RunWith(AndroidJUnit4::class)
class ExportServiceTracksInstrumentedTest {
    @Test fun videoOnlyCutsWithoutScoresStayVideoOnly() = export("overlay-fixture.mp4.b64", false, false)
    @Test fun videoOnlyCutsWithScoresStayVideoOnly() = export("overlay-fixture.mp4.b64", false, true)
    @Test fun aacCutsWithoutScoresKeepSourceAudio() = export("draft-recovery-fixture.mp4.b64", true, false)
    @Test fun aacCutsWithScoresKeepSourceAudio() = export("draft-recovery-fixture.mp4.b64", true, true)

    @Test fun cancellingBlockedPreparationDoesNotStarveNextExport() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val id = UUID.randomUUID().toString()
        val blockedId = UUID.randomUUID().toString()
        val nextId = UUID.randomUUID().toString()
        val directory = File(context.cacheDir, "export-cancel-$id").apply { check(mkdirs()) }
        val input = File(directory, "source-$id.mp4")
        val blockedOutput = File(directory, "final-$blockedId.mp4")
        val nextOutput = File(directory, "final-$nextId.mp4")
        val source = Uri.parse("content://export-cancel-$id/source-$id.mp4")
        val entered = CountDownLatch(1)
        val release = CountDownLatch(1)
        val interrupted = CountDownLatch(1)
        val blockedThread = AtomicReference<Thread>()
        val descriptors = CopyOnWriteArrayList<ParcelFileDescriptor>()
        val provider = object : ContentProvider() {
            override fun onCreate() = true
            override fun getType(uri: Uri) = "video/mp4"
            override fun openFile(uri: Uri, mode: String): ParcelFileDescriptor {
                if (uri != source) {
                    check(uri == Uri.fromFile(blockedOutput) && mode != "r") { "Unexpected fixture request" }
                    return ParcelFileDescriptor.open(blockedOutput, ParcelFileDescriptor.parseMode(mode))
                }
                check(mode == "r") { "Source opened for writing" }
                blockedThread.compareAndSet(null, Thread.currentThread())
                entered.countDown()
                val deadline = android.os.SystemClock.elapsedRealtime() + 45_000
                while (release.count > 0 && android.os.SystemClock.elapsedRealtime() < deadline) {
                    try {
                        release.await(100, TimeUnit.MILLISECONDS)
                    } catch (_: InterruptedException) {
                        // Deliberately emulate a provider that ignores cancellation/interruption.
                        interrupted.countDown()
                    }
                }
                check(release.count == 0L) { "Test did not release the blocked provider" }
                return ParcelFileDescriptor.open(input, ParcelFileDescriptor.MODE_READ_ONLY).also(descriptors::add)
            }
            override fun query(uri: Uri, projection: Array<out String>?, selection: String?,
                selectionArgs: Array<out String>?, sortOrder: String?) = MatrixCursor(arrayOf("_display_name"))
            override fun insert(uri: Uri, values: ContentValues?): Uri? = error("Not supported")
            override fun delete(uri: Uri, selection: String?, selectionArgs: Array<out String>?) =
                if (uri == Uri.fromFile(blockedOutput) && blockedOutput.delete()) 1 else 0
            override fun update(uri: Uri, values: ContentValues?, selection: String?, selectionArgs: Array<out String>?) = 0
        }
        val scenario = ActivityScenario.launch(ComponentActivity::class.java)
        try {
            input.writeBytes(Base64.decode(instrumentation.context.assets.open("overlay-fixture.mp4.b64")
                .bufferedReader().use { it.readText() }, Base64.DEFAULT))
            scenario.onActivity { it.startService(Intent(it, ExportService::class.java)) }
            instrumentation.waitForIdleSync()
            lateinit var running: ExportService
            instrumentation.runOnMainSync {
                running = runningService()
                installResolver(running, ContentResolver.wrap(provider))
                running.onStartCommand(twoCutExportRequest(context, blockedId, source, Uri.fromFile(blockedOutput)), 0, 1)
            }
            assertTrue("Blocked source read did not start", entered.await(5, TimeUnit.SECONDS))
            instrumentation.runOnMainSync {
                running.onStartCommand(twoCutExportRequest(context, nextId, Uri.fromFile(input), Uri.fromFile(nextOutput)), 0, 2)
                assertEquals("queued", ExportService.statusForProject(nextId)?.status)
                running.onStartCommand(Intent(ExportService.ACTION_CANCEL).putExtra(ExportService.EXTRA_JOB_ID, blockedId), 0, 3)
                assertEquals("cancelled", ExportService.statusForProject(blockedId)?.status)
            }
            val deadline = android.os.SystemClock.elapsedRealtime() + 30_000
            while (ExportService.statusForProject(nextId)?.status !in setOf("complete", "failed", "cancelled") &&
                android.os.SystemClock.elapsedRealtime() < deadline) Thread.sleep(20)
            assertEquals("The healthy successor was starved by a cancelled provider read", "complete", ExportService.statusForProject(nextId)?.status)
            assertEquals("The successor must finish while the original provider is still blocked", 1L, release.count)
            assertTrue("Cancelled preparation was not interrupted", interrupted.await(5, TimeUnit.SECONDS))
            val worker = checkNotNull(blockedThread.get())
            assertTrue("The provider did not keep the cancelled read blocked", worker.isAlive)
            val bytes = nextOutput.readBytes()
            assertTrue(bytes.isNotEmpty())
            assertEquals(1, tracks(nextOutput).count { it.mime.startsWith("video/") && it.sampleBytes > 0 })
            assertFalse(tracks(nextOutput).any { it.mime.startsWith("audio/") })
            release.countDown()
            worker.join(5_000)
            assertFalse("Cancelled preparation worker did not finish after its provider returned", worker.isAlive)
            assertTrue("The cancelled source was never released by the provider", descriptors.isNotEmpty())
            assertTrue("A source descriptor leaked after cancellation", descriptors.all { !it.fileDescriptor.valid() })
            instrumentation.waitForIdleSync()
            assertEquals("cancelled", ExportService.statusForProject(blockedId)?.status)
            assertEquals("complete", ExportService.statusForProject(nextId)?.status)
            assertArrayEquals("A late callback changed the successor output", bytes, nextOutput.readBytes())
            assertEquals(0, ExportService.pendingCount())
            assertTrue(!blockedOutput.exists() || blockedOutput.length() == 0L)
        } finally {
            release.countDown()
            context.stopService(Intent(context, ExportService::class.java))
            instrumentation.waitForIdleSync()
            blockedThread.get()?.join(5_000)
            descriptors.forEach { it.close() }
            scenario.close()
            directory.deleteRecursively()
        }
    }

    @Test fun failedAudioTrackProbeFailsTheJobOffMainThread() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val id = UUID.randomUUID().toString()
        val directory = File(context.cacheDir, "export-probe-$id").apply { check(mkdirs()) }
        val input = File(directory, "source-$id.mp4")
        val output = File(directory, "final-$id.mp4")
        val source = Uri.parse("content://export-probe-$id/source-$id.mp4")
        val target = Uri.fromFile(output)
        val colorReads = AtomicInteger()
        val audioProbeFailures = AtomicInteger()
        val unexpectedSourceReads = CopyOnWriteArrayList<String>()
        val mainThreadRead = AtomicReference(false)
        // Identify the production caller, not an assumed number/order of provider opens.
        // Platform retries are permitted within a stage; failure cleanup opens the destination too.
        val provider = object : ContentProvider() {
            override fun onCreate() = true
            override fun getType(uri: Uri) = "audio/unknown"
            override fun openFile(uri: Uri, mode: String): ParcelFileDescriptor {
                if (uri == target) {
                    check(mode != "r") { "Destination opened for reading during probe failure" }
                    return ParcelFileDescriptor.open(output, ParcelFileDescriptor.parseMode(mode))
                }
                check(uri == source) { "Unexpected fixture URI" }
                check(mode == "r") { "Source opened for writing" }
                if (Looper.myLooper() == Looper.getMainLooper()) mainThreadRead.set(true)
                val callers = Thread.currentThread().stackTrace
                if (callers.any { it.className == ExportColorPolicy.Companion::class.java.name && it.methodName == "read" }) {
                    val descriptor = ParcelFileDescriptor.open(input, ParcelFileDescriptor.MODE_READ_ONLY)
                    colorReads.incrementAndGet()
                    return descriptor
                }
                // A stack-frame name adds no binary dependency on the new private method,
                // so the same APK's export cases still work against the old app.
                if (callers.any { it.className == ExportService::class.java.name && it.methodName == "sourceHasAudio" }) {
                    audioProbeFailures.incrementAndGet()
                    throw FileNotFoundException("Synthetic audio-track probe failure")
                }
                unexpectedSourceReads.add(callers.joinToString { "${it.className}.${it.methodName}" })
                throw FileNotFoundException("Unexpected source read outside the color/audio probes")
            }
            override fun query(uri: Uri, projection: Array<out String>?, selection: String?,
                selectionArgs: Array<out String>?, sortOrder: String?) = MatrixCursor(arrayOf("_display_name"))
            override fun insert(uri: Uri, values: ContentValues?): Uri? = error("Not supported")
            override fun delete(uri: Uri, selection: String?, selectionArgs: Array<out String>?) =
                if (uri == target && output.delete()) 1 else 0
            override fun update(uri: Uri, values: ContentValues?, selection: String?, selectionArgs: Array<out String>?) = 0
        }
        val scenario = ActivityScenario.launch(ComponentActivity::class.java)
        try {
            input.writeBytes(Base64.decode(instrumentation.context.assets.open("draft-recovery-fixture.mp4.b64")
                .bufferedReader().use { it.readText() }, Base64.DEFAULT))
            scenario.onActivity { it.startService(Intent(it, ExportService::class.java)) }
            instrumentation.waitForIdleSync()
            instrumentation.runOnMainSync {
                val running = runningService()
                installResolver(running, ContentResolver.wrap(provider))
                running.onStartCommand(Intent(context, ExportService::class.java).apply {
                    putExtra(ExportService.EXTRA_JOB_ID, id)
                    putExtra(ExportService.EXTRA_PROJECT_ID, id)
                    putExtra(ExportService.EXTRA_SOURCE_URI, source.toString())
                    putExtra(ExportService.EXTRA_SOURCE_NAME, input.name)
                    putExtra(ExportService.EXTRA_SOURCE_DURATION_MS, 1_000L)
                    putExtra(ExportService.EXTRA_DESTINATION_URI, target.toString())
                    putExtra(ExportService.EXTRA_INTERVAL_STARTS, longArrayOf(0))
                    putExtra(ExportService.EXTRA_INTERVAL_ENDS, longArrayOf(1_000))
                }, 0, 1)
            }
            val deadline = android.os.SystemClock.elapsedRealtime() + 10_000
            while (ExportService.statusForProject(id)?.status !in setOf("complete", "failed", "cancelled") &&
                android.os.SystemClock.elapsedRealtime() < deadline) Thread.sleep(20)
            assertEquals("Probe failure must not become a successful silent export", "failed", ExportService.statusForProject(id)?.status)
            assertTrue("The production color probe never obtained a readable descriptor", colorReads.get() > 0)
            // In production the audio probe is invoked only after ExportColorPolicy.read returns.
            // Its independently identified request therefore also proves the color read succeeded.
            assertTrue("The production audio-track probe was never faulted", audioProbeFailures.get() > 0)
            assertTrue("Unclassified source reads: $unexpectedSourceReads", unexpectedSourceReads.isEmpty())
            assertFalse("Source track I/O ran on the service main thread", mainThreadRead.get())
            val metrics = JSONObject(checkNotNull(ExportService.statusForProject(id)?.metrics))
            assertEquals("A probe failure must happen before opening the encoder output", "uninitialized", metrics.getString("outputWriteMode"))
            assertTrue("A probe failure must report its error", metrics.getString("errorType").isNotBlank())
            assertTrue("Failed probe published output", !output.exists() || output.length() == 0L)
        } finally {
            context.startService(Intent(context, ExportService::class.java).setAction(ExportService.ACTION_CANCEL)
                .putExtra(ExportService.EXTRA_JOB_ID, id))
            // This test creates the service itself solely to substitute an in-process failing provider.
            context.stopService(Intent(context, ExportService::class.java))
            instrumentation.waitForIdleSync()
            scenario.close()
            directory.deleteRecursively()
        }
    }

    private fun runningService(): ExportService {
        val activityThread = Class.forName("android.app.ActivityThread")
        val thread = activityThread.getDeclaredMethod("currentActivityThread").invoke(null)
        val services = activityThread.getDeclaredField("mServices").apply { isAccessible = true }.get(thread) as Map<*, *>
        return services.values.filterIsInstance<ExportService>().single()
    }

    private fun installResolver(service: ExportService, resolver: ContentResolver) {
        val base = service.baseContext
        ContextWrapper::class.java.getDeclaredField("mBase").apply { isAccessible = true }
            .set(service, object : ContextWrapper(base) {
                override fun getContentResolver() = resolver
            })
    }

    private fun twoCutExportRequest(context: Context, id: String, source: Uri, target: Uri) = Intent(context, ExportService::class.java).apply {
        putExtra(ExportService.EXTRA_JOB_ID, id)
        putExtra(ExportService.EXTRA_PROJECT_ID, id)
        putExtra(ExportService.EXTRA_SOURCE_URI, source.toString())
        putExtra(ExportService.EXTRA_SOURCE_NAME, "source-$id.mp4")
        putExtra(ExportService.EXTRA_SOURCE_DURATION_MS, 1_000L)
        putExtra(ExportService.EXTRA_DESTINATION_URI, target.toString())
        // Use the same proven cut sequence as the export cases; a full-clip transmux
        // of this untagged fixture can independently fail the production color guard.
        putExtra(ExportService.EXTRA_INTERVAL_STARTS, longArrayOf(0, 600))
        putExtra(ExportService.EXTRA_INTERVAL_ENDS, longArrayOf(400, 1_000))
    }

    private fun export(asset: String, hasAudio: Boolean, scores: Boolean) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val id = UUID.randomUUID().toString()
        val retainArtifacts = InstrumentationRegistry.getArguments().getString("retainExportTracksArtifacts") == "true"
        val artifactRoot = if (retainArtifacts) checkNotNull(context.getExternalFilesDir("export-track-evidence")) else context.cacheDir
        val directory = File(artifactRoot, "export-tracks-$id").apply { check(mkdirs()) }
        val input = File(directory, "source-$id.mp4")
        val output = File(directory, "final-$id.mp4")
        val terminal = CountDownLatch(1)
        val result = AtomicReference<Intent>()
        val receiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context, intent: Intent) {
                if (intent.getStringExtra(ExportService.EXTRA_JOB_ID) == id &&
                    intent.getStringExtra(ExportService.EXTRA_STATUS) in setOf("complete", "failed", "cancelled")
                ) {
                    result.set(intent)
                    terminal.countDown()
                }
            }
        }
        context.registerReceiver(receiver, IntentFilter(ExportService.ACTION_PROGRESS), Context.RECEIVER_NOT_EXPORTED)
        val scenario = ActivityScenario.launch(ComponentActivity::class.java)
        try {
            val encoded = instrumentation.context.assets.open(asset).bufferedReader().use { it.readText() }
            input.writeBytes(Base64.decode(encoded, Base64.DEFAULT))
            val sourceTracks = tracks(input)
            assertEquals(1, sourceTracks.count { it.mime.startsWith("video/") })
            assertEquals(if (hasAudio) 1 else 0, sourceTracks.count { it.mime.startsWith("audio/") })
            if (hasAudio) {
                val original = sourceTracks.single { it.mime.startsWith("audio/") }
                assertTrue("The AAC input fixture has no readable samples", original.sampleCount > 0 && original.sampleBytes > 0)
            }
            val request = Intent(context, ExportService::class.java).apply {
                putExtra(ExportService.EXTRA_JOB_ID, id)
                putExtra(ExportService.EXTRA_PROJECT_ID, id)
                putExtra(ExportService.EXTRA_SOURCE_URI, Uri.fromFile(input).toString())
                putExtra(ExportService.EXTRA_SOURCE_NAME, input.name)
                putExtra(ExportService.EXTRA_SOURCE_DURATION_MS, 1_000L)
                putExtra(ExportService.EXTRA_DESTINATION_URI, Uri.fromFile(output).toString())
                // Two real cuts, with a removed middle gap, exercise the production sequence.
                putExtra(ExportService.EXTRA_INTERVAL_STARTS, longArrayOf(0, 600))
                putExtra(ExportService.EXTRA_INTERVAL_ENDS, longArrayOf(400, 1_000))
                if (scores) putExtra(ExportService.EXTRA_SCORE_SNAPSHOT, ScoreExportSnapshotJson.encode(
                    ScoreExportSnapshot(true, scoreTracking = ScoreTracking(team1Name = "Falcons", team2Name = "Wolves"), ignoredIntervals = emptyList(),
                        excludedRallyIds = emptySet(), rallyRanges = emptyList()),
                ))
            }
            scenario.onActivity { it.startForegroundService(request) }
            assertTrue("Service export timed out", terminal.await(60, TimeUnit.SECONDS))
            val finished = checkNotNull(result.get())
            assertEquals(finished.getStringExtra(ExportService.EXTRA_DETAIL), "complete",
                finished.getStringExtra(ExportService.EXTRA_STATUS))
            assertTrue("No final MP4 was published", output.length() > 0)
            val exportedTracks = tracks(output)
            val video = exportedTracks.filter { it.mime.startsWith("video/") }.single()
            assertEquals("video/avc", video.mime)
            assertTrue("Cut video duration was ${video.durationUs}", video.durationUs in 700_000L..900_000L)
            val audio = exportedTracks.filter { it.mime.startsWith("audio/") }
            assertEquals("Video-only sources must not acquire synthesized audio", if (hasAudio) 1 else 0, audio.size)
            val metrics = JSONObject(checkNotNull(finished.getStringExtra(ExportService.EXTRA_METRICS)))
            assertEquals(2, metrics.getInt("intervalCount"))
            assertEquals(800L, metrics.getLong("outputDurationMs"))
            if (hasAudio) {
                val original = sourceTracks.single { it.mime.startsWith("audio/") }
                assertEquals("audio/mp4a-latm", audio.single().mime)
                assertEquals(original.sampleRate, audio.single().sampleRate)
                assertEquals(original.channels, audio.single().channels)
                assertTrue("Source audio lost its samples", audio.single().sampleCount > 0)
                assertTrue("Exported audio has no readable sample bytes", audio.single().sampleBytes > 0)
                assertTrue("Cut audio duration was ${audio.single().durationUs}", audio.single().durationUs in 700_000L..950_000L)
            } else {
                assertTrue("Export metrics still report audio", metrics.isNull("audioMimeType"))
            }
            if (scores) assertScorePixels(output)
        } finally {
            // Cancel only this UUID job; never clear app storage or stop unrelated work.
            if (terminal.count > 0) {
                context.startService(Intent(context, ExportService::class.java).setAction(ExportService.ACTION_CANCEL)
                    .putExtra(ExportService.EXTRA_JOB_ID, id))
                terminal.await(5, TimeUnit.SECONDS)
                instrumentation.waitForIdleSync()
            }
            context.unregisterReceiver(receiver)
            scenario.close()
            if (retainArtifacts) {
                instrumentation.sendStatus(2, Bundle().apply { putString("exportTracksArtifactDirectory", directory.absolutePath) })
            } else directory.deleteRecursively()
        }
    }

    private data class Track(val mime: String, val durationUs: Long, val sampleRate: Int,
        val channels: Int, val sampleCount: Int, val sampleBytes: Long)

    private fun tracks(file: File): List<Track> {
        val metadata = MediaExtractor()
        val formats = try {
            metadata.setDataSource(file.absolutePath)
            (0 until metadata.trackCount).map(metadata::getTrackFormat)
        } finally {
            metadata.release()
        }
        return formats.mapIndexed { index, format ->
            // Seeking a shared extractor after video EOS did not reliably restore audio on device.
            // Each independent extractor starts fresh and reads only this track's actual payload.
            val extractor = MediaExtractor()
            try {
                extractor.setDataSource(file.absolutePath)
                extractor.selectTrack(index)
                val buffer = ByteBuffer.allocate(maxOf(65_536, format.getInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 0)))
                var samples = 0
                var bytes = 0L
                while (extractor.sampleTrackIndex >= 0) {
                    assertEquals("Read a sample from another track", index, extractor.sampleTrackIndex)
                    buffer.clear()
                    val size = extractor.readSampleData(buffer, 0)
                    assertTrue("Track $index has an unreadable sample", size > 0)
                    samples++
                    bytes += size
                    if (!extractor.advance()) break
                }
                Track(checkNotNull(format.getString(MediaFormat.KEY_MIME)), format.getLong(MediaFormat.KEY_DURATION),
                    format.getInteger(MediaFormat.KEY_SAMPLE_RATE, 0), format.getInteger(MediaFormat.KEY_CHANNEL_COUNT, 0), samples, bytes)
            } finally {
                extractor.release()
            }
        }
    }

    private fun assertScorePixels(output: File) {
        val retriever = MediaMetadataRetriever()
        try {
            retriever.setDataSource(output.absolutePath)
            val frame = checkNotNull(retriever.getFrameAtTime(200_000, MediaMetadataRetriever.OPTION_CLOSEST))
            try {
                for ((x, expected) in listOf(10 to ScoreOverlay.TEAM_1_COLOR.toInt(),
                    90 to ScoreOverlay.SCORE_BACKGROUND_COLOR.toInt(), 138 to ScoreOverlay.TEAM_2_COLOR.toInt())) {
                    val actual = frame.getPixel(x, 5)
                    val distance = maxOf(kotlin.math.abs(Color.red(expected) - Color.red(actual)),
                        kotlin.math.abs(Color.green(expected) - Color.green(actual)),
                        kotlin.math.abs(Color.blue(expected) - Color.blue(actual)))
                    assertTrue("Score overlay pixel missing at $x", distance <= 48)
                }
            } finally {
                frame.recycle()
            }
        } finally {
            retriever.release()
        }
    }
}
