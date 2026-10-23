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
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.atomic.AtomicInteger

/** Uses the stable service Intent API so this test APK also runs against the pre-fix app. */
@RunWith(AndroidJUnit4::class)
class ExportServiceTracksInstrumentedTest {
    @Test fun videoOnlyCutsWithoutScoresStayVideoOnly() = export("overlay-fixture.mp4.b64", false, false)
    @Test fun videoOnlyCutsWithScoresStayVideoOnly() = export("overlay-fixture.mp4.b64", false, true)
    @Test fun aacCutsWithoutScoresKeepSourceAudio() = export("draft-recovery-fixture.mp4.b64", true, false)
    @Test fun aacCutsWithScoresKeepSourceAudio() = export("draft-recovery-fixture.mp4.b64", true, true)

    @Test fun failedAudioTrackProbeFailsTheJobOffMainThread() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val id = UUID.randomUUID().toString()
        val directory = File(context.cacheDir, "export-probe-$id").apply { check(mkdirs()) }
        val input = File(directory, "source-$id.mp4")
        val output = File(directory, "final-$id.mp4")
        val opens = AtomicInteger()
        val mainThreadRead = AtomicReference(false)
        // The color probe succeeds; the subsequent actual audio-track probe fails.
        val provider = object : ContentProvider() {
            override fun onCreate() = true
            override fun getType(uri: Uri) = "audio/unknown"
            override fun openFile(uri: Uri, mode: String): ParcelFileDescriptor {
                if (Looper.myLooper() == Looper.getMainLooper()) mainThreadRead.set(true)
                if (opens.incrementAndGet() > 1) throw FileNotFoundException("Synthetic track probe failure")
                return ParcelFileDescriptor.open(input, ParcelFileDescriptor.MODE_READ_ONLY)
            }
            override fun query(uri: Uri, projection: Array<out String>?, selection: String?,
                selectionArgs: Array<out String>?, sortOrder: String?) = MatrixCursor(arrayOf("_display_name"))
            override fun insert(uri: Uri, values: ContentValues?): Uri? = error("Not supported")
            override fun delete(uri: Uri, selection: String?, selectionArgs: Array<out String>?) = 0
            override fun update(uri: Uri, values: ContentValues?, selection: String?, selectionArgs: Array<out String>?) = 0
        }
        val scenario = ActivityScenario.launch(ComponentActivity::class.java)
        try {
            input.writeBytes(Base64.decode(instrumentation.context.assets.open("draft-recovery-fixture.mp4.b64")
                .bufferedReader().use { it.readText() }, Base64.DEFAULT))
            scenario.onActivity { it.startService(Intent(it, ExportService::class.java)) }
            instrumentation.waitForIdleSync()
            instrumentation.runOnMainSync {
                val activityThread = Class.forName("android.app.ActivityThread")
                val thread = activityThread.getDeclaredMethod("currentActivityThread").invoke(null)
                val services = activityThread.getDeclaredField("mServices").apply { isAccessible = true }.get(thread) as Map<*, *>
                val running = services.values.filterIsInstance<ExportService>().single()
                val base = running.baseContext
                val resolver = ContentResolver.wrap(provider)
                ContextWrapper::class.java.getDeclaredField("mBase").apply { isAccessible = true }
                    .set(running, object : ContextWrapper(base) {
                        override fun getContentResolver() = resolver
                    })
                running.onStartCommand(Intent(context, ExportService::class.java).apply {
                    putExtra(ExportService.EXTRA_JOB_ID, id)
                    putExtra(ExportService.EXTRA_PROJECT_ID, id)
                    putExtra(ExportService.EXTRA_SOURCE_URI, "content://export-probe-$id/source-$id.mp4")
                    putExtra(ExportService.EXTRA_SOURCE_NAME, input.name)
                    putExtra(ExportService.EXTRA_SOURCE_DURATION_MS, 1_000L)
                    putExtra(ExportService.EXTRA_DESTINATION_URI, Uri.fromFile(output).toString())
                    putExtra(ExportService.EXTRA_INTERVAL_STARTS, longArrayOf(0))
                    putExtra(ExportService.EXTRA_INTERVAL_ENDS, longArrayOf(1_000))
                }, 0, 1)
            }
            val deadline = android.os.SystemClock.elapsedRealtime() + 10_000
            while (ExportService.statusForProject(id)?.status !in setOf("complete", "failed", "cancelled") &&
                android.os.SystemClock.elapsedRealtime() < deadline) Thread.sleep(20)
            assertEquals("Probe failure must not become a successful silent export", "failed", ExportService.statusForProject(id)?.status)
            assertEquals("Both actual source probes must run", 2, opens.get())
            assertFalse("Source track I/O ran on the service main thread", mainThreadRead.get())
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

    private fun export(asset: String, hasAudio: Boolean, scores: Boolean) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val id = UUID.randomUUID().toString()
        val directory = File(context.cacheDir, "export-tracks-$id").apply { check(mkdirs()) }
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
            directory.deleteRecursively()
        }
    }

    private data class Track(val mime: String, val durationUs: Long, val sampleRate: Int,
        val channels: Int, val sampleCount: Int)

    private fun tracks(file: File): List<Track> {
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(file.absolutePath)
            return (0 until extractor.trackCount).map { index ->
                val format = extractor.getTrackFormat(index)
                extractor.selectTrack(index)
                extractor.seekTo(0, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)
                var samples = 0
                while (extractor.sampleTime >= 0) {
                    samples++
                    if (!extractor.advance()) break
                }
                extractor.unselectTrack(index)
                Track(checkNotNull(format.getString(MediaFormat.KEY_MIME)), format.getLong(MediaFormat.KEY_DURATION),
                    format.getInteger(MediaFormat.KEY_SAMPLE_RATE, 0), format.getInteger(MediaFormat.KEY_CHANNEL_COUNT, 0), samples)
            }
        } finally {
            extractor.release()
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
