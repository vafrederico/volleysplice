package com.volleycut.nativeanalysis

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.net.Uri
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import androidx.annotation.OptIn
import androidx.core.app.NotificationCompat
import androidx.media3.common.MediaItem
import androidx.media3.common.MimeTypes
import androidx.media3.common.util.UnstableApi
import androidx.media3.transformer.Composition
import androidx.media3.transformer.EditedMediaItem
import androidx.media3.transformer.EditedMediaItemSequence
import androidx.media3.transformer.ExportException
import androidx.media3.transformer.ExportResult
import androidx.media3.transformer.ProgressHolder
import androidx.media3.transformer.Transformer
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.Locale

@OptIn(markerClass = [UnstableApi::class])
class ExportService : Service() {
    private val handler = Handler(Looper.getMainLooper())
    private var transformer: Transformer? = null
    private var startedAtMs = 0L
    private var outputDurationMs = 0L
    private var intervalCount = 0
    private var temporaryFile: File? = null
    private var destination: Uri? = null
    private var sourceName = "video"
    private var outputWriteMode = "uninitialized"
    private var wakeLock: PowerManager.WakeLock? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        wakeLock = getSystemService(PowerManager::class.java)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "$packageName:video-export")
            .apply { setReferenceCounted(false) }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_CANCEL) {
            transformer?.cancel()
            finishExport("cancelled", "Export cancelled", null, null)
            return START_NOT_STICKY
        }
        if (transformer != null || intent == null) return START_NOT_STICKY
        val source = intent.getStringExtra(EXTRA_SOURCE_URI)?.let(Uri::parse)
        destination = intent.getStringExtra(EXTRA_DESTINATION_URI)?.let(Uri::parse)
        sourceName = intent.getStringExtra(EXTRA_SOURCE_NAME) ?: "video"
        val starts = intent.getLongArrayExtra(EXTRA_INTERVAL_STARTS)
        val ends = intent.getLongArrayExtra(EXTRA_INTERVAL_ENDS)
        val sourceDurationMs = intent.getLongExtra(EXTRA_SOURCE_DURATION_MS, 0)
        if (source == null || destination == null || starts == null || ends == null ||
            starts.isEmpty() || starts.size != ends.size || sourceDurationMs <= 0
        ) {
            stopSelf()
            return START_NOT_STICKY
        }

        val intervals = starts.indices.map { FinalCutInterval(starts[it], ends[it], emptyList()) }
        outputDurationMs = EditorMath.totalFinalMs(intervals)
        intervalCount = intervals.size
        startedAtMs = SystemClock.elapsedRealtime()
        startForegroundCompat(notification(0, "Preparing export", true))
        if (wakeLock?.isHeld != true) wakeLock?.acquire(WAKE_LOCK_TIMEOUT_MS)
        broadcast("running", 0, "Preparing hardware encoder", null)
        runCatching { startTransformer(source, sourceDurationMs, intervals) }
            .onFailure { error ->
                Log.e(TAG, "Could not start native export", error)
                finishExport("failed", error.message ?: "Could not start export", null, error)
            }
        return START_NOT_STICKY
    }

    private fun startTransformer(source: Uri, sourceDurationMs: Long, intervals: List<FinalCutInterval>) {
        val editedItems = intervals.map { interval ->
            val mediaItem = MediaItem.Builder()
                .setUri(source)
                .setClippingConfiguration(
                    MediaItem.ClippingConfiguration.Builder()
                        .setStartPositionMs(interval.startMs)
                        .setEndPositionMs(interval.endMs)
                        .build(),
                )
                .build()
            EditedMediaItem.Builder(mediaItem)
                .setDurationUs(sourceDurationMs * 1_000)
                .build()
        }
        val sequence = EditedMediaItemSequence.withAudioAndVideoFrom(editedItems)
        val composition = Composition.Builder(sequence).build()
        val target = checkNotNull(destination) { "Destination missing" }
        val directMuxerFactory = DirectUriMp4MuxerFactory(this, target)
        val directOutput = directMuxerFactory.isSeekable()
        val transformerBuilder = Transformer.Builder(this)
            .setVideoMimeType(MimeTypes.VIDEO_H264)
            .setAudioMimeType(MimeTypes.AUDIO_AAC)
        val transformerOutputPath = if (directOutput) {
            outputWriteMode = "direct_document"
            transformerBuilder.setMuxerFactory(directMuxerFactory)
            updateNotification(0, "Encoding directly to selected file")
            broadcast("running", 0, "Encoding directly to selected file", null)
            target.toString()
        } else {
            outputWriteMode = "temporary_then_copy"
            temporaryFile = File.createTempFile("volleycut-export-", ".mp4", externalCacheDir ?: cacheDir)
                .also { file -> check(file.delete()) { "Could not prepare temporary export path" } }
            updateNotification(0, "Destination is not seekable; using cache fallback")
            broadcast("running", 0, "Destination is not seekable; using cache fallback", null)
            temporaryFile!!.absolutePath
        }
        transformer = transformerBuilder
            .addListener(object : Transformer.Listener {
                override fun onCompleted(composition: Composition, exportResult: ExportResult) {
                    completeOutput(exportResult)
                }

                override fun onError(
                    composition: Composition,
                    exportResult: ExportResult,
                    exportException: ExportException,
                ) {
                    Log.e(TAG, "Native export failed", exportException)
                    finishExport("failed", exportException.message ?: "Export failed", exportResult, exportException)
                }
            })
            .build()
        transformer!!.start(composition, transformerOutputPath)
        handler.post(progressPoll)
    }

    private val progressPoll = object : Runnable {
        override fun run() {
            val active = transformer ?: return
            val holder = ProgressHolder()
            val state = active.getProgress(holder)
            if (state != Transformer.PROGRESS_STATE_NOT_STARTED) {
                val progress = if (state == Transformer.PROGRESS_STATE_AVAILABLE) holder.progress else -1
                val elapsedMs = SystemClock.elapsedRealtime() - startedAtMs
                val processedMs = if (progress > 0) outputDurationMs * progress / 100 else 0
                val realtime = if (elapsedMs > 0) processedMs.toDouble() / elapsedMs else 0.0
                val detail = if (progress >= 0) {
                    String.format(Locale.US, "%d%% · %.2fx real time", progress, realtime)
                } else "Encoding · ${compactTime(elapsedMs)}"
                updateNotification(progress.coerceAtLeast(0), detail)
                broadcast("running", progress, detail, null)
            }
            handler.postDelayed(this, 500)
        }
    }

    private fun completeOutput(result: ExportResult) {
        if (outputWriteMode == "direct_document") {
            val target = destination
                ?: return finishExport("failed", "Destination missing", result, null)
            finishExport("complete", "Saved ${target.lastPathSegment ?: sourceName}", result, null)
            return
        }
        publishTemporaryOutput(result)
    }

    private fun publishTemporaryOutput(result: ExportResult) {
        handler.removeCallbacks(progressPoll)
        updateNotification(100, "Finalizing file")
        broadcast("running", 100, "Finalizing file", null)
        val file = temporaryFile ?: return finishExport("failed", "Temporary export missing", result, null)
        val target = destination ?: return finishExport("failed", "Destination missing", result, null)
        runCatching {
            contentResolver.openOutputStream(target, "w")!!.use { output ->
                file.inputStream().use { input -> input.copyTo(output, 4 * 1_024 * 1_024) }
            }
        }.onSuccess {
            finishExport("complete", "Saved ${target.lastPathSegment ?: sourceName}", result, null)
        }.onFailure { error ->
            Log.e(TAG, "Could not publish export", error)
            finishExport("failed", error.message ?: "Could not save export", result, error)
        }
    }

    private fun finishExport(
        status: String,
        detail: String,
        result: ExportResult?,
        error: Throwable?,
    ) {
        handler.removeCallbacks(progressPoll)
        val metrics = resultJson(status, detail, result, error)
        Log.i(TAG, metrics.toString())
        broadcast(status, if (status == "complete") 100 else 0, detail, metrics.toString())
        transformer = null
        temporaryFile?.delete()
        temporaryFile = null
        if (status != "complete") deleteIncompleteDestination()
        val manager = getSystemService(NotificationManager::class.java)
        if (status == "complete") manager.notify(NOTIFICATION_ID, notification(100, detail, false))
        else if (status == "failed") manager.notify(NOTIFICATION_ID, notification(0, detail, false))
        releaseWakeLock()
        stopForeground(STOP_FOREGROUND_DETACH)
        stopSelf()
    }

    private fun resultJson(
        status: String,
        detail: String,
        result: ExportResult?,
        error: Throwable?,
    ) = JSONObject().apply {
        val elapsedMs = (SystemClock.elapsedRealtime() - startedAtMs).coerceAtLeast(0)
        put("schemaVersion", 1)
        put("method", "android-media3-transformer-composition-v1")
        put("status", status)
        put("detail", detail)
        put("sourceName", sourceName)
        put("intervalCount", intervalCount)
        put("outputWriteMode", outputWriteMode)
        put("outputDurationMs", outputDurationMs)
        put("elapsedMilliseconds", elapsedMs)
        put("realtimeRatio", if (elapsedMs > 0) outputDurationMs.toDouble() / elapsedMs else 0)
        if (result != null) {
            put("videoEncoder", result.videoEncoderName ?: JSONObject.NULL)
            put("audioEncoder", result.audioEncoderName ?: JSONObject.NULL)
            put("videoMimeType", result.videoMimeType ?: JSONObject.NULL)
            put("audioMimeType", result.audioMimeType ?: JSONObject.NULL)
            put("videoConversionProcess", result.videoConversionProcess)
            put("audioConversionProcess", result.audioConversionProcess)
            put("videoFrameCount", result.videoFrameCount)
            put("videoFramesPerSecond", if (elapsedMs > 0) result.videoFrameCount / (elapsedMs / 1_000.0) else 0)
            put("width", result.width)
            put("height", result.height)
            put("averageVideoBitrate", result.averageVideoBitrate)
            put("averageAudioBitrate", result.averageAudioBitrate)
            put("fileSizeBytes", result.fileSizeBytes)
            put("optimizationResult", result.optimizationResult)
        }
        if (error != null) {
            put("errorType", error.javaClass.name)
            put("errorMessage", error.message ?: "")
        }
    }

    private fun deleteIncompleteDestination() {
        val target = destination ?: return
        val deleted = runCatching { contentResolver.delete(target, null, null) > 0 }
            .getOrDefault(false)
        if (deleted) return
        val truncated = listOf("rwt", "wt", "w").any { mode ->
            runCatching {
                contentResolver.openFileDescriptor(target, mode)?.use { descriptor ->
                    android.system.Os.ftruncate(descriptor.fileDescriptor, 0)
                } ?: error("Document provider returned no file descriptor")
            }.isSuccess
        }
        if (!truncated) Log.w(TAG, "Could not remove or truncate incomplete export")
    }

    private fun broadcast(status: String, progress: Int, detail: String, metrics: String?) {
        sendBroadcast(Intent(ACTION_PROGRESS).setPackage(packageName).apply {
            putExtra(EXTRA_STATUS, status)
            putExtra(EXTRA_PROGRESS, progress)
            putExtra(EXTRA_DETAIL, detail)
            if (metrics != null) putExtra(EXTRA_METRICS, metrics)
        })
    }

    private fun createNotificationChannel() {
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "Video exports", NotificationManager.IMPORTANCE_LOW),
        )
    }

    private fun notification(progress: Int, detail: String, ongoing: Boolean): Notification {
        val cancelIntent = Intent(this, ExportService::class.java).setAction(ACTION_CANCEL)
        val cancel = PendingIntent.getService(
            this, 1, cancelIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val builder = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_upload)
            .setContentTitle("VolleyCut export")
            .setContentText(detail)
            .setOnlyAlertOnce(true)
            .setOngoing(ongoing)
        if (ongoing) builder.setProgress(100, progress, progress <= 0).addAction(0, "Cancel", cancel)
        return builder.build()
    }

    private fun updateNotification(progress: Int, detail: String) {
        getSystemService(NotificationManager::class.java)
            .notify(NOTIFICATION_ID, notification(progress, detail, true))
    }

    private fun startForegroundCompat(notification: Notification) {
        if (Build.VERSION.SDK_INT >= 35) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING,
            )
        } else startForeground(NOTIFICATION_ID, notification)
    }

    override fun onDestroy() {
        val abandonedExport = transformer != null
        handler.removeCallbacksAndMessages(null)
        transformer?.cancel()
        temporaryFile?.delete()
        transformer = null
        if (abandonedExport) deleteIncompleteDestination()
        releaseWakeLock()
        super.onDestroy()
    }

    private fun releaseWakeLock() {
        if (wakeLock?.isHeld == true) wakeLock?.release()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    companion object {
        private const val TAG = "VolleyCutExport"
        private const val CHANNEL_ID = "volleycut_exports"
        private const val NOTIFICATION_ID = 401
        private const val WAKE_LOCK_TIMEOUT_MS = 12 * 60 * 60 * 1_000L
        const val ACTION_PROGRESS = "com.volleycut.nativeanalysis.EXPORT_PROGRESS"
        const val ACTION_CANCEL = "com.volleycut.nativeanalysis.CANCEL_EXPORT"
        const val EXTRA_SOURCE_URI = "export_source_uri"
        const val EXTRA_SOURCE_NAME = "export_source_name"
        const val EXTRA_SOURCE_DURATION_MS = "export_source_duration_ms"
        const val EXTRA_DESTINATION_URI = "export_destination_uri"
        const val EXTRA_INTERVAL_STARTS = "export_interval_starts"
        const val EXTRA_INTERVAL_ENDS = "export_interval_ends"
        const val EXTRA_STATUS = "export_status"
        const val EXTRA_PROGRESS = "export_progress"
        const val EXTRA_DETAIL = "export_detail"
        const val EXTRA_METRICS = "export_metrics"
    }
}
