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
import java.util.ArrayDeque
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

internal data class ExportJobStatus(
    val jobId: String,
    val projectId: String,
    val status: String,
    val progress: Int,
    val detail: String,
    val metrics: String? = null,
)

private data class ExportJob(
    val id: String,
    val projectId: String,
    val source: Uri,
    val sourceName: String,
    val sourceDurationMs: Long,
    val destination: Uri,
    val intervals: List<FinalCutInterval>,
)

@OptIn(markerClass = [UnstableApi::class])
class ExportService : Service() {
    private val handler = Handler(Looper.getMainLooper())
    private val queue = ArrayDeque<ExportJob>()
    private var currentJob: ExportJob? = null
    private var transformer: Transformer? = null
    private var startedAtMs = 0L
    private var outputDurationMs = 0L
    private var intervalCount = 0
    private var lastProgress = 0
    private var temporaryFile: File? = null
    private var destination: Uri? = null
    private var sourceName = "video"
    private var outputWriteMode = "uninitialized"
    private var wakeLock: PowerManager.WakeLock? = null
    private var foreground = false

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        wakeLock = getSystemService(PowerManager::class.java)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "$packageName:video-export")
            .apply { setReferenceCounted(false) }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_CANCEL) {
            cancelExport(intent.getStringExtra(EXTRA_JOB_ID))
            return START_NOT_STICKY
        }
        val job = intent?.let(::exportJob) ?: return START_NOT_STICKY
        if (currentJob?.id == job.id || queue.any { it.id == job.id }) return START_NOT_STICKY
        queue.addLast(job)
        publishQueuedStatuses()
        if (currentJob == null) startNextExport()
        else updateNotification(currentProgress(), activeNotificationDetail())
        return START_NOT_STICKY
    }

    private fun exportJob(intent: Intent): ExportJob? {
        val id = intent.getStringExtra(EXTRA_JOB_ID)?.takeIf(String::isNotBlank) ?: return null
        val projectId = intent.getStringExtra(EXTRA_PROJECT_ID)?.takeIf(String::isNotBlank) ?: return null
        val source = intent.getStringExtra(EXTRA_SOURCE_URI)?.let(Uri::parse)
        val target = intent.getStringExtra(EXTRA_DESTINATION_URI)?.let(Uri::parse)
        val sourceName = intent.getStringExtra(EXTRA_SOURCE_NAME) ?: "video"
        val starts = intent.getLongArrayExtra(EXTRA_INTERVAL_STARTS)
        val ends = intent.getLongArrayExtra(EXTRA_INTERVAL_ENDS)
        val sourceDurationMs = intent.getLongExtra(EXTRA_SOURCE_DURATION_MS, 0)
        if (source == null || target == null || starts == null || ends == null ||
            starts.isEmpty() || starts.size != ends.size || sourceDurationMs <= 0
        ) return null
        val intervals = starts.indices.map { FinalCutInterval(starts[it], ends[it], emptyList()) }
        if (intervals.any { it.startMs < 0 || it.endMs <= it.startMs || it.endMs > sourceDurationMs }) return null
        return ExportJob(id, projectId, source, sourceName, sourceDurationMs, target, intervals)
    }

    private fun startNextExport() {
        val job = if (queue.isEmpty()) null else queue.removeFirst()
        if (job == null) {
            pendingJobs.set(0)
            releaseWakeLock()
            if (foreground) {
                stopForeground(STOP_FOREGROUND_DETACH)
                foreground = false
            }
            stopSelf()
            return
        }
        currentJob = job
        destination = job.destination
        sourceName = job.sourceName
        outputDurationMs = EditorMath.totalFinalMs(job.intervals)
        intervalCount = job.intervals.size
        lastProgress = 0
        outputWriteMode = "uninitialized"
        startedAtMs = SystemClock.elapsedRealtime()
        if (!foreground) {
            startForegroundCompat(notification(0, "Preparing ${job.sourceName}", true))
            foreground = true
        } else updateNotification(0, "Preparing ${job.sourceName}")
        if (wakeLock?.isHeld != true) wakeLock?.acquire(WAKE_LOCK_TIMEOUT_MS)
        broadcast("running", 0, "Preparing hardware encoder", null)
        publishQueuedStatuses()
        runCatching { startTransformer(job.source, job.sourceDurationMs, job.intervals) }
            .onFailure { error ->
                Log.e(TAG, "Could not start native export", error)
                finishExport("failed", error.message ?: "Could not start export", null, error)
            }
    }

    private fun startTransformer(source: Uri, sourceDurationMs: Long, intervals: List<FinalCutInterval>) {
        val jobId = currentJob?.id ?: error("Export job missing")
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
                    completeOutput(jobId, exportResult)
                }

                override fun onError(
                    composition: Composition,
                    exportResult: ExportResult,
                    exportException: ExportException,
                ) {
                    if (currentJob?.id != jobId) return
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
                if (progress >= 0) lastProgress = progress
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

    private fun completeOutput(jobId: String, result: ExportResult) {
        if (currentJob?.id != jobId) return
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
        val finishedJob = currentJob ?: return
        handler.removeCallbacks(progressPoll)
        val metrics = resultJson(status, detail, result, error)
        Log.i(TAG, metrics.toString())
        transformer = null
        temporaryFile?.delete()
        temporaryFile = null
        if (status != "complete") deleteIncompleteDestination()
        currentJob = null
        broadcast(finishedJob, status, if (status == "complete") 100 else 0, detail, metrics.toString())
        val manager = getSystemService(NotificationManager::class.java)
        if (status == "complete") manager.notify(NOTIFICATION_ID, notification(100, detail, false))
        else if (status == "failed") manager.notify(NOTIFICATION_ID, notification(0, detail, false))
        if (queue.isEmpty()) {
            pendingJobs.set(0)
            releaseWakeLock()
            stopForeground(STOP_FOREGROUND_DETACH)
            foreground = false
            stopSelf()
        } else startNextExport()
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

    private fun deleteIncompleteDestination(target: Uri? = destination) {
        target ?: return
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
        val job = currentJob ?: return
        broadcast(job, status, progress, detail, metrics)
    }

    private fun broadcast(
        job: ExportJob,
        status: String,
        progress: Int,
        detail: String,
        metrics: String?,
    ) {
        val pending = (if (currentJob == null) 0 else 1) + queue.size
        pendingJobs.set(pending)
        latestStatuses[job.projectId] = ExportJobStatus(
            job.id, job.projectId, status, progress, detail, metrics,
        )
        sendBroadcast(Intent(ACTION_PROGRESS).setPackage(packageName).apply {
            putExtra(EXTRA_JOB_ID, job.id)
            putExtra(EXTRA_PROJECT_ID, job.projectId)
            putExtra(EXTRA_STATUS, status)
            putExtra(EXTRA_PROGRESS, progress)
            putExtra(EXTRA_DETAIL, detail)
            putExtra(EXTRA_QUEUE_COUNT, pending)
            if (metrics != null) putExtra(EXTRA_METRICS, metrics)
        })
    }

    private fun publishQueuedStatuses() {
        val activeOffset = if (currentJob == null) 0 else 1
        pendingJobs.set(activeOffset + queue.size)
        queue.forEachIndexed { index, job ->
            val ahead = activeOffset + index
            val detail = if (ahead == 0) "Next to encode" else {
                "Queued behind $ahead ${if (ahead == 1) "export" else "exports"}"
            }
            broadcast(job, "queued", 0, detail, null)
        }
    }

    private fun cancelExport(requestedJobId: String?) {
        val active = currentJob
        if (active != null && (requestedJobId == null || requestedJobId == active.id)) {
            transformer?.cancel()
            finishExport("cancelled", "Export cancelled", null, null)
            return
        }
        val queued = queue.firstOrNull { it.id == requestedJobId } ?: return
        queue.remove(queued)
        deleteIncompleteDestination(queued.destination)
        broadcast(queued, "cancelled", 0, "Queued export cancelled", null)
        publishQueuedStatuses()
        currentJob?.let { updateNotification(currentProgress(), activeNotificationDetail()) }
    }

    private fun currentProgress() = lastProgress.coerceIn(0, 100)

    private fun activeNotificationDetail(): String {
        val active = currentJob ?: return "Preparing export"
        return if (queue.isEmpty()) active.sourceName
        else "${active.sourceName} · ${queue.size} queued"
    }

    private fun createNotificationChannel() {
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "Video exports", NotificationManager.IMPORTANCE_LOW),
        )
    }

    private fun notification(progress: Int, detail: String, ongoing: Boolean): Notification {
        val cancelIntent = Intent(this, ExportService::class.java).setAction(ACTION_CANCEL).apply {
            currentJob?.let { putExtra(EXTRA_JOB_ID, it.id) }
        }
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
        val abandoned = buildList {
            currentJob?.let(::add)
            addAll(queue)
        }
        handler.removeCallbacksAndMessages(null)
        transformer?.cancel()
        temporaryFile?.delete()
        transformer = null
        currentJob = null
        queue.clear()
        abandoned.forEach { job ->
            deleteIncompleteDestination(job.destination)
            broadcast(job, "cancelled", 0, "Export service stopped", null)
        }
        pendingJobs.set(0)
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
        private val latestStatuses = ConcurrentHashMap<String, ExportJobStatus>()
        private val pendingJobs = AtomicInteger(0)
        const val ACTION_PROGRESS = "com.volleycut.nativeanalysis.EXPORT_PROGRESS"
        const val ACTION_CANCEL = "com.volleycut.nativeanalysis.CANCEL_EXPORT"
        const val EXTRA_JOB_ID = "export_job_id"
        const val EXTRA_PROJECT_ID = "export_project_id"
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
        const val EXTRA_QUEUE_COUNT = "export_queue_count"

        internal fun statusForProject(projectId: String): ExportJobStatus? = latestStatuses[projectId]

        internal fun pendingCount(): Int = pendingJobs.get()
    }
}
