package com.volleycut.nativeanalysis

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import android.os.SystemClock
import android.provider.OpenableColumns
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
import androidx.media3.transformer.Effects
import androidx.media3.transformer.ProgressHolder
import androidx.media3.transformer.InAppMp4Muxer
import androidx.media3.transformer.TransformationRequest
import androidx.media3.transformer.Transformer
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.ArrayDeque
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
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
    val scoreSnapshot: ScoreExportSnapshot?,
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
    private val sourceWorker = Executors.newSingleThreadExecutor { task -> Thread(task, "VolleySpliceExportSource") }
    private val publicationWorker = Executors.newSingleThreadExecutor { task -> Thread(task, "VolleySpliceExportPublication") }
    private var publication: ExportFilePublication? = null
    @Volatile private var destroyed = false
    @Volatile private var timedOut = false

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        wakeLock = getSystemService(PowerManager::class.java)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "$packageName:video-export")
            .apply { setReferenceCounted(false) }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (timedOut) return START_NOT_STICKY
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
        val scoreSnapshot = intent.getStringExtra(EXTRA_SCORE_SNAPSHOT)?.let {
            ScoreExportSnapshotJson.decode(it, sourceDurationMs)
        }
        return ExportJob(id, projectId, source, sourceName, sourceDurationMs, target, intervals, scoreSnapshot)
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
        if (ExportPublicationDestinations.isInUse(job.destination.toString())) {
            finishExport("failed", "The previous file save is still stopping. Retry this destination after it finishes.",
                null, null, cleanDestinationOnFailure = false)
            return
        }
        runCatching {
            sourceWorker.execute {
                val sourceInfo = runCatching { ExportColorPolicy.read(this, job.source) to sourceHasAudio(job.source) }
                handler.post {
                    // A cancelled/timed-out job must not start encoding or fail its successor.
                    if (destroyed || timedOut || currentJob !== job) return@post
                    sourceInfo.mapCatching { (colorPolicy, hasAudio) ->
                        startTransformer(job.source, job.sourceDurationMs, job.intervals, job.scoreSnapshot, colorPolicy, hasAudio)
                    }.onFailure(::failToStartExport)
                }
            }
        }.onFailure(::failToStartExport)
    }

    private fun failToStartExport(error: Throwable) {
        Log.e(TAG, "Could not start native export", error)
        finishExport("failed", error.message ?: "Could not start export", null, error)
    }

    /** Inspect actual tracks, including audio/unknown; container MIME/analysis metadata cannot prove absence. */
    private fun sourceHasAudio(source: Uri): Boolean {
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(this, source, null)
            check(extractor.trackCount > 0) { "The recording has no readable tracks." }
            var hasAudio = false
            for (index in 0 until extractor.trackCount) {
                val mime = checkNotNull(extractor.getTrackFormat(index).getString(MediaFormat.KEY_MIME)
                    ?.takeIf(String::isNotBlank)) { "Could not identify a recording track." }
                if (mime.startsWith("audio/")) hasAudio = true
            }
            return hasAudio
        } finally {
            extractor.release()
        }
    }

    private fun startTransformer(
        source: Uri,
        sourceDurationMs: Long,
        intervals: List<FinalCutInterval>,
        scoreSnapshot: ScoreExportSnapshot?,
        colorPolicy: ExportColorPolicy,
        hasAudio: Boolean,
    ) {
        val jobId = currentJob?.id ?: error("Export job missing")
        if (scoreSnapshot?.render == true) colorPolicy.requireHdrEncoder()
        var compositionStartUs = 0L
        val editedItems = intervals.map { interval ->
            val intervalCompositionStartUs = compositionStartUs
            compositionStartUs += (interval.endMs - interval.startMs) * 1_000L
            val mediaItem = MediaItem.Builder()
                .setUri(source)
                .setClippingConfiguration(
                    MediaItem.ClippingConfiguration.Builder()
                        .setStartPositionMs(interval.startMs)
                        .setEndPositionMs(interval.endMs)
                        .build(),
                )
                .build()
            EditedMediaItem.Builder(mediaItem).apply {
                setDurationUs(sourceDurationMs * 1_000)
                if (scoreSnapshot?.render == true) {
                    setEffects(Effects(
                        emptyList(),
                        listOf(ScoreOverlayEffect(ScoreCanvasOverlay(
                            scoreSnapshot,
                            interval.startMs,
                            intervalCompositionStartUs,
                        ), colorPolicy.color.colorTransfer)),
                    ))
                }
            }.build()
        }
        // Requesting audio for a video-only source makes Media3 synthesize silence.
        val sequence = if (hasAudio) EditedMediaItemSequence.withAudioAndVideoFrom(editedItems)
            else EditedMediaItemSequence.withVideoFrom(editedItems)
        val composition = Composition.Builder(sequence).setHdrMode(Composition.HDR_MODE_KEEP_HDR).build()
        val target = checkNotNull(destination) { "Destination missing" }
        val directMuxerFactory = DirectUriMp4MuxerFactory(this, target)
        val directOutput = directMuxerFactory.isSeekable()
        val transformerBuilder = Transformer.Builder(this)
            .setVideoMimeType(colorPolicy.outputMimeType)
            .setAudioMimeType(MimeTypes.AUDIO_AAC)
            .setMuxerFactory(colorPolicy.preservingMuxer(InAppMp4Muxer.Factory()))
        val transformerOutputPath = if (directOutput) {
            outputWriteMode = "direct_document"
            transformerBuilder.setMuxerFactory(colorPolicy.preservingMuxer(directMuxerFactory))
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
                override fun onFallbackApplied(
                    composition: Composition,
                    originalTransformationRequest: TransformationRequest,
                    fallbackTransformationRequest: TransformationRequest,
                ) {
                    if (currentJob?.id != jobId || !colorPolicy.isHdr ||
                        fallbackTransformationRequest.hdrMode == Composition.HDR_MODE_KEEP_HDR
                    ) return
                    transformer?.cancel()
                    finishExport(
                        "failed", "This device cannot export this video in its original HDR color space.",
                        null, null,
                    )
                }

                override fun onCompleted(composition: Composition, exportResult: ExportResult) {
                    completeOutput(jobId, exportResult)
                }

                override fun onError(
                    composition: Composition,
                    exportResult: ExportResult,
                    exportException: ExportException,
                ) {
                    if (currentJob?.id != jobId || publication != null) return
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
                if (progress >= 0) lastProgress = if (outputWriteMode == "temporary_then_copy") progress * 90 / 100 else progress
                val elapsedMs = SystemClock.elapsedRealtime() - startedAtMs
                val processedMs = if (progress > 0) outputDurationMs * progress / 100 else 0
                val realtime = if (elapsedMs > 0) processedMs.toDouble() / elapsedMs else 0.0
                val detail = if (progress >= 0) {
                    String.format(Locale.US, "%d%% · %.2fx real time", progress, realtime)
                } else "Encoding · ${compactTime(elapsedMs)}"
                updateNotification(currentProgress(), detail)
                broadcast("running", if (progress >= 0) currentProgress() else -1, detail, null)
            }
            handler.postDelayed(this, 500)
        }
    }

    private fun completeOutput(jobId: String, result: ExportResult) {
        if (currentJob?.id != jobId) return
        if (outputWriteMode == "direct_document") {
            val target = destination
                ?: return finishExport("failed", "Destination missing", result, null)
            val displayName = destinationDisplayName(target)
            finishExport("complete", "Saved $displayName", result, null, displayName)
            return
        }
        publishTemporaryOutput(jobId, result)
    }

    private fun publishTemporaryOutput(jobId: String, result: ExportResult) {
        handler.removeCallbacks(progressPoll)
        transformer = null
        lastProgress = 90
        updateNotification(90, "Saving encoded file")
        broadcast("running", 90, "Saving encoded file", null)
        val file = temporaryFile ?: return finishExport("failed", "Temporary export missing", result, null)
        val target = destination ?: return finishExport("failed", "Destination missing", result, null)
        val fallbackName = sourceName.substringBeforeLast('.', sourceName).ifBlank { "VolleySplice" } + "-final.mp4"
        val owner = Any()
        val targetKey = target.toString()
        if (!ExportPublicationDestinations.claim(targetKey, owner)) {
            finishExport("failed", "This destination is still being saved by another export. Retry after it finishes.",
                result, null, cleanDestinationOnFailure = false)
            return
        }
        lateinit var task: ExportFilePublication
        task = ExportFilePublication(
            worker = publicationWorker,
            dispatch = { callback -> if (!destroyed) handler.post(callback) },
            openInput = { file.inputStream() },
            openOutput = { checkNotNull(contentResolver.openOutputStream(target, "w")) { "Document provider returned no output stream" } },
            totalBytes = file.length(),
            displayName = { destinationDisplayName(target, fallbackName) },
            removeTemporary = { file.delete(); Unit },
            removeIncomplete = { deleteIncompleteDestination(target) },
            isCurrent = { !destroyed && !timedOut && currentJob?.id == jobId && publication === task },
            onProgress = { bytes, total ->
                lastProgress = 90 + (bytes.coerceAtMost(total) * 9 / total).toInt()
                val detail = "Saving file · ${bytes / 1_024 / 1_024} / ${total / 1_024 / 1_024} MiB"
                updateNotification(lastProgress, detail)
                broadcast("running", lastProgress, detail, null)
            },
            onSuccess = { name ->
                ExportPublicationDestinations.release(targetKey, owner)
                finishExport("complete", "Saved $name", result, null, name)
            },
            onFailure = { error, cancelled ->
                if (!cancelled) Log.e(TAG, "Could not publish export", error)
                finishExport(if (cancelled) "cancelled" else "failed",
                    if (cancelled) "Export cancelled" else error.message ?: "Could not save export",
                    result, if (cancelled) null else error, cleanDestinationOnFailure = false)
            },
            onSettled = { ExportPublicationDestinations.release(targetKey, owner) },
        )
        publication = task
        runCatching { task.start() }.onFailure { error ->
            ExportPublicationDestinations.release(targetKey, owner)
            finishExport("failed", error.message ?: "Could not start file save", result, error, cleanDestinationOnFailure = false)
        }
    }

    private fun finishExport(
        status: String,
        detail: String,
        result: ExportResult?,
        error: Throwable?,
        savedDisplayName: String? = null,
        cleanDestinationOnFailure: Boolean = true,
    ) {
        val finishedJob = currentJob ?: return
        handler.removeCallbacks(progressPoll)
        val metrics = resultJson(status, detail, result, error)
        Log.i(TAG, metrics.toString())
        transformer = null
        publication = null
        temporaryFile?.delete()
        temporaryFile = null
        if (status != "complete" && cleanDestinationOnFailure) deleteIncompleteDestination()
        currentJob = null
        if (status == "complete") {
            NativeProjectStore.recordSuccessfulExport(
                this,
                finishedJob.projectId,
                savedDisplayName ?: "video",
            )
            AppRating.recordSuccessfulExport(this, finishedJob.id)
        }
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

    private fun destinationDisplayName(target: Uri, fallback: String? = null): String {
        val queriedName = runCatching {
            contentResolver.query(
                target,
                arrayOf(OpenableColumns.DISPLAY_NAME),
                null,
                null,
                null,
            )?.use { cursor ->
                if (!cursor.moveToFirst()) null else cursor.getString(0)
            }
        }.getOrNull()?.takeIf(String::isNotBlank)
        return queriedName ?: fallback ?: sourceName.substringBeforeLast('.', sourceName)
            .ifBlank { "VolleySplice" }
            .let { "$it-final.mp4" }
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
            publication?.let {
                it.cancel()
                updateNotification(currentProgress(), "Cancelling file save")
                broadcast("running", currentProgress(), "Cancelling file save", null)
                return
            }
            transformer?.cancel()
            finishExport("cancelled", "Export cancelled", null, null)
            return
        }
        val queued = queue.firstOrNull { it.id == requestedJobId } ?: return
        queue.remove(queued)
        // Queued jobs have never opened their destination. It may already
        // contain a user's file or belong to a still-running publication.
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
            .setContentTitle("VolleySplice export")
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
        destroyed = true
        sourceWorker.shutdownNow()
        val publishing = publication != null
        publication?.cancel()
        publicationWorker.shutdown()
        val abandoned = buildList {
            currentJob?.let(::add)
            addAll(queue)
        }
        val activeId = currentJob?.id
        handler.removeCallbacksAndMessages(null)
        transformer?.cancel()
        if (!publishing) temporaryFile?.delete()
        transformer = null
        currentJob = null
        queue.clear()
        abandoned.forEach { job ->
            // A queued job has not opened its destination and must not delete
            // a pre-existing document selected by the user.
            if (job.id == activeId && !publishing) deleteIncompleteDestination(job.destination)
            broadcast(job, "cancelled", 0, "Export service stopped", null)
        }
        pendingJobs.set(0)
        releaseWakeLock()
        super.onDestroy()
    }

    private fun releaseWakeLock() {
        if (wakeLock?.isHeld == true) wakeLock?.release()
    }

    override fun onTimeout(startId: Int, fgsType: Int) {
        timedOut = true
        val publishing = publication != null
        publication?.cancel()
        val affectedJobs = buildList {
            currentJob?.let(::add)
            addAll(queue)
        }
        val active = currentJob
        val detail = "Android stopped video export after its background time limit. Queue the export again when the app is ready."
        handler.removeCallbacks(progressPoll)
        transformer?.cancel()
        if (!publishing) temporaryFile?.delete()
        temporaryFile = null
        transformer = null
        currentJob = null
        destination = null
        queue.clear()
        affectedJobs.forEach { job ->
            // Queued jobs have not opened their destination yet; only remove
            // the active job's partially written document.
            if (job.id == active?.id && !publishing) deleteIncompleteDestination(job.destination)
            broadcast(job, "failed", 0, detail, null)
        }
        pendingJobs.set(0)
        ProcessingTimeoutTracker.record(
            context = this,
            operation = MediaProcessingOperation.EXPORT,
            projectId = active?.projectId ?: affectedJobs.firstOrNull()?.projectId,
            sourceName = active?.sourceName ?: affectedJobs.firstOrNull()?.sourceName,
            detail = detail,
        )
        if (foreground) {
            stopForeground(STOP_FOREGROUND_REMOVE)
            foreground = false
        }
        releaseWakeLock()
        stopSelf()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    companion object {
        private const val TAG = "VolleySpliceExport"
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
        const val EXTRA_SCORE_SNAPSHOT = "export_score_snapshot"
        const val EXTRA_STATUS = "export_status"
        const val EXTRA_PROGRESS = "export_progress"
        const val EXTRA_DETAIL = "export_detail"
        const val EXTRA_METRICS = "export_metrics"
        const val EXTRA_QUEUE_COUNT = "export_queue_count"

        internal fun statusForProject(projectId: String): ExportJobStatus? = latestStatuses[projectId]

        internal fun pendingCount(): Int = pendingJobs.get()
    }
}
