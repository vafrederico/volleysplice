package com.volleycut.nativeanalysis

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.net.Uri
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import org.json.JSONArray
import org.json.JSONObject
import java.util.ArrayDeque
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.ceil

internal fun servingSideOverallProgress(stage: String, fraction: Double): Double {
    val bounded = fraction.coerceIn(0.0, 1.0)
    return when (stage) {
        "serving-side-frames" -> 0.02 + bounded * 0.80
        "serving-side-features" -> 0.82 + bounded * 0.17
        "serving-side" -> if (bounded >= 1.0) 1.0 else 0.01
        else -> bounded
    }
}

internal fun projectCreationOverallProgress(
    stage: String,
    fraction: Double,
    includeServingSide: Boolean,
): Double {
    val bounded = fraction.coerceIn(0.0, 1.0)
    if (stage == "complete") return 1.0
    if (includeServingSide) {
        return when (stage) {
            "opening" -> 0.02 * bounded
            "video" -> 0.02 + 0.40 * bounded
            "audio" -> 0.42 + 0.16 * bounded
            "normalizing" -> 0.58 + 0.06 * bounded
            "inference" -> 0.64 + 0.06 * bounded
            "serving-side", "serving-side-frames", "serving-side-features" ->
                0.70 + 0.30 * servingSideOverallProgress(stage, bounded)
            else -> bounded
        }
    }
    return when (stage) {
        "opening" -> 0.02 * bounded
        "video" -> 0.02 + 0.63 * bounded
        "audio" -> 0.65 + 0.19 * bounded
        "normalizing" -> 0.84 + 0.07 * bounded
        "inference" -> 0.91 + 0.08 * bounded
        "serving-side" -> 0.99
        else -> bounded
    }
}

internal data class ProjectNotificationStage(
    val label: String,
    val step: Int,
    val stepCount: Int,
    val progress: Double,
) {
    val title: String get() = "$label ($step/$stepCount)"
    val progressPercent: Int get() = (progress.coerceIn(0.0, 1.0) * 100).toInt()
}

internal class NotificationStageSpeedTracker(
    private val nanoTime: () -> Long = System::nanoTime,
) {
    private var stageKey: String? = null
    private var startedNanos = 0L
    private var startedProgress = 0.0

    fun detail(filename: String, key: String, progress: Double): String {
        val bounded = progress.coerceIn(0.0, 1.0)
        val now = nanoTime()
        if (stageKey != key || bounded + 0.001 < startedProgress) {
            stageKey = key
            startedNanos = now
            startedProgress = bounded
        }
        val percent = (bounded * 100).toInt()
        if (bounded >= 1.0) return "$filename · 100% · complete"
        val elapsedSeconds = (now - startedNanos) / 1_000_000_000.0
        val advanced = bounded - startedProgress
        if (elapsedSeconds < 0.5 || advanced <= 0.0) {
            return "$filename · $percent% · measuring speed"
        }
        val fractionPerSecond = advanced / elapsedSeconds
        val etaSeconds = (1.0 - bounded) / fractionPerSecond
        return String.format(
            Locale.US,
            "%s · %d%% · %.1f%%/s · ETA %s",
            filename,
            percent,
            fractionPerSecond * 100.0,
            formatNotificationDuration(etaSeconds),
        )
    }
}

internal fun projectCreationNotificationStage(
    stage: String,
    fraction: Double,
    includeServingSide: Boolean,
): ProjectNotificationStage {
    val bounded = fraction.coerceIn(0.0, 1.0)
    val stepCount = if (includeServingSide) 3 else 2
    return when (stage) {
        "opening", "video" -> ProjectNotificationStage(
            "Video analysis", 1, stepCount, if (stage == "video") bounded else 0.0,
        )
        "audio" -> ProjectNotificationStage("Audio analysis", 2, stepCount, bounded * 0.80)
        "normalizing" -> ProjectNotificationStage("Audio analysis", 2, stepCount, 0.80 + bounded * 0.10)
        "inference" -> ProjectNotificationStage("Audio analysis", 2, stepCount, 0.90 + bounded * 0.10)
        "serving-side", "serving-side-frames", "serving-side-features" -> {
            if (includeServingSide) {
                ProjectNotificationStage(
                    "Serving-side analysis", 3, stepCount,
                    servingSideOverallProgress(stage, bounded),
                )
            } else {
                ProjectNotificationStage("Audio analysis", 2, stepCount, 1.0)
            }
        }
        "complete" -> ProjectNotificationStage("Project ready", stepCount, stepCount, 1.0)
        else -> ProjectNotificationStage("Video analysis", 1, stepCount, bounded)
    }
}

internal fun projectInferenceNotificationDetail(
    filename: String,
    stats: AnalysisTypes.PerformanceStats?,
    progressPercent: Int? = null,
): String {
    val prefix = if (progressPercent == null) filename else "$filename · ${progressPercent.coerceIn(0, 100)}%"
    if (stats == null || stats.framesPerSecond() <= 0.0) return prefix
    val eta = formatNotificationDuration(stats.etaSeconds())
    return String.format(
        Locale.US,
        "%s · %.1f fps · %.2fx realtime · ETA %s",
        prefix,
        stats.framesPerSecond(),
        stats.realtimeRatio(),
        eta,
    )
}

private fun formatNotificationDuration(etaSeconds: Double): String {
    if (!etaSeconds.isFinite() || etaSeconds < 0.0) return "calculating"
    val totalSeconds = ceil(etaSeconds).toLong().coerceAtLeast(0L)
    val hours = totalSeconds / 3_600
    val minutes = totalSeconds % 3_600 / 60
    val seconds = totalSeconds % 60
    return when {
        hours > 0 -> "${hours}h ${minutes}m"
        minutes > 0 -> "${minutes}m ${seconds}s"
        else -> "${seconds}s"
    }
}

/** Serial foreground queue so decoding continues while a different project is edited. */
class ProjectAnalysisService : Service() {
    private data class AnalysisTask(val projectId: String, val servingSideOnly: Boolean)

    private val executor = Executors.newSingleThreadExecutor()
    private val queue = ArrayDeque<AnalysisTask>()
    private val queuedIds = mutableSetOf<String>()
    private val deletingProjects = mutableMapOf<String, NativeProject>()
    private var activeProjectId: String? = null
    private var activeServingSideOnly = false
    private var activeCancellation: AtomicBoolean? = null
    private var workerRunning = false
    private var foreground = false
    @Volatile private var timedOut = false
    private var wakeLock: PowerManager.WakeLock? = null

    override fun onCreate() {
        super.onCreate()
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "Project inference", NotificationManager.IMPORTANCE_LOW),
        )
        wakeLock = getSystemService(PowerManager::class.java)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "$packageName:project-inference")
            .apply { setReferenceCounted(false) }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (timedOut) return START_NOT_STICKY
        when (intent?.action) {
            ACTION_DELETE -> intent.getStringExtra(EXTRA_PROJECT_ID)?.let(::deleteProject)
            ACTION_ENQUEUE -> intent.getStringExtra(EXTRA_PROJECT_ID)?.let(::enqueueProject)
            ACTION_ENQUEUE_SERVING_SIDE -> intent.getStringExtra(EXTRA_PROJECT_ID)
                ?.let(::enqueueServingSideProject)
        }
        return START_NOT_STICKY
    }

    @Synchronized
    private fun enqueueProject(projectId: String) {
        val project = NativeProjectStore.get(this, projectId) ?: return
        if (project.status == ProjectStatus.READY || projectId == activeProjectId || projectId in queuedIds) return
        NativeProjectStore.updateStatus(this, projectId, ProjectStatus.QUEUED)
        queue.addLast(AnalysisTask(projectId, false))
        queuedIds += projectId
        liveProjectIds += projectId
        if (wakeLock?.isHeld != true) wakeLock?.acquire(12 * 60 * 60 * 1_000L)
        val notificationStage = projectCreationNotificationStage(
            "opening", 0.0,
            project.servingSideStatus != ServingSideAnalysisStatus.DISABLED,
        )
        ensureForeground("Queued ${project.source.name}", true, notificationStage.title)
        broadcast(projectId, ProjectStatus.QUEUED, 0.0, "queued", "Waiting for inference")
        if (!workerRunning) {
            workerRunning = true
            executor.execute(::drainQueue)
        }
    }

    @Synchronized
    private fun enqueueServingSideProject(projectId: String) {
        val project = NativeProjectStore.get(this, projectId) ?: return
        if (project.status != ProjectStatus.READY ||
            project.servingSideStatus == ServingSideAnalysisStatus.READY ||
            projectId == activeProjectId || projectId in queuedIds
        ) return
        NativeProjectStore.updateServingSideStatus(
            this,
            projectId,
            ServingSideAnalysisStatus.QUEUED,
        )
        queue.addLast(AnalysisTask(projectId, true))
        queuedIds += projectId
        liveProjectIds += projectId
        if (wakeLock?.isHeld != true) wakeLock?.acquire(12 * 60 * 60 * 1_000L)
        ensureForeground(
            "Queued ${project.source.name}", true,
            ProjectNotificationStage("Serving-side analysis", 1, 1, 0.0).title,
        )
        broadcast(
            projectId, null, 0.0, "serving-side-queued",
            "Waiting to generate serving-side features", servingSideJob = true,
        )
        if (!workerRunning) {
            workerRunning = true
            executor.execute(::drainQueue)
        }
    }

    @Synchronized
    private fun deleteProject(projectId: String) {
        queue.removeAll { it.projectId == projectId }
        queuedIds -= projectId
        if (activeProjectId == projectId) activeCancellation?.set(true)
        NativeProjectStore.get(this, projectId)?.let {
            if (activeProjectId == projectId) deletingProjects[projectId] = it
            NativeProjectStore.delete(this, it)
        }
        broadcast(projectId, null, 0.0, "deleted", "Project deleted")
        if (!workerRunning && queue.isEmpty()) finishForeground()
    }

    private fun drainQueue() {
        while (!timedOut) {
            val task = synchronized(this) {
                val next = if (queue.isEmpty()) null else queue.removeFirst()
                if (next == null) {
                    workerRunning = false
                    null
                } else {
                    queuedIds -= next.projectId
                    activeProjectId = next.projectId
                    activeServingSideOnly = next.servingSideOnly
                    next
                }
            } ?: break
            if (task.servingSideOnly) analyzeServingSide(task.projectId) else analyze(task.projectId)
            synchronized(this) {
                activeProjectId = null
                activeServingSideOnly = false
                activeCancellation = null
            }
            liveProjectIds -= task.projectId
        }
        finishForeground()
    }

    private fun analyze(projectId: String) {
        val project = NativeProjectStore.get(this, projectId) ?: return
        val cancelled = AtomicBoolean(false)
        synchronized(this) { activeCancellation = cancelled }
        NativeProjectStore.updateStatus(this, projectId, ProjectStatus.ANALYZING)
        if (project.servingSideStatus != ServingSideAnalysisStatus.DISABLED) {
            NativeProjectStore.updateServingSideStatus(
                this, projectId, ServingSideAnalysisStatus.ANALYZING,
            )
        }
        broadcast(projectId, ProjectStatus.ANALYZING, 0.0, "opening", "Preparing game-window video + audio inference")
        val includeServingSide = project.servingSideStatus != ServingSideAnalysisStatus.DISABLED
        var latestNotificationStage = projectCreationNotificationStage("opening", 0.0, includeServingSide)
        var latestPerformance: AnalysisTypes.PerformanceStats? = null
        val notificationSpeed = NotificationStageSpeedTracker()
        fun notificationDetail(): String {
            val stage = latestNotificationStage
            return if (stage.step == 1 && latestPerformance != null) {
                projectInferenceNotificationDetail(
                    project.source.name, latestPerformance, stage.progressPercent,
                )
            } else {
                notificationSpeed.detail(
                    project.source.name, "${stage.step}/${stage.stepCount}", stage.progress,
                )
            }
        }
        updateNotification(
            latestNotificationStage.progressPercent,
            notificationDetail(),
            false,
            latestNotificationStage.title,
        )
        try {
            val result = AnalysisEngine(this).analyze(
                Uri.parse(project.source.uri),
                false,
                FeatureSchema.FULL_SOURCE_FRAME_LIMIT,
                AnalysisTypes.VideoDecoderOptions.defaults(),
                NativeFeatureCache.Mode.fromWireName(project.cacheMode),
                project.analysisWindow,
                cancelled,
                object : AnalysisTypes.ProgressListener {
                    override fun onProgress(stage: String, fraction: Double, detail: String) {
                        val overallProgress = projectCreationOverallProgress(
                            stage,
                            fraction,
                            includeServingSide,
                        )
                        latestNotificationStage = projectCreationNotificationStage(
                            stage, fraction, includeServingSide,
                        )
                        updateNotification(
                            latestNotificationStage.progressPercent,
                            notificationDetail(),
                            false,
                            latestNotificationStage.title,
                        )
                        broadcast(
                            projectId,
                            ProjectStatus.ANALYZING,
                            overallProgress,
                            stage,
                            detail,
                        )
                    }

                    override fun onPerformance(stats: AnalysisTypes.PerformanceStats) {
                        latestPerformance = stats
                        updateNotification(
                            latestNotificationStage.progressPercent,
                            notificationDetail(),
                            false,
                            latestNotificationStage.title,
                        )
                        broadcastPerformance(projectId, stats)
                    }
                },
                includeServingSide,
            )
            if (!cancelled.get() && NativeProjectStore.get(this, projectId) != null) {
                NativeProjectStore.complete(this, projectId, result)
                val disagreements = result.ranges().count {
                    ProductionEnsemble.isDisagreement(it.agreement())
                }
                val detail = "${result.ranges().size} merged ranges · $disagreements to validate · features cached"
                Log.i(TAG, resultLog(projectId, project.analysisWindow, result).toString())
                broadcast(projectId, ProjectStatus.READY, 1.0, "complete", detail)
                val completeStage = projectCreationNotificationStage("complete", 1.0, includeServingSide)
                updateNotification(100, project.source.name, false, completeStage.title)
            }
        } catch (error: Exception) {
            if (NativeProjectStore.get(this, projectId) != null) {
                if (cancelled.get() && !timedOut) {
                    NativeProjectStore.updateStatus(this, projectId, ProjectStatus.QUEUED)
                    broadcast(projectId, ProjectStatus.QUEUED, 0.0, "queued", "Inference interrupted; ready to resume")
                } else if (timedOut) {
                    val detail = "Android stopped project inference after its background time limit. Retry when the app is ready."
                    NativeProjectStore.updateStatus(this, projectId, ProjectStatus.ERROR, detail)
                    broadcast(projectId, ProjectStatus.ERROR, 0.0, "timeout", detail)
                } else {
                    val message = error.message ?: error.javaClass.simpleName
                    NativeProjectStore.updateStatus(this, projectId, ProjectStatus.ERROR, message)
                    Log.e(TAG, "Project $projectId failed", error)
                    broadcast(projectId, ProjectStatus.ERROR, 0.0, "failed", message)
                    updateNotification(0, "Failed: ${project.source.name}", false, "Project inference failed")
                }
            }
        } finally {
            synchronized(this) { deletingProjects.remove(projectId) }
                ?.let { NativeProjectStore.delete(this, it) }
        }
    }

    private fun analyzeServingSide(projectId: String) {
        val project = NativeProjectStore.get(this, projectId) ?: return
        if (project.status != ProjectStatus.READY) return
        val cancelled = AtomicBoolean(false)
        synchronized(this) { activeCancellation = cancelled }
        NativeProjectStore.updateServingSideStatus(
            this, projectId, ServingSideAnalysisStatus.ANALYZING,
        )
        broadcast(
            projectId, null, 0.0, "serving-side",
            "Generating serving-side features", servingSideJob = true,
        )
        val notificationSpeed = NotificationStageSpeedTracker()
        updateNotification(
            0, notificationSpeed.detail(project.source.name, "1/1", 0.0), true,
            ProjectNotificationStage("Serving-side analysis", 1, 1, 0.0).title,
        )
        try {
            val output = ServingSideInference.run(
                this,
                Uri.parse(project.source.uri),
                project.media,
                project.roi,
                project.ranges.map {
                    AnalysisTypes.Interval(
                        it.startMs / 1_000.0,
                        it.endMs / 1_000.0,
                        it.confidence,
                        it.agreement,
                    )
                },
                project.productionServeOutputs,
                object : AnalysisTypes.ProgressListener {
                    override fun onProgress(stage: String, fraction: Double, detail: String) {
                        val overallProgress = servingSideOverallProgress(stage, fraction)
                        val percent = (overallProgress * 100).toInt()
                        updateNotification(
                            percent,
                            notificationSpeed.detail(project.source.name, "1/1", overallProgress),
                            false,
                            ProjectNotificationStage("Serving-side analysis", 1, 1, overallProgress).title,
                        )
                        broadcast(
                            projectId, null, overallProgress, stage, detail,
                            servingSideJob = true,
                        )
                    }

                    override fun onPerformance(stats: AnalysisTypes.PerformanceStats) = Unit
                },
                cancelled::get,
            )
            if (!cancelled.get() && NativeProjectStore.get(this, projectId) != null) {
                NativeProjectStore.completeServingSide(this, projectId, output)
                val reviewCount = output.candidates.count {
                    it.verdict == ServingSideVerdict.REVIEW
                }
                val detail = "${output.candidates.size} serve candidates · $reviewCount to review"
                broadcast(
                    projectId, null, 1.0, "serving-side-complete", detail,
                    servingSideJob = true,
                )
                updateNotification(
                    100,
                    notificationSpeed.detail(project.source.name, "1/1", 1.0),
                    false,
                    "Score tracking ready (1/1)",
                )
            }
        } catch (error: Exception) {
            if (NativeProjectStore.get(this, projectId) != null) {
                val detail = when {
                    cancelled.get() && !timedOut -> {
                        NativeProjectStore.updateServingSideStatus(
                            this, projectId, ServingSideAnalysisStatus.QUEUED,
                        )
                        "Serving-side inference interrupted; ready to resume"
                    }
                    timedOut -> {
                        val message = "Android stopped serving-side inference after its background time limit. Retry by enabling score tracking again."
                        NativeProjectStore.updateServingSideStatus(
                            this, projectId, ServingSideAnalysisStatus.ERROR, message,
                        )
                        message
                    }
                    else -> {
                        val message = error.message ?: error.javaClass.simpleName
                        NativeProjectStore.updateServingSideStatus(
                            this, projectId, ServingSideAnalysisStatus.ERROR, message,
                        )
                        Log.e(TAG, "Serving-side inference failed for $projectId", error)
                        message
                    }
                }
                broadcast(
                    projectId, null, 0.0,
                    if (cancelled.get() && !timedOut) "serving-side-queued" else "serving-side-failed",
                    detail,
                    servingSideJob = true,
                )
                updateNotification(0, project.source.name, false, "Score tracking unavailable")
            }
        } finally {
            synchronized(this) { deletingProjects.remove(projectId) }
                ?.let { NativeProjectStore.delete(this, it) }
        }
    }

    private fun resultLog(
        projectId: String,
        analysisWindow: AnalysisTypes.AnalysisWindow,
        result: AnalysisTypes.AnalysisResult,
    ) = JSONObject().apply {
        put("schemaVersion", 1)
        put("method", "android-project-inference-ensemble-v2")
        put("projectId", projectId)
        put("modelId", FeatureSchema.MODEL_ID)
        put("sourceName", result.displayName())
        put("analysisWindow", JSONArray(listOf(analysisWindow.start(), analysisWindow.end())))
        put("totalMilliseconds", result.totalMilliseconds())
        put("sampleRows", result.sampleRows())
        put("decodedSourceFrames", result.decodedSourceFrames())
        put("decodedAudioFrames", result.decodedAudioFrames())
        put("audioFeatureFrames", result.audioFeatureFrames())
        put("audioCodecOperatingRate", result.audioCodecOperatingRate())
        put("audioCodecPriority", result.audioCodecPriority())
        put("audioMultipleFramesSupported", result.audioMultipleFramesSupported())
        put("stageMilliseconds", JSONObject(result.stageMilliseconds()))
        put("featureCache", JSONObject().apply {
            put("visualHit", result.featureCache().visualHit())
            put("audioHit", result.featureCache().audioHit())
            put("contextHit", result.featureCache().contextHit())
            put("complete", result.featureCache().complete())
            put("bytes", result.featureCache().bytes())
        })
        put("ranges", JSONArray().apply {
            result.ranges().forEach { range -> put(JSONObject().apply {
                put("start", range.start())
                put("end", range.end())
                put("confidence", range.confidence().toDouble())
                put("agreement", range.agreement())
            }) }
        })
    }

    private fun broadcast(
        projectId: String,
        status: ProjectStatus?,
        progress: Double,
        stage: String,
        detail: String,
        servingSideJob: Boolean = false,
    ) {
        sendBroadcast(Intent(ACTION_UPDATE).setPackage(packageName).apply {
            putExtra(EXTRA_PROJECT_ID, projectId)
            if (status != null) putExtra(EXTRA_STATUS, status.wireName)
            putExtra(EXTRA_PROGRESS, progress)
            putExtra(EXTRA_STAGE, stage)
            putExtra(EXTRA_DETAIL, detail)
            putExtra(EXTRA_SERVING_SIDE_JOB, servingSideJob)
        })
    }

    private fun broadcastPerformance(projectId: String, stats: AnalysisTypes.PerformanceStats) {
        sendBroadcast(Intent(ACTION_UPDATE).setPackage(packageName).apply {
            putExtra(EXTRA_PROJECT_ID, projectId)
            putExtra(EXTRA_PERFORMANCE, true)
            putExtra(EXTRA_GENERATED_FRAMES, stats.generatedFrames())
            putExtra(EXTRA_TOTAL_FRAMES, stats.totalFrames())
            putExtra(EXTRA_DECODED_FRAMES, stats.decodedSourceFrames())
            putExtra(EXTRA_ELAPSED_SECONDS, stats.elapsedSeconds())
            putExtra(EXTRA_FPS, stats.framesPerSecond())
            putExtra(EXTRA_REALTIME, stats.realtimeRatio())
            putExtra(EXTRA_ETA_SECONDS, stats.etaSeconds())
        })
    }

    private fun ensureForeground(
        detail: String,
        indeterminate: Boolean,
        title: String = "VolleyCut project inference",
    ) {
        if (foreground) {
            updateNotification(0, detail, indeterminate, title)
            return
        }
        val notification = notification(0, detail, true, indeterminate, title)
        if (Build.VERSION.SDK_INT >= 35) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING,
            )
        } else startForeground(NOTIFICATION_ID, notification)
        foreground = true
    }

    private fun updateNotification(
        progress: Int,
        detail: String,
        indeterminate: Boolean,
        title: String = "VolleyCut project inference",
    ) {
        getSystemService(NotificationManager::class.java).notify(
            NOTIFICATION_ID,
            notification(progress, detail, true, indeterminate, title),
        )
    }

    private fun notification(
        progress: Int,
        detail: String,
        ongoing: Boolean,
        indeterminate: Boolean,
        title: String,
    ): Notification {
        val content = PendingIntent.getActivity(
            this,
            0,
            Intent(this, EditorActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_upload)
            .setContentTitle(title)
            .setContentText(detail)
            .setContentIntent(content)
            .setOnlyAlertOnce(true)
            .setOngoing(ongoing)
            .setProgress(100, progress.coerceIn(0, 100), indeterminate)
            .build()
    }

    @Synchronized
    private fun finishForeground() {
        if (foreground) {
            stopForeground(STOP_FOREGROUND_REMOVE)
            foreground = false
        }
        if (wakeLock?.isHeld == true) wakeLock?.release()
        stopSelf()
    }

    override fun onTimeout(startId: Int, fgsType: Int) {
        val activeTask = synchronized(this) {
            timedOut = true
            activeCancellation?.set(true)
            activeProjectId?.let { AnalysisTask(it, activeServingSideOnly) }
        }
        val affectedTasks = synchronized(this) {
            (queue.toList() + listOfNotNull(activeTask)).distinctBy { it.projectId }.also {
                queue.clear()
                queuedIds.clear()
                workerRunning = false
            }
        }
        val detail = "Android stopped project inference after its background time limit. Retry when the app is ready."
        affectedTasks.forEach { task ->
            if (task.servingSideOnly) {
                NativeProjectStore.updateServingSideStatus(
                    this, task.projectId, ServingSideAnalysisStatus.ERROR, detail,
                )
                broadcast(
                    task.projectId, null, 0.0, "serving-side-failed", detail,
                    servingSideJob = true,
                )
            } else {
                NativeProjectStore.updateStatus(this, task.projectId, ProjectStatus.ERROR, detail)
                broadcast(task.projectId, ProjectStatus.ERROR, 0.0, "timeout", detail)
            }
        }
        val project = activeTask?.projectId?.let { NativeProjectStore.get(this, it) }
        ProcessingTimeoutTracker.record(
            context = this,
            operation = MediaProcessingOperation.INFERENCE,
            projectId = project?.id,
            sourceName = project?.source?.name,
            detail = detail,
        )
        executor.shutdownNow()
        if (foreground) {
            stopForeground(STOP_FOREGROUND_REMOVE)
            foreground = false
        }
        if (wakeLock?.isHeld == true) wakeLock?.release()
        stopSelf()
    }

    override fun onDestroy() {
        activeCancellation?.set(true)
        executor.shutdownNow()
        liveProjectIds.clear()
        if (wakeLock?.isHeld == true) wakeLock?.release()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    companion object {
        private const val TAG = "VolleyCutProjects"
        private const val CHANNEL_ID = "volleycut_project_inference"
        private const val NOTIFICATION_ID = 402
        private val liveProjectIds = ConcurrentHashMap.newKeySet<String>()
        const val ACTION_ENQUEUE = "com.volleycut.nativeanalysis.ENQUEUE_PROJECT"
        const val ACTION_ENQUEUE_SERVING_SIDE = "com.volleycut.nativeanalysis.ENQUEUE_SERVING_SIDE"
        const val ACTION_DELETE = "com.volleycut.nativeanalysis.DELETE_PROJECT"
        const val ACTION_UPDATE = "com.volleycut.nativeanalysis.PROJECT_UPDATE"
        const val EXTRA_PROJECT_ID = "project_id"
        const val EXTRA_STATUS = "project_status"
        const val EXTRA_PROGRESS = "project_progress"
        const val EXTRA_STAGE = "project_stage"
        const val EXTRA_DETAIL = "project_detail"
        const val EXTRA_PERFORMANCE = "project_performance"
        const val EXTRA_GENERATED_FRAMES = "project_generated_frames"
        const val EXTRA_TOTAL_FRAMES = "project_total_frames"
        const val EXTRA_DECODED_FRAMES = "project_decoded_frames"
        const val EXTRA_ELAPSED_SECONDS = "project_elapsed_seconds"
        const val EXTRA_FPS = "project_fps"
        const val EXTRA_REALTIME = "project_realtime"
        const val EXTRA_ETA_SECONDS = "project_eta_seconds"
        const val EXTRA_SERVING_SIDE_JOB = "project_serving_side_job"

        fun enqueue(context: Context, projectId: String) {
            ContextCompat.startForegroundService(
                context,
                Intent(context, ProjectAnalysisService::class.java)
                    .setAction(ACTION_ENQUEUE)
                    .putExtra(EXTRA_PROJECT_ID, projectId),
            )
        }

        fun enqueueServingSide(context: Context, projectId: String) {
            ContextCompat.startForegroundService(
                context,
                Intent(context, ProjectAnalysisService::class.java)
                    .setAction(ACTION_ENQUEUE_SERVING_SIDE)
                    .putExtra(EXTRA_PROJECT_ID, projectId),
            )
        }

        fun delete(context: Context, projectId: String) {
            context.startService(
                Intent(context, ProjectAnalysisService::class.java)
                    .setAction(ACTION_DELETE)
                    .putExtra(EXTRA_PROJECT_ID, projectId),
            )
        }

        fun resumePending(context: Context) {
            NativeProjectStore.list(context)
                .filter { it.status == ProjectStatus.QUEUED || it.status == ProjectStatus.ANALYZING }
                .sortedBy { it.createdAtMs }
                .forEach {
                    if (it.id in liveProjectIds) return@forEach
                    if (it.status == ProjectStatus.ANALYZING) {
                        NativeProjectStore.updateStatus(context, it.id, ProjectStatus.QUEUED)
                    }
                    enqueue(context, it.id)
                }
            NativeProjectStore.list(context)
                .filter {
                    it.status == ProjectStatus.READY &&
                        (it.servingSideStatus == ServingSideAnalysisStatus.QUEUED ||
                            it.servingSideStatus == ServingSideAnalysisStatus.ANALYZING)
                }
                .sortedBy { it.createdAtMs }
                .forEach {
                    if (it.id in liveProjectIds) return@forEach
                    if (it.servingSideStatus == ServingSideAnalysisStatus.ANALYZING) {
                        NativeProjectStore.updateServingSideStatus(
                            context, it.id, ServingSideAnalysisStatus.QUEUED,
                        )
                    }
                    enqueueServingSide(context, it.id)
                }
        }
    }
}
