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

/** Serial foreground queue so decoding continues while a different project is edited. */
class ProjectAnalysisService : Service() {
    private val executor = Executors.newSingleThreadExecutor()
    private val queue = ArrayDeque<String>()
    private val queuedIds = mutableSetOf<String>()
    private val deletingProjects = mutableMapOf<String, NativeProject>()
    private var activeProjectId: String? = null
    private var activeCancellation: AtomicBoolean? = null
    private var workerRunning = false
    private var foreground = false
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
        when (intent?.action) {
            ACTION_DELETE -> intent.getStringExtra(EXTRA_PROJECT_ID)?.let(::deleteProject)
            ACTION_ENQUEUE -> intent.getStringExtra(EXTRA_PROJECT_ID)?.let(::enqueueProject)
        }
        return START_NOT_STICKY
    }

    @Synchronized
    private fun enqueueProject(projectId: String) {
        val project = NativeProjectStore.get(this, projectId) ?: return
        if (project.status == ProjectStatus.READY || projectId == activeProjectId || projectId in queuedIds) return
        NativeProjectStore.updateStatus(this, projectId, ProjectStatus.QUEUED)
        queue.addLast(projectId)
        queuedIds += projectId
        liveProjectIds += projectId
        if (wakeLock?.isHeld != true) wakeLock?.acquire(12 * 60 * 60 * 1_000L)
        ensureForeground("Queued ${project.source.name}", true)
        broadcast(projectId, ProjectStatus.QUEUED, 0.0, "queued", "Waiting for inference")
        if (!workerRunning) {
            workerRunning = true
            executor.execute(::drainQueue)
        }
    }

    @Synchronized
    private fun deleteProject(projectId: String) {
        queue.removeAll { it == projectId }
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
        while (true) {
            val projectId = synchronized(this) {
                val next = if (queue.isEmpty()) null else queue.removeFirst()
                if (next == null) {
                    workerRunning = false
                    null
                } else {
                    queuedIds -= next
                    activeProjectId = next
                    next
                }
            } ?: break
            analyze(projectId)
            synchronized(this) {
                activeProjectId = null
                activeCancellation = null
            }
            liveProjectIds -= projectId
        }
        finishForeground()
    }

    private fun analyze(projectId: String) {
        val project = NativeProjectStore.get(this, projectId) ?: return
        val cancelled = AtomicBoolean(false)
        synchronized(this) { activeCancellation = cancelled }
        NativeProjectStore.updateStatus(this, projectId, ProjectStatus.ANALYZING)
        broadcast(projectId, ProjectStatus.ANALYZING, 0.0, "opening", "Preparing game-window video + audio inference")
        updateNotification(0, "Analyzing ${project.source.name}", true)
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
                        val percent = (fraction.coerceIn(0.0, 1.0) * 100).toInt()
                        updateNotification(percent, "${project.source.name} · $detail", true)
                        broadcast(projectId, ProjectStatus.ANALYZING, fraction, stage, detail)
                    }

                    override fun onPerformance(stats: AnalysisTypes.PerformanceStats) {
                        broadcastPerformance(projectId, stats)
                    }
                },
            )
            if (!cancelled.get() && NativeProjectStore.get(this, projectId) != null) {
                NativeProjectStore.complete(this, projectId, result)
                val disagreements = result.ranges().count {
                    ProductionEnsemble.isDisagreement(it.agreement())
                }
                val detail = "${result.ranges().size} merged ranges · $disagreements to validate · features cached"
                Log.i(TAG, resultLog(projectId, project.analysisWindow, result).toString())
                broadcast(projectId, ProjectStatus.READY, 1.0, "complete", detail)
                updateNotification(100, "Ready: ${project.source.name}", false)
            }
        } catch (error: Exception) {
            if (NativeProjectStore.get(this, projectId) != null) {
                if (cancelled.get()) {
                    NativeProjectStore.updateStatus(this, projectId, ProjectStatus.QUEUED)
                    broadcast(projectId, ProjectStatus.QUEUED, 0.0, "queued", "Inference interrupted; ready to resume")
                } else {
                    val message = error.message ?: error.javaClass.simpleName
                    NativeProjectStore.updateStatus(this, projectId, ProjectStatus.ERROR, message)
                    Log.e(TAG, "Project $projectId failed", error)
                    broadcast(projectId, ProjectStatus.ERROR, 0.0, "failed", message)
                    updateNotification(0, "Failed: ${project.source.name}", false)
                }
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
    ) {
        sendBroadcast(Intent(ACTION_UPDATE).setPackage(packageName).apply {
            putExtra(EXTRA_PROJECT_ID, projectId)
            if (status != null) putExtra(EXTRA_STATUS, status.wireName)
            putExtra(EXTRA_PROGRESS, progress)
            putExtra(EXTRA_STAGE, stage)
            putExtra(EXTRA_DETAIL, detail)
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

    private fun ensureForeground(detail: String, indeterminate: Boolean) {
        if (foreground) {
            updateNotification(0, detail, indeterminate)
            return
        }
        val notification = notification(0, detail, true, indeterminate)
        if (Build.VERSION.SDK_INT >= 35) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING,
            )
        } else startForeground(NOTIFICATION_ID, notification)
        foreground = true
    }

    private fun updateNotification(progress: Int, detail: String, indeterminate: Boolean) {
        getSystemService(NotificationManager::class.java).notify(
            NOTIFICATION_ID,
            notification(progress, detail, true, indeterminate),
        )
    }

    private fun notification(progress: Int, detail: String, ongoing: Boolean, indeterminate: Boolean): Notification {
        val content = PendingIntent.getActivity(
            this,
            0,
            Intent(this, EditorActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_upload)
            .setContentTitle("VolleyCut project inference")
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

        fun enqueue(context: Context, projectId: String) {
            ContextCompat.startForegroundService(
                context,
                Intent(context, ProjectAnalysisService::class.java)
                    .setAction(ACTION_ENQUEUE)
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
        }
    }
}
