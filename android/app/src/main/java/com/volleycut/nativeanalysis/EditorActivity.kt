package com.volleycut.nativeanalysis

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.OpenableColumns
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.Orientation
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.draggable
import androidx.compose.foundation.gestures.rememberDraggableState
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.media3.common.MediaItem
import androidx.media3.common.PlaybackParameters
import androidx.media3.common.Player
import androidx.media3.common.util.UnstableApi
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.SeekParameters
import androidx.media3.ui.compose.ContentFrame
import androidx.media3.ui.compose.SURFACE_TYPE_SURFACE_VIEW
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.roundToLong

class EditorActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val seed = seedFromIntent(intent) ?: EditorProjectStore.load(this)
        setContent {
            VolleyCutTheme {
                EditorApp(activity = this, initialSeed = seed)
            }
        }
    }

    private fun seedFromIntent(intent: Intent): EditorSeed? {
        val sourceUri = intent.getStringExtra(EXTRA_SOURCE_URI) ?: return null
        val durationMs = intent.getLongExtra(EXTRA_DURATION_MS, 0)
        val starts = intent.getLongArrayExtra(EXTRA_RANGE_STARTS) ?: longArrayOf()
        val ends = intent.getLongArrayExtra(EXTRA_RANGE_ENDS) ?: longArrayOf()
        val confidence = intent.getFloatArrayExtra(EXTRA_RANGE_CONFIDENCE) ?: floatArrayOf()
        if (durationMs <= 0 || starts.size != ends.size || starts.size != confidence.size) return null
        return EditorSeed(
            sourceUri = sourceUri,
            displayName = intent.getStringExtra(EXTRA_DISPLAY_NAME) ?: "recording.mp4",
            durationMs = durationMs,
            width = intent.getIntExtra(EXTRA_WIDTH, 0),
            height = intent.getIntExtra(EXTRA_HEIGHT, 0),
            rotation = intent.getIntExtra(EXTRA_ROTATION, 0),
            ranges = starts.indices.map { SeedRange(starts[it], ends[it], confidence[it]) },
        )
    }

    companion object {
        private const val EXTRA_SOURCE_URI = "editor_source_uri"
        private const val EXTRA_DISPLAY_NAME = "editor_display_name"
        private const val EXTRA_DURATION_MS = "editor_duration_ms"
        private const val EXTRA_WIDTH = "editor_width"
        private const val EXTRA_HEIGHT = "editor_height"
        private const val EXTRA_ROTATION = "editor_rotation"
        private const val EXTRA_RANGE_STARTS = "editor_range_starts"
        private const val EXTRA_RANGE_ENDS = "editor_range_ends"
        private const val EXTRA_RANGE_CONFIDENCE = "editor_range_confidence"

        @JvmStatic
        fun createIntent(context: Context, result: AnalysisTypes.AnalysisResult): Intent {
            val seed = editorSeedFromResult(result)
            EditorProjectStore.save(context, seed)
            val project = NativeProjectStore.fromResult(context, result)
            NativeProjectStore.save(context, project)
            NativeProjectStore.setSelectedId(context, project.id)
            return intentFromSeed(context, seed)
        }

        @JvmStatic
        fun createResumeIntent(context: Context): Intent? {
            val projects = NativeProjectStore.list(context)
            val selected = NativeProjectStore.selectedId(context)
                ?.let { id -> projects.firstOrNull { it.id == id } }
            return (selected?.editorSeed() ?: projects.firstNotNullOfOrNull { it.editorSeed() }
                ?: EditorProjectStore.load(context))?.let { intentFromSeed(context, it) }
        }

        private fun intentFromSeed(context: Context, seed: EditorSeed) =
            Intent(context, EditorActivity::class.java).apply {
                putExtra(EXTRA_SOURCE_URI, seed.sourceUri)
                putExtra(EXTRA_DISPLAY_NAME, seed.displayName)
                putExtra(EXTRA_DURATION_MS, seed.durationMs)
                putExtra(EXTRA_WIDTH, seed.width)
                putExtra(EXTRA_HEIGHT, seed.height)
                putExtra(EXTRA_ROTATION, seed.rotation)
                putExtra(EXTRA_RANGE_STARTS, seed.ranges.map { it.startMs }.toLongArray())
                putExtra(EXTRA_RANGE_ENDS, seed.ranges.map { it.endMs }.toLongArray())
                putExtra(EXTRA_RANGE_CONFIDENCE, seed.ranges.map { it.confidence }.toFloatArray())
            }
    }
}

private fun editorSeedFromResult(result: AnalysisTypes.AnalysisResult) = EditorSeed(
    sourceUri = result.source().toString(),
    displayName = result.displayName(),
    durationMs = secondsToMs(result.media().durationSeconds()),
    width = result.media().width(),
    height = result.media().height(),
    rotation = result.media().rotation(),
    ranges = result.ranges().map {
        SeedRange(secondsToMs(it.start()), secondsToMs(it.end()), it.confidence())
    },
)

private val Paper = Color(0xFFF7F4EE)
private val Ink = Color(0xFF20201E)
private val Orange = Color(0xFFEF5B35)
private val Green = Color(0xFF26734D)
private val PaleGreen = Color(0xFF9ED5B5)
private val Muted = Color(0xFF77736C)
private val Rail = Color(0xFFE4E0D7)
private val Danger = Color(0xFFB3261E)
private val Warning = Color(0xFFE8A317)

@Composable
private fun VolleyCutTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = lightColorScheme(
            primary = Orange,
            onPrimary = Color.White,
            background = Paper,
            onBackground = Ink,
            surface = Color.White,
            onSurface = Ink,
            error = Danger,
        ),
        content = content,
    )
}

private data class ExportUiState(
    val status: String = "idle",
    val progress: Int = 0,
    val detail: String = "",
    val metrics: String? = null,
)

private data class SourceSelection(val uri: Uri, val displayName: String)

private data class InferenceUiState(
    val projectId: String? = null,
    val running: Boolean = false,
    val progress: Float = 0f,
    val stage: String = "",
    val detail: String = "",
    val performance: AnalysisTypes.PerformanceStats? = null,
    val error: String? = null,
)

@Composable
private fun EditorApp(activity: ComponentActivity, initialSeed: EditorSeed?) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val initialProjects = remember {
        var stored = NativeProjectStore.list(context)
            .map { NativeProjectStore.repairLegacyMetadata(context, it) }
        if (initialSeed != null && stored.none { it.source.uri == initialSeed.sourceUri }) {
            NativeProjectStore.fromSeed(context, initialSeed).also {
                NativeProjectStore.save(context, it)
                stored = NativeProjectStore.list(context)
            }
        }
        stored
    }
    var projects by remember { mutableStateOf(initialProjects) }
    var selectedProjectId by remember {
        mutableStateOf(
            NativeProjectStore.selectedId(context)
                ?.takeIf { id -> initialProjects.any { it.id == id } }
                ?: initialProjects.firstOrNull()?.id,
        )
    }
    var creatingNew by remember { mutableStateOf(selectedProjectId == null) }
    var selectedSource by remember { mutableStateOf<SourceSelection?>(null) }
    var preparingSource by remember { mutableStateOf(false) }
    var inference by remember { mutableStateOf(InferenceUiState()) }
    var useCache by remember { mutableStateOf(true) }
    var cacheBytes by remember { mutableLongStateOf(NativeFeatureCache.totalBytes(context)) }
    var confirmDelete by remember { mutableStateOf<NativeProject?>(null) }

    fun reloadProjects(
        preferredId: String? = selectedProjectId,
        preserveNewProject: Boolean = creatingNew,
    ) {
        projects = NativeProjectStore.list(context)
        if (preserveNewProject) {
            selectedProjectId = null
            NativeProjectStore.setSelectedId(context, null)
            cacheBytes = NativeFeatureCache.totalBytes(context)
            return
        }
        selectedProjectId = preferredId?.takeIf { id -> projects.any { it.id == id } }
            ?: projects.firstOrNull()?.id
        NativeProjectStore.setSelectedId(context, selectedProjectId)
        if (selectedProjectId != null) creatingNew = false
        cacheBytes = NativeFeatureCache.totalBytes(context)
    }

    val sourcePicker = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument(),
    ) { uri ->
        if (uri != null) {
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    uri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION,
                )
            }
            selectedSource = SourceSelection(uri, sourceDisplayName(context, uri))
            inference = InferenceUiState(detail = "Ready to create and queue this project")
        }
    }
    val notificationPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { }

    fun queueSelectedSource() {
        val selected = selectedSource ?: return
        if (preparingSource) return
        preparingSource = true
        inference = InferenceUiState(
            stage = "opening",
            detail = "Reading recording metadata",
        )
        scope.launch {
            try {
                val prepared = withContext(Dispatchers.IO) {
                    val engine = AnalysisEngine(context)
                    val media = engine.probe(selected.uri)
                    val source = NativeProjectStore.source(context, selected.uri, selected.displayName)
                    Triple(source, media, AnalysisEngine.inferRoi(source.name))
                }
                val candidate = NativeProjectStore.newQueued(prepared.first, prepared.second, prepared.third)
                val existing = NativeProjectStore.findMatching(context, candidate)
                val reusable = useCache && existing?.status == ProjectStatus.READY &&
                    existing.modelId == FeatureSchema.MODEL_ID
                if (reusable) {
                    selectedProjectId = existing.id
                    creatingNew = false
                    selectedSource = null
                    inference = InferenceUiState(
                        projectId = existing.id,
                        progress = 1f,
                        stage = "complete",
                        detail = "Opened cached inference; no analysis was run",
                    )
                    reloadProjects(existing.id)
                } else if (existing?.status == ProjectStatus.QUEUED ||
                    existing?.status == ProjectStatus.ANALYZING
                ) {
                    selectedProjectId = existing.id
                    creatingNew = false
                    selectedSource = null
                    inference = InferenceUiState(
                        projectId = existing.id,
                        running = true,
                        stage = existing.status.wireName,
                        detail = "This recording is already in the inference queue",
                    )
                    reloadProjects(existing.id)
                } else {
                    val queued = candidate.copy(
                        id = existing?.id ?: candidate.id,
                        createdAtMs = existing?.createdAtMs ?: candidate.createdAtMs,
                        cacheMode = if (useCache) NativeFeatureCache.Mode.USE.wireName()
                            else NativeFeatureCache.Mode.REFRESH.wireName(),
                    )
                    withContext(Dispatchers.IO) { NativeProjectStore.save(context, queued) }
                    selectedProjectId = queued.id
                    creatingNew = false
                    selectedSource = null
                    inference = InferenceUiState(
                        projectId = queued.id,
                        running = true,
                        stage = "queued",
                        detail = "Waiting for full video + audio inference",
                    )
                    reloadProjects(queued.id)
                    if (Build.VERSION.SDK_INT >= 33 &&
                        context.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
                    ) notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
                    ProjectAnalysisService.enqueue(context, queued.id)
                }
            } catch (error: Exception) {
                inference = inference.copy(
                    running = false,
                    stage = "failed",
                    detail = "",
                    error = error.message ?: error.javaClass.simpleName,
                )
            } finally {
                preparingSource = false
            }
        }
    }

    DisposableEffect(Unit) {
        val receiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context?, intent: Intent?) {
                if (intent?.action != ProjectAnalysisService.ACTION_UPDATE) return
                val projectId = intent.getStringExtra(ProjectAnalysisService.EXTRA_PROJECT_ID) ?: return
                if (intent.getBooleanExtra(ProjectAnalysisService.EXTRA_PERFORMANCE, false)) {
                    if (projectId == selectedProjectId) {
                        inference = inference.copy(
                            projectId = projectId,
                            running = true,
                            performance = AnalysisTypes.PerformanceStats(
                                intent.getIntExtra(ProjectAnalysisService.EXTRA_GENERATED_FRAMES, 0),
                                intent.getIntExtra(ProjectAnalysisService.EXTRA_TOTAL_FRAMES, 0),
                                intent.getIntExtra(ProjectAnalysisService.EXTRA_DECODED_FRAMES, 0),
                                0.0,
                                intent.getDoubleExtra(ProjectAnalysisService.EXTRA_ELAPSED_SECONDS, 0.0),
                                intent.getDoubleExtra(ProjectAnalysisService.EXTRA_FPS, 0.0),
                                intent.getDoubleExtra(ProjectAnalysisService.EXTRA_REALTIME, 0.0),
                                intent.getDoubleExtra(ProjectAnalysisService.EXTRA_ETA_SECONDS, 0.0),
                                0,
                                0,
                            ),
                        )
                    }
                    return
                }
                if (projectId == selectedProjectId) {
                    val status = intent.getStringExtra(ProjectAnalysisService.EXTRA_STATUS)
                        ?.let(ProjectStatus::fromWireName)
                    inference = inference.copy(
                        projectId = projectId,
                        running = status == ProjectStatus.QUEUED || status == ProjectStatus.ANALYZING,
                        progress = intent.getDoubleExtra(ProjectAnalysisService.EXTRA_PROGRESS, 0.0)
                            .coerceIn(0.0, 1.0).toFloat(),
                        stage = intent.getStringExtra(ProjectAnalysisService.EXTRA_STAGE).orEmpty(),
                        detail = intent.getStringExtra(ProjectAnalysisService.EXTRA_DETAIL).orEmpty(),
                        error = if (status == ProjectStatus.ERROR) {
                            intent.getStringExtra(ProjectAnalysisService.EXTRA_DETAIL)
                        } else null,
                    )
                }
                reloadProjects(selectedProjectId)
            }
        }
        ContextCompat.registerReceiver(
            context,
            receiver,
            IntentFilter(ProjectAnalysisService.ACTION_UPDATE),
            ContextCompat.RECEIVER_NOT_EXPORTED,
        )
        ProjectAnalysisService.resumePending(context)
        onDispose { runCatching { context.unregisterReceiver(receiver) } }
    }

    if (confirmDelete != null) {
        val deleting = checkNotNull(confirmDelete)
        AlertDialog(
            onDismissRequest = { confirmDelete = null },
            title = { Text("Delete ${deleting.source.name}?") },
            text = {
                Text("This removes its inference, saved editor changes, and generated feature cache from this device.")
            },
            confirmButton = {
                TextButton(onClick = {
                    val nextId = projects.firstOrNull { it.id != deleting.id }?.id
                    ProjectAnalysisService.delete(context, deleting.id)
                    projects = projects.filterNot { it.id == deleting.id }
                    selectedProjectId = nextId
                    NativeProjectStore.setSelectedId(context, nextId)
                    creatingNew = nextId == null
                    inference = InferenceUiState(detail = "Project deleted")
                    confirmDelete = null
                }) { Text("Delete", color = Danger) }
            },
            dismissButton = { TextButton(onClick = { confirmDelete = null }) { Text("Cancel") } },
        )
    }

    val selectedProject = projects.firstOrNull { it.id == selectedProjectId }
    val queueCount = projects.count {
        it.status == ProjectStatus.QUEUED || it.status == ProjectStatus.ANALYZING
    }
    val projectControls: @Composable () -> Unit = {
        ProjectHeaderBar(
            projects = projects,
            selected = selectedProject,
            creatingNew = creatingNew,
            queueCount = queueCount,
            onSelect = { project ->
                selectedProjectId = project.id
                NativeProjectStore.setSelectedId(context, project.id)
                creatingNew = false
                inference = InferenceUiState(projectId = project.id)
            },
            onNew = {
                creatingNew = true
                selectedProjectId = null
                NativeProjectStore.setSelectedId(context, null)
                selectedSource = null
                inference = InferenceUiState(detail = "Choose a recording for the new project")
            },
            onDelete = { selectedProject?.let { confirmDelete = it } },
        )
    }

    if (creatingNew || selectedProject == null) {
        ProjectShell(projectControls) {
            NewProjectCard(
                selected = selectedSource,
                preparing = preparingSource,
                state = inference,
                useCache = useCache,
                cacheBytes = cacheBytes,
                queueCount = queueCount,
                onSelect = { sourcePicker.launch(arrayOf("video/*")) },
                onQueue = ::queueSelectedSource,
                onUseCache = { useCache = it },
                onClearCache = {
                    NativeFeatureCache.clearAll(context)
                    cacheBytes = 0
                    inference = InferenceUiState(detail = "Feature cache cleared")
                },
                onBenchmark = { context.startActivity(Intent(context, MainActivity::class.java)) },
            )
        }
    } else if (selectedProject.status != ProjectStatus.READY) {
        ProjectShell(projectControls) {
            ProjectInferenceCard(
                project = selectedProject,
                state = inference.takeIf { it.projectId == selectedProject.id } ?: InferenceUiState(),
                onRetry = {
                    val queued = selectedProject.copy(
                        status = ProjectStatus.QUEUED,
                        error = null,
                        updatedAtMs = System.currentTimeMillis(),
                    )
                    NativeProjectStore.save(context, queued)
                    inference = InferenceUiState(
                        projectId = queued.id,
                        running = true,
                        stage = "queued",
                        detail = "Waiting for full video + audio inference",
                    )
                    reloadProjects(queued.id)
                    ProjectAnalysisService.enqueue(context, queued.id)
                },
            )
        }
    } else {
        val currentSeed = checkNotNull(selectedProject.editorSeed())
        key(selectedProject.id, currentSeed.sourceRevision) {
            val store = remember(selectedProject.id, currentSeed.sourceRevision) {
                EditorDraftStore(context, currentSeed)
            }
            val restored = remember(selectedProject.id, currentSeed.sourceRevision) { store.load() }
            EditorScreen(
                activity = activity,
                seed = currentSeed,
                store = store,
                initialDraft = restored ?: EditorMath.newDraft(currentSeed),
                restored = restored != null,
                analysisRunning = queueCount > 0,
                sourceControls = projectControls,
            )
        }
    }
}

@Composable
private fun ProjectHeaderBar(
    projects: List<NativeProject>,
    selected: NativeProject?,
    creatingNew: Boolean,
    queueCount: Int,
    onSelect: (NativeProject) -> Unit,
    onNew: () -> Unit,
    onDelete: () -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }
    Surface(
        color = Color.White,
        shape = RoundedCornerShape(12.dp),
        shadowElevation = 1.dp,
    ) {
        Column(Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("VOLLEYCUT", color = Orange, fontWeight = FontWeight.Black, letterSpacing = 2.sp)
                Spacer(Modifier.weight(1f))
                Text(
                    if (queueCount == 0) "Queue idle" else "$queueCount in queue",
                    color = if (queueCount == 0) Muted else Green,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.SemiBold,
                )
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.weight(1f)) {
                    OutlinedButton(
                        onClick = { expanded = true },
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Text(
                            if (creatingNew || selected == null) "New project"
                            else "${selected.source.name} · ${selected.status.wireName}",
                            maxLines = 1,
                        )
                    }
                    DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                        DropdownMenuItem(
                            text = { Text("＋ Start a new project") },
                            onClick = { expanded = false; onNew() },
                        )
                        projects.forEach { project ->
                            DropdownMenuItem(
                                text = {
                                    Column {
                                        Text(project.source.name, maxLines = 1)
                                        Text(
                                            "${project.status.wireName} · ${project.id.removePrefix("project-")}",
                                            color = Muted,
                                            fontSize = 11.sp,
                                        )
                                    }
                                },
                                onClick = { expanded = false; onSelect(project) },
                            )
                        }
                    }
                }
                if (selected != null && !creatingNew) {
                    TextButton(onClick = onDelete) { Text("Delete", color = Danger) }
                }
            }
        }
    }
}

@Composable
private fun ProjectInferenceCard(
    project: NativeProject,
    state: InferenceUiState,
    onRetry: () -> Unit,
) {
    SectionCard("PROJECT INFERENCE", project.source.name) {
        Text(
            when (project.status) {
                ProjectStatus.QUEUED -> "Queued behind any active project"
                ProjectStatus.ANALYZING -> "Generating video and audio features in the background"
                ProjectStatus.ERROR -> "Inference stopped with an error"
                ProjectStatus.READY -> "Inference ready"
            },
            color = Muted,
        )
        if (project.status == ProjectStatus.ANALYZING || state.progress > 0f) {
            LinearProgressIndicator(progress = { state.progress }, modifier = Modifier.fillMaxWidth())
        }
        val detail = state.detail.ifBlank { project.error.orEmpty() }
        if (state.stage.isNotBlank() || detail.isNotBlank()) Text(
            "${state.stage.ifBlank { project.status.wireName }.uppercase(Locale.US)} · $detail",
            fontSize = 12.sp,
            fontFamily = FontFamily.Monospace,
            color = if (project.status == ProjectStatus.ERROR) Danger else Ink,
        )
        state.performance?.let { stats ->
            Text(
                String.format(
                    Locale.US,
                    "%,d / %,d source frames · %.1f feature frames/s · %.2fx realtime",
                    stats.decodedSourceFrames(), stats.totalFrames(),
                    stats.framesPerSecond(), stats.realtimeRatio(),
                ),
                fontFamily = FontFamily.Monospace,
                fontSize = 12.sp,
            )
            Text(
                "${secondsLabel(stats.elapsedSeconds())} elapsed · ${secondsLabel(stats.etaSeconds())} video ETA",
                fontFamily = FontFamily.Monospace,
                fontSize = 12.sp,
                color = Muted,
            )
        }
        Text(
            "Project ${project.id} · ${compactTime((project.media.durationSeconds() * 1_000).toLong())}",
            color = Muted,
            fontSize = 11.sp,
            fontFamily = FontFamily.Monospace,
        )
        if (project.status == ProjectStatus.ERROR) Button(onClick = onRetry) { Text("Retry") }
    }
}

@Composable
private fun NewProjectCard(
    selected: SourceSelection?,
    preparing: Boolean,
    state: InferenceUiState,
    useCache: Boolean,
    cacheBytes: Long,
    queueCount: Int,
    onSelect: () -> Unit,
    onQueue: () -> Unit,
    onUseCache: (Boolean) -> Unit,
    onClearCache: () -> Unit,
    onBenchmark: () -> Unit,
) {
    SectionCard(
        "NEW PROJECT",
        selected?.displayName ?: "Select a recording to create a persistent project",
    ) {
        Text(
            if (queueCount == 0) "Inference starts immediately."
            else "This waits behind $queueCount ${if (queueCount == 1) "project" else "projects"}; you can edit any ready project meanwhile.",
            color = Muted,
            fontSize = 12.sp,
        )
        Row(
            Modifier.horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            OutlinedButton(enabled = !preparing, onClick = onSelect) { Text("Choose video") }
            Button(enabled = selected != null && !preparing, onClick = onQueue) {
                Text(if (preparing) "Creating…" else "Create & queue")
            }
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("Reuse generated features", fontWeight = FontWeight.SemiBold)
                Text(
                    if (useCache) "Completed inference opens without running anything"
                    else "Regenerate this source's features for an experiment",
                    fontSize = 12.sp,
                    color = Muted,
                )
            }
            Switch(enabled = !preparing, checked = useCache, onCheckedChange = onUseCache)
        }
        if (preparing || state.stage.isNotBlank()) {
            val stageText = state.stage.ifBlank { "analysis" }.uppercase(Locale.US)
            Text("$stageText · ${state.detail}", fontSize = 12.sp, fontFamily = FontFamily.Monospace)
        }
        state.error?.let { Text(it, color = Danger, fontSize = 13.sp) }
        Row(
            Modifier.horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            TextButton(enabled = !preparing && queueCount == 0 && cacheBytes > 0, onClick = onClearCache) {
                Text("Clear ${formatBytes(cacheBytes)} feature cache")
            }
            TextButton(enabled = !preparing, onClick = onBenchmark) { Text("Benchmark tools") }
        }
    }
}

@Composable
private fun ProjectShell(
    projectControls: @Composable () -> Unit,
    content: @Composable ColumnScope.() -> Unit,
) {
    Scaffold(containerColor = Paper) { scaffoldPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(scaffoldPadding)
                .windowInsetsPadding(WindowInsets.statusBars)
                .windowInsetsPadding(WindowInsets.navigationBars)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 12.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            projectControls()
            content()
        }
    }
}

@OptIn(markerClass = [UnstableApi::class])
@Composable
private fun EditorScreen(
    activity: ComponentActivity,
    seed: EditorSeed,
    store: EditorDraftStore,
    initialDraft: EditorDraft,
    restored: Boolean,
    analysisRunning: Boolean,
    sourceControls: @Composable () -> Unit,
) {
    val context = LocalContext.current
    var draft by remember { mutableStateOf(initialDraft) }
    var selectedId by remember { mutableStateOf(initialDraft.cuts.firstOrNull()?.id.orEmpty()) }
    var focusLocked by remember { mutableStateOf(false) }
    var playbackPositionMs by remember { mutableLongStateOf(0L) }
    var isPlaying by remember { mutableStateOf(false) }
    var previewEndMs by remember { mutableStateOf<Long?>(null) }
    var message by remember {
        mutableStateOf(if (restored) "Restored saved edits on this device" else "New on-device draft")
    }
    var exportState by remember { mutableStateOf(ExportUiState()) }
    var confirmReset by remember { mutableStateOf(false) }

    val player = remember {
        ExoPlayer.Builder(context).build().apply {
            setSeekParameters(SeekParameters.EXACT)
            setMediaItem(MediaItem.fromUri(seed.sourceUri))
            prepare()
        }
    }
    val sortedCuts = draft.cuts.sortedWith(compareBy<EditableCut> { it.keepStartMs }.thenBy { it.keepEndMs })
    val selected = sortedCuts.firstOrNull { it.id == selectedId } ?: sortedCuts.firstOrNull()
    val selectedIndex = selected?.let { sortedCuts.indexOf(it) } ?: -1
    val finalIntervals = EditorMath.finalIntervals(draft)
    val effectiveIds = EditorMath.effectiveKeptIds(draft)
    val lowConfidence = sortedCuts.filter {
        it.origin == CutOrigin.INFERRED && it.included && it.id in effectiveIds &&
            it.confidence < draft.confidenceReviewThreshold
    }
    val totalFinalMs = EditorMath.totalFinalMs(finalIntervals)
    val removedCount = draft.cuts.count { !it.included }
    val ignoredCutCount = draft.cuts.count { it.included && it.id !in effectiveIds }
    val detailWindow = EditorMath.detailWindow(selected, playbackPositionMs, seed.durationMs)

    fun updateDraft(mutate: (EditorDraft) -> EditorDraft) {
        draft = mutate(draft).copy(updatedAtMs = System.currentTimeMillis())
    }

    fun updateCut(id: String, mutate: (EditableCut) -> EditableCut) {
        updateDraft { current ->
            current.copy(cuts = current.cuts.map { if (it.id == id) mutate(it) else it })
        }
    }

    fun seekTo(positionMs: Long) {
        val target = positionMs.coerceIn(0, seed.durationMs)
        playbackPositionMs = target
        player.seekTo(target)
    }

    fun setBoundary(side: String, valueMs: Long) {
        val cut = selected ?: return
        updateCut(cut.id) { current ->
            if (current.origin == CutOrigin.MANUAL) {
                if (side == "start") {
                    val start = valueMs.coerceIn(0, current.keepEndMs - MIN_MARK_MS)
                    current.copy(coreStartMs = start, keepStartMs = start)
                } else {
                    val end = valueMs.coerceIn(current.keepStartMs + MIN_MARK_MS, seed.durationMs)
                    current.copy(coreEndMs = end, keepEndMs = end)
                }
            } else if (side == "start") {
                current.copy(keepStartMs = valueMs.coerceIn(0, current.coreStartMs))
            } else {
                current.copy(keepEndMs = valueMs.coerceIn(current.coreEndMs, seed.durationMs))
            }
        }
    }

    val notificationPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { }
    val exportLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.CreateDocument("video/mp4"),
    ) { uri ->
        if (uri != null && finalIntervals.isNotEmpty()) {
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    uri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION,
                )
            }
            if (Build.VERSION.SDK_INT >= 33 &&
                context.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
            ) notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
            val exportIntent = Intent(context, ExportService::class.java).apply {
                putExtra(ExportService.EXTRA_SOURCE_URI, seed.sourceUri)
                putExtra(ExportService.EXTRA_SOURCE_NAME, seed.displayName)
                putExtra(ExportService.EXTRA_SOURCE_DURATION_MS, seed.durationMs)
                putExtra(ExportService.EXTRA_DESTINATION_URI, uri.toString())
                putExtra(ExportService.EXTRA_INTERVAL_STARTS, finalIntervals.map { it.startMs }.toLongArray())
                putExtra(ExportService.EXTRA_INTERVAL_ENDS, finalIntervals.map { it.endMs }.toLongArray())
            }
            context.startForegroundService(exportIntent)
            exportState = ExportUiState("running", 0, "Preparing export")
        }
    }
    val editListLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.CreateDocument("application/json"),
    ) { uri ->
        if (uri != null) runCatching {
            context.contentResolver.openOutputStream(uri, "w")!!.bufferedWriter().use {
                it.write(editListJson(seed, draft, finalIntervals).toString(2))
            }
        }.onSuccess { message = "Saved edit-list JSON" }
            .onFailure { message = it.message ?: "Could not save edit list" }
    }

    DisposableEffect(player) {
        val listener = object : Player.Listener {
            override fun onIsPlayingChanged(playing: Boolean) { isPlaying = playing }
            override fun onPlayerError(error: androidx.media3.common.PlaybackException) {
                message = error.message ?: "Video playback failed"
            }
        }
        player.addListener(listener)
        onDispose {
            player.removeListener(listener)
            player.release()
        }
    }

    DisposableEffect(activity, player) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_STOP) player.pause()
        }
        activity.lifecycle.addObserver(observer)
        onDispose { activity.lifecycle.removeObserver(observer) }
    }

    DisposableEffect(Unit) {
        val receiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context?, intent: Intent?) {
                if (intent?.action != ExportService.ACTION_PROGRESS) return
                exportState = ExportUiState(
                    status = intent.getStringExtra(ExportService.EXTRA_STATUS) ?: "running",
                    progress = intent.getIntExtra(ExportService.EXTRA_PROGRESS, 0),
                    detail = intent.getStringExtra(ExportService.EXTRA_DETAIL).orEmpty(),
                    metrics = intent.getStringExtra(ExportService.EXTRA_METRICS),
                )
            }
        }
        ContextCompat.registerReceiver(
            context,
            receiver,
            IntentFilter(ExportService.ACTION_PROGRESS),
            ContextCompat.RECEIVER_NOT_EXPORTED,
        )
        onDispose { runCatching { context.unregisterReceiver(receiver) } }
    }

    LaunchedEffect(draft) {
        delay(200)
        withContext(Dispatchers.IO) { store.save(draft) }
    }

    LaunchedEffect(draft.playbackRate) {
        player.playbackParameters = PlaybackParameters(draft.playbackRate)
    }

    LaunchedEffect(player, draft.finalPreviewEnabled, finalIntervals, previewEndMs, focusLocked) {
        while (true) {
            delay(33)
            val position = player.currentPosition.coerceAtLeast(0)
            playbackPositionMs = position
            if (!focusLocked) {
                EditorMath.playbackFocusCut(sortedCuts, position)?.let { reached ->
                    if (reached.id != selectedId) selectedId = reached.id
                }
            }
            previewEndMs?.let { end ->
                if (player.isPlaying && position >= end) {
                    player.pause()
                    previewEndMs = null
                }
            }
            if (draft.finalPreviewEnabled && player.isPlaying && previewEndMs == null) {
                val activeIndex = finalIntervals.indexOfLast { position >= it.startMs }
                when {
                    finalIntervals.isEmpty() -> player.pause()
                    activeIndex < 0 -> player.seekTo(finalIntervals.first().startMs)
                    position >= finalIntervals[activeIndex].endMs -> {
                        val next = finalIntervals.getOrNull(activeIndex + 1)
                        if (next == null) player.pause() else player.seekTo(next.startMs)
                    }
                }
            }
        }
    }

    DisposableEffect(isPlaying, exportState.status, analysisRunning) {
        if (isPlaying || exportState.status == "running" || analysisRunning) {
            activity.window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        } else activity.window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        onDispose { }
    }

    if (confirmReset) {
        AlertDialog(
            onDismissRequest = { confirmReset = false },
            title = { Text("Reset this edit?") },
            text = { Text("All boundary, keep/remove, manual-cut, and ignored-section changes will be discarded.") },
            confirmButton = {
                TextButton(onClick = {
                    store.clear()
                    draft = EditorMath.newDraft(seed)
                    selectedId = draft.cuts.firstOrNull()?.id.orEmpty()
                    confirmReset = false
                    message = "Restored inference ranges"
                }) { Text("Reset") }
            },
            dismissButton = { TextButton(onClick = { confirmReset = false }) { Text("Cancel") } },
        )
    }

    Scaffold(containerColor = Paper) { scaffoldPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(scaffoldPadding)
                .windowInsetsPadding(WindowInsets.statusBars)
                .windowInsetsPadding(WindowInsets.navigationBars)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            sourceControls()

            EditorHeader(seed, totalFinalMs, effectiveIds.size, removedCount, ignoredCutCount)

            SectionCard("OUTPUT PADDING", "Applied to inferred ranges only") {
                PaddingControl("Before", draft.beforePaddingMs) { before ->
                    updateDraft { EditorMath.applyPadding(it, before, it.afterPaddingMs, seed.durationMs) }
                }
                PaddingControl("After", draft.afterPaddingMs) { after ->
                    updateDraft { EditorMath.applyPadding(it, it.beforePaddingMs, after, seed.durationMs) }
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text("Play final cut only", fontWeight = FontWeight.SemiBold)
                        Text("Skip removed, ignored, and unselected time", fontSize = 12.sp, color = Muted)
                    }
                    Switch(
                        checked = draft.finalPreviewEnabled,
                        onCheckedChange = { enabled ->
                            previewEndMs = null
                            updateDraft { it.copy(finalPreviewEnabled = enabled) }
                            if (enabled) {
                                EditorMath.nextFinalTime(finalIntervals, player.currentPosition)?.let(::seekTo)
                            }
                        },
                    )
                }
            }

            Card(colors = CardDefaults.cardColors(containerColor = Color.Black)) {
                ContentFrame(
                    player = player,
                    modifier = Modifier.fillMaxWidth().height(176.dp),
                    surfaceType = SURFACE_TYPE_SURFACE_VIEW,
                )
            }
            PlayerControls(
                playing = isPlaying,
                positionMs = playbackPositionMs,
                durationMs = seed.durationMs,
                playbackRate = draft.playbackRate,
                onToggle = {
                    previewEndMs = null
                    if (player.isPlaying) player.pause() else {
                        if (draft.finalPreviewEnabled) {
                            EditorMath.nextFinalTime(finalIntervals, player.currentPosition)
                                ?.let { if (it != player.currentPosition) seekTo(it) }
                        }
                        player.play()
                    }
                },
                onSeekBy = { seekTo(player.currentPosition + it) },
                onRate = { rate -> updateDraft { it.copy(playbackRate = rate) } },
            )

            SectionCard("WHOLE RECORDING", "Tap a range · drag to scrub", compact = true) {
                ConfidenceControl(draft.confidenceReviewThreshold, lowConfidence.size, compact = true, onChange = { threshold ->
                    updateDraft { it.copy(confidenceReviewThreshold = threshold) }
                }, onReviewNext = {
                    if (lowConfidence.isNotEmpty()) {
                        val current = lowConfidence.indexOfFirst { it.id == selected?.id }
                        val next = if (current >= 0) lowConfidence[(current + 1) % lowConfidence.size]
                        else lowConfidence.firstOrNull { it.keepStartMs >= playbackPositionMs } ?: lowConfidence.first()
                        selectedId = next.id
                        seekTo(next.keepStartMs)
                        message = "${next.id} has ${(next.confidence * 100).roundToInt()}% confidence"
                    }
                })
                WholeTimeline(
                    windowStartMs = 0,
                    windowEndMs = seed.durationMs / 2,
                    cuts = sortedCuts,
                    ignored = draft.ignoredIntervals,
                    selectedId = selected?.id,
                    effectiveIds = effectiveIds,
                    confidenceThreshold = draft.confidenceReviewThreshold,
                    playheadMs = playbackPositionMs,
                    onSeek = { time, id ->
                        id?.let { selectedId = it }
                        seekTo(time)
                    },
                )
                TimelineLabels(0, seed.durationMs / 4, seed.durationMs / 2)
                WholeTimeline(
                    windowStartMs = seed.durationMs / 2,
                    windowEndMs = seed.durationMs,
                    cuts = sortedCuts,
                    ignored = draft.ignoredIntervals,
                    selectedId = selected?.id,
                    effectiveIds = effectiveIds,
                    confidenceThreshold = draft.confidenceReviewThreshold,
                    playheadMs = playbackPositionMs,
                    onSeek = { time, id ->
                        id?.let { selectedId = it }
                        seekTo(time)
                    },
                )
                TimelineLabels(seed.durationMs / 2, seed.durationMs * 3 / 4, seed.durationMs)
            }

            SectionCard(
                "FOCUSED RANGE",
                selected?.let { "${it.id} · ${if (it.origin == CutOrigin.MANUAL) "Manual" else "${(it.confidence * 100).roundToInt()}% confidence"}" }
                    ?: "No range selected",
                compact = true,
            ) {
                if (selected == null) {
                    Text("Add a missed cut at the current playhead to begin.", color = Muted)
                } else {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        SmallButton("Previous", enabled = selectedIndex > 0) {
                            sortedCuts.getOrNull(selectedIndex - 1)?.let { selectedId = it.id; seekTo(it.keepStartMs) }
                        }
                        Text("${selectedIndex + 1} / ${sortedCuts.size}", Modifier.padding(horizontal = 8.dp))
                        SmallButton("Next", enabled = selectedIndex < sortedCuts.lastIndex) {
                            sortedCuts.getOrNull(selectedIndex + 1)?.let { selectedId = it.id; seekTo(it.keepStartMs) }
                        }
                        SmallButton(if (selected.included) "Disable rally" else "Enable rally") {
                            updateCut(selected.id) { it.copy(included = !it.included) }
                        }
                        Spacer(Modifier.weight(1f))
                        Checkbox(checked = focusLocked, onCheckedChange = { focusLocked = it })
                        Text("Lock", fontSize = 13.sp)
                    }
                    TimelineLabels(detailWindow.startMs, (detailWindow.startMs + detailWindow.endMs) / 2, detailWindow.endMs)
                    FocusTimeline(
                        cut = selected,
                        window = detailWindow,
                        playheadMs = playbackPositionMs,
                        effective = selected.id in effectiveIds,
                        lowConfidence = selected.confidence < draft.confidenceReviewThreshold,
                        onSeek = ::seekTo,
                        onStartChange = { setBoundary("start", it) },
                        onEndChange = { setBoundary("end", it) },
                    )
                    BoundaryControls("Kept start", selected.keepStartMs) { delta ->
                        setBoundary("start", selected.keepStartMs + delta)
                    }
                    BoundaryControls("Kept end", selected.keepEndMs) { delta ->
                        setBoundary("end", selected.keepEndMs + delta)
                    }
                    Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedButton(onClick = {
                            updateDraft { it.copy(finalPreviewEnabled = false) }
                            seekTo(selected.keepStartMs)
                            previewEndMs = selected.keepEndMs
                            player.play()
                        }) { Text("Preview range") }
                        if (selected.origin == CutOrigin.INFERRED) {
                            OutlinedButton(onClick = {
                                updateCut(selected.id) { cut -> cut.copy(
                                    keepStartMs = (cut.coreStartMs - draft.beforePaddingMs).coerceAtLeast(0),
                                    keepEndMs = (cut.coreEndMs + draft.afterPaddingMs).coerceAtMost(seed.durationMs),
                                ) }
                            }) { Text("Reset padding") }
                        }
                        if (selected.origin == CutOrigin.MANUAL) {
                            OutlinedButton(onClick = {
                                val remaining = sortedCuts.filterNot { it.id == selected.id }
                                updateDraft { it.copy(cuts = it.cuts.filterNot { cut -> cut.id == selected.id }) }
                                selectedId = remaining.getOrNull((selectedIndex - 1).coerceAtLeast(0))?.id.orEmpty()
                            }) { Text("Delete", color = Danger) }
                        }
                    }
                }
            }

            MarkingTools(
                draft = draft,
                playbackPositionMs = playbackPositionMs,
                onDraft = ::updateDraft,
                onManualCompleted = { cut ->
                    selectedId = cut.id
                    message = "Added ${cut.id} from ${preciseTime(cut.keepStartMs)} to ${preciseTime(cut.keepEndMs)}"
                },
                onMessage = { message = it },
            )

            if (message.isNotBlank()) {
                Surface(color = Color(0xFFFFE2D8), shape = RoundedCornerShape(10.dp)) {
                    Text(message, Modifier.padding(12.dp), fontSize = 13.sp)
                }
            }

            if (draft.ignoredIntervals.isNotEmpty()) {
                SectionCard("IGNORED SOURCE", "Removed from the derived edit list") {
                    draft.ignoredIntervals.sortedBy { it.startMs }.forEach { interval ->
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            TextButton(onClick = { seekTo(interval.startMs) }) {
                                Text("${preciseTime(interval.startMs)}–${preciseTime(interval.endMs)}")
                            }
                            Text(interval.reason.replace('-', ' '), Modifier.weight(1f), fontSize = 12.sp)
                            TextButton(onClick = {
                                updateDraft { it.copy(ignoredIntervals = it.ignoredIntervals.filterNot { item -> item.id == interval.id }) }
                            }) { Text("Remove", color = Danger) }
                        }
                    }
                }
            }

            SectionCard("ALL CUTS", "${effectiveIds.size} kept · $ignoredCutCount fully ignored") {
                sortedCuts.forEachIndexed { index, cut ->
                    val state = when {
                        !cut.included -> "Removed"
                        cut.id !in effectiveIds -> "Ignored"
                        else -> "Keep"
                    }
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(8.dp))
                            .background(if (cut.id == selected?.id) Color(0xFFFFEEE8) else Color.Transparent)
                            .clickable { selectedId = cut.id; seekTo(cut.keepStartMs) }
                            .padding(vertical = 7.dp, horizontal = 8.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text((index + 1).toString().padStart(2, '0'), color = Muted, fontFamily = FontFamily.Monospace)
                        Column(Modifier.weight(1f).padding(horizontal = 10.dp)) {
                            Text(cut.id, fontWeight = FontWeight.SemiBold)
                            Text("${preciseTime(cut.keepStartMs)}–${preciseTime(cut.keepEndMs)}", fontSize = 12.sp, color = Muted)
                        }
                        Text(
                            if (cut.origin == CutOrigin.MANUAL) "MANUAL" else "${(cut.confidence * 100).roundToInt()}%",
                            fontSize = 12.sp,
                            color = if (cut.confidence < draft.confidenceReviewThreshold) Warning else Muted,
                        )
                        TextButton(onClick = { updateCut(cut.id) { it.copy(included = !it.included) } }) {
                            Text(state, color = if (state == "Keep") Green else Danger)
                        }
                    }
                    if (index < sortedCuts.lastIndex) HorizontalDivider(color = Rail)
                }
            }

            SectionCard("EXPORT", "Exact intervals · AVC video · AAC audio · MP4") {
                Text(
                    "${finalIntervals.size} merged ranges · ${compactTime(totalFinalMs)} output",
                    fontWeight = FontWeight.SemiBold,
                )
                if (exportState.status == "running") {
                    LinearProgressIndicator(
                        progress = { exportState.progress / 100f },
                        modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp),
                    )
                    Text(exportState.detail, fontFamily = FontFamily.Monospace, fontSize = 12.sp)
                } else if (exportState.detail.isNotBlank()) {
                    Text(exportState.detail, color = if (exportState.status == "failed") Danger else Green)
                    exportState.metrics?.let { Text(it, fontFamily = FontFamily.Monospace, fontSize = 10.sp, color = Muted) }
                }
                Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        enabled = finalIntervals.isNotEmpty() && exportState.status != "running",
                        onClick = { exportLauncher.launch(exportFilename(seed.displayName)) },
                    ) { Text("Export MP4") }
                    OutlinedButton(
                        enabled = finalIntervals.isNotEmpty(),
                        onClick = { editListLauncher.launch(editListFilename(seed.displayName)) },
                    ) { Text("Save edit list") }
                    if (exportState.status == "running") {
                        OutlinedButton(onClick = {
                            context.startService(Intent(context, ExportService::class.java).setAction(ExportService.ACTION_CANCEL))
                        }) { Text("Cancel export", color = Danger) }
                    }
                }
            }

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = { confirmReset = true }) { Text("Reset editor") }
                TextButton(onClick = { context.startActivity(Intent(context, MainActivity::class.java)) }) {
                    Text("Benchmark tools")
                }
            }
            Spacer(Modifier.height(20.dp))
        }
    }
}

@Composable
private fun EditorHeader(seed: EditorSeed, outputMs: Long, kept: Int, removed: Int, ignored: Int) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text("VOLLEYCUT", color = Orange, fontWeight = FontWeight.Black, letterSpacing = 2.sp)
        Text("Native cut editor", fontSize = 28.sp, fontWeight = FontWeight.Bold)
        Text(seed.displayName, color = Muted, fontSize = 13.sp, maxLines = 2)
        Text(
            "${seed.width}×${seed.height} · ${compactTime(seed.durationMs)} source · ${compactTime(outputMs)} output",
            fontFamily = FontFamily.Monospace,
            fontSize = 12.sp,
        )
        Text("$kept kept · $removed removed · $ignored fully ignored", fontSize = 12.sp, color = Muted)
    }
}

@Composable
private fun SectionCard(
    label: String,
    subtitle: String,
    compact: Boolean = false,
    content: @Composable ColumnScope.() -> Unit,
) {
    Card(colors = CardDefaults.cardColors(containerColor = Color.White)) {
        Column(
            Modifier.fillMaxWidth().padding(if (compact) 10.dp else 14.dp),
            verticalArrangement = Arrangement.spacedBy(if (compact) 4.dp else 10.dp),
        ) {
            if (compact) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        label,
                        Modifier.weight(1f),
                        color = Orange,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Black,
                        letterSpacing = .7.sp,
                    )
                    Text(subtitle, color = Muted, fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
                }
            } else {
                Text(label, color = Orange, fontSize = 12.sp, fontWeight = FontWeight.Black, letterSpacing = 1.sp)
                Text(subtitle, fontWeight = FontWeight.SemiBold)
            }
            content()
        }
    }
}

@Composable
private fun PlayerControls(
    playing: Boolean,
    positionMs: Long,
    durationMs: Long,
    playbackRate: Float,
    onToggle: () -> Unit,
    onSeekBy: (Long) -> Unit,
    onRate: (Float) -> Unit,
) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(preciseTime(positionMs), fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
            Text(" / ${preciseTime(durationMs)}", fontFamily = FontFamily.Monospace, color = Muted)
            Spacer(Modifier.weight(1f))
            Button(onClick = onToggle) { Text(if (playing) "Pause" else "Play") }
        }
        Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            listOf(-1_000L to "−1s", -100L to "−0.1s", 100L to "+0.1s", 1_000L to "+1s")
                .forEach { (delta, label) -> SmallButton(label) { onSeekBy(delta) } }
            Spacer(Modifier.width(8.dp))
            listOf(1f, 2f, 4f, 8f).forEach { rate ->
                FilterChip(selected = playbackRate == rate, onClick = { onRate(rate) }, label = { Text("${rate.toInt()}x") })
            }
        }
    }
}

@Composable
private fun PaddingControl(label: String, valueMs: Long, onChange: (Long) -> Unit) {
    Column {
        Row {
            Text(label, Modifier.weight(1f), fontSize = 13.sp)
            Text(String.format(Locale.US, "%.1fs", valueMs / 1_000.0), fontFamily = FontFamily.Monospace)
        }
        Slider(
            value = valueMs / 1_000f,
            onValueChange = { value -> onChange((value * 2).roundToLong() * 500) },
            valueRange = 0f..10f,
            steps = 19,
        )
    }
}

@Composable
private fun ConfidenceControl(
    value: Float,
    count: Int,
    compact: Boolean = false,
    onChange: (Float) -> Unit,
    onReviewNext: () -> Unit,
) {
    if (compact) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Low <${(value * 100).roundToInt()}%", fontSize = 11.sp)
            Slider(
                value = value,
                onValueChange = onChange,
                modifier = Modifier.weight(1f).height(32.dp),
                valueRange = 0f..1f,
                steps = 99,
            )
            SmallButton("Review $count", enabled = count > 0, onClick = onReviewNext)
        }
    } else {
        Column {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Highlight below ${(value * 100).roundToInt()}%", Modifier.weight(1f), fontSize = 13.sp)
                Text("$count ranges", color = Muted, fontSize = 12.sp)
                TextButton(enabled = count > 0, onClick = onReviewNext) { Text("Review next") }
            }
            Slider(value = value, onValueChange = onChange, valueRange = 0f..1f, steps = 99)
        }
    }
}

@Composable
private fun WholeTimeline(
    windowStartMs: Long,
    windowEndMs: Long,
    cuts: List<EditableCut>,
    ignored: List<IgnoredSourceInterval>,
    selectedId: String?,
    effectiveIds: Set<String>,
    confidenceThreshold: Float,
    playheadMs: Long,
    onSeek: (Long, String?) -> Unit,
) {
    BoxWithConstraints(Modifier.fillMaxWidth()) {
        val density = LocalDensity.current
        val widthPx = with(density) { maxWidth.toPx() }.coerceAtLeast(1f)
        val windowSpanMs = (windowEndMs - windowStartMs).coerceAtLeast(1)
        fun timeAt(x: Float) = windowStartMs +
            ((x / widthPx).coerceIn(0f, 1f) * windowSpanMs).roundToLong()
        fun xAt(timeMs: Long) = (timeMs - windowStartMs).toFloat() / windowSpanMs * widthPx
        Canvas(
            Modifier
                .fillMaxWidth()
                .height(34.dp)
                .clip(RoundedCornerShape(9.dp))
                .background(Rail)
                .pointerInput(windowStartMs, windowEndMs, cuts) {
                    detectTapGestures { offset ->
                        val time = timeAt(offset.x)
                        val cut = cuts.lastOrNull { time in it.keepStartMs..it.keepEndMs }
                        onSeek(time, cut?.id)
                    }
                }
                .pointerInput(windowStartMs, windowEndMs) {
                    detectHorizontalDragGestures(
                        onDragStart = { onSeek(timeAt(it.x), null) },
                        onHorizontalDrag = { change, _ -> change.consume(); onSeek(timeAt(change.position.x), null) },
                    )
                },
        ) {
            ignored.forEach { interval ->
                val clippedStart = max(interval.startMs, windowStartMs)
                val clippedEnd = min(interval.endMs, windowEndMs)
                if (clippedEnd <= clippedStart) return@forEach
                val x = xAt(clippedStart)
                val width = max(1f, xAt(clippedEnd) - x)
                drawRect(Danger.copy(alpha = .26f), Offset(x, 0f), Size(width, size.height))
            }
            cuts.forEach { cut ->
                val clippedStart = max(cut.keepStartMs, windowStartMs)
                val clippedEnd = min(cut.keepEndMs, windowEndMs)
                if (clippedEnd <= clippedStart) return@forEach
                val x = xAt(clippedStart)
                val width = max(2f, xAt(clippedEnd) - x)
                val low = cut.origin == CutOrigin.INFERRED && cut.confidence < confidenceThreshold
                val color = when {
                    !cut.included -> Muted.copy(alpha = .55f)
                    cut.id !in effectiveIds -> Danger.copy(alpha = .65f)
                    low -> Warning
                    else -> PaleGreen
                }
                drawRoundRect(color, Offset(x, 16f), Size(width, size.height - 32f), CornerRadius(6f))
                val clippedCoreStart = max(cut.coreStartMs, windowStartMs)
                val clippedCoreEnd = min(cut.coreEndMs, windowEndMs)
                if (clippedCoreEnd > clippedCoreStart) {
                    val coreX = xAt(clippedCoreStart)
                    val coreWidth = max(1f, xAt(clippedCoreEnd) - coreX)
                    drawRoundRect(if (cut.included) Green else Muted, Offset(coreX, 24f), Size(coreWidth, size.height - 48f), CornerRadius(4f))
                }
                if (cut.id == selectedId) {
                    drawRoundRect(Orange, Offset(x, 13f), Size(width, size.height - 26f), CornerRadius(7f), style = Stroke(4f))
                }
            }
            if (playheadMs in windowStartMs..windowEndMs) {
                val playheadX = xAt(playheadMs)
                drawLine(Orange, Offset(playheadX, 0f), Offset(playheadX, size.height), 4f, StrokeCap.Round)
            }
        }
    }
}

@Composable
private fun FocusTimeline(
    cut: EditableCut,
    window: DetailWindow,
    playheadMs: Long,
    effective: Boolean,
    lowConfidence: Boolean,
    onSeek: (Long) -> Unit,
    onStartChange: (Long) -> Unit,
    onEndChange: (Long) -> Unit,
) {
    BoxWithConstraints(Modifier.fillMaxWidth().height(44.dp)) {
        val density = LocalDensity.current
        val widthPx = with(density) { maxWidth.toPx() }.coerceAtLeast(1f)
        val span = (window.endMs - window.startMs).coerceAtLeast(1)
        fun xAt(time: Long) = (time - window.startMs).toFloat() / span * widthPx
        fun timeAt(x: Float) = window.startMs + ((x / widthPx).coerceIn(0f, 1f) * span).roundToLong()
        Canvas(
            Modifier
                .fillMaxSize()
                .clip(RoundedCornerShape(10.dp))
                .background(Rail)
                .pointerInput(window) { detectTapGestures { onSeek(timeAt(it.x)) } }
                .pointerInput(window) {
                    detectHorizontalDragGestures(
                        onDragStart = { onSeek(timeAt(it.x)) },
                        onHorizontalDrag = { change, _ -> change.consume(); onSeek(timeAt(change.position.x)) },
                    )
                },
        ) {
            val keptColor = when {
                !cut.included || !effective -> Danger.copy(alpha = .6f)
                lowConfidence -> Warning.copy(alpha = .8f)
                else -> PaleGreen
            }
            drawRoundRect(
                keptColor,
                Offset(xAt(cut.keepStartMs), 18f),
                Size(max(2f, xAt(cut.keepEndMs) - xAt(cut.keepStartMs)), size.height - 36f),
                CornerRadius(8f),
            )
            drawRoundRect(
                if (cut.included) Green else Muted,
                Offset(xAt(cut.coreStartMs), 33f),
                Size(max(2f, xAt(cut.coreEndMs) - xAt(cut.coreStartMs)), size.height - 66f),
                CornerRadius(5f),
            )
            val playheadX = xAt(playheadMs).coerceIn(0f, size.width)
            drawLine(Orange, Offset(playheadX, 0f), Offset(playheadX, size.height), 4f)
        }
        val handleWidth = 20.dp
        val handlePx = with(density) { handleWidth.toPx() }
        val deltaToMs = { delta: Float -> (delta / widthPx * span).roundToLong() }
        Box(
            Modifier
                .offset { IntOffset((xAt(cut.keepStartMs) - handlePx / 2).roundToInt(), 0) }
                .width(handleWidth)
                .fillMaxHeight()
                .semantics { contentDescription = "Adjust kept start at ${preciseTime(cut.keepStartMs)}" }
                .draggable(
                    orientation = Orientation.Horizontal,
                    state = rememberDraggableState { delta -> onStartChange(cut.keepStartMs + deltaToMs(delta)) },
                )
                .border(3.dp, Orange, RoundedCornerShape(8.dp)),
        )
        Box(
            Modifier
                .offset { IntOffset((xAt(cut.keepEndMs) - handlePx / 2).roundToInt(), 0) }
                .width(handleWidth)
                .fillMaxHeight()
                .semantics { contentDescription = "Adjust kept end at ${preciseTime(cut.keepEndMs)}" }
                .draggable(
                    orientation = Orientation.Horizontal,
                    state = rememberDraggableState { delta -> onEndChange(cut.keepEndMs + deltaToMs(delta)) },
                )
                .border(3.dp, Orange, RoundedCornerShape(8.dp)),
        )
    }
}

@Composable
private fun TimelineLabels(start: Long, middle: Long, end: Long) {
    Row(Modifier.fillMaxWidth()) {
        Text(compactTime(start), fontFamily = FontFamily.Monospace, fontSize = 10.sp, color = Muted)
        Spacer(Modifier.weight(1f))
        Text(compactTime(middle), fontFamily = FontFamily.Monospace, fontSize = 10.sp, color = Muted)
        Spacer(Modifier.weight(1f))
        Text(compactTime(end), fontFamily = FontFamily.Monospace, fontSize = 10.sp, color = Muted)
    }
}

@Composable
private fun BoundaryControls(label: String, valueMs: Long, onNudge: (Long) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.width(104.dp)) {
            Text(label, fontSize = 12.sp, color = Muted)
            Text(preciseTime(valueMs), fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
        }
        Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(5.dp)) {
            listOf(-1_000L to "−1s", -100L to "−0.1s", 100L to "+0.1s", 1_000L to "+1s")
                .forEach { (delta, labelText) -> SmallButton(labelText) { onNudge(delta) } }
        }
    }
}

@Composable
private fun MarkingTools(
    draft: EditorDraft,
    playbackPositionMs: Long,
    onDraft: ((EditorDraft) -> EditorDraft) -> Unit,
    onManualCompleted: (EditableCut) -> Unit,
    onMessage: (String) -> Unit,
) {
    SectionCard("ADD A MISSED CUT", draft.pendingManualStartMs?.let { "Started ${preciseTime(it)}" } ?: "Find the first frame") {
        Text("Seek, mark the start, then seek and mark the end.", fontSize = 12.sp, color = Muted)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = {
                val startMark = draft.pendingManualStartMs
                if (startMark == null) {
                    onDraft { it.copy(pendingManualStartMs = playbackPositionMs, pendingIgnoreStartMs = null) }
                    onMessage("Missed cut starts at ${preciseTime(playbackPositionMs)}")
                } else {
                    val start = min(startMark, playbackPositionMs)
                    val end = max(startMark, playbackPositionMs)
                    if (end - start < MIN_MARK_MS) onMessage("A manual cut must be at least 0.1 seconds")
                    else {
                        val cut = EditableCut(
                            id = EditorMath.nextId("M", draft.cuts.map { it.id }),
                            coreStartMs = start, coreEndMs = end,
                            keepStartMs = start, keepEndMs = end,
                            confidence = 1f, included = true, origin = CutOrigin.MANUAL,
                        )
                        onDraft { it.copy(pendingManualStartMs = null, cuts = it.cuts + cut) }
                        onManualCompleted(cut)
                    }
                }
            }) { Text(if (draft.pendingManualStartMs == null) "Mark start · ${preciseTime(playbackPositionMs)}" else "Mark end · ${preciseTime(playbackPositionMs)}") }
            if (draft.pendingManualStartMs != null) {
                OutlinedButton(onClick = { onDraft { it.copy(pendingManualStartMs = null) } }) { Text("Cancel") }
            }
        }
    }
    SectionCard("IGNORE SOURCE SECTION", draft.pendingIgnoreStartMs?.let { "Started ${preciseTime(it)}" } ?: "Exclude unusable footage") {
        Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(5.dp)) {
            listOf("non-game-content", "camera-gap", "partial-rally", "boundary-ambiguous").forEach { reason ->
                FilterChip(
                    selected = draft.ignoreReason == reason,
                    onClick = { onDraft { it.copy(ignoreReason = reason) } },
                    label = { Text(reason.replace('-', ' '), fontSize = 11.sp) },
                )
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = {
                val startMark = draft.pendingIgnoreStartMs
                if (startMark == null) {
                    onDraft { it.copy(pendingIgnoreStartMs = playbackPositionMs, pendingManualStartMs = null) }
                    onMessage("Ignored section starts at ${preciseTime(playbackPositionMs)}")
                } else {
                    val start = min(startMark, playbackPositionMs)
                    val end = max(startMark, playbackPositionMs)
                    if (end - start < MIN_MARK_MS) onMessage("An ignored section must be at least 0.1 seconds")
                    else {
                        val ignored = IgnoredSourceInterval(
                            EditorMath.nextId("I", draft.ignoredIntervals.map { it.id }),
                            start, end, draft.ignoreReason,
                        )
                        onDraft { it.copy(pendingIgnoreStartMs = null, ignoredIntervals = it.ignoredIntervals + ignored) }
                        onMessage("Ignored ${preciseTime(start)} to ${preciseTime(end)}")
                    }
                }
            }) { Text(if (draft.pendingIgnoreStartMs == null) "Mark start · ${preciseTime(playbackPositionMs)}" else "Mark end · ${preciseTime(playbackPositionMs)}") }
            if (draft.pendingIgnoreStartMs != null) {
                OutlinedButton(onClick = { onDraft { it.copy(pendingIgnoreStartMs = null) } }) { Text("Cancel") }
            }
        }
    }
}

@Composable
private fun SmallButton(label: String, enabled: Boolean = true, onClick: () -> Unit) {
    OutlinedButton(enabled = enabled, onClick = onClick, contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 10.dp, vertical = 5.dp)) {
        Text(label, fontSize = 12.sp)
    }
}

private fun editListJson(seed: EditorSeed, draft: EditorDraft, intervals: List<FinalCutInterval>) =
    JSONObject().apply {
        put("schemaVersion", 1)
        put("method", "android-native-editor-v1")
        put("sourceName", seed.displayName)
        put("sourceUri", seed.sourceUri)
        put("sourceDuration", seed.durationMs / 1_000.0)
        put("sourceRevision", seed.sourceRevision)
        put("beforePaddingSeconds", draft.beforePaddingMs / 1_000.0)
        put("afterPaddingSeconds", draft.afterPaddingMs / 1_000.0)
        put("outputDuration", EditorMath.totalFinalMs(intervals) / 1_000.0)
        put("ranges", JSONArray().apply {
            intervals.forEach { range -> put(JSONObject().apply {
                put("start", range.startMs / 1_000.0)
                put("end", range.endMs / 1_000.0)
                put("cutIds", JSONArray(range.cutIds))
            }) }
        })
        put("cuts", JSONArray().apply {
            draft.cuts.forEach { cut -> put(JSONObject().apply {
                put("id", cut.id)
                put("coreStart", cut.coreStartMs / 1_000.0)
                put("coreEnd", cut.coreEndMs / 1_000.0)
                put("keepStart", cut.keepStartMs / 1_000.0)
                put("keepEnd", cut.keepEndMs / 1_000.0)
                put("confidence", cut.confidence.toDouble())
                put("included", cut.included)
                put("origin", cut.origin.name.lowercase())
            }) }
        })
        put("ignoredIntervals", JSONArray().apply {
            draft.ignoredIntervals.forEach { interval -> put(JSONObject().apply {
                put("id", interval.id)
                put("start", interval.startMs / 1_000.0)
                put("end", interval.endMs / 1_000.0)
                put("reason", interval.reason)
            }) }
        })
    }

private fun exportFilename(sourceName: String): String {
    val base = sourceName.substringBeforeLast('.').replace(Regex("[^A-Za-z0-9._-]+"), "-").trim('-')
    return "${base.ifBlank { "volleycut" }}-cut.mp4"
}

private fun editListFilename(sourceName: String): String = exportFilename(sourceName).removeSuffix(".mp4") + ".edit-list.json"

private fun sourceDisplayName(context: Context, uri: Uri): String {
    runCatching {
        context.contentResolver.query(
            uri,
            arrayOf(OpenableColumns.DISPLAY_NAME),
            null,
            null,
            null,
        )?.use { cursor ->
            val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (index >= 0 && cursor.moveToFirst()) return cursor.getString(index)
        }
    }
    return uri.lastPathSegment ?: "selected-video"
}

private fun formatBytes(bytes: Long): String = when {
    bytes >= 1_073_741_824 -> String.format(Locale.US, "%.1f GiB", bytes / 1_073_741_824.0)
    bytes >= 1_048_576 -> String.format(Locale.US, "%.1f MiB", bytes / 1_048_576.0)
    bytes >= 1_024 -> String.format(Locale.US, "%.1f KiB", bytes / 1_024.0)
    else -> "$bytes B"
}

private fun secondsLabel(seconds: Double): String =
    if (!seconds.isFinite() || seconds < 0) "calculating…"
    else compactTime((seconds * 1_000).roundToLong())
