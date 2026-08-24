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
import androidx.compose.foundation.Image
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
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
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
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RangeSlider
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
import androidx.compose.runtime.mutableIntStateOf
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
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
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
import androidx.media3.ui.compose.SURFACE_TYPE_TEXTURE_VIEW
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale
import java.util.UUID
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
        val agreements = intent.getStringArrayExtra(EXTRA_RANGE_AGREEMENTS)
        if (durationMs <= 0 || starts.size != ends.size || starts.size != confidence.size) return null
        return EditorSeed(
            sourceUri = sourceUri,
            displayName = intent.getStringExtra(EXTRA_DISPLAY_NAME) ?: "recording.mp4",
            durationMs = durationMs,
            width = intent.getIntExtra(EXTRA_WIDTH, 0),
            height = intent.getIntExtra(EXTRA_HEIGHT, 0),
            rotation = intent.getIntExtra(EXTRA_ROTATION, 0),
            ranges = starts.indices.map {
                SeedRange(
                    starts[it], ends[it], confidence[it],
                    agreements?.getOrNull(it)?.takeIf(String::isNotBlank),
                )
            },
            gameStartMs = intent.getLongExtra(EXTRA_GAME_START_MS, 0),
            gameEndMs = intent.getLongExtra(EXTRA_GAME_END_MS, durationMs),
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
        private const val EXTRA_RANGE_AGREEMENTS = "editor_range_agreements"
        private const val EXTRA_GAME_START_MS = "editor_game_start_ms"
        private const val EXTRA_GAME_END_MS = "editor_game_end_ms"

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
                putExtra(EXTRA_RANGE_AGREEMENTS, seed.ranges.map { it.agreement.orEmpty() }.toTypedArray())
                putExtra(EXTRA_GAME_START_MS, seed.gameStartMs)
                putExtra(EXTRA_GAME_END_MS, seed.gameEndMs)
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
        SeedRange(secondsToMs(it.start()), secondsToMs(it.end()), it.confidence(), it.agreement())
    },
    productionComponents = result.productionComponents(),
    productionServeOutputs = result.productionServeOutputs(),
    productionStateOutputs = result.productionStateOutputs(),
    servingSide = result.servingSide(),
    servingSideError = result.servingSideError(),
    sideSwitch = result.sideSwitch(),
    sideSwitchError = result.sideSwitchError(),
    suppression = result.suppression(),
)

private val Paper = Color(0xFFF7F4EE)
private val Ink = Color(0xFF20201E)
private val Orange = Color(0xFFEF5B35)
private val Green = Color(0xFF26734D)
private val PaleGreen = Color(0xFF9ED5B5)
private val Muted = Color(0xFF77736C)
private val Rail = Color(0xFFE4E0D7)
private val Danger = Color(0xFFB3261E)
private val SuppressionRed = Color(0xFFD1242F)
private val Warning = Color(0xFFE8A317)

private fun EditableCut.isModelDisagreement(): Boolean =
    ProductionEnsemble.isDisagreement(agreement)

private fun EditableCut.modelAgreementLabel(): String = when (agreement) {
    ProductionEnsemble.BOTH_MODELS -> "Both models agree"
    ProductionEnsemble.ALL_LABELS_V2_ONLY -> "Check · all-labels v2 only"
    ProductionEnsemble.PREVIOUS_PRODUCTION_ONLY -> "Check · previous model only"
    else -> "Model prediction"
}

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
    val jobId: String? = null,
    val status: String = "idle",
    val progress: Int = 0,
    val detail: String = "",
)

private fun ExportJobStatus.toUiState() = ExportUiState(
    jobId = jobId,
    status = status,
    progress = progress,
    detail = detail,
)

private fun projectStatusLabel(
    project: NativeProject,
    export: ExportJobStatus?,
): String {
    val encoding = when (export?.status) {
        "queued" -> "encoding queued"
        "running" -> "encoding ${export.progress.coerceIn(0, 100)}%"
        "complete" -> "encoding finished"
        "failed" -> "encoding failed"
        "cancelled" -> "encoding cancelled"
        else -> null
    }
    return listOfNotNull(project.status.wireName, encoding).joinToString(" · ")
}

private data class SourceSelection(
    val uri: Uri,
    val displayName: String,
    val source: ProjectSource? = null,
    val media: AnalysisTypes.MediaInfo? = null,
    val roi: AnalysisTypes.Roi? = null,
)

private data class InferenceUiState(
    val projectId: String? = null,
    val running: Boolean = false,
    val servingSideJob: Boolean = false,
    val progress: Float = 0f,
    val stage: String = "",
    val detail: String = "",
    val performance: AnalysisTypes.PerformanceStats? = null,
    val stepMeasurements: List<InferenceStepMeasurement> = emptyList(),
    val error: String? = null,
)

private data class EditorProjectSummary(
    val width: Int,
    val height: Int,
    val sourceDurationMs: Long,
    val outputDurationMs: Long,
    val kept: Int,
    val removed: Int,
    val ignored: Int,
)

private fun setScoreTrackingPreference(
    context: Context,
    project: NativeProject,
    enabled: Boolean,
) {
    val seed = project.editorSeed() ?: return
    val store = EditorDraftStore(context, seed)
    val draft = store.load() ?: EditorMath.newDraft(seed)
    store.save(draft.copy(
        scoreTracking = draft.scoreTracking.copy(enabled = enabled),
        updatedAtMs = System.currentTimeMillis(),
    ))
}

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
    var analyzeServingSide by remember { mutableStateOf(true) }
    var generateSideSwitchMarkers by remember { mutableStateOf(false) }
    var cacheBytes by remember { mutableLongStateOf(NativeFeatureCache.totalBytes(context)) }
    var confirmDelete by remember { mutableStateOf<NativeProject?>(null) }
    var gameStartMs by remember { mutableLongStateOf(0L) }
    var gameEndMs by remember { mutableLongStateOf(0L) }
    val selectedProject = projects.firstOrNull { it.id == selectedProjectId }
    var sourceAvailable by remember(selectedProject?.id, selectedProject?.source?.uri) {
        mutableStateOf<Boolean?>(null)
    }
    var relinkTargetId by remember { mutableStateOf<String?>(null) }
    var relinkingSource by remember { mutableStateOf(false) }
    var relinkMessage by remember { mutableStateOf<String?>(null) }
    var relinkFailed by remember { mutableStateOf(false) }
    var sourceCheckNonce by remember { mutableLongStateOf(0L) }
    var exportQueueCount by remember { mutableIntStateOf(ExportService.pendingCount()) }
    var exportStatuses by remember {
        mutableStateOf(
            initialProjects.mapNotNull { project ->
                ExportService.statusForProject(project.id)?.let { project.id to it }
            }.toMap(),
        )
    }
    var timeoutSnapshot by remember {
        mutableStateOf(ProcessingTimeoutTracker.snapshot(context))
    }

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
            gameStartMs = 0
            gameEndMs = 0
            preparingSource = true
            inference = InferenceUiState(stage = "opening", detail = "Reading recording metadata")
            scope.launch {
                try {
                    val prepared = withContext(Dispatchers.IO) {
                        val engine = AnalysisEngine(context)
                        val media = engine.probe(uri)
                        val source = NativeProjectStore.source(
                            context,
                            uri,
                            sourceDisplayName(context, uri),
                        )
                        SourceSelection(
                            uri = uri,
                            displayName = source.name,
                            source = source,
                            media = media,
                            roi = AnalysisEngine.inferRoi(source.name),
                        )
                    }
                    selectedSource = prepared
                    gameStartMs = 0
                    gameEndMs = secondsToMs(checkNotNull(prepared.media).durationSeconds())
                    inference = InferenceUiState(detail = "Mark the game start and end, then queue inference")
                } catch (error: Exception) {
                    inference = InferenceUiState(
                        stage = "failed",
                        error = error.message ?: error.javaClass.simpleName,
                    )
                } finally {
                    preparingSource = false
                }
            }
        }
    }
    val feedbackImportPicker = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument(),
    ) { uri ->
        if (uri != null) {
            preparingSource = true
            inference = InferenceUiState(stage = "importing", detail = "Validating model feedback")
            scope.launch {
                val imported = runCatching {
                    withContext(Dispatchers.IO) {
                        val text = context.contentResolver.openInputStream(uri)
                            ?.bufferedReader()?.use { it.readText() }
                            ?: error("Could not open model feedback")
                        ModelFeedbackImporter.import(context, text)
                    }
                }
                preparingSource = false
                imported.onSuccess { project ->
                    reloadProjects(project.id, preserveNewProject = false)
                    creatingNew = false
                    inference = InferenceUiState(
                        projectId = project.id,
                        detail = "Feedback imported · re-link the original video for playback",
                    )
                }.onFailure { error ->
                    inference = InferenceUiState(
                        stage = "failed",
                        error = error.message ?: "Could not import model feedback",
                    )
                }
            }
        }
    }
    val relinkPicker = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument(),
    ) { uri ->
        val target = relinkTargetId?.let { id -> projects.firstOrNull { it.id == id } }
        relinkTargetId = null
        if (uri != null && target != null) {
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    uri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION,
                )
            }
            relinkingSource = true
            relinkMessage = "Validating replacement recording…"
            relinkFailed = false
            scope.launch {
                val result = runCatching {
                    withContext(Dispatchers.IO) {
                        val engine = AnalysisEngine(context)
                        val media = engine.probe(uri)
                        val source = NativeProjectStore.source(
                            context,
                            uri,
                            sourceDisplayName(context, uri),
                        )
                        NativeProjectStore.relink(context, target, source, media)
                    }
                }
                relinkingSource = false
                result.onSuccess { updated ->
                    projects = NativeProjectStore.list(context)
                    selectedProjectId = updated.id
                    sourceAvailable = true
                    relinkMessage = "Re-linked ${updated.source.name}; saved edits and inference were preserved."
                    relinkFailed = false
                }.onFailure { error ->
                    sourceAvailable = false
                    relinkMessage = error.message ?: "Could not re-link that recording."
                    relinkFailed = true
                }
            }
        }
    }
    val notificationPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { }

    DisposableEffect(activity) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_RESUME) sourceCheckNonce++
        }
        activity.lifecycle.addObserver(observer)
        onDispose { activity.lifecycle.removeObserver(observer) }
    }

    LaunchedEffect(selectedProject?.id, selectedProject?.source?.uri, sourceCheckNonce) {
        val project = selectedProject
        sourceAvailable = if (project == null) null else withContext(Dispatchers.IO) {
            NativeProjectStore.sourceAvailable(context, project)
        }
    }

    fun queueSelectedSource() {
        val selected = selectedSource ?: return
        if (preparingSource || selected.media == null || selected.source == null || selected.roi == null) return
        if (gameEndMs - gameStartMs < 1_000) {
            inference = InferenceUiState(error = "Mark at least one second of game footage")
            return
        }
        preparingSource = true
        inference = InferenceUiState(
            stage = "opening",
            detail = "Reading recording metadata",
        )
        scope.launch {
            try {
                val analysisWindow = AnalysisTypes.AnalysisWindow(
                    gameStartMs / 1_000.0,
                    gameEndMs / 1_000.0,
                )
                val candidate = NativeProjectStore.newQueued(
                    selected.source,
                    selected.media,
                    selected.roi,
                    analysisWindow,
                    analyzeServingSide,
                    generateSideSwitchMarkers,
                )
                val existing = NativeProjectStore.findMatching(context, candidate)
                val reusable = useCache && existing?.status == ProjectStatus.READY &&
                    existing.modelId == FeatureSchema.MODEL_ID
                if (reusable) {
                    var opened = NativeProjectStore.updateSideSwitchEnabled(
                        context,
                        checkNotNull(existing).id,
                        generateSideSwitchMarkers,
                    ) ?: existing
                    val missingRequestedScoreOutput = opened.servingSide == null ||
                        (opened.sideSwitchEnabled && opened.sideSwitch == null)
                    if (missingRequestedScoreOutput) {
                        opened = NativeProjectStore.updateServingSideStatus(
                            context,
                            opened.id,
                            if (analyzeServingSide) ServingSideAnalysisStatus.QUEUED
                            else ServingSideAnalysisStatus.DISABLED,
                        ) ?: opened
                    }
                    withContext(Dispatchers.IO) {
                        setScoreTrackingPreference(context, opened, analyzeServingSide)
                    }
                    selectedProjectId = opened.id
                    creatingNew = false
                    selectedSource = null
                    inference = InferenceUiState(
                        projectId = opened.id,
                        servingSideJob = opened.servingSideStatus == ServingSideAnalysisStatus.QUEUED,
                        progress = if (opened.servingSideStatus == ServingSideAnalysisStatus.QUEUED) 0f else 1f,
                        stage = if (opened.servingSideStatus == ServingSideAnalysisStatus.QUEUED) {
                            "serving-side-queued"
                        } else "complete",
                        detail = if (opened.servingSideStatus == ServingSideAnalysisStatus.QUEUED) {
                            "Opened cached rally inference; waiting to prepare score tracking"
                        } else "Opened cached inference; no analysis was run",
                    )
                    reloadProjects(opened.id)
                    if (opened.servingSideStatus == ServingSideAnalysisStatus.QUEUED) {
                        ProjectAnalysisService.enqueueServingSide(context, opened.id)
                    }
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
                        detail = "Waiting for game-window video + audio inference",
                    )
                    reloadProjects(queued.id)
                    if (Build.VERSION.SDK_INT >= 33 &&
                        context.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
                    ) notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
                    ProjectAnalysisService.enqueue(context, queued.id)
                }
                GuidedTourStore.moveToEditor(context)
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

    val appContext = context
    DisposableEffect(Unit) {
        val receiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context?, intent: Intent?) {
                if (intent?.action == ProcessingTimeoutTracker.ACTION_UPDATED) {
                    timeoutSnapshot = ProcessingTimeoutTracker.snapshot(appContext)
                    return
                }
                if (intent?.action == ExportService.ACTION_PROGRESS) {
                    exportQueueCount = intent.getIntExtra(
                        ExportService.EXTRA_QUEUE_COUNT,
                        ExportService.pendingCount(),
                    )
                    val projectId = intent.getStringExtra(ExportService.EXTRA_PROJECT_ID)
                    val jobId = intent.getStringExtra(ExportService.EXTRA_JOB_ID)
                    if (projectId != null && jobId != null) {
                        exportStatuses = exportStatuses + (projectId to ExportJobStatus(
                            jobId = jobId,
                            projectId = projectId,
                            status = intent.getStringExtra(ExportService.EXTRA_STATUS) ?: "running",
                            progress = intent.getIntExtra(ExportService.EXTRA_PROGRESS, 0),
                            detail = intent.getStringExtra(ExportService.EXTRA_DETAIL).orEmpty(),
                            metrics = intent.getStringExtra(ExportService.EXTRA_METRICS),
                        ))
                    }
                    return
                }
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
                    val stage = intent.getStringExtra(ProjectAnalysisService.EXTRA_STAGE).orEmpty()
                    val servingSideJob = intent.getBooleanExtra(
                        ProjectAnalysisService.EXTRA_SERVING_SIDE_JOB,
                        false,
                    )
                    val stepMeasurements = InferenceStepMeasurementsJson.decode(
                        intent.getStringExtra(ProjectAnalysisService.EXTRA_STAGE_MEASUREMENTS),
                    )
                    inference = inference.copy(
                        projectId = projectId,
                        servingSideJob = servingSideJob,
                        running = if (servingSideJob) {
                            stage != "serving-side-complete" && stage != "serving-side-failed"
                        } else status == ProjectStatus.QUEUED || status == ProjectStatus.ANALYZING,
                        progress = intent.getDoubleExtra(ProjectAnalysisService.EXTRA_PROGRESS, 0.0)
                            .coerceIn(0.0, 1.0).toFloat(),
                        stage = stage,
                        detail = intent.getStringExtra(ProjectAnalysisService.EXTRA_DETAIL).orEmpty(),
                        stepMeasurements = stepMeasurements.ifEmpty {
                            inference.stepMeasurements
                        },
                        error = if (status == ProjectStatus.ERROR || stage == "serving-side-failed") {
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
            IntentFilter().apply {
                addAction(ProcessingTimeoutTracker.ACTION_UPDATED)
                addAction(ProjectAnalysisService.ACTION_UPDATE)
                addAction(ExportService.ACTION_PROGRESS)
            },
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

    timeoutSnapshot.unacknowledged.maxByOrNull { it.sequence }?.let { event ->
        val source = event.sourceName?.let { " for $it" }.orEmpty()
        AlertDialog(
            onDismissRequest = {
                ProcessingTimeoutTracker.acknowledgeThrough(context, event.sequence)
                timeoutSnapshot = ProcessingTimeoutTracker.snapshot(context)
            },
            title = { Text("Processing stopped by Android") },
            text = {
                Text(
                    "Android stopped ${event.operation.label}$source after its background time limit " +
                        "at ${event.displayTime()}.\n\n${event.detail}\n\n" +
                        "Recorded interruptions: ${timeoutSnapshot.totalCount}.",
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    ProcessingTimeoutTracker.acknowledgeThrough(context, event.sequence)
                    timeoutSnapshot = ProcessingTimeoutTracker.snapshot(context)
                }) { Text("Acknowledge") }
            },
        )
    }

    val queueCount = projects.count {
        it.status == ProjectStatus.QUEUED || it.status == ProjectStatus.ANALYZING ||
            it.servingSideStatus == ServingSideAnalysisStatus.QUEUED ||
            it.servingSideStatus == ServingSideAnalysisStatus.ANALYZING
    }
    val guidedTourTargets = remember { GuidedTourTargets() }
    var guidedTourRestartSignal by remember { mutableIntStateOf(0) }
    val projectControls: @Composable (EditorProjectSummary?) -> Unit = { editorSummary ->
        ProjectHeaderBar(
            projects = projects,
            selected = selectedProject,
            creatingNew = creatingNew,
            queueCount = queueCount,
            exportQueueCount = exportQueueCount,
            exportStatuses = exportStatuses,
            editorSummary = editorSummary,
            onSelect = { project ->
                selectedProjectId = project.id
                NativeProjectStore.setSelectedId(context, project.id)
                creatingNew = false
                inference = InferenceUiState(projectId = project.id)
                relinkMessage = null
                relinkFailed = false
            },
            onNew = {
                creatingNew = true
                selectedProjectId = null
                NativeProjectStore.setSelectedId(context, null)
                selectedSource = null
                gameStartMs = 0
                gameEndMs = 0
                inference = InferenceUiState(detail = "Choose a recording for the new project")
                analyzeServingSide = true
                generateSideSwitchMarkers = false
                relinkMessage = null
                relinkFailed = false
            },
            onDelete = { selectedProject?.let { confirmDelete = it } },
            onRestartTour = {
                GuidedTourStore.restart(
                    context,
                    if (creatingNew) GuidedTourStage.SETUP else GuidedTourStage.EDITOR,
                )
                guidedTourRestartSignal++
            },
        )
    }

    if (creatingNew || selectedProject == null) {
        ProjectShell(
            projectControls = { projectControls(null) },
            guidedTourStage = GuidedTourStage.SETUP,
            guidedTourTargets = guidedTourTargets,
            sourceReady = selectedSource?.media != null,
            guidedTourRestartSignal = guidedTourRestartSignal,
        ) {
            NewProjectCard(
                selected = selectedSource,
                preparing = preparingSource,
                state = inference,
                useCache = useCache,
                analyzeServingSide = analyzeServingSide,
                generateSideSwitchMarkers = generateSideSwitchMarkers,
                cacheBytes = cacheBytes,
                queueCount = queueCount,
                gameStartMs = gameStartMs,
                gameEndMs = gameEndMs,
                onSelect = { sourcePicker.launch(arrayOf("video/*")) },
                onImportFeedback = {
                    feedbackImportPicker.launch(arrayOf("application/json", "text/json"))
                },
                onQueue = ::queueSelectedSource,
                onGameStart = { requested ->
                    gameStartMs = requested.coerceIn(0, (gameEndMs - 1_000).coerceAtLeast(0))
                },
                onGameEnd = { requested ->
                    val duration = selectedSource?.media?.durationSeconds()?.let(::secondsToMs) ?: 0
                    gameEndMs = if (duration >= gameStartMs + 1_000) {
                        requested.coerceIn(gameStartMs + 1_000, duration)
                    } else duration
                },
                onFullVideo = {
                    gameStartMs = 0
                    gameEndMs = selectedSource?.media?.durationSeconds()?.let(::secondsToMs) ?: 0
                },
                onUseCache = { useCache = it },
                onAnalyzeServingSide = { analyzeServingSide = it },
                onGenerateSideSwitchMarkers = { generateSideSwitchMarkers = it },
                onClearCache = {
                    NativeFeatureCache.clearAll(context)
                    cacheBytes = 0
                    inference = InferenceUiState(detail = "Feature cache cleared")
                },
                onBenchmark = if (BuildConfig.DEBUG) {
                    { context.startActivity(Intent(context, MainActivity::class.java)) }
                } else {
                    null
                },
                guidedTourTargets = guidedTourTargets,
            )
        }
    } else if (selectedProject.status != ProjectStatus.READY) {
        ProjectShell(projectControls = { projectControls(null) }) {
            ProjectInferenceCard(
                project = selectedProject,
                state = inference.takeIf { it.projectId == selectedProject.id } ?: InferenceUiState(),
                sourceAvailable = sourceAvailable,
                relinkingSource = relinkingSource,
                relinkMessage = relinkMessage,
                onRelink = {
                    relinkTargetId = selectedProject.id
                    relinkPicker.launch(arrayOf("video/*"))
                },
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
                        detail = "Waiting for game-window video + audio inference",
                    )
                    reloadProjects(queued.id)
                    ProjectAnalysisService.enqueue(context, queued.id)
                },
            )
            if (selectedProject.analysisMeasurements.isNotEmpty()) {
                AnalysisMeasurementsCard(selectedProject.analysisMeasurements)
            }
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
                project = selectedProject,
                seed = currentSeed,
                store = store,
                initialDraft = restored ?: EditorMath.newDraft(currentSeed),
                restored = restored != null,
                analysisRunning = queueCount > 0,
                servingSideProgress = inference.takeIf {
                    it.projectId == selectedProject.id && it.servingSideJob
                }?.progress,
                servingSideProgressDetail = inference.takeIf {
                    it.projectId == selectedProject.id && it.servingSideJob
                }?.detail,
                servingSideStepMeasurements = inference.takeIf {
                    it.projectId == selectedProject.id && it.servingSideJob
                }?.stepMeasurements.orEmpty(),
                sourceAvailable = sourceAvailable,
                relinkingSource = relinkingSource,
                relinkMessage = relinkMessage,
                relinkFailed = relinkFailed,
                onProjectUpdated = { updated ->
                    projects = projects.map { if (it.id == updated.id) updated else it }
                },
                onRelink = {
                    relinkTargetId = selectedProject.id
                    relinkPicker.launch(arrayOf("video/*"))
                },
                sourceControls = projectControls,
                guidedTourTargets = guidedTourTargets,
                guidedTourRestartSignal = guidedTourRestartSignal,
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
    exportQueueCount: Int,
    exportStatuses: Map<String, ExportJobStatus>,
    editorSummary: EditorProjectSummary?,
    onSelect: (NativeProject) -> Unit,
    onNew: () -> Unit,
    onDelete: () -> Unit,
    onRestartTour: () -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }
    val context = LocalContext.current
    val versionName = remember(context) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            context.packageManager.getPackageInfo(
                context.packageName,
                PackageManager.PackageInfoFlags.of(0),
            ).versionName
        } else {
            @Suppress("DEPRECATION")
            context.packageManager.getPackageInfo(context.packageName, 0).versionName
        } ?: "unknown"
    }
    Surface(
        color = Color.White,
        shape = RoundedCornerShape(12.dp),
        shadowElevation = 1.dp,
    ) {
        Column(Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Image(
                    painter = painterResource(R.drawable.volleycut_logo),
                    contentDescription = "VolleyCut",
                    contentScale = ContentScale.Fit,
                    modifier = Modifier.width(116.dp).height(39.dp),
                )
                Spacer(Modifier.width(8.dp))
                Text(
                    "v$versionName",
                    color = Muted,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.SemiBold,
                )
                Spacer(Modifier.weight(1f))
                Text(
                    when {
                        queueCount == 0 && exportQueueCount == 0 -> "Queue idle"
                        queueCount == 0 -> "$exportQueueCount encoding"
                        exportQueueCount == 0 -> "$queueCount analyzing"
                        else -> "$queueCount analyzing · $exportQueueCount encoding"
                    },
                    color = if (queueCount == 0 && exportQueueCount == 0) Muted else Green,
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
                            else "${selected.source.name} · ${projectStatusLabel(selected, exportStatuses[selected.id])}",
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
                                            "${projectStatusLabel(project, exportStatuses[project.id])} · ${project.id.removePrefix("project-")}",
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
                TextButton(onClick = onRestartTour) { Text("? Tour") }
                if (selected != null && !creatingNew) {
                    TextButton(onClick = onDelete) { Text("Delete", color = Danger) }
                }
            }
            editorSummary?.let { summary ->
                Text(
                    "${summary.width}×${summary.height} · ${compactTime(summary.sourceDurationMs)} source · " +
                        "${compactTime(summary.outputDurationMs)} output",
                    modifier = Modifier.padding(top = 4.dp),
                    fontFamily = FontFamily.Monospace,
                    fontSize = 12.sp,
                )
                Text(
                    "${summary.kept} kept · ${summary.removed} removed · ${summary.ignored} fully ignored",
                    color = Muted,
                    fontSize = 12.sp,
                )
            }
        }
    }
}

@Composable
private fun ProjectInferenceCard(
    project: NativeProject,
    state: InferenceUiState,
    sourceAvailable: Boolean?,
    relinkingSource: Boolean,
    relinkMessage: String?,
    onRelink: () -> Unit,
    onRetry: () -> Unit,
) {
    SectionCard("PROJECT INFERENCE", project.source.name) {
        Text(
            when (project.status) {
                ProjectStatus.QUEUED -> "Queued behind any active project"
                ProjectStatus.ANALYZING -> "Generating shared features and running both production models"
                ProjectStatus.ERROR -> "Inference stopped with an error"
                ProjectStatus.READY -> "Inference ready"
            },
            color = Muted,
        )
        if (project.status == ProjectStatus.ANALYZING || state.progress > 0f) {
            if (state.stepMeasurements.isEmpty()) {
                LinearProgressIndicator(progress = { state.progress }, modifier = Modifier.fillMaxWidth())
            }
        }
        val detail = state.detail.ifBlank { project.error.orEmpty() }
        if (state.stepMeasurements.isEmpty() &&
            (state.stage.isNotBlank() || detail.isNotBlank())
        ) Text(
            "${state.stage.ifBlank { project.status.wireName }.uppercase(Locale.US)} · $detail",
            fontSize = 12.sp,
            fontFamily = FontFamily.Monospace,
            color = if (project.status == ProjectStatus.ERROR) Danger else Ink,
        )
        state.performance?.takeIf {
            state.stepMeasurements.isEmpty() && state.stage == "video"
        }?.let { stats ->
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
        if (state.stepMeasurements.isNotEmpty()) {
            InferenceProgressMeasurementsPanel(
                steps = state.stepMeasurements,
                performance = state.performance,
            )
        }
        Text(
            "Project ${project.id} · ${compactTime((project.media.durationSeconds() * 1_000).toLong())}",
            color = Muted,
            fontSize = 11.sp,
            fontFamily = FontFamily.Monospace,
        )
        if (sourceAvailable == false) {
            Text(
                relinkMessage
                    ?: "The saved video location is unavailable. Re-link the original recording before retrying inference.",
                color = Danger,
                fontSize = 12.sp,
            )
            Button(enabled = !relinkingSource, onClick = onRelink) {
                Text(if (relinkingSource) "Validating…" else "Re-link video")
            }
        }
        if (project.status == ProjectStatus.ERROR) {
            Button(enabled = sourceAvailable != false, onClick = onRetry) { Text("Retry") }
        }
    }
}

@Composable
private fun InferenceProgressMeasurementsPanel(
    steps: List<InferenceStepMeasurement>,
    performance: AnalysisTypes.PerformanceStats? = null,
    compact: Boolean = false,
) {
    if (steps.isEmpty()) return
    Surface(
        modifier = Modifier.fillMaxWidth().border(1.dp, Rail, RoundedCornerShape(8.dp)),
        color = Paper,
        shape = RoundedCornerShape(8.dp),
    ) {
        Column(
            Modifier.fillMaxWidth().padding(if (compact) 8.dp else 12.dp),
            verticalArrangement = Arrangement.spacedBy(if (compact) 7.dp else 10.dp),
        ) {
            steps.forEachIndexed { index, step ->
                val percent = (step.fraction * 100).roundToInt()
                val elapsedSeconds = step.elapsedMilliseconds / 1_000.0
                val etaSeconds = if (
                    step.status == InferenceStepStatus.RUNNING &&
                    step.fraction >= .03 && elapsedSeconds >= .5
                ) elapsedSeconds * (1.0 - step.fraction) / step.fraction else null
                val measuredRate = when {
                    step.id == "video" && performance != null && performance.realtimeRatio() > 0.0 ->
                        String.format(Locale.US, "%.2fx realtime", performance.realtimeRatio())
                    step.status == InferenceStepStatus.RUNNING && elapsedSeconds >= .5 && step.fraction > 0.0 ->
                        String.format(Locale.US, "%.1f%%/s", step.fraction * 100.0 / elapsedSeconds)
                    step.status == InferenceStepStatus.COMPLETE -> "Completed"
                    step.status == InferenceStepStatus.ERROR -> "Stopped"
                    step.status == InferenceStepStatus.PENDING -> "Waiting"
                    else -> "Measuring…"
                }
                val metrics = when (step.status) {
                    InferenceStepStatus.RUNNING ->
                        "$percent% · $measuredRate · ${measurementDuration(step.elapsedMilliseconds)} elapsed · " +
                            if (etaSeconds != null) "ETA ${secondsLabel(etaSeconds)}" else "estimating ETA"
                    InferenceStepStatus.COMPLETE -> buildList {
                        add("${measurementDuration(step.elapsedMilliseconds)} elapsed")
                        if (measuredRate != "Completed") add(measuredRate)
                    }.joinToString(" · ")
                    InferenceStepStatus.ERROR ->
                        "Stopped after ${measurementDuration(step.elapsedMilliseconds)}"
                    InferenceStepStatus.PENDING -> null
                }
                Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            (index + 1).toString().padStart(2, '0'),
                            color = Muted,
                            fontFamily = FontFamily.Monospace,
                            fontSize = 10.sp,
                        )
                        Text(
                            step.label,
                            Modifier.weight(1f).padding(horizontal = 8.dp),
                            fontSize = if (compact) 11.sp else 13.sp,
                            fontWeight = FontWeight.SemiBold,
                        )
                        Text(
                            when (step.status) {
                                InferenceStepStatus.PENDING -> "QUEUED"
                                InferenceStepStatus.RUNNING -> "RUNNING"
                                InferenceStepStatus.COMPLETE -> "DONE"
                                InferenceStepStatus.ERROR -> "ERROR"
                            },
                            color = when (step.status) {
                                InferenceStepStatus.RUNNING -> Orange
                                InferenceStepStatus.COMPLETE -> Green
                                InferenceStepStatus.ERROR -> Danger
                                InferenceStepStatus.PENDING -> Muted
                            },
                            fontFamily = FontFamily.Monospace,
                            fontSize = 9.sp,
                            fontWeight = FontWeight.Bold,
                        )
                    }
                    LinearProgressIndicator(
                        progress = { step.fraction.toFloat() },
                        modifier = Modifier.fillMaxWidth(),
                    )
                    metrics?.let {
                        Text(
                            it,
                            color = Muted,
                            fontFamily = FontFamily.Monospace,
                            fontSize = 9.sp,
                        )
                    }
                    if (step.status == InferenceStepStatus.RUNNING ||
                        step.status == InferenceStepStatus.ERROR
                    ) {
                        Text(step.detail, color = Muted, fontSize = 9.sp, maxLines = 2)
                    }
                }
                if (index < steps.lastIndex) HorizontalDivider(color = Rail)
            }
            Text(
                "Total elapsed · ${measurementDuration(steps.sumOf { it.elapsedMilliseconds })}",
                color = Muted,
                fontFamily = FontFamily.Monospace,
                fontSize = 10.sp,
            )
        }
    }
}

@Composable
private fun AnalysisMeasurementsCard(runs: List<AnalysisRunMeasurements>) {
    var expandedRun by remember(runs) { mutableStateOf<String?>(null) }
    SectionCard(
        "ANALYSIS MEASUREMENTS",
        if (runs.isEmpty()) "Stored with this project after inference"
        else "${runs.size} saved ${if (runs.size == 1) "run" else "runs"} · persisted in project information",
    ) {
        if (runs.isEmpty()) {
            Text(
                "This project predates saved measurements. The next full or score-tracking inference run will populate this card.",
                color = Muted,
                fontSize = 12.sp,
            )
            return@SectionCard
        }
        runs.forEachIndexed { runIndex, run ->
            val runKey = "${run.kind.wireName}-${run.completedAtMs}"
            val stages = run.stageMilliseconds.filter { (key, value) ->
                value > 0.0 && !(key == "inference" && "rally_inference" in run.stageMilliseconds)
            }
            Surface(
                modifier = Modifier.fillMaxWidth().border(1.dp, Rail, RoundedCornerShape(6.dp)),
                color = Paper,
                shape = RoundedCornerShape(6.dp),
            ) {
                Column(
                    Modifier.fillMaxWidth().padding(10.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text(run.kind.label, fontWeight = FontWeight.SemiBold)
                            Text(
                                if (run.succeeded) "COMPLETED" else "STOPPED · ${run.error.orEmpty()}",
                                color = if (run.succeeded) Green else Danger,
                                fontFamily = FontFamily.Monospace,
                                fontSize = 10.sp,
                                maxLines = 2,
                            )
                        }
                        Text(
                            measurementDuration(run.totalMilliseconds),
                            color = Ink,
                            fontFamily = FontFamily.Monospace,
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Bold,
                        )
                    }
                    stages.forEach { (key, milliseconds) ->
                        MeasurementRow(measurementLabel(key), milliseconds)
                    }
                    measurementCounterSummary(run.counters)?.let { summary ->
                        Text(summary, color = Muted, fontFamily = FontFamily.Monospace, fontSize = 10.sp)
                    }
                    if (run.profileMilliseconds.isNotEmpty()) {
                        TextButton(onClick = {
                            expandedRun = if (expandedRun == runKey) null else runKey
                        }) {
                            Text(
                                if (expandedRun == runKey) "Hide detailed feature timings"
                                else "Show ${run.profileMilliseconds.size} detailed feature timings",
                            )
                        }
                        if (expandedRun == runKey) {
                            Text(
                                "Profiler measurements can overlap because parent and child operations are both retained.",
                                color = Muted,
                                fontSize = 10.sp,
                            )
                            run.profileMilliseconds.forEach { (key, milliseconds) ->
                                MeasurementRow(measurementLabel(key), milliseconds, compact = true)
                            }
                        }
                    }
                }
            }
            if (runIndex < runs.lastIndex) Spacer(Modifier.height(2.dp))
        }
    }
}

@Composable
private fun MeasurementRow(label: String, milliseconds: Double, compact: Boolean = false) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Text(
            label,
            Modifier.weight(1f),
            color = if (compact) Muted else Ink,
            fontSize = if (compact) 10.sp else 12.sp,
        )
        Text(
            measurementDuration(milliseconds),
            color = Muted,
            fontFamily = FontFamily.Monospace,
            fontSize = if (compact) 10.sp else 12.sp,
        )
    }
}

private fun measurementDuration(milliseconds: Double): String = when {
    !milliseconds.isFinite() || milliseconds < 0.0 -> "—"
    milliseconds < 1.0 -> String.format(Locale.US, "%.2f ms", milliseconds)
    milliseconds < 1_000.0 -> String.format(Locale.US, "%.0f ms", milliseconds)
    else -> compactTime(milliseconds.roundToLong())
}

private fun measurementLabel(key: String): String = when (key) {
    "open" -> "Open recording and cache"
    "video", "video_decode_and_features" -> "Video feature generation"
    "audio", "audio_decode_and_features" -> "Audio feature generation"
    "contextualize" -> "Context and percentile features"
    "rally", "rally_inference" -> "Rally model inference"
    "score_specialists" -> "Score-tracking specialists"
    "score_planning" -> "Score request planning"
    "score_frame_decode" -> "Shared score-frame decoding"
    "serving-side", "serving_side" -> "Serving-side features and model"
    "side-switch", "side_switch" -> "Team-switch features and model"
    else -> key.split('/').joinToString(" · ") { segment ->
        segment.replace('_', ' ').replaceFirstChar { character ->
            character.titlecase(Locale.US)
        }
    }
}

private fun measurementCounterSummary(counters: Map<String, Long>): String? {
    val parts = buildList {
        counters["sample_rows"]?.let { add(String.format(Locale.US, "%,d feature rows", it)) }
        counters["decoded_video_frames"]?.let {
            add(String.format(Locale.US, "%,d decoded video frames", it))
        }
        counters["shared_requested_frames"]?.let {
            add(String.format(Locale.US, "%,d requested score frames", it))
        }
        counters["score_decoder_output_frames"]?.let {
            add(String.format(Locale.US, "%,d score decoder outputs", it))
        }
    }
    return parts.takeIf { it.isNotEmpty() }?.joinToString(" · ")
}

@Composable
private fun NewProjectCard(
    selected: SourceSelection?,
    preparing: Boolean,
    state: InferenceUiState,
    useCache: Boolean,
    analyzeServingSide: Boolean,
    generateSideSwitchMarkers: Boolean,
    cacheBytes: Long,
    queueCount: Int,
    gameStartMs: Long,
    gameEndMs: Long,
    onSelect: () -> Unit,
    onImportFeedback: () -> Unit,
    onQueue: () -> Unit,
    onGameStart: (Long) -> Unit,
    onGameEnd: (Long) -> Unit,
    onFullVideo: () -> Unit,
    onUseCache: (Boolean) -> Unit,
    onAnalyzeServingSide: (Boolean) -> Unit,
    onGenerateSideSwitchMarkers: (Boolean) -> Unit,
    onClearCache: () -> Unit,
    onBenchmark: (() -> Unit)?,
    guidedTourTargets: GuidedTourTargets? = null,
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
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            OutlinedButton(
                enabled = !preparing,
                onClick = onSelect,
                modifier = Modifier
                    .weight(1f)
                    .guidedTourTarget("setup-source", guidedTourTargets),
            ) { Text("Choose video") }
            Button(
                enabled = selected?.media != null && !preparing && gameEndMs - gameStartMs >= 1_000,
                onClick = onQueue,
                modifier = Modifier
                    .weight(1f)
                    .guidedTourTarget("setup-create", guidedTourTargets),
            ) {
                Text(if (preparing) "Creating…" else "Create & queue")
            }
        }
        TextButton(
            enabled = !preparing,
            onClick = onImportFeedback,
            modifier = Modifier.align(Alignment.Start),
        ) {
            Text("Import model feedback instead", color = Muted)
        }
        if (selected?.media != null) {
            Column(
                Modifier.guidedTourTarget("setup-window", guidedTourTargets),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                GameWindowPicker(
                    selected = selected,
                    gameStartMs = gameStartMs,
                    gameEndMs = gameEndMs,
                    enabled = !preparing,
                    onGameStart = onGameStart,
                    onGameEnd = onGameEnd,
                    onFullVideo = onFullVideo,
                )
            }
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("Prepare score tracking", fontWeight = FontWeight.SemiBold)
                Text(
                    if (analyzeServingSide) {
                        if (generateSideSwitchMarkers) {
                            "Generate serve-side and team-switch predictions during project creation"
                        } else "Generate serve-side predictions during project creation"
                    } else {
                        "Create faster; enabling score tracking later will run this analysis"
                    },
                    fontSize = 12.sp,
                    color = Muted,
                )
            }
            Switch(
                enabled = !preparing,
                checked = analyzeServingSide,
                onCheckedChange = onAnalyzeServingSide,
            )
        }
        Surface(
            modifier = Modifier
                .fillMaxWidth()
                .border(1.dp, Rail, RoundedCornerShape(4.dp))
                .clickable(enabled = !preparing && analyzeServingSide) {
                    onGenerateSideSwitchMarkers(!generateSideSwitchMarkers)
                },
            color = Color.White,
            shape = RoundedCornerShape(4.dp),
        ) {
            Row(
                Modifier.padding(horizontal = 10.dp, vertical = 8.dp),
                verticalAlignment = Alignment.Top,
            ) {
                Checkbox(
                    enabled = !preparing && analyzeServingSide,
                    checked = generateSideSwitchMarkers,
                    onCheckedChange = onGenerateSideSwitchMarkers,
                )
                Column(Modifier.padding(top = 4.dp)) {
                    Text("Generate team side-switch markers", fontWeight = FontWeight.SemiBold)
                    Text(
                        "Enable this only for formats where teams change court sides during the recording.",
                        fontSize = 12.sp,
                        color = Muted,
                    )
                }
            }
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("Reuse generated features", fontWeight = FontWeight.SemiBold)
                Text(
                    if (useCache) "Current ensemble inference opens without running anything"
                    else "Regenerate this source's features for an experiment",
                    fontSize = 12.sp,
                    color = Muted,
                )
            }
            Switch(enabled = !preparing, checked = useCache, onCheckedChange = onUseCache)
        }
        Text(
            "Both models share the feature cache. Overlaps merge; one-model ranges are flagged for validation.",
            fontSize = 11.sp,
            color = Muted,
        )
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
            if (onBenchmark != null) {
                TextButton(enabled = !preparing, onClick = onBenchmark) { Text("Benchmark tools") }
            }
        }
    }
}

@OptIn(markerClass = [UnstableApi::class])
@Composable
private fun GameWindowPicker(
    selected: SourceSelection,
    gameStartMs: Long,
    gameEndMs: Long,
    enabled: Boolean,
    onGameStart: (Long) -> Unit,
    onGameEnd: (Long) -> Unit,
    onFullVideo: () -> Unit,
) {
    val context = LocalContext.current
    val durationMs = secondsToMs(checkNotNull(selected.media).durationSeconds())
    var playheadMs by remember(selected.uri) { mutableLongStateOf(0L) }
    var playing by remember(selected.uri) { mutableStateOf(false) }
    val player = remember(selected.uri) {
        ExoPlayer.Builder(context).build().apply {
            setSeekParameters(SeekParameters.EXACT)
            setMediaItem(MediaItem.fromUri(selected.uri))
            prepare()
        }
    }
    DisposableEffect(player) {
        val listener = object : Player.Listener {
            override fun onIsPlayingChanged(isPlaying: Boolean) { playing = isPlaying }
        }
        player.addListener(listener)
        onDispose {
            player.removeListener(listener)
            player.release()
        }
    }
    LaunchedEffect(player) {
        while (true) {
            playheadMs = player.currentPosition.coerceIn(0, durationMs)
            delay(50)
        }
    }
    Card(colors = CardDefaults.cardColors(containerColor = Color.Black)) {
        ContentFrame(
            player = player,
            modifier = Modifier.fillMaxWidth().height(176.dp),
            surfaceType = SURFACE_TYPE_TEXTURE_VIEW,
        )
    }
    Row(verticalAlignment = Alignment.CenterVertically) {
        SmallButton(if (playing) "Pause" else "Play", enabled = enabled) {
            if (player.isPlaying) player.pause() else player.play()
        }
        SmallButton("−10s", enabled = enabled) {
            player.seekTo((player.currentPosition - 10_000).coerceAtLeast(0))
        }
        SmallButton("+10s", enabled = enabled) {
            player.seekTo((player.currentPosition + 10_000).coerceAtMost(durationMs))
        }
        Text(
            "${preciseTime(playheadMs)} / ${preciseTime(durationMs)}",
            modifier = Modifier.padding(start = 8.dp),
            fontFamily = FontFamily.Monospace,
            fontSize = 12.sp,
        )
    }
    Slider(
        value = if (durationMs > 0) playheadMs.toFloat() else 0f,
        onValueChange = {
            playheadMs = it.toLong()
            player.seekTo(playheadMs)
        },
        valueRange = 0f..durationMs.coerceAtLeast(1).toFloat(),
        enabled = enabled,
    )
    Row(verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text("ANALYSIS WINDOW", color = Orange, fontSize = 11.sp, fontWeight = FontWeight.Bold)
            Text("Mark game start & end", fontWeight = FontWeight.Bold)
            Text(
                "Only this range generates features and appears on the editor overview.",
                color = Muted,
                fontSize = 12.sp,
            )
        }
        TextButton(enabled = enabled, onClick = onFullVideo) { Text("Use full video") }
    }
    Row(
        Modifier.horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        OutlinedButton(enabled = enabled, onClick = { onGameStart(playheadMs) }) {
            Text("Set start · ${preciseTime(gameStartMs)}")
        }
        OutlinedButton(enabled = enabled, onClick = { onGameEnd(playheadMs) }) {
            Text("Set end · ${preciseTime(gameEndMs)}")
        }
    }
    val analyzedMs = (gameEndMs - gameStartMs).coerceAtLeast(0)
    Text(
        "${compactTime(analyzedMs)} analyzed · ${compactTime(durationMs - analyzedMs)} skipped",
        color = Green,
        fontSize = 12.sp,
        fontWeight = FontWeight.SemiBold,
    )
}

@Composable
private fun ProjectShell(
    projectControls: @Composable () -> Unit,
    guidedTourStage: GuidedTourStage? = null,
    guidedTourTargets: GuidedTourTargets? = null,
    sourceReady: Boolean = false,
    guidedTourRestartSignal: Int = 0,
    content: @Composable ColumnScope.() -> Unit,
) {
    Box(Modifier.fillMaxSize()) {
        Scaffold(containerColor = Paper) { scaffoldPadding ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(scaffoldPadding)
                    .verticalScroll(rememberScrollState())
                    .padding(horizontal = 12.dp, vertical = 8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                projectControls()
                content()
                LegalFooter()
            }
        }
        if (guidedTourStage != null && guidedTourTargets != null) {
            GuidedTour(
                stage = guidedTourStage,
                targets = guidedTourTargets,
                sourceReady = sourceReady,
                restartSignal = guidedTourRestartSignal,
            )
        }
    }
}

private const val PRIVACY_POLICY_URL = "https://volleycut.vafrederico.com/privacy.html"
private const val APACHE_LICENSE_URL = "https://www.apache.org/licenses/LICENSE-2.0"

@Composable
private fun LegalFooter() {
    val context = LocalContext.current
    var showNotices by remember { mutableStateOf(false) }
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        TextButton(onClick = {
            runCatching {
                context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(PRIVACY_POLICY_URL)))
            }
        }) { Text("Privacy") }
        Text("·", color = Muted)
        TextButton(onClick = { showNotices = true }) { Text("Open source") }
    }
    if (showNotices) {
        AlertDialog(
            onDismissRequest = { showNotices = false },
            title = { Text("Open-source notices") },
            text = {
                Column(
                    modifier = Modifier.verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    Text(
                        "VolleyCut includes the following open-source components. " +
                            "Their licenses remain available to you under their original terms.",
                    )
                    Text("AndroidX and Jetpack Compose — Apache License 2.0")
                    Text("AndroidX Media3 1.10.1 — Apache License 2.0")
                    Text("OpenCV 4.12.0 — Apache License 2.0")
                    Text("Kotlin runtime — Apache License 2.0")
                    Text(
                        "The Android system, device codecs, and other platform components are " +
                            "provided under their respective system licenses.",
                        color = Muted,
                        fontSize = 12.sp,
                    )
                    TextButton(onClick = {
                        runCatching {
                            context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(APACHE_LICENSE_URL)))
                        }
                    }) { Text("View Apache License 2.0") }
                }
            },
            confirmButton = {
                TextButton(onClick = { showNotices = false }) { Text("Close") }
            },
        )
    }
}

@OptIn(markerClass = [UnstableApi::class])
@Composable
private fun EditorScreen(
    activity: ComponentActivity,
    project: NativeProject,
    seed: EditorSeed,
    store: EditorDraftStore,
    initialDraft: EditorDraft,
    restored: Boolean,
    analysisRunning: Boolean,
    servingSideProgress: Float?,
    servingSideProgressDetail: String?,
    servingSideStepMeasurements: List<InferenceStepMeasurement>,
    sourceAvailable: Boolean?,
    relinkingSource: Boolean,
    relinkMessage: String?,
    relinkFailed: Boolean,
    onProjectUpdated: (NativeProject) -> Unit,
    onRelink: () -> Unit,
    sourceControls: @Composable (EditorProjectSummary) -> Unit,
    guidedTourTargets: GuidedTourTargets,
    guidedTourRestartSignal: Int,
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var draft by remember { mutableStateOf(initialDraft) }
    var selectedId by remember { mutableStateOf(initialDraft.cuts.firstOrNull()?.id.orEmpty()) }
    var focusLocked by remember { mutableStateOf(false) }
    var playbackPositionMs by remember { mutableLongStateOf(seed.gameStartMs) }
    var isPlaying by remember { mutableStateOf(false) }
    var previewEndMs by remember { mutableStateOf<Long?>(null) }
    var message by remember {
        mutableStateOf(if (restored) "Restored saved edits on this device" else "New on-device draft")
    }
    var exportState by remember(project.id) {
        mutableStateOf(ExportService.statusForProject(project.id)?.toUiState() ?: ExportUiState())
    }
    var pendingExportIntervals by remember { mutableStateOf<List<FinalCutInterval>?>(null) }
    var feedbackExporting by remember { mutableStateOf(false) }
    var confirmReset by remember { mutableStateOf(false) }
    var selectedSuggestionId by remember { mutableStateOf<String?>(null) }
    var suppressionPreparing by remember { mutableStateOf(false) }
    var selectedScoreMarkerId by remember { mutableStateOf<String?>(null) }
    var manualServingSide by remember { mutableStateOf(ServingSide.NEAR) }

    val player = remember {
        ExoPlayer.Builder(context).build().apply {
            setSeekParameters(SeekParameters.EXACT)
            setMediaItem(MediaItem.fromUri(seed.sourceUri))
            prepare()
            seekTo(seed.gameStartMs)
        }
    }
    val sortedCuts = draft.cuts.sortedWith(compareBy<EditableCut> { it.keepStartMs }.thenBy { it.keepEndMs })
    val selected = sortedCuts.firstOrNull { it.id == selectedId } ?: sortedCuts.firstOrNull()
    val selectedIndex = selected?.let { sortedCuts.indexOf(it) } ?: -1
    val finalMaterialization = EditorMath.materialize(draft, seed.suppression)
    val finalIntervals = finalMaterialization.intervals
    val effectiveIds = EditorMath.effectiveKeptIds(draft, seed.suppression)
    val excludedScoreRallyIds = draft.cuts.asSequence()
        .filter { it.origin == CutOrigin.INFERRED && (!it.included || it.id !in effectiveIds) }
        .map { it.id }
        .toSet()
    val scoreRallyRanges = draft.cuts.filter {
        it.included && (it.origin == CutOrigin.MANUAL || it.id in effectiveIds)
    }.map { ScoreRallyRange(it.coreStartMs, it.coreEndMs, it.keepStartMs, it.keepEndMs) }
    val scoreMergedRanges = finalIntervals.map { ScoreMergedRange(it.startMs, it.endMs) }
    val preparedScoreOverlay = ScoreOverlay.prepare(
        draft.scoreTracking,
        draft.ignoredIntervals,
        excludedScoreRallyIds,
        scoreRallyRanges,
        scoreMergedRanges,
    )
    val visibleScore = ScoreReducer.deriveAt(
        preparedScoreOverlay.tracking,
        ScoreReducer.scoreBoundaryTimestamp(
            playbackPositionMs,
            preparedScoreOverlay.rallyRanges,
            preparedScoreOverlay.tracking,
            preparedScoreOverlay.mergedRanges,
        ),
    )
    val activeSuggestions = EditorMath.activeSuggestions(draft, seed.suppression)
    val selectedSuggestion = activeSuggestions.firstOrNull { it.fragmentId() == selectedSuggestionId }
    val selectedSuggestionIndex = selectedSuggestion?.let(activeSuggestions::indexOf) ?: -1
    val appliedSuggestionCount = activeSuggestions.count {
        EditorMath.suggestionEffectiveDecision(draft, it) == SuppressionDecision.SUPPRESS
    }
    val baselineTotalMs = EditorMath.totalFinalMs(EditorMath.finalIntervals(
        draft.copy(selectedSuppressionPolicy = SuppressionPolicyEngine.Policy.NONE),
        seed.suppression,
    ))
    val suppressionPoliciesEqual = seed.suppression?.let { analysis ->
        listOf(
            SuppressionPolicyEngine.Policy.CONSERVATIVE,
            SuppressionPolicyEngine.Policy.BALANCED,
            SuppressionPolicyEngine.Policy.AGGRESSIVE,
        ).map { policy ->
            SuppressionPolicyEngine.active(analysis, policy)
                .map { "${it.startMs()}:${it.endMs()}:${it.logicalId()}" }
        }.distinct().size == 1
    } == true
    val lowConfidence = sortedCuts.filter {
        it.origin == CutOrigin.INFERRED && it.included && it.id in effectiveIds &&
            (it.isModelDisagreement() || it.confidence < draft.confidenceReviewThreshold)
    }
    val disagreementCount = lowConfidence.count(EditableCut::isModelDisagreement)
    val joinedGaps = finalIntervals.flatMap { it.joinedGaps }
    val totalFinalMs = EditorMath.totalFinalMs(finalIntervals)
    val exportPending = exportState.status == "queued" || exportState.status == "running"
    val removedCount = draft.cuts.count { !it.included }
    val ignoredCutCount = draft.cuts.count { it.included && it.id !in effectiveIds }
    val visibleIgnoredIntervals = draft.ignoredIntervals.filter {
        it.endMs > seed.gameStartMs && it.startMs < seed.gameEndMs
    }
    val detailWindow = selectedSuggestion?.let { suggestion ->
        val span = max(24_000L, suggestion.endMs() - suggestion.startMs() + 10_000L)
        val center = (suggestion.startMs() + suggestion.endMs()) / 2
        var start = max(seed.gameStartMs, center - span / 2)
        val end = min(seed.gameEndMs, start + span)
        start = max(seed.gameStartMs, end - span)
        DetailWindow(start, end)
    } ?: EditorMath.detailWindow(
        selected, playbackPositionMs, seed.durationMs, seed.gameStartMs, seed.gameEndMs,
    )

    fun updateDraft(mutate: (EditorDraft) -> EditorDraft) {
        draft = EditorMath.reconcileTouchedCuts(mutate(draft), seed)
            .copy(updatedAtMs = System.currentTimeMillis())
    }

    LaunchedEffect(
        project.servingSideCacheIdentity,
        project.sideSwitch?.modelFingerprint,
        project.sideSwitchEnabled,
    ) {
        val seeded = ScoreReducer.seedModelMarkers(
            draft.scoreTracking,
            project.servingSide,
            project.sideSwitch,
            project.sideSwitchEnabled,
        )
        if (seeded != draft.scoreTracking) updateDraft { it.copy(scoreTracking = seeded) }
    }

    fun updateCut(id: String, mutate: (EditableCut) -> EditableCut) {
        updateDraft { current ->
            current.copy(cuts = current.cuts.map { if (it.id == id) mutate(it) else it })
        }
    }

    fun seekTo(positionMs: Long) {
        val target = positionMs.coerceIn(seed.gameStartMs, seed.gameEndMs)
        playbackPositionMs = target
        player.seekTo(target)
    }

    fun selectSuggestion(suggestion: AnalysisTypes.SuppressionSuggestion) {
        player.pause()
        previewEndMs = null
        selectedSuggestionId = suggestion.fragmentId()
        val parent = sortedCuts.firstOrNull { cut ->
            cut.coreStartMs < suggestion.endMs() && suggestion.startMs() < cut.coreEndMs
        }
        parent?.let { selectedId = it.id }
        seekTo(max(seed.gameStartMs, suggestion.startMs() - 2_000))
    }

    fun setKeepBoundary(side: String, valueMs: Long) {
        val cut = selected ?: return
        updateCut(cut.id) { current ->
            if (current.origin == CutOrigin.MANUAL) {
                if (side == "start") {
                    val start = valueMs.coerceIn(seed.gameStartMs, current.keepEndMs - MIN_MARK_MS)
                    current.copy(coreStartMs = start, keepStartMs = start)
                } else {
                    val end = valueMs.coerceIn(current.keepStartMs + MIN_MARK_MS, seed.gameEndMs)
                    current.copy(coreEndMs = end, keepEndMs = end)
                }
            } else if (side == "start") {
                current.copy(keepStartMs = valueMs.coerceIn(seed.gameStartMs, current.coreStartMs))
            } else {
                current.copy(keepEndMs = valueMs.coerceIn(current.coreEndMs, seed.gameEndMs))
            }
        }
    }

    fun setCoreBoundary(side: String, valueMs: Long) {
        val cut = selected ?: return
        updateDraft { current ->
            if (side == "start") {
                EditorMath.setCoreStart(current, cut.id, valueMs, seed.gameStartMs)
            } else {
                EditorMath.setCoreEnd(current, cut.id, valueMs, seed.gameEndMs)
            }
        }
    }

    fun splitSelectedAtPlayhead() {
        val cut = selected ?: return
        var split: CutSplitResult? = null
        updateDraft { current ->
            EditorMath.splitCut(
                current,
                cut.id,
                playbackPositionMs,
                seed.gameStartMs,
                seed.gameEndMs,
            )?.also { split = it }?.draft ?: current
        }
        val created = split?.newCut
        if (created == null) {
            message = "Move the playhead inside the rally before splitting"
        } else {
            selectedSuggestionId = null
            selectedId = created.id
            message = "Split ${cut.id} at ${preciseTime(playbackPositionMs)}; both parts can now be trimmed independently"
        }
    }

    val notificationPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { }
    val exportLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.CreateDocument("video/mp4"),
    ) { uri ->
        val requestedIntervals = pendingExportIntervals
        pendingExportIntervals = null
        if (uri != null && !requestedIntervals.isNullOrEmpty()) {
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    uri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION,
                )
            }
            if (Build.VERSION.SDK_INT >= 33 &&
                context.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
            ) notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
            val jobId = UUID.randomUUID().toString()
            val exportIntent = Intent(context, ExportService::class.java).apply {
                putExtra(ExportService.EXTRA_JOB_ID, jobId)
                putExtra(ExportService.EXTRA_PROJECT_ID, project.id)
                putExtra(ExportService.EXTRA_SOURCE_URI, seed.sourceUri)
                putExtra(ExportService.EXTRA_SOURCE_NAME, seed.displayName)
                putExtra(ExportService.EXTRA_SOURCE_DURATION_MS, seed.durationMs)
                putExtra(ExportService.EXTRA_DESTINATION_URI, uri.toString())
                putExtra(ExportService.EXTRA_INTERVAL_STARTS, requestedIntervals.map { it.startMs }.toLongArray())
                putExtra(ExportService.EXTRA_INTERVAL_ENDS, requestedIntervals.map { it.endMs }.toLongArray())
                putExtra(
                    ExportService.EXTRA_SCORE_SNAPSHOT,
                    ScoreExportSnapshotJson.encode(ScoreExportSnapshot(
                        render = draft.renderScoreOverlay && draft.scoreTracking.enabled,
                        renderPointTimeline = draft.renderScoreTimeline,
                        scoreTracking = draft.scoreTracking,
                        ignoredIntervals = draft.ignoredIntervals,
                        excludedRallyIds = excludedScoreRallyIds,
                        rallyRanges = scoreRallyRanges,
                        mergedRanges = requestedIntervals.map {
                            ScoreMergedRange(it.startMs, it.endMs)
                        },
                    )),
                )
            }
            context.startForegroundService(exportIntent)
            exportState = ExportUiState(jobId, "queued", 0, "Added to export queue")
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
    val feedbackSaveLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.CreateDocument("application/json"),
    ) { uri ->
        if (uri != null) scope.launch {
            feedbackExporting = true
            val result = runCatching {
                withContext(Dispatchers.IO) {
                    val bundle = createModelFeedback(context, project, draft, finalIntervals)
                    context.contentResolver.openOutputStream(uri, "w")!!.bufferedWriter().use {
                        it.write(bundle)
                    }
                }
            }
            feedbackExporting = false
            result.onSuccess {
                message = "Saved model feedback with features, inference, and corrections"
            }.onFailure {
                message = it.message ?: "Could not save model feedback"
            }
        }
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
                if (intent.getStringExtra(ExportService.EXTRA_PROJECT_ID) != project.id) return
                exportState = ExportUiState(
                    jobId = intent.getStringExtra(ExportService.EXTRA_JOB_ID),
                    status = intent.getStringExtra(ExportService.EXTRA_STATUS) ?: "running",
                    progress = intent.getIntExtra(ExportService.EXTRA_PROGRESS, 0),
                    detail = intent.getStringExtra(ExportService.EXTRA_DETAIL).orEmpty(),
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
            if (position < seed.gameStartMs) {
                player.seekTo(seed.gameStartMs)
                playbackPositionMs = seed.gameStartMs
                continue
            }
            if (position >= seed.gameEndMs) {
                player.pause()
                if (position != seed.gameEndMs) player.seekTo(seed.gameEndMs)
                playbackPositionMs = seed.gameEndMs
                previewEndMs = null
                continue
            }
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
            text = { Text("All boundary, keep/remove, suppression, score, marker, manual-cut, and ignored-section changes will be discarded.") },
            confirmButton = {
                TextButton(onClick = {
                    store.clear()
                    draft = EditorMath.newDraft(seed)
                    selectedId = draft.cuts.firstOrNull()?.id.orEmpty()
                    selectedSuggestionId = null
                    confirmReset = false
                    message = "Restored inference ranges"
                }) { Text("Reset") }
            },
            dismissButton = { TextButton(onClick = { confirmReset = false }) { Text("Cancel") } },
        )
    }

    Box(Modifier.fillMaxSize()) {
        Scaffold(containerColor = Paper) { scaffoldPadding ->
            Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(scaffoldPadding)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Box(Modifier.guidedTourTarget("editor-header", guidedTourTargets)) {
                    sourceControls(
                        EditorProjectSummary(
                            width = seed.width,
                            height = seed.height,
                            sourceDurationMs = seed.durationMs,
                            outputDurationMs = totalFinalMs,
                            kept = effectiveIds.size,
                            removed = removedCount,
                            ignored = ignoredCutCount,
                        ),
                    )
                }

            if (sourceAvailable == false || relinkMessage != null) {
                SectionCard(
                    if (sourceAvailable == false) "SOURCE UNAVAILABLE" else "SOURCE RE-LINKED",
                    if (sourceAvailable == false) seed.displayName else "Playback and exports restored",
                ) {
                    Text(
                        relinkMessage
                            ?: "VolleyCut cannot open the saved video location. Choose the original recording again to restore playback and MP4 export.",
                        color = if (relinkFailed) Danger else Muted,
                        fontSize = 12.sp,
                    )
                    if (sourceAvailable == false) {
                        Button(
                            enabled = !relinkingSource,
                            onClick = {
                                store.save(draft)
                                onRelink()
                            },
                        ) {
                            Text(if (relinkingSource) "Validating…" else "Re-link video")
                        }
                    }
                }
            }

            SectionCard(
                "OUTPUT",
                "Padding and retained short gaps",
                modifier = Modifier.guidedTourTarget("editor-output", guidedTourTargets),
            ) {
                Column(
                    Modifier.guidedTourTarget("editor-suppression", guidedTourTargets),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    if (seed.suppression == null) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text("Suppression suggestions", fontWeight = FontWeight.SemiBold)
                            Text(
                                "Prepare from the saved feature cache; video and audio are not decoded again.",
                                fontSize = 12.sp,
                                color = Muted,
                            )
                        }
                        OutlinedButton(
                            enabled = !suppressionPreparing,
                            onClick = {
                                suppressionPreparing = true
                                scope.launch {
                                    val result = runCatching {
                                        withContext(Dispatchers.IO) {
                                            SuppressionAugmenter.augment(context, project)
                                        }
                                    }
                                    suppressionPreparing = false
                                    result.onSuccess {
                                        onProjectUpdated(it)
                                        message = "Suppression suggestions are ready"
                                    }.onFailure {
                                        message = it.message ?: "Could not prepare suppression suggestions"
                                    }
                                }
                            },
                        ) { Text(if (suppressionPreparing) "Preparing…" else "Prepare") }
                    }
                } else {
                    Text("Suppression", fontWeight = FontWeight.SemiBold)
                    Row(
                        Modifier.horizontalScroll(rememberScrollState()),
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        SuppressionPolicyEngine.Policy.values().forEach { policy ->
                            FilterChip(
                                selected = draft.selectedSuppressionPolicy == policy,
                                onClick = {
                                    updateDraft { it.copy(selectedSuppressionPolicy = policy) }
                                    if (policy == SuppressionPolicyEngine.Policy.NONE) {
                                        selectedSuggestionId = null
                                    }
                                },
                                label = { Text(policy.label, fontSize = 11.sp) },
                            )
                        }
                    }
                    if (suppressionPoliciesEqual) {
                        Text(
                            "All three levels produce the same ${seed.suppression.suggestions().size} suggestions for this game.",
                            fontSize = 11.sp,
                            color = Muted,
                        )
                    }
                    Text("New suggestions", fontWeight = FontWeight.SemiBold, fontSize = 13.sp)
                    Row(
                        Modifier.horizontalScroll(rememberScrollState()),
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        SuppressionInitialBehavior.entries.forEach { behavior ->
                            FilterChip(
                                selected = draft.suppressionInitialBehavior == behavior,
                                onClick = {
                                    updateDraft { it.copy(suppressionInitialBehavior = behavior) }
                                },
                                label = { Text(behavior.label, fontSize = 11.sp) },
                            )
                        }
                    }
                    Text(
                        if (draft.suppressionInitialBehavior == SuppressionInitialBehavior.HIGHLIGHT_ONLY) {
                            "Untouched suggestions stay in the output until you choose Suppress."
                        } else {
                            "Untouched suggestions start disabled; edited ranges stay protected."
                        },
                        fontSize = 11.sp,
                        color = Muted,
                    )
                    if (draft.selectedSuppressionPolicy != SuppressionPolicyEngine.Policy.NONE) {
                        Text(
                            "${draft.selectedSuppressionPolicy.label} · ${activeSuggestions.size} suggestions · $appliedSuggestionCount applied · ${compactTime(baselineTotalMs - totalFinalMs)} less",
                            fontSize = 12.sp,
                            color = Muted,
                        )
                        if (activeSuggestions.isEmpty()) {
                            Text("No suppression suggestions for this game", color = Muted, fontSize = 12.sp)
                        }
                    }
                }
                }
                Column(Modifier.guidedTourTarget("editor-padding", guidedTourTargets)) {
                    PaddingControl("Before", draft.beforePaddingMs) { before ->
                        updateDraft { EditorMath.applyPadding(
                            it, before, it.afterPaddingMs, seed.durationMs,
                            seed.gameStartMs, seed.gameEndMs,
                        ) }
                    }
                    PaddingControl("After", draft.afterPaddingMs) { after ->
                        updateDraft { EditorMath.applyPadding(
                            it, it.beforePaddingMs, after, seed.durationMs,
                            seed.gameStartMs, seed.gameEndMs,
                        ) }
                    }
                }
                Box(Modifier.guidedTourTarget("editor-join-gaps", guidedTourTargets)) {
                    PaddingControl("Join gaps under", draft.joinGapMs) { joinGap ->
                        updateDraft { it.copy(joinGapMs = joinGap.coerceIn(0, MAX_JOIN_GAP_MS)) }
                        message = if (joinGap == 0L) "Short-gap joining disabled"
                        else "Keeping export gaps shorter than ${String.format(Locale.US, "%.1f", joinGap / 1_000.0)} seconds"
                    }
                }
                Text(
                    "Padding applies to inferred ranges. Gray gaps shorter than this setting remain in preview and export.",
                    fontSize = 11.sp,
                    color = Muted,
                )
                Row(
                    modifier = Modifier.guidedTourTarget("editor-final-preview", guidedTourTargets),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(Modifier.weight(1f)) {
                        Text("Play final cut only", fontWeight = FontWeight.SemiBold)
                        Text("Skip removed, ignored, and gaps at or above the join setting", fontSize = 12.sp, color = Muted)
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
                if (draft.scoreTracking.enabled) {
                    Row(
                        modifier = Modifier
                            .padding(start = 18.dp)
                            .guidedTourTarget("editor-score-overlay", guidedTourTargets),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text("Render score on final video", fontWeight = FontWeight.SemiBold)
                            Text("Preview now and include in the next MP4 export", fontSize = 12.sp, color = Muted)
                        }
                        Switch(
                            checked = draft.renderScoreOverlay,
                            onCheckedChange = { enabled ->
                                updateDraft {
                                    it.copy(
                                        renderScoreOverlay = enabled,
                                        renderScoreTimeline = if (enabled) it.renderScoreTimeline else false,
                                    )
                                }
                            },
                        )
                    }
                    if (draft.renderScoreOverlay) {
                        Row(
                            modifier = Modifier.padding(start = 36.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Column(Modifier.weight(1f)) {
                                Text("Render point timeline", fontWeight = FontWeight.SemiBold)
                                Text(
                                    "Show the two team rails beside the score when a new point starts",
                                    fontSize = 12.sp,
                                    color = Muted,
                                )
                            }
                            Switch(
                                checked = draft.renderScoreTimeline,
                                onCheckedChange = { enabled ->
                                    updateDraft {
                                        it.copy(renderScoreTimeline = it.renderScoreOverlay && enabled)
                                    }
                                },
                            )
                        }
                    }
                }
            }

            ScoreTrackingPanel(
                    enabled = draft.scoreTracking.enabled,
                    tracking = draft.scoreTracking,
                    visibleTracking = preparedScoreOverlay.tracking,
                    visibleScore = visibleScore,
                    selectedMarkerId = selectedScoreMarkerId,
                    manualServingSide = manualServingSide,
                    currentTimestampMs = playbackPositionMs,
                    servingSideStatus = project.servingSideStatus,
                    servingSideError = project.servingSideError,
                    servingSideProgress = servingSideProgress,
                    servingSideProgressDetail = servingSideProgressDetail,
                    servingSideStepMeasurements = servingSideStepMeasurements,
                    sideSwitchEnabled = project.sideSwitchEnabled,
                    modifier = Modifier.guidedTourTarget("editor-score-panel", guidedTourTargets),
                    toggleModifier = Modifier.guidedTourTarget("editor-score-toggle", guidedTourTargets),
                    onEnabledChange = { enabled ->
                        updateDraft { it.copy(scoreTracking = it.scoreTracking.copy(enabled = enabled)) }
                        val missingRequestedScoreOutput = project.servingSide == null ||
                            (project.sideSwitchEnabled && project.sideSwitch == null)
                        if (enabled && missingRequestedScoreOutput &&
                            project.servingSideStatus != ServingSideAnalysisStatus.QUEUED &&
                            project.servingSideStatus != ServingSideAnalysisStatus.ANALYZING
                        ) {
                            if (sourceAvailable == false) {
                                message = "Re-link the source video to prepare score tracking"
                            } else {
                                message = "Queued score-tracking feature generation"
                                scope.launch {
                                    val queued = withContext(Dispatchers.IO) {
                                        NativeProjectStore.updateServingSideStatus(
                                            context,
                                            project.id,
                                            ServingSideAnalysisStatus.QUEUED,
                                        )
                                    }
                                    queued?.let(onProjectUpdated)
                                    ProjectAnalysisService.enqueueServingSide(context, project.id)
                                }
                            }
                        }
                    },
                    onTracking = { tracking -> updateDraft { it.copy(scoreTracking = tracking) } },
                    onSelect = { markerId, timestampMs ->
                        selectedScoreMarkerId = markerId
                        seekTo(timestampMs)
                    },
                    onManualServingSide = { manualServingSide = it },
                )

            Card(
                modifier = Modifier.guidedTourTarget("editor-video", guidedTourTargets),
                colors = CardDefaults.cardColors(containerColor = Color.Black),
            ) {
                BoxWithConstraints(Modifier.fillMaxWidth().height(176.dp)) {
                    ContentFrame(
                        player = player,
                        modifier = Modifier.fillMaxSize(),
                        surfaceType = SURFACE_TYPE_TEXTURE_VIEW,
                    )
                    if (draft.renderScoreOverlay && draft.scoreTracking.enabled) {
                        val (displayWidth, displayHeight) = scoreOverlayDisplaySize(
                            seed.width, seed.height, seed.rotation,
                        )
                        val videoAspect = displayWidth.toFloat() / displayHeight.coerceAtLeast(1)
                        val containerAspect = maxWidth.value / maxHeight.value.coerceAtLeast(.001f)
                        val videoModifier = if (containerAspect > videoAspect) {
                            Modifier.height(maxHeight).width(maxHeight * videoAspect)
                        } else {
                            Modifier.width(maxWidth).height(maxWidth / videoAspect)
                        }
                        ScoreOverlayPreview(
                            snapshot = ScoreOverlay.snapshot(preparedScoreOverlay, playbackPositionMs),
                            timeline = ScoreOverlay.pointTimelineSnapshot(preparedScoreOverlay, playbackPositionMs),
                            renderTimeline = draft.renderScoreTimeline,
                            modifier = videoModifier.align(Alignment.Center),
                        )
                    }
                }
            }
            PlayerControls(
                playing = isPlaying,
                positionMs = playbackPositionMs,
                durationMs = seed.gameEndMs,
                playbackRate = draft.playbackRate,
                onToggle = {
                    previewEndMs = null
                    if (player.isPlaying) player.pause() else {
                        if (player.currentPosition >= seed.gameEndMs) seekTo(seed.gameStartMs)
                        if (draft.finalPreviewEnabled) {
                            EditorMath.nextFinalTime(finalIntervals, player.currentPosition)
                                ?.let { if (it != player.currentPosition) seekTo(it) }
                        }
                        player.play()
                    }
                },
                onSeekBy = { seekTo(player.currentPosition + it) },
                onRate = { rate -> updateDraft { it.copy(playbackRate = rate) } },
                modifier = Modifier.guidedTourTarget("editor-transport", guidedTourTargets),
            )

            SectionCard(
                "GAME WINDOW",
                "Tap a range · gray = joined gap",
                compact = true,
                modifier = Modifier.guidedTourTarget("editor-overview", guidedTourTargets),
            ) {
                ConfidenceControl(draft.confidenceReviewThreshold, lowConfidence.size, disagreementCount,
                    suppressionCount = activeSuggestions.size,
                    compact = true, onChange = { threshold ->
                    updateDraft { it.copy(confidenceReviewThreshold = threshold) }
                }, onReviewNext = {
                    if (lowConfidence.isNotEmpty()) {
                        val current = lowConfidence.indexOfFirst { it.id == selected?.id }
                        val next = if (current >= 0) lowConfidence[(current + 1) % lowConfidence.size]
                        else lowConfidence.firstOrNull { it.keepStartMs >= playbackPositionMs } ?: lowConfidence.first()
                        selectedId = next.id
                        seekTo(next.keepStartMs)
                        message = if (next.isModelDisagreement()) {
                            "${next.id} was detected by only one model; validate or disable it"
                        } else "${next.id} has ${(next.confidence * 100).roundToInt()}% confidence"
                    }
                }, onSuppressionNext = {
                    EditorMath.nextSuppressionSuggestion(
                        activeSuggestions,
                        playbackPositionMs,
                        selectedSuggestionId,
                    )?.let(::selectSuggestion)
                })
                WholeTimeline(
                    windowStartMs = seed.gameStartMs,
                    windowEndMs = (seed.gameStartMs + seed.gameEndMs) / 2,
                    cuts = sortedCuts,
                    joinedGaps = joinedGaps,
                    ignored = visibleIgnoredIntervals,
                    suggestions = activeSuggestions,
                    appliedSuggestionIds = activeSuggestions.filter {
                        EditorMath.suggestionEffectiveDecision(draft, it) == SuppressionDecision.SUPPRESS
                    }.map { it.fragmentId() }.toSet(),
                    selectedSuggestionId = selectedSuggestionId,
                    selectedId = selected?.id,
                    effectiveIds = effectiveIds,
                    confidenceThreshold = draft.confidenceReviewThreshold,
                    playheadMs = playbackPositionMs,
                    serveMarkers = if (draft.scoreTracking.enabled) preparedScoreOverlay.tracking.serveMarkers else emptyList(),
                    sideSwitchMarkers = if (draft.scoreTracking.enabled) preparedScoreOverlay.tracking.sideSwitchMarkers else emptyList(),
                    selectedScoreMarkerId = selectedScoreMarkerId,
                    onMarkerSelect = { markerId, timestampMs ->
                        selectedScoreMarkerId = markerId
                        seekTo(timestampMs)
                    },
                    onSeek = { time, id, suggestionId ->
                        if (suggestionId != null) {
                            val candidate = suggestionId
                            activeSuggestions.firstOrNull { it.fragmentId() == candidate }
                                ?.let(::selectSuggestion)
                        } else {
                            id?.let {
                                selectedSuggestionId = null
                                selectedId = it
                            }
                            seekTo(time)
                        }
                    },
                )
                TimelineLabels(
                    seed.gameStartMs,
                    (seed.gameStartMs * 3 + seed.gameEndMs) / 4,
                    (seed.gameStartMs + seed.gameEndMs) / 2,
                )
                WholeTimeline(
                    windowStartMs = (seed.gameStartMs + seed.gameEndMs) / 2,
                    windowEndMs = seed.gameEndMs,
                    cuts = sortedCuts,
                    joinedGaps = joinedGaps,
                    ignored = visibleIgnoredIntervals,
                    suggestions = activeSuggestions,
                    appliedSuggestionIds = activeSuggestions.filter {
                        EditorMath.suggestionEffectiveDecision(draft, it) == SuppressionDecision.SUPPRESS
                    }.map { it.fragmentId() }.toSet(),
                    selectedSuggestionId = selectedSuggestionId,
                    selectedId = selected?.id,
                    effectiveIds = effectiveIds,
                    confidenceThreshold = draft.confidenceReviewThreshold,
                    playheadMs = playbackPositionMs,
                    serveMarkers = if (draft.scoreTracking.enabled) preparedScoreOverlay.tracking.serveMarkers else emptyList(),
                    sideSwitchMarkers = if (draft.scoreTracking.enabled) preparedScoreOverlay.tracking.sideSwitchMarkers else emptyList(),
                    selectedScoreMarkerId = selectedScoreMarkerId,
                    onMarkerSelect = { markerId, timestampMs ->
                        selectedScoreMarkerId = markerId
                        seekTo(timestampMs)
                    },
                    onSeek = { time, id, suggestionId ->
                        if (suggestionId != null) {
                            val candidate = suggestionId
                            activeSuggestions.firstOrNull { it.fragmentId() == candidate }
                                ?.let(::selectSuggestion)
                        } else {
                            id?.let {
                                selectedSuggestionId = null
                                selectedId = it
                            }
                            seekTo(time)
                        }
                    },
                )
                TimelineLabels(
                    (seed.gameStartMs + seed.gameEndMs) / 2,
                    (seed.gameStartMs + seed.gameEndMs * 3) / 4,
                    seed.gameEndMs,
                )
            }

            if (selectedSuggestion != null) {
                val effectiveDecision = EditorMath.suggestionEffectiveDecision(draft, selectedSuggestion)
                val effectiveScope = EditorMath.suggestionEffectiveScope(draft, selectedSuggestion)
                val explicitDecision = draft.suppressionDecisionOverrides[selectedSuggestion.logicalId()]
                val protectedByEdit = explicitDecision == null && draft.userTouchedCutIds.any { id ->
                    draft.cuts.firstOrNull { it.id == id }?.let { cut ->
                        cut.coreStartMs < selectedSuggestion.endMs() &&
                            selectedSuggestion.startMs() < cut.coreEndMs
                    } == true
                }
                SectionCard(
                    "SUPPRESSION SUGGESTION",
                    "${selectedSuggestionIndex + 1} / ${activeSuggestions.size} · ${(selectedSuggestion.score() * 100).roundToInt()}% model score",
                    compact = true,
                ) {
                    Text(
                        "${preciseTime(selectedSuggestion.startMs())}–${preciseTime(selectedSuggestion.endMs())}",
                        fontFamily = FontFamily.Monospace,
                        fontWeight = FontWeight.Bold,
                    )
                    Text(
                        when {
                            protectedByEdit -> "Edited rally—kept"
                            effectiveDecision == SuppressionDecision.SUPPRESS ->
                                if (effectiveScope == SuppressionScope.WHOLE_RALLY) {
                                    "Whole rally suppressed"
                                } else "Veto region suppressed"
                            else -> "Suggestion kept"
                        },
                        color = if (effectiveDecision == SuppressionDecision.SUPPRESS) SuppressionRed else Muted,
                        fontWeight = FontWeight.SemiBold,
                    )
                    Text("Suppress scope", fontWeight = FontWeight.SemiBold, fontSize = 13.sp)
                    Row(
                        Modifier.horizontalScroll(rememberScrollState()),
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        SuppressionScope.entries.forEach { scopeOption ->
                            FilterChip(
                                selected = effectiveScope == scopeOption,
                                onClick = {
                                    updateDraft { current -> current.copy(
                                        suppressionScopeOverrides =
                                            if (scopeOption == SuppressionScope.WHOLE_RALLY) {
                                                current.suppressionScopeOverrides - selectedSuggestion.logicalId()
                                            } else {
                                                current.suppressionScopeOverrides +
                                                    (selectedSuggestion.logicalId() to scopeOption)
                                            },
                                    ) }
                                },
                                label = { Text(scopeOption.label, fontSize = 11.sp) },
                            )
                        }
                    }
                    Text(
                        if (effectiveScope == SuppressionScope.WHOLE_RALLY) {
                            "Removes the entire inferred rally, including its padding."
                        } else {
                            "Removes only the red veto region; the rest of the rally stays."
                        },
                        color = Muted,
                        fontSize = 11.sp,
                    )
                    Row(
                        Modifier.horizontalScroll(rememberScrollState()),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        SmallButton("Previous") {
                            selectSuggestion(activeSuggestions[
                                (selectedSuggestionIndex - 1 + activeSuggestions.size) % activeSuggestions.size
                            ])
                        }
                        SmallButton("Next") {
                            selectSuggestion(activeSuggestions[(selectedSuggestionIndex + 1) % activeSuggestions.size])
                        }
                        Button(onClick = {
                            updateDraft { current -> current.copy(
                                suppressionDecisionOverrides = current.suppressionDecisionOverrides +
                                    (selectedSuggestion.logicalId() to SuppressionDecision.SUPPRESS),
                            ) }
                        }) { Text("Suppress") }
                        OutlinedButton(onClick = {
                            updateDraft { current -> current.copy(
                                suppressionDecisionOverrides = current.suppressionDecisionOverrides +
                                    (selectedSuggestion.logicalId() to SuppressionDecision.KEEP),
                            ) }
                        }) { Text("Keep") }
                    }
                }
            }

            SectionCard(
                "FOCUSED RANGE",
                selected?.let {
                    val hasSuggestion = activeSuggestions.any { suggestion ->
                        it.coreStartMs < suggestion.endMs() && suggestion.startMs() < it.coreEndMs
                    }
                    "${it.id} · ${if (it.origin == CutOrigin.MANUAL) "Manual" else "${it.modelAgreementLabel()} · ${(it.confidence * 100).roundToInt()}% review confidence"}" +
                        if (hasSuggestion) " · suppression suggestion" else ""
                }
                    ?: "No range selected",
                compact = true,
                modifier = Modifier.guidedTourTarget("editor-focus", guidedTourTargets),
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
                        lowConfidence = selected.isModelDisagreement() ||
                            selected.confidence < draft.confidenceReviewThreshold,
                        suggestions = activeSuggestions,
                        appliedSuggestionIds = activeSuggestions.filter {
                            EditorMath.suggestionEffectiveDecision(draft, it) == SuppressionDecision.SUPPRESS
                        }.map { it.fragmentId() }.toSet(),
                        onSeek = ::seekTo,
                        onSuggestionSelect = { id ->
                            activeSuggestions.firstOrNull { it.fragmentId() == id }
                                ?.let(::selectSuggestion)
                        },
                        onStartChange = { setKeepBoundary("start", it) },
                        onEndChange = { setKeepBoundary("end", it) },
                    )
                    Text(
                        if (selected.origin == CutOrigin.INFERRED) {
                            "Set the green rally edges first. Padding stays outside them."
                        } else {
                            "Set this manual rally's exact start and end."
                        },
                        fontSize = 12.sp,
                        color = Muted,
                    )
                    Row(
                        Modifier.horizontalScroll(rememberScrollState()),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        OutlinedButton(
                            enabled = playbackPositionMs <= selected.coreEndMs - MIN_MARK_MS,
                            onClick = { setCoreBoundary("start", playbackPositionMs) },
                        ) { Text("Set rally start here") }
                        Button(
                            enabled = playbackPositionMs >= selected.coreStartMs + MIN_MARK_MS &&
                                playbackPositionMs <= selected.coreEndMs - MIN_MARK_MS,
                            onClick = ::splitSelectedAtPlayhead,
                        ) { Text("Split at playhead") }
                        OutlinedButton(
                            enabled = playbackPositionMs >= selected.coreStartMs + MIN_MARK_MS,
                            onClick = { setCoreBoundary("end", playbackPositionMs) },
                        ) { Text("Set rally end here") }
                    }
                    RallyRangeSlider(
                        cut = selected,
                        window = detailWindow,
                        onRangeChange = { startMs, endMs ->
                            updateDraft { current -> EditorMath.setCoreRange(
                                current,
                                selected.id,
                                startMs,
                                endMs,
                                seed.gameStartMs,
                                seed.gameEndMs,
                            ) }
                        },
                        modifier = Modifier.guidedTourTarget("editor-trim", guidedTourTargets),
                    )
                    if (selected.origin == CutOrigin.INFERRED) {
                        Text("OUTPUT EDGES (RALLY + PADDING)", fontSize = 11.sp, color = Muted)
                        BoundaryControls("Output start", selected.keepStartMs) { delta ->
                            setKeepBoundary("start", selected.keepStartMs + delta)
                        }
                        BoundaryControls("Output end", selected.keepEndMs) { delta ->
                            setKeepBoundary("end", selected.keepEndMs + delta)
                        }
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
                                    keepStartMs = (cut.coreStartMs - draft.beforePaddingMs)
                                        .coerceAtLeast(seed.gameStartMs),
                                    keepEndMs = (cut.coreEndMs + draft.afterPaddingMs)
                                        .coerceAtMost(seed.gameEndMs),
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
                modifier = Modifier.guidedTourTarget("editor-marking", guidedTourTargets),
            )

            if (message.isNotBlank()) {
                Surface(color = Color(0xFFFFE2D8), shape = RoundedCornerShape(10.dp)) {
                    Text(message, Modifier.padding(12.dp), fontSize = 13.sp)
                }
            }

            if (visibleIgnoredIntervals.isNotEmpty()) {
                SectionCard("IGNORED SOURCE", "Removed from the derived edit list") {
                    visibleIgnoredIntervals.sortedBy { it.startMs }.forEach { interval ->
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

            SectionCard(
                "ALL CUTS",
                "${effectiveIds.size} kept · $ignoredCutCount fully ignored",
                modifier = Modifier.guidedTourTarget("editor-cuts", guidedTourTargets),
            ) {
                Column(
                    Modifier
                        .fillMaxWidth()
                        .heightIn(max = 320.dp)
                        .border(1.dp, Rail, RoundedCornerShape(4.dp))
                        .background(Paper, RoundedCornerShape(4.dp))
                        .verticalScroll(rememberScrollState())
                        .padding(4.dp)
                        .semantics { contentDescription = "All cuts list" },
                ) {
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
                                if (cut.origin == CutOrigin.MANUAL) "MANUAL"
                                else if (cut.isModelDisagreement()) "CHECK"
                                else "${(cut.confidence * 100).roundToInt()}%",
                                fontSize = 12.sp,
                                color = if (cut.isModelDisagreement() ||
                                    cut.confidence < draft.confidenceReviewThreshold
                                ) Warning else Muted,
                            )
                            TextButton(onClick = { updateCut(cut.id) { it.copy(included = !it.included) } }) {
                                Text(state, color = if (state == "Keep") Green else Danger)
                            }
                        }
                        if (index < sortedCuts.lastIndex) HorizontalDivider(color = Rail)
                    }
                }
            }

            SectionCard(
                "EXPORT",
                "MP4 video",
                modifier = Modifier.guidedTourTarget("editor-export", guidedTourTargets),
            ) {
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
                } else if (exportState.status == "queued") {
                    Text(exportState.detail, color = Green, fontFamily = FontFamily.Monospace, fontSize = 12.sp)
                } else if (exportState.detail.isNotBlank()) {
                    Text(exportState.detail, color = if (exportState.status == "failed") Danger else Green)
                }
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        enabled = finalIntervals.isNotEmpty() && !exportPending,
                        onClick = {
                            pendingExportIntervals = finalIntervals
                            exportLauncher.launch(exportFilename(seed.displayName))
                        },
                        modifier = Modifier.weight(1f),
                    ) { Text(if (exportPending) "Export queued" else "Queue MP4 export") }
                    OutlinedButton(
                        enabled = finalIntervals.isNotEmpty(),
                        onClick = { editListLauncher.launch(editListFilename(seed.displayName)) },
                        modifier = Modifier.weight(1f),
                    ) { Text("Save edit list") }
                }
                if (exportPending) {
                    OutlinedButton(
                        onClick = {
                            context.startService(
                                Intent(context, ExportService::class.java)
                                    .setAction(ExportService.ACTION_CANCEL)
                                    .putExtra(ExportService.EXTRA_JOB_ID, exportState.jobId),
                            )
                        },
                        modifier = Modifier.fillMaxWidth(),
                    ) { Text(if (exportState.status == "queued") "Remove from queue" else "Cancel export", color = Danger) }
                }
                TextButton(
                    enabled = !feedbackExporting,
                    onClick = {
                        feedbackSaveLauncher.launch(ModelFeedbackExporter.filename(seed.displayName))
                    },
                    modifier = Modifier.align(Alignment.Start),
                ) {
                    Text(if (feedbackExporting) "Exporting feedback…" else "Export model feedback", color = Muted)
                }
                Text(
                    "Model feedback includes source-aligned audiovisual features, probability traces, initial ranges, and your corrections. It never includes video bytes.",
                    color = Muted,
                    fontSize = 11.sp,
                )
            }

            AnalysisMeasurementsCard(project.analysisMeasurements)

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = { confirmReset = true }) { Text("Reset editor") }
                if (BuildConfig.DEBUG) {
                    TextButton(onClick = { context.startActivity(Intent(context, MainActivity::class.java)) }) {
                        Text("Benchmark tools")
                    }
                }
            }
            LegalFooter()
            Spacer(Modifier.height(20.dp))
            }
        }
        GuidedTour(
            GuidedTourStage.EDITOR,
            guidedTourTargets,
            scoreTrackingEnabled = draft.scoreTracking.enabled,
            restartSignal = guidedTourRestartSignal,
        )
    }
}

@Composable
internal fun ScoreTrackingPanel(
    enabled: Boolean,
    tracking: ScoreTracking,
    visibleTracking: ScoreTracking,
    visibleScore: DerivedScore,
    selectedMarkerId: String?,
    manualServingSide: ServingSide,
    currentTimestampMs: Long,
    servingSideStatus: ServingSideAnalysisStatus,
    servingSideError: String?,
    servingSideProgress: Float?,
    servingSideProgressDetail: String?,
    servingSideStepMeasurements: List<InferenceStepMeasurement> = emptyList(),
    sideSwitchEnabled: Boolean = true,
    onEnabledChange: (Boolean) -> Unit,
    onTracking: (ScoreTracking) -> Unit,
    onSelect: (String, Long) -> Unit,
    onManualServingSide: (ServingSide) -> Unit,
    modifier: Modifier = Modifier,
    toggleModifier: Modifier = Modifier,
) {
    val sortedServes = visibleTracking.serveMarkers.sortedWith(
        compareBy<ServeMarker> { it.timestampMs }.thenBy { it.id },
    )
    val reviewMarkers = sortedServes.filter { it.side == ServingSide.REVIEW }
    val reviewMarkerIds = reviewMarkers.mapTo(mutableSetOf()) { it.id }
    val selected = sortedServes.firstOrNull { it.id == selectedMarkerId }
    val selectedIndex = selected?.let {
        sortedServes.indexOfFirst { marker -> marker.id == it.id }
    } ?: -1
    Card(
        modifier = modifier
            .testTag("score-tracking-panel")
            .semantics { contentDescription = "Score tracking controls" },
        colors = CardDefaults.cardColors(containerColor = Color.White),
    ) {
        Column(
            Modifier.fillMaxWidth().padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(11.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Text(
                        "SCORE TRACKING · BETA",
                        color = Orange,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Black,
                        letterSpacing = .8.sp,
                    )
                    Text("Serve and team-side history", fontSize = 18.sp, fontWeight = FontWeight.Bold)
                }
                Switch(
                    checked = enabled,
                    onCheckedChange = onEnabledChange,
                    modifier = toggleModifier,
                )
            }
            if (!enabled) {
                Text(
                    "Use inferred serve and team-switch markers to track points. This choice is saved with the project.",
                    color = Muted,
                    fontSize = 12.sp,
                )
            } else {
                val reviewCount = reviewMarkers.size
                Text(
                    "${sortedServes.size} serves · ${visibleTracking.sideSwitchMarkers.size} switches" +
                        if (reviewCount > 0) " · $reviewCount need review" else "",
                    color = Muted,
                    fontFamily = FontFamily.Monospace,
                    fontSize = 12.sp,
                )

                Surface(
                    modifier = Modifier.fillMaxWidth().border(1.dp, Rail, RoundedCornerShape(8.dp)),
                    color = Paper,
                    shape = RoundedCornerShape(8.dp),
                ) {
                    Column(
                        Modifier.fillMaxWidth().padding(10.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            OutlinedTextField(
                                value = tracking.team1Name,
                                onValueChange = { name ->
                                    onTracking(tracking.copy(team1Name = name.ifEmpty { "Team 1" }))
                                },
                                label = { Text("Team 1 · starts near") },
                                singleLine = true,
                                modifier = Modifier.weight(1f),
                            )
                            Text(
                                visibleScore.team1Score.toString(),
                                Modifier.width(54.dp),
                                color = Green,
                                fontFamily = FontFamily.Monospace,
                                fontSize = 30.sp,
                                fontWeight = FontWeight.Bold,
                                textAlign = TextAlign.Center,
                            )
                        }
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            OutlinedTextField(
                                value = tracking.team2Name,
                                onValueChange = { name ->
                                    onTracking(tracking.copy(team2Name = name.ifEmpty { "Team 2" }))
                                },
                                label = { Text("Team 2 · starts far") },
                                singleLine = true,
                                modifier = Modifier.weight(1f),
                            )
                            Text(
                                visibleScore.team2Score.toString(),
                                Modifier.width(54.dp),
                                color = Orange,
                                fontFamily = FontFamily.Monospace,
                                fontSize = 30.sp,
                                fontWeight = FontWeight.Bold,
                                textAlign = TextAlign.Center,
                            )
                        }
                        Text(
                            when (visibleScore.servingTeamId) {
                                ScoreTeamId.TEAM_1 -> "Serving now: ${tracking.team1Name} · ${visibleScore.servingSide?.wireName} side"
                                ScoreTeamId.TEAM_2 -> "Serving now: ${tracking.team2Name} · ${visibleScore.servingSide?.wireName} side"
                                null -> if (visibleScore.servingSide == ServingSide.REVIEW) {
                                    "Serving side needs review"
                                } else "Server not established"
                            },
                            color = Muted,
                            fontFamily = FontFamily.Monospace,
                            fontSize = 12.sp,
                        )
                    }
                }

                val servingSideBusy = servingSideStatus == ServingSideAnalysisStatus.QUEUED ||
                    servingSideStatus == ServingSideAnalysisStatus.ANALYZING
                val servingSideFailed = servingSideStatus == ServingSideAnalysisStatus.ERROR
                val reviewPending = reviewCount > 0
                val reportedProgress = servingSideProgress?.coerceIn(0f, 1f) ?: 0f
                Surface(
                    modifier = Modifier.fillMaxWidth().border(
                        1.dp,
                        when {
                            servingSideFailed -> Orange
                            reviewPending -> Warning
                            else -> PaleGreen
                        },
                        RoundedCornerShape(4.dp),
                    ),
                    color = when {
                        servingSideFailed -> Color(0xFFFFEEE8)
                        reviewPending -> Color(0xFFFFF7E6)
                        else -> Color(0xFFEEF6F0)
                    },
                    shape = RoundedCornerShape(4.dp),
                ) {
                    Column(Modifier.padding(10.dp), verticalArrangement = Arrangement.spacedBy(3.dp)) {
                        if (!servingSideBusy || servingSideStepMeasurements.isEmpty()) {
                            Text(
                                when (servingSideStatus) {
                                    ServingSideAnalysisStatus.QUEUED -> "SCORE TRACKING QUEUED"
                                    ServingSideAnalysisStatus.ANALYZING -> "GENERATING SCORE-TRACKING FEATURES"
                                    ServingSideAnalysisStatus.READY -> if (sideSwitchEnabled) {
                                        "SERVE + SWITCH MARKERS READY"
                                    } else "SERVE MARKERS READY"
                                    ServingSideAnalysisStatus.ERROR -> "SCORE-TRACKING ANALYSIS NEEDS ATTENTION"
                                    ServingSideAnalysisStatus.DISABLED,
                                    ServingSideAnalysisStatus.NOT_RUN -> "SCORE MARKERS NOT PREPARED"
                                },
                                fontFamily = FontFamily.Monospace,
                                fontSize = 11.sp,
                                fontWeight = FontWeight.Bold,
                            )
                            Text(
                                when {
                                    servingSideBusy && !servingSideProgressDetail.isNullOrBlank() ->
                                        servingSideProgressDetail
                                    servingSideBusy -> "This can take a while. You can keep editing while it runs."
                                    servingSideFailed -> servingSideError ?: "Score-tracking analysis failed. Disable and re-enable score tracking to retry."
                                    reviewCount > 0 ->
                                        "$reviewCount model ${if (reviewCount == 1) "verdict needs" else "verdicts need"} review."
                                    servingSideStatus == ServingSideAnalysisStatus.READY ->
                                        if (sideSwitchEnabled) {
                                            "Serve-side and team-switch predictions are ready to edit."
                                        } else "Serve-side predictions are ready; automatic team switches are off."
                                    else -> if (sideSwitchEnabled) {
                                        "Enable score tracking to generate serve-side and team-switch predictions."
                                    } else "Enable score tracking to generate serve-side predictions."
                                },
                                color = Muted,
                                fontFamily = FontFamily.Monospace,
                                fontSize = 10.sp,
                            )
                        }
                        if (servingSideBusy && servingSideStepMeasurements.isEmpty()) {
                            Row(
                                Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                verticalAlignment = Alignment.CenterVertically,
                            ) {
                                LinearProgressIndicator(
                                    progress = { reportedProgress },
                                    modifier = Modifier.weight(1f),
                                )
                                Text(
                                    "${(reportedProgress * 100).roundToInt()}%",
                                    color = Muted,
                                    fontFamily = FontFamily.Monospace,
                                    fontSize = 10.sp,
                                )
                            }
                        }
                        if (servingSideStepMeasurements.isNotEmpty()) {
                            InferenceProgressMeasurementsPanel(
                                steps = servingSideStepMeasurements,
                                compact = true,
                            )
                        }
                        if (reviewPending) {
                            OutlinedButton(
                                onClick = {
                                    val selectedReviewIndex = reviewMarkers.indexOfFirst { it.id == selectedMarkerId }
                                    val next = if (selectedReviewIndex >= 0) {
                                        reviewMarkers[(selectedReviewIndex + 1) % reviewMarkers.size]
                                    } else {
                                        reviewMarkers.firstOrNull { it.timestampMs >= currentTimestampMs }
                                            ?: reviewMarkers.first()
                                    }
                                    onSelect(next.id, next.timestampMs)
                                },
                                modifier = Modifier.align(Alignment.Start),
                            ) {
                                Text("Review next · $reviewCount")
                            }
                        }
                    }
                }

                Surface(
                    modifier = Modifier.fillMaxWidth().border(1.dp, Rail, RoundedCornerShape(4.dp)),
                    color = Color.White,
                    shape = RoundedCornerShape(4.dp),
                ) {
                    Column(Modifier.padding(10.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("SELECTED SERVE", color = Orange, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                        if (selected == null) {
                            Text("Select a ball marker on the timeline", color = Muted, fontSize = 12.sp)
                        } else {
                            Text(
                                "${preciseTime(selected.timestampMs)} · ${selected.side.wireName}",
                                fontFamily = FontFamily.Monospace,
                                fontSize = 20.sp,
                                fontWeight = FontWeight.SemiBold,
                            )
                            selected.modelSide?.let { modelSide ->
                                Text(
                                    "Model: ${modelSide.wireName}" + if (selected.side != modelSide) " · corrected" else "",
                                    color = Muted,
                                    fontSize = 11.sp,
                                )
                            }
                            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                listOf(ServingSide.NEAR to "Near", ServingSide.FAR to "Far").forEach { (side, label) ->
                                    FilterChip(
                                        selected = selected.side == side,
                                        onClick = {
                                            onTracking(tracking.copy(serveMarkers = tracking.serveMarkers.map {
                                                if (it.id == selected.id) it.copy(side = side) else it
                                            }))
                                        },
                                        label = { Text(label) },
                                        modifier = Modifier.weight(1f),
                                    )
                                }
                            }
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Checkbox(
                                    checked = selected.ignorePreviousPoint,
                                    enabled = selectedIndex > 0,
                                    onCheckedChange = { ignored ->
                                        onTracking(tracking.copy(serveMarkers = tracking.serveMarkers.map {
                                            if (it.id == selected.id) it.copy(ignorePreviousPoint = ignored) else it
                                        }))
                                    },
                                )
                                Column {
                                    Text("Ignore / replay previous point", fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
                                    Text("Do not change the score at this serve.", color = Muted, fontSize = 10.sp)
                                }
                            }
                        }
                    }
                }

                Text(
                    "MISSING SERVE AT ${preciseTime(currentTimestampMs)}",
                    color = Orange,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                )
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    listOf(
                        ServingSide.NEAR to "Near",
                        ServingSide.FAR to "Far",
                    ).forEach { (side, label) ->
                        FilterChip(
                            selected = manualServingSide == side,
                            onClick = { onManualServingSide(side) },
                            label = { Text(label) },
                            modifier = Modifier.weight(1f),
                        )
                    }
                }
                OutlinedButton(
                    onClick = {
                        val updated = ScoreReducer.addServe(tracking, currentTimestampMs, manualServingSide)
                        onTracking(updated)
                        updated.serveMarkers.firstOrNull { marker ->
                            tracking.serveMarkers.none { it.id == marker.id }
                        }?.let { onSelect(it.id, it.timestampMs) }
                    },
                    modifier = Modifier.fillMaxWidth(),
                ) { Text("+ Add serve") }
                Button(
                    onClick = {
                        val updated = ScoreReducer.addSideSwitch(tracking, currentTimestampMs)
                        onTracking(updated)
                        updated.sideSwitchMarkers.firstOrNull { marker ->
                            tracking.sideSwitchMarkers.none { it.id == marker.id }
                        }?.let { onSelect(it.id, it.timestampMs) }
                    },
                    modifier = Modifier.fillMaxWidth(),
                ) { Text("⇄ Add team side switch · ${preciseTime(currentTimestampMs)}") }

                val awardedPoints = visibleScore.points.mapIndexedNotNull { index, point ->
                    point.takeIf { it.status == ScorePointStatus.COUNTED && it.winnerTeamId != null }
                        ?.let { index + 1 to it }
                }
                Surface(
                    modifier = Modifier.fillMaxWidth().border(1.dp, Rail, RoundedCornerShape(4.dp)),
                    color = Paper,
                    shape = RoundedCornerShape(4.dp),
                ) {
                    Column(Modifier.padding(10.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text("POINT TIMELINE", color = Orange, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                            Spacer(Modifier.weight(1f))
                            Text("${awardedPoints.size} awarded", color = Muted, fontSize = 10.sp)
                        }
                        if (awardedPoints.isEmpty()) {
                            Text("No points awarded yet", color = Muted, fontSize = 11.sp)
                        } else {
                            Column(
                                Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()),
                                verticalArrangement = Arrangement.spacedBy(3.dp),
                            ) {
                                ScoreTimelineLabelRow("RALLY", awardedPoints.map { it.first.toString() })
                                ScoreTimelinePointRow(
                                    tracking.team1Name,
                                    awardedPoints.map { (_, point) ->
                                        point.team1ScoreAfter.toString().takeIf { point.winnerTeamId == ScoreTeamId.TEAM_1 }
                                    },
                                    Green,
                                )
                                ScoreTimelinePointRow(
                                    tracking.team2Name,
                                    awardedPoints.map { (_, point) ->
                                        point.team2ScoreAfter.toString().takeIf { point.winnerTeamId == ScoreTeamId.TEAM_2 }
                                    },
                                    Orange,
                                )
                            }
                        }
                    }
                }

                val markerRows = buildList<Triple<String, Long, String>> {
                    sortedServes.forEachIndexed { index, marker ->
                        val side = when {
                            marker.side == ServingSide.REVIEW -> "Needs review"
                            index == 0 -> "First serve"
                            else -> marker.side.wireName
                        }
                        add(Triple(marker.id, marker.timestampMs, "$side · ${marker.origin.wireName}"))
                    }
                    visibleTracking.sideSwitchMarkers.forEach { marker ->
                        add(Triple(marker.id, marker.timestampMs, "Team side switch"))
                    }
                }.sortedWith(compareBy<Triple<String, Long, String>> { it.second }.thenBy { it.first })
                val markerListState = rememberLazyListState()
                LaunchedEffect(selectedMarkerId, markerRows) {
                    val selectedRowIndex = markerRows.indexOfFirst { it.first == selectedMarkerId }
                    if (selectedRowIndex >= 0) markerListState.animateScrollToItem(selectedRowIndex)
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("MARKERS", color = Orange, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                    Spacer(Modifier.weight(1f))
                    Text("${markerRows.size} · tap to select", color = Muted, fontSize = 10.sp)
                }
                LazyColumn(
                    Modifier
                        .fillMaxWidth()
                        .heightIn(max = 252.dp)
                        .border(1.dp, Rail, RoundedCornerShape(4.dp))
                        .background(Paper, RoundedCornerShape(4.dp))
                        .padding(4.dp)
                        .semantics { contentDescription = "Score marker list" },
                    state = markerListState,
                    verticalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    if (markerRows.isEmpty()) {
                        item {
                            Text("No score markers have been added.", Modifier.padding(10.dp), color = Muted, fontSize = 11.sp)
                        }
                    }
                    items(markerRows, key = { it.first }) { (id, timestamp, label) ->
                        val selectedRow = id == selectedMarkerId
                        val needsReview = id in reviewMarkerIds
                        Row(
                            Modifier
                                .fillMaxWidth()
                                .border(
                                    if (selectedRow) 2.dp else 1.dp,
                                    when {
                                        selectedRow -> Orange
                                        needsReview -> Warning
                                        else -> Rail
                                    },
                                    RoundedCornerShape(3.dp),
                                )
                                .background(
                                    if (needsReview) Color(0xFFFFF7E6) else Color.White,
                                    RoundedCornerShape(3.dp),
                                )
                                .semantics {
                                    contentDescription = if (needsReview) {
                                        "Score marker needs review"
                                    } else {
                                        "Score marker"
                                    }
                                },
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            TextButton(
                                onClick = { onSelect(id, timestamp) },
                                modifier = Modifier.weight(1f),
                            ) {
                                Column(Modifier.fillMaxWidth()) {
                                    Text(
                                        "${if (tracking.serveMarkers.any { it.id == id }) "🏐" else "⇄"} ${preciseTime(timestamp)}",
                                        color = Ink,
                                        fontFamily = FontFamily.Monospace,
                                        fontWeight = FontWeight.SemiBold,
                                    )
                                    Text(label, color = Muted, fontSize = 10.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
                                }
                            }
                            TextButton(
                                onClick = {
                                    onTracking(
                                        if (tracking.serveMarkers.any { it.id == id }) ScoreReducer.removeServe(tracking, id)
                                        else ScoreReducer.removeSideSwitch(tracking, id),
                                    )
                                },
                            ) { Text("Remove", color = Danger, fontSize = 11.sp) }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ScoreTimelineLabelRow(label: String, values: List<String>) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(label, Modifier.width(72.dp), color = Muted, fontSize = 9.sp, fontWeight = FontWeight.Bold)
        values.forEach { value ->
            Text(
                value,
                Modifier.width(32.dp),
                color = Muted,
                fontFamily = FontFamily.Monospace,
                fontSize = 9.sp,
                textAlign = TextAlign.Center,
            )
        }
    }
}

@Composable
private fun ScoreTimelinePointRow(label: String, values: List<String?>, accent: Color) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(
            label,
            Modifier.width(72.dp),
            color = Ink,
            fontSize = 10.sp,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
        values.forEach { value ->
            Box(Modifier.width(32.dp).height(28.dp), contentAlignment = Alignment.Center) {
                if (value == null) {
                    HorizontalDivider(Modifier.width(26.dp), color = Rail)
                } else {
                    Box(
                        Modifier
                            .size(24.dp)
                            .background(accent.copy(alpha = .14f), RoundedCornerShape(50))
                            .border(2.dp, accent, RoundedCornerShape(50)),
                        contentAlignment = Alignment.Center,
                    ) {
                        Text(value, color = Ink, fontFamily = FontFamily.Monospace, fontSize = 9.sp)
                    }
                }
            }
        }
    }
}

@Composable
private fun ScoreOverlayPreview(
    snapshot: ScoreOverlaySnapshot,
    timeline: ScorePointTimelineSnapshot,
    renderTimeline: Boolean,
    modifier: Modifier = Modifier,
) {
    val density = LocalDensity.current
    val measurer = rememberTextMeasurer()
    BoxWithConstraints(modifier) {
        val videoWidthPx = with(density) { maxWidth.toPx().roundToInt() }
        val videoHeightPx = with(density) { maxHeight.toPx().roundToInt() }
        val layout = ScoreOverlay.layout(videoWidthPx, videoHeightPx, snapshot) { text, fontSize ->
            measurer.measure(
                text,
                style = androidx.compose.ui.text.TextStyle(
                    fontSize = with(density) { fontSize.toSp() },
                    fontWeight = FontWeight.Bold,
                ),
            ).size.width.toFloat()
        }
        fun pxDp(value: Int) = with(density) { value.toDp() }
        val rowShape = RoundedCornerShape(bottomEnd = pxDp(layout.radius))
        Box(
            Modifier
                .width(pxDp(layout.width))
                .height(pxDp(layout.height))
                .clip(rowShape)
                .border(pxDp(layout.borderWidth), Color(ScoreOverlay.BORDER_COLOR), rowShape),
        ) {
            Row(Modifier.fillMaxSize(), verticalAlignment = Alignment.CenterVertically) {
                fun Modifier.cell(widthPx: Int, color: Long) = width(pxDp(widthPx))
                    .fillMaxHeight()
                    .background(Color(color))
                @Composable
                fun Cell(text: String, widthPx: Int, background: Long, foreground: Color, padded: Boolean) {
                    Box(
                        Modifier.cell(widthPx, background).then(
                            if (padded) Modifier.padding(horizontal = pxDp(layout.horizontalPadding)) else Modifier,
                        ),
                        contentAlignment = Alignment.Center,
                    ) {
                        Text(
                            text,
                            color = foreground,
                            textAlign = TextAlign.Center,
                            fontSize = with(density) { layout.fontSize.toSp() },
                            fontWeight = FontWeight.Bold,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                    }
                }
                Cell(
                    ScoreOverlay.formatTeamLabel(
                        snapshot.team1Name,
                        snapshot.servingTeamId == ScoreTeamId.TEAM_1,
                    ),
                    layout.team1Width,
                    ScoreOverlay.TEAM_1_COLOR,
                    Color.White,
                    true,
                )
                Cell(snapshot.team1ScoreLabel, layout.scoreWidth, ScoreOverlay.SCORE_BACKGROUND_COLOR, Color.Black, false)
                Cell(
                    ScoreOverlay.formatTeamLabel(
                        snapshot.team2Name,
                        snapshot.servingTeamId == ScoreTeamId.TEAM_2,
                    ),
                    layout.team2Width,
                    ScoreOverlay.TEAM_2_COLOR,
                    Color.White,
                    true,
                )
                Cell(snapshot.team2ScoreLabel, layout.scoreWidth, ScoreOverlay.SCORE_BACKGROUND_COLOR, Color.Black, false)
            }
            Canvas(Modifier.fillMaxSize()) {
                val stroke = layout.borderWidth.toFloat()
                var divider = pxDp(layout.team1Width).toPx()
                drawLine(Color.Black, Offset(divider, 0f), Offset(divider, size.height), stroke)
                divider += pxDp(layout.scoreWidth).toPx()
                drawLine(Color.Black, Offset(divider, 0f), Offset(divider, size.height), stroke)
                divider += pxDp(layout.team2Width).toPx()
                drawLine(Color.Black, Offset(divider, 0f), Offset(divider, size.height), stroke)
            }
        }
        val visibleTimelinePoints = ScoreOverlay.visiblePointTimelineEntries(
            videoWidthPx,
            layout,
            timeline.points,
        )
        if (renderTimeline && visibleTimelinePoints.isNotEmpty() && timeline.opacity > 0f) {
            val pointLayout = ScoreOverlay.pointTimelineLayout(
                videoWidthPx,
                videoHeightPx,
                layout,
                visibleTimelinePoints.size,
            )
            val pointTextStyle = androidx.compose.ui.text.TextStyle(
                color = Color(ScoreOverlay.POINT_TEXT_COLOR).copy(alpha = timeline.opacity),
                fontSize = with(density) { pointLayout.fontSize.toSp() },
                fontWeight = FontWeight.Black,
            )
            val pointTexts = visibleTimelinePoints.map { point ->
                measurer.measure(point.teamPointNumber.toString(), style = pointTextStyle)
            }
            Canvas(Modifier.fillMaxSize()) {
                listOf(
                    Triple(ScoreTeamId.TEAM_1, pointLayout.team1CenterY, Color(ScoreOverlay.TEAM_1_COLOR)),
                    Triple(ScoreTeamId.TEAM_2, pointLayout.team2CenterY, Color(ScoreOverlay.TEAM_2_COLOR)),
                ).forEach { (teamId, y, color) ->
                    val lastPointIndex = visibleTimelinePoints.indexOfLast { it.winnerTeamId == teamId }
                    if (lastPointIndex < 0) return@forEach
                    val lastX = pointLayout.startX + pointLayout.columnSpacing * (lastPointIndex + .5f)
                    drawLine(
                        color.copy(alpha = timeline.opacity),
                        Offset(pointLayout.startX, y),
                        Offset(lastX, y),
                        pointLayout.lineWidth,
                    )
                }
                visibleTimelinePoints.forEachIndexed { index, point ->
                    val x = pointLayout.startX + pointLayout.columnSpacing * (index + .5f)
                    val y = if (point.winnerTeamId == ScoreTeamId.TEAM_1) {
                        pointLayout.team1CenterY
                    } else pointLayout.team2CenterY
                    val fill = if (point.winnerTeamId == ScoreTeamId.TEAM_1) {
                        Color(ScoreOverlay.TEAM_1_COLOR)
                    } else Color(ScoreOverlay.TEAM_2_COLOR)
                    drawCircle(
                        fill.copy(alpha = timeline.opacity),
                        pointLayout.circleRadius,
                        Offset(x, y),
                    )
                    drawCircle(
                        Color(ScoreOverlay.BORDER_COLOR).copy(alpha = timeline.opacity),
                        pointLayout.circleRadius,
                        Offset(x, y),
                        style = Stroke(pointLayout.lineWidth),
                    )
                    val text = pointTexts[index]
                    drawText(
                        text,
                        topLeft = Offset(
                            x - text.size.width / 2f,
                            y - text.size.height / 2f,
                        ),
                    )
                }
            }
        }
    }
}

@Composable
private fun SectionCard(
    label: String,
    subtitle: String,
    modifier: Modifier = Modifier,
    compact: Boolean = false,
    content: @Composable ColumnScope.() -> Unit,
) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(containerColor = Color.White),
    ) {
        Column(
            Modifier.fillMaxWidth().padding(if (compact) 10.dp else 14.dp),
            verticalArrangement = Arrangement.spacedBy(if (compact) 4.dp else 10.dp),
        ) {
            if (compact) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        label,
                        color = Orange,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Black,
                        letterSpacing = .7.sp,
                    )
                    Spacer(Modifier.width(10.dp))
                    Text(
                        subtitle,
                        Modifier.weight(1f),
                        color = Muted,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.SemiBold,
                        textAlign = TextAlign.End,
                    )
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
    modifier: Modifier = Modifier,
) {
    Column(modifier = modifier, verticalArrangement = Arrangement.spacedBy(4.dp)) {
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
    disagreementCount: Int,
    suppressionCount: Int,
    compact: Boolean = false,
    onChange: (Float) -> Unit,
    onReviewNext: () -> Unit,
    onSuppressionNext: () -> Unit,
) {
    if (compact) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("$disagreementCount checks · <${(value * 100).roundToInt()}%", fontSize = 11.sp)
            Slider(
                value = value,
                onValueChange = onChange,
                modifier = Modifier.weight(1f).height(32.dp),
                valueRange = 0f..1f,
                steps = 99,
            )
            SmallButton("Review $count", enabled = count > 0, onClick = onReviewNext)
            SmallButton(
                "Next suppression",
                enabled = suppressionCount > 0,
                onClick = onSuppressionNext,
            )
        }
    } else {
        Column {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Review disagreements and confidence below ${(value * 100).roundToInt()}%", Modifier.weight(1f), fontSize = 13.sp)
                Text("$disagreementCount disagreements · $count ranges", color = Muted, fontSize = 12.sp)
                TextButton(enabled = count > 0, onClick = onReviewNext) { Text("Review next") }
                TextButton(enabled = suppressionCount > 0, onClick = onSuppressionNext) {
                    Text("Next suppression")
                }
            }
            Slider(value = value, onValueChange = onChange, valueRange = 0f..1f, steps = 99)
        }
    }
}

@Composable
internal fun WholeTimeline(
    windowStartMs: Long,
    windowEndMs: Long,
    cuts: List<EditableCut>,
    joinedGaps: List<JoinedGap>,
    ignored: List<IgnoredSourceInterval>,
    suggestions: List<AnalysisTypes.SuppressionSuggestion>,
    appliedSuggestionIds: Set<String>,
    selectedSuggestionId: String?,
    selectedId: String?,
    effectiveIds: Set<String>,
    confidenceThreshold: Float,
    playheadMs: Long,
    serveMarkers: List<ServeMarker>,
    sideSwitchMarkers: List<SideSwitchMarker>,
    selectedScoreMarkerId: String?,
    onMarkerSelect: (String, Long) -> Unit,
    onSeek: (Long, String?, String?) -> Unit,
) {
    BoxWithConstraints(Modifier.fillMaxWidth()) {
        val density = LocalDensity.current
        val markerTextMeasurer = rememberTextMeasurer()
        val serveMarkerText = markerTextMeasurer.measure(
            "🏐",
            style = TextStyle(fontSize = 9.sp),
            maxLines = 1,
        )
        val switchMarkerText = markerTextMeasurer.measure(
            "⇄",
            style = TextStyle(
                color = Ink,
                fontSize = 10.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = FontFamily.Monospace,
            ),
            maxLines = 1,
        )
        val widthPx = with(density) { maxWidth.toPx() }.coerceAtLeast(1f)
        val windowSpanMs = (windowEndMs - windowStartMs).coerceAtLeast(1)
        fun timeAt(x: Float) = windowStartMs +
            ((x / widthPx).coerceIn(0f, 1f) * windowSpanMs).roundToLong()
        fun xAt(timeMs: Long) = (timeMs - windowStartMs).toFloat() / windowSpanMs * widthPx
        Canvas(
            Modifier
                .testTag("whole-timeline")
                .fillMaxWidth()
                .height(68.dp)
                .clip(RoundedCornerShape(9.dp))
                .background(Rail)
                .semantics {
                    contentDescription = if (suggestions.isEmpty()) "Game timeline"
                    else "Game timeline with ${suggestions.size} suppression suggestions: " +
                        suggestions.joinToString { suggestion ->
                            "${preciseTime(suggestion.startMs())} to ${preciseTime(suggestion.endMs())}, " +
                                if (suggestion.fragmentId() in appliedSuggestionIds) "Suppressed" else "Suggestion kept"
                        }
                }
                .pointerInput(windowStartMs, windowEndMs, cuts, suggestions, serveMarkers, sideSwitchMarkers) {
                    detectTapGestures { offset ->
                        val time = timeAt(offset.x)
                        val markerHitRadius = with(density) { 14.dp.toPx() }
                        val markerIconHeight = with(density) { 20.dp.toPx() }
                        val marker = if (offset.y <= markerIconHeight) {
                            buildList<Pair<String, Long>> {
                                serveMarkers.forEach { add(it.id to it.timestampMs) }
                                sideSwitchMarkers.forEach { add(it.id to it.timestampMs) }
                            }.filter { it.second in windowStartMs..windowEndMs }
                                .minByOrNull { kotlin.math.abs(xAt(it.second) - offset.x) }
                                ?.takeIf { kotlin.math.abs(xAt(it.second) - offset.x) <= markerHitRadius }
                        } else null
                        if (marker != null) {
                            onMarkerSelect(marker.first, marker.second)
                            return@detectTapGestures
                        }
                        val suggestion = suggestions.lastOrNull {
                            time >= it.startMs() && time <= it.endMs()
                        }
                        val cut = cuts.lastOrNull { time in it.keepStartMs..it.keepEndMs }
                        onSeek(time, cut?.id, suggestion?.fragmentId())
                    }
                }
                .pointerInput(windowStartMs, windowEndMs) {
                    detectHorizontalDragGestures(
                        onDragStart = { onSeek(timeAt(it.x), null, null) },
                        onHorizontalDrag = { change, _ -> change.consume(); onSeek(timeAt(change.position.x), null, null) },
                    )
                },
        ) {
            val cutTop = 16.dp.toPx()
            val cutHeight = 36.dp.toPx()
            val suppressionTop = 8.dp.toPx()
            val suppressionHeight = 52.dp.toPx()
            ignored.forEach { interval ->
                val clippedStart = max(interval.startMs, windowStartMs)
                val clippedEnd = min(interval.endMs, windowEndMs)
                if (clippedEnd <= clippedStart) return@forEach
                val x = xAt(clippedStart)
                val width = max(1f, xAt(clippedEnd) - x)
                drawRect(Color(0xFF4B4A47).copy(alpha = .28f), Offset(x, 0f), Size(width, size.height))
                clipRect(x, 0f, x + width, size.height) {
                    var hatch = x - size.height
                    while (hatch < x + width) {
                        drawLine(Color(0xFF4B4A47).copy(alpha = .5f), Offset(hatch, size.height), Offset(hatch + size.height, 0f), 2f)
                        hatch += 10f
                    }
                }
            }
            joinedGaps.forEach { gap ->
                val clippedStart = max(gap.startMs, windowStartMs)
                val clippedEnd = min(gap.endMs, windowEndMs)
                if (clippedEnd <= clippedStart) return@forEach
                val x = xAt(clippedStart)
                val width = max(1f, xAt(clippedEnd) - x)
                drawRoundRect(
                    Color(0xFFBBB8B1),
                    Offset(x, cutTop),
                    Size(width, cutHeight),
                    CornerRadius(4f),
                )
            }
            cuts.forEach { cut ->
                val clippedStart = max(cut.keepStartMs, windowStartMs)
                val clippedEnd = min(cut.keepEndMs, windowEndMs)
                if (clippedEnd <= clippedStart) return@forEach
                val x = xAt(clippedStart)
                val width = max(2f, xAt(clippedEnd) - x)
                val low = cut.origin == CutOrigin.INFERRED &&
                    (cut.isModelDisagreement() || cut.confidence < confidenceThreshold)
                val color = when {
                    !cut.included -> Muted.copy(alpha = .55f)
                    cut.id !in effectiveIds -> Danger.copy(alpha = .65f)
                    low -> Warning
                    else -> PaleGreen
                }
                drawRoundRect(color, Offset(x, cutTop), Size(width, cutHeight), CornerRadius(6f))
                val clippedCoreStart = max(cut.coreStartMs, windowStartMs)
                val clippedCoreEnd = min(cut.coreEndMs, windowEndMs)
                if (clippedCoreEnd > clippedCoreStart) {
                    val coreX = xAt(clippedCoreStart)
                    val coreWidth = max(1f, xAt(clippedCoreEnd) - coreX)
                    drawRoundRect(
                        when {
                            !cut.included -> Muted
                            low -> Orange
                            else -> Green
                        },
                        Offset(coreX, cutTop),
                        Size(coreWidth, cutHeight),
                        CornerRadius(4f),
                    )
                }
                if (cut.id == selectedId) {
                    drawRoundRect(
                        Orange,
                        Offset(x, cutTop - 3.dp.toPx()),
                        Size(width, cutHeight + 6.dp.toPx()),
                        CornerRadius(7f),
                        style = Stroke(2.dp.toPx()),
                    )
                }
            }
            suggestions.forEach { suggestion ->
                val clippedStart = max(suggestion.startMs(), windowStartMs)
                val clippedEnd = min(suggestion.endMs(), windowEndMs)
                if (clippedEnd <= clippedStart) return@forEach
                val x = xAt(clippedStart)
                val width = max(2f, xAt(clippedEnd) - x)
                val applied = suggestion.fragmentId() in appliedSuggestionIds
                if (applied) {
                    drawRect(
                        SuppressionRed.copy(alpha = .62f),
                        Offset(x, suppressionTop),
                        Size(width, suppressionHeight),
                    )
                } else {
                    drawRect(
                        SuppressionRed.copy(alpha = .2f),
                        Offset(x, suppressionTop),
                        Size(width, suppressionHeight),
                    )
                    drawRect(
                        SuppressionRed,
                        Offset(x, suppressionTop),
                        Size(width, suppressionHeight),
                        style = Stroke(2.dp.toPx()),
                    )
                }
                clipRect(x, suppressionTop, x + width, suppressionTop + suppressionHeight) {
                    var hatch = x - suppressionHeight
                    while (hatch < x + width) {
                        drawLine(
                            SuppressionRed.copy(alpha = if (applied) .95f else .55f),
                            Offset(hatch, suppressionTop + suppressionHeight),
                            Offset(hatch + suppressionHeight, suppressionTop),
                            2f,
                        )
                        hatch += 10f
                    }
                }
                if (suggestion.fragmentId() == selectedSuggestionId) {
                    drawRect(
                        Color.White,
                        Offset(x, suppressionTop + 2.dp.toPx()),
                        Size(width, suppressionHeight - 4.dp.toPx()),
                        style = Stroke(1.5.dp.toPx()),
                    )
                }
            }
            if (playheadMs in windowStartMs..windowEndMs) {
                val playheadX = xAt(playheadMs)
                drawLine(Orange, Offset(playheadX, 0f), Offset(playheadX, size.height), 4f, StrokeCap.Round)
            }
            val markerCenterY = 10.dp.toPx()
            val markerRadius = 8.dp.toPx()
            val markerStemWidth = 1.dp.toPx()
            val markerBorderWidth = 1.dp.toPx()
            serveMarkers.filter { it.timestampMs in windowStartMs..windowEndMs }.forEach { marker ->
                val x = xAt(marker.timestampMs)
                drawLine(
                    Ink,
                    Offset(x, markerCenterY + markerRadius),
                    Offset(x, size.height),
                    markerStemWidth,
                )
                val needsReview = marker.side == ServingSide.REVIEW
                drawCircle(
                    if (needsReview) Color(0xFFFFE1C7) else Color.White,
                    markerRadius,
                    Offset(x, markerCenterY),
                )
                drawCircle(
                    if (needsReview) Color(0xFFA84A16) else Ink,
                    markerRadius,
                    Offset(x, markerCenterY),
                    style = Stroke(markerBorderWidth),
                )
                drawText(
                    serveMarkerText,
                    topLeft = Offset(
                        x - serveMarkerText.size.width / 2f,
                        markerCenterY - serveMarkerText.size.height / 2f,
                    ),
                )
                if (marker.id == selectedScoreMarkerId) {
                    drawCircle(
                        Orange,
                        markerRadius + 2.dp.toPx(),
                        Offset(x, markerCenterY),
                        style = Stroke(2.dp.toPx()),
                    )
                }
            }
            sideSwitchMarkers.filter { it.timestampMs in windowStartMs..windowEndMs }.forEach { marker ->
                val x = xAt(marker.timestampMs)
                drawLine(
                    Ink,
                    Offset(x, markerCenterY + markerRadius),
                    Offset(x, size.height),
                    markerStemWidth,
                )
                val boxLeft = x - markerRadius
                val boxTop = markerCenterY - markerRadius
                drawRect(
                    Color(0xFFDFFF35),
                    Offset(boxLeft, boxTop),
                    Size(markerRadius * 2f, markerRadius * 2f),
                )
                drawRect(
                    Ink,
                    Offset(boxLeft, boxTop),
                    Size(markerRadius * 2f, markerRadius * 2f),
                    style = Stroke(markerBorderWidth),
                )
                drawText(
                    switchMarkerText,
                    topLeft = Offset(
                        x - switchMarkerText.size.width / 2f,
                        markerCenterY - switchMarkerText.size.height / 2f,
                    ),
                )
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
    suggestions: List<AnalysisTypes.SuppressionSuggestion>,
    appliedSuggestionIds: Set<String>,
    onSeek: (Long) -> Unit,
    onSuggestionSelect: (String) -> Unit,
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
                .semantics {
                    contentDescription = "Focused timeline. " + suggestions.joinToString {
                        "Suppression ${preciseTime(it.startMs())} to ${preciseTime(it.endMs())}, " +
                            if (it.fragmentId() in appliedSuggestionIds) "Suppressed" else "Suggestion kept"
                    }
                }
                .pointerInput(window, suggestions) {
                    detectTapGestures { offset ->
                        val time = timeAt(offset.x)
                        suggestions.lastOrNull {
                            time >= it.startMs() && time <= it.endMs()
                        }?.let { onSuggestionSelect(it.fragmentId()) } ?: onSeek(time)
                    }
                }
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
                when {
                    !cut.included -> Muted
                    lowConfidence -> Orange
                    else -> Green
                },
                Offset(xAt(cut.coreStartMs), 33f),
                Size(max(2f, xAt(cut.coreEndMs) - xAt(cut.coreStartMs)), size.height - 66f),
                CornerRadius(5f),
            )
            suggestions.forEach { suggestion ->
                val clippedStart = max(suggestion.startMs(), window.startMs)
                val clippedEnd = min(suggestion.endMs(), window.endMs)
                if (clippedEnd <= clippedStart) return@forEach
                val x = xAt(clippedStart)
                val width = max(2f, xAt(clippedEnd) - x)
                val applied = suggestion.fragmentId() in appliedSuggestionIds
                drawRect(
                    SuppressionRed.copy(alpha = if (applied) .62f else .18f),
                    Offset(x, 0f), Size(width, size.height),
                )
                drawRect(
                    SuppressionRed, Offset(x, 1f), Size(width, size.height - 2f),
                    style = Stroke(if (applied) 1f else 3f),
                )
                clipRect(x, 0f, x + width, size.height) {
                    var hatch = x - size.height
                    while (hatch < x + width) {
                        drawLine(SuppressionRed.copy(alpha = .75f), Offset(hatch, size.height), Offset(hatch + size.height, 0f), 2f)
                        hatch += 10f
                    }
                }
            }
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
private fun RallyRangeSlider(
    cut: EditableCut,
    window: DetailWindow,
    onRangeChange: (Long, Long) -> Unit,
    modifier: Modifier = Modifier,
) {
    val sliderWindow = remember(cut.id) { window }
    val windowStart = sliderWindow.startMs.toFloat()
    val windowEnd = sliderWindow.endMs
        .coerceAtLeast(sliderWindow.startMs + MIN_MARK_MS)
        .toFloat()
    Column(modifier) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Rally length", fontSize = 12.sp, color = Muted)
            Spacer(Modifier.weight(1f))
            Text(
                "${preciseTime(cut.coreStartMs)} – ${preciseTime(cut.coreEndMs)}",
                fontFamily = FontFamily.Monospace,
                fontWeight = FontWeight.Bold,
            )
        }
        RangeSlider(
            value = cut.coreStartMs.toFloat()..cut.coreEndMs.toFloat(),
            onValueChange = { range ->
                onRangeChange(range.start.roundToLong(), range.endInclusive.roundToLong())
            },
            valueRange = windowStart..windowEnd,
            modifier = Modifier.fillMaxWidth().semantics {
                contentDescription = "Adjust rally start and end"
            },
        )
        Text(
            "Drag either handle to shorten or extend this rally before padding.",
            fontSize = 11.sp,
            color = Muted,
        )
    }
}

@Composable
private fun MarkingTools(
    draft: EditorDraft,
    playbackPositionMs: Long,
    onDraft: ((EditorDraft) -> EditorDraft) -> Unit,
    onManualCompleted: (EditableCut) -> Unit,
    onMessage: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(modifier, verticalArrangement = Arrangement.spacedBy(12.dp)) {
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
}

@Composable
private fun SmallButton(label: String, enabled: Boolean = true, onClick: () -> Unit) {
    OutlinedButton(enabled = enabled, onClick = onClick, contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 10.dp, vertical = 5.dp)) {
        Text(label, fontSize = 12.sp)
    }
}

internal fun editListJson(seed: EditorSeed, draft: EditorDraft, intervals: List<FinalCutInterval>) =
    JSONObject().apply {
        put("schemaVersion", 2)
        put("method", "android-editor-v3-suppression")
        put("sourceName", seed.displayName)
        put("sourceUri", seed.sourceUri)
        put("sourceDuration", seed.durationMs / 1_000.0)
        put("gameStartSeconds", seed.gameStartMs / 1_000.0)
        put("gameEndSeconds", seed.gameEndMs / 1_000.0)
        put("sourceRevision", seed.sourceRevision)
        put("beforePaddingSeconds", draft.beforePaddingMs / 1_000.0)
        put("afterPaddingSeconds", draft.afterPaddingMs / 1_000.0)
        put("joinGapSeconds", draft.joinGapMs / 1_000.0)
        put("outputDuration", EditorMath.totalFinalMs(intervals) / 1_000.0)
        put("policyContractVersion", draft.suppressionContractVersion)
        put("selectedSuppressionPolicy", draft.selectedSuppressionPolicy.wireName)
        put("recordedSuppressionPolicy", draft.selectedSuppressionPolicy.recordedPolicy)
        put("suppressionInitialBehavior", draft.suppressionInitialBehavior.wireName)
        put("defaultSuppressionScope", SuppressionScope.WHOLE_RALLY.wireName)
        put("suppression", seed.suppression?.let { analysis -> JSONObject().apply {
            put("modelId", analysis.modelId())
            put("artifactSha256", analysis.artifactSha256())
            put("weightsSha256", analysis.weightsSha256())
            put("decoderVersion", analysis.decoderVersion())
            put("productionComponents", JSONObject().apply {
                put("allLabelsV2", JSONArray().apply {
                    seed.productionComponents.allLabelsV2().forEach { interval ->
                        put(JSONObject().apply {
                            put("start", interval.start())
                            put("end", interval.end())
                            put("confidence", interval.confidence().toDouble())
                        })
                    }
                })
                put("previousProduction", JSONArray().apply {
                    seed.productionComponents.previousProduction().forEach { interval ->
                        put(JSONObject().apply {
                            put("start", interval.start())
                            put("end", interval.end())
                            put("confidence", interval.confidence().toDouble())
                        })
                    }
                })
            })
            put("suggestions", JSONArray().apply {
                analysis.suggestions().forEach { suggestion -> put(JSONObject().apply {
                    put("logicalId", suggestion.logicalId())
                    put("fragmentId", suggestion.fragmentId())
                    put("start", suggestion.startMs() / 1_000.0)
                    put("end", suggestion.endMs() / 1_000.0)
                    put("score", suggestion.score().toDouble())
                    put("sourceProductionIds", JSONArray(suggestion.sourceProductionIds()))
                    put("eligiblePolicyIds", JSONArray(suggestion.eligiblePolicyIds()))
                    put("active", suggestion.eligiblePolicyIds().contains(draft.selectedSuppressionPolicy.wireName))
                    put("explicitDecision", draft.suppressionDecisionOverrides[suggestion.logicalId()]?.wireName ?: JSONObject.NULL)
                    put("effectiveDecision", EditorMath.suggestionEffectiveDecision(draft, suggestion).wireName)
                    put("effectiveScope", EditorMath.suggestionEffectiveScope(draft, suggestion).wireName)
                }) }
            })
            put("decisionOverrides", JSONObject().apply {
                draft.suppressionDecisionOverrides.toSortedMap().forEach { (id, decision) ->
                    put(id, decision.wireName)
                }
            })
            put("scopeOverrides", JSONObject().apply {
                draft.suppressionScopeOverrides.toSortedMap().forEach { (id, scope) ->
                    put(id, scope.wireName)
                }
            })
            put("userTouchedCutIds", JSONArray(draft.userTouchedCutIds.sorted()))
        }} ?: JSONObject.NULL)
        put("ranges", JSONArray().apply {
            intervals.forEach { range -> put(JSONObject().apply {
                put("start", range.startMs / 1_000.0)
                put("end", range.endMs / 1_000.0)
                put("cutIds", JSONArray(range.cutIds))
                if (range.joinedGaps.isNotEmpty()) put("joinedGaps", JSONArray().apply {
                    range.joinedGaps.forEach { gap -> put(JSONObject().apply {
                        put("start", gap.startMs / 1_000.0)
                        put("end", gap.endMs / 1_000.0)
                    }) }
                })
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
                put("agreement", cut.agreement ?: JSONObject.NULL)
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
        put("materializationProvenance", JSONArray().apply {
            EditorMath.materialize(draft, seed.suppression).provenance.forEach { segment ->
                put(JSONObject().apply {
                    put("start", segment.startMs / 1_000.0)
                    put("end", segment.endMs / 1_000.0)
                    put("kind", segment.kind)
                    put("cutIds", JSONArray(segment.cutIds))
                    put("suggestionIds", JSONArray(segment.suggestionIds))
                })
            }
        })
        put("scoreTracking", ScoreTrackingJson.encodeWire(draft.scoreTracking))
        put("renderScoreOverlay", draft.renderScoreOverlay)
        put("renderScoreTimeline", draft.renderScoreTimeline)
    }

private fun exportFilename(sourceName: String): String {
    val base = sourceName.substringBeforeLast('.').replace(Regex("[^A-Za-z0-9._-]+"), "-").trim('-')
    return "${base.ifBlank { "volleycut" }}-cut.mp4"
}

private fun editListFilename(sourceName: String): String = exportFilename(sourceName).removeSuffix(".mp4") + ".edit-list.json"

private fun createModelFeedback(
    context: Context,
    project: NativeProject,
    draft: EditorDraft,
    intervals: List<FinalCutInterval>,
): String {
    val analysis = ModelFeedbackExporter.loadAnalysis(context, project)
    val fingerprint = ModelFeedbackExporter.sourceFingerprint(context, project)
        ?: project.source.sampledFingerprint
    return ModelFeedbackExporter.createBundle(
        project,
        draft,
        intervals,
        analysis,
        fingerprint,
    ).toString() + "\n"
}

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
