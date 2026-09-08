package com.volleycut.nativeanalysis

import android.Manifest
import android.content.BroadcastReceiver
import android.content.ClipData
import android.content.ClipboardManager
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
import androidx.compose.foundation.BorderStroke
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
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
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
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.runtime.staticCompositionLocalOf
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
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
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
            var uiScale by remember { mutableFloatStateOf(uiScalePreference(this)) }
            val systemDensity = LocalDensity.current
            val scaledDensity = remember(systemDensity.density, systemDensity.fontScale, uiScale) {
                Density(
                    density = systemDensity.density * uiScale,
                    fontScale = systemDensity.fontScale,
                )
            }
            CompositionLocalProvider(
                LocalDensity provides scaledDensity,
                LocalUiScaleSetting provides UiScaleSetting(uiScale) { requestedScale ->
                    val normalizedScale = normalizeUiScale(requestedScale)
                    uiScale = normalizedScale
                    setUiScalePreference(this, normalizedScale)
                },
            ) {
                VolleySpliceTheme {
                    BoxWithConstraints(Modifier.fillMaxSize()) {
                        val desktop = editorLayoutMode(
                            maxWidth.value.roundToInt(),
                            maxHeight.value.roundToInt(),
                        ) == EditorLayoutMode.DESKTOP
                        DesktopStatusBar(activity = this@EditorActivity, hidden = desktop)
                        EditorApp(activity = this@EditorActivity, initialSeed = seed)
                    }
                }
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

@Composable
private fun DesktopStatusBar(activity: ComponentActivity, hidden: Boolean) {
    DisposableEffect(activity, hidden) {
        val controller = WindowCompat.getInsetsController(activity.window, activity.window.decorView)
        if (hidden) {
            controller.systemBarsBehavior =
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            controller.hide(WindowInsetsCompat.Type.statusBars())
        } else {
            controller.show(WindowInsetsCompat.Type.statusBars())
        }
        onDispose {
            if (hidden) controller.show(WindowInsetsCompat.Type.statusBars())
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

private val Paper = Color(0xFFF8F7EE)
private val Ink = Color(0xFF17382E)
private val Orange = Color(0xFFDE7959)
private val Green = Color(0xFF2F6855)
private val Acid = Color(0xFFD9EE9E)
private val PaleGreen = Color(0xFFC8D6CA)
private val Muted = Color(0xFF65766F)
private val Rail = Color(0xFFC8D6CA)
internal val TimelineTrackColor = Color(0xFFE7EBE4)
internal val TimelineKeptPaddingColor = Color(0xFFC8D6CA)
internal val TimelineJoinedGapColor = Color(0xFFAAA9A2)
private val SoftPanel = Color(0xFFF5F8E9)
private val Danger = Color(0xFFB3261E)
private val SuppressionRed = Color(0xFFD1242F)
private val Warning = Color(0xFFE8A317)

internal enum class EditorLayoutMode { COMPACT, DESKTOP }
private enum class DesktopEditorTab { REVIEW, EXPORT }

private const val DESKTOP_PLAYER_DEFAULT_HEIGHT_DP = 320f
internal const val DESKTOP_PLAYER_MIN_HEIGHT_DP = 180f
internal const val DESKTOP_PLAYER_MAX_HEIGHT_DP = 1200f
private const val COMPACT_PLAYER_DEFAULT_HEIGHT_DP = 176f
internal const val COMPACT_PLAYER_MIN_HEIGHT_DP = 140f
internal const val COMPACT_PLAYER_MAX_HEIGHT_DP = 720f
private const val SETUP_PLAYER_COMPACT_DEFAULT_HEIGHT_DP = 176f
private const val SETUP_PLAYER_DESKTOP_DEFAULT_HEIGHT_DP = 320f
internal const val SETUP_PLAYER_MIN_HEIGHT_DP = 140f
internal const val SETUP_PLAYER_COMPACT_MAX_HEIGHT_DP = 720f
internal const val SETUP_PLAYER_DESKTOP_MAX_HEIGHT_DP = 1200f
private const val EDITOR_LAYOUT_PREFERENCES = "desktop_editor_layout"
private const val DESKTOP_LEFT_PANE_DEFAULT_WIDTH_DP = 292f
private const val DESKTOP_RIGHT_PANE_DEFAULT_WIDTH_DP = 292f
internal const val DESKTOP_LEFT_PANE_MIN_WIDTH_DP = 180f
internal const val DESKTOP_LEFT_PANE_MAX_WIDTH_DP = 520f
internal const val DESKTOP_RIGHT_PANE_MIN_WIDTH_DP = 220f
internal const val DESKTOP_RIGHT_PANE_MAX_WIDTH_DP = 560f
private const val DESKTOP_CENTER_PANE_MIN_WIDTH_DP = 220f
private const val DESKTOP_PANE_HORIZONTAL_CHROME_DP = 64f

internal fun resizedDesktopPlayerHeight(
    currentHeightDp: Float,
    dragDeltaPx: Float,
    density: Float,
): Float = resizedPlayerHeight(
    currentHeightDp = currentHeightDp,
    dragDeltaPx = dragDeltaPx,
    density = density,
    minHeightDp = DESKTOP_PLAYER_MIN_HEIGHT_DP,
    maxHeightDp = DESKTOP_PLAYER_MAX_HEIGHT_DP,
)

internal fun resizedPlayerHeight(
    currentHeightDp: Float,
    dragDeltaPx: Float,
    density: Float,
    minHeightDp: Float,
    maxHeightDp: Float,
): Float = (currentHeightDp + dragDeltaPx / density.coerceAtLeast(0.1f))
    .coerceIn(minHeightDp, maxHeightDp.coerceAtLeast(minHeightDp))

internal fun resizedDesktopSidebarWidth(
    currentWidthDp: Float,
    dragDeltaPx: Float,
    density: Float,
    minWidthDp: Float,
    maxWidthDp: Float,
): Float = (currentWidthDp + dragDeltaPx / density.coerceAtLeast(0.1f))
    .coerceIn(minWidthDp, maxWidthDp.coerceAtLeast(minWidthDp))

internal fun editorLayoutMode(widthDp: Int, heightDp: Int): EditorLayoutMode =
    if (widthDp >= 840 && heightDp >= 360) EditorLayoutMode.DESKTOP else EditorLayoutMode.COMPACT

internal fun timelineNeedsReview(
    lowConfidence: Boolean,
    cutId: String,
    reviewedCutIds: Set<String>,
): Boolean = lowConfidence && cutId !in reviewedCutIds

private fun EditableCut.isModelDisagreement(): Boolean =
    ProductionEnsemble.isDisagreement(agreement)

private fun EditableCut.modelAgreementLabel(): String = when (agreement) {
    ProductionEnsemble.BOTH_MODELS -> "Found automatically"
    ProductionEnsemble.ALL_LABELS_V2_ONLY,
    ProductionEnsemble.PREVIOUS_PRODUCTION_ONLY -> "Needs a quick check"
    else -> "Suggested clip"
}

@Composable
private fun VolleySpliceTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = lightColorScheme(
            primary = Green,
            onPrimary = Color.White,
            secondary = Orange,
            background = Paper,
            onBackground = Ink,
            surface = Paper,
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

private fun NativeProject.persistedExportUiState() = if (lastExportedAtMs != null) {
    ExportUiState(
        status = "complete",
        progress = 100,
        detail = lastExportedVideoName?.let { "Saved $it" } ?: "Video saved",
    )
} else ExportUiState()

private fun projectStatusLabel(
    project: NativeProject,
    export: ExportJobStatus?,
): String {
    val encoding = when (export?.status) {
        "queued" -> "video waiting to save"
        "running" -> "saving ${export.progress.coerceIn(0, 100)}%"
        "complete" -> "video saved"
        "failed" -> "save failed"
        "cancelled" -> "save cancelled"
        else -> "video saved".takeIf { project.lastExportedAtMs != null }
    }
    val projectState = when (project.status) {
        ProjectStatus.QUEUED -> "waiting"
        ProjectStatus.ANALYZING -> "finding rallies"
        ProjectStatus.ERROR -> "needs attention"
        ProjectStatus.READY -> "ready to review"
    }
    return listOfNotNull(projectState, encoding).joinToString(" · ")
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
    val outputDurationMs: Long,
    val kept: Int,
    val needsReview: Int,
    val cleanupReviewCount: Int,
    val clipReviewCount: Int,
    val serveReviewCount: Int,
    val onReviewCleanup: () -> Unit,
    val onReviewClips: () -> Unit,
    val onReviewServes: () -> Unit,
    val desktopTab: DesktopEditorTab,
    val onDesktopTab: (DesktopEditorTab) -> Unit,
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

private const val DISPLAY_SETTINGS_PREFERENCES = "volleycut-display-settings"
private const val SHOW_ANALYSIS_MEASUREMENTS = "show-analysis-measurements"
private const val UI_SCALE = "ui-scale"
internal const val UI_SCALE_MIN = 0.75f
internal const val UI_SCALE_MAX = 1.25f
internal const val UI_SCALE_DEFAULT = 1f
private const val UI_SCALE_STEP = 0.05f

private data class UiScaleSetting(
    val scale: Float,
    val onScaleChange: (Float) -> Unit,
)

private val LocalUiScaleSetting = staticCompositionLocalOf {
    UiScaleSetting(UI_SCALE_DEFAULT) {}
}

internal fun normalizeUiScale(scale: Float): Float =
    ((scale.coerceIn(UI_SCALE_MIN, UI_SCALE_MAX) / UI_SCALE_STEP).roundToInt() * UI_SCALE_STEP)
        .coerceIn(UI_SCALE_MIN, UI_SCALE_MAX)

private fun uiScalePreference(context: Context): Float = normalizeUiScale(
    context.getSharedPreferences(DISPLAY_SETTINGS_PREFERENCES, Context.MODE_PRIVATE)
        .getFloat(UI_SCALE, UI_SCALE_DEFAULT),
)

private fun setUiScalePreference(context: Context, scale: Float) {
    context.getSharedPreferences(DISPLAY_SETTINGS_PREFERENCES, Context.MODE_PRIVATE)
        .edit()
        .putFloat(UI_SCALE, normalizeUiScale(scale))
        .apply()
}

private fun showAnalysisMeasurements(context: Context): Boolean =
    context.getSharedPreferences(DISPLAY_SETTINGS_PREFERENCES, Context.MODE_PRIVATE)
        .getBoolean(SHOW_ANALYSIS_MEASUREMENTS, false)

private fun setShowAnalysisMeasurements(context: Context, enabled: Boolean) {
    context.getSharedPreferences(DISPLAY_SETTINGS_PREFERENCES, Context.MODE_PRIVATE)
        .edit()
        .putBoolean(SHOW_ANALYSIS_MEASUREMENTS, enabled)
        .apply()
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
    val useCache = true
    // Match the web app: prepare serve markers with every project so the
    // optional scoreboard is ready when the user turns it on in the editor.
    var analyzeServingSide by remember { mutableStateOf(true) }
    var generateSideSwitchMarkers by remember { mutableStateOf(false) }
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
    var showSettings by remember { mutableStateOf(false) }
    var displayAnalysisMeasurements by remember {
        mutableStateOf(showAnalysisMeasurements(context))
    }
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
            return
        }
        selectedProjectId = preferredId?.takeIf { id -> projects.any { it.id == id } }
            ?: projects.firstOrNull()?.id
        NativeProjectStore.setSelectedId(context, selectedProjectId)
        if (selectedProjectId != null) creatingNew = false
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
            inference = InferenceUiState(stage = "opening", detail = "Getting your video ready")
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
                    inference = InferenceUiState(detail = "Choose the part with the game, then find the rallies")
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
        ActivityResultContracts.OpenMultipleDocuments(),
    ) { uris ->
        if (uris.isNotEmpty()) {
            preparingSource = true
            inference = InferenceUiState(
                stage = "importing",
                detail = "Opening ${if (uris.size == 1) "your saved project" else "${uris.size} saved projects"}",
            )
            scope.launch {
                val results = uris.mapIndexed { index, uri ->
                    inference = InferenceUiState(
                        stage = "importing",
                        detail = "Importing ${index + 1} of ${uris.size} · ${sourceDisplayName(context, uri)}",
                    )
                    uri to withContext(Dispatchers.IO) {
                        runCatching {
                            val text = context.contentResolver.openInputStream(uri)
                                ?.bufferedReader()?.use { it.readText() }
                                ?: error("Could not open that saved project")
                            ModelFeedbackImporter.import(context, text)
                        }
                    }
                }
                preparingSource = false
                val importedProjects = results.mapNotNull { it.second.getOrNull() }
                val failures = results.mapNotNull { (uri, result) ->
                    result.exceptionOrNull()?.let { error ->
                        "${sourceDisplayName(context, uri)}: ${error.message ?: "could not import"}"
                    }
                }
                if (importedProjects.isNotEmpty()) {
                    val selectedImport = importedProjects.last()
                    reloadProjects(selectedImport.id, preserveNewProject = false)
                    creatingNew = false
                    val importedLabel = if (importedProjects.size == 1) {
                        "1 project imported"
                    } else {
                        "${importedProjects.size} projects imported"
                    }
                    relinkMessage = buildString {
                        append(importedLabel)
                        if (failures.isNotEmpty()) {
                            append(" · ${failures.size} failed: ")
                            append(failures.take(2).joinToString("; "))
                            if (failures.size > 2) append("; +${failures.size - 2} more")
                        }
                        append(" · re-link the original video for playback")
                    }
                    relinkFailed = failures.isNotEmpty()
                    inference = InferenceUiState(
                        projectId = selectedImport.id,
                        detail = relinkMessage.orEmpty(),
                    )
                } else {
                    inference = InferenceUiState(
                        stage = "failed",
                        error = failures.joinToString(separator = "\n").ifBlank {
                            "Could not open the selected saved projects"
                        },
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
                    relinkMessage = "${updated.source.name} is connected again. Your saved edits were preserved."
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
                            "Rallies are ready; preparing the optional scoreboard"
                        } else "Your suggested clips are ready",
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
                        detail = "This video is already waiting to be prepared",
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
                        detail = "Waiting to find the rallies",
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
                Text("This removes the video’s suggested clips and all saved edits from this device.")
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

    if (showSettings) {
        AppSettingsDialog(
            displayAnalysisMeasurements = displayAnalysisMeasurements,
            onDisplayAnalysisMeasurements = { enabled ->
                displayAnalysisMeasurements = enabled
                setShowAnalysisMeasurements(context, enabled)
            },
            onDismiss = { showSettings = false },
        )
    }

    val queueCount = projects.count {
        it.status == ProjectStatus.QUEUED || it.status == ProjectStatus.ANALYZING ||
            it.servingSideStatus == ServingSideAnalysisStatus.QUEUED ||
            it.servingSideStatus == ServingSideAnalysisStatus.ANALYZING
    }
    val guidedTourTargets = remember { GuidedTourTargets() }
    var guidedTourRestartSignal by remember { mutableIntStateOf(0) }
    val projectControls: @Composable (EditorProjectSummary?, (() -> Unit)?) -> Unit =
        { editorSummary, onOpenEditorSettings ->
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
                inference = InferenceUiState(detail = "Choose a game video")
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
            guidedTourTargets = guidedTourTargets,
            onOpenSettings = onOpenEditorSettings ?: { showSettings = true },
        )
    }

    if (creatingNew || selectedProject == null) {
        ProjectShell(
            projectControls = { projectControls(null, null) },
            guidedTourStage = GuidedTourStage.SETUP,
            guidedTourTargets = guidedTourTargets,
            sourceReady = selectedSource?.media != null,
            guidedTourRestartSignal = guidedTourRestartSignal,
        ) {
            NewProjectIntro()
            NewProjectCard(
                selected = selectedSource,
                preparing = preparingSource,
                state = inference,
                generateSideSwitchMarkers = generateSideSwitchMarkers,
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
                onGenerateSideSwitchMarkers = { generateSideSwitchMarkers = it },
                guidedTourTargets = guidedTourTargets,
            )
        }
    } else if (selectedProject.status != ProjectStatus.READY) {
        ProjectShell(projectControls = { projectControls(null, null) }) {
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
                        detail = "Waiting to find the rallies",
                    )
                    reloadProjects(queued.id)
                    ProjectAnalysisService.enqueue(context, queued.id)
                },
            )
            if (displayAnalysisMeasurements && selectedProject.analysisMeasurements.isNotEmpty()) {
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
                analysisRunning = queueCount > 0,
                displayAnalysisMeasurements = displayAnalysisMeasurements,
                onDisplayAnalysisMeasurements = { enabled ->
                    displayAnalysisMeasurements = enabled
                    setShowAnalysisMeasurements(context, enabled)
                },
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
    guidedTourTargets: GuidedTourTargets? = null,
    onOpenSettings: () -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }
    var projectMenuExpanded by remember { mutableStateOf(false) }
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
        color = Paper,
        shape = RoundedCornerShape(10.dp),
        modifier = Modifier.border(1.dp, Rail, RoundedCornerShape(10.dp)),
    ) {
        BoxWithConstraints {
            if (maxWidth >= 840.dp && editorSummary != null) {
                val condensed = maxWidth < 1100.dp
                Row(
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 2.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(if (condensed) 5.dp else 8.dp),
                ) {
                    Image(
                        painter = painterResource(R.drawable.volleysplice_logo),
                        contentDescription = "VolleySplice",
                        contentScale = ContentScale.Fit,
                        modifier = Modifier.width(if (condensed) 64.dp else 92.dp).height(26.dp),
                    )
                    Box(Modifier.width(if (condensed) 160.dp else 250.dp)) {
                        Column(verticalArrangement = Arrangement.spacedBy(0.dp)) {
                            Text(
                                "CURRENT REVIEW",
                                color = Green,
                                fontSize = 7.sp,
                                fontWeight = FontWeight.Black,
                                letterSpacing = .7.sp,
                            )
                            OutlinedButton(
                                onClick = { expanded = true },
                                modifier = Modifier.fillMaxWidth().height(32.dp),
                                shape = RoundedCornerShape(5.dp),
                                contentPadding = PaddingValues(horizontal = 8.dp, vertical = 0.dp),
                            ) {
                                Text(
                                    if (creatingNew || selected == null) "＋ Start a new video…" else selected.source.name,
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis,
                                    fontSize = 10.sp,
                                )
                            }
                        }
                        DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                            DropdownMenuItem(
                                text = { Text("＋ Start a new video…") },
                                onClick = { expanded = false; onNew() },
                            )
                            projects.forEach { project ->
                                DropdownMenuItem(
                                    text = {
                                        Column {
                                            Text(project.source.name, maxLines = 1)
                                            Text(
                                                projectStatusLabel(project, exportStatuses[project.id]),
                                                color = Muted,
                                                fontSize = 10.sp,
                                            )
                                        }
                                    },
                                    onClick = { expanded = false; onSelect(project) },
                                )
                            }
                        }
                    }
                    Row(
                        Modifier.width(if (condensed) 112.dp else 200.dp),
                    ) {
                        HeaderWorkflowTab(
                            label = "Review",
                            selected = editorSummary.desktopTab == DesktopEditorTab.REVIEW,
                            onClick = { editorSummary.onDesktopTab(DesktopEditorTab.REVIEW) },
                            modifier = Modifier.weight(1f),
                        )
                        HeaderWorkflowTab(
                            label = "Export",
                            selected = editorSummary.desktopTab == DesktopEditorTab.EXPORT,
                            onClick = { editorSummary.onDesktopTab(DesktopEditorTab.EXPORT) },
                            modifier = Modifier
                                .weight(1f)
                                .guidedTourTarget("editor-export", guidedTourTargets),
                        )
                    }
                    Spacer(Modifier.weight(1f))
                    HeaderReviewQueue(
                        label = "Review cleanup",
                        count = editorSummary.cleanupReviewCount,
                        onClick = editorSummary.onReviewCleanup,
                        condensed = condensed,
                        modifier = Modifier.guidedTourTarget("editor-review-queues", guidedTourTargets),
                    )
                    HeaderReviewQueue(
                        label = "Review clips",
                        count = editorSummary.clipReviewCount,
                        onClick = editorSummary.onReviewClips,
                        condensed = condensed,
                    )
                    HeaderReviewQueue(
                        label = "Review serves",
                        count = editorSummary.serveReviewCount,
                        onClick = editorSummary.onReviewServes,
                        condensed = condensed,
                    )
                    if (!condensed) HeaderStat(editorSummary.kept.toString(), "clips included")
                    HeaderStat(compactTime(editorSummary.outputDurationMs), "planned video")
                    IconButton(
                        onClick = onOpenSettings,
                        modifier = Modifier
                            .size(40.dp)
                            .testTag("open-settings")
                            .guidedTourTarget("editor-settings", guidedTourTargets),
                    ) {
                        Icon(
                            painter = painterResource(R.drawable.ic_settings),
                            contentDescription = "Settings",
                            tint = Ink,
                        )
                    }
                    Box {
                        TextButton(
                            onClick = { projectMenuExpanded = true },
                            modifier = Modifier.height(40.dp),
                            contentPadding = PaddingValues(horizontal = 5.dp, vertical = 0.dp),
                        ) { Text("More", fontSize = 11.sp) }
                        DropdownMenu(
                            expanded = projectMenuExpanded,
                            onDismissRequest = { projectMenuExpanded = false },
                        ) {
                            DropdownMenuItem(
                                text = { Text("Restart guided tour") },
                                onClick = { projectMenuExpanded = false; onRestartTour() },
                            )
                            if (selected != null && !creatingNew) {
                                DropdownMenuItem(
                                    text = { Text("Delete current project", color = Danger) },
                                    onClick = { projectMenuExpanded = false; onDelete() },
                                )
                            }
                        }
                    }
                }
            } else {
        Column(
            Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Image(
                    painter = painterResource(R.drawable.volleysplice_logo),
                    contentDescription = "VolleySplice",
                    contentScale = ContentScale.Fit,
                    modifier = Modifier.width(112.dp).height(36.dp),
                )
                Spacer(Modifier.width(8.dp))
                Surface(color = SoftPanel, shape = RoundedCornerShape(4.dp)) {
                    Text(
                        "VIDEO EDITOR · v$versionName",
                        Modifier.padding(horizontal = 6.dp, vertical = 4.dp),
                        color = Muted,
                        fontSize = 8.sp,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = .7.sp,
                    )
                }
                Spacer(Modifier.weight(1f))
                IconButton(
                    onClick = onOpenSettings,
                    modifier = Modifier
                        .testTag("open-settings")
                        .guidedTourTarget("editor-settings", guidedTourTargets),
                ) {
                    Icon(
                        painter = painterResource(R.drawable.ic_settings),
                        contentDescription = "Settings",
                        tint = Ink,
                    )
                }
            }
            Row(verticalAlignment = Alignment.Bottom) {
                Box(Modifier.weight(1f)) {
                    Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
                        Text(
                            "CURRENT PROJECT",
                            color = Green,
                            fontSize = 9.sp,
                            fontWeight = FontWeight.Black,
                            letterSpacing = .9.sp,
                        )
                        OutlinedButton(
                            onClick = { expanded = true },
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(6.dp),
                        ) {
                            Text(
                                if (creatingNew || selected == null) "＋ Start a new video…"
                                else "${selected.source.name} · ${projectStatusLabel(selected, exportStatuses[selected.id])}",
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis,
                            )
                        }
                    }
                    DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                        DropdownMenuItem(
                            text = { Text("＋ Start a new video…") },
                            onClick = { expanded = false; onNew() },
                        )
                        projects.forEach { project ->
                            DropdownMenuItem(
                                text = {
                                    Column {
                                        Text(project.source.name, maxLines = 1)
                                        Text(
                                            projectStatusLabel(project, exportStatuses[project.id]),
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
                Box {
                    TextButton(onClick = { projectMenuExpanded = true }) { Text("More") }
                    DropdownMenu(
                        expanded = projectMenuExpanded,
                        onDismissRequest = { projectMenuExpanded = false },
                    ) {
                        DropdownMenuItem(
                            text = { Text("Restart guided tour") },
                            onClick = { projectMenuExpanded = false; onRestartTour() },
                        )
                        if (selected != null && !creatingNew) {
                            DropdownMenuItem(
                                text = { Text("Delete current project", color = Danger) },
                                onClick = { projectMenuExpanded = false; onDelete() },
                            )
                        }
                    }
                }
            }
            if (queueCount > 0 || exportQueueCount > 0) {
                Text(
                    when {
                        queueCount == 0 -> "$exportQueueCount saving"
                        exportQueueCount == 0 -> "$queueCount preparing"
                        else -> "$queueCount preparing · $exportQueueCount saving"
                    },
                    color = Green,
                    fontSize = 10.sp,
                    fontFamily = FontFamily.Monospace,
                    fontWeight = FontWeight.SemiBold,
                )
            }
        }
            }
        }
    }
}

@Composable
private fun HeaderStat(value: String, label: String) {
    Column(
        modifier = Modifier
            .width(70.dp)
            .height(40.dp)
            .border(1.dp, Rail, RoundedCornerShape(5.dp))
            .padding(horizontal = 4.dp, vertical = 2.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(value, color = Ink, fontFamily = FontFamily.Monospace, fontSize = 12.sp, fontWeight = FontWeight.Bold)
        Text(label, color = Muted, fontSize = 8.sp, lineHeight = 9.sp, maxLines = 1)
    }
}

@Composable
private fun HeaderWorkflowTab(
    label: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier
            .height(32.dp)
            .clip(RoundedCornerShape(4.dp))
            .clickable(onClick = onClick)
            .padding(top = 4.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Text(
            label,
            color = if (selected) Ink else Muted,
            fontSize = 11.sp,
            fontWeight = FontWeight.Bold,
        )
        Box(
            Modifier
                .fillMaxWidth()
                .height(2.dp)
                .background(if (selected) Green else Rail),
        )
    }
}

@Composable
private fun HeaderReviewQueue(
    label: String,
    count: Int,
    onClick: () -> Unit,
    condensed: Boolean,
    modifier: Modifier = Modifier,
) {
    Button(
        onClick = onClick,
        enabled = count > 0,
        modifier = modifier
            .width(if (condensed) 68.dp else 92.dp)
            .height(40.dp),
        shape = RoundedCornerShape(5.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = Acid,
            contentColor = Ink,
            disabledContainerColor = SoftPanel,
            disabledContentColor = Muted,
        ),
        contentPadding = PaddingValues(horizontal = 3.dp, vertical = 1.dp),
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(count.toString(), fontFamily = FontFamily.Monospace, fontSize = 10.sp, fontWeight = FontWeight.Bold)
            Text(
                label.replace(" ", "\n"),
                fontSize = if (condensed) 7.sp else 8.sp,
                lineHeight = if (condensed) 8.sp else 9.sp,
                maxLines = 2,
                textAlign = TextAlign.Center,
            )
        }
    }
}

@Composable
private fun AppSettingsDialog(
    displayAnalysisMeasurements: Boolean,
    onDisplayAnalysisMeasurements: (Boolean) -> Unit,
    onDismiss: () -> Unit,
) {
    val context = LocalContext.current
    var storeError by remember { mutableStateOf(false) }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Settings") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text("DISPLAY", color = Orange, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                UiScaleControl()
                HorizontalDivider(color = Rail)
                Text("Enjoying VolleySplice? A Google Play rating helps other volleyball players find it.")
                OutlinedButton(
                    onClick = {
                        storeError = !AppRating.openPlayStore(context)
                        if (!storeError) onDismiss()
                    },
                    modifier = Modifier.fillMaxWidth().testTag("settings-rate-app"),
                ) { Text("Rate VolleySplice on Google Play") }
                if (storeError) {
                    Text("Google Play could not be opened on this device.", color = Danger, fontSize = 12.sp)
                }
                AnalysisMeasurementsSetting(
                    checked = displayAnalysisMeasurements,
                    onCheckedChange = onDisplayAnalysisMeasurements,
                )
                Text(
                    "Rating opens Google Play. VolleySplice does not send your videos or rating activity anywhere.",
                    color = Muted,
                    fontSize = 12.sp,
                )
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text("Close") } },
    )
}

private val cleanupPolicies = listOf(
    SuppressionPolicyEngine.Policy.NONE,
    SuppressionPolicyEngine.Policy.CONSERVATIVE,
    SuppressionPolicyEngine.Policy.BALANCED,
    SuppressionPolicyEngine.Policy.AGGRESSIVE,
)

private fun cleanupLabel(policy: SuppressionPolicyEngine.Policy): String = when (policy) {
    SuppressionPolicyEngine.Policy.NONE -> "Off"
    SuppressionPolicyEngine.Policy.CONSERVATIVE -> "Light"
    SuppressionPolicyEngine.Policy.BALANCED -> "Recommended"
    SuppressionPolicyEngine.Policy.AGGRESSIVE -> "Strong"
}

@Composable
private fun EditorSettingsDialog(
    cleanupAvailable: Boolean,
    cleanupPreparing: Boolean,
    cleanupPolicy: SuppressionPolicyEngine.Policy,
    beforePaddingMs: Long,
    afterPaddingMs: Long,
    joinGapMs: Long,
    reviewThreshold: Float,
    displayAnalysisMeasurements: Boolean,
    onCleanupPolicy: (SuppressionPolicyEngine.Policy) -> Unit,
    onPrepareCleanup: () -> Unit,
    onBeforePadding: (Long) -> Unit,
    onAfterPadding: (Long) -> Unit,
    onJoinGap: (Long) -> Unit,
    onReviewThreshold: (Float) -> Unit,
    onDisplayAnalysisMeasurements: (Boolean) -> Unit,
    onDismiss: () -> Unit,
) {
    val context = LocalContext.current
    var storeError by remember { mutableStateOf(false) }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Settings") },
        text = {
            Column(
                Modifier.heightIn(max = 560.dp).verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                Text("DISPLAY", color = Orange, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                UiScaleControl()
                HorizontalDivider(color = Rail)
                Text("FINE-TUNE THE FINAL VIDEO", color = Orange, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("Automatic cleanup", Modifier.weight(1f), fontWeight = FontWeight.SemiBold)
                        Text(
                            if (cleanupAvailable) cleanupLabel(cleanupPolicy) else "Not added",
                            color = if (cleanupAvailable) Green else Muted,
                            fontSize = 12.sp,
                        )
                    }
                    Text(
                        "Leaves out likely non-play moments. You can restore anything while reviewing.",
                        color = Muted,
                        fontSize = 12.sp,
                    )
                    if (cleanupAvailable) {
                        Slider(
                            value = cleanupPolicies.indexOf(cleanupPolicy).coerceAtLeast(0).toFloat(),
                            onValueChange = { value ->
                                onCleanupPolicy(cleanupPolicies[value.roundToInt().coerceIn(cleanupPolicies.indices)])
                            },
                            valueRange = 0f..cleanupPolicies.lastIndex.toFloat(),
                            steps = cleanupPolicies.size - 2,
                        )
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text("Off", color = Muted, fontSize = 10.sp)
                            Text("Strong", color = Muted, fontSize = 10.sp)
                        }
                    } else {
                        OutlinedButton(
                            enabled = !cleanupPreparing,
                            onClick = onPrepareCleanup,
                            modifier = Modifier.fillMaxWidth(),
                        ) { Text(if (cleanupPreparing) "Adding cleanup…" else "Add automatic cleanup") }
                    }
                }
                PaddingControl("Extra time before each clip", beforePaddingMs, onBeforePadding)
                PaddingControl("Extra time after each clip", afterPaddingMs, onAfterPadding)
                PaddingControl("Keep short breaks under", joinGapMs, onJoinGap)
                Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Text("Clips to check", fontWeight = FontWeight.SemiBold, fontSize = 13.sp)
                    Text("Move toward More to have VolleySplice flag more suggested clips for review.", color = Muted, fontSize = 12.sp)
                    Slider(
                        value = reviewThreshold,
                        onValueChange = onReviewThreshold,
                        valueRange = 0f..1f,
                        steps = 19,
                    )
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text("Fewer", color = Muted, fontSize = 10.sp)
                        Text("More", color = Muted, fontSize = 10.sp)
                    }
                }
                HorizontalDivider(color = Rail)
                AnalysisMeasurementsSetting(
                    checked = displayAnalysisMeasurements,
                    onCheckedChange = onDisplayAnalysisMeasurements,
                )
                HorizontalDivider(color = Rail)
                Text("Enjoying VolleySplice? A Google Play rating helps other volleyball players find it.")
                OutlinedButton(
                    onClick = {
                        storeError = !AppRating.openPlayStore(context)
                        if (!storeError) onDismiss()
                    },
                    modifier = Modifier.fillMaxWidth().testTag("settings-rate-app"),
                ) { Text("Rate VolleySplice on Google Play") }
                if (storeError) {
                    Text("Google Play could not be opened on this device.", color = Danger, fontSize = 12.sp)
                }
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text("Done") } },
    )
}

@Composable
private fun UiScaleControl() {
    val setting = LocalUiScaleSetting.current
    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("UI size", Modifier.weight(1f), fontWeight = FontWeight.SemiBold)
            Text(
                "${(setting.scale * 100).roundToInt()}%",
                color = Green,
                fontFamily = FontFamily.Monospace,
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold,
            )
        }
        Text(
            "Scales the interface on top of Android's screen sizing. Use a smaller size to fit more in landscape.",
            color = Muted,
            fontSize = 10.sp,
        )
        Slider(
            value = setting.scale,
            onValueChange = setting.onScaleChange,
            valueRange = UI_SCALE_MIN..UI_SCALE_MAX,
            steps = 9,
            modifier = Modifier.testTag("settings-ui-scale"),
        )
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text("Smaller", color = Muted, fontSize = 10.sp)
            Spacer(Modifier.weight(1f))
            TextButton(
                onClick = { setting.onScaleChange(UI_SCALE_DEFAULT) },
                enabled = setting.scale != UI_SCALE_DEFAULT,
                modifier = Modifier.testTag("settings-ui-scale-default"),
            ) { Text("Default") }
            Spacer(Modifier.weight(1f))
            Text("Larger", color = Muted, fontSize = 10.sp)
        }
    }
}

@Composable
private fun AnalysisMeasurementsSetting(
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
) {
    Row(
        Modifier.fillMaxWidth().clickable { onCheckedChange(!checked) },
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Checkbox(checked = checked, onCheckedChange = onCheckedChange)
        Column(Modifier.weight(1f)) {
            Text("Show analysis measurements", fontWeight = FontWeight.SemiBold)
            Text("Show developer timing details in the editor.", color = Muted, fontSize = 11.sp)
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
    SectionCard("GETTING YOUR CLIPS READY", project.source.name) {
        Text(
            when (project.status) {
                ProjectStatus.QUEUED -> "Waiting to start"
                ProjectStatus.ANALYZING -> "VolleySplice is finding the rallies on this device"
                ProjectStatus.ERROR -> "VolleySplice could not finish preparing this video"
                ProjectStatus.READY -> "Your suggested clips are ready"
            },
            color = Muted,
        )
        if (project.status == ProjectStatus.ANALYZING || state.progress > 0f) {
            if (state.stepMeasurements.isEmpty()) {
                LinearProgressIndicator(progress = { state.progress }, modifier = Modifier.fillMaxWidth())
            }
        }
        val detail = state.detail.ifBlank { project.error.orEmpty() }
        if (project.status == ProjectStatus.ERROR && detail.isNotBlank()) {
            Text(detail, fontSize = 12.sp, color = Danger)
        } else if (project.status == ProjectStatus.ANALYZING) {
            Text("Keep VolleySplice open while it prepares the review.", fontSize = 12.sp, color = Muted)
        }
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
        Text("Your video stays on this device.", color = Muted, fontSize = 11.sp)
        if (sourceAvailable == false) {
            Text(
                relinkMessage
                    ?: "Choose the original video again so VolleySplice can continue.",
                color = Danger,
                fontSize = 12.sp,
            )
            Button(enabled = !relinkingSource, onClick = onRelink) {
                Text(if (relinkingSource) "Checking…" else "Choose original video")
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
                val videoPerformance = performance?.takeIf {
                    step.id == "video" && it.framesPerSecond() > 0.0
                }
                val etaSeconds = if (
                    step.status == InferenceStepStatus.RUNNING &&
                    step.fraction >= .03 && elapsedSeconds >= .5
                ) {
                    videoPerformance?.etaSeconds()?.takeIf { it.isFinite() && it >= 0.0 }
                        ?: (elapsedSeconds * (1.0 - step.fraction) / step.fraction)
                } else null
                val measuredRate = when {
                    videoPerformance != null && videoPerformance.realtimeRatio() > 0.0 ->
                        String.format(
                            Locale.US,
                            "%.1f frames/s · %.2fx realtime",
                            videoPerformance.framesPerSecond(),
                            videoPerformance.realtimeRatio(),
                        )
                    videoPerformance != null ->
                        String.format(Locale.US, "%.1f frames/s", videoPerformance.framesPerSecond())
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
private fun NewProjectIntro() {
    Surface(
        color = Paper,
        shape = RoundedCornerShape(10.dp),
        modifier = Modifier.fillMaxWidth().border(1.dp, Rail, RoundedCornerShape(10.dp)),
    ) {
        Column(
            Modifier.fillMaxWidth().padding(horizontal = 18.dp, vertical = 22.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Text(
                "Bump. Set. Splice.",
                color = Green,
                fontFamily = FontFamily.Monospace,
                fontSize = 10.sp,
                fontWeight = FontWeight.Black,
                letterSpacing = 1.sp,
            )
            Text(
                "Choose the game.\nVolleySplice finds the rallies.",
                color = Ink,
                fontSize = 30.sp,
                fontWeight = FontWeight.Black,
                lineHeight = 32.sp,
                letterSpacing = (-.8).sp,
            )
            Text(
                "Choose a game video, confirm the part to analyze, then let VolleySplice prepare the review timeline.",
                color = Muted,
                fontSize = 13.sp,
                lineHeight = 20.sp,
            )
            Column(
                Modifier.fillMaxWidth().border(1.dp, Rail, RoundedCornerShape(6.dp)),
            ) {
                Row(Modifier.fillMaxWidth()) {
                    PipelineStep(1, "Choose video", Modifier.weight(1f))
                    PipelineStep(2, "Set game window", Modifier.weight(1f))
                }
                HorizontalDivider(color = Rail)
                Row(Modifier.fillMaxWidth()) {
                    PipelineStep(3, "Analyze locally", Modifier.weight(1f))
                    PipelineStep(4, "Review rallies", Modifier.weight(1f))
                }
            }
            Text("Your video stays on this device.", color = Muted, fontSize = 11.sp)
        }
    }
}

@Composable
private fun PipelineStep(number: Int, label: String, modifier: Modifier = Modifier) {
    Row(
        modifier = modifier.padding(horizontal = 10.dp, vertical = 11.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Surface(color = Acid, shape = RoundedCornerShape(6.dp)) {
            Text(
                number.toString(),
                Modifier.size(28.dp).padding(top = 6.dp),
                color = Ink,
                fontFamily = FontFamily.Monospace,
                fontSize = 12.sp,
                fontWeight = FontWeight.Black,
                textAlign = TextAlign.Center,
            )
        }
        Text(label, color = Ink, fontSize = 11.sp, fontWeight = FontWeight.Bold)
    }
}

@Composable
private fun EditorWorkflow() {
    Row(
        Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        WorkflowStep(1, "Video", done = true)
        HorizontalDivider(Modifier.weight(1f).padding(horizontal = 8.dp), color = Rail)
        WorkflowStep(2, "Review", current = true)
        HorizontalDivider(Modifier.weight(1f).padding(horizontal = 8.dp), color = Rail)
        WorkflowStep(3, "Export")
    }
}

@Composable
private fun WorkflowStep(
    number: Int,
    label: String,
    done: Boolean = false,
    current: Boolean = false,
) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(5.dp)) {
        Surface(
            color = if (done || current) Acid else Paper,
            shape = RoundedCornerShape(50),
            modifier = Modifier.border(
                1.dp,
                if (current) Green else Rail,
                RoundedCornerShape(50),
            ),
        ) {
            Text(
                number.toString(),
                Modifier.size(27.dp).padding(top = 5.dp),
                color = Ink,
                fontSize = 11.sp,
                fontWeight = FontWeight.Black,
                textAlign = TextAlign.Center,
            )
        }
        Text(label, color = if (done || current) Ink else Muted, fontSize = 10.sp, fontWeight = FontWeight.Bold)
    }
}

@Composable
private fun CurrentSourceCard(filename: String, rallyCount: Int) {
    Surface(
        color = Paper,
        shape = RoundedCornerShape(10.dp),
        modifier = Modifier.fillMaxWidth().border(1.dp, Rail, RoundedCornerShape(10.dp)),
    ) {
        Column(
            Modifier.fillMaxWidth().padding(horizontal = 14.dp, vertical = 11.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                Column(Modifier.weight(1.2f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Text("NOW REVIEWING", color = Orange, fontSize = 9.sp, fontWeight = FontWeight.Black, letterSpacing = .8.sp)
                    Text(filename, maxLines = 1, overflow = TextOverflow.Ellipsis, fontSize = 13.sp, fontWeight = FontWeight.Bold)
                }
                Column(Modifier.weight(.8f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Text("RALLIES FOUND", color = Orange, fontSize = 9.sp, fontWeight = FontWeight.Black, letterSpacing = .8.sp)
                    Text("$rallyCount suggested clips", fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
                }
            }
            Text("● Saved on this device", color = Green, fontSize = 10.sp, fontWeight = FontWeight.SemiBold)
        }
    }
}

@Composable
private fun NewProjectCard(
    selected: SourceSelection?,
    preparing: Boolean,
    state: InferenceUiState,
    generateSideSwitchMarkers: Boolean,
    queueCount: Int,
    gameStartMs: Long,
    gameEndMs: Long,
    onSelect: () -> Unit,
    onImportFeedback: () -> Unit,
    onQueue: () -> Unit,
    onGameStart: (Long) -> Unit,
    onGameEnd: (Long) -> Unit,
    onFullVideo: () -> Unit,
    onGenerateSideSwitchMarkers: (Boolean) -> Unit,
    guidedTourTargets: GuidedTourTargets? = null,
) {
    SectionCard(
        "STEP 1 OF 3",
        selected?.displayName ?: "Choose your game video",
    ) {
        Text(
            if (queueCount == 0) {
                if (selected == null) "Pick a video from this device. Nothing will be uploaded."
                else "Ready to set up · your video stays on this device."
            }
            else "This video will start after the ${if (queueCount == 1) "current video" else "$queueCount videos"} finishes.",
            color = Muted,
            fontSize = 12.sp,
        )
        if (selected?.media == null) {
            Button(
                enabled = !preparing,
                onClick = onSelect,
                colors = ButtonDefaults.buttonColors(
                    containerColor = Acid,
                    contentColor = Ink,
                ),
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 48.dp)
                    .guidedTourTarget("setup-source", guidedTourTargets),
            ) { Text("Choose video") }
            TextButton(
                enabled = !preparing,
                onClick = onImportFeedback,
                modifier = Modifier.align(Alignment.CenterHorizontally),
            ) {
                Text("Open saved projects", color = Muted)
            }
        } else {
            OutlinedButton(
                enabled = !preparing,
                onClick = onSelect,
                modifier = Modifier
                    .fillMaxWidth()
                    .guidedTourTarget("setup-source", guidedTourTargets),
            ) { Text("Choose a different video") }
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
            Surface(
                modifier = Modifier
                    .fillMaxWidth()
                    .border(1.dp, Rail, RoundedCornerShape(6.dp))
                    .clickable(enabled = !preparing) {
                        onGenerateSideSwitchMarkers(!generateSideSwitchMarkers)
                    },
                color = if (generateSideSwitchMarkers) SoftPanel else Paper,
                shape = RoundedCornerShape(6.dp),
            ) {
                Row(
                    Modifier.padding(horizontal = 10.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.Top,
                ) {
                    Checkbox(
                        enabled = !preparing,
                        checked = generateSideSwitchMarkers,
                        onCheckedChange = onGenerateSideSwitchMarkers,
                    )
                    Column(Modifier.padding(top = 4.dp)) {
                        Text("Teams change court sides", fontWeight = FontWeight.SemiBold)
                        Text(
                            "VolleySplice will find side switches for the optional scoreboard.",
                            fontSize = 12.sp,
                            color = Muted,
                        )
                    }
                }
            }
            Button(
                enabled = !preparing && gameEndMs - gameStartMs >= 1_000,
                onClick = onQueue,
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 48.dp)
                    .guidedTourTarget("setup-create", guidedTourTargets),
            ) {
                Text(if (preparing) "Opening…" else "Find the rallies", fontWeight = FontWeight.Bold)
            }
            Text(
                "This may take a while for a long video. You can review another project while it runs.",
                color = Muted,
                fontSize = 11.sp,
            )
        }
        if (preparing || state.stage.isNotBlank()) {
            Text(state.detail, fontSize = 12.sp, color = Muted)
        }
        state.error?.let { Text(it, color = Danger, fontSize = 13.sp) }
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
    val layoutPreferences = remember(context) {
        context.getSharedPreferences(EDITOR_LAYOUT_PREFERENCES, Context.MODE_PRIVATE)
    }
    val durationMs = secondsToMs(checkNotNull(selected.media).durationSeconds())
    var playheadMs by remember(selected.uri) { mutableLongStateOf(0L) }
    var playing by remember(selected.uri) { mutableStateOf(false) }
    var compactPlayerHeightDp by remember {
        mutableFloatStateOf(
            layoutPreferences.getFloat(
                "setup_player_compact_height_dp",
                SETUP_PLAYER_COMPACT_DEFAULT_HEIGHT_DP,
            ).coerceIn(SETUP_PLAYER_MIN_HEIGHT_DP, SETUP_PLAYER_COMPACT_MAX_HEIGHT_DP),
        )
    }
    var desktopPlayerHeightDp by remember {
        mutableFloatStateOf(
            layoutPreferences.getFloat(
                "setup_player_desktop_height_dp",
                SETUP_PLAYER_DESKTOP_DEFAULT_HEIGHT_DP,
            ).coerceIn(SETUP_PLAYER_MIN_HEIGHT_DP, SETUP_PLAYER_DESKTOP_MAX_HEIGHT_DP),
        )
    }
    val displayDensity = LocalDensity.current.density
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
    LaunchedEffect(compactPlayerHeightDp, desktopPlayerHeightDp) {
        delay(250)
        layoutPreferences.edit()
            .putFloat("setup_player_compact_height_dp", compactPlayerHeightDp)
            .putFloat("setup_player_desktop_height_dp", desktopPlayerHeightDp)
            .apply()
    }
    BoxWithConstraints(Modifier.fillMaxWidth()) {
        val desktop = maxWidth >= 840.dp
        val playerHeightDp = if (desktop) desktopPlayerHeightDp else compactPlayerHeightDp
        val maxPlayerHeightDp = if (desktop) {
            SETUP_PLAYER_DESKTOP_MAX_HEIGHT_DP
        } else {
            SETUP_PLAYER_COMPACT_MAX_HEIGHT_DP
        }
        Column {
            Card(colors = CardDefaults.cardColors(containerColor = Color.Black)) {
                Box(Modifier.fillMaxWidth().height(playerHeightDp.dp)) {
                    if (!BuildConfig.BLACK_VIDEO_PREVIEW) {
                        ContentFrame(
                            player = player,
                            modifier = Modifier.fillMaxSize(),
                            surfaceType = SURFACE_TYPE_TEXTURE_VIEW,
                        )
                    }
                    Box(
                        Modifier
                            .fillMaxSize()
                            .clickable(enabled = enabled) {
                                if (player.isPlaying) player.pause() else player.play()
                            }
                            .semantics {
                                contentDescription = if (playing) "Pause video" else "Play video"
                            },
                    )
                }
            }
            VideoResizeHandle(description = "Resize setup video player height") { dragDeltaPx ->
                val resized = resizedPlayerHeight(
                    currentHeightDp = playerHeightDp,
                    dragDeltaPx = dragDeltaPx,
                    density = displayDensity,
                    minHeightDp = SETUP_PLAYER_MIN_HEIGHT_DP,
                    maxHeightDp = maxPlayerHeightDp,
                )
                if (desktop) {
                    desktopPlayerHeightDp = resized
                } else {
                    compactPlayerHeightDp = resized
                }
            }
        }
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
            Text("STEP 2 OF 3", color = Orange, fontSize = 11.sp, fontWeight = FontWeight.Bold)
            Text("Choose the part with the game", fontWeight = FontWeight.Bold)
            Text(
                "Leave the full video selected, or mark where the game starts and ends.",
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
            Text("Game starts · ${preciseTime(gameStartMs)}")
        }
        OutlinedButton(enabled = enabled, onClick = { onGameEnd(playheadMs) }) {
            Text("Game ends · ${preciseTime(gameEndMs)}")
        }
    }
    val analyzedMs = (gameEndMs - gameStartMs).coerceAtLeast(0)
    Text(
        "${compactTime(analyzedMs)} selected · ${compactTime(durationMs - analyzedMs)} left out",
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
                        "VolleySplice includes the following open-source components. " +
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
    analysisRunning: Boolean,
    displayAnalysisMeasurements: Boolean,
    onDisplayAnalysisMeasurements: (Boolean) -> Unit,
    servingSideProgress: Float?,
    servingSideProgressDetail: String?,
    servingSideStepMeasurements: List<InferenceStepMeasurement>,
    sourceAvailable: Boolean?,
    relinkingSource: Boolean,
    relinkMessage: String?,
    relinkFailed: Boolean,
    onProjectUpdated: (NativeProject) -> Unit,
    onRelink: () -> Unit,
    sourceControls: @Composable (EditorProjectSummary, () -> Unit) -> Unit,
    guidedTourTargets: GuidedTourTargets,
    guidedTourRestartSignal: Int,
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var draft by remember { mutableStateOf(initialDraft) }
    var selectedId by remember { mutableStateOf(initialDraft.cuts.firstOrNull()?.id.orEmpty()) }
    var playbackPositionMs by remember { mutableLongStateOf(seed.gameStartMs) }
    var isPlaying by remember { mutableStateOf(false) }
    var previewEndMs by remember { mutableStateOf<Long?>(null) }
    var message by remember { mutableStateOf("") }
    var exportState by remember(project.id) {
        mutableStateOf(
            ExportService.statusForProject(project.id)?.toUiState()
                ?: project.persistedExportUiState(),
        )
    }
    var pendingExportIntervals by remember { mutableStateOf<List<FinalCutInterval>?>(null) }
    var feedbackExporting by remember { mutableStateOf(false) }
    var confirmReset by remember { mutableStateOf(false) }
    var showRatingPrompt by remember { mutableStateOf(AppRating.hasPendingPrompt(context)) }
    var selectedSuggestionId by remember { mutableStateOf<String?>(null) }
    var suppressionPreparing by remember { mutableStateOf(false) }
    var showEditorSettings by remember { mutableStateOf(false) }
    var showProjectOptions by remember { mutableStateOf(false) }
    var selectedScoreMarkerId by remember { mutableStateOf<String?>(null) }
    var showYouTubeChapters by remember { mutableStateOf(false) }
    var youtubeChaptersStatus by remember { mutableStateOf<String?>(null) }
    var pendingYouTubeChaptersText by remember { mutableStateOf<String?>(null) }
    var desktopTab by remember(project.id) { mutableStateOf(DesktopEditorTab.REVIEW) }
    val desktopLayoutPreferences = remember(context) {
        context.getSharedPreferences(EDITOR_LAYOUT_PREFERENCES, Context.MODE_PRIVATE)
    }
    var compactPlayerHeightDp by remember {
        mutableFloatStateOf(
            desktopLayoutPreferences.getFloat(
                "editor_player_compact_height_dp",
                desktopLayoutPreferences.getFloat(
                    "compact_player_height_dp",
                    COMPACT_PLAYER_DEFAULT_HEIGHT_DP,
                ),
            ).coerceIn(COMPACT_PLAYER_MIN_HEIGHT_DP, COMPACT_PLAYER_MAX_HEIGHT_DP),
        )
    }
    var desktopPlayerHeightDp by remember {
        mutableFloatStateOf(
            desktopLayoutPreferences.getFloat(
                "editor_player_desktop_height_dp",
                desktopLayoutPreferences.getFloat(
                    "player_height_dp",
                    DESKTOP_PLAYER_DEFAULT_HEIGHT_DP,
                ),
            ).coerceIn(DESKTOP_PLAYER_MIN_HEIGHT_DP, DESKTOP_PLAYER_MAX_HEIGHT_DP),
        )
    }
    var desktopLeftPaneWidthDp by remember {
        mutableFloatStateOf(
            desktopLayoutPreferences.getFloat(
                "left_pane_width_dp",
                DESKTOP_LEFT_PANE_DEFAULT_WIDTH_DP,
            ).coerceIn(DESKTOP_LEFT_PANE_MIN_WIDTH_DP, DESKTOP_LEFT_PANE_MAX_WIDTH_DP),
        )
    }
    var desktopRightPaneWidthDp by remember {
        mutableFloatStateOf(
            desktopLayoutPreferences.getFloat(
                "right_pane_width_dp",
                DESKTOP_RIGHT_PANE_DEFAULT_WIDTH_DP,
            ).coerceIn(DESKTOP_RIGHT_PANE_MIN_WIDTH_DP, DESKTOP_RIGHT_PANE_MAX_WIDTH_DP),
        )
    }
    val displayDensity = LocalDensity.current.density

    LaunchedEffect(
        compactPlayerHeightDp,
        desktopPlayerHeightDp,
        desktopLeftPaneWidthDp,
        desktopRightPaneWidthDp,
    ) {
        delay(250)
        desktopLayoutPreferences.edit()
            .putFloat("editor_player_compact_height_dp", compactPlayerHeightDp)
            .putFloat("editor_player_desktop_height_dp", desktopPlayerHeightDp)
            .putFloat("left_pane_width_dp", desktopLeftPaneWidthDp)
            .putFloat("right_pane_width_dp", desktopRightPaneWidthDp)
            .apply()
    }

    val player = remember {
        ExoPlayer.Builder(context).build().apply {
            setSeekParameters(SeekParameters.EXACT)
            setMediaItem(MediaItem.fromUri(seed.sourceUri))
            prepare()
            seekTo(seed.gameStartMs)
        }
    }
    val sortedCuts = draft.cuts.sortedWith(compareBy<EditableCut> { it.keepStartMs }.thenBy { it.keepEndMs })
    val finalMaterialization = EditorMath.materialize(draft, seed.suppression)
    val finalIntervals = finalMaterialization.intervals
    val pendingConfidenceReviewCutIds = sortedCuts.asSequence()
        .filter { cut ->
            cut.origin == CutOrigin.INFERRED && cut.included &&
                cut.id !in draft.reviewedCutIds &&
                (cut.isModelDisagreement() || cut.confidence < draft.confidenceReviewThreshold)
        }
        .map { it.id }
        .toSet()
    val editableRallies = EditorMath.editableRallyGroups(
        sortedCuts,
        finalIntervals,
        draft.scoreTracking.serveMarkers.takeIf { draft.scoreTracking.enabled }.orEmpty(),
        pendingConfidenceReviewCutIds,
    )
    val selectedRally = editableRallies.firstOrNull { selectedId in it.cutIds }
        ?: editableRallies.firstOrNull()
    val selected = selectedRally?.asEditableCut()
    val selectedIndex = selectedRally?.let { editableRallies.indexOf(it) } ?: -1
    val effectiveIds = EditorMath.effectiveKeptIds(draft, seed.suppression)
    val effectiveRallies = editableRallies.filter { rally -> rally.cutIds.any { it in effectiveIds } }
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
    val scoreEffectiveTimestampMs = ScoreReducer.scoreBoundaryTimestamp(
        playbackPositionMs,
        preparedScoreOverlay.rallyRanges,
        preparedScoreOverlay.tracking,
        preparedScoreOverlay.mergedRanges,
    )
    val visibleScore = ScoreReducer.deriveAt(preparedScoreOverlay.tracking, scoreEffectiveTimestampMs)
    val scoreEffectiveServeId = preparedScoreOverlay.tracking.serveMarkers
        .sortedWith(compareBy<ServeMarker> { it.timestampMs }.thenBy { it.id })
        .lastOrNull { it.timestampMs <= scoreEffectiveTimestampMs }
        ?.id
    LaunchedEffect(scoreEffectiveServeId) {
        if (scoreEffectiveServeId != null) selectedScoreMarkerId = scoreEffectiveServeId
    }
    val visibleIgnoredIntervals = draft.ignoredIntervals.filter {
        it.endMs > seed.gameStartMs && it.startMs < seed.gameEndMs
    }
    val activeSuggestions = EditorMath.activeSuggestions(draft, seed.suppression)
    val appliedSuggestionIds = activeSuggestions.filter {
        EditorMath.suggestionEffectiveDecision(draft, it) == SuppressionDecision.SUPPRESS
    }.map { it.fragmentId() }.toSet()
    val userRemovedSuggestionIds = activeSuggestions.filter { suggestion ->
        draft.suppressionDecisionOverrides[suggestion.logicalId()] == SuppressionDecision.SUPPRESS
    }.mapTo(mutableSetOf()) { it.fragmentId() }
    val userRemovedCleanupCutIds = sortedCuts.filter { cut ->
        cut.id !in effectiveIds && activeSuggestions.any { suggestion ->
            suggestion.fragmentId() in userRemovedSuggestionIds &&
                cut.coreStartMs < suggestion.endMs() && suggestion.startMs() < cut.coreEndMs
        }
    }.mapTo(mutableSetOf()) { it.id }
    val reviewableCleanupSuggestions = activeSuggestions.filter { suggestion ->
        EditorMath.firstReviewableTime(
            suggestion.startMs(),
            suggestion.endMs(),
            visibleIgnoredIntervals,
        ) != null
    }
    val pendingCleanupReview = reviewableCleanupSuggestions.filter { suggestion ->
        suggestion.logicalId() !in draft.suppressionDecisionOverrides
    }
    val selectedSuggestion = activeSuggestions.firstOrNull { it.fragmentId() == selectedSuggestionId }
    val selectedClipSuggestion = selectedSuggestion ?: selectedRally?.let { rally ->
        activeSuggestions.firstOrNull { suggestion ->
            rally.cuts.any { cut ->
                cut.coreStartMs < suggestion.endMs() && suggestion.startMs() < cut.coreEndMs
            }
        }
    }
    val lowConfidence = editableRallies.filter { rally ->
        rally.cuts.any {
            it.origin == CutOrigin.INFERRED && it.included && it.id in effectiveIds &&
                (it.isModelDisagreement() || it.confidence < draft.confidenceReviewThreshold)
        }
    }
    val pendingReview = lowConfidence
        .filter { rally -> rally.cutIds.any { it !in draft.reviewedCutIds } }
        .filter { rally ->
            EditorMath.firstReviewableTime(
                rally.keepStartMs,
                rally.keepEndMs,
                visibleIgnoredIntervals,
            ) != null
        }
    val reviewableServeReviews = preparedScoreOverlay.tracking.serveMarkers
        .filter { marker ->
            marker.side == ServingSide.REVIEW &&
                !EditorMath.isTimestampIgnored(marker.timestampMs, visibleIgnoredIntervals)
        }
        .sortedBy { it.timestampMs }
    val reviewAttentionCount = pendingReview.size + pendingCleanupReview.size + reviewableServeReviews.size
    val reviewAttentionBreakdown = buildList {
        if (pendingReview.isNotEmpty()) add("${pendingReview.size} ${if (pendingReview.size == 1) "clip" else "clips"}")
        if (pendingCleanupReview.isNotEmpty()) add("${pendingCleanupReview.size} cleanup")
        if (reviewableServeReviews.isNotEmpty()) add(
            "${reviewableServeReviews.size} ${if (reviewableServeReviews.size == 1) "serve" else "serves"}",
        )
    }.joinToString(" · ")
    val joinedGaps = finalIntervals.flatMap { it.joinedGaps }
    val totalFinalMs = EditorMath.totalFinalMs(finalIntervals)
    val exportPending = exportState.status == "queued" || exportState.status == "running"
    val leftOutRallyCount = editableRallies.count { rally -> rally.cutIds.none { it in effectiveIds } }
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

    fun updateRallyCuts(ids: Set<String>, mutate: (EditableCut) -> EditableCut) {
        updateDraft { current ->
            current.copy(cuts = current.cuts.map { if (it.id in ids) mutate(it) else it })
        }
    }

    fun seekTo(positionMs: Long) {
        val target = positionMs.coerceIn(seed.gameStartMs, seed.gameEndMs)
        playbackPositionMs = target
        player.seekTo(target)
    }

    fun togglePlayback() {
        previewEndMs = null
        if (player.isPlaying) {
            player.pause()
        } else {
            if (player.currentPosition >= seed.gameEndMs) seekTo(seed.gameStartMs)
            if (draft.finalPreviewEnabled) {
                EditorMath.nextFinalTime(finalIntervals, player.currentPosition)
                    ?.let { if (it != player.currentPosition) seekTo(it) }
            }
            player.play()
        }
    }

    fun selectSuggestion(suggestion: AnalysisTypes.SuppressionSuggestion) {
        player.pause()
        previewEndMs = null
        selectedSuggestionId = suggestion.fragmentId()
        val parent = sortedCuts.firstOrNull { cut ->
            cut.coreStartMs < suggestion.endMs() && suggestion.startMs() < cut.coreEndMs
        }
        parent?.let { selectedId = it.id }
        val reviewStart = max(seed.gameStartMs, suggestion.startMs() - 2_000)
        val reviewTime = EditorMath.firstReviewableTime(
            reviewStart,
            suggestion.endMs(),
            visibleIgnoredIntervals,
        ) ?: reviewStart
        seekTo(reviewTime)
    }

    fun setKeepBoundary(side: String, valueMs: Long) {
        val rally = selectedRally ?: return
        val cut = if (side == "start") rally.cuts.minBy { it.keepStartMs }
            else rally.cuts.maxBy { it.keepEndMs }
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
        val rally = selectedRally ?: return
        val cut = if (side == "start") rally.cuts.minBy { it.coreStartMs }
            else rally.cuts.maxBy { it.coreEndMs }
        updateDraft { current ->
            if (side == "start") {
                EditorMath.setCoreStart(current, cut.id, valueMs, seed.gameStartMs)
            } else {
                EditorMath.setCoreEnd(current, cut.id, valueMs, seed.gameEndMs)
            }
        }
    }

    fun splitSelectedAtPlayhead() {
        val cut = selectedRally?.cuts?.firstOrNull {
            playbackPositionMs > it.coreStartMs && playbackPositionMs < it.coreEndMs
        } ?: return
        var split: CutSplitResult? = null
        updateDraft { current ->
            EditorMath.splitCut(
                current,
                cut.id,
                playbackPositionMs,
                seed.gameStartMs,
                seed.gameEndMs,
            )?.also { split = it }?.draft?.copy(
                reviewedCutIds = current.reviewedCutIds - cut.id,
            ) ?: current
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

    fun reviewNextClip() {
        val current = pendingReview.firstOrNull { selectedId in it.cutIds }
        if (current != null) {
            updateDraft { it.copy(reviewedCutIds = it.reviewedCutIds + current.cutIds) }
        }
        val remaining = pendingReview.filterNot { it.id == current?.id }
        val next = remaining.firstOrNull { it.keepStartMs >= playbackPositionMs }
            ?: remaining.firstOrNull()
        if (next == null) {
            message = "Clip review complete"
        } else {
            selectedSuggestionId = null
            selectedId = next.id
            val reviewTime = EditorMath.firstReviewableTime(
                next.keepStartMs,
                next.keepEndMs,
                visibleIgnoredIntervals,
            ) ?: return
            seekTo(reviewTime)
            message = "Check this suggested clip, then keep or remove it"
        }
    }

    fun reviewNextCleanup() {
        EditorMath.nextSuppressionSuggestion(
            pendingCleanupReview,
            playbackPositionMs,
            selectedSuggestionId,
        )?.let(::selectSuggestion)
    }

    fun reviewNextServe() {
        val reviewServes = reviewableServeReviews
        if (reviewServes.isEmpty()) return
        val selectedReviewIndex = reviewServes.indexOfFirst { it.id == selectedScoreMarkerId }
        val next = if (selectedReviewIndex >= 0) {
            reviewServes[(selectedReviewIndex + 1) % reviewServes.size]
        } else {
            reviewServes.firstOrNull { it.timestampMs >= playbackPositionMs }
                ?: reviewServes.first()
        }
        selectedScoreMarkerId = next.id
        seekTo(next.timestampMs)
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
    val youtubeChaptersLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.CreateDocument("text/plain"),
    ) { uri ->
        val text = pendingYouTubeChaptersText
        pendingYouTubeChaptersText = null
        if (uri != null && text != null) runCatching {
            context.contentResolver.openOutputStream(uri, "w")!!.bufferedWriter().use {
                it.write(text)
                it.newLine()
            }
        }.onSuccess {
            youtubeChaptersStatus = "Saved the YouTube chapters text file."
        }.onFailure {
            youtubeChaptersStatus = it.message ?: "Could not save the chapters."
        }
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
                message = "Saved the detailed project file"
            }.onFailure {
                message = it.message ?: "Could not save the detailed project file"
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
            override fun onReceive(receiverContext: Context?, intent: Intent?) {
                if (intent?.action != ExportService.ACTION_PROGRESS) return
                if (intent.getStringExtra(ExportService.EXTRA_PROJECT_ID) != project.id) return
                exportState = ExportUiState(
                    jobId = intent.getStringExtra(ExportService.EXTRA_JOB_ID),
                    status = intent.getStringExtra(ExportService.EXTRA_STATUS) ?: "running",
                    progress = intent.getIntExtra(ExportService.EXTRA_PROGRESS, 0),
                    detail = intent.getStringExtra(ExportService.EXTRA_DETAIL).orEmpty(),
                )
                if (exportState.status == "complete" && AppRating.hasPendingPrompt(context)) {
                    showRatingPrompt = true
                }
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
    if (showRatingPrompt) {
        AlertDialog(
            onDismissRequest = {
                AppRating.deferPrompt(context)
                showRatingPrompt = false
            },
            title = { Text("Enjoying VolleySplice?") },
            text = {
                Text("Your highlight video is ready. If VolleySplice helped, would you rate it on Google Play?")
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        AppRating.openPlayStore(context)
                        showRatingPrompt = false
                    },
                    modifier = Modifier.testTag("export-rate-app"),
                ) { Text("Rate VolleySplice") }
            },
            dismissButton = {
                TextButton(onClick = {
                    AppRating.deferPrompt(context)
                    showRatingPrompt = false
                }) { Text("Not now") }
            },
        )
    }

    LaunchedEffect(draft) {
        delay(200)
        withContext(Dispatchers.IO) { store.save(draft) }
    }

    LaunchedEffect(draft.playbackRate) {
        player.playbackParameters = PlaybackParameters(draft.playbackRate)
    }

    LaunchedEffect(player, draft.finalPreviewEnabled, finalIntervals, editableRallies, previewEndMs) {
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
            editableRallies.filter { it.keepStartMs <= position }.maxByOrNull { it.keepStartMs }?.let { reached ->
                if (reached.id != selectedId) selectedId = reached.id
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
            text = { Text("All clip boundaries, included or excluded choices, score markers, added clips, and omitted sections will be discarded.") },
            confirmButton = {
                TextButton(onClick = {
                    store.clear()
                    draft = EditorMath.newDraft(seed)
                    selectedId = draft.cuts.firstOrNull()?.id.orEmpty()
                    selectedSuggestionId = null
                    confirmReset = false
                    message = "Started the edit over from the suggested clips"
                }) { Text("Reset") }
            },
            dismissButton = { TextButton(onClick = { confirmReset = false }) { Text("Cancel") } },
        )
    }

    if (showEditorSettings) {
        EditorSettingsDialog(
            cleanupAvailable = seed.suppression != null,
            cleanupPreparing = suppressionPreparing,
            cleanupPolicy = draft.selectedSuppressionPolicy,
            beforePaddingMs = draft.beforePaddingMs,
            afterPaddingMs = draft.afterPaddingMs,
            joinGapMs = draft.joinGapMs,
            reviewThreshold = draft.confidenceReviewThreshold,
            displayAnalysisMeasurements = displayAnalysisMeasurements,
            onCleanupPolicy = { policy ->
                updateDraft {
                    it.copy(
                        selectedSuppressionPolicy = policy,
                        suppressionInitialBehavior = if (policy == SuppressionPolicyEngine.Policy.NONE) {
                            it.suppressionInitialBehavior
                        } else SuppressionInitialBehavior.DISABLE_INITIALLY,
                    )
                }
                if (policy == SuppressionPolicyEngine.Policy.NONE) selectedSuggestionId = null
            },
            onPrepareCleanup = {
                suppressionPreparing = true
                scope.launch {
                    val result = runCatching {
                        withContext(Dispatchers.IO) { SuppressionAugmenter.augment(context, project) }
                    }
                    suppressionPreparing = false
                    result.onSuccess {
                        onProjectUpdated(it)
                        message = "Automatic cleanup is ready"
                    }.onFailure {
                        message = it.message ?: "Could not add automatic cleanup"
                    }
                }
            },
            onBeforePadding = { before ->
                updateDraft { EditorMath.applyPadding(
                    it, before, it.afterPaddingMs, seed.durationMs, seed.gameStartMs, seed.gameEndMs,
                ) }
            },
            onAfterPadding = { after ->
                updateDraft { EditorMath.applyPadding(
                    it, it.beforePaddingMs, after, seed.durationMs, seed.gameStartMs, seed.gameEndMs,
                ) }
            },
            onJoinGap = { joinGap -> updateDraft { it.copy(joinGapMs = joinGap.coerceIn(0, MAX_JOIN_GAP_MS)) } },
            onReviewThreshold = { threshold -> updateDraft { it.copy(confidenceReviewThreshold = threshold) } },
            onDisplayAnalysisMeasurements = onDisplayAnalysisMeasurements,
            onDismiss = { showEditorSettings = false },
        )
    }

    if (showProjectOptions) {
        AlertDialog(
            onDismissRequest = { showProjectOptions = false },
            title = { Text("Project options") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text(
                        "Save the edit and score markers as a small project file, or return this review to its original suggestions.",
                        color = Muted,
                        fontSize = 12.sp,
                    )
                    OutlinedButton(
                        enabled = !feedbackExporting,
                        onClick = {
                            showProjectOptions = false
                            feedbackSaveLauncher.launch(ModelFeedbackExporter.filename(seed.displayName))
                        },
                        modifier = Modifier.fillMaxWidth(),
                    ) { Text(if (feedbackExporting) "Saving project…" else "Save project") }
                    OutlinedButton(
                        onClick = {
                            showProjectOptions = false
                            confirmReset = true
                        },
                        modifier = Modifier.fillMaxWidth(),
                    ) { Text("Start this review over", color = Danger) }
                }
            },
            confirmButton = {
                TextButton(onClick = { showProjectOptions = false }) { Text("Close") }
            },
        )
    }

    val editorSummary = EditorProjectSummary(
        outputDurationMs = totalFinalMs,
        kept = effectiveRallies.size,
        needsReview = reviewAttentionCount,
        cleanupReviewCount = pendingCleanupReview.size,
        clipReviewCount = pendingReview.size,
        serveReviewCount = reviewableServeReviews.size,
        onReviewCleanup = ::reviewNextCleanup,
        onReviewClips = ::reviewNextClip,
        onReviewServes = ::reviewNextServe,
        desktopTab = desktopTab,
        onDesktopTab = { desktopTab = it },
    )

    Box(Modifier.fillMaxSize()) {
        Scaffold(containerColor = Paper) { scaffoldPadding ->
            BoxWithConstraints(
                modifier = Modifier.fillMaxSize().padding(scaffoldPadding),
            ) {
                if (editorLayoutMode(maxWidth.value.roundToInt(), maxHeight.value.roundToInt()) ==
                    EditorLayoutMode.DESKTOP
                ) {
                    if (desktopTab == DesktopEditorTab.EXPORT) {
                        LargeScreenExportLayout(
                            header = {
                                Box(Modifier.guidedTourTarget("editor-header", guidedTourTargets)) {
                                    sourceControls(editorSummary) { showEditorSettings = true }
                                }
                            },
                            mp4Option = {
                                SectionCard(
                                    "MP4 VIDEO",
                                    "${compactTime(totalFinalMs)} · ${effectiveRallies.size} clips included",
                                    modifier = Modifier.guidedTourTarget("editor-export", guidedTourTargets),
                                ) {
                                    Text(
                                        "Create the finished cut at the original video size. The recording stays on this device.",
                                        color = Muted,
                                        fontSize = 12.sp,
                                    )
                                    if (draft.scoreTracking.enabled) {
                                        Row(verticalAlignment = Alignment.CenterVertically) {
                                            Column(Modifier.weight(1f)) {
                                                Text("Add score to video", fontWeight = FontWeight.SemiBold)
                                                Text("Include the checked scoreboard", color = Muted, fontSize = 10.sp)
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
                                    }
                                    if (exportState.status == "running") {
                                        LinearProgressIndicator(
                                            progress = { exportState.progress / 100f },
                                            modifier = Modifier.fillMaxWidth(),
                                        )
                                    }
                                    if (exportState.detail.isNotBlank()) {
                                        Text(
                                            exportState.detail,
                                            color = if (exportState.status == "failed") Danger else Green,
                                            fontSize = 11.sp,
                                        )
                                    }
                                    Button(
                                        enabled = finalIntervals.isNotEmpty() && !exportPending,
                                        onClick = {
                                            pendingExportIntervals = finalIntervals
                                            exportLauncher.launch(exportFilename(seed.displayName))
                                        },
                                        modifier = Modifier.fillMaxWidth().heightIn(min = 52.dp),
                                    ) {
                                        Text(if (exportPending) "Creating MP4…" else "Save MP4", fontWeight = FontWeight.Bold)
                                    }
                                }
                            },
                            chaptersOption = {
                                SectionCard("YOUTUBE CHAPTERS", null) {
                                    Text(
                                        "Generate timestamped rally chapters to paste into the YouTube description, or save them as a text file.",
                                        color = Muted,
                                        fontSize = 12.sp,
                                    )
                                    Text(
                                        "${effectiveRallies.size} rally chapters ready",
                                        color = Green,
                                        fontWeight = FontWeight.SemiBold,
                                    )
                                    Button(
                                        enabled = finalIntervals.isNotEmpty(),
                                        onClick = {
                                            youtubeChaptersStatus = null
                                            showYouTubeChapters = true
                                        },
                                        modifier = Modifier.fillMaxWidth().heightIn(min = 52.dp).testTag("youtube-chapters-button"),
                                    ) { Text("Get YouTube chapters", fontWeight = FontWeight.Bold) }
                                }
                            },
                            projectOption = {
                                SectionCard("PROJECT FILE", null) {
                                    Text(
                                        "Save the edit, rally decisions, and score markers. The original video is not included.",
                                        color = Muted,
                                        fontSize = 12.sp,
                                    )
                                    Text(
                                        "Open this file later to continue editing.",
                                        color = Green,
                                        fontWeight = FontWeight.SemiBold,
                                    )
                                    Button(
                                        enabled = !feedbackExporting,
                                        onClick = {
                                            feedbackSaveLauncher.launch(ModelFeedbackExporter.filename(seed.displayName))
                                        },
                                        modifier = Modifier.fillMaxWidth().heightIn(min = 52.dp),
                                    ) {
                                        Text(if (feedbackExporting) "Saving project…" else "Save project", fontWeight = FontWeight.Bold)
                                    }
                                }
                            },
                        )
                    } else {
                    LargeScreenEditorLayout(
                        leftPaneWidthDp = desktopLeftPaneWidthDp,
                        rightPaneWidthDp = desktopRightPaneWidthDp,
                        onLeftPaneWidthChange = { desktopLeftPaneWidthDp = it },
                        onRightPaneWidthChange = { desktopRightPaneWidthDp = it },
                        header = {
                            Box(Modifier.guidedTourTarget("editor-header", guidedTourTargets)) {
                                sourceControls(
                                    editorSummary,
                                    { showEditorSettings = true },
                                )
                            }
                        },
                        leftPane = {
                            ScoreTrackingPanel(
                                enabled = draft.scoreTracking.enabled,
                                tracking = draft.scoreTracking,
                                visibleTracking = preparedScoreOverlay.tracking,
                                visibleScore = visibleScore,
                                selectedMarkerId = selectedScoreMarkerId,
                                currentTimestampMs = playbackPositionMs,
                                scoreEffectiveTimestampMs = scoreEffectiveTimestampMs,
                                servingSideStatus = project.servingSideStatus,
                                servingSideError = project.servingSideError,
                                servingSideProgress = servingSideProgress,
                                servingSideProgressDetail = servingSideProgressDetail,
                                servingSideStepMeasurements = servingSideStepMeasurements,
                                sideSwitchEnabled = project.sideSwitchEnabled,
                                desktop = true,
                                modifier = Modifier.guidedTourTarget("editor-score-panel", guidedTourTargets),
                                toggleModifier = Modifier.guidedTourTarget("editor-score-toggle", guidedTourTargets),
                                onEnabledChange = { enabled ->
                                    updateDraft { it.copy(scoreTracking = it.scoreTracking.copy(enabled = enabled)) }
                                    val missingOutput = project.servingSide == null ||
                                        (project.sideSwitchEnabled && project.sideSwitch == null)
                                    if (enabled && missingOutput &&
                                        project.servingSideStatus != ServingSideAnalysisStatus.QUEUED &&
                                        project.servingSideStatus != ServingSideAnalysisStatus.ANALYZING
                                    ) {
                                        if (sourceAvailable == false) {
                                            message = "Re-link the source video to prepare score tracking"
                                        } else {
                                            message = "Preparing score markers"
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
                            )
                            DesktopRallyLedger(
                                rallies = editableRallies,
                                selectedId = selectedRally?.id,
                                effectiveIds = effectiveIds,
                                reviewedIds = draft.reviewedCutIds,
                                lowConfidenceIds = lowConfidence.mapTo(mutableSetOf()) { it.id },
                                onSelect = { rally ->
                                    selectedSuggestionId = null
                                    selectedId = rally.id
                                    seekTo(rally.keepStartMs)
                                },
                                onToggleIncluded = { rally, included ->
                                    updateRallyCuts(rally.cutIds) { it.copy(included = included) }
                                },
                                modifier = Modifier.guidedTourTarget("editor-cuts", guidedTourTargets),
                            )
                        },
                        centerPane = {
                            if (sourceAvailable == false || relinkMessage != null) {
                                SectionCard(
                                    if (sourceAvailable == false) "SOURCE UNAVAILABLE" else "SOURCE RE-LINKED",
                                    if (sourceAvailable == false) seed.displayName else "Playback and exports restored",
                                    compact = true,
                                ) {
                                    Text(
                                        relinkMessage ?: "Choose the original recording again to restore playback and MP4 export.",
                                        color = if (relinkFailed) Danger else Muted,
                                        fontSize = 11.sp,
                                    )
                                    if (sourceAvailable == false) {
                                        Button(
                                            enabled = !relinkingSource,
                                            onClick = { store.save(draft); onRelink() },
                                        ) { Text(if (relinkingSource) "Validating…" else "Re-link video") }
                                    }
                                }
                            }
                            Column {
                                Card(
                                    modifier = Modifier.guidedTourTarget("editor-video", guidedTourTargets),
                                    colors = CardDefaults.cardColors(containerColor = Color.Black),
                                ) {
                                    BoxWithConstraints(
                                        Modifier
                                            .fillMaxWidth()
                                            .height(desktopPlayerHeightDp.dp),
                                    ) {
                                        if (!BuildConfig.BLACK_VIDEO_PREVIEW) {
                                            ContentFrame(
                                                player = player,
                                                modifier = Modifier.fillMaxSize(),
                                                surfaceType = SURFACE_TYPE_TEXTURE_VIEW,
                                            )
                                        }
                                        if (draft.renderScoreOverlay && draft.scoreTracking.enabled) {
                                            val (displayWidth, displayHeight) = scoreOverlayDisplaySize(
                                                seed.width,
                                                seed.height,
                                                seed.rotation,
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
                                        Box(
                                            Modifier
                                                .fillMaxSize()
                                                .clickable(onClick = ::togglePlayback)
                                                .semantics {
                                                    contentDescription = if (isPlaying) "Pause video" else "Play video"
                                                },
                                        )
                                    }
                                }
                                Box(Modifier.fillMaxWidth().height(36.dp)) {
                                    PlayerControls(
                                        playing = isPlaying,
                                        positionMs = playbackPositionMs,
                                        durationMs = seed.gameEndMs,
                                        playbackRate = draft.playbackRate,
                                        desktop = true,
                                        onToggle = ::togglePlayback,
                                        onRate = { rate -> updateDraft { it.copy(playbackRate = rate) } },
                                        modifier = Modifier.guidedTourTarget("editor-transport", guidedTourTargets),
                                    )
                                    VideoResizeHandle(
                                        modifier = Modifier.align(Alignment.Center).width(96.dp).fillMaxHeight(),
                                    ) { dragDeltaPx ->
                                        desktopPlayerHeightDp = resizedDesktopPlayerHeight(
                                            currentHeightDp = desktopPlayerHeightDp,
                                            dragDeltaPx = dragDeltaPx,
                                            density = displayDensity,
                                        )
                                    }
                                }
                            }
                            Column(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .guidedTourTarget("editor-overview", guidedTourTargets),
                                verticalArrangement = Arrangement.spacedBy(4.dp),
                            ) {
                                Row(
                                    Modifier.fillMaxWidth(),
                                    verticalAlignment = Alignment.CenterVertically,
                                ) {
                                    Text(
                                        "GAME TIMELINE",
                                        color = Orange,
                                        fontSize = 11.sp,
                                        fontWeight = FontWeight.Black,
                                        letterSpacing = .7.sp,
                                    )
                                    Spacer(Modifier.weight(1f))
                                    Text(
                                        "Source time, cuts, and match events",
                                        color = Muted,
                                        fontSize = 12.sp,
                                        fontWeight = FontWeight.SemiBold,
                                        textAlign = TextAlign.End,
                                    )
                                }
                                WholeTimeline(
                                    windowStartMs = seed.gameStartMs,
                                    windowEndMs = (seed.gameStartMs + seed.gameEndMs) / 2,
                                    cuts = sortedCuts,
                                    joinedGaps = joinedGaps,
                                    ignored = visibleIgnoredIntervals,
                                    suggestions = activeSuggestions,
                                    appliedSuggestionIds = appliedSuggestionIds,
                                    userRemovedSuggestionIds = userRemovedSuggestionIds,
                                    userRemovedCleanupCutIds = userRemovedCleanupCutIds,
                                    selectedSuggestionId = selectedSuggestionId,
                                    selectedCutIds = selectedRally?.cutIds.orEmpty(),
                                    effectiveIds = effectiveIds,
                                    confidenceThreshold = draft.confidenceReviewThreshold,
                                    reviewedCutIds = draft.reviewedCutIds,
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
                                            activeSuggestions.firstOrNull { it.fragmentId() == suggestionId }
                                                ?.let(::selectSuggestion)
                                        } else {
                                            id?.let {
                                                selectedSuggestionId = null
                                                selectedId = editableRallies.firstOrNull { rally -> it in rally.cutIds }?.id ?: it
                                            }
                                            seekTo(time)
                                        }
                                    },
                                    heightScale = 0.8f,
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
                                    appliedSuggestionIds = appliedSuggestionIds,
                                    userRemovedSuggestionIds = userRemovedSuggestionIds,
                                    userRemovedCleanupCutIds = userRemovedCleanupCutIds,
                                    selectedSuggestionId = selectedSuggestionId,
                                    selectedCutIds = selectedRally?.cutIds.orEmpty(),
                                    effectiveIds = effectiveIds,
                                    confidenceThreshold = draft.confidenceReviewThreshold,
                                    reviewedCutIds = draft.reviewedCutIds,
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
                                            activeSuggestions.firstOrNull { it.fragmentId() == suggestionId }
                                                ?.let(::selectSuggestion)
                                        } else {
                                            id?.let {
                                                selectedSuggestionId = null
                                                selectedId = editableRallies.firstOrNull { rally -> it in rally.cutIds }?.id ?: it
                                            }
                                            seekTo(time)
                                        }
                                    },
                                    heightScale = 0.8f,
                                )
                                TimelineLabels(
                                    (seed.gameStartMs + seed.gameEndMs) / 2,
                                    (seed.gameStartMs + seed.gameEndMs * 3) / 4,
                                    seed.gameEndMs,
                                )
                                TimelineLegend()
                            }
                            if (message.isNotBlank()) {
                                Surface(color = Color(0xFFFFE2D8), shape = RoundedCornerShape(10.dp)) {
                                    Text(message, Modifier.padding(10.dp), fontSize = 12.sp)
                                }
                            }
                        },
                        rightPane = {
                            DesktopSidebarSection(
                                title = if (selectedIndex >= 0) {
                                    "R${(selectedIndex + 1).toString().padStart(3, '0')}"
                                } else "SELECTED RALLY",
                                modifier = Modifier.guidedTourTarget("editor-focus", guidedTourTargets),
                                headerActions = {
                                    selected?.let {
                                        Text(
                                            "${preciseTime(it.keepStartMs)}–${preciseTime(it.keepEndMs)}",
                                            color = Muted,
                                            fontSize = 9.sp,
                                        )
                                    }
                                    SmallButton("‹", enabled = selectedIndex > 0) {
                                        editableRallies.getOrNull(selectedIndex - 1)?.let {
                                            selectedId = it.id
                                            seekTo(it.keepStartMs)
                                        }
                                    }
                                    SmallButton("›", enabled = selectedIndex in 0..<editableRallies.lastIndex) {
                                        editableRallies.getOrNull(selectedIndex + 1)?.let {
                                            selectedId = it.id
                                            seekTo(it.keepStartMs)
                                        }
                                    }
                                },
                            ) {
                                if (selected == null) {
                                    Text("Add a missed rally at the current playhead to begin.", color = Muted, fontSize = 11.sp)
                                } else {
                                    val activeRally = requireNotNull(selectedRally)
                                    val needsReview = activeRally.cuts.any { cut ->
                                        cut.origin == CutOrigin.INFERRED &&
                                            (cut.isModelDisagreement() || cut.confidence < draft.confidenceReviewThreshold)
                                    }
                                    if (selectedClipSuggestion == null && needsReview) {
                                        ReviewNotice(
                                            title = "CHECK THIS RALLY",
                                            text = if (activeRally.cutIds.all { it in draft.reviewedCutIds }) {
                                                "Checked · your keep/remove choice is saved."
                                            } else {
                                                "VolleySplice is less certain about this rally."
                                            },
                                            warning = true,
                                        )
                                    }
                                    selectedClipSuggestion?.let { suggestion ->
                                        val decision = EditorMath.suggestionEffectiveDecision(draft, suggestion)
                                        KeepRemoveToggle(
                                            keepSelected = decision == SuppressionDecision.KEEP,
                                            removeSelected = decision == SuppressionDecision.SUPPRESS,
                                            automaticRemove = decision == SuppressionDecision.SUPPRESS &&
                                                draft.suppressionDecisionOverrides[suggestion.logicalId()] == null,
                                            onKeep = {
                                                updateDraft { current -> current.copy(
                                                    suppressionDecisionOverrides = current.suppressionDecisionOverrides +
                                                        (suggestion.logicalId() to SuppressionDecision.KEEP),
                                                    reviewedCutIds = current.reviewedCutIds + activeRally.cutIds,
                                                ) }
                                            },
                                            onRemove = {
                                                updateDraft { current -> current.copy(
                                                    suppressionDecisionOverrides = current.suppressionDecisionOverrides +
                                                        (suggestion.logicalId() to SuppressionDecision.SUPPRESS),
                                                    reviewedCutIds = current.reviewedCutIds + activeRally.cutIds,
                                                ) }
                                            },
                                        )
                                    } ?: KeepRemoveToggle(
                                        keepSelected = selected.included,
                                        removeSelected = !selected.included,
                                        automaticRemove = false,
                                        onKeep = {
                                            updateRallyCuts(activeRally.cutIds) { it.copy(included = true) }
                                            updateDraft { it.copy(reviewedCutIds = it.reviewedCutIds + activeRally.cutIds) }
                                        },
                                        onRemove = {
                                            updateRallyCuts(activeRally.cutIds) { it.copy(included = false) }
                                            updateDraft { it.copy(reviewedCutIds = it.reviewedCutIds + activeRally.cutIds) }
                                        },
                                    )
                                    TimelineLabels(
                                        detailWindow.startMs,
                                        (detailWindow.startMs + detailWindow.endMs) / 2,
                                        detailWindow.endMs,
                                    )
                                    Box(Modifier.guidedTourTarget("editor-trim", guidedTourTargets)) {
                                        FocusTimeline(
                                            cut = selected,
                                            window = detailWindow,
                                            playheadMs = playbackPositionMs,
                                            effective = activeRally.cutIds.any { it in effectiveIds },
                                            lowConfidence = needsReview &&
                                                activeRally.cutIds.any { it !in draft.reviewedCutIds },
                                            suggestions = activeSuggestions,
                                            appliedSuggestionIds = appliedSuggestionIds,
                                            userRemovedSuggestionIds = userRemovedSuggestionIds,
                                            userRemoved = activeRally.cutIds.any { it in userRemovedCleanupCutIds },
                                            handlesEnabled = selectedSuggestion == null,
                                            onSeek = ::seekTo,
                                            onSuggestionSelect = { id ->
                                                activeSuggestions.firstOrNull { it.fragmentId() == id }?.let(::selectSuggestion)
                                            },
                                            onStartChange = { setKeepBoundary("start", it) },
                                            onEndChange = { setKeepBoundary("end", it) },
                                            onCoreStartChange = { setCoreBoundary("start", it) },
                                            onCoreEndChange = { setCoreBoundary("end", it) },
                                        )
                                    }
                                    if (selectedSuggestion == null) {
                                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                            OutlinedButton(
                                                enabled = playbackPositionMs <= selected.coreEndMs - MIN_MARK_MS,
                                                onClick = { setCoreBoundary("start", playbackPositionMs) },
                                                modifier = Modifier.weight(1f),
                                            ) { Text("Set start", fontSize = 10.sp) }
                                            OutlinedButton(
                                                enabled = playbackPositionMs >= selected.coreStartMs + MIN_MARK_MS,
                                                onClick = { setCoreBoundary("end", playbackPositionMs) },
                                                modifier = Modifier.weight(1f),
                                            ) { Text("Set end", fontSize = 10.sp) }
                                        }
                                        Button(
                                            enabled = activeRally.cuts.any { cut ->
                                                playbackPositionMs >= cut.coreStartMs + MIN_MARK_MS &&
                                                    playbackPositionMs <= cut.coreEndMs - MIN_MARK_MS
                                            },
                                            onClick = ::splitSelectedAtPlayhead,
                                            modifier = Modifier.fillMaxWidth(),
                                        ) { Text("Split at playhead") }
                                    }
                                }
                            }
                            MarkingTools(
                                draft = draft,
                                playbackPositionMs = playbackPositionMs,
                                flat = true,
                                onDraft = ::updateDraft,
                                onManualCompleted = { cut ->
                                    selectedId = cut.id
                                    message = "Added ${cut.id} from ${preciseTime(cut.keepStartMs)} to ${preciseTime(cut.keepEndMs)}"
                                },
                                onMessage = { message = it },
                                modifier = Modifier.guidedTourTarget("editor-marking", guidedTourTargets),
                            )
                            if (visibleIgnoredIntervals.isNotEmpty()) {
                                DesktopSidebarSection("LEFT-OUT FOOTAGE") {
                                    visibleIgnoredIntervals.sortedBy { it.startMs }.forEach { interval ->
                                        Row(verticalAlignment = Alignment.CenterVertically) {
                                            TextButton(onClick = { seekTo(interval.startMs) }) {
                                                Text("${preciseTime(interval.startMs)}–${preciseTime(interval.endMs)}", fontSize = 10.sp)
                                            }
                                            Spacer(Modifier.weight(1f))
                                            TextButton(onClick = {
                                                updateDraft {
                                                    it.copy(ignoredIntervals = it.ignoredIntervals.filterNot { item -> item.id == interval.id })
                                                }
                                            }) { Text("Remove", color = Danger, fontSize = 10.sp) }
                                        }
                                    }
                                }
                            }
                            if (displayAnalysisMeasurements) AnalysisMeasurementsCard(project.analysisMeasurements)
                            DesktopSidebarSection(title = "PLAYBACK") {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    verticalAlignment = Alignment.CenterVertically,
                                ) {
                                    Text(
                                        "Play only final cut",
                                        modifier = Modifier.weight(1f),
                                        fontSize = 11.sp,
                                        fontWeight = FontWeight.SemiBold,
                                    )
                                    Switch(
                                        checked = draft.finalPreviewEnabled,
                                        onCheckedChange = { enabled ->
                                            previewEndMs = null
                                            updateDraft { it.copy(finalPreviewEnabled = enabled) }
                                            if (enabled) {
                                                EditorMath.nextFinalTime(finalIntervals, player.currentPosition)
                                                    ?.let(::seekTo)
                                            }
                                        },
                                    )
                                }
                            }
                            DesktopSidebarSection(
                                title = "VIDEO OVERLAY",
                                modifier = Modifier.guidedTourTarget(
                                    "editor-score-overlay",
                                    guidedTourTargets,
                                ),
                            ) {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    verticalAlignment = Alignment.CenterVertically,
                                ) {
                                    Text(
                                        "Render scores",
                                        modifier = Modifier.weight(1f),
                                        fontSize = 11.sp,
                                        fontWeight = FontWeight.SemiBold,
                                    )
                                    Switch(
                                        checked = draft.renderScoreOverlay,
                                        enabled = draft.scoreTracking.enabled,
                                        onCheckedChange = { enabled ->
                                            updateDraft {
                                                it.copy(
                                                    renderScoreOverlay = enabled,
                                                    renderScoreTimeline = if (enabled) {
                                                        it.renderScoreTimeline
                                                    } else {
                                                        false
                                                    },
                                                )
                                            }
                                        },
                                    )
                                }
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    verticalAlignment = Alignment.CenterVertically,
                                ) {
                                    Text(
                                        "Render point timeline",
                                        modifier = Modifier.weight(1f),
                                        fontSize = 11.sp,
                                        fontWeight = FontWeight.SemiBold,
                                    )
                                    Switch(
                                        checked = draft.renderScoreTimeline,
                                        enabled = draft.scoreTracking.enabled && draft.renderScoreOverlay,
                                        onCheckedChange = { enabled ->
                                            updateDraft {
                                                it.copy(renderScoreTimeline = it.renderScoreOverlay && enabled)
                                            }
                                        },
                                    )
                                }
                                if (!draft.scoreTracking.enabled) {
                                    Text(
                                        "Turn on Optional scoreboard to enable video overlays.",
                                        color = Muted,
                                        fontSize = 9.sp,
                                    )
                                }
                            }
                            LegalFooter()
                        },
                    )
                    }
                } else {
            Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Box(Modifier.guidedTourTarget("editor-header", guidedTourTargets)) {
                    sourceControls(
                        editorSummary,
                        { showEditorSettings = true },
                    )
                }

            EditorWorkflow()
            CurrentSourceCard(seed.displayName, seed.ranges.size)

            if (sourceAvailable == false || relinkMessage != null) {
                SectionCard(
                    if (sourceAvailable == false) "SOURCE UNAVAILABLE" else "SOURCE RE-LINKED",
                    if (sourceAvailable == false) seed.displayName else "Playback and exports restored",
                ) {
                    Text(
                        relinkMessage
                            ?: "VolleySplice cannot open the saved video location. Choose the original recording again to restore playback and MP4 export.",
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
                "YOUR FINAL VIDEO",
                "${compactTime(totalFinalMs)} · ${effectiveRallies.size} clips included",
                modifier = Modifier.guidedTourTarget("editor-export", guidedTourTargets),
            ) {
                Text(
                    if (reviewAttentionCount == 0) "Everything that needs attention has been checked."
                    else "$reviewAttentionCount ${if (reviewAttentionCount == 1) "item needs" else "items need"} attention · $reviewAttentionBreakdown",
                    color = if (reviewAttentionCount == 0) Green else Ink,
                    fontWeight = FontWeight.SemiBold,
                )
                Text(
                    "Use the Settings gear to fine-tune automatic cleanup, extra time around clips, short breaks, and how many clips are flagged.",
                    color = Muted,
                    fontSize = 12.sp,
                )
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(Modifier.weight(1f)) {
                        Text("Play only the final video", fontWeight = FontWeight.SemiBold)
                        Text("Skip every part that will not be saved", fontSize = 12.sp, color = Muted)
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
                        modifier = Modifier.guidedTourTarget(
                            "editor-score-overlay",
                            guidedTourTargets,
                        ),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text("Add scores to the final video", fontWeight = FontWeight.SemiBold)
                            Text("Preview the scoreboard and include it when you save", fontSize = 12.sp, color = Muted)
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
                            modifier = Modifier.padding(start = 18.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Column(Modifier.weight(1f)) {
                                Text("Show the point history", fontWeight = FontWeight.SemiBold)
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
                HorizontalDivider(color = Rail)
                if (exportState.status == "running") {
                    LinearProgressIndicator(
                        progress = { exportState.progress / 100f },
                        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                    )
                    Text(exportState.detail, color = Muted, fontSize = 11.sp)
                } else if (exportState.status == "queued") {
                    Text(exportState.detail, color = Green, fontSize = 11.sp)
                } else if (exportState.detail.isNotBlank()) {
                    Text(
                        exportState.detail,
                        color = if (exportState.status == "failed") Danger else Green,
                        fontSize = 11.sp,
                    )
                }
                Button(
                    enabled = finalIntervals.isNotEmpty() && !exportPending,
                    onClick = {
                        pendingExportIntervals = finalIntervals
                        exportLauncher.launch(exportFilename(seed.displayName))
                    },
                    modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp),
                ) {
                    Text(if (exportPending) "Creating your video…" else "Save final video", fontWeight = FontWeight.Bold)
                }
                OutlinedButton(
                    enabled = finalIntervals.isNotEmpty(),
                    onClick = {
                        youtubeChaptersStatus = null
                        showYouTubeChapters = true
                    },
                    modifier = Modifier.fillMaxWidth().testTag("youtube-chapters-button"),
                ) { Text("YouTube chapters") }
                if (exportPending) {
                    TextButton(
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
                    onClick = { showProjectOptions = true },
                    modifier = Modifier.align(Alignment.CenterHorizontally),
                ) { Text("Project options", color = Muted) }
                Text(
                    "The finished video and optional project files stay on this device.",
                    color = Muted,
                    fontSize = 11.sp,
                )
            }

            ScoreTrackingPanel(
                    enabled = draft.scoreTracking.enabled,
                    tracking = draft.scoreTracking,
                    visibleTracking = preparedScoreOverlay.tracking,
                    visibleScore = visibleScore,
                    selectedMarkerId = selectedScoreMarkerId,
                    currentTimestampMs = playbackPositionMs,
                    scoreEffectiveTimestampMs = scoreEffectiveTimestampMs,
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
                                message = "Preparing score markers"
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
                )

            Column {
                Card(
                    modifier = Modifier.guidedTourTarget("editor-video", guidedTourTargets),
                    colors = CardDefaults.cardColors(containerColor = Color.Black),
                ) {
                    BoxWithConstraints(Modifier.fillMaxWidth().height(compactPlayerHeightDp.dp)) {
                        if (!BuildConfig.BLACK_VIDEO_PREVIEW) {
                            ContentFrame(
                                player = player,
                                modifier = Modifier.fillMaxSize(),
                                surfaceType = SURFACE_TYPE_TEXTURE_VIEW,
                            )
                        }
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
                        Box(
                            Modifier
                                .fillMaxSize()
                                .clickable(onClick = ::togglePlayback)
                                .semantics {
                                    contentDescription = if (isPlaying) "Pause video" else "Play video"
                                },
                        )
                    }
                }
                VideoResizeHandle(description = "Resize phone video player height") { dragDeltaPx ->
                    compactPlayerHeightDp = resizedPlayerHeight(
                        currentHeightDp = compactPlayerHeightDp,
                        dragDeltaPx = dragDeltaPx,
                        density = displayDensity,
                        minHeightDp = COMPACT_PLAYER_MIN_HEIGHT_DP,
                        maxHeightDp = COMPACT_PLAYER_MAX_HEIGHT_DP,
                    )
                }
            }
            PlayerControls(
                playing = isPlaying,
                positionMs = playbackPositionMs,
                durationMs = seed.gameEndMs,
                playbackRate = draft.playbackRate,
                onToggle = ::togglePlayback,
                onRate = { rate -> updateDraft { it.copy(playbackRate = rate) } },
                modifier = Modifier.guidedTourTarget("editor-transport", guidedTourTargets),
            )
            ReviewQueueControls(
                cleanupCount = pendingCleanupReview.size,
                clipCount = pendingReview.size,
                serveCount = reviewableServeReviews.size,
                scoreTrackingEnabled = draft.scoreTracking.enabled,
                onReviewCleanup = ::reviewNextCleanup,
                onReviewClips = ::reviewNextClip,
                onReviewServes = ::reviewNextServe,
                modifier = Modifier.guidedTourTarget("editor-review-queues", guidedTourTargets),
            )

            SectionCard(
                "GAME TIMELINE",
                "Select a clip to check or adjust it",
                compact = true,
                modifier = Modifier.guidedTourTarget("editor-overview", guidedTourTargets),
            ) {
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
                    userRemovedSuggestionIds = userRemovedSuggestionIds,
                    userRemovedCleanupCutIds = userRemovedCleanupCutIds,
                    selectedSuggestionId = selectedSuggestionId,
                    selectedCutIds = selectedRally?.cutIds.orEmpty(),
                    effectiveIds = effectiveIds,
                    confidenceThreshold = draft.confidenceReviewThreshold,
                    reviewedCutIds = draft.reviewedCutIds,
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
                                selectedId = editableRallies.firstOrNull { rally -> it in rally.cutIds }?.id ?: it
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
                    userRemovedSuggestionIds = userRemovedSuggestionIds,
                    userRemovedCleanupCutIds = userRemovedCleanupCutIds,
                    selectedSuggestionId = selectedSuggestionId,
                    selectedCutIds = selectedRally?.cutIds.orEmpty(),
                    effectiveIds = effectiveIds,
                    confidenceThreshold = draft.confidenceReviewThreshold,
                    reviewedCutIds = draft.reviewedCutIds,
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
                                selectedId = editableRallies.firstOrNull { rally -> it in rally.cutIds }?.id ?: it
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
                TimelineLegend()
            }

            SectionCard(
                "SELECTED RALLY",
                null,
                compact = true,
                modifier = Modifier.guidedTourTarget("editor-focus", guidedTourTargets),
                headerActions = {
                    SmallButton("Previous", enabled = selectedIndex > 0) {
                        editableRallies.getOrNull(selectedIndex - 1)?.let {
                            selectedId = it.id
                            seekTo(it.keepStartMs)
                        }
                    }
                    Text(
                        if (selected == null) "0 / ${editableRallies.size}"
                        else "${selectedIndex + 1} / ${editableRallies.size}",
                        Modifier.padding(horizontal = 4.dp),
                        color = Muted,
                        fontFamily = FontFamily.Monospace,
                        fontSize = 11.sp,
                    )
                    SmallButton("Next", enabled = selectedIndex in 0..<editableRallies.lastIndex) {
                        editableRallies.getOrNull(selectedIndex + 1)?.let {
                            selectedId = it.id
                            seekTo(it.keepStartMs)
                        }
                    }
                },
            ) {
                if (selected == null) {
                    Text("Add a missed rally at the current playhead to begin.", color = Muted)
                } else {
                    val selectedRally = requireNotNull(selectedRally)
                    val selectedNeedsConfidenceReview = selectedRally.cuts.any { cut ->
                        cut.origin == CutOrigin.INFERRED &&
                            (cut.isModelDisagreement() || cut.confidence < draft.confidenceReviewThreshold)
                    }
                    if (selectedClipSuggestion == null && selectedNeedsConfidenceReview) {
                        ReviewNotice(
                            title = "LOW-CONFIDENCE CLIP",
                            text = if (selectedRally.cutIds.all { it in draft.reviewedCutIds }) {
                                "Checked · your keep/remove choice is saved."
                            } else {
                                "VolleySplice is less certain about this rally. Check it, then choose Keep or Remove."
                            },
                            warning = true,
                        )
                    }
                    selectedClipSuggestion?.let { suggestion ->
                        val cleanupDecision = EditorMath.suggestionEffectiveDecision(draft, suggestion)
                        val explicitDecision = draft.suppressionDecisionOverrides[suggestion.logicalId()]
                        ReviewNotice(
                            title = "AUTOMATIC CLEANUP",
                            text = when {
                                cleanupDecision == SuppressionDecision.KEEP ->
                                    "Kept · ${preciseTime(suggestion.startMs())}–${preciseTime(suggestion.endMs())} stays in the video."
                                explicitDecision == SuppressionDecision.SUPPRESS ->
                                    "Removed by you · this cleanup section will be left out."
                                else ->
                                    "Suggested removal · VolleySplice will leave out this likely non-play section unless you keep it."
                            },
                            removal = cleanupDecision == SuppressionDecision.SUPPRESS,
                        )
                        KeepRemoveToggle(
                            keepSelected = cleanupDecision == SuppressionDecision.KEEP,
                            removeSelected = cleanupDecision == SuppressionDecision.SUPPRESS,
                            automaticRemove = cleanupDecision == SuppressionDecision.SUPPRESS && explicitDecision == null,
                            onKeep = {
                                updateDraft { current -> current.copy(
                                    suppressionDecisionOverrides = current.suppressionDecisionOverrides +
                                        (suggestion.logicalId() to SuppressionDecision.KEEP),
                                    reviewedCutIds = current.reviewedCutIds + selectedRally.cutIds,
                                ) }
                            },
                            onRemove = {
                                updateDraft { current -> current.copy(
                                    suppressionDecisionOverrides = current.suppressionDecisionOverrides +
                                        (suggestion.logicalId() to SuppressionDecision.SUPPRESS),
                                    reviewedCutIds = current.reviewedCutIds + selectedRally.cutIds,
                                ) }
                            },
                        )
                    } ?: KeepRemoveToggle(
                        keepSelected = selected.included,
                        removeSelected = !selected.included,
                        automaticRemove = false,
                        onKeep = {
                            updateDraft { current -> current.copy(
                                cuts = current.cuts.map { cut ->
                                    if (cut.id in selectedRally.cutIds) cut.copy(included = true) else cut
                                },
                                reviewedCutIds = current.reviewedCutIds + selectedRally.cutIds,
                            ) }
                        },
                        onRemove = {
                            updateDraft { current -> current.copy(
                                cuts = current.cuts.map { cut ->
                                    if (cut.id in selectedRally.cutIds) cut.copy(included = false) else cut
                                },
                                reviewedCutIds = current.reviewedCutIds + selectedRally.cutIds,
                            ) }
                        },
                    )
                    TimelineLabels(detailWindow.startMs, (detailWindow.startMs + detailWindow.endMs) / 2, detailWindow.endMs)
                    Box(Modifier.guidedTourTarget("editor-trim", guidedTourTargets)) {
                        FocusTimeline(
                            cut = selected,
                            window = detailWindow,
                            playheadMs = playbackPositionMs,
                            effective = selectedRally.cutIds.any { it in effectiveIds },
                            lowConfidence = selectedNeedsConfidenceReview &&
                                selectedRally.cutIds.any { it !in draft.reviewedCutIds },
                            suggestions = activeSuggestions,
                            appliedSuggestionIds = activeSuggestions.filter {
                                EditorMath.suggestionEffectiveDecision(draft, it) == SuppressionDecision.SUPPRESS
                            }.map { it.fragmentId() }.toSet(),
                            userRemovedSuggestionIds = userRemovedSuggestionIds,
                            userRemoved = selectedRally.cutIds.any { it in userRemovedCleanupCutIds },
                            handlesEnabled = selectedSuggestion == null,
                            onSeek = ::seekTo,
                            onSuggestionSelect = { id ->
                                activeSuggestions.firstOrNull { it.fragmentId() == id }
                                    ?.let(::selectSuggestion)
                            },
                            onStartChange = { setKeepBoundary("start", it) },
                            onEndChange = { setKeepBoundary("end", it) },
                            onCoreStartChange = { setCoreBoundary("start", it) },
                            onCoreEndChange = { setCoreBoundary("end", it) },
                        )
                    }
                    if (selectedSuggestion == null) {
                    Text(
                        if (selected.origin == CutOrigin.INFERRED) {
                            "Orange outer handles adjust padding · green inner handles adjust rally length."
                        } else {
                            "Drag the green handles to adjust this rally's exact start and end."
                        },
                        fontSize = 12.sp,
                        color = Muted,
                    )
                    Row(
                        Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        OutlinedButton(
                            enabled = playbackPositionMs <= selected.coreEndMs - MIN_MARK_MS,
                            onClick = { setCoreBoundary("start", playbackPositionMs) },
                            modifier = Modifier.weight(1f),
                        ) { Text("Set rally start here") }
                        OutlinedButton(
                            enabled = playbackPositionMs >= selected.coreStartMs + MIN_MARK_MS,
                            onClick = { setCoreBoundary("end", playbackPositionMs) },
                            modifier = Modifier.weight(1f),
                        ) { Text("Set rally end here") }
                    }
                    Button(
                        enabled = selectedRally.cuts.any { cut ->
                            playbackPositionMs >= cut.coreStartMs + MIN_MARK_MS &&
                                playbackPositionMs <= cut.coreEndMs - MIN_MARK_MS
                        },
                        onClick = ::splitSelectedAtPlayhead,
                        modifier = Modifier.fillMaxWidth(),
                    ) { Text("Split at playhead") }
                    Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedButton(onClick = {
                            updateDraft { it.copy(finalPreviewEnabled = false) }
                            seekTo(selected.keepStartMs)
                            previewEndMs = selected.keepEndMs
                            player.play()
                        }) { Text("Preview rally") }
                        if (selected.origin == CutOrigin.INFERRED) {
                            OutlinedButton(onClick = {
                                updateRallyCuts(selectedRally.cutIds) { cut -> cut.copy(
                                    keepStartMs = (cut.coreStartMs - draft.beforePaddingMs)
                                        .coerceAtLeast(seed.gameStartMs),
                                    keepEndMs = (cut.coreEndMs + draft.afterPaddingMs)
                                        .coerceAtMost(seed.gameEndMs),
                                ) }
                            }) { Text("Reset extra time") }
                        }
                        if (selected.origin == CutOrigin.MANUAL) {
                            OutlinedButton(onClick = {
                                val remaining = editableRallies.filterNot { it.id == selectedRally.id }
                                updateDraft { it.copy(
                                    cuts = it.cuts.filterNot { cut -> cut.id in selectedRally.cutIds },
                                    reviewedCutIds = it.reviewedCutIds - selectedRally.cutIds,
                                ) }
                                selectedId = remaining.getOrNull((selectedIndex - 1).coerceAtLeast(0))?.id.orEmpty()
                            }) { Text("Delete", color = Danger) }
                        }
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
                SectionCard("LEFT-OUT FOOTAGE", null) {
                    visibleIgnoredIntervals.sortedBy { it.startMs }.forEach { interval ->
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            TextButton(onClick = { seekTo(interval.startMs) }) {
                                Text("${preciseTime(interval.startMs)}–${preciseTime(interval.endMs)}")
                            }
                            Spacer(Modifier.weight(1f))
                            TextButton(onClick = {
                                updateDraft { it.copy(ignoredIntervals = it.ignoredIntervals.filterNot { item -> item.id == interval.id }) }
                            }) { Text("Remove", color = Danger) }
                        }
                    }
                }
            }

            val clipListState = rememberLazyListState()
            LaunchedEffect(selectedRally?.id, editableRallies.map { it.id }) {
                val activeIndex = editableRallies.indexOfFirst { it.id == selectedRally?.id }
                if (activeIndex >= 0) clipListState.animateScrollToItem(activeIndex)
            }
            SectionCard(
                "ALL RALLIES",
                "${effectiveRallies.size} included · $leftOutRallyCount left out",
                modifier = Modifier.guidedTourTarget("editor-cuts", guidedTourTargets),
            ) {
                LazyColumn(
                    Modifier
                        .fillMaxWidth()
                        .heightIn(max = 320.dp)
                        .border(1.dp, Rail, RoundedCornerShape(4.dp))
                        .background(Paper, RoundedCornerShape(4.dp))
                        .padding(4.dp)
                        .semantics { contentDescription = "All rallies list" },
                    state = clipListState,
                ) {
                    itemsIndexed(editableRallies, key = { _, rally -> rally.id }) { index, rally ->
                        val cut = rally.asEditableCut()
                        val state = when {
                            !cut.included -> "Left out"
                            rally.cutIds.none { it in effectiveIds } -> "Left out"
                            else -> "Included"
                        }
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(8.dp))
                                .background(if (rally.id == selectedRally?.id) Color(0xFFFFEEE8) else Color.Transparent)
                                .clickable { selectedId = rally.id; seekTo(rally.keepStartMs) }
                                .padding(vertical = 7.dp, horizontal = 8.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Text((index + 1).toString().padStart(2, '0'), color = Muted, fontFamily = FontFamily.Monospace)
                            Column(Modifier.weight(1f).padding(horizontal = 10.dp)) {
                                Text(
                                    if (rally.cuts.size == 1) cut.id
                                    else "${rally.cuts.first().id} + ${rally.cuts.size - 1} joined",
                                    fontWeight = FontWeight.SemiBold,
                                )
                                Text("${preciseTime(cut.keepStartMs)}–${preciseTime(cut.keepEndMs)}", fontSize = 12.sp, color = Muted)
                            }
                            Text(
                                if (cut.origin == CutOrigin.MANUAL) "ADDED"
                                else if (rally.cutIds.all { it in draft.reviewedCutIds }) "CHECKED"
                                else if (rally in lowConfidence) "CHECK"
                                else "READY",
                                fontSize = 12.sp,
                                color = if (rally.cuts.any { it.isModelDisagreement() ||
                                    it.confidence < draft.confidenceReviewThreshold }
                                ) Warning else if (rally.cutIds.all { it in draft.reviewedCutIds }) Green else Muted,
                            )
                            TextButton(onClick = {
                                updateRallyCuts(rally.cutIds) { it.copy(included = state != "Included") }
                            }) {
                                Text(state, color = if (state == "Included") Green else Danger)
                            }
                        }
                        if (index < editableRallies.lastIndex) HorizontalDivider(color = Rail)
                    }
                }
            }

            if (displayAnalysisMeasurements) AnalysisMeasurementsCard(project.analysisMeasurements)

            if (BuildConfig.DEBUG) {
                TextButton(onClick = { context.startActivity(Intent(context, MainActivity::class.java)) }) {
                    Text("Benchmark tools")
                }
            }
            LegalFooter()
            Spacer(Modifier.height(20.dp))
            }
                }
            }
        }
        GuidedTour(
            GuidedTourStage.EDITOR,
            guidedTourTargets,
            scoreTrackingEnabled = draft.scoreTracking.enabled,
            restartSignal = guidedTourRestartSignal,
        )
        if (showYouTubeChapters) {
            YouTubeChaptersDialog(
                sourceFilename = seed.displayName,
                intervals = finalIntervals,
                cuts = draft.cuts.filter {
                    it.included && (it.origin == CutOrigin.MANUAL || it.id in effectiveIds)
                },
                scoreTracking = preparedScoreOverlay.tracking.takeIf { draft.scoreTracking.enabled },
                hasSideSwitches = draft.scoreTracking.enabled &&
                    (project.sideSwitchEnabled || preparedScoreOverlay.tracking.sideSwitchMarkers.isNotEmpty()),
                status = youtubeChaptersStatus,
                onClearStatus = { youtubeChaptersStatus = null },
                onCopy = { text ->
                    val clipboard = context.getSystemService(ClipboardManager::class.java)
                    clipboard.setPrimaryClip(ClipData.newPlainText("YouTube chapters", text))
                    youtubeChaptersStatus = "Copied. Paste these chapters into your YouTube description."
                },
                onSaveTextFile = { text, filename ->
                    pendingYouTubeChaptersText = text
                    youtubeChaptersStatus = null
                    youtubeChaptersLauncher.launch(filename)
                },
                onDismiss = {
                    showYouTubeChapters = false
                    youtubeChaptersStatus = null
                },
            )
        }
    }
}

@Composable
private fun LargeScreenEditorLayout(
    leftPaneWidthDp: Float,
    rightPaneWidthDp: Float,
    onLeftPaneWidthChange: (Float) -> Unit,
    onRightPaneWidthChange: (Float) -> Unit,
    header: @Composable () -> Unit,
    leftPane: @Composable ColumnScope.() -> Unit,
    centerPane: @Composable ColumnScope.() -> Unit,
    rightPane: @Composable ColumnScope.() -> Unit,
) {
    val density = LocalDensity.current.density
    Column(
        modifier = Modifier.fillMaxSize().background(Paper),
    ) {
        Box(
            Modifier.fillMaxWidth().padding(start = 10.dp, top = 4.dp, end = 10.dp, bottom = 2.dp),
        ) { header() }
        BoxWithConstraints(Modifier.fillMaxWidth().weight(1f)) {
            val sidePaneBudgetDp = (
                maxWidth.value - DESKTOP_CENTER_PANE_MIN_WIDTH_DP - DESKTOP_PANE_HORIZONTAL_CHROME_DP
            ).coerceAtLeast(DESKTOP_LEFT_PANE_MIN_WIDTH_DP + DESKTOP_RIGHT_PANE_MIN_WIDTH_DP)
            val resolvedLeftWidthDp = leftPaneWidthDp.coerceIn(
                DESKTOP_LEFT_PANE_MIN_WIDTH_DP,
                minOf(
                    DESKTOP_LEFT_PANE_MAX_WIDTH_DP,
                    sidePaneBudgetDp - DESKTOP_RIGHT_PANE_MIN_WIDTH_DP,
                ),
            )
            val resolvedRightWidthDp = rightPaneWidthDp.coerceIn(
                DESKTOP_RIGHT_PANE_MIN_WIDTH_DP,
                minOf(
                    DESKTOP_RIGHT_PANE_MAX_WIDTH_DP,
                    sidePaneBudgetDp - resolvedLeftWidthDp,
                ),
            )
            val leftMaxWidthDp = minOf(
                DESKTOP_LEFT_PANE_MAX_WIDTH_DP,
                sidePaneBudgetDp - resolvedRightWidthDp,
            )
            val rightMaxWidthDp = minOf(
                DESKTOP_RIGHT_PANE_MAX_WIDTH_DP,
                sidePaneBudgetDp - resolvedLeftWidthDp,
            )

            Row(
                modifier = Modifier.fillMaxSize(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Column(
                    modifier = Modifier
                        .width(resolvedLeftWidthDp.dp)
                        .fillMaxHeight()
                        .verticalScroll(rememberScrollState())
                        .padding(start = 10.dp, bottom = 16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    DesktopPaneHeading("MATCH EVENTS")
                    leftPane()
                }
                DesktopSidebarResizeHandle("Resize match events sidebar") { dragDeltaPx ->
                    onLeftPaneWidthChange(
                        resizedDesktopSidebarWidth(
                            currentWidthDp = resolvedLeftWidthDp,
                            dragDeltaPx = dragDeltaPx,
                            density = density,
                            minWidthDp = DESKTOP_LEFT_PANE_MIN_WIDTH_DP,
                            maxWidthDp = leftMaxWidthDp,
                        ),
                    )
                }
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxHeight()
                        .verticalScroll(rememberScrollState())
                        .padding(bottom = 16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    centerPane()
                }
                DesktopSidebarResizeHandle("Resize current rally sidebar") { dragDeltaPx ->
                    onRightPaneWidthChange(
                        resizedDesktopSidebarWidth(
                            currentWidthDp = resolvedRightWidthDp,
                            dragDeltaPx = -dragDeltaPx,
                            density = density,
                            minWidthDp = DESKTOP_RIGHT_PANE_MIN_WIDTH_DP,
                            maxWidthDp = rightMaxWidthDp,
                        ),
                    )
                }
                Column(
                    modifier = Modifier
                        .width(resolvedRightWidthDp.dp)
                        .fillMaxHeight()
                        .verticalScroll(rememberScrollState())
                        .padding(end = 10.dp, bottom = 16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    DesktopPaneHeading("CURRENT RALLY")
                    rightPane()
                }
            }
        }
    }
}

@Composable
private fun DesktopSidebarResizeHandle(
    description: String,
    onDrag: (Float) -> Unit,
) {
    val dragState = rememberDraggableState(onDelta = onDrag)
    Box(
        modifier = Modifier
            .width(16.dp)
            .fillMaxHeight()
            .draggable(
                state = dragState,
                orientation = Orientation.Horizontal,
            )
            .semantics { contentDescription = description },
        contentAlignment = Alignment.Center,
    ) {
        Box(Modifier.width(1.dp).fillMaxHeight().background(Rail))
        Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
            repeat(3) {
                Box(
                    Modifier
                        .width(6.dp)
                        .height(2.dp)
                        .clip(RoundedCornerShape(1.dp))
                        .background(Muted.copy(alpha = 0.7f)),
                )
            }
        }
    }
}

@Composable
private fun VideoResizeHandle(
    modifier: Modifier = Modifier.fillMaxWidth().height(18.dp),
    description: String = "Resize video player height",
    onDrag: (Float) -> Unit,
) {
    val dragState = rememberDraggableState(onDelta = onDrag)
    Box(
        modifier = modifier
            .draggable(
                state = dragState,
                orientation = Orientation.Vertical,
            )
            .semantics { contentDescription = description },
    ) {
        Box(
            modifier = Modifier
                .align(Alignment.Center)
                .width(54.dp)
                .height(4.dp)
                .clip(RoundedCornerShape(2.dp))
                .background(Muted.copy(alpha = 0.55f)),
        )
    }
}

@Composable
private fun LargeScreenExportLayout(
    header: @Composable () -> Unit,
    mp4Option: @Composable ColumnScope.() -> Unit,
    chaptersOption: @Composable ColumnScope.() -> Unit,
    projectOption: @Composable ColumnScope.() -> Unit,
) {
    Column(
        modifier = Modifier.fillMaxSize().background(Paper),
    ) {
        Box(
            Modifier.fillMaxWidth().padding(start = 10.dp, top = 4.dp, end = 10.dp, bottom = 2.dp),
        ) { header() }
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 24.dp, vertical = 18.dp),
            verticalArrangement = Arrangement.spacedBy(18.dp),
        ) {
            DesktopPaneHeading("EXPORT")
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(16.dp),
                verticalAlignment = Alignment.Top,
            ) {
                Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(8.dp)) { mp4Option() }
                Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(8.dp)) { chaptersOption() }
                Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(8.dp)) { projectOption() }
            }
        }
    }
}

@Composable
private fun DesktopPaneHeading(label: String) {
    Text(
        text = label,
        modifier = Modifier.fillMaxWidth().padding(horizontal = 2.dp, vertical = 2.dp),
        color = Green,
        fontSize = 10.sp,
        fontWeight = FontWeight.Black,
        letterSpacing = 1.sp,
    )
}

@Composable
private fun DesktopSidebarSection(
    title: String,
    modifier: Modifier = Modifier,
    headerActions: @Composable RowScope.() -> Unit = {},
    content: @Composable ColumnScope.() -> Unit,
) {
    Column(modifier = modifier.fillMaxWidth()) {
        HorizontalDivider(color = Rail)
        Column(
            Modifier.fillMaxWidth().padding(horizontal = 2.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    title,
                    modifier = Modifier.weight(1f),
                    color = Orange,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Black,
                    letterSpacing = .7.sp,
                )
                headerActions()
            }
            content()
        }
    }
}

@Composable
private fun DesktopRallyLedger(
    rallies: List<EditableRallyGroup>,
    selectedId: String?,
    effectiveIds: Set<String>,
    reviewedIds: Set<String>,
    lowConfidenceIds: Set<String>,
    onSelect: (EditableRallyGroup) -> Unit,
    onToggleIncluded: (EditableRallyGroup, Boolean) -> Unit,
    modifier: Modifier = Modifier,
) {
    DesktopSidebarSection(
        title = "CLIP REGISTER",
        modifier = modifier,
        headerActions = {
            Text(
                "${rallies.count { rally -> rally.cutIds.any { it in effectiveIds } }} / ${rallies.size} included",
                color = Muted,
                fontSize = 9.sp,
            )
        },
    ) {
        LazyColumn(
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 180.dp, max = 440.dp)
                .semantics { contentDescription = "Clip register" },
        ) {
            itemsIndexed(rallies, key = { _, rally -> rally.id }) { index, rally ->
                val included = rally.cutIds.any { it in effectiveIds }
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(6.dp))
                        .background(if (rally.id == selectedId) Acid.copy(alpha = .5f) else Color.Transparent)
                        .clickable { onSelect(rally) }
                        .padding(horizontal = 7.dp, vertical = 7.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        "R${(index + 1).toString().padStart(3, '0')}",
                        color = if (rally.id == selectedId) Ink else Muted,
                        fontFamily = FontFamily.Monospace,
                        fontSize = 10.sp,
                        fontWeight = FontWeight.Bold,
                    )
                    Column(Modifier.weight(1f).padding(horizontal = 7.dp)) {
                        Text(
                            "${preciseTime(rally.keepStartMs)}–${preciseTime(rally.keepEndMs)}",
                            fontFamily = FontFamily.Monospace,
                            fontSize = 10.sp,
                            maxLines = 1,
                        )
                        Text(
                            when {
                                !included -> "Left out"
                                rally.cutIds.all { it in reviewedIds } -> "Checked"
                                rally.id in lowConfidenceIds -> "Needs review"
                                else -> "Ready"
                            },
                            color = if (rally.id in lowConfidenceIds && rally.cutIds.any { it !in reviewedIds }) Warning else Muted,
                            fontSize = 9.sp,
                        )
                    }
                    TextButton(onClick = { onToggleIncluded(rally, !included) }) {
                        Text(if (included) "On" else "Off", color = if (included) Green else Danger, fontSize = 10.sp)
                    }
                }
                if (index < rallies.lastIndex) HorizontalDivider(color = Rail)
            }
        }
    }
}

@Composable
internal fun ScoreTrackingPanel(
    enabled: Boolean,
    tracking: ScoreTracking,
    visibleTracking: ScoreTracking,
    visibleScore: DerivedScore,
    selectedMarkerId: String?,
    currentTimestampMs: Long,
    scoreEffectiveTimestampMs: Long,
    servingSideStatus: ServingSideAnalysisStatus,
    servingSideError: String?,
    servingSideProgress: Float?,
    servingSideProgressDetail: String?,
    servingSideStepMeasurements: List<InferenceStepMeasurement> = emptyList(),
    sideSwitchEnabled: Boolean = true,
    desktop: Boolean = false,
    onEnabledChange: (Boolean) -> Unit,
    onTracking: (ScoreTracking) -> Unit,
    onSelect: (String, Long) -> Unit,
    modifier: Modifier = Modifier,
    toggleModifier: Modifier = Modifier,
) {
    val sortedServes = visibleTracking.serveMarkers.sortedWith(
        compareBy<ServeMarker> { it.timestampMs }.thenBy { it.id },
    )
    val reviewMarkers = sortedServes.filter { it.side == ServingSide.REVIEW }
    val reviewMarkerIds = reviewMarkers.mapTo(mutableSetOf()) { it.id }
    val selected = sortedServes.lastOrNull { it.timestampMs <= scoreEffectiveTimestampMs }
        ?: sortedServes.firstOrNull()
    val selectedIndex = selected?.let {
        sortedServes.indexOfFirst { marker -> marker.id == it.id }
    } ?: -1
    val panelContent: @Composable ColumnScope.() -> Unit = {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    "SCOREBOARD · BETA",
                    Modifier.weight(1f),
                    color = Orange,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Black,
                    letterSpacing = .8.sp,
                )
                Switch(
                    checked = enabled,
                    onCheckedChange = onEnabledChange,
                    modifier = toggleModifier,
                )
            }
            if (enabled) {
                val reviewCount = reviewMarkers.size

                ScoreMarkerList(
                    tracking = tracking,
                    visibleTracking = visibleTracking,
                    currentTimestampMs = scoreEffectiveTimestampMs,
                    reviewMarkerIds = reviewMarkerIds,
                    onSelect = onSelect,
                    onTracking = onTracking,
                    flat = desktop,
                )
                ScorePointTimeline(
                    visibleScore = visibleScore,
                    tracking = tracking,
                    onPointClick = { point -> onSelect(point.serveMarkerId, point.timestampMs) },
                    flat = desktop,
                )

                SidebarContentContainer(flat = desktop) {
                    Column(
                        Modifier.fillMaxWidth().padding(vertical = if (desktop) 2.dp else 10.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        if (desktop) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                            ) {
                                DesktopTeamScoreEditor(
                                    name = tracking.team1Name,
                                    label = "Team 1",
                                    score = visibleScore.team1Score,
                                    scoreColor = Green,
                                    serving = visibleScore.servingTeamId == ScoreTeamId.TEAM_1,
                                    onNameChange = { name ->
                                        onTracking(tracking.copy(team1Name = name.ifEmpty { "Team 1" }))
                                    },
                                    modifier = Modifier.weight(1f),
                                )
                                DesktopTeamScoreEditor(
                                    name = tracking.team2Name,
                                    label = "Team 2",
                                    score = visibleScore.team2Score,
                                    scoreColor = Orange,
                                    serving = visibleScore.servingTeamId == ScoreTeamId.TEAM_2,
                                    onNameChange = { name ->
                                        onTracking(tracking.copy(team2Name = name.ifEmpty { "Team 2" }))
                                    },
                                    modifier = Modifier.weight(1f),
                                )
                            }
                        } else {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            OutlinedTextField(
                                value = tracking.team1Name,
                                onValueChange = { name ->
                                    onTracking(tracking.copy(team1Name = name.ifEmpty { "Team 1" }))
                                },
                                label = { Text("Team 1") },
                                leadingIcon = {
                                    Text(if (visibleScore.servingTeamId == ScoreTeamId.TEAM_1) "🏐" else " ")
                                },
                                singleLine = true,
                                modifier = Modifier.weight(1f),
                            )
                            Text(
                                visibleScore.team1Score.toString(),
                                Modifier.width(38.dp),
                                color = Green,
                                fontFamily = FontFamily.Monospace,
                                fontSize = 27.sp,
                                fontWeight = FontWeight.Bold,
                                textAlign = TextAlign.Center,
                            )
                            Text(
                                visibleScore.team2Score.toString(),
                                Modifier.width(38.dp),
                                color = Orange,
                                fontFamily = FontFamily.Monospace,
                                fontSize = 27.sp,
                                fontWeight = FontWeight.Bold,
                                textAlign = TextAlign.Center,
                            )
                            OutlinedTextField(
                                value = tracking.team2Name,
                                onValueChange = { name ->
                                    onTracking(tracking.copy(team2Name = name.ifEmpty { "Team 2" }))
                                },
                                label = { Text("Team 2") },
                                trailingIcon = {
                                    Text(if (visibleScore.servingTeamId == ScoreTeamId.TEAM_2) "🏐" else " ")
                                },
                                singleLine = true,
                                modifier = Modifier.weight(1f),
                            )
                        }
                        }
                    }
                }

                val servingSideBusy = servingSideStatus == ServingSideAnalysisStatus.QUEUED ||
                    servingSideStatus == ServingSideAnalysisStatus.ANALYZING
                val servingSideFailed = servingSideStatus == ServingSideAnalysisStatus.ERROR
                val reviewPending = reviewCount > 0
                val reportedProgress = servingSideProgress?.coerceIn(0f, 1f) ?: 0f
                if (servingSideBusy || servingSideFailed) {
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
                                    ServingSideAnalysisStatus.ANALYZING -> "FINDING SCORE MARKERS"
                                    ServingSideAnalysisStatus.READY -> if (sideSwitchEnabled) {
                                        "SERVE + SWITCH MARKERS READY"
                                    } else "SERVE MARKERS READY"
                                    ServingSideAnalysisStatus.ERROR -> "SCORE MARKERS NEED ATTENTION"
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
                                    servingSideFailed -> servingSideError ?: "Score markers could not be prepared. Turn score tracking off and on to try again."
                                    reviewCount > 0 ->
                                        "$reviewCount serve ${if (reviewCount == 1) "marker needs" else "markers need"} a quick check."
                                    servingSideStatus == ServingSideAnalysisStatus.READY ->
                                        if (sideSwitchEnabled) {
                                            "Serve and team side-switch markers are ready to check."
                                        } else "Serve markers are ready to check. Team side switches are off."
                                    else -> if (sideSwitchEnabled) {
                                        "Turn on score tracking to find serves and team side switches."
                                    } else "Turn on score tracking to find serve markers."
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
                    }
                }
                }

                SidebarContentContainer(flat = desktop, containerColor = Paper) {
                    Column(
                        Modifier.padding(vertical = if (desktop) 2.dp else 10.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
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
                            if (selected.modelSide != null && selected.side != selected.modelSide) {
                                Text("Corrected by you", color = Muted, fontSize = 11.sp)
                            }
                            Row(
                                Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(4.dp),
                                verticalAlignment = Alignment.CenterVertically,
                            ) {
                                listOf(ServingSide.NEAR to "Near", ServingSide.FAR to "Far").forEach { (side, label) ->
                                    FilterChip(
                                        selected = selected.side == side,
                                        onClick = {
                                            onTracking(tracking.copy(serveMarkers = tracking.serveMarkers.map {
                                                if (it.id == selected.id) it.copy(side = side) else it
                                            }))
                                        },
                                        label = { Text(label, fontSize = 10.sp) },
                                        modifier = Modifier.weight(1f),
                                    )
                                }
                                Row(
                                    Modifier
                                        .weight(1.5f)
                                        .clickable(enabled = selectedIndex > 0) {
                                            val ignored = !selected.ignorePreviousPoint
                                            onTracking(tracking.copy(serveMarkers = tracking.serveMarkers.map {
                                                if (it.id == selected.id) it.copy(ignorePreviousPoint = ignored) else it
                                            }))
                                        },
                                    verticalAlignment = Alignment.CenterVertically,
                                ) {
                                    Checkbox(
                                        checked = selected.ignorePreviousPoint,
                                        enabled = selectedIndex > 0,
                                        onCheckedChange = { ignored ->
                                            onTracking(tracking.copy(serveMarkers = tracking.serveMarkers.map {
                                                if (it.id == selected.id) it.copy(ignorePreviousPoint = ignored) else it
                                            }))
                                        },
                                    )
                                    Text("Ignore / replay previous point", fontSize = 9.sp, lineHeight = 11.sp)
                                }
                            }
                        }
                    }
                }

                Row(
                    Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    OutlinedButton(
                        onClick = {
                            val updated = ScoreReducer.addServe(tracking, currentTimestampMs, ServingSide.REVIEW)
                            onTracking(updated)
                            updated.serveMarkers.firstOrNull { marker ->
                                tracking.serveMarkers.none { it.id == marker.id }
                            }?.let { onSelect(it.id, it.timestampMs) }
                        },
                        modifier = Modifier.weight(1f),
                    ) { Text("+ Add serve", fontSize = 11.sp) }
                    Button(
                        onClick = {
                            val updated = ScoreReducer.addSideSwitch(tracking, currentTimestampMs)
                            onTracking(updated)
                            updated.sideSwitchMarkers.firstOrNull { marker ->
                                tracking.sideSwitchMarkers.none { it.id == marker.id }
                            }?.let { onSelect(it.id, it.timestampMs) }
                        },
                        modifier = Modifier.weight(1f),
                    ) { Text("⇄ Add team switch", fontSize = 11.sp) }
                }
                Text(
                    "At ${preciseTime(currentTimestampMs)}",
                    color = Muted,
                    fontFamily = FontFamily.Monospace,
                    fontSize = 10.sp,
                )
            }
    }
    if (desktop) {
        Column(
            modifier = modifier
                .fillMaxWidth()
                .testTag("score-tracking-panel")
                .semantics { contentDescription = "Score tracking controls" },
        ) {
            HorizontalDivider(color = Rail)
            Column(
                Modifier.fillMaxWidth().padding(horizontal = 2.dp, vertical = 10.dp),
                verticalArrangement = Arrangement.spacedBy(9.dp),
            ) {
                panelContent()
            }
        }
    } else {
        Card(
            modifier = modifier
                .border(1.dp, Rail, RoundedCornerShape(10.dp))
                .testTag("score-tracking-panel")
                .semantics { contentDescription = "Score tracking controls" },
            shape = RoundedCornerShape(10.dp),
            colors = CardDefaults.cardColors(containerColor = Paper),
            elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
        ) {
            Column(
                Modifier.fillMaxWidth().padding(14.dp),
                verticalArrangement = Arrangement.spacedBy(11.dp),
            ) {
                panelContent()
            }
        }
    }
}

@Composable
private fun DesktopTeamScoreEditor(
    name: String,
    label: String,
    score: Int,
    scoreColor: Color,
    serving: Boolean,
    onNameChange: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(modifier = modifier, verticalArrangement = Arrangement.spacedBy(6.dp)) {
        OutlinedTextField(
            value = name,
            onValueChange = onNameChange,
            label = { Text(label, maxLines = 1, fontSize = 10.sp) },
            singleLine = true,
            modifier = Modifier.fillMaxWidth(),
        )
        Surface(
            modifier = Modifier
                .fillMaxWidth()
                .height(62.dp)
                .border(1.dp, Rail, RoundedCornerShape(7.dp)),
            color = Paper,
            shape = RoundedCornerShape(7.dp),
        ) {
            Box(Modifier.fillMaxSize()) {
                Text(
                    score.toString(),
                    modifier = Modifier.align(Alignment.Center),
                    color = scoreColor,
                    fontFamily = FontFamily.Monospace,
                    fontSize = 29.sp,
                    fontWeight = FontWeight.Bold,
                )
                if (serving) {
                    Text("🏐", Modifier.align(Alignment.TopEnd).padding(5.dp), fontSize = 11.sp)
                }
            }
        }
    }
}

@Composable
private fun ScoreMarkerList(
    tracking: ScoreTracking,
    visibleTracking: ScoreTracking,
    currentTimestampMs: Long,
    reviewMarkerIds: Set<String>,
    onSelect: (String, Long) -> Unit,
    onTracking: (ScoreTracking) -> Unit,
    flat: Boolean = false,
) {
    val sortedServes = visibleTracking.serveMarkers.sortedWith(
        compareBy<ServeMarker> { it.timestampMs }.thenBy { it.id },
    )
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
    val timelineMarkerId = markerRows.lastOrNull { it.second <= currentTimestampMs }?.first
        ?: markerRows.firstOrNull()?.first
    val markerListState = rememberLazyListState()
    LaunchedEffect(timelineMarkerId, markerRows.map { it.first }) {
        val selectedRowIndex = markerRows.indexOfFirst { it.first == timelineMarkerId }
        if (selectedRowIndex >= 0) markerListState.animateScrollToItem(selectedRowIndex)
    }
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text("MARKERS", color = Orange, fontSize = 11.sp, fontWeight = FontWeight.Bold)
        Spacer(Modifier.weight(1f))
        Text("${markerRows.size} · tap to select", color = Muted, fontSize = 10.sp)
    }
    val listModifier = if (flat) {
        Modifier
            .fillMaxWidth()
            .heightIn(max = 220.dp)
            .semantics { contentDescription = "Score marker list" }
    } else {
        Modifier
            .fillMaxWidth()
            .heightIn(max = 220.dp)
            .border(1.dp, Rail, RoundedCornerShape(4.dp))
            .background(Paper, RoundedCornerShape(4.dp))
            .padding(4.dp)
            .semantics { contentDescription = "Score marker list" }
    }
    LazyColumn(
        listModifier,
        state = markerListState,
        verticalArrangement = Arrangement.spacedBy(if (flat) 0.dp else 4.dp),
    ) {
        if (markerRows.isEmpty()) {
            item {
                Text("No score markers have been added.", Modifier.padding(10.dp), color = Muted, fontSize = 11.sp)
            }
        }
        items(markerRows, key = { it.first }) { (id, timestamp, label) ->
            val selectedRow = id == timelineMarkerId
            val needsReview = id in reviewMarkerIds
            val rowModifier = if (flat) {
                Modifier
                    .fillMaxWidth()
                    .background(
                        when {
                            selectedRow -> Acid.copy(alpha = .35f)
                            needsReview -> Color(0xFFFFF7E6)
                            else -> Color.Transparent
                        },
                    )
                    .semantics {
                        contentDescription = if (needsReview) "Score marker needs review" else "Score marker"
                    }
            } else {
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
                        if (needsReview) Color(0xFFFFF7E6) else Paper,
                        RoundedCornerShape(3.dp),
                    )
                    .semantics {
                        contentDescription = if (needsReview) "Score marker needs review" else "Score marker"
                    }
            }
            Column {
            Row(
                rowModifier,
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
            if (flat) HorizontalDivider(color = Rail)
            }
        }
    }
}

@Composable
private fun SidebarContentContainer(
    flat: Boolean,
    containerColor: Color = Paper,
    content: @Composable () -> Unit,
) {
    if (flat) {
        Box(Modifier.fillMaxWidth()) { content() }
    } else {
        Surface(
            modifier = Modifier.fillMaxWidth().border(1.dp, Rail, RoundedCornerShape(4.dp)),
            color = containerColor,
            shape = RoundedCornerShape(4.dp),
        ) {
            Box(Modifier.fillMaxWidth().padding(horizontal = 10.dp)) {
                content()
            }
        }
    }
}

@Composable
private fun ScorePointTimeline(
    visibleScore: DerivedScore,
    tracking: ScoreTracking,
    onPointClick: (DerivedScorePoint) -> Unit,
    flat: Boolean = false,
) {
    val awardedPoints = visibleScore.points.mapIndexedNotNull { index, point ->
        point.takeIf { it.status == ScorePointStatus.COUNTED && it.winnerTeamId != null }
            ?.let { index + 1 to it }
    }
    SidebarContentContainer(flat = flat) {
        Column(
            Modifier.padding(vertical = if (flat) 2.dp else 10.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
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
                        awardedPoints.map { it.second },
                        Green,
                        scoreForPoint = { point ->
                            point.team1ScoreAfter.toString().takeIf { point.winnerTeamId == ScoreTeamId.TEAM_1 }
                        },
                        onPointClick = onPointClick,
                    )
                    ScoreTimelinePointRow(
                        tracking.team2Name,
                        awardedPoints.map { it.second },
                        Orange,
                        scoreForPoint = { point ->
                            point.team2ScoreAfter.toString().takeIf { point.winnerTeamId == ScoreTeamId.TEAM_2 }
                        },
                        onPointClick = onPointClick,
                    )
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
private fun ScoreTimelinePointRow(
    label: String,
    points: List<DerivedScorePoint>,
    accent: Color,
    scoreForPoint: (DerivedScorePoint) -> String?,
    onPointClick: (DerivedScorePoint) -> Unit,
) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(
            label,
            Modifier.width(72.dp),
            color = Ink,
            fontSize = 10.sp,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
        points.forEach { point ->
            val value = scoreForPoint(point)
            Box(Modifier.width(32.dp).height(28.dp), contentAlignment = Alignment.Center) {
                if (value == null) {
                    HorizontalDivider(Modifier.width(26.dp), color = Rail)
                } else {
                    Box(
                        Modifier
                            .size(24.dp)
                            .background(accent.copy(alpha = .14f), RoundedCornerShape(50))
                            .border(2.dp, accent, RoundedCornerShape(50))
                            .semantics {
                                contentDescription = "Seek to point at ${preciseTime(point.timestampMs)}"
                            }
                            .clickable { onPointClick(point) },
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
        // Store screenshots intentionally hide the source footage. Give the
        // remaining score UI more visual weight so it survives Play resizing.
        val overlayLayoutHeightPx = if (BuildConfig.BLACK_VIDEO_PREVIEW) {
            videoHeightPx * 2
        } else videoHeightPx
        val layout = ScoreOverlay.layout(videoWidthPx, overlayLayoutHeightPx, snapshot) { text, fontSize ->
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
                overlayLayoutHeightPx,
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
    subtitle: String?,
    modifier: Modifier = Modifier,
    compact: Boolean = false,
    headerActions: (@Composable RowScope.() -> Unit)? = null,
    content: @Composable ColumnScope.() -> Unit,
) {
    Card(
        modifier = modifier.border(1.dp, Rail, RoundedCornerShape(10.dp)),
        shape = RoundedCornerShape(10.dp),
        colors = CardDefaults.cardColors(containerColor = Paper),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
    ) {
        Column(
            Modifier.fillMaxWidth().padding(if (compact) 10.dp else 14.dp),
            verticalArrangement = Arrangement.spacedBy(if (compact) 4.dp else 10.dp),
        ) {
            if (compact) {
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        label,
                        color = Orange,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Black,
                        letterSpacing = .7.sp,
                    )
                    Spacer(Modifier.weight(1f))
                    if (!subtitle.isNullOrBlank()) {
                        Text(
                            subtitle,
                            color = Muted,
                            fontSize = 12.sp,
                            fontWeight = FontWeight.SemiBold,
                            textAlign = TextAlign.End,
                        )
                        Spacer(Modifier.width(8.dp))
                    }
                    headerActions?.invoke(this)
                }
            } else {
                Text(label, color = Orange, fontSize = 12.sp, fontWeight = FontWeight.Black, letterSpacing = 1.sp)
                if (!subtitle.isNullOrBlank()) Text(subtitle, fontWeight = FontWeight.SemiBold)
                headerActions?.let { actions ->
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) { actions() }
                }
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
    desktop: Boolean = false,
    onToggle: () -> Unit,
    onRate: (Float) -> Unit,
    modifier: Modifier = Modifier,
) {
    var rateExpanded by remember { mutableStateOf(false) }
    if (desktop) {
        Row(
            modifier = modifier.fillMaxWidth().heightIn(min = 36.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                preciseTime(positionMs),
                fontFamily = FontFamily.Monospace,
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold,
            )
            Text(
                " / ${preciseTime(durationMs)}",
                fontFamily = FontFamily.Monospace,
                fontSize = 12.sp,
                color = Muted,
            )
            Spacer(Modifier.weight(1f))
            Box(Modifier.width(104.dp)) {
                OutlinedButton(
                    onClick = { rateExpanded = true },
                    modifier = Modifier.fillMaxWidth().height(34.dp),
                    contentPadding = PaddingValues(horizontal = 10.dp, vertical = 0.dp),
                ) { Text("Speed · ${playbackRate.toInt()}x", fontSize = 10.sp, maxLines = 1) }
                DropdownMenu(
                    expanded = rateExpanded,
                    onDismissRequest = { rateExpanded = false },
                ) {
                    listOf(1f, 2f, 4f, 8f).forEach { rate ->
                        DropdownMenuItem(
                            text = { Text("${rate.toInt()}x") },
                            onClick = {
                                rateExpanded = false
                                onRate(rate)
                            },
                        )
                    }
                }
            }
            Button(
                onClick = onToggle,
                modifier = Modifier.width(84.dp).height(34.dp),
                contentPadding = PaddingValues(horizontal = 10.dp, vertical = 0.dp),
            ) {
                Text(if (playing) "Pause" else "Play", fontSize = 10.sp, fontWeight = FontWeight.Bold)
            }
        }
    } else {
        Column(modifier = modifier, verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text(preciseTime(positionMs), fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
                Text(" / ${preciseTime(durationMs)}", fontFamily = FontFamily.Monospace, color = Muted)
            }
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Box(Modifier.weight(1f)) {
                    OutlinedButton(
                        onClick = { rateExpanded = true },
                        modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp),
                    ) { Text("Speed · ${playbackRate.toInt()}x") }
                    DropdownMenu(
                        expanded = rateExpanded,
                        onDismissRequest = { rateExpanded = false },
                    ) {
                        listOf(1f, 2f, 4f, 8f).forEach { rate ->
                            DropdownMenuItem(
                                text = { Text("${rate.toInt()}x") },
                                onClick = {
                                    rateExpanded = false
                                    onRate(rate)
                                },
                            )
                        }
                    }
                }
                Button(
                    onClick = onToggle,
                    modifier = Modifier.weight(1f).heightIn(min = 48.dp),
                ) { Text(if (playing) "Pause" else "Play", fontWeight = FontWeight.Bold) }
            }
        }
    }
}

@Composable
private fun ReviewQueueControls(
    cleanupCount: Int,
    clipCount: Int,
    serveCount: Int,
    scoreTrackingEnabled: Boolean,
    onReviewCleanup: () -> Unit,
    onReviewClips: () -> Unit,
    onReviewServes: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Row(
        modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        ReviewQueueButton("Review cleanup", cleanupCount, onReviewCleanup, Modifier.weight(1f))
        ReviewQueueButton("Review clips", clipCount, onReviewClips, Modifier.weight(1f))
        ReviewQueueButton(
            "Review serves",
            serveCount,
            onReviewServes,
            Modifier.weight(1f),
            enabled = scoreTrackingEnabled,
        )
    }
}

@Composable
private fun ReviewQueueButton(
    label: String,
    count: Int,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
) {
    Button(
        onClick = onClick,
        enabled = enabled && count > 0,
        modifier = modifier.heightIn(min = 44.dp),
        shape = RoundedCornerShape(8.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = Acid,
            contentColor = Ink,
            disabledContainerColor = SoftPanel,
            disabledContentColor = Muted,
        ),
    ) {
        Text("$label · $count", fontSize = 10.sp, maxLines = 1)
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
private fun TimelineLegend() {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .horizontalScroll(rememberScrollState())
            .padding(top = 4.dp, bottom = 2.dp),
        horizontalArrangement = Arrangement.spacedBy(14.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        TimelineLegendItem("Rally core") {
            TimelineLegendBlock(Green)
        }
        TimelineLegendItem("Padding") {
            TimelineLegendBlock(TimelineKeptPaddingColor)
        }
        TimelineLegendItem("Needs review") {
            TimelineLegendBlock(Orange)
        }
        TimelineLegendItem("Serve") {
            Text("🏐", fontSize = 13.sp, maxLines = 1)
        }
        TimelineLegendItem("Side switch") {
            Text("⇄", color = Ink, fontSize = 13.sp, fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
        }
        TimelineLegendItem("Excluded") {
            Canvas(Modifier.width(20.dp).height(10.dp)) {
                var x = -size.height
                while (x < size.width) {
                    drawLine(
                        color = SuppressionRed,
                        start = Offset(x, size.height),
                        end = Offset(x + size.height, 0f),
                        strokeWidth = 2.dp.toPx(),
                    )
                    x += 6.dp.toPx()
                }
            }
        }
        TimelineLegendItem("Joined gap") {
            Canvas(Modifier.width(20.dp).height(10.dp)) {
                val barWidth = 2.dp.toPx()
                val gap = 3.dp.toPx()
                var x = 1.dp.toPx()
                while (x < size.width) {
                    drawRoundRect(
                        color = TimelineJoinedGapColor,
                        topLeft = Offset(x, 0f),
                        size = Size(barWidth, size.height),
                        cornerRadius = CornerRadius(barWidth / 2f),
                    )
                    x += barWidth + gap
                }
            }
        }
    }
}

@Composable
private fun TimelineLegendItem(
    label: String,
    marker: @Composable () -> Unit,
) {
    Row(
        horizontalArrangement = Arrangement.spacedBy(6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        marker()
        Text(label, color = Muted, fontSize = 10.sp, fontWeight = FontWeight.SemiBold, maxLines = 1)
    }
}

@Composable
private fun TimelineLegendBlock(color: Color) {
    Box(
        Modifier
            .width(20.dp)
            .height(9.dp)
            .clip(RoundedCornerShape(3.dp))
            .background(color),
    )
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
    userRemovedSuggestionIds: Set<String>,
    userRemovedCleanupCutIds: Set<String>,
    selectedSuggestionId: String?,
    selectedCutIds: Set<String>,
    effectiveIds: Set<String>,
    confidenceThreshold: Float,
    reviewedCutIds: Set<String>,
    playheadMs: Long,
    serveMarkers: List<ServeMarker>,
    sideSwitchMarkers: List<SideSwitchMarker>,
    selectedScoreMarkerId: String?,
    onMarkerSelect: (String, Long) -> Unit,
    onSeek: (Long, String?, String?) -> Unit,
    heightScale: Float = 1f,
) {
    val resolvedHeightScale = heightScale.coerceIn(0.5f, 1f)
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
                .height(68.dp * resolvedHeightScale)
                .clip(RoundedCornerShape(9.dp))
                .background(TimelineTrackColor)
                .semantics {
                    contentDescription = if (suggestions.isEmpty()) "Game timeline"
                    else "Game timeline with ${suggestions.size} cleanup suggestions: " +
                        suggestions.joinToString { suggestion ->
                            "${preciseTime(suggestion.startMs())} to ${preciseTime(suggestion.endMs())}, " +
                                when (suggestion.fragmentId()) {
                                    in userRemovedSuggestionIds -> "removed by you"
                                    in appliedSuggestionIds -> "left out"
                                    else -> "included"
                                }
                        }
                }
                .pointerInput(windowStartMs, windowEndMs, cuts, suggestions, serveMarkers, sideSwitchMarkers) {
                    detectTapGestures { offset ->
                        val time = timeAt(offset.x)
                        val markerHitRadius = with(density) { 14.dp.toPx() }
                        val markerIconHeight = with(density) { (20.dp * resolvedHeightScale).toPx() }
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
            val cutTop = (16.dp * resolvedHeightScale).toPx()
            val cutHeight = (36.dp * resolvedHeightScale).toPx()
            val suppressionTop = (8.dp * resolvedHeightScale).toPx()
            val suppressionHeight = (52.dp * resolvedHeightScale).toPx()
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
            cuts.forEach { cut ->
                val clippedStart = max(cut.keepStartMs, windowStartMs)
                val clippedEnd = min(cut.keepEndMs, windowEndMs)
                if (clippedEnd <= clippedStart) return@forEach
                val x = xAt(clippedStart)
                val width = max(2f, xAt(clippedEnd) - x)
                val low = cut.origin == CutOrigin.INFERRED &&
                    (cut.isModelDisagreement() || cut.confidence < confidenceThreshold)
                val needsReview = timelineNeedsReview(low, cut.id, reviewedCutIds)
                val color = when {
                    !cut.included || cut.id in userRemovedCleanupCutIds -> Muted.copy(alpha = .55f)
                    cut.id !in effectiveIds -> Danger.copy(alpha = .65f)
                    needsReview -> Warning
                    else -> TimelineKeptPaddingColor
                }
                drawRoundRect(color, Offset(x, cutTop), Size(width, cutHeight), CornerRadius(6f))
                val clippedCoreStart = max(cut.coreStartMs, windowStartMs)
                val clippedCoreEnd = min(cut.coreEndMs, windowEndMs)
                if (clippedCoreEnd > clippedCoreStart) {
                    val coreX = xAt(clippedCoreStart)
                    val coreWidth = max(1f, xAt(clippedCoreEnd) - coreX)
                    drawRoundRect(
                        when {
                            !cut.included || cut.id in userRemovedCleanupCutIds -> Muted
                            needsReview -> Orange
                            else -> Green
                        },
                        Offset(coreX, cutTop),
                        Size(coreWidth, cutHeight),
                        CornerRadius(4f),
                    )
                }
            }
            // Joined gaps are retained output, so keep their connector above the
            // individual clip blocks just like the production web timeline.
            joinedGaps.forEach { gap ->
                val clippedStart = max(gap.startMs, windowStartMs)
                val clippedEnd = min(gap.endMs, windowEndMs)
                if (clippedEnd <= clippedStart) return@forEach
                val x = xAt(clippedStart)
                val width = max(1f, xAt(clippedEnd) - x)
                drawRoundRect(
                    TimelineJoinedGapColor,
                    Offset(x, cutTop),
                    Size(width, cutHeight),
                    CornerRadius(4f),
                )
            }
            val selectedCuts = cuts.filter { it.id in selectedCutIds }
            if (selectedCuts.isNotEmpty()) {
                val clippedStart = max(selectedCuts.minOf { it.keepStartMs }, windowStartMs)
                val clippedEnd = min(selectedCuts.maxOf { it.keepEndMs }, windowEndMs)
                if (clippedEnd > clippedStart) {
                    val x = xAt(clippedStart)
                    drawRoundRect(
                        Orange,
                        Offset(x, cutTop - (3.dp * resolvedHeightScale).toPx()),
                        Size(
                            max(2f, xAt(clippedEnd) - x),
                            cutHeight + (6.dp * resolvedHeightScale).toPx(),
                        ),
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
                val removedByUser = suggestion.fragmentId() in userRemovedSuggestionIds
                val suggestionColor = if (removedByUser) Muted else SuppressionRed
                if (applied) {
                    drawRect(
                        suggestionColor.copy(alpha = .62f),
                        Offset(x, suppressionTop),
                        Size(width, suppressionHeight),
                    )
                } else {
                    drawRect(
                        suggestionColor.copy(alpha = .2f),
                        Offset(x, suppressionTop),
                        Size(width, suppressionHeight),
                    )
                    drawRect(
                        suggestionColor,
                        Offset(x, suppressionTop),
                        Size(width, suppressionHeight),
                        style = Stroke(2.dp.toPx()),
                    )
                }
                clipRect(x, suppressionTop, x + width, suppressionTop + suppressionHeight) {
                    var hatch = x - suppressionHeight
                    while (hatch < x + width) {
                        drawLine(
                            suggestionColor.copy(alpha = if (applied) .95f else .55f),
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
                        Offset(x, suppressionTop + (2.dp * resolvedHeightScale).toPx()),
                        Size(width, suppressionHeight - (4.dp * resolvedHeightScale).toPx()),
                        style = Stroke(1.5.dp.toPx()),
                    )
                }
            }
            if (playheadMs in windowStartMs..windowEndMs) {
                val playheadX = xAt(playheadMs)
                drawLine(Orange, Offset(playheadX, 0f), Offset(playheadX, size.height), 4f, StrokeCap.Round)
            }
            val markerCenterY = (10.dp * resolvedHeightScale).toPx()
            val markerRadius = (8.dp * resolvedHeightScale).toPx()
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
                    if (needsReview) Color(0xFFFFD84D) else Color.White,
                    markerRadius,
                    Offset(x, markerCenterY),
                )
                drawCircle(
                    if (needsReview) Color(0xFF8A6500) else Ink,
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
private fun ReviewNotice(
    title: String,
    text: String,
    warning: Boolean = false,
    removal: Boolean = false,
) {
    val borderColor = when {
        removal -> Color(0xFFD69B8D)
        warning -> Warning
        else -> Rail
    }
    val backgroundColor = when {
        removal -> Color(0xFFFFF1F2)
        warning -> Color(0xFFFFF7E6)
        else -> SoftPanel
    }
    Surface(
        color = backgroundColor,
        shape = RoundedCornerShape(8.dp),
        modifier = Modifier.fillMaxWidth().border(1.dp, borderColor, RoundedCornerShape(8.dp)),
    ) {
        Column(Modifier.padding(10.dp), verticalArrangement = Arrangement.spacedBy(3.dp)) {
            Text(
                title,
                color = if (removal) SuppressionRed else Orange,
                fontSize = 10.sp,
                fontWeight = FontWeight.Black,
                letterSpacing = .7.sp,
            )
            Text(text, color = if (removal) Color(0xFF651019) else Muted, fontSize = 11.sp)
        }
    }
}

@Composable
private fun KeepRemoveToggle(
    keepSelected: Boolean,
    removeSelected: Boolean,
    automaticRemove: Boolean,
    onKeep: () -> Unit,
    onRemove: () -> Unit,
) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        OutlinedButton(
            onClick = onKeep,
            modifier = Modifier.weight(1f).heightIn(min = 44.dp),
            border = BorderStroke(1.dp, if (keepSelected) Green else Rail),
            colors = ButtonDefaults.outlinedButtonColors(
                containerColor = if (keepSelected) Color(0xFFE6F4DF) else Color.White,
                contentColor = if (keepSelected) Green else Ink,
            ),
        ) { Text(if (keepSelected) "✓ Keep" else "Keep", fontWeight = FontWeight.Bold) }
        OutlinedButton(
            onClick = onRemove,
            modifier = Modifier.weight(1f).heightIn(min = 44.dp),
            border = BorderStroke(1.dp, if (removeSelected) SuppressionRed else Rail),
            colors = ButtonDefaults.outlinedButtonColors(
                containerColor = when {
                    !removeSelected -> Color.White
                    automaticRemove -> Color(0xFFFFE5E7)
                    else -> Color(0xFFBD1826)
                },
                contentColor = if (removeSelected && !automaticRemove) Color.White else SuppressionRed,
            ),
        ) {
            Text(
                when {
                    removeSelected && automaticRemove -> "Remove · auto"
                    removeSelected -> "✓ Remove"
                    else -> "Remove"
                },
                fontWeight = FontWeight.Bold,
            )
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
    userRemovedSuggestionIds: Set<String>,
    userRemoved: Boolean,
    handlesEnabled: Boolean,
    onSeek: (Long) -> Unit,
    onSuggestionSelect: (String) -> Unit,
    onStartChange: (Long) -> Unit,
    onEndChange: (Long) -> Unit,
    onCoreStartChange: (Long) -> Unit,
    onCoreEndChange: (Long) -> Unit,
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
                .background(TimelineTrackColor)
                .semantics {
                    contentDescription = "Focused timeline. " + suggestions.joinToString {
                        "Cleanup suggestion ${preciseTime(it.startMs())} to ${preciseTime(it.endMs())}, " +
                            when (it.fragmentId()) {
                                in userRemovedSuggestionIds -> "removed by you"
                                in appliedSuggestionIds -> "left out"
                                else -> "included"
                            }
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
                !cut.included || userRemoved -> Muted.copy(alpha = .6f)
                !effective -> Danger.copy(alpha = .6f)
                lowConfidence -> Warning.copy(alpha = .8f)
                else -> TimelineKeptPaddingColor
            }
            drawRoundRect(
                keptColor,
                Offset(xAt(cut.keepStartMs), 18f),
                Size(max(2f, xAt(cut.keepEndMs) - xAt(cut.keepStartMs)), size.height - 36f),
                CornerRadius(8f),
            )
            drawRoundRect(
                when {
                    !cut.included || userRemoved -> Muted
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
                val removedByUser = suggestion.fragmentId() in userRemovedSuggestionIds
                val suggestionColor = if (removedByUser) Muted else SuppressionRed
                drawRect(
                    suggestionColor.copy(alpha = if (applied) .62f else .18f),
                    Offset(x, 0f), Size(width, size.height),
                )
                drawRect(
                    suggestionColor, Offset(x, 1f), Size(width, size.height - 2f),
                    style = Stroke(if (applied) 1f else 3f),
                )
                clipRect(x, 0f, x + width, size.height) {
                    var hatch = x - size.height
                    while (hatch < x + width) {
                        drawLine(suggestionColor.copy(alpha = .75f), Offset(hatch, size.height), Offset(hatch + size.height, 0f), 2f)
                        hatch += 10f
                    }
                }
            }
            val playheadX = xAt(playheadMs).coerceIn(0f, size.width)
            drawLine(Orange, Offset(playheadX, 0f), Offset(playheadX, size.height), 4f)
        }
        if (handlesEnabled) {
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
        val coreHandleWidth = 12.dp
        val coreHandlePx = with(density) { coreHandleWidth.toPx() }
        Box(
            Modifier
                .align(Alignment.BottomStart)
                .offset { IntOffset((xAt(cut.coreStartMs) - coreHandlePx / 2).roundToInt(), 0) }
                .width(coreHandleWidth)
                .height(27.dp)
                .semantics { contentDescription = "Adjust rally start at ${preciseTime(cut.coreStartMs)}" }
                .draggable(
                    orientation = Orientation.Horizontal,
                    state = rememberDraggableState { delta ->
                        onCoreStartChange(cut.coreStartMs + deltaToMs(delta))
                    },
                )
                .background(Color.White.copy(alpha = .75f), RoundedCornerShape(6.dp))
                .border(3.dp, Green, RoundedCornerShape(6.dp)),
        )
        Box(
            Modifier
                .align(Alignment.BottomStart)
                .offset { IntOffset((xAt(cut.coreEndMs) - coreHandlePx / 2).roundToInt(), 0) }
                .width(coreHandleWidth)
                .height(27.dp)
                .semantics { contentDescription = "Adjust rally end at ${preciseTime(cut.coreEndMs)}" }
                .draggable(
                    orientation = Orientation.Horizontal,
                    state = rememberDraggableState { delta ->
                        onCoreEndChange(cut.coreEndMs + deltaToMs(delta))
                    },
                )
                .background(Color.White.copy(alpha = .75f), RoundedCornerShape(6.dp))
                .border(3.dp, Green, RoundedCornerShape(6.dp)),
        )
        }
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
private fun MarkingTools(
    draft: EditorDraft,
    playbackPositionMs: Long,
    flat: Boolean = false,
    onDraft: ((EditorDraft) -> EditorDraft) -> Unit,
    onManualCompleted: (EditableCut) -> Unit,
    onMessage: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    val toolsContent: @Composable ColumnScope.() -> Unit = {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Button(
                    onClick = {
                        val startMark = draft.pendingManualStartMs
                        if (startMark == null) {
                            onDraft { it.copy(pendingManualStartMs = playbackPositionMs, pendingIgnoreStartMs = null) }
                            onMessage("Missed rally starts at ${preciseTime(playbackPositionMs)}")
                        } else {
                            val start = min(startMark, playbackPositionMs)
                            val end = max(startMark, playbackPositionMs)
                            if (end - start < MIN_MARK_MS) onMessage("A rally must be at least 0.1 seconds")
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
                    },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text(if (draft.pendingManualStartMs == null) "Add rally start" else "Add rally end", fontSize = 11.sp)
                }
                if (draft.pendingManualStartMs != null) {
                    Text("Started ${preciseTime(draft.pendingManualStartMs)}", color = Muted, fontSize = 10.sp)
                    TextButton(onClick = { onDraft { it.copy(pendingManualStartMs = null) } }) { Text("Cancel") }
                }
            }
            Box(Modifier.width(1.dp).heightIn(min = 48.dp).background(Rail))
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Button(
                    onClick = {
                        val startMark = draft.pendingIgnoreStartMs
                        if (startMark == null) {
                            onDraft { it.copy(pendingIgnoreStartMs = playbackPositionMs, pendingManualStartMs = null) }
                            onMessage("Left-out section starts at ${preciseTime(playbackPositionMs)}")
                        } else {
                            val start = min(startMark, playbackPositionMs)
                            val end = max(startMark, playbackPositionMs)
                            if (end - start < MIN_MARK_MS) onMessage("A section must be at least 0.1 seconds")
                            else {
                                val ignored = IgnoredSourceInterval(
                                    EditorMath.nextId("I", draft.ignoredIntervals.map { it.id }),
                                    start, end, "manually-excluded",
                                )
                                onDraft { it.copy(pendingIgnoreStartMs = null, ignoredIntervals = it.ignoredIntervals + ignored) }
                                onMessage("Left out ${preciseTime(start)} to ${preciseTime(end)}")
                            }
                        }
                    },
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.buttonColors(containerColor = Orange, contentColor = Color.White),
                ) {
                    Text(if (draft.pendingIgnoreStartMs == null) "Leave out start" else "Leave out end", fontSize = 11.sp)
                }
                if (draft.pendingIgnoreStartMs != null) {
                    Text("Started ${preciseTime(draft.pendingIgnoreStartMs)}", color = Muted, fontSize = 10.sp)
                    TextButton(onClick = { onDraft { it.copy(pendingIgnoreStartMs = null) } }) { Text("Cancel") }
                }
            }
        }
    }
    if (flat) {
        DesktopSidebarSection(
            title = "FIX MISSED OR EXTRA FOOTAGE",
            modifier = modifier,
            content = toolsContent,
        )
    } else {
        SectionCard(
            label = "FIX MISSED OR EXTRA FOOTAGE",
            subtitle = null,
            modifier = modifier,
            content = toolsContent,
        )
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
    return "${base.ifBlank { "volleysplice" }}-cut.mp4"
}

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
