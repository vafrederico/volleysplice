package com.volleycut.nativeanalysis

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.*
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.interaction.DragInteraction
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyListState
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.mapSaver
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.*
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.*
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.unit.TextUnit
import kotlinx.coroutines.flow.collectLatest
import kotlin.math.abs

internal val CourtsideGround = Color(0xFFEEF3F4)
private val CourtsideInk = Color(0xFF131416)
private val CourtsideMuted = Color(0xFF5D5F60)
private val CourtsideAccent = Color(0xFF1B6C80)
private val CourtsideControl = Color(0xFFCDCED1)
private val CourtsideRaised = Color(0xFFF4F6F8)
private val CourtsideFont = FontFamily(
    Font(R.font.courtside_sans_400, FontWeight.Normal),
    Font(R.font.courtside_sans_500, FontWeight.Medium),
    Font(R.font.courtside_sans_600, FontWeight.SemiBold),
)

@Composable
internal fun CourtsideTheme(content: @Composable () -> Unit) {
    val t = MaterialTheme.typography
    MaterialTheme(
        colorScheme = MaterialTheme.colorScheme.copy(
            primary = CourtsideInk, onPrimary = Color.White,
            surface = CourtsideGround, onSurface = CourtsideInk,
            surfaceContainerHighest = CourtsideRaised,
            secondaryContainer = CourtsideRaised, onSecondaryContainer = CourtsideInk,
            outline = CourtsideControl, outlineVariant = Color(0xFFE8EAEC),
        ),
        typography = t.copy(
            bodyLarge = t.bodyLarge.copy(fontFamily = CourtsideFont),
            bodyMedium = t.bodyMedium.copy(fontFamily = CourtsideFont),
            bodySmall = t.bodySmall.copy(fontFamily = CourtsideFont),
            titleLarge = t.titleLarge.copy(fontFamily = CourtsideFont),
            titleMedium = t.titleMedium.copy(fontFamily = CourtsideFont),
            titleSmall = t.titleSmall.copy(fontFamily = CourtsideFont),
            labelLarge = t.labelLarge.copy(fontFamily = CourtsideFont),
            labelMedium = t.labelMedium.copy(fontFamily = CourtsideFont),
            labelSmall = t.labelSmall.copy(fontFamily = CourtsideFont),
        ),
        content = content,
    )
}

@Composable
internal fun EditorVersionPill(rallies: Boolean, exporting: Boolean, onChange: (Boolean) -> Unit,
    onExport: () -> Unit, exportModifier: Modifier = Modifier) {
    Row(Modifier.selectableGroup().clip(RoundedCornerShape(50))
        .background(if (rallies) Color(0xFFEBEDEE) else Rail.copy(alpha = .4f))
        .padding(3.dp).testTag("editor-version-pill")) {
        listOf("Rallies", "Editor", "Export").forEach { label ->
            val chosen = if (label == "Export") exporting else !exporting && rallies == (label == "Rallies")
            Box(
                Modifier.clip(RoundedCornerShape(50))
                    .background(if (chosen) Color.White else Color.Transparent)
                    .selectable(selected = chosen, role = Role.Tab,
                        onClick = { if (label == "Export") onExport() else onChange(label == "Rallies") })
                    .testTag("workspace-" + label.lowercase(java.util.Locale.ROOT) + "-tab")
                    .then(if (label == "Export") exportModifier else Modifier)
                    .heightIn(min = 44.dp).padding(horizontal = 8.dp),
                contentAlignment = Alignment.Center,
            ) {
                Text(label, color = if (rallies) CourtsideInk else Ink, fontSize = 12.sp,
                    fontFamily = if (rallies) CourtsideFont else FontFamily.Default)
            }
        }
    }
}

/** Only navigation lives here. Project edits always go through EditorScreen's single draft. */
@Stable
internal class RallyWorkspaceState(val list: LazyListState = LazyListState()) {
    var initialized by mutableStateOf(false)
    var focus by mutableStateOf<RallyReviewItem?>(null)
    var lastSelection by mutableStateOf<String?>(null)
    var browsing by mutableStateOf(false)
    var editingRally by mutableStateOf(false)
    var showTimeline by mutableStateOf(true)
    var addingMissing by mutableStateOf(false)
    var reviewKeys by mutableStateOf<Set<String>>(emptySet())
    var following by mutableStateOf(true)
    companion object {
        val Saver = mapSaver(
            save = { state -> mapOf(
                "initialized" to state.initialized, "browsing" to state.browsing,
                "editing" to state.editingRally, "timeline" to state.showTimeline,
                "adding" to state.addingMissing, "following" to state.following,
                "last" to state.lastSelection.orEmpty(), "keys" to ArrayList(state.reviewKeys),
                "focus" to state.focus?.key.orEmpty(), "rally" to state.focus?.rallyId.orEmpty(),
                "start" to (state.focus?.startMs ?: 0L), "end" to (state.focus?.endMs ?: 0L),
                "cleanup" to ArrayList(state.focus?.cleanupIds.orEmpty()),
                "serves" to ArrayList(state.focus?.serveIds.orEmpty()),
                "listIndex" to state.list.firstVisibleItemIndex,
                "listOffset" to state.list.firstVisibleItemScrollOffset,
            ) },
            restore = { values ->
                RallyWorkspaceState(LazyListState(values["listIndex"] as Int, values["listOffset"] as Int)).apply {
                    initialized = values["initialized"] as Boolean
                    browsing = values["browsing"] as Boolean
                    editingRally = values["editing"] as Boolean
                    showTimeline = values["timeline"] as Boolean
                    addingMissing = values["adding"] as Boolean
                    following = values["following"] as Boolean
                    lastSelection = (values["last"] as String).ifEmpty { null }
                    reviewKeys = (values["keys"] as List<*>).filterIsInstance<String>().toSet()
                    val key = values["focus"] as String
                    if (key.isNotEmpty()) focus = RallyReviewItem(
                        key, (values["rally"] as String).ifEmpty { null },
                        values["start"] as Long, values["end"] as Long,
                        cleanupIds = (values["cleanup"] as List<*>).filterIsInstance<String>().toSet(),
                        serveIds = (values["serves"] as List<*>).filterIsInstance<String>().toSet(),
                    )
                }
            },
        )
    }
}

@Composable
internal fun RallyReviewWorkspace(
    state: RallyWorkspaceState,
    seed: EditorSeed,
    rallies: List<EditableRallyGroup>,
    queue: List<RallyReviewItem>,
    selectedId: String?,
    effectiveIds: Set<String>,
    scoreOverlay: PreparedScoreOverlay,
    tracking: ScoreTracking,
    ignored: List<IgnoredSourceInterval>,
    positionMs: Long,
    playing: Boolean,
    playbackRate: Double,
    sourceAvailable: Boolean,
    sourceStatus: String?,
    onRelink: () -> Unit,
    onSelect: (RallyReviewItem) -> Unit,
    onSelectRally: (EditableRallyGroup) -> Unit,
    onSeek: (Long) -> Unit,
    onScrubbingChange: (Boolean) -> Unit,
    onPlay: (Long, Long) -> Unit,
    onPlaybackRateChange: (Double) -> Unit,
    onDecision: (RallyReviewItem, Boolean) -> Unit,
    onServe: (ServeMarker, ServingSide?) -> Unit,
    onNewServe: (EditableRallyGroup, ServingSide) -> Unit,
    onAdd: (Long, Long) -> Boolean,
    onExport: () -> Unit,
    onTrim: (String, Long) -> Unit,
    guidedTourTargets: GuidedTourTargets,
    player: @Composable () -> Unit,
) {
    val selected = rallies.firstOrNull { selectedId in it.cutIds }
    val focus = state.focus
    val rally = if (!state.browsing && !state.editingRally && focus?.rallyId == null) null else selected
    val pending = queue.firstOrNull { it.key == focus?.key }
    val index = rallies.indexOfFirst { it.id == rally?.id }
    val occupied = rallies.map { it.keepStartMs to it.keepEndMs } + ignored.map { it.startMs to it.endMs }
    val start = (rally?.keepStartMs ?: focus?.startMs ?: seed.gameStartMs).coerceAtLeast(seed.gameStartMs)
    val end = (rally?.keepEndMs ?: focus?.endMs ?: seed.gameEndMs).coerceAtMost(seed.gameEndMs)
    val activeServes = if (tracking.enabled) {
        (rally?.let { rallyServeMarkers(it, tracking.serveMarkers) }.orEmpty() +
            tracking.serveMarkers.filter { it.id in focus?.serveIds.orEmpty() && !state.browsing })
            .distinctBy { it.id }
    } else emptyList()

    SideEffect { if (!state.reviewKeys.containsAll(queue.map { it.key })) state.reviewKeys += queue.map { it.key } }
    LaunchedEffect(selectedId) {
        if (!state.initialized) {
            state.initialized = true
            state.reviewKeys = queue.mapTo(mutableSetOf()) { it.key }
            state.focus = queue.firstOrNull()
            state.focus?.let(onSelect)
        } else if (state.lastSelection != selectedId && !state.browsing &&
            state.focus?.rallyId !in selected?.cutIds.orEmpty()) {
            state.focus = queue.firstOrNull { it.rallyId in selected?.cutIds.orEmpty() }
            state.editingRally = state.focus == null && selected != null
        }
        state.lastSelection = selectedId
    }
    BackHandler(state.addingMissing || state.browsing || state.editingRally) {
        when {
            state.addingMissing -> state.addingMissing = false
            state.editingRally -> { state.editingRally = false; state.browsing = true }
            else -> state.browsing = false
        }
    }
    fun browse() { state.browsing = true; state.editingRally = false }
    fun reviewNext() {
        val after = rally?.coreStartMs ?: focus?.startMs ?: -1
        val next = queue.firstOrNull { it.startMs > after } ?: queue.firstOrNull()
        state.focus = next
        state.editingRally = false
        state.browsing = false
        next?.let(onSelect)
    }
    fun resumeReview() {
        val next = queue.firstOrNull { it.key == state.focus?.key } ?: queue.firstOrNull()
        state.focus = next
        state.editingRally = false
        state.browsing = false
        next?.let(onSelect)
    }
    fun seekSource(time: Long) {
        state.following = true
        onSeek(time)
    }
    fun openAdd() {
        state.addingMissing = true
    }
    fun editRally(rally: EditableRallyGroup) {
        state.focus = queue.firstOrNull { it.rallyId in rally.cutIds }
        state.editingRally = true
        state.browsing = false
        onSelectRally(rally)
    }

    CourtsideTheme {
        BoxWithConstraints(Modifier.fillMaxSize().background(CourtsideGround).testTag("rally-review-workspace")) {
            val wide = maxWidth >= 720.dp || (maxWidth >= 560.dp && maxHeight < 480.dp)
            val stackedTransport = wide && maxWidth < 720.dp
            val compactProgress = wide && maxHeight < 480.dp
            val compactControls = (maxHeight * .49f).coerceIn(240.dp, 360.dp)
            val progress: @Composable () -> Unit = {
                Column(Modifier.fillMaxWidth().guidedTourTarget("rally-tour-progress", guidedTourTargets)
                    .padding(horizontal = 16.dp, vertical = if (compactProgress) 4.dp else 8.dp)) {
                    if (!state.browsing && !state.addingMissing && !compactProgress) {
                        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                            Text(if (queue.isEmpty() && focus == null && !state.editingRally) "Review complete" else "Review rallies",
                                Modifier.weight(1f), fontSize = 16.sp, fontWeight = FontWeight.SemiBold, color = CourtsideInk)
                            TextButton(onClick = ::browse, modifier = Modifier.heightIn(min = 44.dp).testTag("rally-browse-all")
                                .guidedTourTarget("rally-tour-all", guidedTourTargets)) {
                                Text("All " + rallies.size + "  →", color = CourtsideInk, fontSize = 13.sp)
                            }
                        }
                    }
                    val checked = state.reviewKeys.count { key -> queue.none { it.key == key } }
                    Row(Modifier.fillMaxWidth().padding(bottom = 6.dp), verticalAlignment = Alignment.CenterVertically) {
                        Text(if (queue.isEmpty()) "Review complete" else "" + queue.size + " need review",
                            Modifier.weight(1f), color = CourtsideMuted, fontSize = 12.sp)
                        Text("$checked / " + state.reviewKeys.size + " checked", color = CourtsideMuted,
                            fontSize = 12.sp, modifier = Modifier.testTag("rally-review-progress"))
                        if (compactProgress && !state.browsing && !state.addingMissing) {
                            TextButton(onClick = ::browse, modifier = Modifier.heightIn(min = 44.dp).testTag("rally-browse-all")
                                .guidedTourTarget("rally-tour-all", guidedTourTargets)) {
                                Text("All " + rallies.size + "  →", color = CourtsideInk, fontSize = 13.sp)
                            }
                        }
                    }
                    LinearProgressIndicator(
                        progress = { if (state.reviewKeys.isEmpty()) 0f else checked.toFloat() / state.reviewKeys.size },
                        modifier = Modifier.fillMaxWidth().height(3.dp),
                        color = CourtsideAccent, trackColor = Color(0xFFDCE3E5),
                        drawStopIndicator = {},
                    )
                }
            }
            val controls: @Composable () -> Unit = {
                Surface(Modifier.fillMaxWidth().padding(8.dp), shape = RoundedCornerShape(16.dp),
                    color = Color.White, border = BorderStroke(1.dp, Color(0xFFE8EAEC))) {
                    Column(Modifier.guidedTourTarget("rally-tour-controls", guidedTourTargets)
                        .heightIn(max = if (wide) maxHeight - 24.dp else compactControls)
                        .verticalScroll(rememberScrollState()).padding(12.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        when {
                            state.addingMissing -> {
                                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                                    Text("Add missing rally", Modifier.weight(1f), fontSize = 17.sp, fontWeight = FontWeight.SemiBold)
                                    TextButton(onClick = { state.addingMissing = false }) { Text("Cancel") }
                                }
                                Text("Source position " + preciseTime(positionMs), fontSize = 13.sp, color = CourtsideMuted)
                                WholeTimeline(
                                    windowStartMs = seed.gameStartMs, windowEndMs = seed.gameEndMs,
                                    cuts = rallies.flatMap { it.cuts }, joinedGaps = emptyList(), ignored = ignored,
                                    suggestions = emptyList(), appliedSuggestionIds = emptySet(),
                                    userRemovedSuggestionIds = emptySet(), userRemovedCleanupCutIds = emptySet(),
                                    selectedSuggestionId = null, selectedCutIds = emptySet(), effectiveIds = effectiveIds,
                                    confidenceThreshold = 1f,
                                    reviewedCutIds = rallies.filter { rally -> queue.none { it.rallyId == rally.id } }
                                        .flatMapTo(mutableSetOf()) { it.cutIds },
                                    playheadMs = positionMs, serveMarkers = emptyList(), sideSwitchMarkers = emptyList(),
                                    selectedScoreMarkerId = null, onMarkerSelect = { _, time -> onSeek(time) },
                                    onSeek = { time, _, _ -> onSeek(time) },
                                    onScrubbingChange = onScrubbingChange,
                                )
                                val candidate = missingRallyRange(positionMs, seed.gameStartMs, seed.gameEndMs, occupied)
                                if (candidate != null) {
                                    Text("New rally " + preciseTime(candidate.startMs) + " – " + preciseTime(candidate.endMs),
                                        fontSize = 13.sp, color = CourtsideMuted)
                                } else Text("Choose a gap between rallies to add missing footage.", color = CourtsideMuted, fontSize = 13.sp)
                                Button(
                                    onClick = {
                                        candidate?.let { range ->
                                            if (onAdd(range.startMs, range.endMs)) {
                                                state.addingMissing = false
                                                browse()
                                                state.following = true
                                            }
                                        }
                                    }, enabled = candidate != null, modifier = Modifier.fillMaxWidth().testTag("add-missing-rally-confirm"),
                                    shape = RoundedCornerShape(6.dp),
                                ) { Text("Add rally here") }
                            }
                            focus == null && !state.editingRally -> {
                                Text(if (queue.isEmpty()) "All checked." else "Continue reviewing", fontSize = 22.sp, fontWeight = FontWeight.SemiBold)
                                Text(if (queue.isEmpty()) "Browse the recording for anything missed." else "Review the remaining decisions.",
                                    fontSize = 14.sp, color = CourtsideMuted)
                                Button(onClick = { if (queue.isEmpty()) browse() else reviewNext() },
                                    modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(6.dp)) {
                                    Text(if (queue.isEmpty()) "Browse rallies" else "Continue review")
                                }
                                OutlinedButton(onClick = onExport, modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(6.dp)) { Text("Export") }
                            }
                            else -> {
                                if (rally != null) {
                                    RallyBoundsTimeline(rallies, rally, "Rally " + (index + 1), rally.coreStartMs, rally.coreEndMs,
                                        seed.gameStartMs, seed.gameEndMs, positionMs, effectiveIds,
                                        pending?.clipReview == true, onSeek, onTrim, onScrubbingChange,
                                        startMax = rally.cuts.minBy { it.coreStartMs }.coreEndMs,
                                        endMin = rally.cuts.maxBy { it.coreEndMs }.coreStartMs)
                                } else Text("Review footage", fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
                                val decision = if (state.editingRally && rally != null) {
                                    RallyReviewItem("rally-" + rally.id, rally.id, rally.keepStartMs, rally.keepEndMs)
                                } else focus
                                if (decision != null && (rally != null || decision.cleanupIds.isNotEmpty())) {
                                    val kept = rally?.cutIds?.any { it in effectiveIds } == true
                                    val unresolved = pending?.clipReview == true || pending?.cleanupIds?.isNotEmpty() == true
                                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                        RallyChoice("Keep", !unresolved && kept, { onDecision(decision, true) }, Modifier.weight(1f))
                                        RallyChoice("Remove", !unresolved && !kept, { onDecision(decision, false) }, Modifier.weight(1f))
                                    }
                                }
                                activeServes.forEach { marker ->
                                    if (activeServes.size > 1) Text("Serve " + preciseTime(marker.timestampMs), fontSize = 12.sp, color = CourtsideMuted)
                                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                        RallyChoice("Near", marker.side == ServingSide.NEAR, { onServe(marker, ServingSide.NEAR) }, Modifier.weight(1f))
                                        RallyChoice("Far", marker.side == ServingSide.FAR, { onServe(marker, ServingSide.FAR) }, Modifier.weight(1f))
                                        RallyChoice("Not a serve", false, { onServe(marker, null) }, Modifier.weight(1f))
                                    }
                                }
                                if (activeServes.isEmpty() && tracking.enabled && rally != null) {
                                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                        RallyChoice("Near", false, { onNewServe(rally, ServingSide.NEAR) }, Modifier.weight(1f))
                                        RallyChoice("Far", false, { onNewServe(rally, ServingSide.FAR) }, Modifier.weight(1f))
                                        RallyChoice("Not a serve", true, {}, Modifier.weight(1f))
                                    }
                                }
                                val neighbours = rally?.let {
                                    rallyTimelineContext(rallies, it, effectiveIds, seed.gameStartMs, seed.gameEndMs)
                                }
                                Row(Modifier.fillMaxWidth().guidedTourTarget("rally-tour-navigation", guidedTourTargets), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    OutlinedButton(onClick = { neighbours?.previous?.let(::editRally) },
                                        enabled = neighbours?.previous != null,
                                        modifier = Modifier.weight(1f).heightIn(min = 48.dp).testTag("previous-rally"),
                                        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 8.dp),
                                        shape = RoundedCornerShape(6.dp)) { Text("Previous rally", fontSize = 12.sp) }
                                    Button(onClick = ::reviewNext, enabled = pending == null,
                                        modifier = Modifier.weight(1f).heightIn(min = 48.dp).testTag("rally-next-review"),
                                        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 8.dp),
                                        shape = RoundedCornerShape(6.dp)) { Text("Next review", fontSize = 12.sp) }
                                    OutlinedButton(onClick = { neighbours?.next?.let(::editRally) },
                                        enabled = neighbours?.next != null,
                                        modifier = Modifier.weight(1f).heightIn(min = 48.dp).testTag("next-rally"),
                                        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 8.dp),
                                        shape = RoundedCornerShape(6.dp)) { Text("Next rally", fontSize = 12.sp) }
                                }
                            }
                        }
                    }
                }
            }
            val sourcePlayback = state.browsing || state.addingMissing
            val video: @Composable (Modifier) -> Unit = { modifier ->
                Column(modifier) {
                    if (!sourceAvailable || sourceStatus != null) {
                        Row(Modifier.padding(horizontal = 16.dp), verticalAlignment = Alignment.CenterVertically) {
                            Text(sourceStatus ?: "Re-link the original video.", Modifier.weight(1f), color = Danger, fontSize = 12.sp)
                            if (!sourceAvailable) TextButton(onClick = onRelink) { Text("Re-link video") }
                        }
                    }
                    Box(Modifier.weight(1f).fillMaxWidth().background(Color(0xFF17191A))
                        .clickable(enabled = sourceAvailable, role = Role.Button,
                            onClickLabel = if (playing) "Pause" else "Play") {
                            onPlay(if (sourcePlayback) seed.gameStartMs else start, if (sourcePlayback) seed.gameEndMs else end)
                        }.testTag("rally-review-player")) {
                        player()
                        if (stackedTransport) {
                            Text(preciseTime(positionMs), color = Color.White, fontSize = 14.sp,
                                modifier = Modifier.align(Alignment.BottomEnd).padding(4.dp)
                                    .background(Color.Black.copy(alpha = .7f), RoundedCornerShape(4.dp))
                                    .padding(horizontal = 6.dp, vertical = 2.dp))
                        }
                    }
                    val durationLabel = if (sourcePlayback) compactTime(seed.gameEndMs - seed.gameStartMs) + " source"
                        else ((end - start) / 1000f).let { "%.1f".format(java.util.Locale.US, it) } + " s"
                    Column(Modifier.fillMaxWidth().background(Color(0xFF17191A))
                        .padding(horizontal = if (stackedTransport) 8.dp else 16.dp, vertical = 4.dp)) {
                    Row(Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(if (stackedTransport) 8.dp else 12.dp)) {
                        IconButton(onClick = { onPlay(if (sourcePlayback) seed.gameStartMs else start, if (sourcePlayback) seed.gameEndMs else end) },
                            enabled = sourceAvailable, modifier = Modifier.size(48.dp).border(1.dp, Color(0xFF777777), RoundedCornerShape(50)),
                        ) {
                            Canvas(Modifier.size(22.dp).semantics { contentDescription = if (playing) "Pause" else "Play" }) {
                                if (playing) {
                                    val bar = 4.dp.toPx(); val height = 16.dp.toPx(); val gap = 4.dp.toPx()
                                    drawRect(Color.White, Offset(center.x - gap / 2 - bar, center.y - height / 2), Size(bar, height))
                                    drawRect(Color.White, Offset(center.x + gap / 2, center.y - height / 2), Size(bar, height))
                                } else {
                                    val halfWidth = 7.dp.toPx(); val halfHeight = 9.dp.toPx()
                                    val triangle = Path().apply {
                                        moveTo(center.x - halfWidth, center.y - halfHeight)
                                        lineTo(center.x + halfWidth, center.y)
                                        lineTo(center.x - halfWidth, center.y + halfHeight)
                                        close()
                                    }
                                    drawPath(triangle, Color.White)
                                }
                            }
                        }
                        if (!stackedTransport) {
                            Text(preciseTime(positionMs), color = Color.White, fontSize = 14.sp, modifier = Modifier.weight(1f))
                            Text(durationLabel, color = Color(0xFFC4C7C8), fontSize = 12.sp)
                        } else {
                            Text(durationLabel, Modifier.weight(1f), color = Color(0xFFC4C7C8), fontSize = 12.sp,
                                textAlign = androidx.compose.ui.text.style.TextAlign.End)
                        }
                        RallySpeedSelector(playbackRate, onPlaybackRateChange)
                    }
                    }
                }
            }
            Column(Modifier.fillMaxSize()) {
                progress()
                if (wide) {
                    Row(Modifier.weight(1f)) {
                        video(Modifier.weight(1f).fillMaxHeight())
                        if (state.browsing && !state.addingMissing) {
                            RallyBrowser(state, seed, rallies, effectiveIds, queue, scoreOverlay, tracking.enabled,
                                selected, positionMs, ::seekSource, onScrubbingChange, {
                                    state.following = true; onSelectRally(it)
                                }, ::editRally,
                                ::resumeReview, ::openAdd, Modifier.width(340.dp).fillMaxHeight(), compact = compactProgress,
                                guidedTourTargets = guidedTourTargets)
                        } else Column(Modifier.width(350.dp).fillMaxHeight()) { controls() }
                    }
                } else {
                    video(Modifier.weight(if (state.browsing) .95f else 1f).fillMaxWidth())
                    if (state.browsing && !state.addingMissing) {
                        RallyBrowser(state, seed, rallies, effectiveIds, queue, scoreOverlay, tracking.enabled,
                            selected, positionMs, ::seekSource, onScrubbingChange, {
                                state.following = true; onSelectRally(it)
                            }, ::editRally,
                            ::resumeReview, ::openAdd, Modifier.weight(1f).fillMaxWidth(), guidedTourTargets = guidedTourTargets)
                    } else controls()
                }
            }
        }
    }
}

@Composable
private fun RallySpeedSelector(rate: Double, onChange: (Double) -> Unit) {
    var expanded by remember { mutableStateOf(false) }
    Box {
        TextButton(onClick = { expanded = true },
            modifier = Modifier.heightIn(min = 48.dp).widthIn(min = 64.dp)
                .border(1.dp, Color(0xFF777777), RoundedCornerShape(6.dp))
                .testTag("rally-playback-speed").semantics {
                    contentDescription = "Playback speed"
                    stateDescription = PortableReviewPreferences.playbackRateLabel(rate)
                }, contentPadding = PaddingValues(horizontal = 8.dp)) {
            Text("${PortableReviewPreferences.playbackRateLabel(rate)}  ▾", color = Color.White, fontSize = 13.sp)
        }
        DropdownMenu(expanded, onDismissRequest = { expanded = false }) {
            PortableReviewPreferences.playbackRates(rate).forEach { option ->
                DropdownMenuItem(text = { Text(PortableReviewPreferences.playbackRateLabel(option)) },
                    modifier = Modifier.semantics { selected = rate == option },
                    onClick = { expanded = false; onChange(option) })
            }
        }
    }
}

@Composable
private fun RallyChoice(label: String, chosen: Boolean, onClick: () -> Unit, modifier: Modifier) {
    OutlinedButton(onClick = onClick, modifier = modifier.heightIn(min = 48.dp).semantics { selected = chosen },
        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 8.dp),
        shape = RoundedCornerShape(6.dp), border = BorderStroke(1.dp, if (chosen) CourtsideInk else CourtsideControl),
        colors = ButtonDefaults.outlinedButtonColors(containerColor = if (chosen) CourtsideInk else Color.White,
            contentColor = if (chosen) Color.White else CourtsideInk)) { Text(label, fontSize = 14.sp) }
}

@Composable
private fun RallyBoundsTimeline(
    rallies: List<EditableRallyGroup>, rally: EditableRallyGroup, title: String,
    startMs: Long, endMs: Long, sourceStart: Long, sourceEnd: Long,
    positionMs: Long, effectiveIds: Set<String>, needsReview: Boolean,
    onSeek: (Long) -> Unit, onTrim: (String, Long) -> Unit,
    onScrubbingChange: (Boolean) -> Unit,
    startMax: Long = endMs, endMin: Long = startMs,
) {
    val identity = rally.id
    val context = rallyTimelineContext(rallies, rally, effectiveIds, sourceStart, sourceEnd)
    var scale by remember(identity) { mutableStateOf(RallyTimelineScale(context.window)) }
    val windowStart = scale.startMs
    val windowEnd = scale.endMs
    var dragging by remember(identity) { mutableStateOf(false) }
    val liveStart by rememberUpdatedState(startMs)
    val liveEnd by rememberUpdatedState(endMs)
    val liveStartMax by rememberUpdatedState(startMax)
    val liveEndMin by rememberUpdatedState(endMin)
    val trim by rememberUpdatedState(onTrim)
    val seek by rememberUpdatedState(onSeek)
    val scrub by rememberUpdatedState(onScrubbingChange)
    val paddingBeforeView = rally.keepStartMs < windowStart
    val paddingAfterView = rally.keepEndMs > windowEnd
    LaunchedEffect(context, dragging) {
        if (!dragging) {
            scale = RallyTimelineScale(context.window)
        }
    }
    fun change(side: String, value: Long) {
        val bounded = rallyBoundaryValue(side, value, if (side == "end") endMin else startMs,
            if (side == "start") startMax else endMs, sourceStart, sourceEnd)
        onTrim(side, bounded)
    }
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Column {
                Text("Start", fontSize = 12.sp, color = CourtsideMuted)
                Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    RallyNudge("−1s", "Start earlier by 1 second") { change("start", startMs - 1_000) }
                    RallyNudge("+1s", "Start later by 1 second") { change("start", startMs + 1_000) }
                }
        }
        Text(title, Modifier.weight(1f), fontWeight = FontWeight.SemiBold, fontSize = 15.sp,
            textAlign = androidx.compose.ui.text.style.TextAlign.Center)
        Column(horizontalAlignment = Alignment.End) {
            Text("End", fontSize = 12.sp, color = CourtsideMuted)
            Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                RallyNudge("−1s", "End earlier by 1 second") { change("end", endMs - 1_000) }
                RallyNudge("+1s", "End later by 1 second") { change("end", endMs + 1_000) }
            }
        }
    }
    val labelMeasurer = rememberTextMeasurer()
    val neighbours = listOfNotNull(context.previous, context.next)
    val neighbourLabels = neighbours.associate { neighbour ->
        neighbour.id to labelMeasurer.measure("Rally " + (rallies.indexOfFirst { it.id == neighbour.id } + 1),
            style = TextStyle(fontFamily = CourtsideFont, fontSize = 11.sp, color = CourtsideMuted))
    }
    Box(Modifier.fillMaxWidth().height(64.dp)) {
        Canvas(Modifier.fillMaxSize().testTag("rally-bounds-timeline").clip(RoundedCornerShape(9.dp)).background(TimelineTrackColor)
            .semantics {
                contentDescription = "Rally timeline. Start " + preciseTime(startMs) + "; end " + preciseTime(endMs)
                stateDescription = "Previous rally " + (context.previous?.let { rallies.indexOf(it) + 1 } ?: "none") +
                    "; next rally " + (context.next?.let { rallies.indexOf(it) + 1 } ?: "none") +
                    when {
                        positionMs < windowStart -> "; video at ${preciseTime(positionMs)} before this view"
                        positionMs > windowEnd -> "; video at ${preciseTime(positionMs)} after this view"
                        else -> ""
                    }
                progressBarRangeInfo = ProgressBarRangeInfo(positionMs.coerceIn(windowStart, windowEnd).toFloat(),
                    windowStart.toFloat()..windowEnd.toFloat())
                setProgress { value -> onSeek(value.toLong().coerceIn(windowStart, windowEnd)); true }
            }.pointerInput(identity, sourceStart, sourceEnd) {
            awaitEachGesture {
                val down = awaitFirstDown(requireUnconsumed = false)
                val gestureScale = scale
                val inset = 22.dp.toPx()
                val width = (size.width - inset * 2).coerceAtLeast(1f)
                fun x(time: Long) = inset + gestureScale.fraction(time) * width
                fun time(at: Float) = gestureScale.time((at - inset) / width)
                val a = abs(down.position.x - x(liveStart)); val b = abs(down.position.x - x(liveEnd))
                // Labels and neighbour bars seek; only the selected handles trim.
                val side = if (down.position.y >= 18.dp.toPx() && minOf(a, b) <= 24.dp.toPx()) {
                    if (a <= b) "start" else "end"
                } else null
                val offset = if (side == "start") liveStart - time(down.position.x) else liveEnd - time(down.position.x)
                dragging = true
                scrub(true)
                down.consume()
                if (side == null) seek(time(down.position.x).coerceIn(sourceStart, sourceEnd))
                try {
                    while (true) {
                        val event = awaitPointerEvent()
                        val pointer = event.changes.firstOrNull { it.id == down.id } ?: break
                        if (side != null) {
                            val value = rallyBoundaryValue(side, time(pointer.position.x) + offset,
                                if (side == "end") liveEndMin else liveStart,
                                if (side == "start") liveStartMax else liveEnd, sourceStart, sourceEnd)
                            trim(side, value)
                        } else seek(time(pointer.position.x).coerceIn(sourceStart, sourceEnd))
                        pointer.consume()
                        if (!pointer.pressed) break
                    }
                } finally { dragging = false; scrub(false) }
            }
        }) {
            val inset = 22.dp.toPx(); val width = (size.width - inset * 2).coerceAtLeast(1f)
            fun x(t: Long) = inset + scale.fraction(t) * width
            val a = x(startMs); val b = x(endMs)
            neighbours.forEach { neighbour ->
                neighbour.cuts.filter { it.included && it.id in effectiveIds &&
                    it.keepEndMs > windowStart && it.keepStartMs < windowEnd }.forEach { cut ->
                    drawRoundRect(TimelineKeptPaddingColor.copy(alpha = .6f), Offset(x(cut.keepStartMs), 22.dp.toPx()),
                        Size((x(cut.keepEndMs) - x(cut.keepStartMs)).coerceAtLeast(1f), 36.dp.toPx()), CornerRadius(4.dp.toPx()))
                    drawRoundRect(Green.copy(alpha = .45f), Offset(x(cut.coreStartMs), 30.dp.toPx()),
                        Size((x(cut.coreEndMs) - x(cut.coreStartMs)).coerceAtLeast(1f), 20.dp.toPx()), CornerRadius(3.dp.toPx()))
                }
                neighbourLabels[neighbour.id]?.let { label ->
                    val labelX = x((neighbour.coreStartMs + neighbour.coreEndMs) / 2) - label.size.width / 2f
                    drawText(label, topLeft = Offset(labelX.coerceIn(0f, (size.width - label.size.width).coerceAtLeast(0f)), 4.dp.toPx()))
                }
            }
            rally.cuts.forEach { cut ->
                val removed = !cut.included
                val paddingColor = when {
                    removed -> Muted.copy(alpha = .6f)
                    cut.id !in effectiveIds -> Danger.copy(alpha = .6f)
                    needsReview -> Warning.copy(alpha = .8f)
                    else -> TimelineKeptPaddingColor
                }
                val coreColor = when {
                    removed -> Muted
                    needsReview -> Orange
                    else -> Green
                }
                drawRoundRect(paddingColor, Offset(x(cut.keepStartMs), 22.dp.toPx()),
                    Size((x(cut.keepEndMs) - x(cut.keepStartMs)).coerceAtLeast(1f), 36.dp.toPx()), CornerRadius(4.dp.toPx()))
                drawRoundRect(coreColor, Offset(x(cut.coreStartMs), 30.dp.toPx()),
                    Size((x(cut.coreEndMs) - x(cut.coreStartMs)).coerceAtLeast(1f), 20.dp.toPx()), CornerRadius(3.dp.toPx()))
            }
            listOf(a, b).forEach { at ->
                drawRoundRect(Color.White, Offset(at - 5.dp.toPx(), 34.dp.toPx()), Size(10.dp.toPx(), 28.dp.toPx()), CornerRadius(3.dp.toPx()))
                drawRoundRect(Green, Offset(at - 3.dp.toPx(), 36.dp.toPx()), Size(6.dp.toPx(), 24.dp.toPx()), CornerRadius(2.dp.toPx()))
            }
            val p = x(positionMs)
            drawLine(Orange, Offset(p, 0f), Offset(p, size.height), 2.dp.toPx())
            if (positionMs !in windowStart..windowEnd) {
                val direction = if (positionMs < windowStart) -1 else 1
                drawPath(Path().apply {
                    moveTo(p + direction * 8.dp.toPx(), 8.dp.toPx())
                    lineTo(p, 2.dp.toPx()); lineTo(p, 14.dp.toPx()); close()
                }, Orange)
            }
        }
    }
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text("Start " + preciseTime(startMs), fontSize = 12.sp, color = CourtsideMuted, modifier = Modifier.testTag("rally-start-readout"))
        Text("End " + preciseTime(endMs), fontSize = 12.sp, color = CourtsideMuted, modifier = Modifier.testTag("rally-end-readout"))
    }
    if (paddingBeforeView || paddingAfterView) {
        Text(when {
            paddingBeforeView && paddingAfterView -> "Padding continues before and after this view."
            paddingBeforeView -> "Start padding continues before this view."
            else -> "End padding continues after this view."
        }, Modifier.testTag("rally-padding-outside-view"), fontSize = 11.sp, color = CourtsideMuted)
    }
}

@Composable
private fun RallyNudge(text: String, description: String, onClick: () -> Unit) {
    Box(Modifier.size(44.dp).clip(RoundedCornerShape(6.dp)).background(CourtsideRaised)
        .border(1.dp, CourtsideControl, RoundedCornerShape(6.dp)).clickable(role = Role.Button, onClick = onClick)
        .semantics { contentDescription = description }, contentAlignment = Alignment.Center) {
        Text(text, fontSize = 13.sp, color = CourtsideInk)
    }
}

@Composable
private fun RallyBrowser(
    state: RallyWorkspaceState, seed: EditorSeed, rallies: List<EditableRallyGroup>, effectiveIds: Set<String>,
    queue: List<RallyReviewItem>, scoreOverlay: PreparedScoreOverlay, scoresEnabled: Boolean,
    selectedRally: EditableRallyGroup?, positionMs: Long,
    onSeek: (Long) -> Unit, onScrubbingChange: (Boolean) -> Unit, onSelect: (EditableRallyGroup) -> Unit,
    onEdit: (EditableRallyGroup) -> Unit, onReview: () -> Unit, onAdd: () -> Unit, modifier: Modifier,
    compact: Boolean = false,
    guidedTourTargets: GuidedTourTargets? = null,
) {
    val list = state.list
    val hit = rallyAtTime(rallies, positionMs)
    val target = nearestRallyIndex(rallies, positionMs)
    val centerIndex by remember(list) { derivedStateOf {
        val layout = list.layoutInfo
        val center = (layout.viewportStartOffset + layout.viewportEndOffset) / 2
        layout.visibleItemsInfo.minByOrNull { abs(it.offset + it.size / 2 - center) }?.index
    } }
    val centered = rallies.getOrNull(centerIndex ?: -1)
    val centerTime = centered?.let { (it.coreStartMs + it.coreEndMs) / 2 }
    LaunchedEffect(list) {
        list.interactionSource.interactions.collectLatest { if (it is DragInteraction.Start) state.following = false }
    }
    // Restart for a new rally or when resuming follow, not for every pointer move:
    // repeated cancellation before the layout frame can starve scrolling during a seek.
    LaunchedEffect(state.following, target) {
        if (state.following && target >= 0) {
            withFrameNanos { }
            if (!state.following) return@LaunchedEffect
            val layout = list.layoutInfo
            val itemHeight = layout.visibleItemsInfo.firstOrNull()?.size ?: 78
            val height = layout.viewportEndOffset - layout.viewportStartOffset
            list.scrollToItem(target, -((height - itemHeight).coerceAtLeast(0) / 2))
        }
    }
    Column(modifier.background(Color.White).testTag("rally-browser")) {
        val sourceLabel = if (hit == null) "No rally at " + preciseTime(positionMs)
            else "Rally " + (rallies.indexOf(hit) + 1) + " · " + preciseTime(positionMs)
        Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("" + rallies.size + " rallies", fontWeight = FontWeight.SemiBold, fontSize = 16.sp,
                    lineHeight = if (compact) 20.sp else TextUnit.Unspecified)
                if (compact) Text(sourceLabel, color = if (hit == null) CourtsideAccent else CourtsideMuted,
                    fontSize = 12.sp, lineHeight = 16.sp, modifier = Modifier.testTag("rally-source-status"))
            }
            if (queue.isNotEmpty()) {
                TextButton(onClick = onReview, modifier = Modifier.testTag("rally-browser-review")) {
                    Text("Review " + queue.size, fontSize = 12.sp)
                }
            }
            TextButton(onClick = { state.showTimeline = !state.showTimeline }) {
                Text(if (state.showTimeline) "Hide timeline" else "Show timeline", fontSize = 12.sp)
            }
        }
        if (!compact) Text(sourceLabel,
            Modifier.padding(horizontal = 16.dp, vertical = 4.dp).testTag("rally-source-status"),
            color = if (hit == null) CourtsideAccent else CourtsideMuted, fontSize = 12.sp)
        Row(Modifier.weight(1f)) {
            if (state.showTimeline) {
                Column(Modifier.width(64.dp).fillMaxHeight().padding(horizontal = 5.dp, vertical = 8.dp),
                    horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(compactTime(seed.gameStartMs), color = CourtsideMuted, fontSize = 11.sp,
                        lineHeight = if (compact) 14.sp else TextUnit.Unspecified)
                    RallySourceTimeline(seed, rallies, queue, effectiveIds, positionMs, centerTime, centerIndex, onSeek, onScrubbingChange,
                        Modifier.weight(1f).fillMaxWidth().guidedTourTarget("rally-tour-source", guidedTourTargets))
                    Text(compactTime(seed.gameEndMs), color = CourtsideMuted, fontSize = 11.sp,
                        lineHeight = if (compact) 14.sp else TextUnit.Unspecified)
                }
            }
            Column(Modifier.weight(1f)) {
                Row(Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = if (compact) 2.dp else 4.dp), horizontalArrangement = Arrangement.SpaceBetween) {
                    Text("● Play", fontSize = 11.sp, color = CourtsideAccent,
                        lineHeight = if (compact) 14.sp else TextUnit.Unspecified)
                    Text("◇ List · " + (centerIndex?.plus(1)?.toString() ?: "—"), fontSize = 11.sp, color = CourtsideMuted,
                        lineHeight = if (compact) 14.sp else TextUnit.Unspecified,
                        modifier = Modifier.testTag("rally-list-center"))
                }
                LazyColumn(
                    state = list, modifier = Modifier.weight(1f).testTag("rally-card-list")
                        .guidedTourTarget("rally-tour-list", guidedTourTargets)
                        .pointerInput(state) {
                            awaitPointerEventScope {
                                while (true) {
                                    val event = awaitPointerEvent(PointerEventPass.Initial)
                                    if (event.type == PointerEventType.Scroll || event.changes.any { it.pressed && !it.previousPressed }) {
                                        state.following = false
                                    }
                                }
                            }
                        },
                    contentPadding = PaddingValues(horizontal = 8.dp), verticalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    itemsIndexed(rallies, key = { _, rally -> rally.id }) { index, rally ->
                        val score = remember(rally.coreStartMs, scoreOverlay) {
                            ScoreReducer.deriveAt(scoreOverlay.tracking, ScoreReducer.scoreBoundaryTimestamp(
                                rally.coreStartMs, scoreOverlay.rallyRanges, scoreOverlay.tracking, scoreOverlay.mergedRanges))
                        }
                        val active = selectedRally?.id == rally.id
                        val kept = rally.cutIds.any { it in effectiveIds }
                        val needsReview = queue.any { it.rallyId == rally.id }
                        Row(Modifier.fillMaxWidth().heightIn(min = 48.dp).clip(RoundedCornerShape(6.dp))
                            .background(if (active) Color(0xFFEBEDEE) else Color.White)
                            .selectable(active, role = Role.Button, onClick = { onSelect(rally) })
                            .padding(horizontal = 6.dp, vertical = if (compact) 6.dp else 10.dp).testTag("rally-card-" + rally.id),
                            verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                                Text("Rally " + (index + 1), fontSize = 14.sp, fontWeight = FontWeight.Medium,
                                    lineHeight = if (compact) 18.sp else TextUnit.Unspecified)
                                Text(compactTime(rally.coreStartMs), color = CourtsideMuted, fontSize = 12.sp,
                                    lineHeight = if (compact) 16.sp else TextUnit.Unspecified)
                            }
                            Column(horizontalAlignment = Alignment.End, verticalArrangement = Arrangement.spacedBy(4.dp)) {
                                Text(if (scoresEnabled) "" + score.team1Score + " : " + score.team2Score else "—", fontSize = 15.sp,
                                    lineHeight = if (compact) 18.sp else TextUnit.Unspecified)
                                Text(if (needsReview) "Review" else if (kept) "Keep" else "Remove",
                                    fontSize = 11.sp, color = if (!kept) Danger else CourtsideMuted,
                                    lineHeight = if (compact) 16.sp else TextUnit.Unspecified)
                            }
                        }
                        HorizontalDivider(color = Color(0xFFE8EAEC))
                    }
                }
            }
        }
        Row(Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = if (compact) 4.dp else 8.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = { selectedRally?.let(onEdit) },
                enabled = selectedRally != null,
                modifier = Modifier.weight(1f).testTag("edit-selected-rally")
                    .semantics { contentDescription = "Edit selected rally" },
                shape = RoundedCornerShape(6.dp)) {
                val selectedIndex = rallies.indexOfFirst { it.id == selectedRally?.id }
                Text(if (selectedIndex >= 0) "Edit rally " + (selectedIndex + 1) else "Edit selected rally", fontSize = 12.sp)
            }
            Button(onClick = onAdd, modifier = Modifier.weight(1.5f).testTag("add-missing-rally")
                .guidedTourTarget("rally-tour-missing", guidedTourTargets),
                colors = ButtonDefaults.buttonColors(containerColor = if (hit == null) CourtsideAccent else CourtsideRaised,
                    contentColor = if (hit == null) Color.White else CourtsideInk),
                shape = RoundedCornerShape(6.dp)) { Text("+ Add missing rally", fontSize = 12.sp) }
        }
    }
}

@Composable
private fun RallySourceTimeline(
    seed: EditorSeed, rallies: List<EditableRallyGroup>, queue: List<RallyReviewItem>, effectiveIds: Set<String>,
    positionMs: Long, listTime: Long?, listIndex: Int?, onSeek: (Long) -> Unit,
    onScrubbingChange: (Boolean) -> Unit, modifier: Modifier,
) {
    val seek by rememberUpdatedState(onSeek)
    val scrub by rememberUpdatedState(onScrubbingChange)
    Canvas(modifier.testTag("rally-vertical-timeline").semantics {
        contentDescription = "Full recording timeline"
        stateDescription = "Playback " + preciseTime(positionMs) + "; list center rally " + (listIndex?.plus(1) ?: 0)
        progressBarRangeInfo = ProgressBarRangeInfo(positionMs.toFloat(), seed.gameStartMs.toFloat()..seed.gameEndMs.toFloat())
        setProgress { value -> onSeek(value.toLong().coerceIn(seed.gameStartMs, seed.gameEndMs)); true }
    }.pointerInput(seed.gameStartMs, seed.gameEndMs) {
        awaitEachGesture {
            val down = awaitFirstDown(requireUnconsumed = false)
            fun at(y: Float) = seed.gameStartMs + ((y / size.height.coerceAtLeast(1)).coerceIn(0f, 1f) * (seed.gameEndMs - seed.gameStartMs)).toLong()
            scrub(true)
            try {
                down.consume(); seek(at(down.position.y))
                while (true) {
                    val pointer = awaitPointerEvent().changes.firstOrNull { it.id == down.id } ?: break
                    seek(at(pointer.position.y)); pointer.consume()
                    if (!pointer.pressed) break
                }
            } finally { scrub(false) }
        }
    }) {
        val duration = (seed.gameEndMs - seed.gameStartMs).coerceAtLeast(1)
        fun y(time: Long) = ((time - seed.gameStartMs).toFloat() / duration).coerceIn(0f, 1f) * size.height
        val middle = size.width / 2
        drawRoundRect(CourtsideControl, Offset(middle - 2.dp.toPx(), 0f), Size(4.dp.toPx(), size.height), CornerRadius(2.dp.toPx()))
        rallies.forEach { rally ->
            val color = when {
                queue.any { it.rallyId == rally.id } -> CourtsideAccent
                rally.cutIds.any { it in effectiveIds } -> Color(0xFF879296)
                else -> Danger
            }
            drawRoundRect(color, Offset(middle - 9.dp.toPx(), y(rally.keepStartMs)),
                Size(18.dp.toPx(), (y(rally.keepEndMs) - y(rally.keepStartMs)).coerceAtLeast(2.dp.toPx())), CornerRadius(1.dp.toPx()))
        }
        val play = y(positionMs)
        drawLine(CourtsideAccent, Offset(6.dp.toPx(), play), Offset(middle, play), 2.dp.toPx())
        drawCircle(CourtsideAccent, 4.dp.toPx(), Offset(6.dp.toPx(), play))
        listTime?.let {
            val cy = y(it); val cx = size.width - 6.dp.toPx(); val r = 5.dp.toPx()
            val diamond = Path().apply { moveTo(cx, cy - r); lineTo(cx + r, cy); lineTo(cx, cy + r); lineTo(cx - r, cy); close() }
            drawLine(CourtsideInk, Offset(middle + 3.dp.toPx(), cy), Offset(cx, cy), 1.dp.toPx())
            drawPath(diamond, Color.White); drawPath(diamond, CourtsideInk, style = Stroke(2.dp.toPx()))
        }
    }
}
