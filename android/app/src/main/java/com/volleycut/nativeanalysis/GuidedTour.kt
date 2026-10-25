package com.volleycut.nativeanalysis

import android.content.Context
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.relocation.BringIntoViewRequester
import androidx.compose.foundation.relocation.bringIntoViewRequester
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.snapshotFlow
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.composed
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.boundsInRoot
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalWindowInfo
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.edit
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.withTimeoutOrNull

internal enum class GuidedTourStage {
    SETUP,
    EDITOR,
}

internal enum class GuidedTourWorkspace { RALLY, RALLY_LIST, ADD_RALLY, EDITOR, EXPORT }

internal enum class GuidedTourStep(
    val stage: GuidedTourStage,
    val target: String,
    val title: String,
    val body: String,
    val action: String,
) {
    SETUP_SOURCE(
        GuidedTourStage.SETUP,
        "setup-source",
        "Choose your video",
        "Pick a game video from this device. Your video stays here and is not uploaded.",
        "Choose video first",
    ),
    SETUP_WINDOW(
        GuidedTourStage.SETUP,
        "setup-window",
        "Choose the part with the game",
        "If the whole video is game footage, leave it set to the full video. Otherwise, move to the game’s start and end and mark them.",
        "Next",
    ),
    SETUP_CREATE(
        GuidedTourStage.SETUP,
        "setup-create",
        "Let VolleySplice find the rallies",
        "Tap Find rallies. Keep VolleySplice open while it prepares the suggested clips; rally review opens when they are ready.",
        "Find rallies first",
    ),
    RALLIES_REVIEW(
        GuidedTourStage.EDITOR, "rally-tour-progress", "Start with rallies to review",
        "Rallies is the default view. Start with the rallies needing review; the counters show what remains and how many you have checked. Watch each rally in the large player.", "Next",
    ),
    RALLIES_TAG(
        GuidedTourStage.EDITOR, "rally-tour-controls", "Tag the rally",
        "Choose Keep or Remove. For a serve, choose Near or Far for the serving side, or Not a serve. Resolve every highlighted decision before using Next review. Open the Settings cogwheel to edit team names in the top Project settings section.", "Next",
    ),
    RALLIES_TIMELINE(
        GuidedTourStage.EDITOR, "rally-tour-controls", "Seek and trim a rally",
        "Tap or drag inside the timeline to seek the video. Drag the end handles or use the start/end one-second buttons to trim. Trimming keeps your viewed frame until a boundary passes it. Muted bars show the previous and next kept rallies.", "Next",
    ),
    RALLIES_NAVIGATION(
        GuidedTourStage.EDITOR, "rally-tour-navigation", "Move between rallies",
        "Previous rally and Next rally move between kept rallies and preserve playback. Next review moves to the next unresolved rally after you tag this one. Play pauses at the rally end; the speed selector is beside its duration.", "Next",
    ),
    RALLIES_ALL(
        GuidedTourStage.EDITOR, "rally-tour-all", "View all rallies",
        "Tap All to open the rally list whenever you want to check the rest of the recording, even while reviews remain. Review returns to the outstanding decisions.", "Next",
    ),
    RALLIES_LIST(
        GuidedTourStage.EDITOR, "rally-tour-list", "Browse and edit a rally",
        "Scroll through rally numbers, scores and source times. Tap a row to select and seek to it, then use Edit rally below the list to adjust that rally. Scrolling the list alone leaves playback where it is.", "Next",
    ),
    RALLIES_SOURCE(
        GuidedTourStage.EDITOR, "rally-tour-source", "Seek the whole recording",
        "Drag the vertical timeline to seek the video; the list follows the time you choose. The filled dot is playback. The outlined diamond marks the rally at the list center. Hide timeline collapses the sidebar.", "Next",
    ),
    RALLIES_MISSING(
        GuidedTourStage.EDITOR, "rally-tour-missing", "Find a missing rally",
        "Seek into a gap to check footage outside the detected rallies. The Add missing rally button highlights there. Seeking stays in the list; tap this button when you want to add a rally.", "Next",
    ),
    RALLIES_ADD(
        GuidedTourStage.EDITOR, "rally-tour-controls", "Add the missed footage",
        "Seek this source timeline into the missed rally. Add rally here creates the displayed suggested range; then select it and use Edit rally to refine its bounds. Cancel returns without adding anything. The tutorial adds nothing for you.", "Next",
    ),
    EDITOR_SETTINGS(
        GuidedTourStage.EDITOR,
        "workspace-tour-settings",
        "Fine-tune only when you need to",
        "The Settings gear contains optional automatic cleanup, extra time around clips, short-break joining, and review sensitivity. The defaults are ready for most games.",
        "Finish tutorial",
    ),
    EDITOR_EXPORT(
        GuidedTourStage.EDITOR,
        "workspace-tour-export",
        "Save the finished video",
        "Use Export to save the MP4, get YouTube chapters, or save the project. Choose whether scores and point history appear, and whether scores fade. Rallies returns to rally review; Editor opens the full timeline.",
        "Next",
    ),
    EDITOR_SCORE_TOGGLE(
        GuidedTourStage.EDITOR,
        "editor-score-toggle",
        "Add a scoreboard if you want one",
        "Score tracking is optional. Turn it on to check serve markers, add anything missing, and include a scoreboard in the saved video.",
        "Next: check the score",
    ),
    EDITOR_SCORE_PANEL(
        GuidedTourStage.EDITOR,
        "editor-score-panel",
        "Check the score markers",
        "Correct which side serves, add missed serves, and add a side switch whenever the teams change court sides.",
        "Next",
    ),
    EDITOR_VIDEO(
        GuidedTourStage.EDITOR,
        "editor-video",
        "Watch the video",
        "Play, pause, or move to any moment. Turn on Play only the final video above to skip every part that will not be saved.",
        "Next",
    ),
    EDITOR_REVIEW_QUEUES(
        GuidedTourStage.EDITOR,
        "editor-review-queues",
        "Start with what needs attention",
        "These queues contain automatic cleanups, clips, and serves that still need review. Ignored footage is skipped, and the counts shrink as you make decisions.",
        "Next",
    ),
    EDITOR_OVERVIEW(
        GuidedTourStage.EDITOR,
        "editor-overview",
        "Use the full Editor",
        "The Editor pill opens the full timeline workspace for editing any part of the recording, manual bounds, cleanup and score markers. Edits here and in Rallies share the same project. Use Rallies to return to focused rally review.",
        "Next",
    ),
    EDITOR_FOCUS(
        GuidedTourStage.EDITOR,
        "editor-focus",
        "Fix one clip",
        "Include or leave out the selected clip, preview it, and adjust where it starts or ends. Tap Looks good when a flagged clip is correct.",
        "Next",
    ),
    EDITOR_MARKING(
        GuidedTourStage.EDITOR,
        "editor-marking",
        "Add anything VolleySplice missed",
        "For a missed rally, mark its start and end. Leave out a section for camera gaps, breaks, or other footage that should not appear in the final video.",
        "Finish tutorial",
    ),
}

internal val guidedTourSteps = listOf(
    GuidedTourStep.SETUP_SOURCE, GuidedTourStep.SETUP_WINDOW, GuidedTourStep.SETUP_CREATE,
    GuidedTourStep.RALLIES_REVIEW, GuidedTourStep.RALLIES_TAG, GuidedTourStep.RALLIES_TIMELINE,
    GuidedTourStep.RALLIES_NAVIGATION, GuidedTourStep.RALLIES_ALL, GuidedTourStep.RALLIES_LIST,
    GuidedTourStep.RALLIES_SOURCE, GuidedTourStep.RALLIES_MISSING, GuidedTourStep.RALLIES_ADD,
    GuidedTourStep.EDITOR_OVERVIEW, GuidedTourStep.EDITOR_EXPORT, GuidedTourStep.EDITOR_SETTINGS,
)
private val rallyOnlyTourSteps = setOf(
    GuidedTourStep.RALLIES_TAG, GuidedTourStep.RALLIES_TIMELINE, GuidedTourStep.RALLIES_NAVIGATION,
)
internal val GuidedTourStep.workspace: GuidedTourWorkspace
    get() = when (this) {
        GuidedTourStep.RALLIES_LIST, GuidedTourStep.RALLIES_SOURCE, GuidedTourStep.RALLIES_MISSING -> GuidedTourWorkspace.RALLY_LIST
        GuidedTourStep.RALLIES_ADD -> GuidedTourWorkspace.ADD_RALLY
        GuidedTourStep.EDITOR_OVERVIEW -> GuidedTourWorkspace.EDITOR
        GuidedTourStep.EDITOR_EXPORT -> GuidedTourWorkspace.EXPORT
        else -> GuidedTourWorkspace.RALLY
    }
private val TourInk = Color(0xFF20201E)
private val TourOrange = Color(0xFFEF5B35)
private val TourMuted = Color(0xFF77736C)
private const val TOUR_PREFERENCES = "volleycut-guided-tour-v2"
private const val TOUR_STATE_KEY = "state"
private const val TOUR_DONE = "done"
private const val TOUR_DISMISSED = "dismissed"

internal fun nextGuidedTourStep(
    step: GuidedTourStep,
    scoreTrackingEnabled: Boolean = true,
    hasRallies: Boolean = true,
): GuidedTourStep? {
    val index = guidedTourSteps.indexOf(step)
    if (index < 0) return GuidedTourStep.RALLIES_REVIEW
    return guidedTourSteps.drop(index + 1).firstOrNull { hasRallies || it !in rallyOnlyTourSteps }
}

internal object GuidedTourStore {
    fun restart(context: Context, stage: GuidedTourStage) {
        write(
            context,
            if (stage == GuidedTourStage.SETUP) GuidedTourStep.SETUP_SOURCE.name
            else GuidedTourStep.RALLIES_REVIEW.name,
        )
    }

    fun moveToEditor(context: Context) {
        if (read(context) != TOUR_DISMISSED && read(context) != TOUR_DONE) {
            write(context, GuidedTourStep.RALLIES_REVIEW.name)
        }
    }

    private fun preferences(context: Context) =
        context.getSharedPreferences(TOUR_PREFERENCES, Context.MODE_PRIVATE)

    fun read(context: Context): String? {
        val prefs = preferences(context)
        val stored = prefs.getString(TOUR_STATE_KEY, null)
        // Resume an unfinished old Editor tour at the new rally workflow.
        if (prefs.getInt("flowVersion", 0) < 3 && stored?.startsWith("EDITOR_") == true) {
            write(context, GuidedTourStep.RALLIES_REVIEW.name)
            return GuidedTourStep.RALLIES_REVIEW.name
        }
        return stored
    }

    fun write(context: Context, value: String) {
        preferences(context).edit { putString(TOUR_STATE_KEY, value); putInt("flowVersion", 3) }
    }
}

internal class GuidedTourTargets {
    private val requesters = mutableMapOf<String, BringIntoViewRequester>()
    val bounds = mutableStateMapOf<String, Rect>()

    fun register(target: String, requester: BringIntoViewRequester) {
        requesters[target] = requester
        bounds.remove(target)
    }
    fun update(target: String, requester: BringIntoViewRequester, value: Rect) {
        if (requesters[target] === requester) bounds[target] = value
    }
    fun unregister(target: String, requester: BringIntoViewRequester) {
        if (requesters[target] === requester) {
            requesters.remove(target)
            bounds.remove(target)
        }
    }
    suspend fun reveal(target: String) { requesters[target]?.bringIntoView() }
}

internal fun Modifier.guidedTourTarget(target: String, targets: GuidedTourTargets?): Modifier {
    if (targets == null) return this
    return composed {
        val requester = remember(target, targets) { BringIntoViewRequester() }
        DisposableEffect(target, targets, requester) {
            targets.register(target, requester)
            onDispose { targets.unregister(target, requester) }
        }
        Modifier.bringIntoViewRequester(requester).onGloballyPositioned { coordinates ->
            targets.update(target, requester, coordinates.boundsInRoot())
        }
    }
}

@Composable
internal fun GuidedTour(
    stage: GuidedTourStage,
    targets: GuidedTourTargets,
    sourceReady: Boolean = false,
    scoreTrackingEnabled: Boolean = true,
    restartSignal: Int = 0,
    hasRallies: Boolean = true,
    onStepChanged: (GuidedTourStep?) -> Unit = {},
) {
    val context = LocalContext.current
    val screenHeightPx = LocalWindowInfo.current.containerSize.height.toFloat()
    val maximumCardHeight = (with(LocalDensity.current) { screenHeightPx.toDp() } - 64.dp).coerceAtLeast(120.dp)
    val stepChanged by rememberUpdatedState(onStepChanged)
    var state by remember(stage) {
        val stored = GuidedTourStore.read(context)
        val initial = when {
            stored == null -> if (stage == GuidedTourStage.SETUP) {
                GuidedTourStep.SETUP_SOURCE.name
            } else GuidedTourStep.RALLIES_REVIEW.name
            stage == GuidedTourStage.EDITOR && GuidedTourStep.entries.any {
                it.name == stored && it.stage == GuidedTourStage.SETUP
            } -> GuidedTourStep.RALLIES_REVIEW.name
            else -> stored
        }
        if (initial != stored) GuidedTourStore.write(context, initial)
        mutableStateOf(initial)
    }
    val storedStep = GuidedTourStep.entries.firstOrNull { it.name == state && it.stage == stage }
    val activeTourSteps = guidedTourSteps.filter {
        it.stage == stage && (hasRallies || it !in rallyOnlyTourSteps)
    }
    val step = if (storedStep != null && storedStep !in activeTourSteps) {
        nextGuidedTourStep(storedStep, scoreTrackingEnabled, hasRallies)?.also { nextTarget ->
            LaunchedEffect(state, hasRallies) {
                GuidedTourStore.write(context, nextTarget.name)
                state = nextTarget.name
            }
        }
    } else storedStep
    val visible = step != null

    fun save(value: String) {
        GuidedTourStore.write(context, value)
        state = value
    }

    fun dismiss() = save(TOUR_DISMISSED)

    fun restart() {
        GuidedTourStore.restart(context, stage)
        state = if (stage == GuidedTourStage.SETUP) GuidedTourStep.SETUP_SOURCE.name
        else GuidedTourStep.RALLIES_REVIEW.name
    }

    fun advance() {
        val current = step ?: return
        if (current == GuidedTourStep.SETUP_SOURCE && !sourceReady) return
        if (current == GuidedTourStep.SETUP_CREATE) return
        val next = nextGuidedTourStep(current, scoreTrackingEnabled, hasRallies)
        save(next?.name ?: TOUR_DONE)
    }

    BackHandler(enabled = visible, onBack = ::dismiss)

    LaunchedEffect(step) {
        stepChanged(step)
        val target = step?.target ?: return@LaunchedEffect
        delay(80)
        withTimeoutOrNull(2_000) {
            snapshotFlow { targets.bounds[target] }.first { it != null }
            targets.reveal(target)
        }
    }

    LaunchedEffect(restartSignal) {
        if (restartSignal > 0) restart()
    }

    if (!visible) return

    val current = checkNotNull(step)
    val targetBounds = targets.bounds[current.target]
    Box(Modifier.fillMaxSize()) {
        GuidedTourSpotlight(targetBounds)
        Card(
            modifier = Modifier
                .align(
                    if (targetBounds != null && targetBounds.center.y > screenHeightPx / 2f) {
                        Alignment.TopCenter
                    } else Alignment.BottomCenter,
                )
                .fillMaxWidth()
                .testTag("guided-tour-card")
                .safeDrawingPadding()
                .padding(horizontal = 12.dp, vertical = 16.dp),
            colors = CardDefaults.cardColors(containerColor = Color.White),
            shape = RoundedCornerShape(2.dp),
            elevation = CardDefaults.cardElevation(defaultElevation = 10.dp),
        ) {
            Column(
                Modifier
                    .fillMaxWidth()
                    .heightIn(max = maximumCardHeight)
                    .verticalScroll(rememberScrollState())
                    .border(1.dp, TourInk, RoundedCornerShape(2.dp))
                    .padding(horizontal = 18.dp, vertical = 15.dp),
                verticalArrangement = Arrangement.spacedBy(9.dp),
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        "TUTORIAL · ${activeTourSteps.indexOf(current) + 1} OF ${activeTourSteps.size}",
                        modifier = Modifier.weight(1f),
                        color = TourOrange,
                        fontFamily = FontFamily.Monospace,
                        fontSize = 10.sp,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = 1.sp,
                    )
                    TextButton(onClick = ::dismiss, modifier = Modifier.testTag("guided-tour-close")) { Text("Close", color = TourInk) }
                }
                Text(
                    current.title,
                    color = TourInk,
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold,
                    lineHeight = 25.sp,
                )
                Text(if (current == GuidedTourStep.RALLIES_TAG && !scoreTrackingEnabled)
                    "Choose Keep or Remove for this rally. Resolve every highlighted decision before using Next review. Enable score tracking in Editor if you want to review serving sides and scores."
                    else current.body, color = TourMuted, fontSize = 13.sp, lineHeight = 19.sp)
                Row(
                    Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(3.dp),
                ) {
                    activeTourSteps.forEachIndexed { index, _ ->
                        Box(
                            Modifier
                                .weight(1f)
                                .height(4.dp)
                                .background(
                                    if (index <= activeTourSteps.indexOf(current)) TourOrange
                                    else Color(0xFFD7D5CC),
                                ),
                        )
                    }
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    TextButton(onClick = ::dismiss) { Text("Skip tour", color = TourMuted) }
                    Spacer(Modifier.weight(1f))
                    Button(
                        modifier = Modifier.testTag("guided-tour-next"),
                        enabled = when (current) {
                            GuidedTourStep.SETUP_SOURCE -> sourceReady
                            GuidedTourStep.SETUP_CREATE -> false
                            else -> true
                        },
                        onClick = ::advance,
                    ) {
                        Text(
                            if (current == GuidedTourStep.SETUP_SOURCE && sourceReady) {
                                "Next"
                            } else if (
                                current == GuidedTourStep.EDITOR_SCORE_TOGGLE &&
                                !scoreTrackingEnabled
                            ) {
                                "Next"
                            } else current.action,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun GuidedTourSpotlight(target: Rect?) {
    Canvas(Modifier.fillMaxSize()) {
        if (target == null) {
            drawRect(Color(0x6B11130F))
            return@Canvas
        }
        val padding = 7.dp.toPx()
        val left = (target.left - padding).coerceIn(0f, size.width)
        val top = (target.top - padding).coerceIn(0f, size.height)
        val right = (target.right + padding).coerceIn(left, size.width)
        val bottom = (target.bottom + padding).coerceIn(top, size.height)
        val scrim = Color(0x6B11130F)
        drawRect(scrim, size = Size(size.width, top))
        drawRect(scrim, Offset(0f, bottom), Size(size.width, size.height - bottom))
        drawRect(scrim, Offset(0f, top), Size(left, bottom - top))
        drawRect(scrim, Offset(right, top), Size(size.width - right, bottom - top))
        drawRoundRect(
            color = TourOrange,
            topLeft = Offset(left, top),
            size = Size(right - left, bottom - top),
            cornerRadius = CornerRadius(5.dp.toPx()),
            style = Stroke(width = 2.dp.toPx()),
        )
    }
}
