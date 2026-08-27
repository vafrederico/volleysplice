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
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.relocation.BringIntoViewRequester
import androidx.compose.foundation.relocation.bringIntoViewRequester
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.boundsInRoot
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalWindowInfo
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.edit
import kotlinx.coroutines.delay

internal enum class GuidedTourStage {
    SETUP,
    EDITOR,
}

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
        "Let VolleyCut find the rallies",
        "Tap Find rallies. Keep VolleyCut open while it prepares the suggested clips; the editor opens when they are ready.",
        "Find rallies first",
    ),
    EDITOR_SETTINGS(
        GuidedTourStage.EDITOR,
        "editor-settings",
        "Fine-tune only when you need to",
        "The Settings gear contains optional automatic cleanup, extra time around clips, short-break joining, and review sensitivity. The defaults are ready for most games.",
        "Next",
    ),
    EDITOR_VIDEO(
        GuidedTourStage.EDITOR,
        "editor-video",
        "Watch the video",
        "Play, pause, or move to any moment. Turn on Play only the final video above to skip every part that will not be saved.",
        "Next",
    ),
    EDITOR_OVERVIEW(
        GuidedTourStage.EDITOR,
        "editor-overview",
        "Review the suggested clips",
        "Each block is a clip planned for the final video. Start with Check next clip, then review anything else only if it looks wrong.",
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
        "Add anything VolleyCut missed",
        "For a missed rally, mark its start and end. Leave out a section for camera gaps, breaks, or other footage that should not appear in the final video.",
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
        "Next: save video",
    ),
    EDITOR_EXPORT(
        GuidedTourStage.EDITOR,
        "editor-export",
        "Save the finished video",
        "Tap Save final video when the review looks right. Choose the original video first if VolleyCut asks you to reconnect it.",
        "Finish tutorial",
    ),
}

private val guidedTourSteps = GuidedTourStep.entries
private val scoreGuidedTourSteps = setOf(
    GuidedTourStep.EDITOR_SCORE_PANEL,
)
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
): GuidedTourStep? {
    val steps = if (scoreTrackingEnabled) guidedTourSteps else guidedTourSteps.filterNot {
        it in scoreGuidedTourSteps
    }
    return steps.getOrNull(steps.indexOf(step) + 1)
}

internal object GuidedTourStore {
    fun restart(context: Context, stage: GuidedTourStage) {
        write(
            context,
            if (stage == GuidedTourStage.SETUP) GuidedTourStep.SETUP_SOURCE.name
            else GuidedTourStep.EDITOR_SETTINGS.name,
        )
    }

    fun moveToEditor(context: Context) {
        if (read(context) != TOUR_DISMISSED && read(context) != TOUR_DONE) {
            write(context, GuidedTourStep.EDITOR_SETTINGS.name)
        }
    }

    private fun preferences(context: Context) =
        context.getSharedPreferences(TOUR_PREFERENCES, Context.MODE_PRIVATE)

    fun read(context: Context): String? = preferences(context).getString(TOUR_STATE_KEY, null)

    fun write(context: Context, value: String) {
        preferences(context).edit { putString(TOUR_STATE_KEY, value) }
    }
}

internal class GuidedTourTargets {
    private val requesters = mutableMapOf<String, BringIntoViewRequester>()
    val bounds = mutableStateMapOf<String, Rect>()

    fun requester(target: String): BringIntoViewRequester =
        requesters.getOrPut(target) { BringIntoViewRequester() }

    suspend fun reveal(target: String) {
        requesters[target]?.bringIntoView()
    }
}

internal fun Modifier.guidedTourTarget(
    target: String,
    targets: GuidedTourTargets?,
): Modifier {
    if (targets == null) return this
    return bringIntoViewRequester(targets.requester(target))
        .onGloballyPositioned { coordinates ->
            targets.bounds[target] = coordinates.boundsInRoot()
        }
}

@Composable
internal fun GuidedTour(
    stage: GuidedTourStage,
    targets: GuidedTourTargets,
    sourceReady: Boolean = false,
    scoreTrackingEnabled: Boolean = true,
    restartSignal: Int = 0,
) {
    val context = LocalContext.current
    val screenHeightPx = LocalWindowInfo.current.containerSize.height.toFloat()
    var state by remember(stage) {
        val stored = GuidedTourStore.read(context)
        val initial = when {
            stored == null -> if (stage == GuidedTourStage.SETUP) {
                GuidedTourStep.SETUP_SOURCE.name
            } else GuidedTourStep.EDITOR_SETTINGS.name
            stage == GuidedTourStage.EDITOR && GuidedTourStep.entries.any {
                it.name == stored && it.stage == GuidedTourStage.SETUP
            } -> GuidedTourStep.EDITOR_SETTINGS.name
            else -> stored
        }
        if (initial != stored) GuidedTourStore.write(context, initial)
        mutableStateOf(initial)
    }
    val storedStep = GuidedTourStep.entries.firstOrNull { it.name == state && it.stage == stage }
    val activeTourSteps = (if (scoreTrackingEnabled) guidedTourSteps else guidedTourSteps.filterNot {
        it in scoreGuidedTourSteps
    }).filter { it.stage == stage }
    val step = if (!scoreTrackingEnabled && storedStep in scoreGuidedTourSteps) {
        GuidedTourStep.EDITOR_EXPORT.also { saveTarget ->
            LaunchedEffect(state) {
                GuidedTourStore.write(context, saveTarget.name)
                state = saveTarget.name
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
        else GuidedTourStep.EDITOR_SETTINGS.name
    }

    fun advance() {
        val current = step ?: return
        if (current == GuidedTourStep.SETUP_SOURCE && !sourceReady) return
        if (current == GuidedTourStep.SETUP_CREATE) return
        val next = nextGuidedTourStep(current, scoreTrackingEnabled)
        save(next?.name ?: TOUR_DONE)
    }

    BackHandler(enabled = visible, onBack = ::dismiss)

    LaunchedEffect(step?.target) {
        val target = step?.target ?: return@LaunchedEffect
        delay(80)
        targets.reveal(target)
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
                .safeDrawingPadding()
                .padding(horizontal = 12.dp, vertical = 16.dp),
            colors = CardDefaults.cardColors(containerColor = Color.White),
            shape = RoundedCornerShape(2.dp),
            elevation = CardDefaults.cardElevation(defaultElevation = 10.dp),
        ) {
            Column(
                Modifier
                    .fillMaxWidth()
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
                    TextButton(onClick = ::dismiss) { Text("Close", color = TourInk) }
                }
                Text(
                    current.title,
                    color = TourInk,
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold,
                    lineHeight = 25.sp,
                )
                Text(current.body, color = TourMuted, fontSize = 13.sp, lineHeight = 19.sp)
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
