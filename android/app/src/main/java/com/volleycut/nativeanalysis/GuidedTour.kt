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
        "Select a volleyball video",
        "Choose one recording from this device. VolleyCut keeps the file local, reads its metadata, and prepares the controls you need before inference.",
        "Choose video first",
    ),
    SETUP_WINDOW(
        GuidedTourStage.SETUP,
        "setup-window",
        "Choose the game start and end",
        "Play or seek the preview, then copy the playhead into the start and end boundaries. Only this window is analyzed and shown in the editor.",
        "Next: create project",
    ),
    SETUP_CREATE(
        GuidedTourStage.SETUP,
        "setup-create",
        "Create the project and queue inference",
        "When the game window looks right, tap Create & queue. Analysis continues on this device, and the editor opens as soon as the project is ready.",
        "Create project first",
    ),
    EDITOR_HEADER(
        GuidedTourStage.EDITOR,
        "editor-header",
        "Navigate projects from the header",
        "Use the project menu to switch recordings or start another one. The header also shows analysis and export queue activity and keeps delete controls with the selected project.",
        "Next: output settings",
    ),
    EDITOR_OUTPUT(
        GuidedTourStage.EDITOR,
        "editor-output",
        "Build the final edit",
        "This panel controls which footage remains in the result. It combines suppression, padding, gap joining, and final-cut playback into one saved edit.",
        "Next: suppression",
    ),
    EDITOR_SUPPRESSION(
        GuidedTourStage.EDITOR,
        "editor-suppression",
        "Review suppression suggestions",
        "Suppression levels identify likely false positives. Suggestions can begin highlighted or disabled, and you can keep or suppress each one while reviewing the timeline.",
        "Next: padding",
    ),
    EDITOR_PADDING(
        GuidedTourStage.EDITOR,
        "editor-padding",
        "Add context around rallies",
        "Before and After padding extend inferred rallies without changing their core boundaries. The padded edges are used by preview and export.",
        "Next: gap joining",
    ),
    EDITOR_JOIN_GAPS(
        GuidedTourStage.EDITOR,
        "editor-join-gaps",
        "Join nearby cuts",
        "Light-gray gaps shorter than this threshold stay in the output so nearby rallies export as one continuous range. Set it to zero to keep every gap as a cut.",
        "Next: final-cut preview",
    ),
    EDITOR_FINAL_PREVIEW(
        GuidedTourStage.EDITOR,
        "editor-final-preview",
        "Preview only the final cut",
        "Turn this on to skip removed rallies, ignored footage, and gaps that are not being joined while the source plays.",
        "Next: video player",
    ),
    EDITOR_VIDEO(
        GuidedTourStage.EDITOR,
        "editor-video",
        "Watch the source video",
        "The original recording stays on this device. Playback is constrained to the game window so you can compare each model range with the footage.",
        "Next: transport controls",
    ),
    EDITOR_TRANSPORT(
        GuidedTourStage.EDITOR,
        "editor-transport",
        "Move precisely through the video",
        "Play or pause, nudge by one second or one tenth of a second, and change playback speed without changing any saved boundary.",
        "Next: game window",
    ),
    EDITOR_OVERVIEW(
        GuidedTourStage.EDITOR,
        "editor-overview",
        "Read the game-window overview",
        "The two rails show the first and second halves of the game. Tap a range to focus it, and use the review controls to find disagreements and low-confidence cuts.",
        "Next: focused range",
    ),
    EDITOR_FOCUS(
        GuidedTourStage.EDITOR,
        "editor-focus",
        "Refine the focused range",
        "This enlarged timeline follows the selected rally. Move between cuts, keep or remove them, preview one range, and adjust its exact output edges.",
        "Next: trim and split",
    ),
    EDITOR_TRIM(
        GuidedTourStage.EDITOR,
        "editor-trim",
        "Change or split a rally",
        "Drag the rally handles or set either edge at the playhead for frame-accurate corrections. Split at the playhead when one prediction contains two rallies.",
        "Next: marking tools",
    ),
    EDITOR_MARKING(
        GuidedTourStage.EDITOR,
        "editor-marking",
        "Add misses or ignore unusable footage",
        "Mark a missed rally from start to end, or mark camera gaps and non-game footage as ignored. Ignored time is excluded rather than treated as a negative label.",
        "Next: all cuts",
    ),
    EDITOR_CUTS(
        GuidedTourStage.EDITOR,
        "editor-cuts",
        "Use the all-cuts list",
        "Every inferred or manual range appears here. Tap a row to focus it, then toggle whether it contributes to the final edit.",
        "Next: export",
    ),
    EDITOR_EXPORT(
        GuidedTourStage.EDITOR,
        "editor-export",
        "Export the result",
        "Queue the edited MP4, save the exact edit list, or export model feedback containing features and corrections without including video bytes.",
        "Finish tour",
    ),
}

private val guidedTourSteps = GuidedTourStep.entries
private val TourInk = Color(0xFF20201E)
private val TourOrange = Color(0xFFEF5B35)
private val TourMuted = Color(0xFF77736C)
private const val TOUR_PREFERENCES = "volleycut-guided-tour-v1"
private const val TOUR_STATE_KEY = "state"
private const val TOUR_DONE = "done"
private const val TOUR_DISMISSED = "dismissed"

internal fun nextGuidedTourStep(step: GuidedTourStep): GuidedTourStep? =
    guidedTourSteps.getOrNull(guidedTourSteps.indexOf(step) + 1)

internal object GuidedTourStore {
    fun restart(context: Context, stage: GuidedTourStage) {
        write(
            context,
            if (stage == GuidedTourStage.SETUP) GuidedTourStep.SETUP_SOURCE.name
            else GuidedTourStep.EDITOR_HEADER.name,
        )
    }

    fun moveToEditor(context: Context) {
        if (read(context) != TOUR_DISMISSED && read(context) != TOUR_DONE) {
            write(context, GuidedTourStep.EDITOR_HEADER.name)
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
    restartSignal: Int = 0,
) {
    val context = LocalContext.current
    val screenHeightPx = LocalWindowInfo.current.containerSize.height.toFloat()
    var state by remember(stage) {
        val stored = GuidedTourStore.read(context)
        val initial = when {
            stored == null -> if (stage == GuidedTourStage.SETUP) {
                GuidedTourStep.SETUP_SOURCE.name
            } else GuidedTourStep.EDITOR_HEADER.name
            stage == GuidedTourStage.EDITOR && GuidedTourStep.entries.any {
                it.name == stored && it.stage == GuidedTourStage.SETUP
            } -> GuidedTourStep.EDITOR_HEADER.name
            else -> stored
        }
        if (initial != stored) GuidedTourStore.write(context, initial)
        mutableStateOf(initial)
    }
    val step = GuidedTourStep.entries.firstOrNull { it.name == state && it.stage == stage }
    val visible = step != null

    fun save(value: String) {
        GuidedTourStore.write(context, value)
        state = value
    }

    fun dismiss() = save(TOUR_DISMISSED)

    fun restart() {
        GuidedTourStore.restart(context, stage)
        state = if (stage == GuidedTourStage.SETUP) GuidedTourStep.SETUP_SOURCE.name
        else GuidedTourStep.EDITOR_HEADER.name
    }

    fun advance() {
        val current = step ?: return
        if (current == GuidedTourStep.SETUP_SOURCE && !sourceReady) return
        if (current == GuidedTourStep.SETUP_CREATE) return
        val next = nextGuidedTourStep(current)
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
                        "WELCOME TOUR · ${guidedTourSteps.indexOf(current) + 1} OF ${guidedTourSteps.size}",
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
                    guidedTourSteps.forEachIndexed { index, _ ->
                        Box(
                            Modifier
                                .weight(1f)
                                .height(4.dp)
                                .background(
                                    if (index <= guidedTourSteps.indexOf(current)) TourOrange
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
                                "Next: game window"
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
