package com.volleycut.nativeanalysis

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Checkbox
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import kotlin.math.abs

internal data class YouTubeChapterOptions(
    val includeRallyNumber: Boolean,
    val includeServeNumber: Boolean,
    val includeScore: Boolean,
    val includeServingTeam: Boolean,
    val includeSideSwitches: Boolean,
)

internal data class YouTubeChapter(
    val kind: Kind,
    val outputMs: Long,
    val sourceTimestampMs: Long,
    val title: String,
) {
    enum class Kind { RALLY, SIDE_SWITCH }
}

internal object YouTubeChapters {
    fun defaultOptions(hasScoreTracking: Boolean, hasSideSwitches: Boolean) = YouTubeChapterOptions(
        includeRallyNumber = !hasScoreTracking,
        includeServeNumber = false,
        includeScore = hasScoreTracking,
        includeServingTeam = hasScoreTracking,
        includeSideSwitches = hasSideSwitches,
    )

    fun build(
        intervals: List<FinalCutInterval>,
        cuts: List<EditableCut>,
        scoreTracking: ScoreTracking?,
        options: YouTubeChapterOptions,
    ): List<YouTubeChapter> {
        var outputStartMs = 0L
        val outputIntervals = intervals.map { interval ->
            OutputInterval(interval, outputStartMs).also {
                outputStartMs += (interval.endMs - interval.startMs).coerceAtLeast(0)
            }
        }
        val markers = scoreTracking?.serveMarkers
            ?.sortedWith(compareBy<ServeMarker> { it.timestampMs }.thenBy { it.id })
            .orEmpty()
        val visibleRallies = EditorMath.editableRallyGroups(cuts, intervals, markers).mapNotNull { rally ->
            val interval = outputIntervals.firstOrNull { output ->
                rally.cutIds.any { it in output.interval.cutIds }
            }
                ?: return@mapNotNull null
            val sourceTimestampMs = rally.keepStartMs.coerceIn(interval.interval.startMs, interval.interval.endMs)
            VisibleRally(
                rally,
                interval.outputStartMs + sourceTimestampMs - interval.interval.startMs,
                sourceTimestampMs,
            )
        }.sortedWith(
            compareBy<VisibleRally> { it.outputMs }
                .thenBy { it.rally.coreStartMs }
                .thenBy { it.rally.id },
        )
        val serveNumbers = markers.mapIndexed { index, marker -> marker.id to index + 1 }.toMap()
        val redoMarkerIds = markers.mapIndexedNotNull { index, marker ->
            marker.id.takeIf { markers.getOrNull(index + 1)?.ignorePreviousPoint == true }
        }.toSet()
        val claimedMarkerIds = mutableSetOf<String>()
        val rallyChapters = visibleRallies.mapIndexed { index, visible ->
            val marker = markerForRally(visible.rally, markers, claimedMarkerIds)
            if (marker != null) claimedMarkerIds += marker.id
            YouTubeChapter(
                YouTubeChapter.Kind.RALLY,
                visible.outputMs,
                visible.sourceTimestampMs,
                rallyTitle(
                    index + 1,
                    marker,
                    marker?.id in redoMarkerIds,
                    serveNumbers,
                    scoreTracking,
                    options,
                ),
            )
        }
        val switchChapters = if (options.includeSideSwitches && scoreTracking != null) {
            scoreTracking.sideSwitchMarkers.mapNotNull { marker ->
                outputPointAtOrAfter(marker.timestampMs, outputIntervals)
            }.mapIndexed { index, point ->
                YouTubeChapter(
                    YouTubeChapter.Kind.SIDE_SWITCH,
                    point.outputMs,
                    point.sourceTimestampMs,
                    "Side switch ${index + 1}",
                )
            }
        } else emptyList()

        return (rallyChapters + switchChapters)
            .sortedWith(
                compareBy<YouTubeChapter> { it.outputMs }
                    .thenBy { if (it.kind == YouTubeChapter.Kind.RALLY) 0 else 1 },
            )
            .fold(mutableListOf()) { merged, chapter ->
                val wholeSecondMs = (chapter.outputMs / 1_000L).coerceAtLeast(0) * 1_000L
                val previous = merged.lastOrNull()
                if (previous != null && previous.outputMs == wholeSecondMs) {
                    if (chapter.title !in previous.title.split(" / ")) {
                        merged[merged.lastIndex] = previous.copy(title = "${previous.title} / ${chapter.title}")
                    }
                } else {
                    merged += chapter.copy(outputMs = wholeSecondMs)
                }
                merged
            }
    }

    fun text(chapters: List<YouTubeChapter>): String = chapters.joinToString("\n") {
        "${formatTimestamp(it.outputMs)} ${it.title}"
    }

    fun filename(sourceFilename: String): String {
        val base = sourceFilename.substringBeforeLast('.', sourceFilename)
            .replace(Regex("[^A-Za-z0-9._-]+"), "-")
            .trim('-')
            .ifBlank { "volleysplice" }
        return "$base-youtube-chapters.txt"
    }

    fun formatTimestamp(milliseconds: Long): String {
        val totalSeconds = milliseconds.coerceAtLeast(0) / 1_000L
        val hours = totalSeconds / 3_600L
        val minutes = totalSeconds % 3_600L / 60L
        val seconds = totalSeconds % 60L
        return if (hours > 0) "%d:%02d:%02d".format(hours, minutes, seconds)
        else "%d:%02d".format(minutes, seconds)
    }

    private data class OutputInterval(val interval: FinalCutInterval, val outputStartMs: Long)
    private data class OutputPoint(val outputMs: Long, val sourceTimestampMs: Long)
    private data class VisibleRally(
        val rally: EditableRallyGroup,
        val outputMs: Long,
        val sourceTimestampMs: Long,
    )

    private fun outputPointAtOrAfter(sourceTimestampMs: Long, intervals: List<OutputInterval>): OutputPoint? {
        val containing = intervals.firstOrNull {
            sourceTimestampMs >= it.interval.startMs && sourceTimestampMs < it.interval.endMs
        }
        if (containing != null) return OutputPoint(
            containing.outputStartMs + sourceTimestampMs - containing.interval.startMs,
            sourceTimestampMs,
        )
        val next = intervals.firstOrNull { it.interval.startMs > sourceTimestampMs } ?: return null
        return OutputPoint(next.outputStartMs, next.interval.startMs)
    }

    private fun markerForRally(
        rally: EditableRallyGroup,
        markers: List<ServeMarker>,
        claimedMarkerIds: Set<String>,
    ): ServeMarker? {
        val available = markers.filter { it.id !in claimedMarkerIds }
        available.filter { it.rallyId in rally.cutIds }.minWithOrNull(
            compareBy<ServeMarker> { abs(it.timestampMs - rally.coreStartMs) }
                .thenBy { it.timestampMs }
                .thenBy { it.id },
        )?.let { return it }
        val insideCore = available.filter {
            it.timestampMs >= rally.coreStartMs && it.timestampMs < rally.coreEndMs
        }
        val insideKept = available.filter {
            it.timestampMs >= rally.keepStartMs && it.timestampMs < rally.keepEndMs
        }
        return (insideCore.ifEmpty { insideKept }).minWithOrNull(
            compareBy<ServeMarker> { abs(it.timestampMs - rally.coreStartMs) }
                .thenBy { it.timestampMs }
                .thenBy { it.id },
        )
    }

    private fun rallyTitle(
        rallyNumber: Int,
        marker: ServeMarker?,
        isRedo: Boolean,
        serveNumbers: Map<String, Int>,
        scoreTracking: ScoreTracking?,
        options: YouTubeChapterOptions,
    ): String = buildList {
        if (options.includeRallyNumber) add("Rally $rallyNumber")
        if (options.includeServeNumber && marker != null) serveNumbers[marker.id]?.let { add("Serve $it") }
        if (marker != null && scoreTracking != null && (options.includeScore || options.includeServingTeam)) {
            val score = ScoreReducer.deriveAt(scoreTracking, marker.timestampMs)
            if (options.includeScore) add("${score.team1Score}–${score.team2Score}")
            if (options.includeServingTeam) {
                when (score.servingTeamId) {
                    ScoreTeamId.TEAM_1 -> scoreTracking.team1Name
                    ScoreTeamId.TEAM_2 -> scoreTracking.team2Name
                    null -> null
                }?.let { add("$it serving") }
            }
        }
        if (isRedo) add("Re-do")
    }.ifEmpty { listOf("Rally $rallyNumber") }.joinToString(" - ")
}

@Composable
internal fun YouTubeChaptersDialog(
    sourceFilename: String,
    intervals: List<FinalCutInterval>,
    cuts: List<EditableCut>,
    scoreTracking: ScoreTracking?,
    hasSideSwitches: Boolean,
    status: String?,
    onClearStatus: () -> Unit,
    onCopy: (String) -> Unit,
    onSaveTextFile: (String, String) -> Unit,
    onDismiss: () -> Unit,
) {
    val hasScoreTracking = !scoreTracking?.serveMarkers.isNullOrEmpty()
    var options by remember(hasScoreTracking, hasSideSwitches) {
        mutableStateOf(YouTubeChapters.defaultOptions(hasScoreTracking, hasSideSwitches))
    }
    val chapters = YouTubeChapters.build(intervals, cuts, scoreTracking, options)
    val chapterText = YouTubeChapters.text(chapters)

    fun updateOptions(next: YouTubeChapterOptions) {
        options = next
        onClearStatus()
    }

    Dialog(onDismissRequest = onDismiss, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        Surface(
            modifier = Modifier.fillMaxWidth(.94f).heightIn(max = 780.dp).testTag("youtube-chapters-dialog"),
            shape = RoundedCornerShape(18.dp),
            color = Color(0xFFFFFCF7),
            shadowElevation = 12.dp,
        ) {
            Column(Modifier.padding(20.dp)) {
                Row(verticalAlignment = Alignment.Top) {
                    Column(Modifier.weight(1f)) {
                        Text("YOUTUBE EXPORT", color = Color(0xFFB3261E), fontSize = 12.sp, fontWeight = FontWeight.Bold)
                        Text("Create video chapters", fontSize = 24.sp, fontWeight = FontWeight.Bold)
                    }
                    TextButton(onClick = onDismiss) { Text("Close") }
                }
                Text(
                    "These timestamps match the final MP4 after removed footage is cut out. Choose what each chapter title should contain.",
                    color = Color(0xFF66615A),
                    fontSize = 13.sp,
                )
                Column(
                    Modifier.weight(1f, fill = false).verticalScroll(rememberScrollState()).padding(top = 14.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    Text("Chapter title format", fontWeight = FontWeight.Bold)
                    ChapterCheckbox("Rally number", "Rally 1, Rally 2, …", options.includeRallyNumber) {
                        updateOptions(options.copy(includeRallyNumber = it))
                    }
                    ChapterCheckbox("Serve number", "Uses your ordered serve markers.", options.includeServeNumber) {
                        updateOptions(options.copy(includeServeNumber = it))
                    }
                    if (hasScoreTracking) {
                        ChapterCheckbox("Score at the serve", "Shown in Team 1 – Team 2 order.", options.includeScore) {
                            updateOptions(options.copy(includeScore = it))
                        }
                        ChapterCheckbox("Serving team", "Shows the serving team by name.", options.includeServingTeam) {
                            updateOptions(options.copy(includeServingTeam = it))
                        }
                    }
                    if (hasSideSwitches) {
                        ChapterCheckbox(
                            "Side-switch chapters",
                            "Switches in removed footage attach to the next visible clip.",
                            options.includeSideSwitches,
                        ) { updateOptions(options.copy(includeSideSwitches = it)) }
                    }
                    Column(
                        Modifier.fillMaxWidth().background(Color(0xFFF1EEE7), RoundedCornerShape(10.dp))
                            .border(1.dp, Color(0xFFE0DCD3), RoundedCornerShape(10.dp)).padding(14.dp),
                    ) {
                        Row {
                            Text("Preview", fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
                            Text("${chapters.size} ${if (chapters.size == 1) "chapter" else "chapters"}", color = Color(0xFF66615A), fontSize = 12.sp)
                        }
                        Text(
                            chapterText.ifBlank { "No visible rallies are available for chapter export." },
                            modifier = Modifier.padding(top = 10.dp).testTag("youtube-chapters-preview"),
                            fontFamily = FontFamily.Monospace,
                            fontSize = 13.sp,
                        )
                        if (chapters.any { "Re-do" in it.title }) {
                            Text(
                                "“Re-do” means the following serve marker ignores the previous point; that point is not added to the score.",
                                modifier = Modifier.padding(top = 8.dp), color = Color(0xFF66615A), fontSize = 11.sp,
                            )
                        }
                    }
                }
                status?.let {
                    Text(it, modifier = Modifier.padding(top = 10.dp), color = Color(0xFF26734D), fontSize = 12.sp)
                }
                Row(
                    Modifier.fillMaxWidth().padding(top = 14.dp),
                    horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.End),
                ) {
                    TextButton(onClick = onDismiss) { Text("Cancel") }
                    OutlinedButton(
                        enabled = chapterText.isNotBlank(),
                        onClick = {
                            onSaveTextFile(chapterText, YouTubeChapters.filename(sourceFilename))
                        },
                        modifier = Modifier.testTag("youtube-chapters-save-file"),
                    ) { Text("Save text file") }
                    Button(
                        enabled = chapterText.isNotBlank(),
                        onClick = { onCopy(chapterText) },
                        modifier = Modifier.testTag("youtube-chapters-copy"),
                    ) { Text("Copy chapters") }
                }
            }
        }
    }
}

@Composable
private fun ChapterCheckbox(title: String, detail: String, checked: Boolean, onChecked: (Boolean) -> Unit) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Checkbox(checked = checked, onCheckedChange = onChecked)
        Column(Modifier.weight(1f)) {
            Text(title, fontWeight = FontWeight.SemiBold)
            Text(detail, color = Color(0xFF66615A), fontSize = 11.sp)
        }
    }
}
