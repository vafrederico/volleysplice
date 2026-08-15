package com.volleycut.nativeanalysis

import java.security.MessageDigest
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToLong

internal const val EDITOR_DRAFT_VERSION = 2
internal const val DEFAULT_BEFORE_PADDING_MS = 2_000L
internal const val DEFAULT_AFTER_PADDING_MS = 2_000L
internal const val DEFAULT_JOIN_GAP_MS = 3_000L
internal const val MAX_PADDING_MS = 10_000L
internal const val MAX_JOIN_GAP_MS = 10_000L
internal const val MIN_MARK_MS = 100L

internal enum class CutOrigin { INFERRED, MANUAL }

internal data class EditorSeed(
    val sourceUri: String,
    val displayName: String,
    val durationMs: Long,
    val width: Int,
    val height: Int,
    val rotation: Int,
    val ranges: List<SeedRange>,
) {
    val sourceRevision: String by lazy {
        val canonical = buildString {
            append(sourceUri).append('|').append(displayName).append('|').append(durationMs).append('|')
            append(width).append('x').append(height).append('@').append(rotation).append('|')
            ranges.forEach {
                append(it.startMs).append(':').append(it.endMs).append(':').append(it.confidence)
                    .append(':').append(it.agreement).append(';')
            }
        }
        MessageDigest.getInstance("SHA-256")
            .digest(canonical.toByteArray())
            .joinToString("") { "%02x".format(it) }
    }
}

internal data class SeedRange(
    val startMs: Long,
    val endMs: Long,
    val confidence: Float,
    val agreement: String? = null,
)

internal data class EditableCut(
    val id: String,
    val coreStartMs: Long,
    val coreEndMs: Long,
    val keepStartMs: Long,
    val keepEndMs: Long,
    val confidence: Float,
    val included: Boolean,
    val origin: CutOrigin,
    val agreement: String? = null,
)

internal data class IgnoredSourceInterval(
    val id: String,
    val startMs: Long,
    val endMs: Long,
    val reason: String,
)

internal data class EditorDraft(
    val version: Int = EDITOR_DRAFT_VERSION,
    val sourceRevision: String,
    val updatedAtMs: Long,
    val beforePaddingMs: Long = DEFAULT_BEFORE_PADDING_MS,
    val afterPaddingMs: Long = DEFAULT_AFTER_PADDING_MS,
    val joinGapMs: Long = DEFAULT_JOIN_GAP_MS,
    val pendingManualStartMs: Long? = null,
    val pendingIgnoreStartMs: Long? = null,
    val ignoreReason: String = "non-game-content",
    val finalPreviewEnabled: Boolean = false,
    val playbackRate: Float = 1f,
    val confidenceReviewThreshold: Float = .7f,
    val cuts: List<EditableCut>,
    val ignoredIntervals: List<IgnoredSourceInterval> = emptyList(),
)

internal data class JoinedGap(val startMs: Long, val endMs: Long)

internal data class FinalCutInterval(
    val startMs: Long,
    val endMs: Long,
    val cutIds: List<String>,
    val joinedGaps: List<JoinedGap> = emptyList(),
)

internal data class DetailWindow(val startMs: Long, val endMs: Long)

internal object EditorMath {
    fun newDraft(seed: EditorSeed): EditorDraft = EditorDraft(
        sourceRevision = seed.sourceRevision,
        updatedAtMs = 0,
        cuts = seed.ranges.mapNotNull { range ->
            val coreStart = range.startMs.coerceIn(0, seed.durationMs)
            val coreEnd = range.endMs.coerceIn(0, seed.durationMs)
            if (coreEnd <= coreStart) return@mapNotNull null
            EditableCut(
                id = "",
                coreStartMs = coreStart,
                coreEndMs = coreEnd,
                keepStartMs = (coreStart - DEFAULT_BEFORE_PADDING_MS).coerceAtLeast(0),
                keepEndMs = (coreEnd + DEFAULT_AFTER_PADDING_MS).coerceAtMost(seed.durationMs),
                confidence = range.confidence.coerceIn(0f, 1f),
                included = true,
                origin = CutOrigin.INFERRED,
                agreement = range.agreement,
            )
        }.mapIndexed { index, cut -> cut.copy(id = "R${(index + 1).toString().padStart(3, '0')}") },
    )

    fun applyPadding(
        draft: EditorDraft,
        beforeMs: Long,
        afterMs: Long,
        durationMs: Long,
    ): EditorDraft {
        val before = beforeMs.coerceIn(0, MAX_PADDING_MS)
        val after = afterMs.coerceIn(0, MAX_PADDING_MS)
        return draft.copy(
            beforePaddingMs = before,
            afterPaddingMs = after,
            cuts = draft.cuts.map { cut ->
                if (cut.origin == CutOrigin.MANUAL) cut else cut.copy(
                    keepStartMs = (cut.coreStartMs - before).coerceAtLeast(0),
                    keepEndMs = (cut.coreEndMs + after).coerceAtMost(durationMs),
                )
            },
        )
    }

    fun finalIntervals(draft: EditorDraft): List<FinalCutInterval> {
        val joinGapMs = draft.joinGapMs.coerceIn(0, MAX_JOIN_GAP_MS)
        val merged = mutableListOf<FinalCutInterval>()
        draft.cuts.asSequence()
            .filter { it.included && it.keepEndMs > it.keepStartMs }
            .sortedWith(compareBy<EditableCut> { it.keepStartMs }.thenBy { it.keepEndMs })
            .forEach { cut ->
                val previous = merged.lastOrNull()
                val gap = previous?.let { cut.keepStartMs - it.endMs } ?: Long.MAX_VALUE
                if (previous == null || (gap > 0 && gap >= joinGapMs)) {
                    merged += FinalCutInterval(cut.keepStartMs, cut.keepEndMs, listOf(cut.id))
                } else {
                    merged[merged.lastIndex] = previous.copy(
                        endMs = max(previous.endMs, cut.keepEndMs),
                        cutIds = previous.cutIds + cut.id,
                        joinedGaps = if (gap > 0) {
                            previous.joinedGaps + JoinedGap(previous.endMs, cut.keepStartMs)
                        } else previous.joinedGaps,
                    )
                }
            }

        var remaining = merged.toList()
        for (ignored in mergeIgnored(draft.ignoredIntervals)) {
            remaining = remaining.flatMap { interval ->
                if (ignored.endMs <= interval.startMs || ignored.startMs >= interval.endMs) {
                    listOf(interval)
                } else {
                    buildList {
                        if (ignored.startMs > interval.startMs) {
                            add(interval.copy(endMs = min(interval.endMs, ignored.startMs)))
                        }
                        if (ignored.endMs < interval.endMs) {
                            add(interval.copy(startMs = max(interval.startMs, ignored.endMs)))
                        }
                    }
                }
            }
        }
        val byId = draft.cuts.associateBy { it.id }
        return remaining.map { interval ->
            interval.copy(
                cutIds = interval.cutIds.filter { id ->
                    byId[id]?.let { cut ->
                        min(cut.keepEndMs, interval.endMs) - max(cut.keepStartMs, interval.startMs) > 0
                    } == true
                },
                joinedGaps = interval.joinedGaps.mapNotNull { gap ->
                    val start = max(interval.startMs, gap.startMs)
                    val end = min(interval.endMs, gap.endMs)
                    if (end > start) JoinedGap(start, end) else null
                },
            )
        }
    }

    fun effectiveKeptIds(draft: EditorDraft): Set<String> {
        val ignored = mergeIgnored(draft.ignoredIntervals)
        return draft.cuts.asSequence()
            .filter { it.included && it.keepEndMs > it.keepStartMs }
            .filter { cut ->
                val ignoredMs = ignored.sumOf { interval ->
                    max(0, min(cut.keepEndMs, interval.endMs) - max(cut.keepStartMs, interval.startMs))
                }
                cut.keepEndMs - cut.keepStartMs - ignoredMs > 0
            }
            .map { it.id }
            .toSet()
    }

    fun totalFinalMs(intervals: List<FinalCutInterval>): Long =
        intervals.sumOf { it.endMs - it.startMs }

    fun nextFinalTime(intervals: List<FinalCutInterval>, positionMs: Long): Long? {
        for (interval in intervals) {
            if (positionMs in interval.startMs until interval.endMs) return positionMs
            if (positionMs < interval.startMs) return interval.startMs
        }
        return null
    }

    fun playbackFocusCut(cuts: List<EditableCut>, positionMs: Long): EditableCut? =
        cuts.filter { it.keepStartMs <= positionMs }.maxByOrNull { it.keepStartMs }

    fun detailWindow(cut: EditableCut?, positionMs: Long, durationMs: Long): DetailWindow {
        val minimumSpan = min(24_000L, if (durationMs > 0) durationMs else 24_000L)
        val center = cut?.let { (it.keepStartMs + it.keepEndMs) / 2 } ?: positionMs
        val contentSpan = cut?.let { it.keepEndMs - it.keepStartMs + 10_000L } ?: minimumSpan
        val span = min(if (durationMs > 0) durationMs else minimumSpan, max(minimumSpan, contentSpan))
        var start = max(0, center - span / 2)
        var end = min(durationMs, start + span)
        start = max(0, end - span)
        if (end <= start) end = (start + 1).coerceAtMost(durationMs.coerceAtLeast(1))
        return DetailWindow(start, end)
    }

    fun nextId(prefix: String, ids: Collection<String>): String {
        val used = ids.toHashSet()
        for (index in 1 until 10_000) {
            val candidate = prefix + index.toString().padStart(3, '0')
            if (candidate !in used) return candidate
        }
        return prefix + System.currentTimeMillis()
    }

    private fun mergeIgnored(intervals: List<IgnoredSourceInterval>): List<IgnoredSourceInterval> {
        val merged = mutableListOf<IgnoredSourceInterval>()
        intervals.sortedWith(compareBy<IgnoredSourceInterval> { it.startMs }.thenBy { it.endMs })
            .forEach { interval ->
                val previous = merged.lastOrNull()
                if (previous == null || interval.startMs > previous.endMs) {
                    merged += interval
                } else {
                    merged[merged.lastIndex] = previous.copy(endMs = max(previous.endMs, interval.endMs))
                }
            }
        return merged
    }
}

internal fun secondsToMs(seconds: Double): Long = (seconds * 1_000.0).roundToLong()

internal fun preciseTime(milliseconds: Long): String {
    val safe = milliseconds.coerceAtLeast(0)
    val totalSeconds = safe / 1_000.0
    val hours = (totalSeconds / 3_600).toInt()
    val minutes = ((totalSeconds % 3_600) / 60).toInt()
    val remainder = totalSeconds % 60
    val prefix = if (hours > 0) "%d:%02d".format(hours, minutes) else minutes.toString()
    return "%s:%04.1f".format(prefix, remainder)
}

internal fun compactTime(milliseconds: Long): String {
    val totalSeconds = milliseconds.coerceAtLeast(0) / 1_000
    val hours = totalSeconds / 3_600
    val minutes = (totalSeconds % 3_600) / 60
    val seconds = totalSeconds % 60
    return if (hours > 0) "%d:%02d:%02d".format(hours, minutes, seconds)
    else "%d:%02d".format(minutes, seconds)
}
