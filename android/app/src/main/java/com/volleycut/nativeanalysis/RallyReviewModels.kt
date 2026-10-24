package com.volleycut.nativeanalysis

/** One review stop can contain a clip decision, cleanup decisions and several serves. */
internal data class RallyReviewItem(
    val key: String,
    val rallyId: String?,
    val startMs: Long,
    val endMs: Long,
    val clipReview: Boolean = false,
    val cleanupIds: Set<String> = emptySet(),
    val serveIds: Set<String> = emptySet(),
)

internal data class RallyReviewCleanup(val id: String, val startMs: Long, val endMs: Long)

/** Group related decisions without dropping cleanup/serve evidence outside detected rallies. */
internal fun rallyReviewQueue(
    rallies: List<EditableRallyGroup>,
    pendingClipIds: Set<String>,
    cleanup: List<RallyReviewCleanup>,
    serves: List<ServeMarker>,
    ignored: List<IgnoredSourceInterval>,
): List<RallyReviewItem> {
    val items = linkedMapOf<String, RallyReviewItem>()
    fun add(rally: EditableRallyGroup?, item: RallyReviewItem) {
        val key = rally?.let { "rally-${it.id}" } ?: item.key
        val existing = items[key]
        items[key] = item.copy(
            key = key,
            rallyId = rally?.id,
            startMs = minOf(existing?.startMs ?: Long.MAX_VALUE, rally?.keepStartMs ?: item.startMs, item.startMs),
            endMs = maxOf(existing?.endMs ?: 0, rally?.keepEndMs ?: item.endMs, item.endMs),
            clipReview = existing?.clipReview == true || item.clipReview,
            cleanupIds = existing?.cleanupIds.orEmpty() + item.cleanupIds,
            serveIds = existing?.serveIds.orEmpty() + item.serveIds,
        )
    }
    rallies.filter { it.id in pendingClipIds }.forEach { rally ->
        if (EditorMath.firstReviewableTime(rally.keepStartMs, rally.keepEndMs, ignored) != null) {
            add(rally, RallyReviewItem("", rally.id, rally.keepStartMs, rally.keepEndMs, clipReview = true))
        }
    }
    cleanup.forEach { suggestion ->
        if (EditorMath.firstReviewableTime(suggestion.startMs, suggestion.endMs, ignored) == null) return@forEach
        val rally = rallies.filter { it.coreStartMs < suggestion.endMs && suggestion.startMs < it.coreEndMs }
            .maxByOrNull { minOf(it.coreEndMs, suggestion.endMs) - maxOf(it.coreStartMs, suggestion.startMs) }
        add(rally, RallyReviewItem(
            "cleanup-${suggestion.id}", rally?.id, suggestion.startMs, suggestion.endMs,
            cleanupIds = setOf(suggestion.id),
        ))
    }
    serves.filter { it.side == ServingSide.REVIEW && !EditorMath.isTimestampIgnored(it.timestampMs, ignored) }
        .forEach { marker ->
            // Identity wins over overlapping padding from adjacent rallies.
            val rally = rallies.firstOrNull { marker.rallyId in it.cutIds }
                ?: rallies.firstOrNull { marker.timestampMs >= it.coreStartMs && marker.timestampMs < it.coreEndMs }
                ?: rallies.firstOrNull { marker.timestampMs >= it.keepStartMs && marker.timestampMs < it.keepEndMs }
            add(rally, RallyReviewItem(
                "serve-${marker.id}", rally?.id, marker.timestampMs, marker.timestampMs + 2_000,
                serveIds = setOf(marker.id),
            ))
        }
    return items.values.sortedWith(compareBy<RallyReviewItem> { it.startMs }.thenBy { it.key })
}

internal fun rallyAtTime(rallies: List<EditableRallyGroup>, timestampMs: Long): EditableRallyGroup? =
    rallies.filter { timestampMs >= it.keepStartMs && timestampMs < it.keepEndMs }
        .minByOrNull { kotlin.math.abs(timestampMs - it.coreStartMs) }

internal fun rallyServeMarkers(rally: EditableRallyGroup, markers: List<ServeMarker>): List<ServeMarker> =
    markers.filter { marker ->
        marker.rallyId in rally.cutIds ||
            (marker.rallyId == null && marker.timestampMs >= rally.coreStartMs && marker.timestampMs < rally.coreEndMs)
    }.sortedBy { it.timestampMs }

internal const val RALLY_MIN_DURATION_MS = 500L

internal data class RallyTimelineContext(
    val previous: EditableRallyGroup?,
    val next: EditableRallyGroup?,
    val window: DetailWindow,
)

/** Keep the selected rally editable while showing its nearest retained neighbours. */
internal fun rallyTimelineContext(
    rallies: List<EditableRallyGroup>, selected: EditableRallyGroup, effectiveIds: Set<String>,
    sourceStartMs: Long, sourceEndMs: Long,
): RallyTimelineContext {
    val ordered = (rallies.filter { it.id != selected.id } + selected)
        .sortedWith(compareBy<EditableRallyGroup> { it.coreStartMs }.thenBy { it.coreEndMs }.thenBy { it.id })
    val index = ordered.indexOfFirst { it.id == selected.id }
    fun retained(rally: EditableRallyGroup) =
        rally.keepEndMs > sourceStartMs && rally.keepStartMs < sourceEndMs &&
            rally.cuts.any { it.included && it.id in effectiveIds }
    val previous = ordered.take(index).lastOrNull(::retained)
    val next = ordered.drop(index + 1).firstOrNull(::retained)
    return RallyTimelineContext(previous, next, DetailWindow(
        maxOf(sourceStartMs, minOf(selected.coreStartMs - 8_000, selected.keepStartMs,
            previous?.let { it.keepStartMs - 2_000 } ?: Long.MAX_VALUE)),
        minOf(sourceEndMs, maxOf(selected.coreEndMs + 8_000, selected.keepEndMs,
            next?.let { it.keepEndMs + 2_000 } ?: Long.MIN_VALUE)),
    ))
}

internal fun rallyBoundaryValue(
    side: String, valueMs: Long, startMs: Long, endMs: Long, sourceStartMs: Long, sourceEndMs: Long,
): Long = if (side == "start") {
    valueMs.coerceIn(sourceStartMs, (endMs - RALLY_MIN_DURATION_MS).coerceAtLeast(sourceStartMs))
} else {
    valueMs.coerceIn((startMs + RALLY_MIN_DURATION_MS).coerceAtMost(sourceEndMs), sourceEndMs)
}

/** Distance to an interval, so a gap browses its chronological neighbour without selecting it. */
internal fun nearestRallyIndex(rallies: List<EditableRallyGroup>, timestampMs: Long): Int =
    rallies.indices.minByOrNull { index ->
        val rally = rallies[index]
        maxOf(rally.keepStartMs - timestampMs, timestampMs - rally.keepEndMs, 0L)
    } ?: -1

internal data class MissingRallyRange(val startMs: Long, val endMs: Long)

/** Never create a rally over existing footage or a deliberately ignored interval. */
internal fun missingRallyRange(
    timestampMs: Long,
    sourceStartMs: Long,
    sourceEndMs: Long,
    occupied: List<Pair<Long, Long>>,
): MissingRallyRange? {
    if (timestampMs !in sourceStartMs..sourceEndMs) return null
    if (occupied.any { (start, end) -> timestampMs >= start && timestampMs < end }) return null
    val left = occupied.filter { it.second <= timestampMs }.maxOfOrNull { it.second }
        ?.coerceAtLeast(sourceStartMs) ?: sourceStartMs
    val right = occupied.filter { it.first > timestampMs }.minOfOrNull { it.first }
        ?.coerceAtMost(sourceEndMs) ?: sourceEndMs
    if (right - left < RALLY_MIN_DURATION_MS) return null
    var start = maxOf(left, timestampMs - 3_000)
    var end = minOf(right, timestampMs + 5_000)
    if (end - start < RALLY_MIN_DURATION_MS) {
        start = maxOf(left, end - RALLY_MIN_DURATION_MS)
        end = minOf(right, start + RALLY_MIN_DURATION_MS)
    }
    return MissingRallyRange(start, end).takeIf { end - start >= RALLY_MIN_DURATION_MS }
}
