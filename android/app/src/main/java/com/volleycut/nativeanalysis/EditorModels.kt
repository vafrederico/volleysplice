package com.volleycut.nativeanalysis

import java.security.MessageDigest
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToLong

internal const val EDITOR_DRAFT_VERSION = 8
internal const val DEFAULT_BEFORE_PADDING_MS = 2_000L
internal const val DEFAULT_AFTER_PADDING_MS = 2_000L
internal const val DEFAULT_JOIN_GAP_MS = 3_000L
internal const val MAX_PADDING_MS = 10_000L
internal const val MAX_JOIN_GAP_MS = 10_000L
internal const val MIN_MARK_MS = 100L

internal enum class CutOrigin { INFERRED, MANUAL }
internal enum class SuppressionDecision(val wireName: String) {
    KEEP("keep"), SUPPRESS("suppress");

    companion object {
        fun fromWireName(value: String) = entries.firstOrNull { it.wireName == value }
    }
}

internal enum class SuppressionInitialBehavior(val wireName: String, val label: String) {
    HIGHLIGHT_ONLY("highlight-only", "Highlight only"),
    DISABLE_INITIALLY("disable-initially", "Disable initially");

    companion object {
        fun fromWireName(value: String) = entries.firstOrNull { it.wireName == value }
            ?: DISABLE_INITIALLY
    }
}

internal enum class SuppressionScope(val wireName: String, val label: String) {
    WHOLE_RALLY("whole-rally", "Whole clip"),
    VETO_REGION("veto-region", "Only this part");

    companion object {
        fun fromWireName(value: String) = entries.firstOrNull { it.wireName == value }
            ?: WHOLE_RALLY
    }
}

internal data class EditorSeed(
    val sourceUri: String,
    val displayName: String,
    val durationMs: Long,
    val width: Int,
    val height: Int,
    val rotation: Int,
    val ranges: List<SeedRange>,
    val gameStartMs: Long = 0,
    val gameEndMs: Long = durationMs,
    val productionComponents: AnalysisTypes.ProductionComponents =
        AnalysisTypes.ProductionComponents.empty(),
    val productionServeOutputs: AnalysisTypes.ProductionServeOutputs =
        AnalysisTypes.ProductionServeOutputs.empty(),
    val productionStateOutputs: AnalysisTypes.ProductionStateOutputs =
        AnalysisTypes.ProductionStateOutputs.empty(),
    val servingSide: ServingSideOutput? = null,
    val servingSideError: String? = null,
    val sideSwitch: SideSwitchOutput? = null,
    val sideSwitchError: String? = null,
    val sideSwitchEnabled: Boolean = false,
    val scoreTrackingInitiallyEnabled: Boolean = false,
    val suppression: AnalysisTypes.SuppressionAnalysis? = null,
) {
    val sourceRevision: String by lazy {
        val canonical = buildString {
            append(sourceUri).append('|').append(displayName).append('|').append(durationMs).append('|')
            append(width).append('x').append(height).append('@').append(rotation).append('|')
            if (gameStartMs != 0L || gameEndMs != durationMs) {
                append("window=").append(gameStartMs).append(':').append(gameEndMs).append('|')
            }
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
    val reviewedCutIds: Set<String> = emptySet(),
    val cuts: List<EditableCut>,
    val ignoredIntervals: List<IgnoredSourceInterval> = emptyList(),
    val selectedSuppressionPolicy: SuppressionPolicyEngine.Policy =
        SuppressionPolicyEngine.Policy.AGGRESSIVE,
    val suppressionInitialBehavior: SuppressionInitialBehavior =
        SuppressionInitialBehavior.DISABLE_INITIALLY,
    val suppressionDecisionOverrides: Map<String, SuppressionDecision> = emptyMap(),
    val suppressionScopeOverrides: Map<String, SuppressionScope> = emptyMap(),
    val userTouchedCutIds: Set<String> = emptySet(),
    val suppressionContractVersion: String = FeatureSchema.SUPPRESSION_POLICY_CONTRACT_VERSION,
    val scoreTracking: ScoreTracking = ScoreTracking(),
    val renderScoreOverlay: Boolean = true,
    val renderScoreTimeline: Boolean = true,
)

internal data class JoinedGap(val startMs: Long, val endMs: Long)

internal data class FinalCutInterval(
    val startMs: Long,
    val endMs: Long,
    val cutIds: List<String>,
    val joinedGaps: List<JoinedGap> = emptyList(),
)

internal data class EditableRallyGroup(val cuts: List<EditableCut>) {
    init {
        require(cuts.isNotEmpty())
    }

    val id: String get() = cuts.first().id
    val cutIds: Set<String> get() = cuts.mapTo(linkedSetOf()) { it.id }
    val keepStartMs: Long get() = cuts.minOf { it.keepStartMs }
    val keepEndMs: Long get() = cuts.maxOf { it.keepEndMs }
    val coreStartMs: Long get() = cuts.minOf { it.coreStartMs }
    val coreEndMs: Long get() = cuts.maxOf { it.coreEndMs }

    fun asEditableCut(): EditableCut {
        val representative = cuts.first()
        return representative.copy(
            coreStartMs = coreStartMs,
            coreEndMs = coreEndMs,
            keepStartMs = keepStartMs,
            keepEndMs = keepEndMs,
            confidence = cuts.minOf { it.confidence },
            included = cuts.all { it.included },
            origin = if (cuts.all { it.origin == CutOrigin.MANUAL }) CutOrigin.MANUAL else CutOrigin.INFERRED,
        )
    }
}

internal data class DetailWindow(val startMs: Long, val endMs: Long)

internal data class CutSplitResult(
    val draft: EditorDraft,
    val newCut: EditableCut,
)

internal data class MaterializationProvenance(
    val startMs: Long,
    val endMs: Long,
    val kind: String,
    val cutIds: List<String> = emptyList(),
    val suggestionIds: List<String> = emptyList(),
)

internal data class FinalMaterialization(
    val intervals: List<FinalCutInterval>,
    val provenance: List<MaterializationProvenance>,
)

internal object EditorMath {
    fun newDraft(seed: EditorSeed): EditorDraft = EditorDraft(
        sourceRevision = seed.sourceRevision,
        updatedAtMs = 0,
        cuts = seed.ranges.mapNotNull { range ->
            val coreStart = range.startMs.coerceIn(seed.gameStartMs, seed.gameEndMs)
            val coreEnd = range.endMs.coerceIn(seed.gameStartMs, seed.gameEndMs)
            if (coreEnd <= coreStart) return@mapNotNull null
            EditableCut(
                id = "",
                coreStartMs = coreStart,
                coreEndMs = coreEnd,
                keepStartMs = (coreStart - DEFAULT_BEFORE_PADDING_MS).coerceAtLeast(seed.gameStartMs),
                keepEndMs = (coreEnd + DEFAULT_AFTER_PADDING_MS).coerceAtMost(seed.gameEndMs),
                confidence = range.confidence.coerceIn(0f, 1f),
                included = true,
                origin = CutOrigin.INFERRED,
                agreement = range.agreement,
            )
        }.mapIndexed { index, cut -> cut.copy(id = "R${(index + 1).toString().padStart(3, '0')}") },
        ignoredIntervals = buildList {
            if (seed.gameStartMs > 0) {
                add(IgnoredSourceInterval("G001", 0, seed.gameStartMs, "outside-game-window"))
            }
            if (seed.gameEndMs < seed.durationMs) {
                add(IgnoredSourceInterval("G002", seed.gameEndMs, seed.durationMs, "outside-game-window"))
            }
        },
        scoreTracking = ScoreReducer.seedModelMarkers(
            ScoreTracking(enabled = seed.scoreTrackingInitiallyEnabled),
            seed.servingSide,
            seed.sideSwitch,
            seed.sideSwitchEnabled,
        ),
        renderScoreOverlay = true,
        renderScoreTimeline = true,
    )

    fun applyPadding(
        draft: EditorDraft,
        beforeMs: Long,
        afterMs: Long,
        durationMs: Long,
        minimumMs: Long = 0,
        maximumMs: Long = durationMs,
    ): EditorDraft {
        val before = beforeMs.coerceIn(0, MAX_PADDING_MS)
        val after = afterMs.coerceIn(0, MAX_PADDING_MS)
        return draft.copy(
            beforePaddingMs = before,
            afterPaddingMs = after,
            cuts = draft.cuts.map { cut ->
                if (cut.origin == CutOrigin.MANUAL) cut else cut.copy(
                    keepStartMs = (cut.coreStartMs - before).coerceAtLeast(minimumMs),
                    keepEndMs = (cut.coreEndMs + after).coerceAtMost(maximumMs),
                )
            },
        )
    }

    fun setCoreStart(
        draft: EditorDraft,
        cutId: String,
        valueMs: Long,
        minimumMs: Long,
    ): EditorDraft = draft.copy(cuts = draft.cuts.map { cut ->
        if (cut.id != cutId) return@map cut
        val maximumStart = cut.coreEndMs - MIN_MARK_MS
        if (maximumStart < minimumMs) return@map cut
        val start = valueMs.coerceIn(minimumMs, maximumStart)
        val beforePadding = cut.coreStartMs - cut.keepStartMs
        cut.copy(
            coreStartMs = start,
            keepStartMs = (start - beforePadding).coerceAtLeast(minimumMs),
        )
    }).let(::alignRallyServeMarkers)

    fun setCoreEnd(
        draft: EditorDraft,
        cutId: String,
        valueMs: Long,
        maximumMs: Long,
    ): EditorDraft = draft.copy(cuts = draft.cuts.map { cut ->
        if (cut.id != cutId) return@map cut
        val minimumEnd = cut.coreStartMs + MIN_MARK_MS
        if (minimumEnd > maximumMs) return@map cut
        val end = valueMs.coerceIn(minimumEnd, maximumMs)
        val afterPadding = cut.keepEndMs - cut.coreEndMs
        cut.copy(
            coreEndMs = end,
            keepEndMs = (end + afterPadding).coerceAtMost(maximumMs),
        )
    })

    fun setCoreRange(
        draft: EditorDraft,
        cutId: String,
        startMs: Long,
        endMs: Long,
        minimumMs: Long,
        maximumMs: Long,
    ): EditorDraft {
        if (maximumMs - minimumMs < MIN_MARK_MS) return draft
        return draft.copy(cuts = draft.cuts.map { cut ->
            if (cut.id != cutId) return@map cut
            val start = startMs.coerceIn(minimumMs, maximumMs - MIN_MARK_MS)
            val end = endMs.coerceIn(start + MIN_MARK_MS, maximumMs)
            val beforePadding = cut.coreStartMs - cut.keepStartMs
            val afterPadding = cut.keepEndMs - cut.coreEndMs
            cut.copy(
                coreStartMs = start,
                coreEndMs = end,
                keepStartMs = (start - beforePadding).coerceAtLeast(minimumMs),
                keepEndMs = (end + afterPadding).coerceAtMost(maximumMs),
            )
        }).let(::alignRallyServeMarkers)
    }

    fun splitCut(
        draft: EditorDraft,
        cutId: String,
        positionMs: Long,
        minimumMs: Long,
        maximumMs: Long,
    ): CutSplitResult? {
        val cutIndex = draft.cuts.indexOfFirst { it.id == cutId }
        if (cutIndex < 0) return null
        val cut = draft.cuts[cutIndex]
        if (positionMs < cut.coreStartMs + MIN_MARK_MS ||
            positionMs > cut.coreEndMs - MIN_MARK_MS
        ) return null

        val beforePadding = cut.coreStartMs - cut.keepStartMs
        val afterPadding = cut.keepEndMs - cut.coreEndMs
        val prefix = if (cut.origin == CutOrigin.INFERRED) "R" else "M"
        val left = cut.copy(
            coreEndMs = positionMs,
            keepEndMs = (positionMs + afterPadding).coerceAtMost(maximumMs),
        )
        val right = cut.copy(
            id = nextId(prefix, draft.cuts.map { it.id }),
            coreStartMs = positionMs,
            keepStartMs = (positionMs - beforePadding).coerceAtLeast(minimumMs),
        )
        val updatedCuts = draft.cuts.toMutableList().apply {
            this[cutIndex] = left
            add(cutIndex + 1, right)
        }
        return CutSplitResult(draft.copy(cuts = updatedCuts), right)
    }

    fun finalIntervals(
        draft: EditorDraft,
        suppression: AnalysisTypes.SuppressionAnalysis? = null,
    ): List<FinalCutInterval> = materialize(draft, suppression).intervals

    fun editableRallyGroups(
        cuts: List<EditableCut>,
        intervals: List<FinalCutInterval>,
        serveMarkers: List<ServeMarker>,
        hardBoundaryCutIds: Set<String> = emptySet(),
    ): List<EditableRallyGroup> {
        val intervalIndexesByCutId = buildMap<String, Set<Int>> {
            cuts.forEach { cut ->
                put(cut.id, intervals.mapIndexedNotNull { index, interval ->
                    index.takeIf { cut.id in interval.cutIds }
                }.toSet())
            }
        }
        val groups = mutableListOf<EditableRallyGroup>()
        cuts.sortedWith(compareBy<EditableCut> { it.keepStartMs }.thenBy { it.keepEndMs }).forEach { cut ->
            val previous = groups.lastOrNull()
            val previousTail = previous?.cuts?.lastOrNull()
            val sharesOutputInterval = previous != null &&
                previous.cutIds.flatMap { intervalIndexesByCutId[it].orEmpty() }.toSet()
                    .intersect(intervalIndexesByCutId[cut.id].orEmpty()).isNotEmpty()
            val separatedByServe = previousTail != null && serveMarkers.any { marker ->
                marker.rallyId == cut.id ||
                    (marker.rallyId == null &&
                        marker.timestampMs >= cut.keepStartMs && marker.timestampMs < cut.coreEndMs)
            }
            val separatedByHardBoundary = cut.id in hardBoundaryCutIds ||
                previous?.cutIds?.any { it in hardBoundaryCutIds } == true
            if (previous == null || !sharesOutputInterval || separatedByServe || separatedByHardBoundary) {
                groups += EditableRallyGroup(listOf(cut))
            } else {
                groups[groups.lastIndex] = EditableRallyGroup(previous.cuts + cut)
            }
        }
        return groups
    }

    fun materialize(
        draft: EditorDraft,
        suppression: AnalysisTypes.SuppressionAnalysis? = null,
    ): FinalMaterialization {
        val joinGapMs = draft.joinGapMs.coerceIn(0, MAX_JOIN_GAP_MS)
        val activeSuggestions = activeSuggestions(draft, suppression)
        val appliedSuggestions = activeSuggestions.filter { suggestion ->
            suggestionEffectiveDecision(draft, suggestion) == SuppressionDecision.SUPPRESS
        }
        val wholeRallySuggestions = appliedSuggestions.filter { suggestion ->
            suggestionEffectiveScope(draft, suggestion) == SuppressionScope.WHOLE_RALLY
        }
        val vetoRegionSuggestions = appliedSuggestions.filter { suggestion ->
            suggestionEffectiveScope(draft, suggestion) == SuppressionScope.VETO_REGION
        }
        val wholeRallyCutIds = draft.cuts.asSequence()
            .filter { it.origin == CutOrigin.INFERRED }
            .filter { cut ->
                wholeRallySuggestions.any { suggestion ->
                    cut.coreStartMs < suggestion.endMs() && suggestion.startMs() < cut.coreEndMs
                }
            }
            .map { it.id }
            .toSet()
        val suppressionBarriers = buildList {
            vetoRegionSuggestions.forEach { suggestion ->
                add(suggestion.startMs() to suggestion.endMs())
            }
            draft.cuts.filter { it.id in wholeRallyCutIds }.forEach { cut ->
                add(cut.keepStartMs to cut.keepEndMs)
            }
        }
        val sourceIntervals = mutableListOf<FinalCutInterval>()
        val provenance = mutableListOf<MaterializationProvenance>()
        draft.cuts.asSequence().filter { it.included }.forEach { cut ->
            if (cut.origin == CutOrigin.MANUAL) {
                if (cut.keepEndMs > cut.keepStartMs) {
                    sourceIntervals += FinalCutInterval(cut.keepStartMs, cut.keepEndMs, listOf(cut.id))
                    provenance += MaterializationProvenance(
                        cut.keepStartMs, cut.keepEndMs, "manual", listOf(cut.id),
                    )
                }
                return@forEach
            }
            if (cut.id in wholeRallyCutIds) return@forEach
            var fragments = listOf(cut.coreStartMs to cut.coreEndMs)
            vetoRegionSuggestions.forEach { suggestion ->
                fragments = fragments.flatMap { (start, end) ->
                    subtract(start, end, suggestion.startMs(), suggestion.endMs())
                }
            }
            fragments.forEach { (coreStart, coreEnd) ->
                if (coreEnd <= coreStart) return@forEach
                val paddedStart = if (coreStart == cut.coreStartMs) cut.keepStartMs
                    else (coreStart - draft.beforePaddingMs).coerceAtLeast(0)
                val paddedEnd = if (coreEnd == cut.coreEndMs) cut.keepEndMs
                    else coreEnd + draft.afterPaddingMs
                if (paddedEnd > paddedStart) {
                    val hardClipped = vetoRegionSuggestions.fold(
                        listOf(paddedStart to paddedEnd),
                    ) { fragments, suggestion ->
                        fragments.flatMap { (start, end) ->
                            subtract(start, end, suggestion.startMs(), suggestion.endMs())
                        }
                    }
                    hardClipped.forEach { (start, end) ->
                        if (end > start) {
                            sourceIntervals += FinalCutInterval(start, end, listOf(cut.id))
                        }
                    }
                    provenance += MaterializationProvenance(
                        coreStart, coreEnd, "inferred-core", listOf(cut.id),
                    )
                    if (paddedStart < coreStart) provenance += MaterializationProvenance(
                        paddedStart, coreStart, "padding", listOf(cut.id),
                    )
                    if (paddedEnd > coreEnd) provenance += MaterializationProvenance(
                        coreEnd, paddedEnd, "padding", listOf(cut.id),
                    )
                }
            }
        }
        appliedSuggestions.forEach { suggestion ->
            if (suggestionEffectiveScope(draft, suggestion) == SuppressionScope.WHOLE_RALLY) {
                draft.cuts.filter { cut ->
                    cut.id in wholeRallyCutIds && cut.coreStartMs < suggestion.endMs() &&
                        suggestion.startMs() < cut.coreEndMs
                }.forEach { cut ->
                    provenance += MaterializationProvenance(
                        cut.keepStartMs, cut.keepEndMs, "suppression-whole-rally",
                        cutIds = listOf(cut.id),
                        suggestionIds = listOf(suggestion.logicalId()),
                    )
                }
            } else {
                provenance += MaterializationProvenance(
                    suggestion.startMs(), suggestion.endMs(), "suppression-veto-region",
                    suggestionIds = listOf(suggestion.logicalId()),
                )
            }
        }

        val merged = mutableListOf<FinalCutInterval>()
        sourceIntervals.asSequence()
            .sortedWith(compareBy<FinalCutInterval> { it.startMs }.thenBy { it.endMs })
            .forEach { interval ->
                val previous = merged.lastOrNull()
                val gap = previous?.let { interval.startMs - it.endMs } ?: Long.MAX_VALUE
                val crossesSuppression = previous != null && gap > 0 &&
                    suppressionBarriers.any { (start, end) ->
                        start < interval.startMs && previous.endMs < end
                    }
                if (previous == null || (gap > 0 && (gap >= joinGapMs || crossesSuppression))) {
                    merged += interval
                } else {
                    merged[merged.lastIndex] = previous.copy(
                        endMs = max(previous.endMs, interval.endMs),
                        cutIds = (previous.cutIds + interval.cutIds).distinct(),
                        joinedGaps = if (gap > 0) {
                            previous.joinedGaps + JoinedGap(previous.endMs, interval.startMs)
                        } else previous.joinedGaps,
                    )
                    if (gap > 0) provenance += MaterializationProvenance(
                        previous.endMs, interval.startMs, "joined-gap",
                        (previous.cutIds + interval.cutIds).distinct(),
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
        val sourceById = sourceIntervals.flatMap { source ->
            source.cutIds.map { id -> id to source }
        }.groupBy({ it.first }, { it.second })
        val result = remaining.map { interval ->
            interval.copy(
                cutIds = interval.cutIds.filter { id ->
                    sourceById[id].orEmpty().any { source ->
                        min(source.endMs, interval.endMs) > max(source.startMs, interval.startMs)
                    }
                },
                joinedGaps = interval.joinedGaps.mapNotNull { gap ->
                    val start = max(interval.startMs, gap.startMs)
                    val end = min(interval.endMs, gap.endMs)
                    if (end > start) JoinedGap(start, end) else null
                },
            )
        }
        val clippedProvenance = provenance.flatMap { item ->
            if (item.kind.startsWith("suppression-")) listOf(item) else result.mapNotNull { interval ->
                    val start = max(item.startMs, interval.startMs)
                    val end = min(item.endMs, interval.endMs)
                    if (end > start) item.copy(startMs = start, endMs = end) else null
                }
        }
        return FinalMaterialization(result, clippedProvenance)
    }

    fun effectiveKeptIds(
        draft: EditorDraft,
        suppression: AnalysisTypes.SuppressionAnalysis? = null,
    ): Set<String> = finalIntervals(draft, suppression).flatMap { it.cutIds }.toSet()

    fun activeSuggestions(
        draft: EditorDraft,
        suppression: AnalysisTypes.SuppressionAnalysis?,
    ): List<AnalysisTypes.SuppressionSuggestion> = SuppressionPolicyEngine.active(
        suppression,
        draft.selectedSuppressionPolicy,
    )

    fun suggestionEffectiveDecision(
        draft: EditorDraft,
        suggestion: AnalysisTypes.SuppressionSuggestion,
    ): SuppressionDecision {
        draft.suppressionDecisionOverrides[suggestion.logicalId()]?.let { return it }
        val overlapsTouched = draft.cuts.any { cut ->
            cut.origin == CutOrigin.INFERRED && cut.id in draft.userTouchedCutIds &&
                cut.coreStartMs < suggestion.endMs() && suggestion.startMs() < cut.coreEndMs
        }
        if (overlapsTouched) return SuppressionDecision.KEEP
        return if (draft.suppressionInitialBehavior == SuppressionInitialBehavior.DISABLE_INITIALLY) {
            SuppressionDecision.SUPPRESS
        } else SuppressionDecision.KEEP
    }

    fun suggestionEffectiveScope(
        draft: EditorDraft,
        suggestion: AnalysisTypes.SuppressionSuggestion,
    ): SuppressionScope = draft.suppressionScopeOverrides[suggestion.logicalId()]
        ?: SuppressionScope.WHOLE_RALLY

    fun reconcileTouchedCuts(draft: EditorDraft, seed: EditorSeed): EditorDraft {
        val seedCuts = applyPadding(
            newDraft(seed), draft.beforePaddingMs, draft.afterPaddingMs, seed.durationMs,
            seed.gameStartMs, seed.gameEndMs,
        ).cuts.associateBy { it.id }
        val changed = draft.cuts.asSequence().filter { cut ->
            if (cut.origin != CutOrigin.INFERRED) return@filter false
            val original = seedCuts[cut.id] ?: return@filter true
            cut.coreStartMs != original.coreStartMs || cut.coreEndMs != original.coreEndMs ||
                cut.keepStartMs != original.keepStartMs || cut.keepEndMs != original.keepEndMs ||
                cut.included != original.included
        }.map { it.id }.toSet()
        return alignRallyServeMarkers(draft.copy(userTouchedCutIds = draft.userTouchedCutIds + changed))
    }

    /** Linked serves mark the corrected rally start, not its padding or model anchor.
     * Also repairs older drafts after model markers are reseeded on load.
     * Explicit rally IDs avoid moving independent manual serves or side switches.
     */
    fun alignRallyServeMarkers(draft: EditorDraft): EditorDraft {
        val starts = draft.cuts.associate { it.id to it.coreStartMs }
        val markers = draft.scoreTracking.serveMarkers.map { marker ->
            val start = starts[marker.rallyId]
            if (start == null || start == marker.timestampMs) marker
            else marker.copy(timestampMs = start)
        }
        if (markers == draft.scoreTracking.serveMarkers) return draft
        return draft.copy(scoreTracking = draft.scoreTracking.copy(
            serveMarkers = markers.sortedWith(compareBy<ServeMarker> { it.timestampMs }.thenBy { it.id }),
        ))
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

    fun isTimestampIgnored(
        timestampMs: Long,
        ignoredIntervals: List<IgnoredSourceInterval>,
    ): Boolean = ignoredIntervals.any { interval ->
        timestampMs >= interval.startMs && timestampMs < interval.endMs
    }

    fun firstReviewableTime(
        startMs: Long,
        endMs: Long,
        ignoredIntervals: List<IgnoredSourceInterval>,
    ): Long? {
        if (endMs <= startMs) return startMs.takeUnless {
            isTimestampIgnored(it, ignoredIntervals)
        }
        var candidate = startMs
        for (interval in mergeIgnored(ignoredIntervals)) {
            if (interval.endMs <= candidate) continue
            if (interval.startMs >= endMs) break
            if (candidate < interval.startMs) return candidate
            candidate = max(candidate, interval.endMs)
            if (candidate >= endMs) return null
        }
        return candidate.takeIf { it < endMs }
    }

    fun nextSuppressionSuggestion(
        suggestions: List<AnalysisTypes.SuppressionSuggestion>,
        positionMs: Long,
        selectedFragmentId: String?,
        reviewPrerollMs: Long = 2_000,
    ): AnalysisTypes.SuppressionSuggestion? {
        if (suggestions.isEmpty()) return null
        val ordered = suggestions.sortedWith(
            compareBy<AnalysisTypes.SuppressionSuggestion> { it.startMs() }
                .thenBy { it.endMs() },
        )
        val selectedIndex = ordered.indexOfFirst { it.fragmentId() == selectedFragmentId }
        if (selectedIndex >= 0) {
            val selected = ordered[selectedIndex]
            if (positionMs in (selected.startMs() - reviewPrerollMs).coerceAtLeast(0)..selected.endMs()) {
                return ordered[(selectedIndex + 1) % ordered.size]
            }
        }
        return ordered.firstOrNull { it.startMs() >= positionMs } ?: ordered.first()
    }

    fun playbackFocusCut(cuts: List<EditableCut>, positionMs: Long): EditableCut? =
        cuts.filter { it.keepStartMs <= positionMs }.maxByOrNull { it.keepStartMs }

    fun detailWindow(
        cut: EditableCut?,
        positionMs: Long,
        durationMs: Long,
        minimumMs: Long = 0,
        maximumMs: Long = durationMs,
    ): DetailWindow {
        val windowDuration = (maximumMs - minimumMs).coerceAtLeast(0)
        val minimumSpan = min(24_000L, if (windowDuration > 0) windowDuration else 24_000L)
        val center = cut?.let { (it.keepStartMs + it.keepEndMs) / 2 } ?: positionMs
        val contentSpan = cut?.let { it.keepEndMs - it.keepStartMs + 10_000L } ?: minimumSpan
        val span = min(if (windowDuration > 0) windowDuration else minimumSpan, max(minimumSpan, contentSpan))
        var start = max(minimumMs, center - span / 2)
        var end = min(maximumMs, start + span)
        start = max(minimumMs, end - span)
        if (end <= start) end = (start + 1).coerceAtMost(maximumMs.coerceAtLeast(minimumMs + 1))
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

    private fun subtract(
        start: Long,
        end: Long,
        removeStart: Long,
        removeEnd: Long,
    ): List<Pair<Long, Long>> {
        if (removeEnd <= start || removeStart >= end) return listOf(start to end)
        return buildList {
            if (removeStart > start) add(start to min(end, removeStart))
            if (removeEnd < end) add(max(start, removeEnd) to end)
        }
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
