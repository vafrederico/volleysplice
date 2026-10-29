package com.volleycut.nativeanalysis

/** Draft-dependent editor calculations shared by the phone and desktop layouts. */
internal class EditorDerivedState(
    draft: EditorDraft,
    suppression: AnalysisTypes.SuppressionAnalysis?,
    windowStartMs: Long,
    windowEndMs: Long,
) {
    val sortedCuts = draft.cuts.sortedWith(compareBy<EditableCut> { it.keepStartMs }.thenBy { it.keepEndMs })
    val finalIntervals = EditorMath.finalIntervals(draft, suppression)
    val editableRallies = rallyGroupsForDraft(draft, suppression, finalIntervals)
    val effectiveIds = finalIntervals.flatMap { it.cutIds }.toSet()
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
    val visibleIgnoredIntervals = draft.ignoredIntervals.filter {
        it.endMs > windowStartMs && it.startMs < windowEndMs
    }
    val activeSuggestions = EditorMath.activeSuggestions(draft, suppression)
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
    val lowConfidence = editableRallies.filter { rally ->
        rally.cuts.any {
            it.origin == CutOrigin.INFERRED && it.included && it.id in effectiveIds &&
                (ProductionEnsemble.isDisagreement(it.agreement) || it.confidence < draft.confidenceReviewThreshold)
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
    val rallyQueue = rallyReviewQueue(
        editableRallies,
        pendingReview.mapTo(mutableSetOf()) { it.id },
        pendingCleanupReview.map { RallyReviewCleanup(it.logicalId(), it.startMs(), it.endMs()) },
        reviewableServeReviews.takeIf { draft.scoreTracking.enabled }.orEmpty(),
        visibleIgnoredIntervals,
    )
    val reviewAttentionBreakdown = buildList {
        if (pendingReview.isNotEmpty()) add("${pendingReview.size} ${if (pendingReview.size == 1) "clip" else "clips"}")
        if (pendingCleanupReview.isNotEmpty()) add("${pendingCleanupReview.size} cleanup")
        if (reviewableServeReviews.isNotEmpty()) add(
            "${reviewableServeReviews.size} ${if (reviewableServeReviews.size == 1) "serve" else "serves"}",
        )
    }.joinToString(" · ")
    val joinedGaps = finalIntervals.flatMap { it.joinedGaps }
    val totalFinalMs = EditorMath.totalFinalMs(finalIntervals)
    val leftOutRallyCount = editableRallies.count { rally -> rally.cutIds.none { it in effectiveIds } }
    val orderedServeMarkers = preparedScoreOverlay.tracking.serveMarkers
        .sortedWith(compareBy<ServeMarker> { it.timestampMs }.thenBy { it.id })
}
