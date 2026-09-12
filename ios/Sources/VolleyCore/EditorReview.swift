import Foundation

/// Android EditorMath review contracts. These operate on source time, including
/// excluded footage, so cleanup can still be reviewed when it removes a rally.
public enum EditorReview {
    public static func firstReviewableTime(startMs: Int64, endMs: Int64, ignored: [IgnoredSourceInterval]) -> Int64? {
        if endMs <= startMs { return ignored.contains { $0.startMs <= startMs && startMs < $0.endMs } ? nil : startMs }
        var candidate = startMs
        for span in ignored.sorted(by: { ($0.startMs, $0.endMs) < ($1.startMs, $1.endMs) }) {
            if span.endMs <= candidate { continue }
            if span.startMs >= endMs { break }
            if candidate < span.startMs { return candidate }
            candidate = max(candidate, span.endMs)
            if candidate >= endMs { return nil }
        }
        return candidate < endMs ? candidate : nil
    }
    public static func activeSuggestions(_ suggestions: [SuppressionRegion], draft: EditorDraft) -> [SuppressionRegion] {
        suggestions.filter { draft.selectedSuppressionPolicy != "none" && $0.eligiblePolicyIds.contains(draft.selectedSuppressionPolicy) }
            .sorted { ($0.startMs, $0.endMs, $0.id) < ($1.startMs, $1.endMs, $1.id) }
    }
    public static func pendingCleanup(_ active: [SuppressionRegion], draft: EditorDraft) -> [SuppressionRegion] {
        active.filter { draft.suppressionDecisionOverrides[$0.logicalId] == nil &&
            firstReviewableTime(startMs: $0.startMs, endMs: $0.endMs, ignored: draft.ignoredIntervals) != nil }
    }
    public static func nextCleanup(_ suggestions: [SuppressionRegion], positionMs: Int64, selectedId: String?) -> SuppressionRegion? {
        let ordered = suggestions.sorted { ($0.startMs, $0.endMs, $0.id) < ($1.startMs, $1.endMs, $1.id) }
        guard !ordered.isEmpty else { return nil }
        if let index = ordered.firstIndex(where: { $0.id == selectedId }) {
            let current = ordered[index]
            if positionMs >= max(0, current.startMs - 2_000), positionMs <= current.endMs {
                return ordered[(index + 1) % ordered.count]
            }
        }
        return ordered.first { $0.startMs >= positionMs } ?? ordered.first
    }
    /// A cleanup decision applies to every fragment of the logical suggestion.
    /// It does not change cut inclusion: a partial veto must preserve the rest.
    public static func decideCleanup(_ draft: EditorDraft, suggestion: SuppressionRegion,
                                     keep: Bool, reviewedCutIds: Set<String>) -> EditorDraft {
        var result = draft
        result.suppressionDecisionOverrides[suggestion.logicalId] = keep ? "keep" : "suppress"
        result.reviewedCutIds.formUnion(reviewedCutIds)
        return result
    }
}

/// Materialized once per draft change, never from a playback or resize callback.
public struct EditorPresentation: Sendable {
    public let intervals: [FinalCutInterval]
    public let keptIds: Set<String>
    public let orderedCuts: [EditableCut]
    public let pendingConfidenceIds: Set<String>
    public let groups: [EditableRallyGroup]
    public let reviewCuts: [EditableCut]
    public let activeSuggestions: [SuppressionRegion]
    public let pendingCleanup: [SuppressionRegion]
    public let appliedSuggestionIds: Set<String>
    public let userRemovedSuggestionIds: Set<String>
    public let userRemovedCleanupCutIds: Set<String>
    public let score: ScoreTracking
    public let visibleScore: ScoreTracking
    public let scoreRallyRanges: [ScoreRallyRange]
    public let reviewServes: [ServeMarker]
    public init(draft: EditorDraft, suggestions: [SuppressionRegion], bounds: TimeRange, durationMs: Int64) {
        let intervals = EditorMath.finalIntervals(draft, suppression: suggestions, bounds: bounds)
        self.intervals = intervals
        let kept = Set(intervals.flatMap(\.cutIds)); keptIds = kept
        let cuts = draft.cuts.sorted { ($0.keepStartMs, $0.keepEndMs, $0.id) < ($1.keepStartMs, $1.keepEndMs, $1.id) }
        orderedCuts = cuts
        let pending = Set(cuts.filter { $0.origin == .inferred && $0.included && !draft.reviewedCutIds.contains($0.id) &&
            ($0.confidence < draft.confidenceReviewThreshold || ProductionEnsemble.isDisagreement($0.agreement)) }.map(\.id))
        pendingConfidenceIds = pending
        let score = (try? ScoreTracking.fromJSON(draft.scoreTracking, durationMs: durationMs)) ?? ScoreTracking(enabled: false)
        self.score = score
        let groups = EditorMath.editableRallyGroups(cuts: cuts, intervals: intervals,
            serveMarkers: score.enabled ? score.serveMarkers : [], hardBoundaryCutIds: pending)
        self.groups = groups
        reviewCuts = groups.filter { !$0.cutIds.isDisjoint(with: pending.intersection(kept)) &&
            EditorReview.firstReviewableTime(startMs: $0.keepStartMs, endMs: $0.keepEndMs, ignored: draft.ignoredIntervals) != nil }.map { $0.asEditableCut() }
        let active = EditorReview.activeSuggestions(suggestions, draft: draft)
        activeSuggestions = active
        pendingCleanup = EditorReview.pendingCleanup(active, draft: draft)
        appliedSuggestionIds = Set(active.filter { EditorMath.effectiveDecision(draft, suggestion: $0) == "suppress" }.map(\.id))
        let removed = active.filter { draft.suppressionDecisionOverrides[$0.logicalId] == "suppress" }
        userRemovedSuggestionIds = Set(removed.map(\.id))
        userRemovedCleanupCutIds = Set(cuts.filter { cut in !kept.contains(cut.id) && removed.contains {
            cut.coreStartMs < $0.endMs && $0.startMs < cut.coreEndMs } }.map(\.id))
        let visible = ScoreReducer.visibleTracking(score, ignoredIntervals: draft.ignoredIntervals,
            excludedRallyIds: Set(cuts.filter { $0.origin == .inferred && (!$0.included || !kept.contains($0.id)) }.map(\.id)))
        visibleScore = visible
        scoreRallyRanges = cuts.filter { $0.included && ($0.origin == .manual || kept.contains($0.id)) }.map {
            .init(coreStartMs: $0.coreStartMs, coreEndMs: $0.coreEndMs, keepStartMs: $0.keepStartMs, keepEndMs: $0.keepEndMs) }
        reviewServes = visible.serveMarkers.filter { $0.side == .review }.sorted { ($0.timestampMs, $0.id) < ($1.timestampMs, $1.id) }
    }
}
