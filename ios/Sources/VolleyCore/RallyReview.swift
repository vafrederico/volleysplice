import Foundation

/// Source-time review stops. Cleanup and serve evidence outside detected rallies
/// remain reachable; visiting a stop never resolves it.
public struct RallyReviewStop: Equatable, Identifiable, Sendable {
    public var id: String
    public var rallyId: String?
    public var startMs: Int64
    public var endMs: Int64
    public var clipReview = false
    public var cleanupIds: Set<String> = []
    public var serveIds: Set<String> = []
}

public enum RallyReview {
    public static let minimumDurationMs: Int64 = 500

    public static func queue(presentation p: EditorPresentation, ignored: [IgnoredSourceInterval]) -> [RallyReviewStop] {
        var stops: [String: RallyReviewStop] = [:]
        func add(_ rally: EditableRallyGroup?, _ stop: RallyReviewStop) {
            guard EditorReview.firstReviewableTime(startMs: stop.startMs, endMs: stop.endMs, ignored: ignored) != nil else { return }
            let key = rally.map { "rally-" + $0.id } ?? stop.id
            var value = stops[key] ?? RallyReviewStop(id: key, rallyId: rally?.id,
                startMs: rally?.keepStartMs ?? stop.startMs, endMs: rally?.keepEndMs ?? stop.endMs)
            value.startMs = min(value.startMs, stop.startMs); value.endMs = max(value.endMs, stop.endMs)
            value.clipReview = value.clipReview || stop.clipReview
            value.cleanupIds.formUnion(stop.cleanupIds); value.serveIds.formUnion(stop.serveIds)
            stops[key] = value
        }
        for cut in p.reviewCuts {
            let rally = p.groups.first { $0.cutIds.contains(cut.id) }
            add(rally, .init(id: "clip-" + cut.id, startMs: cut.keepStartMs, endMs: cut.keepEndMs, clipReview: true))
        }
        for cleanup in p.pendingCleanup {
            let rally = p.groups.filter { $0.coreStartMs < cleanup.endMs && cleanup.startMs < $0.coreEndMs }.max {
                min($0.coreEndMs, cleanup.endMs) - max($0.coreStartMs, cleanup.startMs) <
                min($1.coreEndMs, cleanup.endMs) - max($1.coreStartMs, cleanup.startMs)
            }
            add(rally, .init(id: "cleanup-" + cleanup.id, startMs: cleanup.startMs, endMs: cleanup.endMs, cleanupIds: [cleanup.id]))
        }
        for serve in p.reviewServes {
            let rally = p.groups.first { $0.cutIds.contains(serve.rallyId ?? "") } ??
                p.groups.first { $0.coreStartMs <= serve.timestampMs && serve.timestampMs < $0.coreEndMs } ??
                at(serve.timestampMs, in: p.groups)
            add(rally, .init(id: "serve-" + serve.id, startMs: serve.timestampMs, endMs: serve.timestampMs + 2_000, serveIds: [serve.id]))
        }
        return stops.values.sorted { ($0.startMs, $0.id) < ($1.startMs, $1.id) }
    }

    public static func at(_ time: Int64, in rallies: [EditableRallyGroup]) -> EditableRallyGroup? {
        rallies.filter { $0.keepStartMs <= time && time < $0.keepEndMs }
            .min { abs($0.coreStartMs - time) < abs($1.coreStartMs - time) }
    }
    public static func nearest(_ time: Int64, in rallies: [EditableRallyGroup]) -> EditableRallyGroup? {
        rallies.min { max($0.keepStartMs - time, time - $0.keepEndMs, 0) < max($1.keepStartMs - time, time - $1.keepEndMs, 0) }
    }
    public static func serves(for rally: EditableRallyGroup, in markers: [ServeMarker]) -> [ServeMarker] {
        markers.filter { rally.cutIds.contains($0.rallyId ?? "") ||
            ($0.rallyId == nil && rally.coreStartMs <= $0.timestampMs && $0.timestampMs < rally.coreEndMs) }
            .sorted { ($0.timestampMs, $0.id) < ($1.timestampMs, $1.id) }
    }
    public static func retained(_ rally: EditableRallyGroup, effectiveIds: Set<String>) -> Bool {
        rally.cuts.contains { $0.included && effectiveIds.contains($0.id) }
    }
    public static func neighbor(of rally: EditableRallyGroup, direction: Int, rallies: [EditableRallyGroup], effectiveIds: Set<String>) -> EditableRallyGroup? {
        guard let index = rallies.firstIndex(where: { $0.id == rally.id }) else { return nil }
        if direction < 0 { return rallies.prefix(index).last { retained($0, effectiveIds: effectiveIds) } }
        return rallies.dropFirst(index + 1).first { retained($0, effectiveIds: effectiveIds) }
    }
    public static func context(for rally: EditableRallyGroup, rallies: [EditableRallyGroup], effectiveIds: Set<String>, bounds: TimeRange) -> TimeRange {
        // Keep short rallies editable even when their retained neighbours are
        // hours away. The view represents distant neighbours as edge actions.
        return .init(startMs: max(bounds.startMs, min(rally.coreStartMs - 8_000, rally.keepStartMs)),
                     endMs: min(bounds.endMs, max(rally.coreEndMs + 8_000, rally.keepEndMs)))
    }
    public static func missingRange(at time: Int64, bounds: TimeRange, occupied: [TimeRange]) -> TimeRange? {
        guard bounds.startMs <= time, time < bounds.endMs,
              !occupied.contains(where: { $0.startMs <= time && time < $0.endMs }) else { return nil }
        let left = max(bounds.startMs, occupied.filter { $0.endMs <= time }.map(\.endMs).max() ?? bounds.startMs)
        let right = min(bounds.endMs, occupied.filter { $0.startMs > time }.map(\.startMs).min() ?? bounds.endMs)
        guard right - left >= minimumDurationMs else { return nil }
        var start = max(left, time - 3_000), end = min(right, time + 5_000)
        if end - start < minimumDurationMs {
            start = max(left, end - minimumDurationMs); end = min(right, start + minimumDurationMs)
        }
        return .init(startMs: start, endMs: end)
    }
    /// A trim pauses for inspection, retaining the viewed frame unless excluded.
    public static func frameAfterTrim(_ frame: Int64, range: TimeRange) -> Int64 { min(max(frame, range.startMs), range.endMs) }

    /// Explicit Keep/Remove resolves clip and cleanup work, never serve work.
    public static func decide(_ draft: EditorDraft, rally: EditableRallyGroup?, cleanup: [SuppressionRegion], keep: Bool) -> EditorDraft {
        var result = draft
        for cut in rally?.cuts ?? [] { result = EditorMath.setIncluded(result, cutId: cut.id, included: keep) }
        result.reviewedCutIds.formUnion(rally?.cutIds ?? [])
        for suggestion in cleanup { result.suppressionDecisionOverrides[suggestion.logicalId] = keep ? "keep" : "suppress" }
        return result
    }
}

/// Opening or seeking in Add only proposes a range. Confirmation is a separate
/// draft operation, and existing/ignored footage remains seekable but not addable.
public struct MissingRallySelection: Equatable, Sendable {
    public private(set) var positionMs: Int64
    public var range: TimeRange?
    public init(positionMs: Int64, bounds: TimeRange, occupied: [TimeRange]) {
        self.positionMs = min(max(positionMs, bounds.startMs), bounds.endMs)
        range = RallyReview.missingRange(at: self.positionMs, bounds: bounds, occupied: occupied)
    }
    public mutating func seek(to positionMs: Int64, bounds: TimeRange, occupied: [TimeRange]) {
        self = .init(positionMs: positionMs, bounds: bounds, occupied: occupied)
    }
}

/// Navigation is view state, separate from the persisted editing draft.
public struct RallyDeckNavigation: Equatable, Sendable {
    public var selectedId: String?
    public var focus: RallyReviewStop?
    public var browsing = false
    public var showTimeline = true
    public var following = true
    public var listCenterId: String?
    public var followRequest = 0
    public var initialReviewKeys: Set<String> = []
    public init() {}
    public mutating func follow(time: Int64, rallies: [EditableRallyGroup]) {
        if let rally = RallyReview.at(time, in: rallies) { selectedId = rally.id }
        listCenterId = RallyReview.nearest(time, in: rallies)?.id
        following = true; followRequest += 1
    }
    public mutating func visit(_ stop: RallyReviewStop) {
        focus = stop; selectedId = stop.rallyId; browsing = false
    }
}
