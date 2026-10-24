import Foundation

public enum EditorPlaybackPolicy {
    public static let rates: [Float] = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 4, 8]
}

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
        let previous = neighbor(of: rally, direction: -1, rallies: rallies, effectiveIds: effectiveIds)
        let next = neighbor(of: rally, direction: 1, rallies: rallies, effectiveIds: effectiveIds)
        return .init(startMs: max(bounds.startMs, min(rally.coreStartMs - 8_000, rally.keepStartMs, previous.map { $0.keepStartMs - 2_000 } ?? Int64.max)),
                     endMs: min(bounds.endMs, max(rally.coreEndMs + 8_000, rally.keepEndMs, next.map { $0.keepEndMs + 2_000 } ?? Int64.min)))
    }
    public static func coreWindow(_ core: TimeRange, bounds: TimeRange) -> TimeRange {
        let margin = min(8_000, max(0, core.endMs - core.startMs) / 2)
        return .init(startMs: max(bounds.startMs, core.startMs - margin), endMs: min(bounds.endMs, core.endMs + margin))
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

    /// Apply the edit and reconcile all grouping, cleanup and score boundaries
    /// before finding the selected group's retained bounds. Old group membership
    /// chooses the edited cut only; it must never choose the post-edit clamp.
    public static func trimBoundary(_ draft: EditorDraft, selectedCutIds: Set<String>, isStart: Bool, valueMs: Int64,
                            minimumDurationMs: Int64 = EditorMath.minimumMarkMs, frameMs: Int64,
                            suggestions: [SuppressionRegion], bounds: TimeRange, durationMs: Int64) throws -> RallyTrimResult? {
        let before = EditorPresentation(draft: draft, suggestions: suggestions, bounds: bounds, durationMs: durationMs)
        guard let original = before.groups.first(where: { !$0.cutIds.isDisjoint(with: selectedCutIds) }) else { return nil }
        let members = original.cuts
        guard let first = members.min(by: { $0.coreStartMs < $1.coreStartMs }),
              let last = members.max(by: { $0.coreEndMs < $1.coreEndMs }) else { return nil }
        var updated = draft
        if isStart, valueMs != first.coreStartMs {
            let value = min(max(bounds.startMs, valueMs), last.coreEndMs - minimumDurationMs)
            updated = EditorMath.setCoreStart(updated, cutId: first.id, valueMs: value, minimumMs: bounds.startMs)
        }
        if !isStart, valueMs != last.coreEndMs {
            let value = max(min(bounds.endMs, valueMs), first.coreStartMs + minimumDurationMs)
            updated = EditorMath.setCoreEnd(updated, cutId: last.id, valueMs: value, maximumMs: bounds.endMs)
        }
        try updated.validate(durationMs: durationMs)
        let reconciled = EditorPresentation(draft: updated, suggestions: suggestions, bounds: bounds, durationMs: durationMs)
        guard let selected = reconciled.groups.first(where: { $0.cutIds.contains(first.id) }) else { return nil }
        return .init(draft: updated, presentation: reconciled, selectedGroup: selected,
                     frameMs: frameAfterTrim(frameMs, range: .init(startMs: selected.keepStartMs, endMs: selected.keepEndMs)))
    }
    public static func focusedPlayback(frameMs: Int64, retained: TimeRange) -> (positionMs: Int64, endMs: Int64) {
        (frameMs < retained.startMs || frameMs >= retained.endMs ? retained.startMs : frameMs, retained.endMs)
    }

    /// Explicit Keep/Remove resolves clip and cleanup work, never serve work.
    public static func decide(_ draft: EditorDraft, rally: EditableRallyGroup?, cleanup: [SuppressionRegion], keep: Bool) -> EditorDraft {
        var result = draft
        for cut in rally?.cuts ?? [] { result = EditorMath.setIncluded(result, cutId: cut.id, included: keep) }
        result.reviewedCutIds.formUnion(rally?.cutIds ?? [])
        for suggestion in cleanup { result.suppressionDecisionOverrides[suggestion.logicalId] = keep ? "keep" : "suppress" }
        return result
    }
}

public struct RallyTrimResult: Sendable {
    public let draft: EditorDraft
    public let presentation: EditorPresentation
    public let selectedGroup: EditableRallyGroup
    public let frameMs: Int64
}

/// A linear local source-time scale devoted to the core, independent of export
/// padding. At narrow widths it reduces context to keep touch targets distinct.
public struct RallyTrimScale: Equatable, Sendable {
    public static let handleWidth = 44.0
    public static let minimumCenterGap = 52.0
    public let window: TimeRange
    public let width: Double
    public var inset: Double { Self.handleWidth / 2 }
    public var usableWidth: Double { max(1, width - Self.handleWidth) }
    public init(core: TimeRange, bounds: TimeRange, width: Double, context: TimeRange? = nil) {
        self.width = max(Self.handleWidth * 2, width)
        if let context {
            window = .init(startMs: max(bounds.startMs, context.startMs), endMs: min(bounds.endMs, context.endMs))
            return
        }
        let duration = max(1, core.endMs - core.startMs)
        let usable = max(1, self.width - Self.handleWidth)
        let separation = min(Self.minimumCenterGap, usable)
        let allowedMargin = Int64((Double(duration) * max(0, usable / separation - 1) / 2).rounded(.down))
        let margin = min(8_000, duration / 2, allowedMargin)
        window = .init(startMs: max(bounds.startMs, core.startMs - margin), endMs: min(bounds.endMs, core.endMs + margin))
    }
    public func x(_ timeMs: Int64) -> Double {
        inset + Double(timeMs - window.startMs) / Double(max(1, window.endMs - window.startMs)) * usableWidth
    }
    public func time(atX x: Double) -> Int64 {
        window.startMs + Int64((min(1, max(0, (x - inset) / usableWidth)) * Double(max(1, window.endMs - window.startMs))).rounded())
    }
    public func handles(for core: TimeRange, spreadForTouch: Bool = true) -> RallyTrimHandles {
        let left = inset, right = width - inset
        let a = min(right, max(left, x(core.startMs))), b = min(right, max(left, x(core.endMs)))
        let gap = min(Self.minimumCenterGap, usableWidth)
        guard spreadForTouch, b - a < gap else { return .init(startX: a, endX: b, spreadForTouch: false) }
        let middle = min(right - gap / 2, max(left + gap / 2, (a + b) / 2))
        return .init(startX: middle - gap / 2, endX: middle + gap / 2, spreadForTouch: true)
    }
    public func hiddenPadding(_ kept: TimeRange) -> (beforeMs: Int64, afterMs: Int64) {
        (max(0, window.startMs - kept.startMs), max(0, kept.endMs - window.endMs))
    }
}

public struct RallyTrimHandles: Equatable, Sendable {
    public let startX: Double, endX: Double
    public let spreadForTouch: Bool
}

/// Translation is anchored to the actual edge time, not the larger touch target.
/// Touch-down anywhere in a target therefore cannot jump a short core boundary.
public struct RallyTrimDrag: Sendable {
    public let scale: RallyTrimScale
    public let initialValueMs: Int64
    public let isStart: Bool
    public init(scale: RallyTrimScale, initialValueMs: Int64, isStart: Bool) {
        self.scale = scale; self.initialValueMs = initialValueMs; self.isStart = isStart
    }
    public func value(translationX: Double) -> Int64 {
        initialValueMs + Int64((translationX / scale.usableWidth * Double(max(1, scale.window.endMs - scale.window.startMs))).rounded())
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
    /// Inspecting context footage must not retarget the rally being trimmed.
    public mutating func sourceSeek(time: Int64, rallies: [EditableRallyGroup]) {
        if browsing { follow(time: time, rallies: rallies) }
    }
    public mutating func follow(time: Int64, rallies: [EditableRallyGroup]) {
        if let rally = RallyReview.at(time, in: rallies) { selectedId = rally.id }
        listCenterId = RallyReview.nearest(time, in: rallies)?.id
        following = true; followRequest += 1
    }
    public mutating func visit(_ stop: RallyReviewStop) {
        focus = stop; selectedId = stop.rallyId; browsing = false
    }
}
