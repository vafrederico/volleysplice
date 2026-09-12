import Foundation

public struct FinalCutInterval: Codable, Equatable, Sendable {
    public var startMs: Int64
    public var endMs: Int64
    public var cutIds: [String]
    public var joinedGaps: [TimeRange]
    public init(startMs: Int64, endMs: Int64, cutIds: [String] = [], joinedGaps: [TimeRange] = []) {
        self.startMs = startMs; self.endMs = endMs; self.cutIds = cutIds; self.joinedGaps = joinedGaps
    }
}
public struct SuppressionRegion: Codable, Equatable, Sendable {
    public var id: String
    public var logicalId: String
    public var startMs: Int64
    public var endMs: Int64
    public var eligiblePolicyIds: [String]
    public init(id: String, logicalId: String, startMs: Int64, endMs: Int64, eligiblePolicyIds: [String]) {
        self.id = id; self.logicalId = logicalId; self.startMs = startMs; self.endMs = endMs
        self.eligiblePolicyIds = eligiblePolicyIds
    }
}

public enum EditorMath {
    public static let minimumMarkMs: Int64 = 100
    public static let defaultPaddingMs: Int64 = 2_000
    public static let defaultJoinGapMs: Int64 = 3_000
    static func clamp(_ value: Int64, _ low: Int64, _ high: Int64) -> Int64 { min(max(value, low), high) }

    public static func newDraft(ranges: [EditableCut], durationMs: Int64, gameWindow: TimeRange? = nil,
                                sourceRevision: String = UUID().uuidString) -> EditorDraft {
        let window = gameWindow ?? TimeRange(startMs: 0, endMs: durationMs)
        var draft = EditorDraft(sourceRevision: sourceRevision)
        draft.cuts = ranges.compactMap { range -> EditableCut? in
            let start = clamp(range.coreStartMs, window.startMs, window.endMs)
            let end = clamp(range.coreEndMs, window.startMs, window.endMs)
            guard end > start else { return nil }
            return EditableCut(id: "", coreStartMs: start, coreEndMs: end,
                               keepStartMs: max(window.startMs, start - defaultPaddingMs),
                               keepEndMs: min(window.endMs, end + defaultPaddingMs),
                               confidence: min(max(range.confidence, 0), 1), agreement: range.agreement)
        }.enumerated().map { index, cut in var cut = cut; cut.id = String(format: "R%03d", index + 1); return cut }
        if window.startMs > 0 { draft.ignoredIntervals.append(.init(id: "G001", startMs: 0, endMs: window.startMs, reason: "outside-game-window")) }
        if window.endMs < durationMs { draft.ignoredIntervals.append(.init(id: "G002", startMs: window.endMs, endMs: durationMs, reason: "outside-game-window")) }
        return draft
    }

    public static func applyPadding(_ draft: EditorDraft, beforeMs: Int64, afterMs: Int64,
                                    durationMs: Int64, gameWindow: TimeRange? = nil) -> EditorDraft {
        var result = draft
        let window = gameWindow ?? TimeRange(startMs: 0, endMs: durationMs)
        result.beforePaddingMs = clamp(beforeMs, 0, 10_000); result.afterPaddingMs = clamp(afterMs, 0, 10_000)
        result.cuts = draft.cuts.map { cut in
            guard cut.origin == .inferred else { return cut }
            var cut = cut
            cut.keepStartMs = max(window.startMs, cut.coreStartMs - result.beforePaddingMs)
            cut.keepEndMs = min(window.endMs, cut.coreEndMs + result.afterPaddingMs)
            return cut
        }
        return result
    }
    public static func setCoreRange(_ draft: EditorDraft, cutId: String, startMs: Int64, endMs: Int64,
                                    bounds: TimeRange) -> EditorDraft {
        guard bounds.endMs - bounds.startMs >= minimumMarkMs else { return draft }
        var result = draft
        result.cuts = draft.cuts.map { cut in
            guard cut.id == cutId else { return cut }
            var updated = cut
            updated.coreStartMs = clamp(startMs, bounds.startMs, bounds.endMs - minimumMarkMs)
            updated.coreEndMs = clamp(endMs, updated.coreStartMs + minimumMarkMs, bounds.endMs)
            updated.keepStartMs = max(bounds.startMs, updated.coreStartMs - (cut.coreStartMs - cut.keepStartMs))
            updated.keepEndMs = min(bounds.endMs, updated.coreEndMs + (cut.keepEndMs - cut.coreEndMs))
            return updated
        }
        if result.cuts != draft.cuts { result.userTouchedCutIds.insert(cutId) }
        return alignRallyServeMarkers(result)
    }
    public static func setIncluded(_ draft: EditorDraft, cutId: String, included: Bool) -> EditorDraft {
        var result = draft
        if let i = result.cuts.firstIndex(where: { $0.id == cutId }) {
            result.cuts[i].included = included; result.userTouchedCutIds.insert(cutId)
        }
        return result
    }
    /// Moving one core edge never moves the opposite core edge; padding follows the edited edge.
    public static func setCoreStart(_ draft: EditorDraft, cutId: String, valueMs: Int64, minimumMs: Int64) -> EditorDraft {
        guard let cut = draft.cuts.first(where: { $0.id == cutId }), cut.coreEndMs - minimumMarkMs >= minimumMs else { return draft }
        return setCoreRange(draft, cutId: cutId, startMs: clamp(valueMs, minimumMs, cut.coreEndMs - minimumMarkMs),
                            endMs: cut.coreEndMs, bounds: .init(startMs: minimumMs, endMs: cut.keepEndMs))
    }
    public static func setCoreEnd(_ draft: EditorDraft, cutId: String, valueMs: Int64, maximumMs: Int64) -> EditorDraft {
        guard let cut = draft.cuts.first(where: { $0.id == cutId }), cut.coreStartMs + minimumMarkMs <= maximumMs else { return draft }
        return setCoreRange(draft, cutId: cutId, startMs: cut.coreStartMs,
                            endMs: clamp(valueMs, cut.coreStartMs + minimumMarkMs, maximumMs),
                            bounds: .init(startMs: cut.keepStartMs, endMs: maximumMs))
    }
    /// Outer handles change export padding. Manual ranges have no separate padding, so their core follows.
    public static func setKeepBoundary(_ draft: EditorDraft, cutId: String, valueMs: Int64, isStart: Bool, bounds: TimeRange) -> EditorDraft {
        guard let index = draft.cuts.firstIndex(where: { $0.id == cutId }) else { return draft }
        var result = draft; var cut = draft.cuts[index]
        if isStart {
            let maximum = cut.origin == .manual ? cut.keepEndMs - minimumMarkMs : cut.coreStartMs
            guard maximum >= bounds.startMs else { return draft }
            cut.keepStartMs = clamp(valueMs, bounds.startMs, maximum)
            if cut.origin == .manual { cut.coreStartMs = cut.keepStartMs }
        } else {
            let minimum = cut.origin == .manual ? cut.keepStartMs + minimumMarkMs : cut.coreEndMs
            guard minimum <= bounds.endMs else { return draft }
            cut.keepEndMs = clamp(valueMs, minimum, bounds.endMs)
            if cut.origin == .manual { cut.coreEndMs = cut.keepEndMs }
        }
        if cut != draft.cuts[index] { result.cuts[index] = cut; result.userTouchedCutIds.insert(cutId) }
        return alignRallyServeMarkers(result)
    }
    public static func resetExtraTime(_ draft: EditorDraft, cutIds: Set<String>, bounds: TimeRange) -> EditorDraft {
        var result = draft
        for index in result.cuts.indices where cutIds.contains(result.cuts[index].id) && result.cuts[index].origin == .inferred {
            let original = result.cuts[index]
            result.cuts[index].keepStartMs = max(bounds.startMs, original.coreStartMs - draft.beforePaddingMs)
            result.cuts[index].keepEndMs = min(bounds.endMs, original.coreEndMs + draft.afterPaddingMs)
            if original != result.cuts[index] { result.userTouchedCutIds.insert(original.id) }
        }
        return result
    }
    public static func deleteManualCuts(_ draft: EditorDraft, cutIds: Set<String>) -> EditorDraft {
        let removed = Set(draft.cuts.filter { cutIds.contains($0.id) && $0.origin == .manual }.map(\.id))
        var result = draft
        result.cuts.removeAll { removed.contains($0.id) }
        result.reviewedCutIds.subtract(removed); result.userTouchedCutIds.subtract(removed)
        return result
    }
    public static func splitCut(_ draft: EditorDraft, cutId: String, positionMs: Int64,
                                bounds: TimeRange) -> EditorDraft? {
        guard let i = draft.cuts.firstIndex(where: { $0.id == cutId }) else { return nil }
        let cut = draft.cuts[i]
        guard positionMs >= cut.coreStartMs + minimumMarkMs, positionMs <= cut.coreEndMs - minimumMarkMs else { return nil }
        var result = draft; var left = cut; var right = cut
        left.coreEndMs = positionMs; left.keepEndMs = min(bounds.endMs, positionMs + cut.keepEndMs - cut.coreEndMs)
        right.id = nextId(prefix: cut.origin == .inferred ? "R" : "M", ids: draft.cuts.map(\.id))
        right.coreStartMs = positionMs; right.keepStartMs = max(bounds.startMs, positionMs - cut.coreStartMs + cut.keepStartMs)
        result.cuts[i] = left; result.cuts.insert(right, at: i + 1)
        result.userTouchedCutIds.formUnion([left.id, right.id])
        return result
    }
    public static func addManual(_ draft: EditorDraft, startMs: Int64, endMs: Int64, bounds: TimeRange) -> EditorDraft {
        let start = clamp(min(startMs, endMs), bounds.startMs, bounds.endMs)
        let end = clamp(max(startMs, endMs), bounds.startMs, bounds.endMs)
        guard end - start >= minimumMarkMs else { return draft }
        var result = draft
        result.cuts.append(.init(id: nextId(prefix: "M", ids: draft.cuts.map(\.id)), coreStartMs: start, coreEndMs: end, origin: .manual))
        result.pendingManualStartMs = nil
        return result
    }
    public static func addIgnored(_ draft: EditorDraft, startMs: Int64, endMs: Int64, durationMs: Int64, reason: String) -> EditorDraft {
        let start = clamp(min(startMs, endMs), 0, durationMs); let end = clamp(max(startMs, endMs), 0, durationMs)
        guard end - start >= minimumMarkMs else { return draft }
        var result = draft
        result.ignoredIntervals.append(.init(id: nextId(prefix: "I", ids: draft.ignoredIntervals.map(\.id)), startMs: start, endMs: end, reason: reason))
        result.pendingIgnoreStartMs = nil
        return result
    }
    public static func nextId(prefix: String, ids: [String]) -> String {
        let maximum = ids.filter { $0.hasPrefix(prefix) }.compactMap { Int($0.dropFirst(prefix.count)) }.max() ?? 0
        return String(format: "%@%03d", prefix, maximum + 1)
    }
    public static func alignRallyServeMarkers(_ draft: EditorDraft) -> EditorDraft {
        guard let markers = draft.scoreTracking["serveMarkers"]?.array else { return draft }
        var result = draft
        result.scoreTracking["serveMarkers"] = .array(markers.map { marker in
            var marker = marker
            if let id = marker["rallyId"]?.string, let cut = draft.cuts.first(where: { $0.id == id }) {
                marker["timestamp"] = .number(Double(cut.coreStartMs) / 1_000)
            }
            return marker
        }.sorted { ($0["timestamp"]?.double ?? 0) < ($1["timestamp"]?.double ?? 0) })
        return result
    }
    public static func effectiveDecision(_ draft: EditorDraft, suggestion: SuppressionRegion) -> String {
        if let explicit = draft.suppressionDecisionOverrides[suggestion.logicalId] { return explicit }
        if draft.cuts.contains(where: { $0.origin == .inferred && draft.userTouchedCutIds.contains($0.id) &&
            $0.coreStartMs < suggestion.endMs && suggestion.startMs < $0.coreEndMs }) { return "keep" }
        return draft.suppressionInitialBehavior == "disable-initially" ? "suppress" : "keep"
    }
    /// Suppress, pad/clip, union/join strict positive gaps, subtract ignored spans; never rejoin.
    /// App/export callers must pass the known analysis bounds: the last kept cut cannot establish source duration.
    /// The optional default preserves the legacy pure-interval API for callers with no padding beyond outer edges.
    public static func finalIntervals(_ draft: EditorDraft, suppression: [SuppressionRegion] = [], bounds: TimeRange? = nil) -> [FinalCutInterval] {
        let active = suppression.filter { draft.selectedSuppressionPolicy != "none" &&
            $0.eligiblePolicyIds.contains(draft.selectedSuppressionPolicy) && effectiveDecision(draft, suggestion: $0) == "suppress" }
        let whole = active.filter { (draft.suppressionScopeOverrides[$0.logicalId] ?? "whole-rally") == "whole-rally" }
        let veto = active.filter { draft.suppressionScopeOverrides[$0.logicalId] == "veto-region" }
        let removed = draft.cuts.filter { cut in cut.origin == .inferred && whole.contains {
            cut.coreStartMs < $0.endMs && $0.startMs < cut.coreEndMs } }
        let barriers = veto.map { TimeRange(startMs: $0.startMs, endMs: $0.endMs) } + removed.map { TimeRange(startMs: $0.keepStartMs, endMs: $0.keepEndMs) }
        let source: [FinalCutInterval] = draft.cuts.filter(\.included).flatMap { cut -> [FinalCutInterval] in
            if cut.origin == .manual { return [.init(startMs: cut.keepStartMs, endMs: cut.keepEndMs, cutIds: [cut.id])] }
            if removed.contains(where: { $0.id == cut.id }) { return [] }
            var fragments = [TimeRange(startMs: cut.coreStartMs, endMs: cut.coreEndMs)]
            for region in veto { fragments = fragments.flatMap { subtract($0, TimeRange(startMs: region.startMs, endMs: region.endMs)) } }
            return fragments.flatMap { fragment -> [FinalCutInterval] in
                let rawStart = fragment.startMs == cut.coreStartMs ? cut.keepStartMs : max(0, fragment.startMs - draft.beforePaddingMs)
                let sum = fragment.endMs.addingReportingOverflow(draft.afterPaddingMs)
                let rawEnd = fragment.endMs == cut.coreEndMs ? cut.keepEndMs : sum.overflow ? Int64.max : sum.partialValue
                let start = max(bounds?.startMs ?? 0, rawStart)
                let end = min(bounds?.endMs ?? Int64.max, rawEnd)
                guard end > start else { return [] }
                var padded = [TimeRange(startMs: start, endMs: end)]
                for region in veto { padded = padded.flatMap { subtract($0, TimeRange(startMs: region.startMs, endMs: region.endMs)) } }
                return padded.map { .init(startMs: $0.startMs, endMs: $0.endMs, cutIds: [cut.id]) }
            }
        }.filter { $0.endMs > $0.startMs }
        var merged: [FinalCutInterval] = []
        let threshold = clamp(draft.joinGapMs, 0, 10_000)
        for interval in source.sorted(by: { ($0.startMs, $0.endMs) < ($1.startMs, $1.endMs) }) {
            guard var previous = merged.last else { merged.append(interval); continue }
            let gap = interval.startMs - previous.endMs
            let barrier = gap > 0 && barriers.contains { $0.startMs < interval.startMs && previous.endMs < $0.endMs }
            if gap > 0 && (gap >= threshold || barrier) { merged.append(interval) }
            else {
                if gap > 0 { previous.joinedGaps.append(.init(startMs: previous.endMs, endMs: interval.startMs)) }
                previous.endMs = max(previous.endMs, interval.endMs)
                previous.cutIds += interval.cutIds.filter { !previous.cutIds.contains($0) }
                merged[merged.count - 1] = previous
            }
        }
        for ignored in draft.ignoredIntervals {
            merged = merged.flatMap { interval in
                subtract(.init(startMs: interval.startMs, endMs: interval.endMs), .init(startMs: ignored.startMs, endMs: ignored.endMs)).map {
                    FinalCutInterval(startMs: $0.startMs, endMs: $0.endMs, cutIds: interval.cutIds, joinedGaps: interval.joinedGaps)
                }
            }
        }
        return merged.map { interval in
            var result = interval
            result.cutIds = interval.cutIds.filter { id in source.contains {
                $0.cutIds.contains(id) && min($0.endMs, interval.endMs) > max($0.startMs, interval.startMs) } }
            result.joinedGaps = interval.joinedGaps.compactMap { gap in
                let start = max(gap.startMs, interval.startMs); let end = min(gap.endMs, interval.endMs)
                return end > start ? .init(startMs: start, endMs: end) : nil
            }
            return result
        }
    }
    public static func subtract(_ range: TimeRange, _ excluded: TimeRange) -> [TimeRange] {
        guard excluded.startMs < range.endMs, excluded.endMs > range.startMs else { return [range] }
        var result: [TimeRange] = []
        if excluded.startMs > range.startMs { result.append(.init(startMs: range.startMs, endMs: min(range.endMs, excluded.startMs))) }
        if excluded.endMs < range.endMs { result.append(.init(startMs: max(range.startMs, excluded.endMs), endMs: range.endMs)) }
        return result
    }
    public static func totalFinalMs(_ intervals: [FinalCutInterval]) -> Int64 { intervals.reduce(0) { $0 + $1.endMs - $1.startMs } }
}
