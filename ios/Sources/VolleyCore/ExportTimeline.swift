import Foundation

/// Exact source/output mapping shared by video export and its score overlay.
public struct ExportTimeline: Equatable, Sendable {
    public struct Span: Equatable, Sendable {
        public var source: TimeRange
        public var outputStartMs: Int64
        public var outputEndMs: Int64 { outputStartMs + source.endMs - source.startMs }
    }
    public struct ScoreSpan: Equatable, Sendable {
        public var source: TimeRange
        public var outputStartMs: Int64
        public var outputEndMs: Int64 { outputStartMs + source.endMs - source.startMs }
        public var score: DerivedScore
    }
    public let spans: [Span]
    public var durationMs: Int64 { spans.last?.outputEndMs ?? 0 }
    public init(intervals: [FinalCutInterval]) throws {
        var cursor: Int64 = 0, result: [Span] = [], previousEnd: Int64 = 0
        for interval in intervals {
            guard interval.startMs >= previousEnd, interval.endMs > interval.startMs else {
                throw ProjectError.invalid("Export intervals must be ordered, positive and disjoint")
            }
            let (next, overflow) = cursor.addingReportingOverflow(interval.endMs - interval.startMs)
            guard !overflow else { throw ProjectError.invalid("Export timeline duration overflow") }
            result.append(.init(source: .init(startMs: interval.startMs, endMs: interval.endMs), outputStartMs: cursor))
            cursor = next; previousEnd = interval.endMs
        }
        spans = result
    }
    public func sourceTimestamp(atOutputMs timestamp: Int64) -> Int64? {
        guard let span = spans.first(where: { timestamp >= $0.outputStartMs && timestamp < $0.outputEndMs }) else { return nil }
        return span.source.startMs + timestamp - span.outputStartMs
    }
    public func scoreSpans(tracking: ScoreTracking, rallyRanges: [ScoreRallyRange]) -> [ScoreSpan] {
        let changes = Set(tracking.serveMarkers.map(\.timestampMs) +
            rallyRanges.flatMap { [$0.keepStartMs, $0.coreStartMs, $0.coreEndMs, $0.keepEndMs] })
        return spans.flatMap { span in
            let boundaries = ([span.source.startMs, span.source.endMs] +
                Array(changes.filter { $0 > span.source.startMs && $0 < span.source.endMs })).sorted()
            return zip(boundaries, boundaries.dropFirst()).map { start, end in
                let boundary = ScoreReducer.scoreBoundaryTimestamp(start, rallyRanges: rallyRanges, tracking: tracking)
                return ScoreSpan(source: .init(startMs: start, endMs: end), outputStartMs: span.outputStartMs + start - span.source.startMs,
                                 score: ScoreReducer.deriveAt(tracking, sourceTimestampMs: boundary))
            }
        }
    }
    /// Matches Android's score timeline reveal in leading padding, in source time.
    public static func pointRevealTimestamp(_ timestampMs: Int64, tracking: ScoreTracking, rallyRanges: [ScoreRallyRange]) -> Int64 {
        let matching = rallyRanges.filter { timestampMs >= $0.keepStartMs && timestampMs < $0.keepEndMs }
        let containing = matching.min { a, b in
            let aCore = timestampMs >= a.coreStartMs && timestampMs < a.coreEndMs ? 0 : 1
            let bCore = timestampMs >= b.coreStartMs && timestampMs < b.coreEndMs ? 0 : 1
            return (aCore, abs(a.coreStartMs - timestampMs)) < (bCore, abs(b.coreStartMs - timestampMs))
        }
        guard let containing else { return timestampMs }
        let candidate = containing.keepStartMs
        return ScoreReducer.scoreBoundaryTimestamp(candidate, rallyRanges: rallyRanges, tracking: tracking) >= timestampMs ? candidate : timestampMs
    }
    public static func pointTimelineOpacity(sourceTimestampMs: Int64, revealTimestampMs: Int64) -> Float {
        let age = sourceTimestampMs - revealTimestampMs
        if age < 0 || age >= 2600 { return 0 }
        if age < 250 { return Float(age) / 250 }
        if age < 2250 { return 1 }
        return 1 - Float(age - 2250) / 350
    }
}
