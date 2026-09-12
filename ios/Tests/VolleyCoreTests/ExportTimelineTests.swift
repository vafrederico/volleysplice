import XCTest
@testable import VolleyCore

final class ExportTimelineTests: XCTestCase {
    func testExactCutBoundaryMapsToNextSourceAndRetainedGapSurvives() throws {
        let timeline = try ExportTimeline(intervals: [.init(startMs: 3000, endMs: 17000), .init(startMs: 28000, endMs: 44000)])
        XCTAssertEqual(timeline.durationMs, 30000)
        XCTAssertEqual(timeline.sourceTimestamp(atOutputMs: 0), 3000)
        XCTAssertEqual(timeline.sourceTimestamp(atOutputMs: 13999), 16999)
        XCTAssertEqual(timeline.sourceTimestamp(atOutputMs: 14000), 28000)
        XCTAssertNil(timeline.sourceTimestamp(atOutputMs: 30000))
        XCTAssertNil(timeline.sourceTimestamp(atOutputMs: -1))
        let joined = try ExportTimeline(intervals: [.init(startMs: 3000, endMs: 44000)])
        XCTAssertEqual(joined.sourceTimestamp(atOutputMs: 17000), 20000)
        XCTAssertThrowsError(try ExportTimeline(intervals: [.init(startMs: 3, endMs: 2)]))
        XCTAssertThrowsError(try ExportTimeline(intervals: [.init(startMs: 0, endMs: 10), .init(startMs: 9, endMs: 20)]))
    }
    func testScoreAppearsInLeadingPaddingButDoesNotJumpInsideJoinedDeadGap() throws {
        let tracking = ScoreTracking(serveMarkers: [.init(id: "S1", timestampMs: 5000, side: .near),
            .init(id: "S2", timestampMs: 20000, side: .far)])
        let ranges: [ScoreRallyRange] = [.init(coreStartMs: 5000, coreEndMs: 12000, keepStartMs: 3000, keepEndMs: 14000),
            .init(coreStartMs: 20000, coreEndMs: 25000, keepStartMs: 16000, keepEndMs: 27000)]
        let timeline = try ExportTimeline(intervals: [.init(startMs: 3000, endMs: 27000)])
        let spans = timeline.scoreSpans(tracking: tracking, rallyRanges: ranges)
        for span in spans {
            for source in [span.source.startMs, span.source.endMs - 1] {
                let boundary = ScoreReducer.scoreBoundaryTimestamp(source, rallyRanges: ranges, tracking: tracking)
                XCTAssertEqual(span.score, ScoreReducer.deriveAt(tracking, sourceTimestampMs: boundary))
            }
        }
        XCTAssertEqual(spans.first { $0.source.startMs == 14000 }?.score.team2Score, 0)
        XCTAssertEqual(spans.first { $0.source.startMs == 16000 }?.score.team2Score, 1)
        XCTAssertEqual(ExportTimeline.pointRevealTimestamp(20000, tracking: tracking, rallyRanges: ranges), 16000)
    }
    func testPointTimelineFadeMatchesAndroidSourceAge() {
        let ages: [Int64] = [-1, 0, 125, 250, 2249, 2250, 2425, 2600]
        let expected: [Float] = [0, 0, 0.5, 1, 1, 1, 0.5, 0]
        XCTAssertEqual(ages.map { ExportTimeline.pointTimelineOpacity(sourceTimestampMs: 1000 + $0, revealTimestampMs: 1000) }, expected)
    }
}
