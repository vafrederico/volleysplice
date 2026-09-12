import XCTest
@testable import VolleyCore

final class ScoreOverlayTests: XCTestCase {
    private func prepared() -> PreparedScoreOverlay {
        .init(tracking: .init(team1Name: "Falcons", team2Name: "Waves", serveMarkers: [
            .init(id: "S1", timestampMs: 1000, side: .near),
            .init(id: "S2", timestampMs: 5000, side: .near),
            .init(id: "S3", timestampMs: 10000, side: .far)]), rallyRanges: [
                .init(coreStartMs: 1000, coreEndMs: 3000, keepStartMs: 500, keepEndMs: 3500),
                .init(coreStartMs: 5000, coreEndMs: 8000, keepStartMs: 4000, keepEndMs: 8500),
                .init(coreStartMs: 10000, coreEndMs: 13000, keepStartMs: 9000, keepEndMs: 13500)])
    }
    func testAndroidScoreLabelsAndPalette() {
        XCTAssertEqual([0, 9, 10, 123, -1].map(ScoreOverlay.formatScore), ["00", "09", "10", "123", "00"])
        XCTAssertEqual(ScoreOverlay.formatTeamLabel("Falcons", serving: true), "Falcons \u{1F3D0}")
        XCTAssertEqual(ScoreOverlay.formatTeamLabel("Falcons", serving: false), "Falcons")
        XCTAssertEqual(ScoreOverlay.team1RGB, 0xd9342b)
        XCTAssertEqual(ScoreOverlay.team2RGB, 0x2367c9)
    }
    func testPointTimelineAppearsInLeadingPaddingAndFadesInSourceTime() {
        let model = prepared()
        XCTAssertTrue(model.snapshot(at: 3999).points.isEmpty)
        let first = model.snapshot(at: 4000)
        XCTAssertEqual(first.points.map(\.serveMarkerId), ["S2"])
        XCTAssertEqual(first.points.map(\.teamPointNumber), [1])
        XCTAssertEqual(first.revealTimestampMs, 4000)
        XCTAssertEqual(first.effectiveTimestampMs, 5000)
        XCTAssertEqual(first.currentServeId, "S2")
        XCTAssertEqual(model.snapshot(at: 5125).effectiveTimestampMs, 5125)
        XCTAssertEqual([4000, 4125, 4250, 6250, 6425, 6600].map { model.snapshot(at: $0).opacity }, [0, 0.5, 1, 1, 0.5, 0])
        XCTAssertEqual(model.snapshot(at: 8999).score.servingTeamId, .team1)
        XCTAssertEqual(model.snapshot(at: 9000).score.servingTeamId, .team2)
        XCTAssertEqual(model.snapshot(at: 9250).points.map(\.teamPointNumber), [1, 1])
        XCTAssertEqual(model.snapshot(at: 9250).points.map(\.winnerTeamId), [.team1, .team2])
    }
    func testCutMappingDoesNotRestartFadeAndExactCutUsesNextSource() throws {
        let model = prepared()
        let timeline = try ExportTimeline(intervals: [.init(startMs: 4000, endMs: 5000), .init(startMs: 6000, endMs: 6800), .init(startMs: 9000, endMs: 10000)])
        XCTAssertEqual(model.snapshot(at: try XCTUnwrap(timeline.sourceTimestamp(atOutputMs: 125))).opacity, 0.5)
        XCTAssertEqual(timeline.sourceTimestamp(atOutputMs: 1000), 6000)
        XCTAssertEqual(model.snapshot(at: try XCTUnwrap(timeline.sourceTimestamp(atOutputMs: 1000))).opacity, 1)
        XCTAssertEqual(model.snapshot(at: try XCTUnwrap(timeline.sourceTimestamp(atOutputMs: 1425))).opacity, 0.5)
        XCTAssertEqual(model.snapshot(at: try XCTUnwrap(timeline.sourceTimestamp(atOutputMs: 1600))).opacity, 0)
        XCTAssertEqual(model.snapshot(at: try XCTUnwrap(timeline.sourceTimestamp(atOutputMs: 1925))).opacity, 0.5)
    }
    func testAndroidIntegerMeasurementsAndTrailingPointWindow() {
        let snapshot = PreparedScoreOverlay(tracking: .init(), rallyRanges: []).snapshot(at: 0)
        let layout = ScoreOverlay.layout(videoWidth: 1920, videoHeight: 1080, snapshot: snapshot) { _, _ in 200 }
        XCTAssertEqual(layout.team1Width, 234); XCTAssertEqual(layout.team2Width, 234)
        XCTAssertEqual(layout.width, 648); XCTAssertEqual(layout.height, 69)
        let points = (0..<38).map { ScorePointTimelineEntry(serveMarkerId: "S\($0 + 1)", winnerTeamId: $0 % 2 == 0 ? .team1 : .team2, teamPointNumber: $0 / 2 + 1) }
        let visible = ScoreOverlay.visiblePoints(videoWidth: 1920, layout: layout, points: points)
        XCTAssertEqual(visible.count, 31)
        XCTAssertEqual(visible.first?.serveMarkerId, "S8"); XCTAssertEqual(visible.last?.serveMarkerId, "S38")
        let point = ScoreOverlay.pointLayout(videoWidth: 1920, videoHeight: 1080, score: layout, pointCount: visible.count)
        XCTAssertEqual(point.startX, 648)
        XCTAssertEqual(point.columnSpacing, Float(layout.height) * 0.58, accuracy: 0.0001)
        XCTAssertLessThanOrEqual(point.columnSpacing * Float(visible.count), Float(1920 - layout.width))
        XCTAssertGreaterThan(point.team2CenterY, point.team1CenterY)
        XCTAssertGreaterThan(point.circleRadius, Float(layout.height) * 0.17)
        XCTAssertGreaterThanOrEqual(point.lineWidth, 2)
    }
    func testLongNamesAndPortraitNeverReserveNegativePointWidth() {
        let model = PreparedScoreOverlay(tracking: .init(team1Name: String(repeating: "Long first ", count: 10),
            team2Name: String(repeating: "Long second ", count: 12)), rallyRanges: [])
        for (width, height) in [(640, 480), (1920, 1080), (3840, 2160), (240, 320)] {
            let layout = ScoreOverlay.layout(videoWidth: width, videoHeight: height, snapshot: model.snapshot(at: 0)) { text, font in Double(text.count * font) * 0.55 }
            XCTAssertTrue((36...76).contains(layout.height))
            XCTAssertLessThanOrEqual(layout.width, max(Int((Double(layout.height) * 2.25).rounded()) * 2 + layout.scoreWidth * 2, Int(Double(width) * 0.96)))
            XCTAssertEqual(layout.borderWidth, min(width, height) >= 720 ? 2 : 1)
            let point = ScoreOverlay.pointLayout(videoWidth: width, videoHeight: height, score: layout, pointCount: 10)
            XCTAssertGreaterThanOrEqual(point.columnSpacing, 0)
        }
    }
    func testIgnoredReviewAndExcludedPointsDoNotBecomeTimelineEntries() {
        let tracking = ScoreTracking(serveMarkers: [
            .init(id: "S1", timestampMs: 1000, side: .near),
            .init(id: "S2", timestampMs: 5000, side: .near, ignorePreviousPoint: true),
            .init(id: "S3", timestampMs: 10000, side: .review),
            .init(id: "S4", timestampMs: 15000, side: .far),
            .init(id: "S5", timestampMs: 20000, side: .near),
            .init(id: "S6", timestampMs: 23000, side: .near, rallyId: "R6")])
        let visible = ScoreReducer.visibleTracking(tracking,
            ignoredIntervals: [.init(id: "I1", startMs: 19000, endMs: 21000, reason: "")], excludedRallyIds: ["R6"])
        let snapshot = PreparedScoreOverlay(tracking: visible, rallyRanges: []).snapshot(at: 25000)
        XCTAssertEqual(snapshot.points.map(\.serveMarkerId), ["S4"])
        XCTAssertEqual(snapshot.points.map(\.teamPointNumber), [1])
        XCTAssertEqual(snapshot.score.ignoredPointCount, 1); XCTAssertEqual(snapshot.score.reviewPointCount, 1)
        XCTAssertEqual(snapshot.revealTimestampMs, 15000); XCTAssertEqual(snapshot.opacity, 0)
    }
}
