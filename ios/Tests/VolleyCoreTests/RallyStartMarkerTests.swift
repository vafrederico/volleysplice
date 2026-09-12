import XCTest
@testable import VolleyCore

/// Android RallyStartMarkerTest fixtures: moving a rally start is a linked-marker
/// edit, not a new model inference or a reset of the user's correction.
final class RallyStartMarkerTests: XCTestCase {
    private let bounds = TimeRange(startMs: 0, endMs: 40000)
    private func base() throws -> EditorDraft {
        var draft = EditorDraft(sourceRevision: "fixture", cuts: [
            .init(id: "R001", coreStartMs: 10000, coreEndMs: 20000, keepStartMs: 8000, keepEndMs: 22000),
            .init(id: "R002", coreStartMs: 25000, coreEndMs: 30000, keepStartMs: 23000, keepEndMs: 32000)])
        draft.scoreTracking = try ScoreTracking(enabled: false, serveMarkers: [
            .init(id: "serve-R001", timestampMs: 10000, side: .far, origin: .model, modelSide: .near, ignorePreviousPoint: true, rallyId: "R001"),
            .init(id: "S001", timestampMs: 10000, side: .near),
            .init(id: "serve-R002", timestampMs: 25000, side: .far, origin: .model, modelSide: .near, ignorePreviousPoint: true, rallyId: "R002")],
            sideSwitchMarkers: [.init(id: "X001", timestampMs: 10000)], removedModelMarkerIds: ["serve-R003"]).jsonValue()
        return draft
    }
    private func tracking(_ draft: EditorDraft) throws -> ScoreTracking { try ScoreTracking.fromJSON(draft.scoreTracking, durationMs: 40000) }
    private func linked(_ draft: EditorDraft) throws -> ServeMarker { try XCTUnwrap(tracking(draft).serveMarkers.first { $0.id == "serve-R001" }) }

    func testForwardAndBackwardStartsPreserveCorrectionsReplayIndependentMarkersAndToggle() throws {
        let original = try base(), originalTracking = try tracking(original)
        let forward = EditorMath.setCoreStart(original, cutId: "R001", valueMs: 12000, minimumMs: 0)
        XCTAssertEqual(forward.cuts[0].keepStartMs, 10000)
        var expected = try linked(original); expected.timestampMs = 12000
        XCTAssertEqual(try linked(forward), expected)
        let backward = EditorMath.setCoreStart(forward, cutId: "R001", valueMs: 8000, minimumMs: 0)
        XCTAssertEqual(backward.cuts[0].keepStartMs, 6000)
        expected.timestampMs = 8000; XCTAssertEqual(try linked(backward), expected)
        let restored = try tracking(backward)
        XCTAssertEqual(restored.serveMarkers.filter { $0.id != expected.id }, originalTracking.serveMarkers.filter { $0.id != expected.id })
        XCTAssertEqual(restored.sideSwitchMarkers, originalTracking.sideSwitchMarkers)
        XCTAssertEqual(restored.removedModelMarkerIds, originalTracking.removedModelMarkerIds)
        XCTAssertFalse(restored.enabled)
        XCTAssertEqual(try JSONDecoder().decode(EditorDraft.self, from: JSONEncoder().encode(backward)), backward)
    }
    func testClampedAndRangeEditsMoveMarkerToActualCoreStart() throws {
        let original = try base()
        XCTAssertEqual(try linked(EditorMath.setCoreStart(original, cutId: "R001", valueMs: -1000, minimumMs: 1000)).timestampMs, 1000)
        XCTAssertEqual(try linked(EditorMath.setCoreStart(original, cutId: "R001", valueMs: 50000, minimumMs: 0)).timestampMs, 19900)
        XCTAssertEqual(try linked(EditorMath.setCoreRange(original, cutId: "R001", startMs: 13000, endMs: 18000, bounds: bounds)).timestampMs, 13000)
    }
    func testPaddingEndAndSplitDoNotMoveOrDuplicateServeMarkers() throws {
        let original = try base(), originalTracking = try tracking(original)
        XCTAssertEqual(try tracking(EditorMath.applyPadding(original, beforeMs: 3000, afterMs: 1000, durationMs: 40000)), originalTracking)
        XCTAssertEqual(try tracking(EditorMath.setCoreEnd(original, cutId: "R001", valueMs: 18000, maximumMs: 40000)), originalTracking)
        let split = try XCTUnwrap(EditorMath.splitCut(original, cutId: "R001", positionMs: 15000, bounds: bounds))
        let newId = try XCTUnwrap(split.cuts.first { $0.id != "R001" && $0.id != "R002" }?.id)
        XCTAssertEqual(try tracking(EditorMath.setCoreStart(split, cutId: newId, valueMs: 16000, minimumMs: 0)), originalTracking)
    }
    func testAlignmentRepairsStaleLinkedMarkersAndNeverResurrectsRemovedMarkers() throws {
        var stale = try base(); stale.cuts[0].coreStartMs = 12000; stale.cuts[0].keepStartMs = 10000
        let repaired = EditorMath.alignRallyServeMarkers(stale)
        XCTAssertEqual(try linked(repaired).timestampMs, 12000)
        XCTAssertEqual(EditorMath.alignRallyServeMarkers(repaired), repaired)
        var removed = try base()
        removed.scoreTracking = try ScoreReducer.removeServe(tracking(removed), markerId: "serve-R001").jsonValue()
        XCTAssertEqual(try tracking(EditorMath.setCoreStart(removed, cutId: "R001", valueMs: 12000, minimumMs: 0)), try tracking(removed))
        var manual = EditorDraft(sourceRevision: "fixture", cuts: [.init(id: "M001", coreStartMs: 10000, coreEndMs: 20000, origin: .manual)])
        manual.scoreTracking = try ScoreTracking(serveMarkers: [.init(id: "S001", timestampMs: 10000, side: .near, rallyId: "M001")]).jsonValue()
        let moved = EditorMath.setKeepBoundary(manual, cutId: "M001", valueMs: 9000, isStart: true, bounds: bounds)
        XCTAssertEqual(try tracking(moved).serveMarkers.first?.timestampMs, 9000)
    }
}
