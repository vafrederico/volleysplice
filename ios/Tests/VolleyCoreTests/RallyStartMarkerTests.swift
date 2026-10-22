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

    private func importedNeuralProject(_ model: RallyModel = .maximumCoverage) throws -> ProjectDocument {
        let ranges: [EditableCut] = [
            .init(id: "N007", coreStartMs: 10000, coreEndMs: 20000, confidence: 0.9, agreement: "neural"),
            .init(id: "N042", coreStartMs: 25000, coreEndMs: 30000, confidence: 0.8, agreement: "neural")]
        var current = EditorDraft(sourceRevision: "imported-analysis", cuts: ranges)
        current.scoreTracking = try ScoreTracking(serveMarkers: [
            .init(id: "serve-N007", timestampMs: 12000, side: .far, origin: .model, modelSide: .near, rallyId: "N007"),
            .init(id: "serve-N042", timestampMs: 25000, side: .near, origin: .model, modelSide: .near, rallyId: "N042")],
            sideSwitchMarkers: [.init(id: "X001", timestampMs: 22000)], removedModelMarkerIds: ["serve-N099"]).jsonValue()
        current.renderScoreOverlay = false; current.renderScoreTimeline = false; current.fadeScoreOverlay = true
        let bundle = try ProjectArchive.create(source: ProjectArchive.source(projectId: "synthetic-neural",
            name: "synthetic.mp4", sizeBytes: 100, lastModifiedMs: 0, duration: 40, width: 320, height: 240),
            initialInference: ProjectArchive.initialInference(ranges: ranges, rallyModel: model), draft: current, analysis: nil)
        var project = try ProjectArchive.importBundle(bundle)
        // Restore is invoked after edits; keep a corrected side and intentionally stale linked time.
        project.draft.cuts[0].coreStartMs = 12000
        project.draft.cuts[0].keepStartMs = 11000
        project.draft.scoreTracking = current.scoreTracking
        project.draft.renderScoreOverlay = false; project.draft.renderScoreTimeline = false; project.draft.fadeScoreOverlay = true
        return project
    }

    func testRestorePreservesImportedIdsAndTheirScoreLinksThroughTrimRemovalAndReimport() throws {
        for model in RallyModel.allCases {
            var project = try importedNeuralProject(model)
            let originalTracking = try tracking(project.draft)
            let restored = try ProjectArchive.restoredModelDraft(project)
            XCTAssertEqual(restored.cuts.map(\.id), ["N007", "N042"])
            XCTAssertEqual(restored.cuts.map(\.coreStartMs), [10000, 25000])
            XCTAssertEqual(restored.sourceRevision, project.draft.sourceRevision)
            XCTAssertFalse(restored.renderScoreOverlay); XCTAssertFalse(restored.renderScoreTimeline)
            XCTAssertEqual(restored.fadeScoreOverlay, true)
            var expected = originalTracking; expected.serveMarkers[0].timestampMs = 10000
            XCTAssertEqual(try tracking(restored), expected)
            let trimmed = EditorMath.setCoreStart(restored, cutId: "N007", valueMs: 13000, minimumMs: 0)
            XCTAssertEqual(try tracking(trimmed).serveMarkers.first?.timestampMs, 13000)
            let beforeRemoval = EditorPresentation(draft: trimmed, suggestions: [], bounds: bounds, durationMs: 40000)
            XCTAssertEqual(ScoreReducer.deriveAt(beforeRemoval.visibleScore).points.filter { $0.status == .counted }.count, 1)
            let removed = EditorMath.setIncluded(trimmed, cutId: "N007", included: false)
            let presentation = EditorPresentation(draft: removed, suggestions: [], bounds: bounds, durationMs: 40000)
            XCTAssertEqual(presentation.visibleScore.serveMarkers.map(\.rallyId), ["N042"])
            XCTAssertEqual(ScoreReducer.deriveAt(presentation.visibleScore).points.filter { $0.status == .counted }.count, 0)
            project.draft = removed
            let reopened = try JSONDecoder().decode(ProjectDocument.self, from: JSONEncoder().encode(project))
            var bundle = try XCTUnwrap(reopened.feedback)
            try ProjectArchive.updateCorrections(&bundle, draft: reopened.draft)
            let imported = try ProjectArchive.importBundle(bundle)
            XCTAssertEqual(imported.draft.cuts, removed.cuts)
            XCTAssertEqual(try tracking(imported.draft), try tracking(removed))
            XCTAssertEqual(bundle["corrections"]?["scoreTracking"]?["excludedRallyIds"]?.array?.compactMap(\.string), ["N007"])
        }
    }

    func testRestoreKeepsManualMarkersCorrectionsAndRemovalTombstones() throws {
        var project = try importedNeuralProject()
        var score = try tracking(project.draft)
        score = ScoreReducer.removeServe(score, markerId: "serve-N007")
        score.serveMarkers.append(.init(id: "manual-independent", timestampMs: 15000, side: .review, ignorePreviousPoint: true))
        score.enabled = false
        project.draft.scoreTracking = try score.jsonValue()
        score.serveMarkers.sort { ($0.timestampMs, $0.id) < ($1.timestampMs, $1.id) }
        XCTAssertEqual(try tracking(ProjectArchive.restoredModelDraft(project)), score)
    }

    func testRestoreUsesLegacyFallbackOnlyForMissingIdsWithoutRenumberingAfterWindowClipping() throws {
        var project = try importedNeuralProject()
        project.gameWindow = .init(startMs: 21000, endMs: 35000)
        var ranges = try XCTUnwrap(project.feedback?["initialInference"]?["ranges"]?.array)
        ranges[0]["id"] = nil; ranges[1]["id"] = nil
        project.feedback?["initialInference"]?["ranges"] = .array(ranges)
        XCTAssertEqual(try ProjectArchive.restoredModelDraft(project).cuts.map(\.id), ["R002"])
    }

    func testRestoreRejectsInvalidOrCollidingOriginalIdsWithoutChangingCurrentReview() throws {
        let original = try importedNeuralProject()
        for value in [JSONValue.string(""), .number(7), .null, .string("N042")] {
            var project = original
            project.feedback?["initialInference"]?["ranges"] = .array([
                .object(["id": value, "start": .number(10), "end": .number(20), "confidence": .number(1)]),
                .object(["id": .string("N042"), "start": .number(25), "end": .number(30), "confidence": .number(1)])])
            XCTAssertThrowsError(try ProjectArchive.restoredModelDraft(project))
            XCTAssertEqual(project.draft, original.draft)
        }
        var collision = original
        collision.feedback?["initialInference"]?["ranges"] = .array([
            .object(["start": .number(10), "end": .number(20)]),
            .object(["id": .string("R001"), "start": .number(25), "end": .number(30)])])
        XCTAssertThrowsError(try ProjectArchive.restoredModelDraft(collision))
    }
}
