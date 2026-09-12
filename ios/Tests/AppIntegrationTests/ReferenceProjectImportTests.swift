import XCTest
@testable import VolleySplice

/// Opt-in lab acceptance using the real full iPad export and its original movie.
/// Stage both files in the simulator's Documents; no inference is performed.
@MainActor final class ReferenceProjectImportTests: XCTestCase {
    func testImportFullIPadReferenceWithoutAnalysis() async throws {
        let model = WorkspaceModel()
        let input = model.documents.appendingPathComponent("Reference-iPad-full.model-feedback.json")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: input.path), "Stage the real iPad reference export first")
        let original = try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: input))
        let source = model.documents.appendingPathComponent("tds6-reference.mp4")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: source.path), "Stage the matching original reference video")
        let jobs = model.queue.jobs
        model.openProject(input)
        let deadline = Date().addingTimeInterval(60)
        while model.busy && Date() < deadline { try await Task.sleep(for: .milliseconds(100)) }
        XCTAssertFalse(model.busy)
        XCTAssertNil(model.error)
        XCTAssertFalse(model.reconnecting)
        XCTAssertTrue(model.showEditor)
        let project = try XCTUnwrap(model.project)
        XCTAssertEqual(project.durationMs, 1_105_817)
        XCTAssertEqual(project.gameWindow, TimeRange(startMs: 0, endMs: 1_105_817))
        XCTAssertEqual(project.feedback?["features"], original["features"])
        XCTAssertEqual(project.feedback?["initialInference"], original["initialInference"])
        XCTAssertEqual(model.queue.jobs, jobs, "Import must not queue analysis")
        let fingerprint = try await SourceVideoLibrary.inspect(SourceVideoAccess(source))
        XCTAssertEqual(fingerprint, original["source"]?["file"]?["sampledFingerprint"]?.string)
        let intervals = EditorMath.finalIntervals(project.draft,
            suppression: try ProjectArchive.suppressionRegions(original), bounds: project.gameWindow)
        let receipt: [String: Any] = ["projectId": project.id, "cutCount": project.draft.cuts.count,
            "finalDurationMs": EditorMath.totalFinalMs(intervals), "newAnalysisJobs": 0]
        try JSONSerialization.data(withJSONObject: receipt, options: .prettyPrinted)
            .write(to: model.documents.appendingPathComponent("reference-import-receipt.json"))
        model.player.pause()
    }
}
