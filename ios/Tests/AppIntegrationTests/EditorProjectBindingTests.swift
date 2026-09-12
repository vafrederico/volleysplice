import SwiftUI
import XCTest
@testable import VolleySplice

@MainActor final class EditorProjectBindingTests: XCTestCase {
    func testNewAnalysisUsesFullFrameAndSavedProjectRetainsItsRegion() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = WorkspaceModel(documents: root, support: root.appendingPathComponent("support"))
        let full = AnalysisRegion(x: 0, y: 0, width: 1, height: 1)
        XCTAssertEqual(model.roi, full)
        var saved = project("saved-region")
        saved.feedback = .object(["source": .object(["featureRoi": .object([
            "x": .number(0.1), "y": .number(0.2), "width": .number(0.8), "height": .number(0.7)
        ])])])
        model.restoreSetup(saved)
        XCTAssertEqual(model.roi, AnalysisRegion(x: 0.1, y: 0.2, width: 0.8, height: 0.7))
        model.beginNewProject()
        XCTAssertEqual(model.roi, full)
    }
    func testDeleteCurrentProjectCancelsAutosaveAndResetsEditor() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = WorkspaceModel(documents: root, support: root.appendingPathComponent("support"))
        let original = project("delete-test"), url = model.projectURL(original)
        model.project = try ProjectPersistence.publish(ProjectPersistence.stage(original, to: url))
        model.showEditor = true
        model.project?.draft.cuts[0].included = false
        model.saveProject()
        let pending = model.saveTask
        model.deleteCurrentProject()
        await pending?.value
        XCTAssertNil(model.project); XCTAssertFalse(model.showEditor)
        XCTAssertNil(model.source); XCTAssertNil(model.media); XCTAssertNil(model.error)
        XCTAssertTrue(model.flushProject())
        model.saveProject()
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: ProjectPersistence.checkpointURL(for: url).path))
    }
    func testDeleteKeepsProjectWithUnfinishedQueueWork() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let support = root.appendingPathComponent("support")
        let store = try ProcessingJobStore(folder: support.appendingPathComponent("ProcessingJobs"))
        var ledger = ProcessingJobLedger()
        var job = ProcessingJob(kind: .scores, projectId: "queued-project", sourceName: "match.mp4", sourceFingerprint: "sampled-sha256-v1:test")
        job.state = .failed
        ledger.jobs = [job]; try store.save(ledger)
        let model = WorkspaceModel(documents: root, support: support)
        let original = project(job.projectId), url = model.projectURL(original)
        model.project = try ProjectPersistence.publish(ProjectPersistence.stage(original, to: url))
        model.showEditor = true
        model.deleteCurrentProject()
        XCTAssertEqual(model.project?.id, original.id); XCTAssertTrue(model.showEditor)
        XCTAssertTrue(model.error?.contains("Processing queue") == true)
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
    }
    private func project(_ id: String) -> ProjectDocument {
        .init(id: id, sourceName: "match.mp4", durationMs: 2000,
              draft: EditorDraft(sourceRevision: "fixture", cuts: [.init(id: "R1", coreStartMs: 100, coreEndMs: 1000)]))
    }

    func testOutgoingEditorBindingSurvivesResetWithoutRestoringProject() {
        let original = project("old")
        var current: ProjectDocument? = original
        let editor = WorkspaceView.editorBinding(Binding(get: { current }, set: { current = $0 }), snapshot: original)
        var edited = original
        edited.draft.cuts[0].included = false
        editor.wrappedValue = edited
        XCTAssertEqual(current, edited)
        XCTAssertEqual(editor.wrappedValue, edited)

        // Start a new video clears the model before SwiftUI finishes tearing down
        // its editor. A retained binding must still be readable in that interval.
        current = nil
        XCTAssertEqual(editor.wrappedValue, original)
        editor.wrappedValue = edited
        XCTAssertNil(current)
    }

    func testOutgoingEditorCannotReadOrOverwriteReplacementProject() {
        let original = project("old"), replacement = project("new")
        var current: ProjectDocument? = original
        let editor = WorkspaceView.editorBinding(Binding(get: { current }, set: { current = $0 }), snapshot: original)
        current = replacement
        XCTAssertEqual(editor.wrappedValue.id, original.id)
        var lateEdit = original
        lateEdit.draft.cuts[0].included = false
        editor.wrappedValue = lateEdit
        XCTAssertEqual(current, replacement)
    }
}
