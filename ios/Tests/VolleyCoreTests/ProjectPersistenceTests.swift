import Foundation
import XCTest
@testable import VolleyCore

final class ProjectPersistenceTests: XCTestCase {
    func testDeletionRemovesOnlyProjectAndCheckpointAndRejectsLateWrites() throws {
        let directory = try folder(); defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("project.volleyproject.json")
        let source = directory.appendingPathComponent("match.mp4"), exported = directory.appendingPathComponent("export.json")
        try Data([1, 2, 3]).write(to: source); try Data([4, 5]).write(to: exported)
        var project = try initial(url)
        project.draft.cuts[0].included = false
        try ProjectPersistence.checkpoint(project, to: url)
        let pending = try ProjectPersistence.stage(project, to: url)
        try ProjectPersistence.delete(project, at: url)
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: ProjectPersistence.checkpointURL(for: url).path))
        XCTAssertThrowsError(try ProjectPersistence.publish(pending, current: project))
        XCTAssertThrowsError(try ProjectPersistence.checkpoint(project, to: url))
        XCTAssertEqual(try Data(contentsOf: source), Data([1, 2, 3]))
        XCTAssertEqual(try Data(contentsOf: exported), Data([4, 5]))
    }
    func testStaleEditorCannotDeleteNewProjectRevision() throws {
        let directory = try folder(); defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("project.volleyproject.json")
        let original = try initial(url)
        let latest = try ProjectPersistence.publish(ProjectPersistence.stage(original, to: url), current: original)
        XCTAssertThrowsError(try ProjectPersistence.delete(original, at: url))
        XCTAssertEqual(try ProjectDocument.load(from: url), latest)
    }
    private func folder() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("persistence-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true); return url
    }
    private func fixture() -> ProjectDocument {
        .init(id: "same-id", sourceName: "match.mp4", durationMs: 40000,
            draft: .init(sourceRevision: "source-a", cuts: [.init(id: "R001", coreStartMs: 10000, coreEndMs: 20000)]),
            feedback: .object(["rawFeatures": .string(String(repeating: "0123456789", count: 10000)), "model": .string("original")]))
    }
    private func initial(_ url: URL) throws -> ProjectDocument {
        try ProjectPersistence.publish(ProjectPersistence.stage(fixture(), to: url))
    }
    func testDraftCheckpointIsSmallPreservesRawBaseAndLoadsLatestEdits() throws {
        let directory = try folder(); defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("project.volleyproject.json")
        var project = try initial(url)
        let base = try Data(contentsOf: url), read = try ProjectPersistence.read(from: url)
        project.draft.cuts[0].included = false; project.draft.updatedAtMs = 1234
        project.sourceName = "renamed.mp4"
        try ProjectPersistence.checkpoint(project, to: url)
        XCTAssertEqual(try Data(contentsOf: url), base)
        XCTAssertLessThan(try Data(contentsOf: ProjectPersistence.checkpointURL(for: url)).count, 10000)
        XCTAssertFalse(try ProjectPersistence.isCurrent(read))
        XCTAssertEqual(try ProjectDocument.load(from: url), project)
        XCTAssertEqual(try ProjectPersistence.listMetadata(from: url).sourceName, "renamed.mp4")
        XCTAssertEqual(try ProjectPersistence.listMetadata(from: url).updatedAtMs, 1234)
    }
    func testOldPreparedSaveAndCheckpointCannotOverwriteNewScoreFeedback() throws {
        let directory = try folder(); defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("project.volleyproject.json")
        let original = try initial(url), oldPrepared = try ProjectPersistence.stage(original, to: url)
        var modelResult = original; modelResult.feedback?["model"] = .string("new-scores")
        let newPrepared = try ProjectPersistence.stage(modelResult, to: url)
        let published = try ProjectPersistence.publish(newPrepared, current: original)
        XCTAssertThrowsError(try ProjectPersistence.publish(oldPrepared, current: original))
        XCTAssertThrowsError(try ProjectPersistence.checkpoint(original, to: url))
        XCTAssertEqual(try ProjectDocument.load(from: url), published)
        XCTAssertEqual(published.feedback?["model"]?.string, "new-scores")
        // The full base still carries exact raw feedback; no external blob is required.
        let raw = try JSONDecoder().decode(ProjectDocument.self, from: Data(contentsOf: url))
        XCTAssertEqual(raw.feedback, modelResult.feedback)
    }
    func testLateDraftEditsAndTombstonesSurvivePreparedScorePublication() throws {
        let directory = try folder(); defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("project.volleyproject.json")
        var current = try initial(url), modelResult = current
        modelResult.feedback?["model"] = .string("new-scores")
        let prepared = try ProjectPersistence.stage(modelResult, to: url)
        // These edits occur after the expensive model JSON was staged.
        current.draft.cuts[0].included = false
        current.draft.ignoredIntervals = [.init(id: "I1", startMs: 30000, endMs: 31000)]
        current.draft.scoreTracking = try ScoreTracking(enabled: false, team1Name: "User rename",
            serveMarkers: [.init(id: "S001", timestampMs: 5000, side: .far, ignorePreviousPoint: true)],
            removedModelMarkerIds: ["serve-R001"]).jsonValue()
        try ProjectPersistence.checkpoint(current, to: url)
        let output = ServingSideOutput(candidates: [.init(id: "R001", anchor: 10, intervalEnd: 20, nearProbability: 0.9, side: .near, verdict: .near)])
        let published = try ProjectPersistence.publish(prepared, current: current) { draft in
            let tracking = try ScoreTracking.fromJSON(draft.scoreTracking)
            draft.scoreTracking = try ScoreReducer.seedModelMarkers(tracking, output: output).jsonValue()
        }
        let restored = try ProjectDocument.load(from: url)
        XCTAssertEqual(restored, published); XCTAssertFalse(restored.draft.cuts[0].included)
        XCTAssertEqual(restored.draft.ignoredIntervals, current.draft.ignoredIntervals)
        let score = try ScoreTracking.fromJSON(restored.draft.scoreTracking)
        XCTAssertEqual(score.team1Name, "User rename"); XCTAssertFalse(score.enabled)
        XCTAssertEqual(score.serveMarkers.map(\.id), ["S001"]); XCTAssertTrue(score.serveMarkers[0].ignorePreviousPoint)
        XCTAssertEqual(score.removedModelMarkerIds, ["serve-R001"])
        XCTAssertEqual(restored.feedback?["model"]?.string, "new-scores")
    }
    func testSameIDImportAndLegacyRevisionCannotReuseAnotherSourcesJournal() throws {
        let directory = try folder(); defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("project.volleyproject.json")
        var old = try initial(url); old.draft.cuts[0].included = false
        try ProjectPersistence.checkpoint(old, to: url)
        var imported = fixture(); imported.draft.sourceRevision = "source-b"; imported.sourceName = "different.mp4"
        let replacement = try ProjectPersistence.publish(ProjectPersistence.stage(imported, to: url))
        XCTAssertEqual(try ProjectDocument.load(from: url), replacement)
        XCTAssertTrue(replacement.draft.cuts[0].included)
        XCTAssertThrowsError(try ProjectPersistence.checkpoint(old, to: url))
        // External legacy replacement has no generation; a managed old journal
        // cannot attach merely because its project ID happens to be the same.
        try imported.save(to: url)
        XCTAssertEqual(try ProjectDocument.load(from: url), imported)
        var legacyEdit = imported; legacyEdit.draft.cuts[0].included = false
        try ProjectPersistence.checkpoint(legacyEdit, to: url)
        XCTAssertEqual(try ProjectDocument.load(from: url), legacyEdit)
        try imported.save(to: url)
        XCTAssertEqual(try ProjectDocument.load(from: url), imported, "A replacement legacy file cannot inherit the previous file's checkpoint")
        var wrong = imported; wrong.draft.sourceRevision = "source-a"
        XCTAssertThrowsError(try ProjectPersistence.checkpoint(wrong, to: url))
    }
    func testTwoEntryJournalRecoversBothSidesOfAtomicBaseRename() throws {
        let directory = try folder(); defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("project.volleyproject.json")
        var old = fixture(); old.persistenceRevision = "old-base"
        var new = old; new.persistenceRevision = "new-base"; new.feedback?["model"] = .string("new-model")
        var oldCheckpoint = old; oldCheckpoint.draft.cuts[0].included = false
        var newCheckpoint = new; newCheckpoint.draft.cuts[0].included = false; newCheckpoint.draft.renderScoreTimeline = false
        func entry(_ project: ProjectDocument) throws -> [String: Any] {
            let raw = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(project)) as? [String: Any])
            return ["projectId": project.id, "sourceRevision": project.draft.sourceRevision, "durationMs": project.durationMs,
                    "gameWindow": raw["gameWindow"]!, "baseRevision": project.persistenceRevision!, "sourceName": project.sourceName, "draft": raw["draft"]!]
        }
        let journal = try JSONSerialization.data(withJSONObject: ["schema": "volleysplice-draft-checkpoint-v1", "entries": [entry(oldCheckpoint), entry(newCheckpoint)]])
        try old.save(to: url)
        try journal.write(to: ProjectPersistence.checkpointURL(for: url), options: .atomic)
        XCTAssertEqual(try ProjectDocument.load(from: url), oldCheckpoint, "Crash before rename keeps old feedback and all preceding user edits")
        try new.save(to: url)
        XCTAssertEqual(try ProjectDocument.load(from: url), newCheckpoint, "Crash after rename uses the pending draft with new feedback")
    }
    func testCancelledPublicationAndInvalidProjectLeaveDurableStateUntouched() async throws {
        let directory = try folder(); defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("project.volleyproject.json")
        let original = try initial(url), prepared = try ProjectPersistence.stage(original, to: url)
        let result = await Task.detached {
            withUnsafeCurrentTask { $0?.cancel() }
            return Result { try ProjectPersistence.publish(prepared, current: original) }
        }.value
        XCTAssertThrowsError(try result.get())
        XCTAssertEqual(try ProjectDocument.load(from: url), original)
        var invalid = original; invalid.id = ""
        XCTAssertThrowsError(try ProjectPersistence.stage(invalid, to: url))
        XCTAssertEqual(try ProjectDocument.load(from: url), original)
    }
}
