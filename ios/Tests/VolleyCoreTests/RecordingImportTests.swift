import Foundation
import XCTest
@testable import VolleyCore

final class RecordingImportTests: XCTestCase {
    private func folders() throws -> (root: URL, documents: URL, external: URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("recording-import-test-" + UUID().uuidString)
        let documents = root.appendingPathComponent("Documents"), external = root.appendingPathComponent("External")
        try FileManager.default.createDirectory(at: documents, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: external, withIntermediateDirectories: true)
        return (root, documents, external)
    }
    func testDifferentSameNameMovieGetsUnusedSuffixAndPreservesEveryExistingFile() async throws {
        let dirs = try folders(); defer { try? FileManager.default.removeItem(at: dirs.root) }
        let original = dirs.documents.appendingPathComponent("match.mp4")
        let second = dirs.documents.appendingPathComponent("match (2).mp4")
        let external = dirs.external.appendingPathComponent("match.mp4")
        try Data("original".utf8).write(to: original)
        try Data("another existing movie".utf8).write(to: second)
        let importedBytes = Data(repeating: 73, count: 4 * 1024 * 1024 + 17)
        try importedBytes.write(to: external)
        let imported = try await RecordingImport.prepare(source: external, documents: dirs.documents)
        XCTAssertTrue(imported.createdCopy)
        XCTAssertEqual(imported.url.lastPathComponent, "match (3).mp4")
        XCTAssertEqual(try Data(contentsOf: imported.url), importedBytes)
        XCTAssertEqual(try Data(contentsOf: original), Data("original".utf8))
        XCTAssertEqual(try Data(contentsOf: second), Data("another existing movie".utf8))
        await RecordingImport.discard(imported)
        XCTAssertFalse(FileManager.default.fileExists(atPath: imported.url.path))
        XCTAssertEqual(try Data(contentsOf: external), importedBytes)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: dirs.documents.path).sorted(), ["match (2).mp4", "match.mp4"])
    }
    func testResolvedLocalPickerURLIsReusedAndCannotBeDiscarded() async throws {
        let dirs = try folders(); defer { try? FileManager.default.removeItem(at: dirs.root) }
        let source = dirs.documents.appendingPathComponent("match.mp4")
        try Data("local movie".utf8).write(to: source)
        let alias = dirs.external.appendingPathComponent("provider-alias.mp4")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: source)
        let imported = try await RecordingImport.prepare(source: alias, documents: dirs.documents)
        XCTAssertFalse(imported.createdCopy)
        XCTAssertEqual(imported.url, source.resolvingSymlinksInPath().standardizedFileURL)
        await RecordingImport.discard(imported)
        XCTAssertEqual(try Data(contentsOf: source), Data("local movie".utf8))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: dirs.documents.path), ["match.mp4"])
    }
    func testCancellationDrainsDetachedWriterAndRemovesOnlyOwnedStage() async throws {
        let dirs = try folders(); defer { try? FileManager.default.removeItem(at: dirs.root) }
        let existing = dirs.documents.appendingPathComponent("match.mp4")
        let source = dirs.external.appendingPathComponent("match.mp4")
        try Data("preserve".utf8).write(to: existing)
        try Data(repeating: 12, count: 10000).write(to: source)
        let entered = expectation(description: "detached copy opened its owned stage")
        let gate = DispatchSemaphore(value: 0)
        let task = Task {
            try await RecordingImport.prepare(source: source, documents: dirs.documents, beforeCopy: {
                entered.fulfill(); gate.wait()
            })
        }
        await fulfillment(of: [entered], timeout: 5)
        task.cancel(); gate.signal()
        do { _ = try await task.value; XCTFail("Cancelled import must not publish a movie") }
        catch is CancellationError { }
        XCTAssertEqual(try Data(contentsOf: existing), Data("preserve".utf8))
        XCTAssertEqual(try Data(contentsOf: source).count, 10000)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: dirs.documents.path), ["match.mp4"])
    }
    func testDiscardDoesNotDeleteLaterReplacementAtImportedPath() async throws {
        let dirs = try folders(); defer { try? FileManager.default.removeItem(at: dirs.root) }
        let source = dirs.external.appendingPathComponent("match.mov")
        try Data("imported".utf8).write(to: source)
        let imported = try await RecordingImport.prepare(source: source, documents: dirs.documents)
        // Keep the original inode alive elsewhere to make the replacement's
        // distinct ownership deterministic even on filesystems recycling inodes.
        let retained = dirs.external.appendingPathComponent("retained.mov")
        try FileManager.default.moveItem(at: imported.url, to: retained)
        try Data("replacement".utf8).write(to: imported.url)
        await RecordingImport.discard(imported)
        XCTAssertEqual(try Data(contentsOf: imported.url), Data("replacement".utf8))
        XCTAssertEqual(try Data(contentsOf: retained), Data("imported".utf8))
    }
}
