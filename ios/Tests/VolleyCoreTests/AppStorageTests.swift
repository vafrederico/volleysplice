import Foundation
import XCTest
@testable import VolleyCore

final class AppStorageTests: XCTestCase {
    private func fixture() throws -> (URL, AppStorageInventory.Roots) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("storage-test-" + UUID().uuidString).resolvingSymlinksInPath()
        let roots = AppStorageInventory.Roots(documents: root.appendingPathComponent("Documents"), features: root.appendingPathComponent("Caches/features"),
                                     queue: root.appendingPathComponent("Support/ProcessingJobs"), temporary: root.appendingPathComponent("tmp"))
        for directory in roots.all { try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true) }
        return (root, roots)
    }
    private func write(_ url: URL, _ value: String = "fixture") throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(value.utf8).write(to: url)
    }
    func testInventoryCountsPrivateFilesAndGroupsProjectWithItsEdits() throws {
        let (root, roots) = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let project = roots.documents.appendingPathComponent("project-game.volleyproject.json")
        try write(project, "project")
        try write(ProjectPersistence.checkpointURL(for: project), "edits")
        try write(roots.documents.appendingPathComponent("match.mp4"), "video")
        try write(roots.features.appendingPathComponent("features.plist"), "features")
        try write(roots.documents.appendingPathComponent(".recording-import-123/recording.partial"), "partial")
        try write(roots.queue.appendingPathComponent("queue.json"), "ledger")
        let items = try AppStorageInventory.scan(roots: roots)
        XCTAssertEqual(items.reduce(0) { $0 + $1.bytes }, 38)
        XCTAssertEqual(items.filter { $0.category == .projects }.count, 1)
        let saved = try XCTUnwrap(items.first { $0.category == .projects })
        XCTAssertEqual(saved.bytes, 12)
        XCTAssertEqual(items.first { $0.name == "recording.partial" }?.category, .temporary)
        XCTAssertFalse(try XCTUnwrap(items.first { $0.name == "queue.json" }).removable)
        try AppStorageInventory.remove([saved], roots: roots, protection: .init())
        XCTAssertFalse(FileManager.default.fileExists(atPath: project.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: ProjectPersistence.checkpointURL(for: project).path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: roots.documents.appendingPathComponent("match.mp4").path))
    }
    func testUnfinishedInputsAndQueueLedgerAreProtectedButUnusedSnapshotsCanBeRemoved() throws {
        let (root, roots) = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let activeID = UUID().uuidString, unusedID = UUID().uuidString
        let movie = roots.documents.appendingPathComponent("match.mp4")
        let active = roots.queue.appendingPathComponent(activeID + ".project.json")
        let unused = roots.queue.appendingPathComponent(unusedID + ".project.json")
        try write(movie); try write(active); try write(unused); try write(roots.queue.appendingPathComponent("queue.json"))
        let protection = AppStorageInventory.Protection(urls: [movie], queueJobIDs: [activeID])
        let items = try AppStorageInventory.scan(roots: roots, protection: protection)
        XCTAssertEqual(items.filter(\.removable).map(\.name), [unused.lastPathComponent])
        XCTAssertThrowsError(try AppStorageInventory.remove(items.filter { $0.name == "match.mp4" }, roots: roots, protection: protection))
        try AppStorageInventory.remove(items.filter(\.removable), roots: roots, protection: protection)
        XCTAssertTrue(FileManager.default.fileExists(atPath: active.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: movie.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: unused.path))
    }
    func testExternalSymlinksAndReplacedDirectoriesNeverDeleteExternalData() throws {
        let (root, roots) = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let external = root.appendingPathComponent("External"), externalMovie = external.appendingPathComponent("match.mp4")
        try write(externalMovie, "external")
        try FileManager.default.createSymbolicLink(at: roots.documents.appendingPathComponent("linked.mp4"), withDestinationURL: externalMovie)
        try FileManager.default.createSymbolicLink(at: roots.documents.appendingPathComponent("linked-folder"), withDestinationURL: external)
        XCTAssertTrue(try AppStorageInventory.scan(roots: roots).isEmpty)
        let owned = roots.documents.appendingPathComponent("Owned/match.mp4")
        try write(owned, "owned")
        let items = try AppStorageInventory.scan(roots: roots)
        try FileManager.default.moveItem(at: owned.deletingLastPathComponent(), to: root.appendingPathComponent("Retained"))
        try FileManager.default.createSymbolicLink(at: owned.deletingLastPathComponent(), withDestinationURL: external)
        XCTAssertThrowsError(try AppStorageInventory.remove(items, roots: roots, protection: .init()))
        XCTAssertEqual(try String(contentsOf: externalMovie, encoding: .utf8), "external")
    }
    func testDeletionRechecksBusyProtectionAndFileIdentityAfterConfirmation() throws {
        let (root, roots) = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let movie = roots.documents.appendingPathComponent("match.mp4")
        try write(movie, "original")
        let items = try AppStorageInventory.scan(roots: roots)
        XCTAssertThrowsError(try AppStorageInventory.remove(items, roots: roots, protection: .init(blocksRemoval: true)))
        XCTAssertThrowsError(try AppStorageInventory.remove(items, roots: roots, protection: .init(urls: [movie])))
        try FileManager.default.moveItem(at: movie, to: root.appendingPathComponent("retained.mp4"))
        try write(movie, "replacement")
        XCTAssertThrowsError(try AppStorageInventory.remove(items, roots: roots, protection: .init()))
        XCTAssertEqual(try String(contentsOf: movie, encoding: .utf8), "replacement")
    }
    func testBroaderSupportAndCacheAccountingDoesNotDuplicateOrEnableRemovalOfUnknownData() throws {
        let (root, managed) = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let roots = AppStorageInventory.Roots(documents: managed.documents, features: managed.features, queue: managed.queue,
                                     temporary: managed.temporary, support: root.appendingPathComponent("Support"), caches: root.appendingPathComponent("Caches"))
        try write(roots.features.appendingPathComponent("features.plist"), "features")
        try write(roots.queue.appendingPathComponent("queue.json"), "queue")
        try write(root.appendingPathComponent("Support/SourceLinks/link.json"), "bookmark")
        // A movie extension in an unknown system cache does not make it an owned recording.
        try write(root.appendingPathComponent("Caches/System/match.mp4"), "cache")
        let items = try AppStorageInventory.scan(roots: roots)
        XCTAssertEqual(items.count, 4)
        XCTAssertEqual(items.reduce(0) { $0 + $1.bytes }, 26)
        XCTAssertEqual(items.first { $0.name == "features.plist" }?.category, .features)
        XCTAssertEqual(items.first { $0.name == "queue.json" }?.category, .queue)
        XCTAssertEqual(items.filter { $0.category == .other }.count, 2)
        XCTAssertTrue(items.filter { $0.category == .other }.allSatisfy { !$0.removable })
        try AppStorageInventory.remove(items.filter(\.removable), roots: roots, protection: .init())
        XCTAssertEqual(try String(contentsOf: root.appendingPathComponent("Support/SourceLinks/link.json"), encoding: .utf8), "bookmark")
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("Caches/System/match.mp4").path))
    }
    func testRootsCapturedBeforeCreationMatchExistingDirectoryURLs() throws {
        let (root, beforeCreation) = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let movie = beforeCreation.documents.appendingPathComponent("match.mp4")
        try write(movie, "movie")
        let afterCreation = AppStorageInventory.Roots(
            documents: URL(fileURLWithPath: beforeCreation.documents.path, isDirectory: true),
            features: URL(fileURLWithPath: beforeCreation.features.path, isDirectory: true),
            queue: URL(fileURLWithPath: beforeCreation.queue.path, isDirectory: true),
            temporary: URL(fileURLWithPath: beforeCreation.temporary.path, isDirectory: true))
        let before = try AppStorageInventory.scan(roots: beforeCreation)
        let after = try AppStorageInventory.scan(roots: afterCreation)
        XCTAssertEqual(before.count, 1)
        XCTAssertEqual(before.map(\.id), after.map(\.id))
        XCTAssertEqual(before.map(\.bytes), [5])
        try AppStorageInventory.remove(before, roots: afterCreation, protection: .init())
        XCTAssertFalse(FileManager.default.fileExists(atPath: movie.path))
    }
    func testRecentlyDeletedSeparatesHiddenCopiesAndRemovalPreservesCurrentRecording() throws {
        let (root, roots) = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let current = roots.documents.appendingPathComponent("match.mp4")
        let deleted = roots.documents.appendingPathComponent(".Trash/match.mp4")
        let nested = roots.documents.appendingPathComponent(".Trash/old-export/chapters.txt")
        let similarName = roots.documents.appendingPathComponent(".Trash-other/match.mp4")
        try write(current, "current"); try write(deleted, "deleted"); try write(nested, "chapters"); try write(similarName, "separate")
        let items = try AppStorageInventory.scan(roots: roots)
        let trash = items.filter { $0.category == .recentlyDeleted }
        XCTAssertEqual(trash.count, 2)
        XCTAssertEqual(trash.reduce(0) { $0 + $1.bytes }, 15)
        XCTAssertEqual(items.first { $0.id == current.path }?.location, "Documents")
        XCTAssertEqual(items.first { $0.id == deleted.path }?.location, "Documents/.Trash")
        XCTAssertEqual(items.first { $0.id == nested.path }?.location, "Documents/.Trash/old-export")
        XCTAssertEqual(items.first { $0.id == similarName.path }?.category, .recordings)
        try AppStorageInventory.remove(trash, roots: roots, protection: .init())
        XCTAssertFalse(FileManager.default.fileExists(atPath: deleted.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: nested.path))
        XCTAssertEqual(try String(contentsOf: current, encoding: .utf8), "current")
        XCTAssertEqual(try String(contentsOf: similarName, encoding: .utf8), "separate")
    }
    func testRecentlyDeletedRetainsProtectionAndGroupsProjectEdits() throws {
        let (root, roots) = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let project = roots.documents.appendingPathComponent(".Trash/project-game.volleyproject.json")
        let checkpoint = ProjectPersistence.checkpointURL(for: project)
        try write(project, "project"); try write(checkpoint, "edits")
        let items = try AppStorageInventory.scan(roots: roots, protection: .init(urls: [checkpoint]))
        let item = try XCTUnwrap(items.first)
        XCTAssertEqual(items.count, 1)
        XCTAssertEqual(item.category, .recentlyDeleted)
        XCTAssertEqual(item.bytes, 12)
        XCTAssertFalse(item.removable)
        XCTAssertThrowsError(try AppStorageInventory.remove(items, roots: roots, protection: .init(urls: [checkpoint])))
        XCTAssertTrue(FileManager.default.fileExists(atPath: project.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: checkpoint.path))
    }
    func testEnumeratedPathAliasesUseSameRepresentationForCategoryAndLocation() throws {
        let (root, roots) = try fixture(); defer { try? FileManager.default.removeItem(at: root) }
        let trash = roots.documents.appendingPathComponent(".Trash/match.mp4")
        let staged = roots.temporary.appendingPathComponent("photo-video-123/match.mp4")
        try write(trash); try write(staged)
        // On Darwin, contentsOfDirectory may return /private/var/... URLs for
        // roots standardized to /var/.... Exercise the real enumerator rather
        // than assuming its raw URL spelling matches the supplied root.
        let entries = try FileManager.default.contentsOfDirectory(at: trash.deletingLastPathComponent(), includingPropertiesForKeys: nil)
        let enumerated = try XCTUnwrap(entries.first)
        let items = try AppStorageInventory.scan(roots: roots)
        let deleted = try XCTUnwrap(items.first { $0.id == enumerated.standardizedFileURL.path })
        XCTAssertEqual(deleted.category, .recentlyDeleted)
        XCTAssertEqual(deleted.location, "Documents/.Trash")
        let temporary = try XCTUnwrap(items.first { $0.id == staged.standardizedFileURL.path })
        XCTAssertEqual(temporary.category, .temporary)
        XCTAssertEqual(temporary.location, "Temporary files/photo-video-123")
    }
}
