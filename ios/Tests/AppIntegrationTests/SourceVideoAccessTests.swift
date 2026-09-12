import Foundation
@preconcurrency import AVFoundation
import XCTest
@testable import VolleySplice

final class SourceVideoAccessTests: XCTestCase {
    private func folders() throws -> (URL, URL, URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("source-link-test-" + UUID().uuidString)
        let documents = root.appendingPathComponent("Documents"), external = root.appendingPathComponent("Provider")
        try FileManager.default.createDirectory(at: documents, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: external, withIntermediateDirectories: true)
        return (root, documents, external)
    }
    func testSelectionAndRelaunchReadOriginalWithoutCopyingVideo() async throws {
        let (root, documents, external) = try folders(); defer { try? FileManager.default.removeItem(at: root) }
        let source = external.appendingPathComponent("match.mp4"), bytes = Data(repeating: 73, count: 100_000)
        try bytes.write(to: source)
        let links = root.appendingPathComponent("Links"), access = SourceVideoAccess(source)
        let fingerprint = try SourceVideoLibrary.fingerprint(access)
        let library = SourceVideoLibrary(folder: links, documents: documents)
        await library.remember(access, fingerprint: fingerprint)
        let relaunched = SourceVideoLibrary(folder: links, documents: documents)
        let restored = try await relaunched.resolve(name: "match.mp4", fingerprint: fingerprint)
        XCTAssertEqual(restored.url.resolvingSymlinksInPath(), source.resolvingSymlinksInPath())
        let read = try await CoordinatedSourceRead.perform(restored) { try Data(contentsOf: $0) }
        XCTAssertEqual(read, bytes)
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: documents.path).isEmpty)
        XCTAssertEqual(try Data(contentsOf: source), bytes)
    }
    func testMissingSourceRequiresRelinkAndSameNamedDifferentFileIsRejected() async throws {
        let (root, documents, external) = try folders(); defer { try? FileManager.default.removeItem(at: root) }
        let source = external.appendingPathComponent("match.mp4"), bytes = Data("original match".utf8)
        try bytes.write(to: source)
        let links = root.appendingPathComponent("Links"), access = SourceVideoAccess(source)
        let fingerprint = try SourceVideoLibrary.fingerprint(access)
        let library = SourceVideoLibrary(folder: links, documents: documents)
        await library.remember(access, fingerprint: fingerprint)
        try FileManager.default.removeItem(at: source)
        let collision = documents.appendingPathComponent("match.mp4")
        try Data("different recording".utf8).write(to: collision)
        let relaunched = SourceVideoLibrary(folder: links, documents: documents)
        do { _ = try await relaunched.resolve(name: "match.mp4", fingerprint: fingerprint); XCTFail("Must ask for re-link") }
        catch { XCTAssertTrue(error is SourceVideoUnavailable) }
        let renamed = external.appendingPathComponent("relocated.mp4")
        try bytes.write(to: renamed)
        await relaunched.remember(SourceVideoAccess(renamed), fingerprint: fingerprint)
        let restored = try await relaunched.resolve(name: "match.mp4", fingerprint: fingerprint)
        XCTAssertEqual(restored.url, renamed)
        XCTAssertEqual(try Data(contentsOf: collision), Data("different recording".utf8))
    }
    func testPhotosRelaunchRefetchesIdentifierWithoutPersistingPrivateURLOrCopying() async throws {
        let (root, documents, external) = try folders(); defer { try? FileManager.default.removeItem(at: root) }
        let original = external.appendingPathComponent("private-original.mov")
        let relocated = external.appendingPathComponent("private-relocated.mov")
        let bytes = Data("same original recording".utf8)
        try bytes.write(to: original)
        let selection = PhotoVideoSelection(assetIdentifier: "photos-local-id", asset: AVURLAsset(url: original), filename: "game.mov")
        let access = SourceVideoAccess(photo: selection)
        let fingerprint = try await SourceVideoLibrary.inspect(access)
        let links = root.appendingPathComponent("Links")
        let library = SourceVideoLibrary(folder: links, documents: documents)
        await library.remember(access, fingerprint: fingerprint)
        let linkURL = try XCTUnwrap(FileManager.default.contentsOfDirectory(at: links, includingPropertiesForKeys: nil).first)
        let data = try Data(contentsOf: linkURL)
        let link = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(Set(link.keys), ["fingerprint", "photoIdentifier"])
        XCTAssertEqual(link["photoIdentifier"] as? String, selection.assetIdentifier)
        XCTAssertFalse(String(decoding: data, as: UTF8.self).contains(original.path))
        try FileManager.default.moveItem(at: original, to: relocated)
        let relaunched = SourceVideoLibrary(folder: links, documents: documents, resolvePhoto: { identifier in
            XCTAssertEqual(identifier, "photos-local-id")
            return PhotoVideoSelection(assetIdentifier: identifier, asset: AVURLAsset(url: relocated), filename: "game.mov")
        })
        let restored = try await relaunched.resolve(name: "game.mov", fingerprint: fingerprint)
        XCTAssertEqual(restored.url, relocated)
        XCTAssertEqual(restored.filename, "game.mov")
        XCTAssertNotNil(restored.photo)
        XCTAssertEqual(try Data(contentsOf: restored.url), bytes)
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: documents.path).isEmpty)
    }
    func testUnavailablePhotosCanRelinkThroughFilesAndChangedContentIsRejected() async throws {
        let (root, documents, external) = try folders(); defer { try? FileManager.default.removeItem(at: root) }
        let source = external.appendingPathComponent("game.mov")
        let bytes = Data("original recording".utf8); try bytes.write(to: source)
        let access = SourceVideoAccess(photo: .init(assetIdentifier: "photos-id", asset: AVURLAsset(url: source), filename: "game.mov"))
        let fingerprint = try await SourceVideoLibrary.inspect(access)
        let links = root.appendingPathComponent("Links")
        let library = SourceVideoLibrary(folder: links, documents: documents)
        await library.remember(access, fingerprint: fingerprint)
        let denied = SourceVideoLibrary(folder: links, documents: documents, resolvePhoto: { _ in throw PhotoKitVideoSource.Failure.unavailable })
        do { _ = try await denied.resolve(name: "game.mov", fingerprint: fingerprint); XCTFail("Must request re-link") }
        catch PhotoKitVideoSource.Failure.unavailable {}
        let changed = SourceVideoLibrary(folder: links, documents: documents, resolvePhoto: { _ in access.photo! })
        try Data("different recording".utf8).write(to: source)
        do { _ = try await changed.resolve(name: "game.mov", fingerprint: fingerprint); XCTFail("Different content must not open") }
        catch { XCTAssertTrue(error is SourceVideoUnavailable) }
        try bytes.write(to: source)
        await denied.remember(SourceVideoAccess(source), fingerprint: fingerprint)
        let restored = try await denied.resolve(name: "game.mov", fingerprint: fingerprint)
        XCTAssertNil(restored.photo)
        XCTAssertEqual(restored.url, source)
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: documents.path).isEmpty)
    }
    func testSameBasenameSourcesHaveIndependentBookmarksAndLegacyOwnedFilesStillOpen() async throws {
        let (root, documents, external) = try folders(); defer { try? FileManager.default.removeItem(at: root) }
        let owned = documents.appendingPathComponent("match.mp4"), linked = external.appendingPathComponent("match.mp4")
        try Data("owned".utf8).write(to: owned); try Data("external".utf8).write(to: linked)
        let library = SourceVideoLibrary(folder: root.appendingPathComponent("Links"), documents: documents)
        let ownedID = try SourceVideoLibrary.fingerprint(SourceVideoAccess(owned))
        let linkedID = try SourceVideoLibrary.fingerprint(SourceVideoAccess(linked))
        await library.remember(SourceVideoAccess(linked), fingerprint: linkedID)
        let first = try await library.resolve(name: "match.mp4", fingerprint: ownedID)
        let second = try await library.resolve(name: "match.mp4", fingerprint: linkedID)
        XCTAssertEqual(first.url, owned); XCTAssertEqual(second.url, linked)
    }
    func testCancellationDrainsCoordinatedReaderBeforeReturning() async throws {
        let (root, _, external) = try folders(); defer { try? FileManager.default.removeItem(at: root) }
        let source = external.appendingPathComponent("match.mp4"); try Data("source".utf8).write(to: source)
        let entered = expectation(description: "Reading original"), drained = expectation(description: "Reader drained")
        let task = Task {
            try await CoordinatedSourceRead.perform(SourceVideoAccess(source)) { _ in
                entered.fulfill()
                defer { drained.fulfill() }
                try await Task.sleep(for: .seconds(60))
            }
        }
        await fulfillment(of: [entered], timeout: 5); task.cancel()
        do { try await task.value; XCTFail("Expected cancellation") } catch { XCTAssertTrue(error is CancellationError) }
        await fulfillment(of: [drained], timeout: 1)
        XCTAssertEqual(try Data(contentsOf: source), Data("source".utf8))
    }

    func testPreCancelledResolutionPropagatesCancellationInsteadOfAskingToRelink() async throws {
        let (root, documents, external) = try folders(); defer { try? FileManager.default.removeItem(at: root) }
        let source = external.appendingPathComponent("match.mp4")
        try Data("source".utf8).write(to: source)
        let access = SourceVideoAccess(source)
        let fingerprint = try await SourceVideoLibrary.inspect(access)
        let library = SourceVideoLibrary(folder: root.appendingPathComponent("Links"), documents: documents)
        await library.remember(access, fingerprint: fingerprint)
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await library.resolve(name: source.lastPathComponent, fingerprint: fingerprint)
        }
        do { _ = try await task.value; XCTFail("Expected cancellation") }
        catch { XCTAssertTrue(error is CancellationError, String(describing: error)) }
    }

    func testInspectionCancellationDoesNotWaitForConflictingWriter() async throws {
        let (root, _, external) = try folders(); defer { try? FileManager.default.removeItem(at: root) }
        let source = external.appendingPathComponent("match.mp4")
        try Data("source".utf8).write(to: source)
        let entered = expectation(description: "Conflicting writer holds source")
        let released = expectation(description: "Conflicting writer released source")
        let releaseWriter = DispatchSemaphore(value: 0)
        defer { releaseWriter.signal() }
        DispatchQueue.global(qos: .userInitiated).async {
            var error: NSError?
            NSFileCoordinator().coordinate(writingItemAt: source, options: .forReplacing, error: &error) { _ in
                entered.fulfill()
                releaseWriter.wait()
            }
            released.fulfill()
        }
        await fulfillment(of: [entered], timeout: 5)
        let finished = expectation(description: "Inspection cancels before writer releases")
        let task = Task {
            defer { finished.fulfill() }
            return try await SourceVideoLibrary.inspect(SourceVideoAccess(source))
        }
        // Let the read request contend with the known-held writer; cancellation
        // must reach NSFileCoordinator, before the reader accessor can begin.
        try await Task.sleep(for: .milliseconds(100))
        task.cancel()
        await fulfillment(of: [finished], timeout: 3)
        releaseWriter.signal()
        do { _ = try await task.value; XCTFail("Expected cancellation") }
        catch { XCTAssertTrue(error is CancellationError, String(describing: error)) }
        await fulfillment(of: [released], timeout: 3)
    }

    func testCombinedExportCanReadSourceAndWriteItsParentFolder() async throws {
        let (root, _, external) = try folders(); defer { try? FileManager.default.removeItem(at: root) }
        let source = external.appendingPathComponent("match.mp4"), bytes = Data("original recording".utf8)
        try bytes.write(to: source)
        let bookmark = try external.bookmarkData(options: .minimalBookmark, includingResourceValuesForKeys: nil, relativeTo: nil)
        let finished = expectation(description: "Combined same-folder coordination completes")
        let task = Task {
            defer { finished.fulfill() }
            return try await CoordinatedVideoExport.inFolder(bookmark: bookmark, reading: SourceVideoAccess(source)) { recording, folder in
                let read = try Data(contentsOf: recording)
                return try await CoordinatedVideoExport.updateFolder(folder, reading: recording) { destination in
                    let output = destination.appendingPathComponent("export.mp4")
                    try read.write(to: output)
                    return output
                }
            }
        }
        await fulfillment(of: [finished], timeout: 5)
        task.cancel()
        let output = try await task.value
        XCTAssertEqual(try Data(contentsOf: output), bytes)
        XCTAssertEqual(try Data(contentsOf: source), bytes)
        XCTAssertNotEqual(output, source)
    }

    func testCombinedExportCancellationDrainsAccessor() async throws {
        let (root, _, external) = try folders(); defer { try? FileManager.default.removeItem(at: root) }
        let source = external.appendingPathComponent("match.mp4"); try Data("source".utf8).write(to: source)
        let bookmark = try external.bookmarkData(options: .minimalBookmark, includingResourceValuesForKeys: nil, relativeTo: nil)
        let entered = expectation(description: "Combined accessor started"), drained = expectation(description: "Combined accessor drained")
        let task = Task {
            try await CoordinatedVideoExport.inFolder(bookmark: bookmark, reading: SourceVideoAccess(source)) { _, _ in
                entered.fulfill()
                defer { drained.fulfill() }
                try await Task.sleep(for: .seconds(60))
            }
        }
        await fulfillment(of: [entered], timeout: 5); task.cancel()
        do { try await task.value; XCTFail("Expected cancellation") }
        catch { XCTAssertTrue(error is CancellationError, String(describing: error)) }
        await fulfillment(of: [drained], timeout: 1)
        XCTAssertEqual(try Data(contentsOf: source), Data("source".utf8))
    }

    func testSameFolderExportAllowsProviderAndAVAssetReadsBetweenMutations() async throws {
        let (root, _, external) = try folders(); defer { try? FileManager.default.removeItem(at: root) }
        let fixture = try XCTUnwrap(Bundle.main.url(forResource: "overlay-fixture", withExtension: "mp4"))
        let source = external.appendingPathComponent("match.mp4")
        try FileManager.default.copyItem(at: fixture, to: source)
        let bookmark = try external.bookmarkData(options: .minimalBookmark, includingResourceValuesForKeys: nil, relativeTo: nil)
        let completed = expectation(description: "Metadata and provider read finish without a held folder writer")
        let task = Task {
            defer { completed.fulfill() }
            return try await CoordinatedVideoExport.inFolder(bookmark: bookmark, reading: SourceVideoAccess(source)) { recording, folder in
                // A provider may coordinate its directory while AVFoundation opens
                // a movie. The former whole-encode directory writer blocks this.
                try await CoordinatedSourceRead.perform(SourceVideoAccess(folder)) { _ in
                    let asset = AVURLAsset(url: recording)
                    return try await withTaskCancellationHandler {
                        try Task.checkCancellation()
                        let video = try await asset.loadTracks(withMediaType: .video)
                        let duration = try await asset.load(.duration)
                        guard !video.isEmpty, duration.seconds > 0 else { throw ProjectError.invalid("Invalid movie fixture") }
                        return duration.seconds
                    } onCancel: { asset.cancelLoading() }
                }
            }
        }
        await fulfillment(of: [completed], timeout: 10)
        task.cancel()
        let duration = try await task.value
        XCTAssertGreaterThan(duration, 0)
        XCTAssertEqual(try Data(contentsOf: source), try Data(contentsOf: fixture))
    }
}
