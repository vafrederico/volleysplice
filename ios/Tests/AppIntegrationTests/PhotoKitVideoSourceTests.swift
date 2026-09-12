import AVFoundation
import Photos
import XCTest
@testable import VolleySplice

final class PhotoKitVideoSourceTests: XCTestCase {
    func testDirectAssetIdentityAndURLAreRetainedWithoutCreatingFiles() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("photo-direct-test-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = directory.appendingPathComponent("camera.mov")
        try Data("source bytes".utf8).write(to: source)
        let asset = AVURLAsset(url: source)
        let selection = try await PhotoKitVideoSource.request(identifier: "local-only", filename: "original.mov", start: { callback in
            callback(asset, nil) // PhotoKit may finish before returning its ID.
            callback(nil, [PHImageErrorKey: CocoaError(.fileReadUnknown)])
            return 17
        }, cancel: { _ in XCTFail("A completed request must not be cancelled") })
        XCTAssertTrue(selection.asset === asset)
        XCTAssertEqual(selection.asset.url, source)
        XCTAssertEqual(selection.filename, "original.mov")
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path), ["camera.mov"])
    }

    func testNonFileAndComposedRepresentationsAreRejectedWithoutSilentImport() async throws {
        let representations: [AVAsset] = [AVMutableComposition(), AVURLAsset(url: URL(string: "https://example.invalid/video.mov")!)]
        for asset in representations {
            do {
                _ = try await PhotoKitVideoSource.request(identifier: "fixture", filename: "video.mov", start: { callback in
                    callback(asset, nil); return 18
                }, cancel: { _ in })
                XCTFail("Unsupported direct representation must fail")
            } catch PhotoKitVideoSource.Failure.unsupportedRepresentation {} 
        }
    }

    func testCancellationBeforeRequestIDArrivesCancelsExactlyThatRequest() async throws {
        let started = expectation(description: "Native request started")
        let cancelled = expectation(description: "Late native request ID cancelled")
        let gate = DispatchSemaphore(value: 0)
        let task = Task.detached {
            try await PhotoKitVideoSource.request(identifier: "fixture", filename: "video.mov", start: { callback in
                started.fulfill()
                _ = gate.wait(timeout: .now() + 5)
                callback(nil, [PHImageCancelledKey: true])
                return 63
            }, cancel: { id in
                XCTAssertEqual(id, 63)
                cancelled.fulfill()
            })
        }
        await fulfillment(of: [started], timeout: 5)
        task.cancel(); gate.signal()
        do { _ = try await task.value; XCTFail("Cancelled request must stop") }
        catch { XCTAssertTrue(error is CancellationError) }
        await fulfillment(of: [cancelled], timeout: 5)
    }

    func testAlreadyCancelledTaskDoesNotStartPhotoKitRequest() async throws {
        let task = Task.detached {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await PhotoKitVideoSource.request(identifier: "fixture", filename: "video.mov", start: { _ in
                XCTFail("Already cancelled task must not request Photos data"); return 99
            }, cancel: { _ in XCTFail("No native request exists") })
        }
        do { _ = try await task.value; XCTFail("Cancellation expected") }
        catch { XCTAssertTrue(error is CancellationError) }
    }
}
