import XCTest
import AVFoundation
import Photos
import UIKit
@testable import VolleySplice

final class RenderingIntegrationTests: XCTestCase {
    @MainActor func testAndroidScoreAndPointPixelsAndYUVBlend() throws {
        try ScoreOverlayRenderer.checkPixels()
        try YUVScoreCompositor.checkKernels()
    }
    func testAndroidRotationFixturesDecodeAtQuarterSecondTargets() async {
        let result = await DeviceMediaCheck.run()
        XCTAssertFalse(result.contains("FAIL"), result)
        XCTAssertTrue(result.contains("4/4 fixtures passed"), result)
    }
    @MainActor func testRemovingQueueEntryPreservesSourceAndFinishedOutput() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let queue = ProcessingQueue(folder: folder)
        let source = folder.appendingPathComponent("source.mp4"), output = folder.appendingPathComponent("export.mp4")
        try Data([1,2,3]).write(to: source); try Data([4,5,6]).write(to: output)
        let job = ProcessingJob(kind: .videoExport, projectId: "fixture", sourceName: "source.mp4", sourceFingerprint: "sampled-sha256-v1:test")
        let project = ProjectDocument(id: "fixture", sourceName: "source.mp4", durationMs: 2000,
            draft: EditorDraft(sourceRevision: "fixture", cuts: []))
        // No execute closure: this job stays queued so removal exercises the durable store directly.
        try await queue.enqueue(job, snapshot: project)
        XCTAssertTrue(FileManager.default.fileExists(atPath: folder.appendingPathComponent(job.id + ".project.json").path))
        queue.remove(job.id)
        XCTAssertTrue(queue.jobs.isEmpty)
        XCTAssertTrue(try ProcessingJobStore(folder: folder).load().jobs.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: folder.appendingPathComponent(job.id + ".project.json").path))
        XCTAssertEqual(try Data(contentsOf: source), Data([1,2,3]))
        XCTAssertEqual(try Data(contentsOf: output), Data([4,5,6]))
    }
    func testCoordinatedFolderWritesIntoChosenDestination() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let bookmark = try folder.bookmarkData(options: .minimalBookmark, includingResourceValuesForKeys: nil, relativeTo: nil)
        let written: URL = try await CoordinatedVideoExport.inFolder(bookmark: bookmark) { destination in
            let file = destination.appendingPathComponent("chosen.mp4")
            try Data([7,8,9]).write(to: file, options: .atomic)
            return file
        }
        XCTAssertEqual(written.standardizedFileURL, folder.appendingPathComponent("chosen.mp4").standardizedFileURL)
        XCTAssertEqual(try Data(contentsOf: written), Data([7,8,9]))
    }
    func testCameraRollMoveRemovesCompletedPrivateFile() async throws {
        guard PHPhotoLibrary.authorizationStatus(for: .addOnly) == .authorized else {
            throw XCTSkip("Grant simulator photos-add permission before running this integration check")
        }
        let fixture = try XCTUnwrap(Bundle.main.url(forResource: "overlay-fixture", withExtension: "mp4"))
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".mp4")
        try FileManager.default.copyItem(at: fixture, to: file)
        defer { try? FileManager.default.removeItem(at: file) }
        let identifier = try await PhotoExportDelivery.move(file)
        XCTAssertFalse(identifier.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: file.path), "shouldMoveFile must remove the private source after successful import")
    }

    @MainActor func testEncodedPointTimelineAppearsAndPreservesSourceOutsideOverlay() async throws {
        let fixture = try XCTUnwrap(Bundle.main.url(forResource: "ios-export-fixture", withExtension: "mp4"))
        let directory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let destination = directory.appendingPathComponent("simulator-export-\(UUID().uuidString).mp4")
        let output = try await DeviceExportCheck.run(source: fixture, destination: destination) { _, _ in }
        XCTAssertEqual(output.durationMs, 13500)
        let project = try ProjectDocument.load(from: destination.deletingPathExtension().appendingPathExtension("volleyproject.json"))
        let tracking = try ScoreTracking.fromJSON(project.draft.scoreTracking, durationMs: project.durationMs)
        let prepared = PreparedScoreOverlay(tracking: tracking, rallyRanges: [
            .init(coreStartMs: 2000, coreEndMs: 4000, keepStartMs: 1000, keepEndMs: 5000),
            .init(coreStartMs: 8000, coreEndMs: 10000, keepStartMs: 7000, keepEndMs: 11000),
            .init(coreStartMs: 15000, coreEndMs: 17000, keepStartMs: 14000, keepEndMs: 18000)])
        let snapshot = prepared.snapshot(at: 9500)
        XCTAssertFalse(snapshot.points.isEmpty)
        let raster = try ScoreOverlayRenderer.raster(snapshot: snapshot, videoWidth: 320, videoHeight: 240, renderTimeline: true)
        XCTAssertGreaterThan(raster.width, raster.scoreWidth)
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: destination))
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero; generator.requestedTimeToleranceAfter = .zero
        let requestedTime = CMTime(seconds: 8, preferredTimescale: 600)
        let frame = try await generator.image(at: requestedTime)
        let image = frame.image
        XCTAssertEqual(frame.actualTime.seconds, 8, accuracy: 0.001)
        let controlURL = destination.deletingPathExtension().appendingPathExtension("overlay-off.mp4")
        let controlGenerator = AVAssetImageGenerator(asset: AVURLAsset(url: controlURL))
        controlGenerator.appliesPreferredTrackTransform = true
        controlGenerator.requestedTimeToleranceBefore = .zero; controlGenerator.requestedTimeToleranceAfter = .zero
        let controlFrame = try await controlGenerator.image(at: requestedTime)
        let control = controlFrame.image
        XCTAssertEqual(controlFrame.actualTime.seconds, 8, accuracy: 0.001)
        func pixels(_ image: CGImage) throws -> [UInt8] {
            var bytes = [UInt8](repeating: 0, count: image.width * image.height * 4)
            let color = CGColorSpaceCreateDeviceRGB()
            try bytes.withUnsafeMutableBytes { buffer in
                let context = try XCTUnwrap(CGContext(data: buffer.baseAddress, width: image.width, height: image.height, bitsPerComponent: 8,
                    bytesPerRow: image.width * 4, space: color,
                    bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue))
                // Drawing an existing CGImage preserves its scanline order.
                // UIKit's text/path y-flip would move the encoded top strip to
                // the bottom of this byte array and invalidate both ROIs.
                context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
            }
            return bytes
        }
        // Independent asymmetric image guards the test's top-row-first contract.
        let scanlines: [UInt8] = [255, 0, 0, 255, 0, 0, 255, 255]
        let provider = try XCTUnwrap(CGDataProvider(data: Data(scanlines) as CFData))
        let rowFixture = try XCTUnwrap(CGImage(width: 1, height: 2, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: 4, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        XCTAssertEqual(try pixels(rowFixture), scanlines, "Pixel row zero must represent the top of the decoded image")
        let on = try pixels(image), off = try pixels(control)
        XCTAssertEqual(image.width, 320); XCTAssertEqual(image.height, 240)
        XCTAssertEqual(control.width, image.width); XCTAssertEqual(control.height, image.height)
        var pointDelta = 0, backgroundDelta = 0, backgroundCount = 0
        for y in 0..<image.height { for x in 0..<image.width {
            let i = (y * image.width + x) * 4
            let delta = (0..<3).reduce(0) { $0 + abs(Int(on[i + $1]) - Int(off[i + $1])) }
            if x >= raster.scoreWidth && x < raster.width && y < raster.height { pointDelta += delta }
            if y >= raster.height + 8 { backgroundDelta += delta; backgroundCount += 3 }
        } }
        XCTAssertGreaterThan(pointDelta, 1000, "The encoded video must contain point graphics right of the score cells")
        XCTAssertLessThan(Double(backgroundDelta) / Double(backgroundCount), 3, "Overlay encoding should preserve source colors below its strip")
        for (name, frame) in [("encoded-overlay-at-8s", image), ("encoded-control-at-8s", control)] {
            let attachment = XCTAttachment(image: UIImage(cgImage: frame))
            attachment.name = name; attachment.lifetime = .keepAlways; add(attachment)
        }
        let artifact = XCTAttachment(string: destination.path); artifact.name = "encoded-overlay-artifact"; artifact.lifetime = .keepAlways; add(artifact)
    }

    @MainActor func testInstalledBundleHasAppIconAndTransparentBrandMark() throws {
        let icons = try XCTUnwrap(Bundle.main.object(forInfoDictionaryKey: "CFBundleIcons") as? [String: Any])
        let primary = try XCTUnwrap(icons["CFBundlePrimaryIcon"] as? [String: Any])
        XCTAssertEqual(primary["CFBundleIconName"] as? String, "AppIcon")
        let brand = try XCTUnwrap(UIImage(named: "BrandMark")?.cgImage)
        XCTAssertTrue([CGImageAlphaInfo.premultipliedFirst, .premultipliedLast, .first, .last].contains(brand.alphaInfo), "Brand mark must retain its transparent background")
    }

    func testFileSourceAnalysisCompletesDetailedStagesWithoutOptionalSideSwitchWork() async throws {
        let source = try XCTUnwrap(Bundle.main.url(forResource: "ios-editor-fixture", withExtension: "mp4"))
        let cache = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: cache) }
        let output = try await AnalysisPipeline.analyze(url: source, roi: AnalysisRegion(x: 0, y: 0, width: 1, height: 1),
            start: 0, end: 20, cacheFolder: cache, prepareScore: true, generateSideSwitchMarkers: false) { _, _ in }
        XCTAssertEqual(output.start, 0); XCTAssertEqual(output.end, 20)
        XCTAssertEqual(output.times.count, 80)
        XCTAssertNil(output.sideSwitch, "Optional court-side inference must honor the setup choice")
        XCTAssertTrue(output.base.allSatisfy(\.isFinite))
        XCTAssertTrue(output.stageSeconds.values.allSatisfy { $0.isFinite && $0 >= 0 })
        XCTAssertNotNil(output.stageSeconds["inference"])
    }

}
