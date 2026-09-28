import CoreVideo
import XCTest
@testable import VolleySplice

final class MediaPreprocessingIntegrationTests: XCTestCase {
    private func fixture() throws -> CVPixelBuffer {
        var result: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, 6, 4,
                                        kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
                                        nil, &result)
        XCTAssertEqual(status, kCVReturnSuccess)
        let buffer = try XCTUnwrap(result)
        CVBufferSetAttachment(buffer, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_709_2, .shouldPropagate)
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let y = try XCTUnwrap(CVPixelBufferGetBaseAddressOfPlane(buffer, 0)?.assumingMemoryBound(to: UInt8.self))
        let uv = try XCTUnwrap(CVPixelBufferGetBaseAddressOfPlane(buffer, 1)?.assumingMemoryBound(to: UInt8.self))
        let ys = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0), uvs = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1)
        for row in 0..<4 { for x in 0..<6 { y[row * ys + x] = UInt8((row * 67 + x * 29) % 256) } }
        for row in 0..<2 { for x in 0..<3 {
            let index = row * 3 + x
            uv[row * uvs + x * 2] = UInt8(40 + index * 29)
            uv[row * uvs + x * 2 + 1] = UInt8(205 - index * 23)
        } }
        return buffer
    }

    func testActualAppNV12AreaMatchesAndroidIncludingRotation() throws {
        let buffer = try fixture()
        // Generated with the current Android YuvAreaResampler. Nonintegral
        // transpose cases catch resizing in the wrong coordinate system.
        let expected: [[UInt8]] = [
            [175,27,0,255,201,88,10,255,221,162,109,255,206,189,191,255,98,117,158,255,56,106,191,255],
            [164,156,173,255,195,121,93,255,159,26,0,255,39,86,168,255,196,176,162,255,203,124,63,255],
            [56,106,191,255,98,117,158,255,206,189,191,255,221,162,109,255,201,88,10,255,175,27,0,255],
            [203,124,63,255,196,176,162,255,39,86,168,255,159,26,0,255,195,121,93,255,164,156,173,255]
        ]
        for (index, rotation) in [0,90,180,270].enumerated() {
            XCTAssertEqual(try MediaDecoder.sampleRGBA(buffer, roi: .fullFrame, rotation: rotation,
                                                       outputWidth: 3, outputHeight: 2), expected[index])
        }
    }

    func testActualAppNeuralPixelsKeepFractionalValuesAndBlackLetterbox() throws {
        let frame = try MediaDecoder.sampleEmbeddingFrame(fixture(), roi: .fullFrame, rotation: 0)
        let pixels = frame.pixels, geometry = frame.geometry
        XCTAssertEqual(geometry, try NeuralRallyContract.geometry(width: 6, height: 4))
        XCTAssertEqual(pixels.count, 224 * 224 * 3)
        XCTAssertTrue(pixels.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 255 })
        XCTAssertTrue(pixels.contains { abs($0 - $0.rounded()) > 0.1 })
        XCTAssertEqual(Array(pixels.prefix(geometry.top * 224 * 3)), [Float](repeating: 0, count: geometry.top * 224 * 3))
        let contentEnd = (geometry.top + geometry.resizedHeight) * 224 * 3
        XCTAssertEqual(Array(pixels[contentEnd...]), [Float](repeating: 0, count: pixels.count - contentEnd))
        let normalized = try NeuralRallyContract.normalizedCHW(rgb: pixels)
        let quality = try NeuralRallyContract.quality(chw: normalized, geometry: geometry)
        XCTAssertEqual(quality[0], Float(geometry.resizedWidth * geometry.resizedHeight) / Float(224 * 224))
        XCTAssertTrue(quality.allSatisfy(\.isFinite))
    }

    func testEmbeddingGeometryFollowsCleanApertureInsteadOfEncodedPadding() throws {
        let buffer = try fixture()
        let clean: [String: Any] = [kCVImageBufferCleanApertureWidthKey as String: 4,
                                   kCVImageBufferCleanApertureHeightKey as String: 2,
                                   kCVImageBufferCleanApertureHorizontalOffsetKey as String: 0,
                                   kCVImageBufferCleanApertureVerticalOffsetKey as String: 0]
        CVBufferSetAttachment(buffer, kCVImageBufferCleanApertureKey, clean as CFDictionary, .shouldPropagate)
        XCTAssertEqual(CVImageBufferGetCleanRect(buffer).width, 4)
        XCTAssertEqual(CVImageBufferGetCleanRect(buffer).height, 2)
        let frame = try MediaDecoder.sampleEmbeddingFrame(buffer, roi: .fullFrame, rotation: 90)
        XCTAssertEqual(frame.geometry, try NeuralRallyContract.geometry(width: 4, height: 2, rotation: 90))
        XCTAssertEqual(frame.geometry.resizedWidth, 112)
        XCTAssertEqual(frame.geometry.resizedHeight, 224)
    }

    func testSaturatedFullHDInterpolationRemainsValidForNormalization() throws {
        var result: CVPixelBuffer?
        XCTAssertEqual(CVPixelBufferCreate(kCFAllocatorDefault, 1920, 1080,
                                          kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                                          nil, &result), kCVReturnSuccess)
        let buffer = try XCTUnwrap(result)
        CVPixelBufferLockBaseAddress(buffer, [])
        for plane in 0..<2 {
            let pointer = try XCTUnwrap(CVPixelBufferGetBaseAddressOfPlane(buffer, plane)?.assumingMemoryBound(to: UInt8.self))
            let count = CVPixelBufferGetBytesPerRowOfPlane(buffer, plane) * CVPixelBufferGetHeightOfPlane(buffer, plane)
            pointer.update(repeating: plane == 0 ? 255 : 128, count: count)
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])
        let frame = try MediaDecoder.sampleEmbeddingFrame(buffer, roi: .fullFrame, rotation: 0)
        // Float bilinear weights previously produced 255.000015 in saturated
        // regions at this geometry, making an otherwise valid video fail.
        XCTAssertEqual(frame.pixels.max(), 255)
        XCTAssertTrue(frame.pixels.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 255 })
        let normalized = try NeuralRallyContract.normalizedCHW(rgb: frame.pixels)
        XCTAssertTrue(normalized.allSatisfy(\.isFinite))
    }

    func testSpecialistCachesFromNearestPixelSamplerAreRebuilt() async throws {
        struct SavedServing: Codable {
            let schema: String
            let candidates: [ServingSideInference.CandidateInterval]
            let rawFeatures: [Double]
        }
        struct SavedSwitch: Codable {
            let schema: String
            let plan: SideSwitchInference.FramePlan
            let visual: [Double]
        }
        let source = try XCTUnwrap(Bundle.main.url(forResource: "ios-editor-fixture", withExtension: "mp4"))
        let media = try await MediaDecoder.describe(source), end = min(4, media.duration)
        let ranges = [Interval(start: 0.1, end: end * 0.35, confidence: 0.9, agreement: "neural"),
                      Interval(start: end * 0.6, end: end - 0.05, confidence: 0.9, agreement: "neural")]
        let times = try VideoFrameSelection.targets(start: 0, end: end, fps: 4)
        let rally = [Float](repeating: 0.9, count: times.count), dead = [Float](repeating: 0.1, count: times.count)
        let input = SideSwitchAnalysisInput(intervals: ranges, allLabelsV2Components: ranges, previousProductionComponents: ranges,
            allLabelsV2State: ProductionStateOutput(modelId: "model-1ca43e38eefc", times: times, rallyProbabilities: rally, deadStateProbabilities: dead),
            previousProductionState: ProductionStateOutput(modelId: "model-9c92b8e9333f", times: times, rallyProbabilities: rally, deadStateProbabilities: dead))
        let servingPlan = try ServingSideInference.framePlan(ranges: ranges, duration: media.duration)
        let switchAsset = try XCTUnwrap(Bundle.main.url(forResource: "side-switch-c2570481c30d", withExtension: "json"))
        let switchPlan = try SideSwitchInference(data: Data(contentsOf: switchAsset)).framePlan(input: input, duration: media.duration)
        XCTAssertFalse(switchPlan.candidates.isEmpty)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("specialist-cache-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let cache = root.appendingPathComponent("serving.plist"), switchCache = root.appendingPathComponent("serving.switch.plist")
        let staleServing = [Double](repeating: 314.25, count: servingPlan.candidates.count * ServingSideInference.columns)
        let staleSwitch = [Double](repeating: 314.25, count: switchPlan.candidates.count * 22)
        let encoder = PropertyListEncoder(); encoder.outputFormat = .binary
        try encoder.encode(SavedServing(schema: "ios-serving-gray-opencv412-v1", candidates: servingPlan.candidates,
                                        rawFeatures: staleServing)).write(to: cache)
        try encoder.encode(SavedSwitch(schema: "ios-switch-bgr-opencv412-v1", plan: switchPlan,
                                       visual: staleSwitch)).write(to: switchCache)
        let result = try await ScoreAnalysis.run(url: source, media: media, roi: .fullFrame, ranges: ranges,
            all: ProductionServeOutput(modelId: "model-1ca43e38eefc", times: times, probabilities: rally),
            previous: ProductionServeOutput(modelId: "model-9c92b8e9333f", times: times, probabilities: rally),
            cacheURL: cache, switchInput: input, progress: { _, _ in })
        XCTAssertNil(result.error)
        XCTAssertEqual(result.servingSide.rows, servingPlan.candidates.count)
        XCTAssertNotEqual(result.servingSide.rawFeatures, staleServing)
        let servingSaved = try PropertyListDecoder().decode(SavedServing.self, from: Data(contentsOf: cache))
        let switchSaved = try PropertyListDecoder().decode(SavedSwitch.self, from: Data(contentsOf: switchCache))
        XCTAssertEqual(servingSaved.schema, "ios-serving-gray-area-color-opencv412-v2")
        XCTAssertEqual(switchSaved.schema, "ios-switch-bgr-area-color-opencv412-v2")
        XCTAssertNotEqual(switchSaved.visual, staleSwitch)
        XCTAssertTrue(switchSaved.visual.allSatisfy(\.isFinite))
    }
}
