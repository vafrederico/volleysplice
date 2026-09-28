import CryptoKit
import Foundation
import XCTest
@testable import VolleyCore

final class NeuralRallyContractTests: XCTestCase {
    private var fixtures: URL {
        if let path = ProcessInfo.processInfo.environment["VOLLEY_FIXTURES"] { return URL(fileURLWithPath: path) }
        return URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Fixtures")
    }
    private func fixture(_ path: String) throws -> Data {
        try Data(contentsOf: fixtures.appendingPathComponent("rally-models").appendingPathComponent(path))
    }
    private func syntheticConfig(_ changes: [String: Any] = [:], model: RallyModel = .maximumCoverage) throws -> NeuralPipelineConfig {
        var values: [String: Any] = ["modelIdentity": "dino-distilled-mobilenet-v3-large-tcn", "selectionMode": "recall",
            "recallTargetPercent": 99, "tokenDimension": 3840, "mean": [Double](repeating: 0.1, count: 112),
            "scale": [Double](repeating: 0.2, count: 112), "weightsSha256": String(repeating: "a", count: 64),
            "encoderWeightsSha256": String(repeating: "b", count: 64),
            "decoder": ["smoothing": 0.5, "minimum": 1, "enter": 0.2, "boundary": true]]
        values.merge(changes) { _, replacement in replacement }
        return try NeuralPipelineConfig(data: JSONSerialization.data(withJSONObject: values), model: model)
    }
    private func floatHash(_ values: [Float]) -> String {
        var bytes = Data(capacity: values.count * 4)
        for value in values {
            var bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { bytes.append(contentsOf: $0) }
        }
        return SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    }

    func testDefaultAndStableIdentityPreserveSavedSelections() throws {
        XCTAssertEqual(RallyModel.default, .maximumCoverage)
        XCTAssertEqual(RallyModel.default.modelId, "distilled-large-recall-v1")
        XCTAssertEqual(RallyModel.maximumCoverage.displayName, "Maximum coverage · BETA")
        XCTAssertEqual(RallyModel.balanced.displayName, "Balanced · BETA")
        XCTAssertEqual(RallyModel.legacy.displayName, "Legacy model")
        XCTAssertEqual(RallyModel.legacy.modelId, ProjectArchive.modelId)
        XCTAssertEqual(RallyModel(modelId: ProjectArchive.modelId), .legacy)
        XCTAssertNil(RallyModel(modelId: "unavailable-model"))
        for model in RallyModel.allCases {
            XCTAssertEqual(try JSONDecoder().decode(RallyModel.self, from: JSONEncoder().encode(model)), model)
        }
        XCTAssertEqual(try JSONDecoder().decode(RallyModel.self, from: Data("\"distilled-large-f1-v1\"".utf8)), .balanced)
        XCTAssertThrowsError(try JSONDecoder().decode(RallyModel.self, from: Data("\"unavailable-model\"".utf8)))
    }

    func testFrozenBundlesPairEncoderHeadScalersAndDecoder() throws {
        let manifest = try NeuralRallyManifest(data: fixture("manifest.json"))
        XCTAssertEqual(manifest.defaultVariant, RallyModel.default.variantKey)
        let recall = try manifest.bundle(for: .maximumCoverage), balanced = try manifest.bundle(for: .balanced)
        XCTAssertNotEqual(recall.files.encoder.sha256, balanced.files.encoder.sha256)
        XCTAssertNotEqual(recall.files.temporal.sha256, balanced.files.temporal.sha256)
        XCTAssertNotEqual(recall.files.pipeline.sha256, balanced.files.pipeline.sha256)
        XCTAssertEqual(recall.embeddingPrecision, "fp32")
        let config = try NeuralPipelineConfig(data: fixture("recall/pipeline.json"), model: .maximumCoverage, bundle: recall)
        XCTAssertEqual(config.decoder.enter, 0.2)
        XCTAssertEqual(config.decoder.smoothing, 0.5)
        XCTAssertEqual(config.decoder.minimum, 1)
        XCTAssertTrue(config.decoder.boundary)
        _ = try NeuralPipelineConfig(data: fixture("f1/pipeline.json"), model: .balanced, bundle: balanced)
        XCTAssertThrowsError(try manifest.bundle(for: .legacy))
        XCTAssertThrowsError(try NeuralPipelineConfig(data: fixture("recall/pipeline.json"), model: .balanced, bundle: balanced))
        XCTAssertThrowsError(try NeuralPipelineConfig(data: fixture("recall/pipeline.json"), model: .maximumCoverage, bundle: balanced))
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: fixture("manifest.json")) as? [String: Any])
        json["defaultVariant"] = "high-f1"
        XCTAssertThrowsError(try NeuralRallyManifest(data: JSONSerialization.data(withJSONObject: json)))
        XCTAssertThrowsError(try syntheticConfig(["scale": [Double](repeating: 0, count: 112)]))
        XCTAssertThrowsError(try syntheticConfig(["tokenDimension": 2304]))
    }

    func testAlignedEmbeddingScheduleIncludesPrecedingCausalImage() throws {
        XCTAssertEqual(try NeuralRallyContract.embeddingTimes(duration: 120, start: 3.2, end: 5), [3, 3.5, 4, 4.5])
        XCTAssertEqual(try NeuralRallyContract.embeddingTimes(duration: 120, start: 3.5, end: 5.01), [3.5, 4, 4.5, 5])
        XCTAssertEqual(try NeuralRallyContract.embeddingTimes(duration: 120, start: 0, end: 120).count, 240)
        XCTAssertThrowsError(try NeuralRallyContract.embeddingTimes(duration: 120, start: 10, end: 9))
        XCTAssertThrowsError(try NeuralRallyContract.embeddingTimes(duration: 120, start: 0, end: 121))
    }

    // Frozen checksums produced by compiling the existing Android RegionalPoolWeights.java
    // and executing its unmodified implementation, not by the Swift implementation under test.
    func testGeometryAndRegionalPoolingMatchNativeGoldenAcrossRotation() throws {
        let landscape = try NeuralRallyContract.geometry(width: 1920, height: 1080)
        XCTAssertEqual(landscape.resizedWidth, 224); XCTAssertEqual(landscape.resizedHeight, 126)
        XCTAssertEqual(landscape.left, 0); XCTAssertEqual(landscape.top, 49)
        let portrait = try NeuralRallyContract.geometry(width: 1920, height: 1080, rotation: 90)
        XCTAssertEqual(portrait, try NeuralRallyContract.geometry(width: 1080, height: 1920))
        let weights = try NeuralRallyContract.regionalPoolWeights(box: landscape.box)
        XCTAssertEqual(floatHash(weights), "ac0af3a32532146b396d6dbb9dcbcc3e18196e00c55dcf37d0d30f9f5d81c0be")
        XCTAssertEqual(floatHash(try NeuralRallyContract.regionalPoolWeights(box: portrait.box)), "7efbf0aae9bec48ff3fc938204bc8fb953d75b787ef23fb335b0f80ba65e6eb2")
        for region in 0..<4 {
            XCTAssertEqual(weights[(region * 49)..<((region + 1) * 49)].map(Double.init).reduce(0, +), 1, accuracy: 1e-6)
        }
        XCTAssertEqual(weights[0], 0); XCTAssertEqual(weights[48], 0)
        let halfPixel = try NeuralRallyContract.geometry(width: 5, height: 5, roi: [0.1, 0.1, 0.5, 0.5])
        XCTAssertEqual(halfPixel.x, 1); XCTAssertEqual(halfPixel.y, 1); XCTAssertEqual(halfPixel.cropWidth, 3)
        XCTAssertThrowsError(try NeuralRallyContract.geometry(width: 1920, height: 1080, rotation: 45))
        XCTAssertThrowsError(try NeuralRallyContract.geometry(width: 1920, height: 1080, roi: [0, 0, 0, 1]))
    }

    func testFractionalImageNormalizationKeepsInterpolationAndPaddingSeparate() throws {
        let geometry = try NeuralRallyContract.geometry(width: 1920, height: 1080)
        var rgb = [Float](repeating: 127.5, count: geometry.resizedWidth * geometry.resizedHeight * 3)
        rgb[0] = 3.25
        let chw = try NeuralRallyContract.normalizedLetterbox(rgb: rgb, geometry: geometry)
        let first = geometry.top * 224 + geometry.left
        XCTAssertEqual(chw[0], -Float(0.485) / Float(0.229))
        XCTAssertEqual(chw[first], (Float(3.25) / 255 - Float(0.485)) / Float(0.229))
        XCTAssertNotEqual(chw[first], (Float(3) / 255 - Float(0.485)) / Float(0.229))
        var padded = [Float](repeating: 0, count: 224 * 224 * 3)
        for y in 0..<geometry.resizedHeight { for x in 0..<geometry.resizedWidth { for c in 0..<3 {
            padded[((y + geometry.top) * 224 + x + geometry.left) * 3 + c] = rgb[(y * geometry.resizedWidth + x) * 3 + c]
        } } }
        XCTAssertEqual(chw, try NeuralRallyContract.normalizedCHW(rgb: padded))
        XCTAssertThrowsError(try NeuralRallyContract.normalizedCHW(rgb: [0, .nan, 0]))
    }

    // Exact six Float bit patterns from Android NeuralVideoEncoder.quality. Synthetic ramp
    // exercises grayscale reconstruction, clipping, reflected borders and laplacian variance.
    func testImageQualityMatchesNativeGolden() throws {
        let geometry = try NeuralRallyContract.geometry(width: 1920, height: 1080)
        let plane = 224 * 224
        let means: [Float] = [0.485, 0.456, 0.406], scales: [Float] = [0.229, 0.224, 0.225]
        var image = [Float](repeating: 0, count: plane * 3)
        for c in 0..<3 { for i in 0..<plane { image[c * plane + i] = (Float(i % 256) / 255 - means[c]) / scales[c] } }
        XCTAssertEqual(try NeuralRallyContract.quality(chw: image, geometry: geometry).map(\.bitPattern),
                       [1058013184, 1056964608, 1049926094, 1049499855, 1019301470, 0])
    }

    // Golden hash from the unchanged Android NeuralRallyPipeline.fuse method. Normalizer
    // uses Double subtraction/division followed by Float assignment, as native production does.
    func testBoundedFusionMatchesNativeGoldenAndKeepsAbsoluteEmbeddingGrid() throws {
        let times = [5.25, 5.5, 5.75, 6.0, 6.25]
        let av = (0..<(times.count * 520)).map { Float($0 % 37) / 37 }
        let tokens = (0..<(3 * 3840)).map { Float($0 % 113 - 56) / 8 }
        var quality: [Float] = []
        for i in 0..<3 { for c in 0..<6 { quality.append(c < 5 ? Float(c + 1) / 10 : Float(i) * 0.01) } }
        let config = try syntheticConfig()
        let whole = try NeuralRallyContract.fuse(times: times, contextual: av, embeddingTimes: [5, 5.5, 6], tokens: tokens, quality: quality, config: config)
        XCTAssertEqual(floatHash(whole), "2978f57a45158c174ca29d443070421666df290df1defdea0c4ec4eaacf0d586")
        let part = try NeuralRallyContract.fuse(times: times, contextual: av, embeddingTimes: [5, 5.5, 6], tokens: tokens, quality: quality, config: config, rows: 1..<4)
        XCTAssertEqual(part, Array(whole[3952..<(4 * 3952)]))
        XCTAssertEqual(whole[3950], 0.75)
        XCTAssertEqual(whole[3952 + 3950], -0.5)
        XCTAssertThrowsError(try NeuralRallyContract.fuse(times: times, contextual: av, embeddingTimes: [5.5, 6, 6.5], tokens: tokens, quality: quality, config: config))
        XCTAssertThrowsError(try NeuralRallyContract.fuse(times: times, contextual: av, embeddingTimes: [5, 6, 6.5], tokens: tokens, quality: quality, config: config))
    }

    func testTemporalChunksRetainRealContextWithoutDuplicatingOrPaddingOutputRows() throws {
        let chunks = try NeuralRallyContract.chunkPlan(rows: 301)
        XCTAssertEqual(chunks.map(\.core), [0..<128, 128..<256, 256..<301])
        XCTAssertEqual(chunks.map(\.input), [0..<190, 66..<301, 194..<301])
        XCTAssertEqual(chunks.flatMap { Array($0.core) }, Array(0..<301))
        XCTAssertTrue(chunks.allSatisfy { $0.input.count <= 252 && $0.input.lowerBound <= $0.core.lowerBound && $0.input.upperBound >= $0.core.upperBound })
        XCTAssertTrue(try NeuralRallyContract.chunkPlan(rows: 0).isEmpty)
        XCTAssertThrowsError(try NeuralRallyContract.chunkPlan(rows: -1))
    }

    func testSigmoidAndFrozenDecoderMatchNativeGolden() throws {
        let scores = try NeuralRallyContract.probabilities(logits: [-100, 0, 2, 100])
        XCTAssertEqual(scores[1], 0.5); XCTAssertEqual(scores[2], 0.8807971, accuracy: 1e-7)
        XCTAssertEqual(scores[3], 1)
        XCTAssertThrowsError(try NeuralRallyContract.probabilities(logits: [0, 0, 0, .infinity]))
        let times = (0..<24).map { Double($0) / 4 }
        var p = [Float](repeating: 0, count: 96)
        for i in 0..<24 { p[i * 4] = (i >= 3 && i < 11) || (i >= 13 && i < 18) ? 0.7 : 0.01 }
        p[3 * 4 + 1] = 0.8; p[18 * 4 + 2] = 0.85
        let config = try NeuralDecoderConfig(smoothing: 0.5, minimum: 1, enter: 0.2, boundary: true)
        let ranges = try NeuralRallyContract.decode(times: times, probabilities: p, duration: 6, decoder: config)
        XCTAssertEqual(ranges.count, 1)
        XCTAssertEqual(ranges[0].start, 0.75); XCTAssertEqual(ranges[0].end, 4.5)
        XCTAssertEqual(ranges[0].confidence, Float(0.57062495))
        XCTAssertEqual(ranges[0].agreement, "neural")
        XCTAssertTrue(try NeuralRallyContract.decode(times: [], probabilities: [], duration: 6, decoder: config).isEmpty)
        // Float(0.9) is below Double(0.9), so this one-frame event must not bypass the minimum.
        let short = try NeuralDecoderConfig(smoothing: 0, minimum: 1, enter: 0.2, boundary: false)
        XCTAssertTrue(try NeuralRallyContract.decode(times: [0], probabilities: [0.9, 0, 0, 0], duration: 1, decoder: short).isEmpty)
        XCTAssertEqual(try NeuralRallyContract.decode(times: [0], probabilities: [0.91, 0, 0, 0], duration: 1, decoder: short).count, 1)
        XCTAssertThrowsError(try NeuralRallyContract.decode(times: [0], probabilities: [1, 0, 0], duration: 1, decoder: config))
    }
}
