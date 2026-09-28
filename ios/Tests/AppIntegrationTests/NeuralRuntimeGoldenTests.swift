import CryptoKit
import Foundation
import OnnxRuntimeBindings
import XCTest
@testable import VolleySplice

/// Independent WASM-vs-native graph qualification on public synthetic inputs.
/// This intentionally runs on CPU and does not claim GPU speed or rally accuracy.
final class NeuralRuntimeGoldenTests: XCTestCase {
    private struct FloatArray: Decodable {
        let encoding: String, byteOrder: String, dataType: String, count: Int, sha256: String, data: String
        func decoded() throws -> [Float] {
            XCTAssertEqual(encoding, "base64"); XCTAssertEqual(byteOrder, "little-endian"); XCTAssertEqual(dataType, "float32")
            let bytes = try XCTUnwrap(Data(base64Encoded: data))
            XCTAssertEqual(bytes.count, count * 4); XCTAssertEqual(NeuralRuntimeGoldenTests.hash(bytes), sha256)
            return NeuralRuntimeGoldenTests.floats(bytes)
        }
    }
    private struct Variant: Decodable {
        let modelId: String, encoderSha256: String, temporalSha256: String
        let tokens: FloatArray, logits: FloatArray
        let maximumChunkAbsoluteError: Double
    }
    private struct Golden: Decodable {
        struct Reference: Decodable { let backend: String, version: String, threads: Int }
        struct Tolerance: Decodable { let absolute: Double, relative: Double }
        let schemaVersion: Int, reference: Reference
        let dimensions: [String: Int], inputSha256: [String: String], tolerance: Tolerance
        let variants: [String: Variant]
    }
    private func fixture() throws -> Golden {
        let url = try XCTUnwrap(Bundle.main.url(forResource: "neural-runtime-golden", withExtension: "json"))
        let value = try JSONDecoder().decode(Golden.self, from: Data(contentsOf: url))
        XCTAssertEqual(value.schemaVersion, 1)
        XCTAssertEqual(value.reference.backend, "onnxruntime-web/wasm")
        XCTAssertEqual(value.reference.version, "1.22.0")
        XCTAssertEqual(value.reference.threads, 1)
        XCTAssertEqual(value.tolerance.absolute, 1e-4); XCTAssertEqual(value.tolerance.relative, 1e-4)
        return value
    }
    private static func hash(_ bytes: Data) -> String { SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined() }
    private static func bytes(_ values: [Float]) -> Data {
        var bytes = Data(capacity: values.count * 4)
        for value in values {
            var bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { bytes.append(contentsOf: $0) }
        }
        return bytes
    }
    private static func floats(_ data: Data) -> [Float] {
        data.withUnsafeBytes { bytes in
            (0..<(bytes.count / 4)).map { Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) }
        }
    }
    private var assets: URL { get throws { try XCTUnwrap(Bundle.main.resourceURL).appendingPathComponent("rally-models") } }
    private func manifest() throws -> NeuralRallyManifest {
        try NeuralRallyManifest(data: Data(contentsOf: assets.appendingPathComponent("manifest.json")))
    }
    private func session(_ env: ORTEnv, _ bundle: NeuralRallyBundle, _ asset: NeuralRallyAsset) throws -> ORTSession {
        let url = try assets.appendingPathComponent(bundle.directory).appendingPathComponent(asset.name)
        let data = try Data(contentsOf: url)
        XCTAssertEqual(data.count, asset.sizeBytes); XCTAssertEqual(Self.hash(data), asset.sha256)
        let options = try ORTSessionOptions()
        try options.setIntraOpNumThreads(4)
        try options.setGraphOptimizationLevel(.all)
        return try ORTSession(env: env, modelPath: url.path, sessionOptions: options)
    }
    private func tensor(_ values: [Float], shape: [Int]) throws -> ORTValue {
        let data = NSMutableData(data: Self.bytes(values))
        return try ORTValue(tensorData: data, elementType: .float, shape: shape.map(NSNumber.init(value:)))
    }
    private func output(_ value: ORTValue?, shape: [Int]) throws -> [Float] {
        let value = try XCTUnwrap(value)
        XCTAssertEqual(try value.tensorTypeAndShapeInfo().shape.map(\.intValue), shape)
        let data = try value.tensorData() as Data
        let values = Self.floats(data)
        XCTAssertEqual(values.count, shape.reduce(1, *))
        XCTAssertTrue(values.allSatisfy(\.isFinite))
        return values
    }
    private func compare(_ actual: [Float], _ expected: [Float], tolerance: Golden.Tolerance, name: String,
                         file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.count, expected.count, name, file: file, line: line)
        var maximum = 0.0, bad = 0, worst = 0
        for (index, values) in zip(actual, expected).enumerated() {
            let error = abs(Double(values.0) - Double(values.1))
            if error > maximum { maximum = error; worst = index }
            if !values.0.isFinite || error > tolerance.absolute + tolerance.relative * abs(Double(values.1)) { bad += 1 }
        }
        let summary = "\(name): \(actual.count) floats, maximum absolute error \(maximum) at \(worst), outside tolerance \(bad)"
        print(summary)
        let attachment = XCTAttachment(string: summary); attachment.lifetime = .keepAlways; add(attachment)
        XCTAssertEqual(bad, 0, summary, file: file, line: line)
    }

    func testBothNativeEncodersMatchAllSyntheticWasmTokens() throws {
        let golden = try fixture(), manifest = try manifest(), env = try ORTEnv(loggingLevel: .warning)
        XCTAssertEqual(golden.dimensions["imageSize"], 224)
        XCTAssertEqual(golden.dimensions["contentHeight"], 126)
        let geometry = try NeuralRallyContract.geometry(width: 1920, height: 1080)
        var rgb = [Float](repeating: 0, count: 224 * 224 * 3)
        for y in 49..<175 { for x in 0..<224 { for channel in 0..<3 {
            rgb[(y * 224 + x) * 3 + channel] = Float((x * 17 + (y - 49) * 29 + channel * 41) % 1021) / 4
        } } }
        let image = try NeuralRallyContract.normalizedCHW(rgb: rgb)
        let pool = try NeuralRallyContract.regionalPoolWeights(box: geometry.box)
        XCTAssertEqual(Self.hash(Self.bytes(image)), golden.inputSha256["image"])
        XCTAssertEqual(Self.hash(Self.bytes(pool)), golden.inputSha256["pool"])
        for model in [RallyModel.maximumCoverage, .balanced] {
            try autoreleasepool {
                let bundle = try manifest.bundle(for: model)
                let expected = try XCTUnwrap(golden.variants[model.variantKey!])
                XCTAssertEqual(expected.modelId, model.modelId)
                XCTAssertEqual(expected.encoderSha256, bundle.files.encoder.sha256)
                let session = try session(env, bundle, bundle.files.encoder)
                let outputs = try session.run(withInputs: ["image": try tensor(image, shape: [1, 3, 224, 224]),
                    "pool_weights": try tensor(pool, shape: [1, 4, 7, 7])], outputNames: ["tokens"], runOptions: nil)
                let tokens = try output(outputs["tokens"], shape: [1, 4, 960])
                compare(tokens, try expected.tokens.decoded(), tolerance: golden.tolerance, name: model.rawValue + " encoder")
            }
        }
    }

    func testBothTemporalHeadsMatchWasmAndWholeRunAcrossChunkSeams() throws {
        let golden = try fixture(), manifest = try manifest(), env = try ORTEnv(loggingLevel: .warning)
        XCTAssertEqual(golden.dimensions["featureRows"], 301)
        XCTAssertEqual(golden.dimensions["featureColumns"], NeuralRallyContract.fusedDimension)
        XCTAssertEqual(golden.dimensions["chunkRows"], NeuralRallyContract.chunkRows)
        XCTAssertEqual(golden.dimensions["contextRows"], NeuralRallyContract.contextRows)
        var features = [Float](repeating: 0, count: 301 * 3952)
        for row in 0..<301 { for column in 0..<3952 {
            features[row * 3952 + column] = Float((row * 31 + column * 17 + (column / 104) * 7) % 257 - 128) / 128
        } }
        XCTAssertEqual(Self.hash(Self.bytes(features)), golden.inputSha256["features"])
        for model in [RallyModel.maximumCoverage, .balanced] {
            try autoreleasepool {
                let bundle = try manifest.bundle(for: model)
                let expected = try XCTUnwrap(golden.variants[model.variantKey!])
                XCTAssertEqual(expected.temporalSha256, bundle.files.temporal.sha256)
                let session = try session(env, bundle, bundle.files.temporal)
                let whole = try session.run(withInputs: ["features": try tensor(features, shape: [1, 301, 3952])],
                    outputNames: ["logits"], runOptions: nil)
                let logits = try output(whole["logits"], shape: [1, 301, 4])
                compare(logits, try expected.logits.decoded(), tolerance: golden.tolerance, name: model.rawValue + " temporal")
                var chunked = [Float](repeating: 0, count: 301 * 4)
                for chunk in try NeuralRallyContract.chunkPlan(rows: 301) {
                    try autoreleasepool {
                        let input = Array(features[(chunk.input.lowerBound * 3952)..<(chunk.input.upperBound * 3952)])
                        let outputs = try session.run(withInputs: ["features": try tensor(input, shape: [1, chunk.input.count, 3952])],
                            outputNames: ["logits"], runOptions: nil)
                        let values = try output(outputs["logits"], shape: [1, chunk.input.count, 4])
                        for row in chunk.core { for channel in 0..<4 {
                            chunked[row * 4 + channel] = values[(row - chunk.input.lowerBound) * 4 + channel]
                        } }
                    }
                }
                // Includes both boundary halos, internal seams at128/256, and the partial final chunk.
                compare(chunked, logits, tolerance: golden.tolerance, name: model.rawValue + " native chunk seams")
                compare(chunked, try expected.logits.decoded(), tolerance: golden.tolerance, name: model.rawValue + " chunked versus WASM")
                let config = try NeuralPipelineConfig(data: Data(contentsOf: assets.appendingPathComponent(bundle.directory).appendingPathComponent("pipeline.json")), model: model, bundle: bundle)
                let times = (0..<301).map { Double($0) / 4 }
                let fullRanges = try NeuralRallyContract.decode(times: times, probabilities: NeuralRallyContract.probabilities(logits: logits), duration: 75.25, decoder: config.decoder)
                let chunkRanges = try NeuralRallyContract.decode(times: times, probabilities: NeuralRallyContract.probabilities(logits: chunked), duration: 75.25, decoder: config.decoder)
                XCTAssertEqual(chunkRanges.map(\.start), fullRanges.map(\.start))
                XCTAssertEqual(chunkRanges.map(\.end), fullRanges.map(\.end))
            }
        }
    }
}
