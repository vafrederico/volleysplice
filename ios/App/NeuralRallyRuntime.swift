import Foundation
import CryptoKit
import OnnxRuntimeBindings

/// One matched, frozen model per analysis. FP32 CPU is the initial native path;
/// hardware provider qualification is deliberately separate from simulator timing.
final class NeuralRallyRuntime {
    static let preprocessingVersion = "ios-nv12-linear224-nearest2hz-fp32-v2"
    private struct Cache: Codable {
        var identity: String
        var times: [Double]
        var tokens: Data
        var quality: [Float]
    }
    let model: RallyModel
    let bundle: NeuralRallyBundle
    let config: NeuralPipelineConfig
    let expectedTimes: [Double]
    private let root: URL, cacheURL: URL, identity: String
    private let environment: ORTEnv
    private var encoder: ORTSession?
    private var tokens: [Float] = [], quality: [Float] = [], times: [Double] = []
    private(set) var cacheHit = false
    private var persisted = false
    private(set) var stageSeconds: [String: Double] = [:]
    var ready: Bool { times == expectedTimes && tokens.count == times.count * 3840 }
    var completedFrames: Int { times.count }

    init(model: RallyModel, media: MediaDescription, roi: AnalysisRegion, start: Double, end: Double,
         sourceIdentity: String, cacheFolder: URL, assets: URL? = nil) throws {
        self.model = model
        guard model.isNeural, let assets = assets ?? Bundle.main.resourceURL?.appendingPathComponent("rally-models") else {
            throw AnalysisError.invalid("Selected rally model is unavailable")
        }
        let manifest = try NeuralRallyManifest(data: Data(contentsOf: assets.appendingPathComponent("manifest.json")))
        bundle = try manifest.bundle(for: model)
        root = assets.appendingPathComponent(bundle.directory)
        config = try NeuralPipelineConfig(data: Self.verified(root, bundle.files.pipeline), model: model, bundle: bundle)
        expectedTimes = try NeuralRallyContract.embeddingTimes(duration: media.duration, start: start, end: end)
        identity = Self.hash(Data((sourceIdentity + Self.preprocessingVersion + bundle.files.encoder.sha256).utf8))
        cacheURL = cacheFolder.appendingPathComponent(identity + "-embeddings.plist")
        environment = try ORTEnv(loggingLevel: .warning)
        if let bytes = try? Data(contentsOf: cacheURL), let saved = try? PropertyListDecoder().decode(Cache.self, from: bytes),
           saved.identity == identity, saved.times == expectedTimes,
           saved.tokens.count == expectedTimes.count * 3840 * 4,
           saved.quality.count == expectedTimes.count * 6, saved.quality.allSatisfy(\.isFinite) {
            let decoded = Self.floats(saved.tokens)
            if decoded.allSatisfy(\.isFinite) {
                times = saved.times; tokens = decoded; quality = saved.quality; cacheHit = true
            }
        }
    }

    private static func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    private static func verified(_ root: URL, _ asset: NeuralRallyAsset) throws -> Data {
        let bytes = try Data(contentsOf: root.appendingPathComponent(asset.name), options: .mappedIfSafe)
        guard bytes.count == asset.sizeBytes, hash(bytes) == asset.sha256 else {
            throw AnalysisError.invalid("Selected rally model failed its integrity check")
        }
        return bytes
    }
    private func session(_ asset: NeuralRallyAsset) throws -> ORTSession {
        _ = try Self.verified(root, asset)
        let options = try ORTSessionOptions()
        try options.setIntraOpNumThreads(4)
        try options.setGraphOptimizationLevel(.all)
        return try ORTSession(env: environment, modelPath: root.appendingPathComponent(asset.name).path, sessionOptions: options)
    }
    private static func tensor(_ values: [Float], _ shape: [Int]) throws -> ORTValue {
        let bytes = values.withUnsafeBytes { NSMutableData(bytes: $0.baseAddress!, length: $0.count) }
        return try ORTValue(tensorData: bytes, elementType: .float, shape: shape.map(NSNumber.init(value:)))
    }
    private static func floats(_ data: Data) -> [Float] {
        data.withUnsafeBytes { bytes in
            (0..<(bytes.count / 4)).map { Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) }
        }
    }
    private static func output(_ value: ORTValue?, shape: [Int]) throws -> [Float] {
        guard let value, try value.tensorTypeAndShapeInfo().shape.map(\.intValue) == shape else {
            throw AnalysisError.invalid("Neural model output shape changed")
        }
        let values = floats(try value.tensorData() as Data)
        guard values.allSatisfy(\.isFinite) else { throw AnalysisError.invalid("Non-finite neural model output") }
        return values
    }
    func encode(target: Double, presentation: Double, rgb: [Float], geometry: NeuralImageGeometry) throws {
        try Task.checkCancellation()
        guard times.count < expectedTimes.count, abs(target - expectedTimes[times.count]) < 1e-8 else {
            throw AnalysisError.invalid("Neural image samples arrived out of order")
        }
        if encoder == nil {
            let began = Date(); encoder = try session(bundle.files.encoder)
            stageSeconds["encoderLoad"] = Date().timeIntervalSince(began)
        }
        try autoreleasepool {
            var began = Date()
            let chw = try NeuralRallyContract.normalizedCHW(rgb: rgb)
            var rowQuality = try NeuralRallyContract.quality(chw: chw, geometry: geometry)
            // MediaCodec's contract records integer-microsecond presentation times.
            rowQuality[5] = Float((presentation * 1_000_000).rounded() / 1_000_000 - target)
            let pool = try NeuralRallyContract.regionalPoolWeights(box: geometry.box)
            let inputs = ["image": try Self.tensor(chw, [1, 3, 224, 224]), "pool_weights": try Self.tensor(pool, [1, 4, 7, 7])]
            stageSeconds["imagePreparation", default: 0] += Date().timeIntervalSince(began)
            began = Date()
            let output = try encoder!.run(withInputs: inputs, outputNames: ["tokens"], runOptions: nil)
            let row = try Self.output(output["tokens"], shape: [1, 4, 960])
            stageSeconds["encoderInference", default: 0] += Date().timeIntervalSince(began)
            times.append(target); tokens.append(contentsOf: row); quality.append(contentsOf: rowQuality)
        }
    }
    func finishEmbeddings() throws {
        guard ready else { throw AnalysisError.invalid("Video did not produce all required neural image features") }
        encoder = nil
        if !cacheHit && !persisted {
            let bytes = tokens.withUnsafeBytes { Data($0) }
            let writer = PropertyListEncoder(); writer.outputFormat = .binary
            try writer.encode(Cache(identity: identity, times: times, tokens: bytes, quality: quality)).write(to: cacheURL, options: .atomic)
            persisted = true
        }
    }
    func run(times avTimes: [Double], contextual: [Float], start: Double, end: Double,
             progress: (Double, String) -> Void) throws -> (intervals: [Interval], scores: NeuralRallyScores) {
        try finishEmbeddings()
        let began = Date(), temporal = try session(bundle.files.temporal)
        var probabilities = [Float](repeating: 0, count: avTimes.count * 4)
        let chunks = try NeuralRallyContract.chunkPlan(rows: avTimes.count)
        for (index, chunk) in chunks.enumerated() {
            try Task.checkCancellation()
            try autoreleasepool {
                let values = try NeuralRallyContract.fuse(times: avTimes, contextual: contextual, embeddingTimes: times,
                    tokens: tokens, quality: quality, config: config, rows: chunk.input)
                let outputs = try temporal.run(withInputs: ["features": try Self.tensor(values, [1, chunk.input.count, 3952])], outputNames: ["logits"], runOptions: nil)
                let logits = try Self.output(outputs["logits"], shape: [1, chunk.input.count, 4])
                let decoded = try NeuralRallyContract.probabilities(logits: logits)
                for row in chunk.core { for column in 0..<4 {
                    probabilities[row * 4 + column] = decoded[(row - chunk.input.lowerBound) * 4 + column]
                } }
            }
            progress(Double(index + 1) / Double(chunks.count), "Finding rallies \(index + 1)/\(chunks.count)")
        }
        stageSeconds["temporalInference"] = Date().timeIntervalSince(began)
        let ranges = try NeuralRallyContract.decode(times: avTimes, probabilities: probabilities, duration: end, decoder: config.decoder).compactMap { value -> Interval? in
            let left = max(start, value.start), right = min(end, value.end)
            return right > left ? Interval(start: left, end: right, confidence: value.confidence, agreement: "neural") : nil
        }
        return (ranges, try NeuralRallyScores(modelId: model.modelId, times: avTimes, probabilities: probabilities))
    }
}
