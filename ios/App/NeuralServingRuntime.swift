import Foundation
import OnnxRuntimeBindings

enum NeuralServingRuntime {
    private static func session(_ bundle: NeuralServingBundle, resources: URL, environment: ORTEnv) throws -> ORTSession {
        let root = resources.appendingPathComponent("serving-models").appendingPathComponent(bundle.directory)
        _ = try NeuralRallyRuntime.verified(root, bundle.file)
        let options = try ORTSessionOptions()
        try options.setIntraOpNumThreads(4); try options.setGraphOptimizationLevel(.all)
        return try ORTSession(env: environment, modelPath: root.appendingPathComponent(bundle.file.name).path, sessionOptions: options)
    }
    private static func infer(_ session: ORTSession, windows: [Float], features: [Float], rows: Int) throws -> [Float] {
        let outputs = try session.run(withInputs: [
            "windows": NeuralRallyRuntime.tensor(windows, [rows, 13, 3843]),
            "features": NeuralRallyRuntime.tensor(features, [rows, 249])
        ], outputNames: ["logits"], runOptions: nil)
        return try NeuralRallyRuntime.output(outputs["logits"], shape: [rows, 3, 2])
    }
    #if DEBUG
    static func checkGolden() throws {
        struct Golden: Decodable {
            struct Variant: Decodable { let modelId: String, logits: [Float] }
            let rows: Int, variants: [String: Variant]
        }
        let resources = Bundle.main.resourceURL!
        let golden = try JSONDecoder().decode(Golden.self, from: Data(contentsOf: resources.appendingPathComponent("neural-serving-golden.json")))
        let catalog = try NeuralServingManifest(data: Data(contentsOf: resources.appendingPathComponent("serving-models/manifest.json")))
        let environment = try ORTEnv(loggingLevel: .warning)
        var windows = (0..<(golden.rows * 13 * 3843)).map { Float($0 % 101 - 50) / 100 }
        for row in 0..<golden.rows { for frame in 0..<13 {
            let base = (row * 13 + frame) * 3843
            windows[base + 3840] = 1; windows[base + 3841] = Float(frame - 6) / 6; windows[base + 3842] = 0
        } }
        let features = (0..<(golden.rows * 249)).map { Float($0 % 19 - 9) / 9 }
        for (key, bundle) in catalog.variants {
            let expected = golden.variants[key]!
            let actual = try infer(session(bundle, resources: resources, environment: environment), windows: windows, features: features, rows: golden.rows)
            guard expected.modelId == bundle.id, zip(actual, expected.logits).allSatisfy({ abs($0 - $1) < 0.00001 }) else {
                throw AnalysisError.invalid("Neural serving logits differ from the independent ONNX golden")
            }
        }
    }
    #endif
    static func evaluate(_ specialist: ServingSideOutput, model: RallyModel, url: URL, media: MediaDescription,
                         roi: AnalysisRegion, cacheFolder: URL, rallyRuntime: NeuralRallyRuntime? = nil,
                         progress: @Sendable (Double, String) -> Void) async throws -> ServingSideOutput {
        guard model.isNeural, let resources = Bundle.main.resourceURL else { throw AnalysisError.invalid("Missing neural serving-side resources") }
        let rallyAssets = resources.appendingPathComponent("rally-models")
        let rally = try NeuralRallyManifest(data: Data(contentsOf: rallyAssets.appendingPathComponent("manifest.json"))).bundle(for: model)
        let assets = resources.appendingPathComponent("serving-models")
        let bundle = try NeuralServingManifest(data: Data(contentsOf: assets.appendingPathComponent("manifest.json"))).bundle(for: rally)
        var result = specialist
        result.modelId = bundle.id; result.modelFingerprint = bundle.fingerprint
        guard !result.candidates.isEmpty else { return result }
        let needed = try NeuralServingContract.embeddingTimes(anchors: result.candidates.map(\.anchor), duration: media.duration)
        var embedding = rallyRuntime?.servingEmbeddings
        if rallyRuntime?.model != model || !Set(needed).isSubset(of: Set(embedding?.times ?? [])) {
            // Deferred scoring and clipped games need samples outside the rally window.
            // Use the exact same encoder/preprocessing with a sparse source-wide grid.
            let source = try ProjectArchive.sampledFingerprint(url: url)
            let geometry = String(decoding: try JSONEncoder().encode([roi.x, roi.y, roi.width, roi.height, media.duration, Double(media.rotation)]), as: UTF8.self)
            let runtime = try NeuralRallyRuntime(model: model, media: media, roi: roi, start: 0, end: media.duration,
                sourceIdentity: source + geometry + "serving-v1", cacheFolder: cacheFolder, servingTimes: needed)
            if !runtime.ready {
                try await MediaDecoder.embeddings(url: url, media: media, roi: roi, start: 0, end: media.duration, requestedTimes: needed,
                    progress: { progress($0 * 0.8, $1) }, embeddingFrame: { target, pts, rgb, geometry in
                        guard abs(pts - target) <= 0.25 + 1e-9 else { throw AnalysisError.invalid("Serving-side image timing is too sparse") }
                        try runtime.encode(target: target, presentation: pts, rgb: rgb, geometry: geometry)
                    })
            }
            try runtime.finishEmbeddings(); embedding = runtime.servingEmbeddings
        }
        guard let embedding else { throw AnalysisError.invalid("Missing serving-side image features") }
        let environment = try ORTEnv(loggingLevel: .warning)
        let session = try session(bundle, resources: resources, environment: environment)
        let features = try NeuralServingContract.features(raw: result.rawFeatures, rows: result.candidates.count)
        for start in stride(from: 0, to: result.candidates.count, by: 8) {
            try Task.checkCancellation()
            try autoreleasepool {
                let count = min(8, result.candidates.count - start)
                var windows: [Float] = []; windows.reserveCapacity(count * 13 * 3843)
                for row in start..<(start + count) {
                    windows += try NeuralServingContract.window(times: embedding.times, tokens: embedding.tokens,
                        anchor: result.candidates[row].anchor, duration: media.duration)
                }
                let logits = try infer(session, windows: windows, features: Array(features[(start * 249)..<((start + count) * 249)]), rows: count)
                for row in 0..<count {
                    result.candidates[start + row] = try NeuralServingContract.apply(
                        NeuralServingContract.decision(logits: Array(logits[(row * 6)..<((row + 1) * 6)]), bundle: bundle), to: result.candidates[start + row])
                }
            }
            progress(0.8 + 0.2 * Double(min(start + 8, result.candidates.count)) / Double(result.candidates.count), "Checking serve sides")
        }
        return result
    }
}
