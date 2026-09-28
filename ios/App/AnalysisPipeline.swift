import Foundation
import CryptoKit

struct FeatureCacheRecord: Codable {
    var schema = "ios-avfoundation-nv12-area-nearest-opencv412-audio-percentile-v2"
    var identity: String
    var complete = false
    var times: [Double] = []
    var decodedTimes: [Double] = []
    var visual: [Float] = []
    var audio: [Float] = []
    var base: [Float] = []
    var contextual: [Float] = []
}

struct AnalysisResult: Codable {
    var rallyModel: RallyModel?
    var neuralScores: NeuralRallyScores?
    var sourceName: String?
    var sourceSHA256: String?
    var modelSHA256: [String: String]?
    var decoderSchema: String?
    var media: MediaDescription
    var roi: AnalysisRegion
    var start: Double
    var end: Double
    var cacheIdentity: String
    var times: [Double]
    var decodedTimes: [Double]
    var base: [Float]
    var contextual: [Float]
    var allLabels: ModelRunner.RunResult
    var previous: ModelRunner.RunResult
    var intervals: [Interval]
    var suppression: SuppressionAnalysis?
    var servingSide: ServingSideOutput?
    var sideSwitch: SideSwitchOutput?
    var scoreError: String?
    var stageSeconds: [String: Double]
    var cacheHit: Bool
}

enum AnalysisPipeline {
    static let modelHashes = [
        "model-1ca43e38eefc": "d2c2c11e8fed8b6c6ad77d244b613e81d5bab101939a8f57be5166b45ebca78f",
        "model-9c92b8e9333f": "d8cc42f70bc10576a5e03251b05981ceeee1a61a15c61cc5dfb68dd631e6f90d"
    ]
    static func loadModel(_ name: String) throws -> ModelRunner {
        guard let url = Bundle.main.url(forResource: name, withExtension: "json") else { throw AnalysisError.invalid("Missing model \(name)") }
        let data = try Data(contentsOf: url)
        guard SHA256.hash(data: data).map({ String(format: "%02x", $0) }).joined() == modelHashes[name] else {
            throw AnalysisError.invalid("Model checksum mismatch")
        }
        return try ModelRunner(data: data)
    }

    static func analyze(url: URL, roi: AnalysisRegion, start requestedStart: Double, end requestedEnd: Double,
                        cacheFolder: URL, rallyModel: RallyModel = .default, prepareScore: Bool = true, generateSideSwitchMarkers: Bool = true,
                        stepProgress: @escaping @Sendable (AnalysisProgressEvent) -> Void = { _ in },
                        progress: @escaping @Sendable (Double, String) -> Void) async throws -> AnalysisResult {
        let media = try await MediaDecoder.describe(url)
        try roi.validate()
        let window = try AnalysisWindowBounds.normalize(start: requestedStart, end: requestedEnd, duration: media.duration)
        let start = Double(window.startMs) / 1000, end = Double(window.endMs) / 1000
        try FileManager.default.createDirectory(at: cacheFolder, withIntermediateDirectories: true)
        progress(0, "Identifying source")
        stepProgress(.init(.preparation, fraction: 0, detail: "Identifying source"))
        let hash = try sourceHash(url) { fraction in
            progress(fraction * 0.03, "Preparing video \(Int(fraction * 100))%")
            stepProgress(.init(.preparation, fraction: fraction * 0.9, detail: "Reading source \(Int(fraction * 100))%"))
        }
        let settings = try JSONEncoder().encode([start, end, roi.x, roi.y, roi.width, roi.height, Double(media.rotation)])
        let identity = SHA256.hash(data: Data((hash + FeatureCacheRecord(identity: "").schema).utf8) + settings).map { String(format: "%02x", $0) }.joined()
        let cacheURL = cacheFolder.appendingPathComponent(identity + ".plist")
        var cache = FeatureCacheRecord(identity: identity)
        if let data = try? Data(contentsOf: cacheURL) {
            let saved = try PropertyListDecoder().decode(FeatureCacheRecord.self, from: data)
            guard saved.schema == cache.schema, saved.identity == identity,
                  saved.visual.count == saved.times.count * 73,
                  saved.decodedTimes.count == saved.times.count else { throw AnalysisError.invalid("Feature cache is incompatible or damaged") }
            if !saved.times.isEmpty { try FeatureMath.validate(times: saved.times, values: saved.visual, columns: 73) }
            cache = saved
        }
        let cacheHit = cache.complete
        let neural = try rallyModel.isNeural ? NeuralRallyRuntime(model: rallyModel, media: media, roi: roi, start: start, end: end,
            sourceIdentity: identity, cacheFolder: cacheFolder) : nil
        // A partial AV checkpoint has no corresponding partial embedding contract.
        // Rebuild the shared pass rather than splice incompatible feature histories.
        if neural != nil && !cache.complete && neural?.ready == false { cache = FeatureCacheRecord(identity: identity) }
        stepProgress(.init(.preparation, fraction: 1, detail: "Video ready for analysis"))
        let encoder = PropertyListEncoder(); encoder.outputFormat = .binary
        var stages: [String: Double] = [:]
        let resumeSeconds = max(0, (cache.times.last ?? start) - start)
        stepProgress(.init(.video, fraction: cache.complete ? 1 : resumeSeconds / (end - start), detail: cache.complete ? "Reusing video features" : "Scanning video",
                           completedFrames: Double(cache.times.count), processedVideoSeconds: Double(cache.times.count) / Double(FeatureSchema.analysisFPS), resetsMeasurement: true))
        if !cache.complete {
            var began = Date()
            let savedDecoded = cache.decodedTimes
            let frameConsumer: ((Double, Double, [Float], NeuralImageGeometry) throws -> Void)? = neural?.ready == false ? { target, pts, rgb, geometry in
                try neural!.encode(target: target, presentation: pts, rgb: rgb, geometry: geometry)
            } : nil
            let visual = try await MediaDecoder.video(url: url, media: media, roi: roi, start: start, end: end,
                resumeTimes: cache.times, resumeVisual: cache.visual, embeddingFrame: frameConsumer,
                measurement: { fraction, frames, seconds in
                    let detail = neural.map { "Decoded \(frames) samples; image features \($0.completedFrames)/\($0.expectedTimes.count)" }
                        ?? "Decoded \(frames) samples"
                    stepProgress(.init(.video, fraction: fraction, detail: detail, completedFrames: Double(frames), processedVideoSeconds: seconds))
                },
                progress: { progress(0.03 + $0 * 0.57, neural == nil ? $1 : "\($1); image features \(neural!.completedFrames)/\(neural!.expectedTimes.count)") }) { times, decoded, visual in
                    cache.times = times; cache.decodedTimes = savedDecoded + decoded; cache.visual = visual
                    try encoder.encode(cache).write(to: cacheURL, options: .atomic)
                }
            cache.times = visual.times; cache.decodedTimes = savedDecoded + visual.decodedTimes; cache.visual = visual.visual
            stages["video"] = Date().timeIntervalSince(began)
            stepProgress(.init(.video, fraction: 1, detail: "Video scan complete", completedFrames: Double(cache.times.count), processedVideoSeconds: Double(cache.times.count) / Double(FeatureSchema.analysisFPS)))
            began = Date()
            stepProgress(.init(.audio, fraction: 0, detail: "Listening for play"))
            if cache.audio.isEmpty {
                cache.audio = try await MediaDecoder.audio(url: url, times: cache.times, start: start, end: end,
                    progress: { fraction, detail in
                        progress(0.60 + fraction * 0.15, detail)
                        stepProgress(.init(.audio, fraction: fraction, detail: detail))
                    })
                try encoder.encode(cache).write(to: cacheURL, options: .atomic)
            }
            stages["audio"] = Date().timeIntervalSince(began)
            stepProgress(.init(.audio, fraction: 1, detail: "Audio features ready"))
            began = Date()
            progress(0.75, "Preparing features")
            try FeatureMath.validate(times: cache.times, values: cache.audio, columns: 27)
            let temporal = try TemporalFeatures.generate(visual: cache.visual, rows: cache.times.count)
            var base: [Float] = []; base.reserveCapacity(cache.times.count * 104)
            for row in cache.times.indices {
                base += cache.visual[(row * 73)..<((row + 1) * 73)]
                base += temporal[(row * 4)..<((row + 1) * 4)]
                base += cache.audio[(row * 27)..<((row + 1) * 27)]
            }
            cache.base = base
            try Task.checkCancellation()
            cache.contextual = try FeatureMath.contextualize(times: cache.times, base: base, names: FeatureSchema.base)
            try Task.checkCancellation()
            cache.complete = true
            try encoder.encode(cache).write(to: cacheURL, options: .atomic)
            stages["context"] = Date().timeIntervalSince(began)
        } else {
            stepProgress(.init(.audio, fraction: 1, detail: "Reusing audio features"))
        }
        if let neural {
            stepProgress(.init(.embedding, fraction: neural.ready ? 1 : 0, detail: neural.ready ? "Image features ready" : "Generating image features"))
            if !neural.ready {
                let began = Date()
                try await MediaDecoder.embeddings(url: url, media: media, roi: roi, start: start, end: end,
                    progress: { fraction, detail in
                        progress(0.75 + fraction * 0.15, detail)
                        stepProgress(.init(.embedding, fraction: fraction, detail: detail, completedFrames: Double(neural.completedFrames)))
                    }, embeddingFrame: { target, pts, rgb, geometry in try neural.encode(target: target, presentation: pts, rgb: rgb, geometry: geometry) })
                stages["embeddingVideoPass"] = Date().timeIntervalSince(began)
            }
            try neural.finishEmbeddings()
            stepProgress(.init(.embedding, fraction: 1, detail: neural.cacheHit ? "Reusing image features" : "Image features ready"))
        }
        try FeatureMath.validate(times: cache.times, values: cache.base, columns: 104)
        try FeatureMath.validate(times: cache.times, values: cache.contextual, columns: 520)
        progress(0.90, "Finding rallies")
        stepProgress(.init(.rally, fraction: 0, detail: "Finding rallies"))
        try Task.checkCancellation()
        let began = Date()
        // Team-switch features include legacy component intervals. Preserve that
        // auxiliary contract when requested; none of these ranges override neural cuts.
        let allModel = try loadModel("model-1ca43e38eefc")
        let previousModel = try loadModel("model-9c92b8e9333f")
        let fullLegacy = neural == nil || (prepareScore && generateSideSwitchMarkers)
        let all = try fullLegacy ? allModel.run(times: cache.times, contextual: cache.contextual, duration: end)
            : allModel.runEvidence(times: cache.times, contextual: cache.contextual, duration: end)
        stepProgress(.init(.rally, fraction: 0.05, detail: neural == nil ? "Checking rally agreement" : "Preparing serve and court-state evidence"))
        try Task.checkCancellation()
        let previous = try fullLegacy ? previousModel.run(times: cache.times, contextual: cache.contextual, duration: end)
            : previousModel.runEvidence(times: cache.times, contextual: cache.contextual, duration: end)
        try Task.checkCancellation()
        var merged = ProductionEnsemble.merge(allLabelsV2: all.intervals, previousProduction: previous.intervals).compactMap { interval -> Interval? in
            let lower = max(start, interval.start), upper = min(end, interval.end)
            return upper > lower ? Interval(start: lower, end: upper, confidence: interval.confidence, agreement: interval.agreement) : nil
        }
        var neuralScores: NeuralRallyScores?
        var suppression: SuppressionAnalysis?
        if let neural {
            let output = try neural.run(times: cache.times, contextual: cache.contextual, start: start, end: end) { fraction, detail in
                progress(0.90 + fraction * 0.04, detail)
                stepProgress(.init(.rally, fraction: 0.10 + fraction * 0.90, detail: detail))
            }
            merged = output.intervals; neuralScores = output.scores
            for (key, value) in neural.stageSeconds { stages["neural/" + key] = value }
        } else {
            stepProgress(.init(.rally, fraction: 0.8, detail: "Preparing cleanup suggestions"))
            let suppressionData = try Data(contentsOf: Bundle.main.url(forResource: SuppressionModelRunner.modelId, withExtension: "json")!)
            guard SHA256.hash(data: suppressionData).map({ String(format: "%02x", $0) }).joined() == SuppressionModelRunner.assetSha256 else {
                throw AnalysisError.invalid("Suppression asset checksum mismatch")
            }
            let suppressionRun = try SuppressionModelRunner(data: suppressionData).run(times: cache.times, contextual: cache.contextual, duration: end)
            suppression = try SuppressionPolicyEngine.build(allLabelsV2: all.intervals, previousProduction: previous.intervals,
                probabilities: suppressionRun.probabilities, decoded: suppressionRun.decodedIntervals, duration: end)
        }
        try Task.checkCancellation()
        stages["inference"] = Date().timeIntervalSince(began)
        stepProgress(.init(.rally, fraction: 1, detail: "\(merged.count) rallies ready"))
        var servingSide: ServingSideOutput?, sideSwitch: SideSwitchOutput?, scoreError: String?
        if prepareScore {
            let scoreStart = Date()
            do {
                let score = try await ScoreAnalysis.run(url: url, media: media, roi: roi, ranges: merged,
                    all: ProductionServeOutput(modelId: "model-1ca43e38eefc", times: cache.times, probabilities: all.serveProbabilities, detections: all.serveDetections),
                    previous: ProductionServeOutput(modelId: "model-9c92b8e9333f", times: cache.times, probabilities: previous.serveProbabilities, detections: previous.serveDetections),
                    cacheURL: cacheFolder.appendingPathComponent(identity + "-" + rallyModel.rawValue + "-serving.plist"),
                    switchInput: generateSideSwitchMarkers ? ScoreAnalysis.input(ranges: merged, times: cache.times, all: all, previous: previous) : nil,
                    stepProgress: stepProgress) { progress(0.94 + $0 * 0.06, $1) }
                servingSide = score.servingSide; sideSwitch = score.sideSwitch; scoreError = score.error
            } catch is CancellationError { throw CancellationError() }
            catch {
                scoreError = String(describing: error)
                stepProgress(.init(.servingSide, fraction: 0, detail: "Serve-marker preparation stopped", failed: true))
            }
            stages["scoreSpecialists"] = Date().timeIntervalSince(scoreStart)
        }
        progress(1, "\(merged.count) rallies ready")
        return AnalysisResult(rallyModel: rallyModel, neuralScores: neuralScores, sourceName: url.lastPathComponent, sourceSHA256: hash, modelSHA256: modelHashes,
                              decoderSchema: cache.schema, media: media, roi: roi, start: start, end: end, cacheIdentity: identity,
                              times: cache.times, decodedTimes: cache.decodedTimes, base: cache.base, contextual: cache.contextual,
                              allLabels: all, previous: previous, intervals: merged, suppression: suppression,
                              servingSide: servingSide, sideSwitch: sideSwitch, scoreError: scoreError, stageSeconds: stages, cacheHit: cacheHit)
    }

    static func sourceHash(_ url: URL, progress: (Double) -> Void = { _ in }) throws -> String {
        let handle = try FileHandle(forReadingFrom: url); defer { try? handle.close() }
        let size = try handle.seekToEnd(); try handle.seek(toOffset: 0)
        var consumed: UInt64 = 0
        var hash = SHA256()
        while let block = try handle.read(upToCount: 4 * 1024 * 1024), !block.isEmpty {
            try Task.checkCancellation(); hash.update(data: block)
            consumed += UInt64(block.count); progress(size == 0 ? 1 : Double(consumed) / Double(size))
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    #if DEBUG
    static func golden() throws -> String {
        try ServingSideFeatureExtractor.validateStaticGrayFixture()
        try SideSwitchFeatureExtractor.validateStaticFixture()
        struct Golden: Decodable { var rows: Int; var columns: Int; var duration: Double; var rallies: [Interval] }
        let metadata = try JSONDecoder().decode(Golden.self, from: Data(contentsOf: Bundle.main.url(forResource: "golden", withExtension: "json")!))
        let data = try Data(contentsOf: Bundle.main.url(forResource: "base", withExtension: "bin")!)
        let times: [Double] = (0..<metadata.rows).map { row in data.withUnsafeBytes { Double(bitPattern: UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: row * 8, as: UInt64.self))) } }
        let base: [Float] = (0..<(metadata.rows * metadata.columns)).map { i in data.withUnsafeBytes { Float(bitPattern: UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: metadata.rows * 8 + i * 4, as: UInt32.self))) } }
        let context = try FeatureMath.contextualize(times: times, base: base, names: FeatureSchema.base)
        let all = try loadModel("model-1ca43e38eefc").run(times: times, contextual: context, duration: metadata.duration)
        guard all.intervals.count == 42 else { throw AnalysisError.invalid("Golden rally count \(all.intervals.count), expected 42") }
        for (actual, expected) in zip(all.intervals, metadata.rallies) {
            guard abs(actual.start - expected.start) <= 0.00051, abs(actual.end - expected.end) <= 0.00051,
                  abs(actual.confidence - expected.confidence) <= 0.00002 else { throw AnalysisError.invalid("Golden interval mismatch") }
        }
        let previous = try loadModel("model-9c92b8e9333f").run(times: times, contextual: context, duration: metadata.duration)
        return "PASS: 42 canonical rallies; \(ProductionEnsemble.merge(allLabelsV2: all.intervals, previousProduction: previous.intervals).count) ensemble rallies; 3474 × 104 features"
    }
    #endif
}
