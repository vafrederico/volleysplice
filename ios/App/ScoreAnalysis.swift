import Foundation
import CryptoKit

enum ScoreAnalysis {
    struct Result: Sendable { var servingSide: ServingSideOutput; var sideSwitch: SideSwitchOutput?; var error: String? }
    static func prepare(project: ProjectDocument, source: URL, cacheFolder: URL,
                        stepProgress: @escaping @Sendable (AnalysisProgressEvent) -> Void = { _ in },
                        progress: @escaping @Sendable (Double, String) -> Void) async throws -> Result {
        stepProgress(.init(.servingSide, fraction: 0, detail: "Reusing project features for score preparation"))
        guard let feedback = project.feedback,
              let analysis = try ProjectArchive.retainedAnalysis(feedback, expectedFeatureNames: FeatureSchema.base),
              let initial = feedback["initialInference"]?["ranges"]?.array,
              let region = feedback["source"]?["featureRoi"] else {
            throw ProjectError.invalid("Preparing scores requires the original project features and inference")
        }
        let media = try await MediaDecoder.describe(source)
        let roi = AnalysisRegion(x: region["x"]?.double ?? 0, y: region["y"]?.double ?? 0,
                                 width: region["width"]?.double ?? 1, height: region["height"]?.double ?? 1)
        let ranges = try initial.map { range -> Interval in
            guard let start = range["start"]?.double, let end = range["end"]?.double,
                  start.isFinite, end.isFinite, start >= 0, end > start else { throw ProjectError.invalid("Missing score anchor") }
            return Interval(start: start, end: end, confidence: Float(range["confidence"]?.double ?? 1), agreement: range["agreement"]?.string)
        }
        progress(0, "Reusing project features for score preparation")
        let contextual = try FeatureMath.contextualize(times: analysis.timestamps, base: analysis.baseFeatures, names: FeatureSchema.base)
        let all = try AnalysisPipeline.loadModel("model-1ca43e38eefc").run(times: analysis.timestamps, contextual: contextual, duration: Double(project.gameWindow.endMs) / 1000)
        let previous = try AnalysisPipeline.loadModel("model-9c92b8e9333f").run(times: analysis.timestamps, contextual: contextual, duration: Double(project.gameWindow.endMs) / 1000)
        try FileManager.default.createDirectory(at: cacheFolder, withIntermediateDirectories: true)
        let safeID = project.id.filter { $0.isLetter || $0.isNumber || $0 == "-" }
        let sourceKey = try ProjectArchive.sampledFingerprint(url: source)
        let geometry = try JSONEncoder().encode([roi.x, roi.y, roi.width, roi.height, Double(media.rotation), media.duration])
        let cacheKey = SHA256.hash(data: Data(sourceKey.utf8) + geometry).map { String(format: "%02x", $0) }.joined()
        return try await run(url: source, media: media, roi: roi, ranges: ranges,
            all: ProductionServeOutput(modelId: "model-1ca43e38eefc", times: analysis.timestamps, probabilities: all.serveProbabilities, detections: all.serveDetections),
            previous: ProductionServeOutput(modelId: "model-9c92b8e9333f", times: analysis.timestamps, probabilities: previous.serveProbabilities, detections: previous.serveDetections),
            cacheURL: cacheFolder.appendingPathComponent("project-\(safeID)-\(cacheKey)-serving.plist"),
            switchInput: (feedback["source"]?["generateSideSwitchMarkers"]?.bool ?? true)
                ? input(ranges: ranges, times: analysis.timestamps, all: all, previous: previous) : nil, stepProgress: stepProgress, progress: progress)
    }
    static func input(ranges: [Interval], times: [Double], all: ModelRunner.RunResult, previous: ModelRunner.RunResult) -> SideSwitchAnalysisInput {
        SideSwitchAnalysisInput(intervals: ranges, allLabelsV2Components: all.intervals, previousProductionComponents: previous.intervals,
            allLabelsV2State: ProductionStateOutput(modelId: "model-1ca43e38eefc", times: times, rallyProbabilities: all.rallyProbabilities, deadStateProbabilities: all.deadStateProbabilities),
            previousProductionState: ProductionStateOutput(modelId: "model-9c92b8e9333f", times: times, rallyProbabilities: previous.rallyProbabilities, deadStateProbabilities: previous.deadStateProbabilities))
    }
    private struct Cache: Codable {
        var schema = "ios-serving-gray-opencv412-v1"
        var candidates: [ServingSideInference.CandidateInterval]
        var rawFeatures: [Double] = []
    }
    private struct SwitchCache: Codable {
        var schema = "ios-switch-bgr-opencv412-v1"
        var plan: SideSwitchInference.FramePlan
        var visual: [Double]
    }
    static func run(url: URL, media: MediaDescription, roi: AnalysisRegion, ranges: [Interval],
                        all: ProductionServeOutput, previous: ProductionServeOutput, cacheURL: URL,
                        switchInput: SideSwitchAnalysisInput? = nil,
                        stepProgress: @escaping @Sendable (AnalysisProgressEvent) -> Void = { _ in },
                        progress: @escaping @Sendable (Double, String) -> Void) async throws -> Result {
        stepProgress(.init(.servingSide, fraction: 0, detail: "Finding serve markers"))
        let plan = try ServingSideInference.framePlan(ranges: ranges, duration: media.duration)
        let modelURL = Bundle.main.url(forResource: "serving-side-85bc3325fbd4", withExtension: "json")!
        let model = try ServingSideInference(data: Data(contentsOf: modelURL))
        let switchModel = try SideSwitchInference(data: Data(contentsOf: Bundle.main.url(forResource: "side-switch-c2570481c30d", withExtension: "json")!))
        let switchPlan = try switchInput.map { try switchModel.framePlan(input: $0, duration: media.duration) }
        let switchCacheURL = cacheURL.deletingPathExtension().appendingPathExtension("switch.plist")
        var switchVisual: [Double]?
        var switchError: String?
        if let switchPlan, let data = try? Data(contentsOf: switchCacheURL),
           let saved = try? PropertyListDecoder().decode(SwitchCache.self, from: data),
           saved.schema == "ios-switch-bgr-opencv412-v1", saved.plan == switchPlan,
           saved.visual.count == switchPlan.candidates.count * 22, saved.visual.allSatisfy(\.isFinite) { switchVisual = saved.visual }
        var cache = Cache(candidates: plan.candidates)
        if let data = try? Data(contentsOf: cacheURL), let saved = try? PropertyListDecoder().decode(Cache.self, from: data),
           saved.schema == cache.schema, saved.candidates == cache.candidates,
           saved.rawFeatures.count % ServingSideInference.columns == 0,
           saved.rawFeatures.count <= plan.candidates.count * ServingSideInference.columns,
           saved.rawFeatures.allSatisfy(\.isFinite) { cache = saved }
        let completed = cache.rawFeatures.count / ServingSideInference.columns
        let pending = Array(plan.candidates.dropFirst(completed))
        let targets = try ServingSideInference.requestedTimestamps(candidates: pending, duration: media.duration)
        let switchTargets = switchVisual == nil ? switchPlan?.requestedTimes ?? [] : []
        if !targets.isEmpty || !switchTargets.isEmpty {
            let frames = try await SpecialistFrameDecoder.decode(url: url, media: media, roi: roi, servingTimes: targets, sideSwitchTimes: switchTargets) {
                progress($0 * 0.6, $1)
                stepProgress(.init(.servingSide, fraction: $0 * 0.68, detail: $1))
            }
            let encoder = PropertyListEncoder(); encoder.outputFormat = .binary
            for (index, candidate) in pending.enumerated() {
                try Task.checkCancellation()
                cache.rawFeatures += try ServingSideFeatureExtractor.extract(candidate: candidate, duration: media.duration, grayByTime: frames.servingGray)
                try encoder.encode(cache).write(to: cacheURL, options: .atomic)
                progress(0.6 + Double(index + 1) / Double(pending.count) * 0.25, "Serving side \(completed + index + 1)/\(plan.candidates.count)")
                stepProgress(.init(.servingSide, fraction: 0.68 + Double(index + 1) / Double(pending.count) * 0.30,
                                   detail: "Serve markers \(completed + index + 1)/\(plan.candidates.count)"))
            }
            // Evaluate serves before starting the separate team-switch step.
            stepProgress(.init(.servingSide, fraction: 0.99, detail: "Checking serve markers"))
            let serving = try model.evaluate(candidates: plan.candidates, rawFeatures: cache.rawFeatures, allLabelsV2: all, previousProduction: previous)
            stepProgress(.init(.servingSide, fraction: 1, detail: "Serve markers ready"))
            if let switchPlan, switchVisual == nil {
                progress(0.86, "Generating team-switch features")
                stepProgress(.init(.sideSwitch, fraction: 0, detail: "Generating team-switch features"))
                do {
                    switchVisual = try SideSwitchFeatureExtractor.visualFeatures(plan: switchPlan, duration: media.duration, bgrByTime: frames.sideSwitchBGR) {
                        progress(0.86 + $0 * 0.14, $1)
                        stepProgress(.init(.sideSwitch, fraction: $0 * 0.98, detail: $1))
                    }
                    try encoder.encode(SwitchCache(plan: switchPlan, visual: switchVisual!)).write(to: switchCacheURL, options: .atomic)
                } catch is CancellationError { throw CancellationError() }
                catch { switchError = "Team-switch features: \(error)" }
            }
            return try finish(serving: serving, switchModel: switchModel, switchPlan: switchPlan, switchInput: switchInput,
                              switchVisual: switchVisual, switchError: switchError, stepProgress: stepProgress)
        }
        let serving = try model.evaluate(candidates: plan.candidates, rawFeatures: cache.rawFeatures, allLabelsV2: all, previousProduction: previous)
        stepProgress(.init(.servingSide, fraction: 1, detail: "Serve markers ready"))
        return try finish(serving: serving, switchModel: switchModel, switchPlan: switchPlan, switchInput: switchInput,
                          switchVisual: switchVisual, switchError: switchError, stepProgress: stepProgress)
    }
    private static func finish(serving: ServingSideOutput, switchModel: SideSwitchInference, switchPlan: SideSwitchInference.FramePlan?,
                               switchInput: SideSwitchAnalysisInput?, switchVisual: [Double]?, switchError: String?,
                               stepProgress: @Sendable (AnalysisProgressEvent) -> Void) throws -> Result {
        var switchError = switchError
        var switches: SideSwitchOutput?
        if let switchPlan, let switchInput, switchVisual != nil || switchPlan.candidates.isEmpty {
            stepProgress(.init(.sideSwitch, fraction: 0.98, detail: "Checking team switches"))
            do { switches = try switchModel.evaluate(input: switchInput, plan: switchPlan, visualFeatures: switchVisual ?? []) }
            catch is CancellationError { throw CancellationError() }
            catch { switchError = "Team-switch inference: \(error)" }
        }
        if switchInput != nil {
            stepProgress(.init(.sideSwitch, fraction: switchError == nil ? 1 : 0.98,
                               detail: switchError ?? "Team switches ready", failed: switchError != nil))
        }
        return Result(servingSide: serving, sideSwitch: switches, error: switchError)
    }
}
