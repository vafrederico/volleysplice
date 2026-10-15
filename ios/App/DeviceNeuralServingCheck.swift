#if DEBUG
import Foundation

/// Opt-in physical-device diagnostic. Private inputs and receipts stay in Documents.
/// Exercises deferred scoring with native extraction, then replays the saved browser
/// Float64 specialist matrix through the same encoder/head to isolate classifier parity.
enum DeviceNeuralServingCheck {
    struct Plan: Decodable { let source: String, projects: [String] }
    static func run(documents: URL, cache: URL, progress: @escaping @Sendable (Double, String) -> Void) async throws -> String {
        try NeuralServingRuntime.checkGolden()
        let plan = try JSONDecoder().decode(Plan.self, from: Data(contentsOf: documents.appendingPathComponent("serving-validation.json")))
        func input(_ name: String) throws -> URL {
            guard !name.isEmpty, name == URL(fileURLWithPath: name).lastPathComponent else { throw AnalysisError.invalid("Validation inputs must be local document names") }
            return documents.appendingPathComponent(name)
        }
        let source = try input(plan.source), media = try await MediaDecoder.describe(source)
        var receipts: [JSONValue] = []
        for (index, name) in plan.projects.enumerated() {
            var project = try ProjectArchive.importData(Data(contentsOf: input(name)), expectedFeatureNames: FeatureSchema.base)
            project.feedback?["source"]?["generateSideSwitchMarkers"] = .bool(false)
            let model = ProjectArchive.rallyModel(project.feedback), began = Date()
            guard model.isNeural, let feedback = project.feedback,
                  let saved = feedback["initialInference"]?["servingSide"],
                  let rows = saved["candidates"]?.array, let raw = saved["features"]?["values"],
                  let region = feedback["source"]?["featureRoi"] else { throw AnalysisError.invalid("Validation requires neural feedback with serving features") }
            let roi = AnalysisRegion(x: region["x"]?.double ?? 0, y: region["y"]?.double ?? 0,
                width: region["width"]?.double ?? 1, height: region["height"]?.double ?? 1)
            let pipeline = try await AnalysisPipeline.analyze(url: source, roi: roi, start: 0, end: min(30, media.duration),
                cacheFolder: cache.appendingPathComponent("serving-pipeline-check"), rallyModel: model,
                prepareScore: true, generateSideSwitchMarkers: false, progress: progress)
            guard pipeline.scoreError == nil, pipeline.servingSide?.modelId == "serving-side-centered-\(model.assetDirectory!)-v1",
                  pipeline.servingSide?.candidates.count == pipeline.intervals.count else {
                throw AnalysisError.invalid("Fresh neural analysis did not run its paired serving-side head")
            }
            let native = try await ScoreAnalysis.prepare(project: project, source: source, cacheFolder: cache) { fraction, detail in
                progress((Double(index) + fraction * 0.9) / Double(plan.projects.count), detail)
            }
            var original = ServingSideOutput(candidates: try rows.map { row in
                guard let id = row["id"]?.string, let anchor = row["anchor"]?.double, let end = row["interval"]?["end"]?.double,
                      let side = row["side"]?.string.flatMap(ServingSide.init(rawValue:)),
                      let verdict = row["verdict"]?.string.flatMap(ServingSideVerdict.init(rawValue:)),
                      let decision = row["serveDecisionSource"]?.string.flatMap(ServingSideDecisionSource.init(rawValue:)) else {
                    throw AnalysisError.invalid("Invalid validation serving candidate")
                }
                return ServingSideCandidate(id: id, anchor: anchor, intervalEnd: end, agreement: row["interval"]?["agreement"]?.string,
                    nearProbability: row["nearProbability"]?.double ?? 0.5, side: side, verdict: verdict, serveDecisionSource: decision,
                    reviewReasons: row["reviewReasons"]?.array?.compactMap { $0.string.flatMap(ServingSideReviewReason.init(rawValue:)) } ?? [],
                    allLabelsV2Evidence: row["serveEvidence"]?["allLabelsV2"], previousProductionEvidence: row["serveEvidence"]?["previousProduction"])
            }, rawFeatures: try FeedbackNumericArray.decode(raw, shape: [rows.count, 237], type: "float64"))
            original = try await NeuralServingRuntime.evaluate(original, model: model, url: source, media: media, roi: roi,
                cacheFolder: cache) { _, detail in progress(Double(index + 1) / Double(plan.projects.count), detail) }
            receipts.append(.object(["modelId": .string(model.modelId), "seconds": .number(Date().timeIntervalSince(began)),
                "pipelineCandidates": .number(Double(pipeline.intervals.count)),
                "native": try ProjectSession.servingFeedback(native.servingSide), "replay": try ProjectSession.servingFeedback(original)]))
            try JSONEncoder().encode(JSONValue.array(receipts)).write(to: documents.appendingPathComponent("serving-validation-output.json"), options: .atomic)
        }
        return "PASS: \(receipts.count) paired serving models; native extraction and saved-feature replay ready"
    }
}
#endif
