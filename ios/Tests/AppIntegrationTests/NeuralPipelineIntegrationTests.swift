import Foundation
import XCTest
@testable import VolleySplice

/// Bounded software-decoder/CPU inference qualification. These small synthetic
/// clips validate contracts and cache reuse, not real-device GPU throughput.
final class NeuralPipelineIntegrationTests: XCTestCase {
    private func source() throws -> URL {
        try XCTUnwrap(Bundle.main.url(forResource: "ios-editor-fixture", withExtension: "mp4"))
    }

    private func temporaryRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("neural-pipeline-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func run(_ source: URL, model: RallyModel, start: Double = 0, end: Double, cache: URL) async throws -> AnalysisResult {
        let began = Date()
        let result = try await AnalysisPipeline.analyze(url: source, roi: .fullFrame, start: start, end: end,
                                                       cacheFolder: cache, rallyModel: model, prepareScore: false,
                                                       generateSideSwitchMarkers: false, progress: { _, _ in })
        let receipt: [String: Any] = ["fixture": "synthetic-editor", "modelId": model.rawValue,
            "windowSeconds": end - start, "elapsedSeconds": Date().timeIntervalSince(began),
            "stageSeconds": result.stageSeconds, "rallies": result.intervals.count,
            "analysisRows": result.times.count, "avCacheHit": result.cacheHit,
            "scope": "CPU simulator correctness; score specialists disabled"]
        let bytes = try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys])
        print("NEURAL_PIPELINE_BENCHMARK " + String(decoding: bytes, as: UTF8.self))
        return result
    }

    private func validate(_ result: AnalysisResult, model: RallyModel, start: Double = 0, end: Double,
                          file: StaticString = #filePath, line: UInt = #line) throws {
        let expectedTimes = try VideoFrameSelection.targets(start: start, end: end, fps: 4)
        XCTAssertEqual(result.rallyModel, model, file: file, line: line)
        XCTAssertEqual(result.roi, .fullFrame, file: file, line: line)
        XCTAssertEqual(result.times, expectedTimes, file: file, line: line)
        XCTAssertEqual(result.decodedTimes.count, expectedTimes.count, file: file, line: line)
        XCTAssertTrue(result.decodedTimes.allSatisfy { $0.isFinite && $0 < end }, file: file, line: line)
        XCTAssertEqual(result.base.count, expectedTimes.count * 104, file: file, line: line)
        XCTAssertEqual(result.contextual.count, expectedTimes.count * 520, file: file, line: line)
        XCTAssertTrue(result.base.allSatisfy(\.isFinite), file: file, line: line)
        XCTAssertTrue(result.contextual.allSatisfy(\.isFinite), file: file, line: line)
        XCTAssertTrue(result.intervals.allSatisfy { $0.start >= start && $0.end <= end && $0.end > $0.start }, file: file, line: line)
        if model.isNeural {
            let scores = try XCTUnwrap(result.neuralScores, file: file, line: line)
            XCTAssertEqual(scores.modelId, model.rawValue, file: file, line: line)
            XCTAssertEqual(scores.times, expectedTimes, file: file, line: line)
            XCTAssertEqual(scores.probabilities.count, expectedTimes.count * 4, file: file, line: line)
            XCTAssertTrue(scores.probabilities.allSatisfy { $0.isFinite && (0...1).contains($0) }, file: file, line: line)
            XCTAssertNil(result.suppression, file: file, line: line)
        } else {
            XCTAssertNil(result.neuralScores, file: file, line: line)
            XCTAssertNotNil(result.suppression, file: file, line: line)
        }
        XCTAssertNil(result.servingSide, file: file, line: line)
        XCTAssertNil(result.sideSwitch, file: file, line: line)
        XCTAssertNil(result.scoreError, file: file, line: line)
    }

    func testAllChoicesSharedDecodeAndCachedAVProduceIdenticalNeuralResults() async throws {
        let url = try source(), media = try await MediaDecoder.describe(url)
        let end = min(4, media.duration), root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        // Prepare AV only once. Neural runs in this folder must use the separate
        // embeddings-only pass, while each cold folder must share its AV pass.
        let reused = root.appendingPathComponent("reused-av")
        let legacy = try await run(url, model: .legacy, end: end, cache: reused)
        try validate(legacy, model: .legacy, end: end)
        XCTAssertFalse(legacy.cacheHit)
        for model in [RallyModel.maximumCoverage, .balanced] {
            let cold = try await run(url, model: model, end: end, cache: root.appendingPathComponent(model.rawValue))
            try validate(cold, model: model, end: end)
            XCTAssertFalse(cold.cacheHit)
            XCTAssertNotNil(cold.stageSeconds["video"])
            XCTAssertNil(cold.stageSeconds["embeddingVideoPass"], "Cold analysis must share its video decode")
            let fromAV = try await run(url, model: model, end: end, cache: reused)
            try validate(fromAV, model: model, end: end)
            XCTAssertTrue(fromAV.cacheHit)
            XCTAssertNil(fromAV.stageSeconds["video"])
            XCTAssertNotNil(fromAV.stageSeconds["embeddingVideoPass"])
            XCTAssertEqual(fromAV.base, cold.base)
            XCTAssertEqual(fromAV.neuralScores, cold.neuralScores)
            XCTAssertEqual(fromAV.intervals, cold.intervals)
            let cached = try await run(url, model: model, end: end, cache: reused)
            XCTAssertTrue(cached.cacheHit)
            XCTAssertNil(cached.stageSeconds["embeddingVideoPass"], "Matched encoder cache must skip decoding")
            XCTAssertEqual(cached.neuralScores, fromAV.neuralScores)
            XCTAssertEqual(cached.intervals, fromAV.intervals)
        }
    }

    func testFractionalWindowKeepsSourceTicksAndCancellationDoesNotPublishResults() async throws {
        let url = try source(), media = try await MediaDecoder.describe(url)
        let root = try temporaryRoot(), start = 0.26, end = min(2.1, media.duration)
        defer { try? FileManager.default.removeItem(at: root) }
        let result = try await run(url, model: .maximumCoverage, start: start, end: end, cache: root.appendingPathComponent("fractional"))
        try validate(result, model: .maximumCoverage, start: start, end: end)
        XCTAssertEqual(result.times.first, 0.5)
        let cancelled = Task {
            try await AnalysisPipeline.analyze(url: url, roi: .fullFrame, start: 0, end: end,
                                              cacheFolder: root.appendingPathComponent("cancelled"),
                                              rallyModel: .maximumCoverage, prepareScore: false, progress: { _, _ in })
        }
        cancelled.cancel()
        do { _ = try await cancelled.value; XCTFail("Cancelled analysis published a result") }
        catch is CancellationError { }
    }

    /// The lab runner resolves this source through its external ledger. The test
    /// deliberately emits no source path, filename, fingerprint or recording ID.
    func testOptionalVideoBenchmark() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let path = environment["VOLLEYCUT_IOS_BENCHMARK_SOURCE"], !path.isEmpty else {
            throw XCTSkip("External indexed video benchmark is not configured")
        }
        guard FileManager.default.fileExists(atPath: path) else {
            XCTFail("Configured external benchmark source is unavailable")
            return
        }
        let configuredSeconds = Double(environment["VOLLEYCUT_IOS_BENCHMARK_SECONDS"] ?? "30") ?? 30
        guard configuredSeconds.isFinite, configuredSeconds > 0 else {
            XCTFail("Benchmark duration must be finite and positive")
            return
        }
        let url = URL(fileURLWithPath: path), media = try await MediaDecoder.describe(url)
        let end = min(120, configuredSeconds, media.duration), root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let expectedTimes = try VideoFrameSelection.targets(start: 0, end: end, fps: 4)
        for model in [RallyModel.legacy, .maximumCoverage, .balanced] {
            let cache = root.appendingPathComponent(model.rawValue), began = Date()
            let result = try await AnalysisPipeline.analyze(url: url, roi: .fullFrame, start: 0, end: end,
                cacheFolder: cache, rallyModel: model, prepareScore: true, generateSideSwitchMarkers: true,
                progress: { _, _ in })
            let readySeconds = Date().timeIntervalSince(began)
            XCTAssertEqual(result.rallyModel, model)
            XCTAssertFalse(result.cacheHit)
            XCTAssertEqual(result.roi, .fullFrame)
            XCTAssertEqual(result.times, expectedTimes)
            XCTAssertEqual(result.base.count, expectedTimes.count * 104)
            XCTAssertEqual(result.contextual.count, expectedTimes.count * 520)
            XCTAssertTrue(result.base.allSatisfy(\.isFinite))
            XCTAssertTrue(result.contextual.allSatisfy(\.isFinite))
            XCTAssertNil(result.scoreError)
            let serving = try XCTUnwrap(result.servingSide)
            let servingPlan = try ServingSideInference.framePlan(ranges: result.intervals, duration: media.duration)
            XCTAssertEqual(serving.candidates.count, servingPlan.candidates.count)
            XCTAssertEqual(serving.candidates.map(\.id), servingPlan.candidates.map(\.id))
            XCTAssertTrue(serving.candidates.allSatisfy { $0.nearProbability.isFinite && (0...1).contains($0.nearProbability) })
            XCTAssertTrue(serving.rawFeatures.allSatisfy(\.isFinite))
            let switches = try XCTUnwrap(result.sideSwitch)
            XCTAssertTrue(switches.features.allSatisfy(\.isFinite))
            if model.isNeural {
                let scores = try XCTUnwrap(result.neuralScores)
                XCTAssertEqual(scores.modelId, model.modelId)
                XCTAssertEqual(scores.times, expectedTimes)
                XCTAssertEqual(scores.probabilities.count, expectedTimes.count * 4)
                XCTAssertTrue(scores.probabilities.allSatisfy { $0.isFinite && (0...1).contains($0) })
                XCTAssertNil(result.suppression)
                XCTAssertFalse(serving.candidates.contains { $0.verdict == .notServe }, "Neural rallies must retain serve-review recovery")
                XCTAssertNil(result.stageSeconds["embeddingVideoPass"], "Cold analysis must share video decoding")
            } else {
                XCTAssertNil(result.neuralScores)
                XCTAssertNotNil(result.suppression)
            }
            let project = try ProjectSession.create(sourceURL: url, result: result)
            XCTAssertEqual(ProjectArchive.rallyModel(project.feedback), model)
            XCTAssertEqual(project.feedback?["initialInference"]?["modelId"]?.string, model.modelId)
            let exported = try JSONEncoder().encode(project)
            let roundTrip = try JSONDecoder().decode(ProjectDocument.self, from: exported)
            XCTAssertEqual(ProjectArchive.rallyModel(roundTrip.feedback), model)
            if model.isNeural {
                XCTAssertEqual(roundTrip.feedback?["initialInference"]?["neuralScores"]?["modelId"]?.string, model.modelId)
            }
            let cacheFiles = try FileManager.default.contentsOfDirectory(at: cache, includingPropertiesForKeys: [.fileSizeKey])
            let embeddingBytes = try cacheFiles.filter { $0.lastPathComponent.hasSuffix("-embeddings.plist") }.reduce(0) {
                $0 + (try $1.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0)
            }
            let receipt: [String: Any] = [
                "fixture": "external-indexed-fixture", "modelId": model.modelId,
                "videoSeconds": end, "allReadySeconds": readySeconds,
                "stageSeconds": result.stageSeconds, "rallies": result.intervals.count,
                "servingCandidates": serving.candidates.count, "sideSwitchCandidates": switches.candidates.count,
                "analysisRows": result.times.count, "avCacheHit": result.cacheHit,
                "baseFeatureBytes": result.base.count * MemoryLayout<Float>.size,
                "contextualFeatureBytes": result.contextual.count * MemoryLayout<Float>.size,
                "timestampBytes": (result.times.count + result.decodedTimes.count) * MemoryLayout<Double>.size,
                "servingFeatureBytes": serving.rawFeatures.count * MemoryLayout<Double>.size,
                "sideSwitchFeatureBytes": switches.features.count * MemoryLayout<Double>.size,
                "neuralScoreBytes": (result.neuralScores?.probabilities.count ?? 0) * MemoryLayout<Float>.size,
                "embeddingCacheBytes": embeddingBytes, "projectJSONBytes": exported.count,
                "scope": "CPU simulator; cold cache per model; serve-side and side-switch included"
            ]
            let bytes = try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys])
            print("NEURAL_PIPELINE_BENCHMARK " + String(decoding: bytes, as: UTF8.self))
        }
    }
}
