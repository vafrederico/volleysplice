import Foundation
import XCTest
@testable import VolleySplice

final class NeuralWindowBoundsIntegrationTests: XCTestCase {
    func testRoundedOlderQueueEndIsClampedBeforeActualNeuralInference() async throws {
        let source = try XCTUnwrap(Bundle.main.url(forResource: "ios-editor-fixture", withExtension: "mp4"))
        let media = try await MediaDecoder.describe(source)
        let cache = FileManager.default.temporaryDirectory.appendingPathComponent("window-bounds-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: cache) }
        let start = max(0, media.duration - 0.5)
        let result = try await AnalysisPipeline.analyze(url: source, roi: .fullFrame,
            start: start, end: media.duration + 0.001, cacheFolder: cache, rallyModel: .maximumCoverage,
            prepareScore: false, generateSideSwitchMarkers: false, progress: { _, _ in })
        XCTAssertLessThanOrEqual(result.end, media.duration)
        XCTAssertEqual(result.end, Double(try AnalysisWindowBounds.maximumEndMilliseconds(duration: media.duration)) / 1000)
        XCTAssertTrue(result.times.allSatisfy { $0 < result.end })
        XCTAssertEqual(result.neuralScores?.times, result.times)
        let project = try ProjectSession.create(sourceURL: source, result: result)
        XCTAssertNoThrow(try ProjectArchive.importBundle(XCTUnwrap(project.feedback)))
    }

    func testProjectCreationClipsRoundedCoreAndGameEndToFractionalMediaDuration() throws {
        let source = try XCTUnwrap(Bundle.main.url(forResource: "ios-editor-fixture", withExtension: "mp4"))
        let times = (0..<16).map { Double($0) / 4 }, zeros = [Float](repeating: 0, count: 16)
        let evidence = ModelRunner.RunResult(intervals: [], rallyProbabilities: zeros, serveProbabilities: zeros,
            deadStateProbabilities: zeros, serveDetections: [], profileMilliseconds: [:])
        // Deliberately model an older result/queued end rounded above its raw duration.
        // The media payload is synthetic; this tests archive coordinates, not predictions.
        let result = AnalysisResult(rallyModel: .maximumCoverage,
            neuralScores: try NeuralRallyScores(modelId: RallyModel.maximumCoverage.modelId, times: times,
                probabilities: [Float](repeating: 0.5, count: 64)),
            sourceName: nil, sourceSHA256: nil, modelSHA256: nil, decoderSchema: nil,
            media: MediaDescription(duration: 4.0006, width: 320, height: 240, rotation: 0, audio: false,
                videoCodec: "video/avc", audioCodec: nil), roi: .fullFrame, start: 0, end: 4.001, cacheIdentity: "synthetic-window",
            times: times, decodedTimes: times, base: [Float](repeating: 0, count: 16 * 104),
            contextual: [Float](repeating: 0, count: 16 * 520), allLabels: evidence, previous: evidence,
            intervals: [Interval(start: 3.25, end: 4.0006, confidence: 0.9, agreement: "neural")],
            suppression: nil, servingSide: nil, sideSwitch: nil, scoreError: nil, stageSeconds: [:], cacheHit: false)
        let project = try ProjectSession.create(sourceURL: source, result: result)
        XCTAssertEqual(project.gameWindow.endMs, 4000)
        XCTAssertEqual(project.draft.cuts.last?.coreEndMs, 4000)
        let feedback = try XCTUnwrap(project.feedback)
        XCTAssertEqual(feedback["source"]?["media"]?["duration"]?.double, 4.0006)
        XCTAssertEqual(feedback["source"]?["gameWindow"]?["end"]?.double, 4)
        let restored = try ProjectArchive.importBundle(feedback)
        XCTAssertEqual(restored.gameWindow, project.gameWindow)
        XCTAssertEqual(try ProjectArchive.retainedNeuralScores(feedback), result.neuralScores)
    }
}
