import Foundation
import XCTest
@testable import VolleyCore

final class ServingSideInferenceTests: XCTestCase {
    private func asset() throws -> Data {
        let root = ProcessInfo.processInfo.environment["VOLLEY_FIXTURES"].map { URL(fileURLWithPath: $0) }
            ?? URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent().appendingPathComponent("Fixtures")
        return try Data(contentsOf: root.appendingPathComponent(ServingSideInference.runtimeAsset))
    }
    func testFrozenFeatureSchemaModelAndCRLFChecksum() throws {
        let data = try asset(), model = try ServingSideInference(data: data)
        XCTAssertEqual(ServingSideInference.courtFeatureNames.count, 82)
        XCTAssertEqual(ServingSideInference.flightFeatureNames.count, 155)
        XCTAssertEqual(ServingSideInference.featureNames.count, 237)
        let raw = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(raw["featureNames"] as? [String], ServingSideInference.featureNames)
        XCTAssertEqual(try model.nearProbability(ranked: [Double](repeating: 0.5, count: 237)), 0.5428339645988896, accuracy: 1e-14)
        let text = try XCTUnwrap(String(data: data, encoding: .utf8)).replacingOccurrences(of: "\r\n", with: "\n")
        XCTAssertNoThrow(try ServingSideInference(data: Data(text.replacingOccurrences(of: "\n", with: "\r\n").utf8)))
        var modified = raw; modified["featureVersion"] = "wrong"
        XCTAssertThrowsError(try ServingSideInference(data: JSONSerialization.data(withJSONObject: modified)))
    }
    func testDoubleRanksPreserveSubFloatDifferencesSingletonAndRejectNonfinite() throws {
        XCTAssertEqual(try ServingSideInference.tiedPercentileRanks([2, 1, 2, 3, 4, 2], rows: 3, columns: 2), [0.25, 0, 0.25, 1, 1, 0.5])
        XCTAssertEqual(try ServingSideInference.tiedPercentileRanks([8, 9, 10], rows: 1, columns: 3), [0.5, 0.5, 0.5])
        XCTAssertEqual(try ServingSideInference.tiedPercentileRanks([1, 1 + 1e-10, 1 + 2e-10], rows: 3, columns: 1), [0, 0.5, 1])
        XCTAssertThrowsError(try ServingSideInference.tiedPercentileRanks([.nan], rows: 1, columns: 1))
        XCTAssertThrowsError(try ServingSideInference.tiedPercentileRanks([1, 2], rows: 2, columns: 2))
    }
    func testEvidenceInclusiveBoundaryFirstPeakNearestFallback() throws {
        let output = ProductionServeOutput(modelId: "model", times: [3, 4, 5, 7], probabilities: [0.9, 0.2, 0.9, 1],
            detections: [.init(time: 4.4, confidence: 0.91), .init(time: 6, confidence: 0.99)])
        let evidence = try ServingSideInference.serveHeadEvidence(output, anchor: 4)
        XCTAssertEqual(evidence.peakTime, 3)
        XCTAssertTrue(evidence.crossesThreshold)
        XCTAssertEqual(evidence.nearestDetection?.time, 4.4)
        let fallback = try ServingSideInference.serveHeadEvidence(output, anchor: 10, windowSeconds: 0.1)
        XCTAssertEqual(fallback.peakTime, 7)
        let tiedNearest = try ServingSideInference.serveHeadEvidence(output, anchor: 6, windowSeconds: 0)
        XCTAssertEqual(tiedNearest.peakTime, 5)
        XCTAssertThrowsError(try ServingSideInference.serveHeadEvidence(.init(modelId: "x", times: [], probabilities: []), anchor: 1))
    }
    func testHybridGateReviewBoundariesAndReasonOrder() throws {
        let high = try ServingSideInference.serveHeadEvidence(.init(modelId: "high", times: [1], probabilities: [0.9]), anchor: 1)
        let low = try ServingSideInference.serveHeadEvidence(.init(modelId: "low", times: [1], probabilities: [0.2]), anchor: 1)
        let agreed = ServingSideInference.CandidateInterval(id: "R001", start: 1, end: 2, agreement: ProductionEnsemble.bothModels)
        let single = ServingSideInference.CandidateInterval(id: "R002", start: 1, end: 2, agreement: ProductionEnsemble.allLabelsV2Only)
        func verdict(_ p: Double, _ candidate: ServingSideInference.CandidateInterval, _ evidence: ServingSideHeadEvidence) throws -> ServingSideCandidate {
            try ServingSideInference.decision(candidate: candidate, nearProbability: p, allLabelsV2Evidence: evidence, previousProductionEvidence: low)
        }
        XCTAssertEqual(try verdict(ServingSideInference.farReviewThreshold.nextDown, single, high).verdict, .far)
        XCTAssertEqual(try verdict(ServingSideInference.farReviewThreshold, single, high).verdict, .review)
        XCTAssertEqual(try verdict(ServingSideInference.nearReviewThreshold.nextDown, single, high).verdict, .review)
        XCTAssertEqual(try verdict(ServingSideInference.nearReviewThreshold, single, high).verdict, .near)
        XCTAssertEqual(try verdict(ServingSideInference.sideThreshold, single, high).side, .near)
        let recovered = try verdict(0.4, agreed, low)
        XCTAssertEqual(recovered.serveDecisionSource, .productionRallyRecovery)
        XCTAssertEqual(recovered.verdict, .review)
        XCTAssertEqual(recovered.side, .far)
        XCTAssertEqual(recovered.reviewReasons, [.sideScore, .productionRallyRecovery])
        let rejected = try verdict(0.4, single, low)
        XCTAssertEqual(rejected.verdict, .notServe)
        XCTAssertEqual(rejected.side, .far)
        XCTAssertEqual(rejected.reviewReasons, [.sideScore])
        XCTAssertEqual(rejected.intervalStart, single.start)
        XCTAssertEqual(rejected.intervalEnd, single.end)
        XCTAssertEqual(recovered.allLabelsV2Evidence?["peakTime"], .number(1))
    }
    func testFramePlanStableIdsMillisecondAnchorsAndSharedDeduplication() throws {
        let plan = try ServingSideInference.framePlan(ranges: [
            .init(start: 9.9004, end: 10, confidence: 1), .init(start: -1, end: 0, confidence: 1),
            .init(start: 0.1001, end: 1, confidence: 1)
        ], duration: 10)
        XCTAssertEqual(plan.candidates.map(\.id), ["R003", "R001"])
        XCTAssertEqual(plan.candidates.map(\.start), [0.1, 9.9])
        XCTAssertEqual(plan.requestedTimes.first, 0)
        XCTAssertEqual(plan.requestedTimes.last, 9.99)
        XCTAssertEqual(plan.requestedTimes.count, Set(plan.requestedTimes).count)
        XCTAssertEqual(try ServingSideInference.sharedRequestedTimestamps(serving: [1, 2], sideSwitch: [2, 3]), [1, 2, 3])
    }
    func testImportedFloat64MatrixMatchesSevenFrozenBrowserScoresAndReuseIdentity() throws {
        let model = try ServingSideInference(data: asset())
        let raw: [Double] = (0..<(7 * 237)).map { i in
            let row = i / 237, column = i % 237
            return Double((row * (column % 11 + 1) + column % 5) % 9)
        }
        var bytes = Data()
        for value in raw {
            var bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { bytes.append(contentsOf: $0) }
        }
        let ranges = (0..<7).map { Interval(start: Double($0 * 10 + 1), end: Double($0 * 10 + 5), confidence: 1) }
        let candidates = try ServingSideInference.candidates(ranges: ranges)
        let times = candidates.map(\.start)
        let all = ProductionServeOutput(modelId: "model-1ca43e38eefc", times: times, probabilities: [Float](repeating: 0.9, count: 7))
        let previous = ProductionServeOutput(modelId: "model-9c92b8e9333f", times: times, probabilities: [Float](repeating: 0.2, count: 7))
        let output = try model.evaluate(candidates: candidates, rawFloat64LE: bytes, allLabelsV2: all, previousProduction: previous)
        // Independently generated with prod/src/lib/on-device/serving-side-model.ts.
        let expected = [0.20540487942997615, 0.6839680699042503, 0.48821232093902167, 0.8725868020359305,
                        0.34505174122483795, 0.5597399936640005, 0.5761448365994998]
        for (actual, expected) in zip(output.candidates, expected) { XCTAssertEqual(actual.nearProbability, expected, accuracy: 1e-14) }
        XCTAssertEqual(output.candidates.map(\.verdict), [.far, .near, .review, .near, .review, .near, .near])
        XCTAssertEqual(output.rawFeatures, raw)
        XCTAssertTrue(ServingSideInference.isReusable(output, ranges: ranges))
        var moved = ranges; moved[0].start += 0.001
        XCTAssertFalse(ServingSideInference.isReusable(output, ranges: moved))
        var wrong = previous; wrong.times[0] += 0.01
        XCTAssertThrowsError(try model.evaluate(candidates: candidates, rawFeatures: raw, allLabelsV2: all, previousProduction: wrong))
        XCTAssertThrowsError(try model.evaluate(candidates: candidates, rawFloat64LE: bytes.dropLast(), allLabelsV2: all, previousProduction: previous))
        var malformed = raw; malformed[0] = .infinity
        XCTAssertThrowsError(try model.evaluate(candidates: candidates, rawFeatures: malformed, allLabelsV2: all, previousProduction: previous))
    }
}
