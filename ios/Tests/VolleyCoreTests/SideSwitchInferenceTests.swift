import Foundation
import XCTest
@testable import VolleyCore

final class SideSwitchInferenceTests: XCTestCase {
    private func asset() throws -> Data {
        let root = ProcessInfo.processInfo.environment["VOLLEY_FIXTURES"].map { URL(fileURLWithPath: $0) }
            ?? URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent().appendingPathComponent("Fixtures")
        return try Data(contentsOf: root.appendingPathComponent(SideSwitchInference.runtimeAsset))
    }
    private func input(peaks: Bool = false) -> SideSwitchAnalysisInput {
        let times = (0...30).map(Double.init), rally = [Float](repeating: 0.4, count: 31)
        var dead = [Float](repeating: 0.8, count: 31)
        if peaks { dead[5] = 0.99; dead[6] = 1; dead[25] = 0.99 }
        let ranges = [Interval(start: 0, end: 10, confidence: 0.9, agreement: ProductionEnsemble.bothModels),
                      Interval(start: 20, end: 30, confidence: 0.9, agreement: ProductionEnsemble.bothModels)]
        return .init(intervals: ranges, allLabelsV2Components: ranges, previousProductionComponents: ranges,
            allLabelsV2State: .init(modelId: "model-1ca43e38eefc", times: times, rallyProbabilities: rally, deadStateProbabilities: dead),
            previousProductionState: .init(modelId: "model-9c92b8e9333f", times: times, rallyProbabilities: rally, deadStateProbabilities: dead))
    }
    func testFrozenRuntimeClassifierAndSchema() throws {
        let data = try asset(), model = try SideSwitchInference(data: data)
        let raw = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let classifier = try XCTUnwrap(raw["classifier"] as? [String: Any])
        XCTAssertEqual(classifier["featureNames"] as? [String], SideSwitchInference.featureNames)
        XCTAssertEqual(try XCTUnwrap(model.predict(features: [Double](repeating: 0, count: 34)).first), 0.0002845953292207428, accuracy: 1e-16)
        XCTAssertEqual(try XCTUnwrap(model.predict(features: [Double](repeating: 0.5, count: 34)).first), 0.9966496738167576, accuracy: 1e-15)
        var modified = raw; modified["fingerprint"] = "changed"
        XCTAssertThrowsError(try SideSwitchInference(data: JSONSerialization.data(withJSONObject: modified)))
        XCTAssertThrowsError(try model.predict(features: [0]))
    }
    func testBoundaryAndScoreRankedInternalPeaksAndSampling() throws {
        let model = try SideSwitchInference(data: asset()), input = input(peaks: true)
        let plan = try model.framePlan(input: input, duration: 30)
        XCTAssertEqual(plan.candidates.map(\.transitionTime), [6, 15, 25])
        XCTAssertEqual(plan.candidates.map(\.kind), [.internalDeadStatePeak, .adjacentRallyBoundary, .internalDeadStatePeak])
        XCTAssertEqual(plan.candidates[0].id, "switch:internal-dead-peak:R001:6000")
        XCTAssertEqual(plan.candidates[1].id, "switch:boundary:R001:R002")
        XCTAssertEqual(plan.candidates[0].beforeStart, 2); XCTAssertEqual(plan.candidates[0].afterEnd, 10)
        XCTAssertEqual(plan.calibrationTimes.count, 14)
        XCTAssertEqual(plan.requestedTimes.count, Set(plan.requestedTimes).count)
        XCTAssertTrue(plan.requestedTimes.allSatisfy { $0 >= 0 && $0 <= 29.99 })
        let expected = [0.2, 1.8083333333333333, 3.4166666666666665, 5.025, 6.633333333333333, 8.241666666666667, 9.85]
        for (actual, expected) in zip(SideSwitchInference.sampleTimes(start: 0, end: 10), expected) { XCTAssertEqual(actual, expected, accuracy: 1e-12) }
        XCTAssertEqual(try JSONDecoder().decode(SideSwitchInference.FramePlan.self, from: JSONEncoder().encode(plan)), plan)
    }
    func testStateFeaturesAndChronologicalOrdinalSuppression() throws {
        let model = try SideSwitchInference(data: asset()), input = input()
        let boundary = try XCTUnwrap(model.generateCandidates(input: input).first)
        let state = try SideSwitchInference.stateFeatures(candidate: boundary, input: input)
        let expected = [2.0, 2, 2, 0.4, 0, 0.4, 0.4, 0.8, 0.8, 10]
        for (actual, expected) in zip(state, expected) { XCTAssertEqual(actual, expected, accuracy: 1e-7) }
        let proposals = (0..<4).map { i in var c = boundary; c.id = "switch-\(i)"; c.transitionTime = Double(i); return c }
        XCTAssertEqual(try model.decode(candidates: proposals, probabilities: [0.9, 0.89, 0.88, 0.87]), [0, 2])
        XCTAssertThrowsError(try model.decode(candidates: proposals, probabilities: [0.9]))
        XCTAssertThrowsError(try model.decode(candidates: proposals, probabilities: [.nan, 0, 0, 0]))
        let many = (0..<20).map { i in var c = boundary; c.id = String(format: "switch-%02d", i); c.transitionTime = Double(i); return c }
        XCTAssertEqual(try model.decode(candidates: many, probabilities: [Double](repeating: 0.45, count: 20)), [0, 2, 4, 6, 8, 10])
        XCTAssertEqual(try model.decode(candidates: [boundary], probabilities: [0.39884973953581804]), [0])
    }
    func testCompleteFeatureAssemblyPreservesAllRowsIncludingRejectedMarkers() throws {
        let model = try SideSwitchInference(data: asset()), input = input(peaks: true)
        let plan = try model.framePlan(input: input, duration: 30)
        let visual = [Double](repeating: 0, count: plan.candidates.count * 22)
        let output = try model.evaluate(input: input, plan: plan, visualFeatures: visual)
        XCTAssertEqual(output.rows, 3)
        XCTAssertEqual(output.features.count, 102)
        XCTAssertEqual(output.features[32], 1)
        XCTAssertEqual(output.features[33], 1)
        XCTAssertEqual(output.features[34 + 32], 0)
        XCTAssertEqual(output.features[34 + 33], 0)
        XCTAssertEqual(output.modelFingerprint, "sha256:c2570481c30dec62f56ac3284cd4028763ffd8e72a9ad07e9908fec10df387e3")
        for candidate in output.candidates { XCTAssertTrue(plan.candidates.contains { $0.id == candidate.id && $0.transitionTime == candidate.timestamp }) }
        XCTAssertThrowsError(try model.evaluate(input: input, plan: plan, visualFeatures: [Double](repeating: .nan, count: visual.count)))
        var wrong = input.previousProductionState; wrong.times[0] = 0.01
        let incompatible = SideSwitchAnalysisInput(intervals: input.intervals, allLabelsV2Components: input.allLabelsV2Components,
            previousProductionComponents: input.previousProductionComponents, allLabelsV2State: input.allLabelsV2State, previousProductionState: wrong)
        XCTAssertThrowsError(try model.framePlan(input: incompatible, duration: 30))
    }
}
