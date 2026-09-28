import Foundation
import XCTest
@testable import VolleyCore

final class NeuralAuxiliaryEvidenceTests: XCTestCase {
    private var fixtures: URL {
        if let path = ProcessInfo.processInfo.environment["VOLLEY_FIXTURES"] { return URL(fileURLWithPath: path) }
        return URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Fixtures")
    }

    func testEvidenceOnlyMatchesBothLegacyHeadsWithoutRallyDecoding() throws {
        let times = (0..<40).map { Double($0) / 4 }
        let contextual = (0..<(40 * 520)).map { Float($0 % 31) / 31 }
        for id in ProjectArchive.componentIds {
            let model = try ModelRunner(data: Data(contentsOf: fixtures.appendingPathComponent(id + ".json")))
            let full = try model.run(times: times, contextual: contextual, duration: 10)
            let evidence = try model.runEvidence(times: times, contextual: contextual, duration: 10)
            XCTAssertEqual(evidence.rallyProbabilities, full.rallyProbabilities)
            XCTAssertEqual(evidence.serveProbabilities, full.serveProbabilities)
            XCTAssertEqual(evidence.deadStateProbabilities, full.deadStateProbabilities)
            XCTAssertEqual(evidence.serveDetections, full.serveDetections)
            XCTAssertTrue(evidence.intervals.isEmpty)
            XCTAssertEqual(Set(evidence.profileMilliseconds.keys), Set(["rally_head", "serve_head", "dead_state_head", "serve_decode"]))
            XCTAssertTrue(full.profileMilliseconds.keys.contains("serve_composition"))
            XCTAssertTrue(full.profileMilliseconds.keys.contains("dead_state_refinement"))
            XCTAssertThrowsError(try model.runEvidence(times: times, contextual: Array(contextual.dropLast()), duration: 10))
            XCTAssertThrowsError(try model.runEvidence(times: times, contextual: contextual, duration: 1))
            XCTAssertTrue(try model.runEvidence(times: [], contextual: [], duration: 0).intervals.isEmpty)
        }
    }

    func testWeakServeHeadNeuralRallyRemainsVisibleForReviewAndLegacyUnchanged() throws {
        let low = try ServingSideInference.serveHeadEvidence(.init(modelId: "low", times: [1], probabilities: [0.2]), anchor: 1)
        let high = try ServingSideInference.serveHeadEvidence(.init(modelId: "high", times: [1], probabilities: [0.9]), anchor: 1)
        let neural = ServingSideInference.CandidateInterval(id: "R001", start: 1, end: 4, agreement: "neural")
        let legacy = ServingSideInference.CandidateInterval(id: "R002", start: 1, end: 4, agreement: ProductionEnsemble.allLabelsV2Only)
        func decide(_ candidate: ServingSideInference.CandidateInterval, p: Double, evidence: ServingSideHeadEvidence) throws -> ServingSideCandidate {
            try ServingSideInference.decision(candidate: candidate, nearProbability: p,
                allLabelsV2Evidence: evidence, previousProductionEvidence: low)
        }
        let recovered = try decide(neural, p: 0.4, evidence: low)
        XCTAssertEqual(recovered.verdict, .review)
        XCTAssertEqual(recovered.side, .far)
        XCTAssertEqual(recovered.serveDecisionSource, .neuralRallyRecovery)
        XCTAssertEqual(recovered.reviewReasons, [.sideScore, .neuralRallyRecovery])
        XCTAssertEqual(recovered.allLabelsV2Evidence, low.json)
        XCTAssertEqual(recovered.withNeuralRallyRecovery(), recovered)
        let certainSide = try decide(neural, p: 0.9, evidence: low)
        XCTAssertEqual(certainSide.side, .near)
        XCTAssertEqual(certainSide.verdict, .review)
        XCTAssertEqual(certainSide.reviewReasons, [.neuralRallyRecovery])
        XCTAssertEqual(try decide(neural, p: 0.9, evidence: high).verdict, .near)
        XCTAssertEqual(try decide(neural, p: 0.9, evidence: high).serveDecisionSource, .serveHead)
        let rejectedLegacy = try decide(legacy, p: 0.9, evidence: low)
        XCTAssertEqual(rejectedLegacy.verdict, .notServe)
        XCTAssertEqual(rejectedLegacy.withNeuralRallyRecovery(), rejectedLegacy)
        XCTAssertEqual(try JSONDecoder().decode(ServingSideCandidate.self, from: JSONEncoder().encode(recovered)), recovered)
    }

    func testOlderNeuralNotServeScoresRecoverDuringMarkerSeedingWithoutErasingCorrection() throws {
        let old = ServingSideCandidate(id: "R001", anchor: 1, intervalEnd: 4, agreement: "neural", nearProbability: 0.8,
            side: .near, verdict: .notServe)
        let tracking = try ScoreReducer.seedModelMarkers(ScoreTracking(), output: ServingSideOutput(candidates: [old]))
        XCTAssertEqual(tracking.serveMarkers.count, 1)
        XCTAssertEqual(tracking.serveMarkers[0].side, .review)
        XCTAssertEqual(tracking.serveMarkers[0].modelSide, .review)
        XCTAssertEqual(tracking.serveMarkers[0].rallyId, "R001")
        var corrected = tracking
        corrected.serveMarkers[0].side = .far
        let rerun = try ScoreReducer.seedModelMarkers(corrected, output: ServingSideOutput(candidates: [old]))
        XCTAssertEqual(rerun.serveMarkers[0].side, .far)
        XCTAssertEqual(rerun.serveMarkers[0].modelSide, .review)
        var legacy = old; legacy.agreement = ProductionEnsemble.allLabelsV2Only
        XCTAssertTrue(try ScoreReducer.seedModelMarkers(ScoreTracking(), output: ServingSideOutput(candidates: [legacy])).serveMarkers.isEmpty)
    }
}
