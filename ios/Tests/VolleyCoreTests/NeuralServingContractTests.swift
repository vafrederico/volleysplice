import Foundation
import XCTest
@testable import VolleyCore

final class NeuralServingContractTests: XCTestCase {
    private func bundles() throws -> (NeuralServingBundle, NeuralServingBundle) {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Fixtures")
        let manifest = try NeuralServingManifest(data: Data(contentsOf: root.appendingPathComponent("serving-models/manifest.json")))
        let rally = try NeuralRallyManifest(data: Data(contentsOf: root.appendingPathComponent("rally-models/manifest.json")))
        return (try manifest.bundle(for: rally.bundle(for: .maximumCoverage)), try manifest.bundle(for: rally.bundle(for: .balanced)))
    }
    func testEncoderSpecificConsensusAndIndependentServeReview() throws {
        let (coverage, balanced) = try bundles()
        for (model, bundle) in [(RallyModel.maximumCoverage, coverage), (.balanced, balanced)] {
            XCTAssertEqual(NeuralServingManifest.identity(for: model).id, bundle.id)
            XCTAssertEqual(NeuralServingManifest.identity(for: model).fingerprint, bundle.fingerprint)
        }
        func logits(_ far: Double, _ near: Double) -> [Float] {
            Array(repeating: [Float(log(far / (1 - far))), Float(log(near / (1 - near)))], count: 3).flatMap { $0 }
        }
        let marginal = logits(0.01, 0.46)
        XCTAssertTrue(try NeuralServingContract.decision(logits: marginal, bundle: coverage).accepted)
        XCTAssertFalse(try NeuralServingContract.decision(logits: marginal, bundle: balanced).accepted)
        let accepted = try NeuralServingContract.decision(logits: logits(0.01, 0.7), bundle: balanced)
        XCTAssertTrue(accepted.accepted)
        var disagree = logits(0.01, 0.7); disagree.replaceSubrange(4..<6, with: logits(0.7, 0.01).prefix(2))
        XCTAssertFalse(try NeuralServingContract.decision(logits: disagree, bundle: coverage).accepted)
        XCTAssertThrowsError(try NeuralServingContract.decision(logits: logits(0.9, 0.9), bundle: coverage))
        let candidate = ServingSideCandidate(id: "N001", anchor: 1, intervalEnd: 3, agreement: "neural", side: .far,
            verdict: .review, serveDecisionSource: .neuralRallyRecovery, reviewReasons: [.sideScore, .neuralRallyRecovery])
        let updated = NeuralServingContract.apply(accepted, to: candidate)
        XCTAssertEqual(updated.side, .near); XCTAssertEqual(updated.verdict, .review)
        XCTAssertEqual(updated.reviewReasons, [.neuralRallyRecovery])
        var sideOnly = candidate; sideOnly.reviewReasons = [.sideScore]
        XCTAssertEqual(NeuralServingContract.apply(accepted, to: sideOnly).verdict, .near)
        sideOnly.verdict = .notServe
        XCTAssertEqual(NeuralServingContract.apply(accepted, to: sideOnly).verdict, .notServe)
    }
    func testDeferredCandidateIdentityPreservesCorrectionsAndTombstones() throws {
        let ranges = [Interval(start: 2, end: 4, confidence: 1, agreement: "neural")]
        let candidate = try XCTUnwrap(ServingSideInference.candidates(ranges: ranges, ids: ["N001"]).first)
        let output = ServingSideOutput(candidates: [.init(id: candidate.id, anchor: candidate.start, intervalEnd: candidate.end,
            agreement: "neural", side: .far, verdict: .far)])
        var tracking = ScoreTracking(serveMarkers: [.init(id: "serve-N001", timestampMs: 2100, side: .near, origin: .model, modelSide: .far, rallyId: "N001")])
        XCTAssertEqual(try ScoreReducer.seedModelMarkers(tracking, output: output).serveMarkers.first?.side, .near)
        tracking.serveMarkers = []; tracking.removedModelMarkerIds = ["serve-N001"]
        XCTAssertTrue(try ScoreReducer.seedModelMarkers(tracking, output: output).serveMarkers.isEmpty)
        XCTAssertThrowsError(try ServingSideInference.candidates(ranges: ranges, ids: []))
    }
    func testRecordingRanksAndUnrankedQuality() throws {
        let raw = [Double](repeating: -2, count: 237) + [Double](repeating: 4, count: 237)
        let features = try NeuralServingContract.features(raw: raw, rows: 2)
        XCTAssertEqual(Array(features[0..<237]), [Float](repeating: 0, count: 237))
        XCTAssertEqual(Array(features[249..<486]), [Float](repeating: 1, count: 237))
        XCTAssertEqual(Array(features[237..<249]), [-2,-2,-2,2,-2,-2,-2,2,-2,-2,-2,2])
        XCTAssertThrowsError(try NeuralServingContract.features(raw: [.nan], rows: 1))
    }
    func testNearestGridTieAndVideoBoundaryMasksAgainstIndependentSearch() throws {
        let duration = 10.2, times = (0..<21).map { Double($0) / 2 }
        let tokens = (0..<(21 * 3840)).map { Float($0 / 3840) }
        for anchor in [0, 0.1, 0.25, 1.25, 3.13, 4.25, 7.999, 10.1] {
            let actual = try NeuralServingContract.window(times: times, tokens: tokens, anchor: anchor, duration: duration)
            let sparse = try NeuralServingContract.embeddingTimes(anchors: [anchor], duration: duration)
            let sparseTokens = sparse.flatMap { [Float](repeating: Float($0 * 2), count: 3840) }
            XCTAssertEqual(actual, try NeuralServingContract.window(times: sparse, tokens: sparseTokens, anchor: anchor, duration: duration))
            for frame in 0..<13 {
                let requested = anchor - 3 + Double(frame) / 2
                let nearest = times.min { abs($0 - requested) < abs($1 - requested) }!
                let valid = requested >= 0 && requested < duration && abs(nearest - requested) <= 0.250001
                XCTAssertEqual(actual[frame * 3843], valid ? Float(nearest * 2) : 0)
                XCTAssertEqual(actual[frame * 3843 + 3840], valid ? 1 : 0)
            }
        }
        XCTAssertThrowsError(try NeuralServingContract.embeddingTimes(anchors: [duration], duration: duration))
    }
}
