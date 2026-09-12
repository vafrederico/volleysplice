import CryptoKit
import Foundation
import XCTest
@testable import VolleyCore

final class SuppressionTests: XCTestCase {
    private var fixtures: URL {
        if let path = ProcessInfo.processInfo.environment["VOLLEY_FIXTURES"] { return URL(fileURLWithPath: path) }
        return URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Fixtures")
    }
    private func data(_ name: String) throws -> Data { try Data(contentsOf: fixtures.appendingPathComponent(name)) }
    private func hash(_ bytes: Data) -> String { SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined() }
    private func append<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        var value = value.littleEndian; withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
    }
    private struct PolicyFixture: Decodable {
        struct Components: Decodable { let allLabelsV2: [Interval], previousProduction: [Interval] }
        let duration: Double, productionComponents: Components, decodedSuppression: [Interval]
        let expectedPolicyIntervals: [String: [[Double]]]
    }
    func testSharedGoldenPolicyRangesAndAndroidLogicalIdentity() throws {
        let golden = try JSONDecoder().decode(PolicyFixture.self, from: data("suppression-policy-golden.json"))
        let analysis = try SuppressionPolicyEngine.build(allLabelsV2: golden.productionComponents.allLabelsV2,
            previousProduction: golden.productionComponents.previousProduction, probabilities: [],
            decoded: golden.decodedSuppression, duration: golden.duration)
        for policy in [SuppressionPolicyEngine.Policy.conservative, .balanced, .aggressive] {
            let actual = SuppressionPolicyEngine.active(analysis, policy: policy)
            let expected = try XCTUnwrap(golden.expectedPolicyIntervals[policy.rawValue])
            XCTAssertEqual(actual.map { [Double($0.startMs) / 1000, Double($0.endMs) / 1000] }, expected, policy.rawValue)
        }
        XCTAssertEqual(analysis.suggestions.count, 4)
        XCTAssertEqual(Set(analysis.suggestions.map(\.logicalId)), ["S-e03614a47102f9fc4ffe"])
        XCTAssertEqual(analysis.suggestions[0].fragmentId, "S-e03614a47102f9fc4ffe:10000:11000")
        XCTAssertEqual(analysis.suggestions[0].sourceProductionIds, ["previous-production:0002:10000:11000"])
        XCTAssertTrue(SuppressionPolicyEngine.active(analysis, policy: .none).isEmpty)
        XCTAssertTrue(SuppressionPolicyEngine.active(nil, policy: .aggressive).isEmpty)
        XCTAssertEqual(SuppressionPolicyEngine.Policy.fromWireName("unknown"), .none)
        XCTAssertEqual(try JSONDecoder().decode(SuppressionAnalysis.self, from: JSONEncoder().encode(analysis)), analysis)
    }
    func testTouchingProductionAgreementAndExactJoinBoundary() throws {
        // Raw-connected Android policy treats touching source components as agreement.
        let touching = try SuppressionPolicyEngine.build(allLabelsV2: [.init(start: 1, end: 2, confidence: 1)],
            previousProduction: [.init(start: 2, end: 3, confidence: 1)], probabilities: [],
            decoded: [.init(start: 0, end: 5, confidence: 1)], duration: 5)
        XCTAssertTrue(touching.suggestions.isEmpty)
        // Balanced padding makes the positive gap exactly 500 ms: it must remain separated.
        let boundary = try SuppressionPolicyEngine.build(allLabelsV2: [.init(start: 1, end: 2, confidence: 1)],
            previousProduction: [.init(start: 5.5, end: 6.5, confidence: 1)], probabilities: [],
            decoded: [.init(start: 0, end: 10, confidence: 1)], duration: 10)
        XCTAssertEqual(SuppressionPolicyEngine.active(boundary, policy: .balanced).count, 2)
        XCTAssertTrue(SuppressionPolicyEngine.active(boundary, policy: .conservative).isEmpty)
    }
    func testEligibleRallyDisabledInitiallyTouchedKeepAndExplicitVeto() throws {
        let analysis = try SuppressionPolicyEngine.build(allLabelsV2: [.init(start: 10, end: 20, confidence: 1)],
            previousProduction: [], probabilities: [], decoded: [.init(start: 13, end: 14, confidence: 1)], duration: 30)
        let region = try XCTUnwrap(analysis.suggestions.first).region
        var draft = EditorDraft(sourceRevision: "test", cuts: [.init(id: "R001", coreStartMs: 10000, coreEndMs: 20000)])
        draft.beforePaddingMs = 0; draft.afterPaddingMs = 0
        XCTAssertEqual(draft.selectedSuppressionPolicy, "aggressive")
        XCTAssertEqual(draft.suppressionInitialBehavior, "disable-initially")
        XCTAssertTrue(EditorMath.finalIntervals(draft, suppression: [region]).isEmpty)
        draft.suppressionInitialBehavior = "highlight-only"
        XCTAssertEqual(EditorMath.finalIntervals(draft, suppression: [region]).count, 1)
        draft.suppressionInitialBehavior = "disable-initially"
        draft.userTouchedCutIds.insert("R001")
        XCTAssertEqual(EditorMath.effectiveDecision(draft, suggestion: region), "keep")
        XCTAssertEqual(EditorMath.finalIntervals(draft, suppression: [region]).count, 1)
        draft.suppressionDecisionOverrides[region.logicalId] = "suppress"
        XCTAssertTrue(EditorMath.finalIntervals(draft, suppression: [region]).isEmpty)
        draft.suppressionScopeOverrides[region.logicalId] = "veto-region"
        let veto = EditorMath.finalIntervals(draft, suppression: [region])
        XCTAssertEqual(veto.map { [$0.startMs, $0.endMs] }, [[10000, 13000], [14000, 20000]])
        // The 1-second excluded veto gap must survive the normal 3-second join rule.
        draft.selectedSuppressionPolicy = "none"
        XCTAssertEqual(EditorMath.finalIntervals(draft, suppression: [region]).map { [$0.startMs, $0.endMs] }, [[10000, 20000]])
    }

    private struct BaseFixture: Decodable { let rows: Int, columns: Int, timesBytes: Int, duration: Double, names: [String] }
    func testFrozenSpecialistEveryProbabilityAndDecodedRangeOnCanonicalVideo() throws {
        let asset = try data("suppression-overlap-exclusion-retrained.json")
        XCTAssertEqual(hash(asset), SuppressionModelRunner.assetSha256)
        let runner = try SuppressionModelRunner(data: asset)
        let fixture = try JSONDecoder().decode(BaseFixture.self, from: data("golden.json"))
        let binary = try data("base.bin")
        let times: [Double] = binary.withUnsafeBytes { bytes in
            (0..<fixture.rows).map { Double(bitPattern: UInt64(littleEndian: bytes.loadUnaligned(fromByteOffset: $0 * 8, as: UInt64.self))) }
        }
        let base: [Float] = binary.withUnsafeBytes { bytes in
            (0..<(fixture.rows * fixture.columns)).map {
                Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: fixture.timesBytes + $0 * 4, as: UInt32.self)))
            }
        }
        let contextual = try FeatureMath.contextualize(times: times, base: base, names: fixture.names)
        let result = try runner.run(times: times, contextual: contextual, duration: fixture.duration)
        var probabilityBytes = Data(), intervalBytes = Data()
        for value in result.probabilities { append(value.bitPattern, to: &probabilityBytes) }
        for value in result.decodedIntervals {
            append(value.start.bitPattern, to: &intervalBytes); append(value.end.bitPattern, to: &intervalBytes)
            append(value.confidence.bitPattern, to: &intervalBytes)
        }
        // Fixed output from the existing TS scalar Float32 head/decoder using the source upper-median FPS.
        // This fixture's four-sample smoothing makes Java's division and TS multiplication identical.
        XCTAssertEqual(hash(probabilityBytes), "e1505b23794bc17d655da41d985661394bb0d7ef38104ea563c7c7762b97d4d7")
        XCTAssertEqual(result.decodedIntervals.count, 41)
        XCTAssertEqual(hash(intervalBytes), "3a6027428389b5c1624473d9c27b75c7d13254be60ca094c3248134b63e77563")
        XCTAssertTrue(try runner.run(times: [], contextual: [], duration: 0).decodedIntervals.isEmpty)
        XCTAssertThrowsError(try runner.run(times: [0], contextual: [], duration: 1))
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: asset) as? [String: Any])
        var decoder = try XCTUnwrap(json["decoder"] as? [String: Any]); decoder["enter_threshold"] = 0.7; json["decoder"] = decoder
        XCTAssertThrowsError(try SuppressionModelRunner(data: JSONSerialization.data(withJSONObject: json)))
    }
}
