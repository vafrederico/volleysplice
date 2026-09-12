import CryptoKit
import Foundation
import XCTest
@testable import VolleyCore

final class ModelRunnerTests: XCTestCase {
    private struct Golden: Decodable {
        let schemaVersion: Int, rows: Int, columns: Int, timesBytes: Int
        let duration: Double, names: [String], rallies: [Interval]
    }
    private var fixtures: URL {
        if let path = ProcessInfo.processInfo.environment["VOLLEY_FIXTURES"] { return URL(fileURLWithPath: path) }
        return URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Fixtures")
    }
    private func data(_ name: String) throws -> Data { try Data(contentsOf: fixtures.appendingPathComponent(name)) }
    private func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    private func append<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        var value = value.littleEndian
        withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
    }
    private func floatHash(_ values: [Float]) -> String {
        var bytes = Data(); for value in values { append(value.bitPattern, to: &bytes) }; return hash(bytes)
    }
    private func intervalHash(_ values: [Interval]) -> String {
        var bytes = Data()
        for value in values {
            append(value.start.bitPattern, to: &bytes); append(value.end.bitPattern, to: &bytes)
            append(value.confidence.bitPattern, to: &bytes)
        }
        return hash(bytes)
    }
    private func serveHash(_ values: [Serve]) -> String {
        var bytes = Data()
        for value in values { append(value.time.bitPattern, to: &bytes); append(value.confidence.bitPattern, to: &bytes) }
        return hash(bytes)
    }

    // Frozen checksums generated from the existing lib/on-device/model.ts Float32 implementation
    // on the canonical Y9 fixture. Hashes cover EVERY probability, boundary, confidence and serve,
    // not just summaries or a second implementation of Swift's own algorithm.
    func testCanonicalVideoBothModelsAndProductionEnsemble() throws {
        let metadata = try JSONDecoder().decode(Golden.self, from: data("golden.json"))
        XCTAssertEqual(metadata.schemaVersion, 1)
        XCTAssertEqual(metadata.rows, 3474); XCTAssertEqual(metadata.columns, 104)
        XCTAssertEqual(metadata.names, FeatureSchema.base)
        XCTAssertEqual(metadata.timesBytes, metadata.rows * 8)
        let binary = try data("base.bin")
        XCTAssertEqual(binary.count, metadata.timesBytes + metadata.rows * metadata.columns * 4)
        let times: [Double] = binary.withUnsafeBytes { bytes in
            (0..<metadata.rows).map { Double(bitPattern: UInt64(littleEndian: bytes.loadUnaligned(fromByteOffset: $0 * 8, as: UInt64.self))) }
        }
        let base: [Float] = binary.withUnsafeBytes { bytes in
            (0..<(metadata.rows * metadata.columns)).map {
                Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: metadata.timesBytes + $0 * 4, as: UInt32.self)))
            }
        }
        let contextual = try FeatureMath.contextualize(times: times, base: base, names: metadata.names)
        XCTAssertEqual(floatHash(contextual), "17e9efebfe5a8d76f9c254197d0fc0b6474fffb4931133f5088c4640e87d894f")
        let allData = try data("model-1ca43e38eefc.json"), previousData = try data("model-9c92b8e9333f.json")
        XCTAssertEqual(hash(allData), "d2c2c11e8fed8b6c6ad77d244b613e81d5bab101939a8f57be5166b45ebca78f")
        XCTAssertEqual(hash(previousData), "d8cc42f70bc10576a5e03251b05981ceeee1a61a15c61cc5dfb68dd631e6f90d")
        let all = try ModelRunner(data: allData).run(times: times, contextual: contextual, duration: metadata.duration)
        let previous = try ModelRunner(data: previousData).run(times: times, contextual: contextual, duration: metadata.duration)
        XCTAssertEqual(all.intervals.count, 42)
        for (actual, expected) in zip(all.intervals, metadata.rallies) {
            XCTAssertEqual((actual.start * 1000).rounded() / 1000, expected.start)
            XCTAssertEqual((actual.end * 1000).rounded() / 1000, expected.end)
            XCTAssertEqual(actual.confidence, expected.confidence, accuracy: 2e-5)
        }
        XCTAssertEqual(floatHash(all.rallyProbabilities), "6da53296d24fd84de1678ef283d80207467894feb5f15d8123532a07fe739626")
        XCTAssertEqual(floatHash(all.serveProbabilities), "49b0f0ef35ac342e5ab84970ef527071570a43f91514f17bc25b279c66c8019d")
        XCTAssertEqual(floatHash(all.deadStateProbabilities), "4d68cd779ddaf2e002d2feec09f9fac059cf60334709f33b293fab58a5ce3594")
        XCTAssertEqual(intervalHash(all.intervals), "a0a198cf0fb7c7240eff90c164d23326c1ca5fc48d94aedb34efb9d66ab60224")
        XCTAssertEqual(serveHash(all.serveDetections), "1f670436a2a8071cc91b53cc0db72d1be42f920718f3966bec7739382d6304b7")
        XCTAssertEqual(previous.intervals.count, 37)
        XCTAssertEqual(floatHash(previous.rallyProbabilities), "89001af0a0351139cf9e9b4ed9dd8295136b902193e076770c50968a673f3edc")
        XCTAssertEqual(floatHash(previous.serveProbabilities), "4b2a7617c78eb2df2275b21e55d6c79fbda8d141dc5e2896e034f35ad8affd8d")
        XCTAssertEqual(floatHash(previous.deadStateProbabilities), "96163c1015fc6b744b36495e236d4689494b22aa58f810d3acd98f16fc17e617")
        XCTAssertEqual(intervalHash(previous.intervals), "e57d6d716222a9495ca64bf21c90e7d1dce5de99c7549ebf500a9f1bbd576d2f")
        XCTAssertEqual(serveHash(previous.serveDetections), "c3a577310b6d1737126f07e5eca73e6f460bd44fc2fedefc4983a04c8188ea2f")
        let ensemble = ProductionEnsemble.merge(allLabelsV2: all.intervals, previousProduction: previous.intervals)
        XCTAssertEqual(ensemble.count, 40)
        XCTAssertEqual(intervalHash(ensemble), "99207598e93c7b4344c959f72174c2121edf266691dfebf7d0d5e4d10f238ba1")
        XCTAssertEqual(hash(Data(ensemble.map { $0.agreement! }.joined(separator: "\n").utf8)), "74820a558ca1e76b8f3bc359530e27bc00ae7945354254b55e3ba996617808e2")
        XCTAssertEqual(ensemble.filter { ProductionEnsemble.isDisagreement($0.agreement) }.count, 3)
        XCTAssertEqual(all.profileMilliseconds.count, 8)
    }

    func testEnsembleStrictOverlapTransitivityAndConfidenceCap() {
        let merged = ProductionEnsemble.merge(allLabelsV2: [
            Interval(start: 10, end: 15, confidence: 0.9), Interval(start: 18, end: 22, confidence: 0.8),
            Interval(start: 30, end: 35, confidence: 0.8), Interval(start: .nan, end: 100, confidence: 1)
        ], previousProduction: [
            Interval(start: 14, end: 19, confidence: 0.7), Interval(start: 35, end: 40, confidence: 0.95),
            Interval(start: 5, end: 5, confidence: 1)
        ])
        XCTAssertEqual(merged.count, 3)
        XCTAssertEqual(merged[0].start, 10); XCTAssertEqual(merged[0].end, 22)
        XCTAssertEqual(merged[0].confidence, 0.8, accuracy: 1e-6)
        XCTAssertEqual(merged.map(\.agreement), [ProductionEnsemble.bothModels, ProductionEnsemble.allLabelsV2Only, ProductionEnsemble.previousProductionOnly])
        XCTAssertEqual(merged[1].confidence, 0.48, accuracy: 1e-6)
        XCTAssertEqual(merged[2].confidence, 0.49)
        XCTAssertEqual(ProductionEnsemble.merge(allLabelsV2: [], previousProduction: []), [])
    }

    func testModelRejectsSchemaDriftAndMalformedMatrices() throws {
        let original = try data("model-1ca43e38eefc.json")
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: original) as? [String: Any])
        json["schemaVersion"] = 99
        XCTAssertThrowsError(try ModelRunner(data: JSONSerialization.data(withJSONObject: json)))
        XCTAssertThrowsError(try ModelRunner(data: original, expectedFeatureNames: Array(FeatureSchema.contextualNames.reversed())))
        let runner = try ModelRunner(data: original)
        XCTAssertThrowsError(try runner.run(times: [0], contextual: [0], duration: 1))
        XCTAssertThrowsError(try runner.run(times: [0, 0], contextual: [Float](repeating: 0, count: 1040), duration: 1))
        XCTAssertThrowsError(try runner.run(times: [], contextual: [], duration: .infinity))
        XCTAssertTrue(try runner.run(times: [], contextual: [], duration: 0).intervals.isEmpty)
    }
}
