import XCTest
@testable import VolleySplice

/// Native extractor coverage, independent of inference outputs and video decoding.
final class FeatureExtractionIntegrationTests: XCTestCase {
    func testAndroidStaticGrayFixtureProducesFrozenCourtAndFlightBanks() throws {
        // Exact fixture from ServingSideFeatureExtractorInstrumentedTest:
        // 192x108 grayscale bytes of value 73, eight court and nine flight frames.
        XCTAssertEqual(ServingSideFeatureExtractor.width, 192)
        XCTAssertEqual(ServingSideFeatureExtractor.height, 108)
        XCTAssertEqual(ServingSideInference.courtFlowOffsets.count, 8)
        XCTAssertEqual(ServingSideInference.flightOffsets.count, 9)
        let frame = [UInt8](repeating: 73, count: 192 * 108)
        let courtFrames = [[UInt8]](repeating: frame, count: 8)
        let flightFrames = [[UInt8]](repeating: frame, count: 9)

        // Call the actual app extractor and linked native flow/morphology code;
        // do not call its diagnostic validator or reuse its expected-value helper.
        let court = try ServingSideFeatureExtractor.courtFlow(courtFrames)
        XCTAssertEqual(court, [Double](repeating: 0, count: 82))

        var expectedFlight = [Double](repeating: 0, count: 155)
        for phase in 0..<3 {
            expectedFlight[phase * 43 + 30] = 0.5 // centroidX in an empty motion field
            expectedFlight[phase * 43 + 31] = 0.5 // centroidY in an empty motion field
        }
        let flight = try ServingSideFeatureExtractor.flight(flightFrames)
        XCTAssertEqual(flight.count, 155)
        for (index, pair) in zip(flight, expectedFlight).enumerated() {
            XCTAssertEqual(pair.0, pair.1, accuracy: 1e-12, "Flight feature \(index)")
        }

        let combined = try ServingSideFeatureExtractor.extract(courtGray: courtFrames, flightGray: flightFrames)
        XCTAssertEqual(combined.count, 237)
        XCTAssertEqual(Array(combined.prefix(82)), [Double](repeating: 0, count: 82))
        let combinedFlight = Array(combined.dropFirst(82))
        XCTAssertEqual(combinedFlight.count, 155)
        for (index, pair) in zip(combinedFlight, expectedFlight).enumerated() {
            XCTAssertEqual(pair.0, pair.1, accuracy: 1e-12, "Combined flight feature \(index)")
        }
    }
}
