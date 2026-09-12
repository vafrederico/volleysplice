import XCTest
@testable import VolleyCore

final class FeatureMathTests: XCTestCase {
    func testTiesAbsoluteColumnsAndContextEdges() throws {
        let times = [0.0, 0.25, 0.5]
        let base: [Float] = [2, 0.1, 2, 0.2, 9, 0.3]
        XCTAssertEqual(try FeatureMath.ranks(base, rows: 3, columns: 2), [0.25, 0, 0.25, 0.5, 1, 1])
        let context = try FeatureMath.contextualize(times: times, base: base, names: ["motion", "focus_quality"])
        XCTAssertEqual(Array(context[0..<2]), [0.25, 0.1])
        XCTAssertEqual(Array(context[8..<10]), [1, 0.3])
        XCTAssertEqual(FeatureMath.nearest(times, 0.125), 0)
    }

    func testRejectsInvalidMatrices() {
        XCTAssertThrowsError(try FeatureMath.contextualize(times: [0, 0], base: [1, 2], names: ["motion"]))
        XCTAssertThrowsError(try FeatureMath.ranks([.nan], rows: 1, columns: 1))
    }
}
