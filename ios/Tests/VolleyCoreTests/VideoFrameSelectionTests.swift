import XCTest
@testable import VolleyCore

final class VideoFrameSelectionTests: XCTestCase {
    func testSourceGridExclusiveEndAndVariableRateEarlierTie() throws {
        XCTAssertEqual(try VideoFrameSelection.targets(start: 0.26, end: 1.01, fps: 4), [0.5, 0.75, 1])
        XCTAssertEqual(try VideoFrameSelection.targets(start: 0, end: 1, fps: 4), [0, 0.25, 0.5, 0.75])
        // The image at the exclusive end cannot satisfy the final target.
        let times = try VideoFrameSelection.select(presentationTimes: [0, 0.1, 0.4, 0.4, 0.9, 1],
                                                   targets: [0, 0.25, 0.5, 0.75, 0.99], end: 1)
        XCTAssertEqual(times, [0, 0.1, 0.4, 0.9, 0.9])
    }

    func testSparseFramesRepeatAndNoFramesFailClosed() throws {
        XCTAssertEqual(try VideoFrameSelection.select(presentationTimes: [0.12], targets: [0, 0.25, 0.5], end: 1), [0.12, 0.12, 0.12])
        XCTAssertEqual(try VideoFrameSelection.select(presentationTimes: [], targets: [0, 0.25], end: 1), [])
        XCTAssertThrowsError(try VideoFrameSelection.select(presentationTimes: [1, 0], targets: [0], end: 2))
        XCTAssertThrowsError(try VideoFrameSelection.targets(start: 0, end: .nan, fps: 4))
    }
}
