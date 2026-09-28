import Foundation
import XCTest
@testable import VolleyCore

final class NeuralWindowBoundsTests: XCTestCase {
    func testFullDurationAndOlderRoundedQueueEndResolveToSameRepresentableWindow() throws {
        for duration in [10.0004, 10.0006, 10.001, 10.0016, 10.9996] {
            let roundedQueueEnd = (duration * 1000).rounded() / 1000
            let current = try AnalysisWindowBounds.normalize(start: 0, end: duration, duration: duration)
            let restored = try AnalysisWindowBounds.normalize(start: 0, end: roundedQueueEnd, duration: duration)
            XCTAssertEqual(current, restored)
            let end = Double(current.endMs) / 1000
            XCTAssertLessThanOrEqual(end, duration)
            XCTAssertLessThan(duration - end, 0.001)
            XCTAssertEqual(try AnalysisWindowBounds.normalize(start: 0, end: end, duration: duration), current)
            XCTAssertNoThrow(try NeuralRallyContract.embeddingTimes(duration: duration, start: 0, end: end))
        }
        XCTAssertEqual(try AnalysisWindowBounds.maximumEndMilliseconds(duration: 10.001), 10001)
        XCTAssertEqual(try AnalysisWindowBounds.normalize(start: 0, end: 10.001, duration: 10.0006).endMs, 10000)
    }

    func testBoundsKeepFractionalStartRoundTripAndRejectGenuinelyInvalidWindows() throws {
        let bounds = try AnalysisWindowBounds.normalize(start: 0.2601, end: 2.1001, duration: 3)
        XCTAssertEqual(bounds, TimeRange(startMs: 260, endMs: 2100))
        XCTAssertEqual(try VideoFrameSelection.targets(start: Double(bounds.startMs) / 1000, end: Double(bounds.endMs) / 1000, fps: 4),
                       [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
        XCTAssertThrowsError(try AnalysisWindowBounds.normalize(start: 0, end: 10.002, duration: 10))
        XCTAssertThrowsError(try AnalysisWindowBounds.normalize(start: -0.001, end: 1, duration: 1))
        XCTAssertThrowsError(try AnalysisWindowBounds.normalize(start: 0.9998, end: 1.0001, duration: 1.0001))
        XCTAssertThrowsError(try AnalysisWindowBounds.normalize(start: 0, end: .infinity, duration: 10))
        XCTAssertThrowsError(try AnalysisWindowBounds.maximumEndMilliseconds(duration: .nan))
    }

    func testArchiveDefaultGameWindowNeverExceedsRawFractionalMediaDuration() throws {
        let source = ProjectArchive.source(projectId: "synthetic", name: "synthetic.mp4", sizeBytes: 0,
            lastModifiedMs: 0, duration: 10.0006, width: 1920, height: 1080)
        XCTAssertEqual(source["gameWindow"]?["end"]?.double, 10)
        XCTAssertEqual(source["media"]?["duration"]?.double, 10.0006)
    }
}
