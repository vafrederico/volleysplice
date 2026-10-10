import XCTest
@testable import VolleyCore

final class SpecialistFrameSelectionTests: XCTestCase {
    func testIsolatedTerminalTargetIncludesLastFrameInReaderWindow() throws {
        var selection = try SpecialistFrameSelection(targets: [99.99], duration: 100)
        XCTAssertEqual(selection.readStart, 98.99, accuracy: 1e-9)
        XCTAssertNil(selection.accept(presentationTime: 99.966667))
        XCTAssertEqual(selection.finish(completed: true), 0..<1)
        XCTAssertTrue(selection.isComplete)
    }

    func testLastImageCanSatisfyBothPreviousAndTerminalRequest() throws {
        var selection = try SpecialistFrameSelection(targets: [99.96, 99.99], duration: 100)
        XCTAssertEqual(selection.accept(presentationTime: 99.966667), 0..<1)
        XCTAssertEqual(selection.finish(completed: true), 1..<2)
        XCTAssertEqual(selection.terminalPTS, 99.966667)
    }

    func testNormalTargetsKeepFirstPTSPlusOneMillisecondRule() throws {
        var selection = try SpecialistFrameSelection(targets: [5, 5.001, 5.03], duration: 100)
        XCTAssertNil(selection.accept(presentationTime: 4.998))
        XCTAssertEqual(selection.accept(presentationTime: 5), 0..<2)
        XCTAssertEqual(selection.accept(presentationTime: 5.033333), 2..<3)
        XCTAssertNil(selection.finish(completed: true))
    }

    func testFailureOrCancellationCannotUseTerminalFallback() throws {
        var selection = try SpecialistFrameSelection(targets: [99.99], duration: 100)
        XCTAssertNil(selection.accept(presentationTime: 99.966667))
        XCTAssertNil(selection.finish(completed: false))
        XCTAssertFalse(selection.isComplete)
    }

    func testTruncatedOrEmptyMiddleSegmentCannotRepeatAFrame() throws {
        var selection = try SpecialistFrameSelection(targets: [50], duration: 100)
        XCTAssertNil(selection.accept(presentationTime: 49.966667))
        XCTAssertNil(selection.finish(completed: true))
        var empty = try SpecialistFrameSelection(targets: [99.99], duration: 100)
        XCTAssertNil(empty.finish(completed: true))
        var distant = try SpecialistFrameSelection(targets: [99.99], duration: 100)
        XCTAssertNil(distant.accept(presentationTime: 98.5))
        XCTAssertNil(distant.finish(completed: true))
    }

    func testStartClampsToZeroAndInvalidTargetsFailClosed() throws {
        XCTAssertEqual(try SpecialistFrameSelection(targets: [0], duration: 1).readStart, 0)
        XCTAssertThrowsError(try SpecialistFrameSelection(targets: [1], duration: 1))
        XCTAssertThrowsError(try SpecialistFrameSelection(targets: [0.5, 0.1], duration: 1))
        XCTAssertThrowsError(try SpecialistFrameSelection(targets: [.nan], duration: 1))
    }
}
