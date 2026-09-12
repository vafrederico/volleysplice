import XCTest
@testable import VolleyCore

/// Android pane bounds, with the requested iPhone landscape breakpoint extension.
final class EditorLayoutTests: XCTestCase {
    func testPhonePlayerReservesMeasuredTimelineLegendAndControls() {
        XCTAssertEqual(EditorLayout.phoneLandscapePlayerHeight(preferred: 240, paneHeight: 450, timelineHeight: 210, chromeHeight: 80), 142)
        XCTAssertEqual(EditorLayout.phoneLandscapePlayerHeight(preferred: 900, paneHeight: 450, timelineHeight: 210, chromeHeight: 80), 142)
        XCTAssertEqual(EditorLayout.phoneLandscapePlayerHeight(preferred: 140, paneHeight: 600, timelineHeight: 210, chromeHeight: 80), 140)
        XCTAssertEqual(EditorLayout.phoneLandscapePlayerHeight(preferred: 80, paneHeight: 450, timelineHeight: 210, chromeHeight: 80), 80)
    }
    func testManualPhonePlayerCanExceedFitWithoutChangingAutomaticOrIPadBounds() {
        let savedHeight = 900.0
        XCTAssertEqual(EditorLayout.phoneLandscapePlayerHeight(preferred: savedHeight, paneHeight: 450,
            timelineHeight: 210, chromeHeight: 80), 142)
        XCTAssertEqual(EditorLayout.phoneLandscapePlayerHeight(preferred: savedHeight, paneHeight: 450,
            timelineHeight: 210, chromeHeight: 80, manual: true), savedHeight)
        XCTAssertEqual(EditorLayout.phoneLandscapePlayerHeight(preferred: 5000, paneHeight: 450,
            timelineHeight: 210, chromeHeight: 80, manual: true), 1200)
        XCTAssertEqual(EditorLayout.phoneLandscapePlayerHeight(preferred: -20, paneHeight: 450,
            timelineHeight: 210, chromeHeight: 80, manual: true), 48)
        XCTAssertEqual(EditorLayout.playerHeight(140, desktop: true), 180)
        XCTAssertEqual(EditorLayout.playerHeight(savedHeight, desktop: true), savedHeight)
    }
    func testPhoneRightMinimumCountsReclaimedSpaceAtBothInterfaceScales() {
        // A 62pt inset leaves 12pt for rounded corners: reclaim 50 physical pt.
        for (scale, expectedBase) in [(0.60, 100.0), (0.75, 113.33333333333333)] {
            let reclaimed = 50 / scale
            let minimum = EditorLayout.phoneRightSidebarMinimum(reclaimedSpace: reclaimed)
            XCTAssertEqual(minimum, expectedBase, accuracy: 0.0001)
            XCTAssertEqual(EditorLayout.phoneRightSidebarMinimum(reclaimedSpace: 0), 180)
            let widths = EditorLayout.sidebarWidths(availableWidth: 640, left: 520, right: 0, rightMinimum: minimum)
            XCTAssertEqual(widths.right, minimum, accuracy: 0.0001)
            XCTAssertEqual(widths.leftMaximum, 640 - 220 - 64 - minimum, accuracy: 0.0001)
            XCTAssertGreaterThanOrEqual(widths.right + reclaimed, 180)
            XCTAssertEqual(EditorLayout.resizedSidebarWidth(current: 220, translation: 1000,
                side: .right, maximum: widths.rightMaximum, rightMinimum: minimum), widths.right, accuracy: 0.0001)
            XCTAssertEqual(EditorLayout.resizedSidebarWidth(current: 292, translation: -1000,
                side: .left, maximum: widths.leftMaximum, rightMinimum: minimum), 180)
        }
        XCTAssertEqual(EditorLayout.phoneRightSidebarMinimum(reclaimedSpace: -50), 180)
        XCTAssertEqual(EditorLayout.phoneRightSidebarMinimum(reclaimedSpace: 1000), 100)
    }
    func testSavedPhoneSidebarWidthReturnsAfterTemporaryOrientationClamp() {
        let savedRight = 120.0
        let reclaimedMinimum = EditorLayout.phoneRightSidebarMinimum(reclaimedSpace: 50 / 0.75)
        let homeEdge = EditorLayout.sidebarWidths(availableWidth: 1000, left: 180, right: savedRight,
            rightMinimum: reclaimedMinimum)
        XCTAssertEqual(homeEdge.right, savedRight)
        let cameraEdge = EditorLayout.sidebarWidths(availableWidth: 1000, left: 180, right: savedRight,
            rightMinimum: EditorLayout.phoneRightSidebarMinimum(reclaimedSpace: 0))
        XCTAssertEqual(cameraEdge.right, 180)
        let restored = EditorLayout.sidebarWidths(availableWidth: 1000, left: 180, right: savedRight,
            rightMinimum: reclaimedMinimum)
        XCTAssertEqual(restored, homeEdge)
        // Existing iPad callers retain their 220pt floor and drag behavior.
        XCTAssertEqual(EditorLayout.sidebarWidths(availableWidth: 1000, left: 180, right: savedRight).right, 220)
        XCTAssertEqual(EditorLayout.resizedSidebarWidth(current: 292, translation: 1000, side: .right,
            maximum: 560), 220)
    }
    func testPortraitPhonesAndSmallWindowsRemainCompact() {
        XCTAssertEqual(EditorLayout.mode(width: 412, height: 915), .compact)
        XCTAssertEqual(EditorLayout.mode(width: 600, height: 839), .compact)
        XCTAssertEqual(EditorLayout.mode(width: 639, height: 375), .compact)
        XCTAssertEqual(EditorLayout.mode(width: 874, height: 299), .compact)
    }
    func testLandscapePhonesUseDesktopWithinSafeAreas() {
        XCTAssertEqual(EditorLayout.mode(width: 748, height: 381), .desktop)
        XCTAssertEqual(EditorLayout.mode(width: 667, height: 354), .desktop)
        XCTAssertEqual(EditorLayout.mode(width: 640, height: 300), .desktop)
        XCTAssertEqual(EditorLayout.mode(width: 381, height: 748), .compact)
    }
    func testDesktopIncludesExactUsableWindowBoundary() {
        XCTAssertEqual(EditorLayout.mode(width: 840, height: 360), .desktop)
        XCTAssertEqual(EditorLayout.mode(width: 1280, height: 800), .desktop)
        XCTAssertEqual(EditorLayout.mode(width: 1280, height: 359), .desktop)
    }
    func testDesktopReviewPlayerUsesAndroidBounds() {
        // Android's 100 px / density 2 is SwiftUI's 50-point translation.
        XCTAssertEqual(EditorLayout.resizedPlayerHeight(current: 320, translation: 50, desktop: true), 370)
        XCTAssertEqual(EditorLayout.resizedPlayerHeight(current: 200, translation: -500, desktop: true), 180)
        XCTAssertEqual(EditorLayout.resizedPlayerHeight(current: 600, translation: 1000, desktop: true), 1200)
    }
    func testCompactAndSetupHaveTheirOwnPlayerBounds() {
        XCTAssertEqual(EditorLayout.resizedPlayerHeight(current: 176, translation: 50, desktop: false), 226)
        XCTAssertEqual(EditorLayout.resizedPlayerHeight(current: 176, translation: -500, desktop: false), 140)
        XCTAssertEqual(EditorLayout.playerHeight(1000, desktop: false), 720)
        XCTAssertEqual(EditorLayout.playerHeight(100, desktop: true, setup: true), 140)
        XCTAssertEqual(EditorLayout.resizedPlayerHeight(current: 600, translation: 1000, desktop: true, setup: true), 1200)
        XCTAssertEqual(EditorLayout.playerHeight(1000, desktop: false, setup: true), 720)
    }
    func testSidebarDragDirectionAndBounds() {
        XCTAssertEqual(EditorLayout.resizedSidebarWidth(current: 292, translation: 50, side: .left, maximum: 520), 342)
        XCTAssertEqual(EditorLayout.resizedSidebarWidth(current: 292, translation: -500, side: .left, maximum: 520), 180)
        XCTAssertEqual(EditorLayout.resizedSidebarWidth(current: 292, translation: -500, side: .right, maximum: 560), 560)
        XCTAssertEqual(EditorLayout.resizedSidebarWidth(current: 292, translation: 50, side: .right, maximum: 560), 242)
        XCTAssertEqual(EditorLayout.resizedSidebarWidth(current: 292, translation: 500, side: .right, maximum: 560), 220)
    }
    func testSidebarsLeaveRoomForCenterAtDesktopBreakpoint() {
        let narrow = EditorLayout.sidebarWidths(availableWidth: 840, left: 292, right: 292)
        XCTAssertEqual(narrow.left, 292)
        XCTAssertEqual(narrow.right, 264)
        XCTAssertEqual(narrow.leftMaximum, 292)
        XCTAssertEqual(narrow.rightMaximum, 264)
        let wide = EditorLayout.sidebarWidths(availableWidth: 1366, left: 1000, right: 1000)
        XCTAssertEqual(wide.left, 520)
        XCTAssertEqual(wide.right, 560)
    }
    func testFullMatchScoreControlsAndClipRowsShareTallEventsColumn() {
        // Physical iPad evidence: 520pt cap hid corrections, so exercise a representative
        // 746pt scoreboard in a 1000pt inner column.
        let scoreHeight = 746.0
        let clips = EditorLayout.clipRegisterHeight(availableHeight: 1000, scoreContentHeight: scoreHeight)
        XCTAssertGreaterThanOrEqual(clips, 180)
        XCTAssertLessThanOrEqual(scoreHeight + clips + 51, 1000)
        // Extra width may shorten wrapped labels; return that room to clips.
        XCTAssertGreaterThan(EditorLayout.clipRegisterHeight(availableHeight: 1000, scoreContentHeight: 600), clips)
    }
    func testShortEventsColumnScrollsRatherThanCompressingCorrectionsOrClipRows() {
        let clips = EditorLayout.clipRegisterHeight(availableHeight: 400, scoreContentHeight: 746)
        XCTAssertGreaterThanOrEqual(clips, 180)
        XCTAssertGreaterThan(746 + clips + 51, 400)
        // Disabled tracking does not reserve the former fixed 55% score pane.
        XCTAssertGreaterThan(EditorLayout.clipRegisterHeight(availableHeight: 1000, scoreContentHeight: 31), 900)
    }
}
