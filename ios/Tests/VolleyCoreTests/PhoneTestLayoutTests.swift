import XCTest
@testable import VolleyCore

final class PhoneTestLayoutTests: XCTestCase {
    func testPortraitUsesPhonePointsAndReservedSafeAreas() {
        let layout = PhoneTestLayout(landscape: false)
        XCTAssertEqual(layout.width, 390); XCTAssertEqual(layout.height, 844)
        XCTAssertEqual(layout.contentWidth, 390); XCTAssertEqual(layout.contentHeight, 763)
        XCTAssertEqual(layout.top + layout.contentHeight + layout.bottom, layout.height)
    }
    func testLandscapeRotatesTheSamePhoneAndReservesCameraAndHomeEdges() {
        let layout = PhoneTestLayout(landscape: true)
        XCTAssertEqual(layout.width, 844); XCTAssertEqual(layout.height, 390)
        XCTAssertEqual(layout.contentWidth, 750); XCTAssertEqual(layout.contentHeight, 369)
        XCTAssertEqual(layout.leading + layout.contentWidth + layout.trailing, layout.width)
    }
    func testIPadDoesNotEnlargePhoneLogicalCoordinates() {
        XCTAssertEqual(PhoneTestLayout(landscape: false).fit(width: 1024, height: 1300), 1)
        XCTAssertEqual(PhoneTestLayout(landscape: true).fit(width: 1366, height: 980), 1)
    }
    func testSmallWindowsFitTheWholePhoneWithoutChangingItsLayoutDimensions() {
        let layout = PhoneTestLayout(landscape: false)
        XCTAssertEqual(layout.fit(width: 300, height: 600), 600 / 844.0, accuracy: 0.0001)
        XCTAssertEqual(layout.width, 390); XCTAssertEqual(layout.height, 844)
    }
    func testPhoneAutomaticScaleMatchesNativePhoneInBothOrientations() {
        for landscape in [false, true] {
            let layout = PhoneTestLayout(landscape: landscape)
            XCTAssertEqual(InterfaceScale.automatic(isPhone: true, width: layout.width, height: layout.height), landscape ? 0.75 : 1)
            XCTAssertEqual(EditorLayout.mode(width: layout.contentWidth / (landscape ? 0.75 : 1),
                height: layout.contentHeight / (landscape ? 0.75 : 1)), landscape ? .desktop : .compact)
        }
    }
}
