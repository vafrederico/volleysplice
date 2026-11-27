import XCTest
@testable import VolleyCore

final class InterfaceScaleTests: XCTestCase {
    func testAutomaticPhoneLandscapeDefaultAndExplicitOverrides() {
        XCTAssertEqual(InterfaceScale.automatic(isPhone: true, width: 748, height: 381), 0.75)
        XCTAssertEqual(InterfaceScale.automatic(isPhone: true, width: 402, height: 780), 1)
        XCTAssertEqual(InterfaceScale.automatic(isPhone: false, width: 1192, height: 820), 1)
        XCTAssertEqual(InterfaceScale.resolved(1, custom: false, automatic: 0.75), 0.75)
        XCTAssertEqual(InterfaceScale.resolved(1, custom: true, automatic: 0.75), 1)
        XCTAssertEqual(InterfaceScale.resolved(0.8, custom: false, automatic: 0.75), 0.8, accuracy: 0.0001)
    }
    func testExtendedRangeKeepsFivePercentStepsAndRejectsInvalidPreferences() {
        for (input, expected) in [(0.5, 0.6), (0.6, 0.6), (0.61, 0.6), (0.64, 0.65),
                                  (0.69, 0.7), (0.74, 0.75), (0.81, 0.8), (0.99, 1.0), (1.5, 1.25)] {
            XCTAssertEqual(InterfaceScale.normalized(input), expected, accuracy: 0.0001)
        }
        XCTAssertEqual(InterfaceScale.normalized(.nan), 1)
        XCTAssertEqual(InterfaceScale.normalized(.infinity), 1)
    }
    func testExistingManualPreferencesAndNewMinimumOverrideAutomaticDefaults() {
        for percent in stride(from: 60, through: 125, by: 5) {
            let preference = Double(percent) / 100
            for automatic in [0.75, 1.0] {
                XCTAssertEqual(InterfaceScale.resolved(preference, custom: true, automatic: automatic),
                               preference, accuracy: 0.0001)
                // Legacy non-default values remain explicit preferences even
                // when saved before the custom-override flag was introduced.
                if percent != 100 {
                    XCTAssertEqual(InterfaceScale.resolved(preference, custom: false, automatic: automatic),
                                   preference, accuracy: 0.0001)
                }
            }
        }
    }
    func testSmallerInterfaceAddsLogicalRoomWithoutChangingPortraitPhoneMode() {
        XCTAssertEqual(EditorLayout.mode(width: 402 / 0.75, height: 780 / 0.75), .compact)
        XCTAssertEqual(EditorLayout.mode(width: 748 / 0.75, height: 381 / 0.75), .desktop)
        XCTAssertEqual(EditorLayout.mode(width: 748 / 1.25, height: 381 / 1.25), .compact)
        XCTAssertEqual(EditorLayout.mode(width: 402 / 0.6, height: 780 / 0.6), .compact)
        XCTAssertEqual(EditorLayout.mode(width: 748 / 0.6, height: 381 / 0.6), .desktop)
    }
    func testIPhoneDuoRetainsFullScaleWhenUnfoldedAndPhoneScaleWhenFolded() {
        let foldedPortrait = EditorLayout.usesPhoneInterface(
            idiomIsPhone: true, phoneTestActive: false, horizontalRegular: false, verticalRegular: true,
            width: 466, height: 678)
        let foldedLandscape = EditorLayout.usesPhoneInterface(
            idiomIsPhone: true, phoneTestActive: false, horizontalRegular: false, verticalRegular: false,
            width: 678, height: 466)
        let unfoldedLandscape = EditorLayout.usesPhoneInterface(
            idiomIsPhone: true, phoneTestActive: false, horizontalRegular: true, verticalRegular: true,
            width: 951, height: 669)
        let unfoldedPortrait = EditorLayout.usesPhoneInterface(
            idiomIsPhone: true, phoneTestActive: false, horizontalRegular: true, verticalRegular: true,
            width: 669, height: 951)

        XCTAssertEqual(InterfaceScale.automatic(isPhone: foldedPortrait, width: 466, height: 678), 1)
        XCTAssertEqual(InterfaceScale.automatic(isPhone: foldedLandscape, width: 678, height: 466), 0.75)
        XCTAssertEqual(InterfaceScale.automatic(isPhone: unfoldedLandscape, width: 951, height: 669), 1)
        XCTAssertEqual(InterfaceScale.automatic(isPhone: unfoldedPortrait, width: 669, height: 951), 1)
    }
}
