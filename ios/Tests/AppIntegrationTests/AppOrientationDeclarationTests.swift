import Foundation
import XCTest

/// Verify the processed host application's plist, including Xcode generation
/// and device-qualified keys, rather than duplicating a Swift layout decision.
final class AppOrientationDeclarationTests: XCTestCase {
    func testIPadIsLandscapeOnlyAndIPhoneRetainsPortrait() throws {
        // Bundle.infoDictionary resolves device-qualified keys for the current
        // phone, hiding the iPad declaration. Inspect the processed plist bytes.
        let data = try Data(contentsOf: Bundle.main.bundleURL.appendingPathComponent("Info.plist"))
        let info = try XCTUnwrap(PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any])
        XCTAssertEqual(info["CFBundleIdentifier"] as? String, "com.volleysplice.VolleySplice")
        let landscapes = Set(["UIInterfaceOrientationLandscapeLeft", "UIInterfaceOrientationLandscapeRight"])
        let tablet = try XCTUnwrap(info["UISupportedInterfaceOrientations~ipad"] as? [String])
        XCTAssertEqual(Set(tablet), landscapes)
        XCTAssertEqual(tablet.count, 2)
        let phone = try XCTUnwrap((info["UISupportedInterfaceOrientations~iphone"] ?? info["UISupportedInterfaceOrientations"]) as? [String])
        XCTAssertEqual(Set(phone), landscapes.union(["UIInterfaceOrientationPortrait"]))
        XCTAssertEqual(phone.count, 3)
        XCTAssertEqual(info["UIRequiresFullScreen"] as? Bool, true,
                       "Legacy iPad multitasking must not override the landscape-only declaration")
        XCTAssertNil(info["UIRequiresFullScreenIgnoredStartingWithVersion"],
                     "Do not silently opt the supported iPadOS versions back into unrestricted resizing")
    }
}
