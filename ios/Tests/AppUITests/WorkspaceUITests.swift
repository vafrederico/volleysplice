import XCTest
import UIKit

final class WorkspaceUITests: XCTestCase {
    override func setUpWithError() throws { continueAfterFailure = false }
    @MainActor func testSetupTutorialAndFilePickerInBothOrientations() throws {
        XCUIDevice.shared.orientation = .portrait
        let app = XCUIApplication()
        app.launch()
        let skip = app.buttons["tutorialSkip"]
        if skip.waitForExistence(timeout: 15) { skip.tap() }
        // P25 requests phone portrait. The VM's landscape framebuffer/activation
        // coordinates are unreliable; iPad retains its structural rotation check.
        let orientations: [UIDeviceOrientation] = app.frame.width > 600 ? [.landscapeLeft, .portrait] : [.portrait]
        for orientation in orientations {
            XCUIDevice.shared.orientation = orientation
            let choose = app.buttons["chooseRecording"]
            XCTAssertTrue(choose.waitForExistence(timeout: 30))
            XCTAssertTrue(choose.isHittable)
            let shot = XCTAttachment(screenshot: app.screenshot())
            shot.name = orientation == .portrait ? "setup-portrait" : "setup-landscape"
            shot.lifetime = .keepAlways; add(shot)
            choose.tap()
            XCTAssertTrue(app.buttons["Camera Roll"].waitForExistence(timeout: 15))
            app.buttons["Files"].tap()
            let cancel = app.buttons["Cancel"].firstMatch
            XCTAssertTrue(cancel.waitForExistence(timeout: 30), app.debugDescription)
            cancel.tap()
            XCTAssertTrue(choose.waitForExistence(timeout: 15))
        }
        app.buttons["openSavedProjects"].tap()
        XCTAssertTrue(app.buttons["Cancel"].firstMatch.waitForExistence(timeout: 15))
        XCTAssertFalse(app.buttons["importProject"].exists)
        XCTAssertFalse(app.buttons["refreshRecordings"].exists)
        app.buttons["Cancel"].firstMatch.tap()
        app.buttons["setupMore"].tap()
        app.buttons["replayTutorial"].tap()
        XCTAssertTrue(app.buttons["tutorialNext"].waitForExistence(timeout: 15))
        let tutorial = XCTAttachment(screenshot: app.screenshot()); tutorial.name = "setup-tutorial"; tutorial.lifetime = .keepAlways; add(tutorial)
        app.buttons["tutorialSkip"].tap()
    }
}
