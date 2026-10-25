import XCTest
import UIKit

/// Apple-runtime checks to run when available. The Windows compatible preview
/// cannot execute these tests or validate native media/VoiceOver behavior.
final class CourtsideDeckUITests: XCTestCase {
    override func setUpWithError() throws { continueAfterFailure = false }
    @MainActor private func launch() -> XCUIApplication {
        let app = XCUIApplication()
        app.launchArguments = ["--parity-editor"]
        app.launch()
        if app.buttons["tutorialSkip"].waitForExistence(timeout: 5) { app.buttons["tutorialSkip"].tap() }
        XCTAssertTrue(app.buttons["workspace-rallies-tab"].waitForExistence(timeout: 30))
        return app
    }
    @MainActor func testRalliesIsDefaultAndTeamEditsAreSharedWithFullEditor() {
        let app = launch(); defer { app.terminate() }
        XCTAssertTrue(app.buttons["workspace-rallies-tab"].isSelected)
        XCTAssertFalse(app.buttons["Edit Synthetic A"].exists)
        app.buttons["editorSettings"].tap()
        let input = app.textFields["settingsTeam1Name"]
        XCTAssertTrue(input.waitForExistence(timeout: 10))
        XCTAssertFalse(app.buttons["projectOptions"].exists)
        input.tap(); input.press(forDuration: 1)
        // Replace through the text field's selection rather than relying on
        // keyboard deletion counts or the current team name's length.
        if app.menuItems["Select All"].exists { app.menuItems["Select All"].tap() }
        else { input.typeText(String(repeating: XCUIKeyboardKey.delete.rawValue, count: (input.value as? String ?? "").count)) }
        input.typeText("Practice A")
        app.buttons["settingsDone"].tap()
        app.buttons["workspace-editor-tab"].tap()
        let name = app.textFields["team1Name"]
        XCTAssertTrue(name.waitForExistence(timeout: 10))
        XCTAssertEqual(name.value as? String, "Practice A")
        app.buttons["workspace-export-tab"].tap()
        XCTAssertTrue(app.buttons["exportVideo"].waitForExistence(timeout: 10))
        app.buttons["workspace-rallies-tab"].tap()
        XCTAssertFalse(app.buttons["Edit Practice A"].exists)
        app.buttons["editorSettings"].tap()
        XCTAssertEqual(app.textFields["settingsTeam1Name"].value as? String, "Practice A")
    }
    @MainActor func testAddSourceSliderCanSeekOutOfExistingFootageWithoutAdding() {
        let app = launch(); defer { app.terminate() }
        app.buttons["deckBrowse"].tap()
        let seek = app.sliders["deckSourceSeek"]
        XCTAssertTrue(seek.waitForExistence(timeout: 10))
        seek.adjust(toNormalizedSliderPosition: 0.22)
        let before = app.staticTexts.matching(NSPredicate(format: "label MATCHES %@", "[0-9]+ rallies")).firstMatch.label
        app.buttons["deckAddMissing"].tap()
        let add = app.buttons["Add rally"]
        XCTAssertTrue(add.waitForExistence(timeout: 10))
        XCTAssertFalse(add.isEnabled)
        let source = app.sliders["deckAddSourceSeek"]
        XCTAssertTrue(source.exists)
        // The fixture ends with unoccupied source time. Native UISlider exposes
        // adjustable and set-value actions; this exercises its set-value path.
        source.adjust(toNormalizedSliderPosition: 0.98)
        let enabled = XCTNSPredicateExpectation(predicate: NSPredicate(format: "enabled == true"), object: add)
        XCTAssertEqual(XCTWaiter.wait(for: [enabled], timeout: 10), .completed)
        app.buttons["Cancel"].tap()
        XCTAssertTrue(app.staticTexts[before].exists, "Seeking and cancellation must not create a rally")
    }
}
