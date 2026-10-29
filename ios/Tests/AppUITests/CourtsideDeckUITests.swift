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
    @MainActor private func seekSource(_ app: XCUIApplication, fraction: Double) {
        let position = min(2, max(0, fraction * 2))
        let index = min(1, Int(position))
        let column = app.otherElements["deckSourceTimelineColumn-\(index)"]
        XCTAssertTrue(column.waitForExistence(timeout: 10))
        let scale = column.frame.width / 44
        let height = max(1, column.frame.height)
        let y = (8 * scale + (height - 16 * scale) * CGFloat(position - Double(index))) / height
        column.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: y)).tap()
    }
    @MainActor private func seekMissing(_ app: XCUIApplication, fraction: Double) {
        let position = min(3, max(0, fraction * 3))
        let index = min(2, Int(position))
        let row = app.otherElements["deckAddTimelineRow-\(index)"]
        XCTAssertTrue(row.waitForExistence(timeout: 10))
        row.coordinate(withNormalizedOffset: CGVector(dx: position - Double(index), dy: 0.5)).tap()
    }
    @MainActor func testAddStaysInlineAndCancellationDoesNotCreateARally() {
        let app = launch(); defer { app.terminate() }
        XCTAssertTrue(app.staticTexts["deckReviewCount"].label.hasPrefix("Reviewing "))
        XCTAssertEqual(app.buttons["deckBrowse"].label, "All rallies (6)")
        XCTAssertFalse(app.progressIndicators.firstMatch.exists)
        app.buttons["deckBrowse"].tap()
        seekSource(app, fraction: 0.22)
        app.buttons["deckAddMissing"].tap()
        let mark = app.buttons["deckAddMark"]
        XCTAssertTrue(mark.waitForExistence(timeout: 10))
        XCTAssertTrue(mark.label.hasPrefix("Mark start at "))
        XCTAssertFalse(mark.isEnabled)
        XCTAssertFalse(app.buttons["deckAddReset"].isEnabled)
        XCTAssertTrue(app.buttons["deckPlay"].isHittable, "The original player stays available above the inline picker")
        XCTAssertFalse(app.navigationBars["Add missing rally"].exists)
        XCTAssertFalse(app.sliders["deckAddSourceSeek"].exists)
        seekMissing(app, fraction: 0.675)
        let enabled = XCTNSPredicateExpectation(predicate: NSPredicate(format: "enabled == true"), object: mark)
        XCTAssertEqual(XCTWaiter.wait(for: [enabled], timeout: 10), .completed)
        mark.tap()
        XCTAssertEqual(mark.label, "Mark end & add rally")
        XCTAssertFalse(mark.isEnabled)
        app.buttons["deckAddCancel"].tap()
        XCTAssertEqual(app.staticTexts["deckReviewCount"].label, "All rallies")
        app.buttons["deckBrowse"].tap()
        XCTAssertEqual(app.buttons["deckBrowse"].label, "All rallies (6)", "Marking and cancellation must not create a rally")
        app.buttons["deckBrowse"].tap()
        app.buttons["deckAddMissing"].tap()
        XCTAssertTrue(app.buttons["deckAddMark"].label.hasPrefix("Mark start at "), "Reopening starts a fresh selection")
    }
    @MainActor func testMarkStartThenEndAddsAndImmediatelyOpensRallyEditing() {
        let app = launch(); defer { app.terminate() }
        app.buttons["deckBrowse"].tap()
        app.buttons["deckAddMissing"].tap()
        XCTAssertFalse(app.sliders["deckAddSourceSeek"].exists)
        seekMissing(app, fraction: 0.675)
        let mark = app.buttons["deckAddMark"]
        mark.tap()
        let start = app.staticTexts["deckAddStartTime"].label
        seekMissing(app, fraction: 0.22)
        XCTAssertFalse(mark.isEnabled, "An end before the start cannot add a rally")
        XCTAssertEqual(app.staticTexts["deckAddStartTime"].label, start)
        seekMissing(app, fraction: 0.71)
        XCTAssertTrue(mark.isEnabled)
        mark.tap()
        XCTAssertTrue(app.otherElements["deckTrimStart"].waitForExistence(timeout: 10) || app.buttons["deckTrimStart"].exists)
        XCTAssertFalse(app.buttons["deckAddMark"].exists)
        XCTAssertTrue(app.staticTexts["deckReviewCount"].label.hasPrefix("Editing rally "))
        XCTAssertEqual(app.buttons["deckBrowse"].label, "Review rallies")
        app.buttons["deckBrowse"].tap()
        XCTAssertEqual(app.buttons["deckBrowse"].label, "All rallies (7)")
        app.buttons["deckBrowse"].tap()
        XCTAssertEqual(app.staticTexts["deckReviewCount"].label, "All rallies")
    }
    @MainActor func testTwoColumnSourceSeekingAndReviewModeForPendingRallies() {
        let app = launch(); defer { app.terminate() }
        app.buttons["deckBrowse"].tap()
        XCTAssertEqual(app.staticTexts["deckReviewCount"].label, "All rallies")
        XCTAssertFalse(app.sliders["deckSourceSeek"].exists)
        seekSource(app, fraction: 0.775)
        XCTAssertTrue(app.buttons["deckRallyRow-R005"].isSelected)
        XCTAssertEqual(app.buttons["deckRallyRow-R005"].value as? String, "List center")
        XCTAssertEqual(app.buttons["deckEditSelected"].label, "Edit rally 5")
        app.buttons["deckEditSelected"].tap()
        XCTAssertTrue(app.staticTexts["deckReviewCount"].label.hasPrefix("Reviewing "))
        XCTAssertEqual(app.buttons["deckBrowse"].label, "All rallies (6)")
        app.buttons["deckBrowse"].tap()
        app.buttons["deckToggleSourceTimeline"].tap()
        XCTAssertFalse(app.otherElements["deckSourceTimelineColumn-0"].exists)
        XCTAssertFalse(app.otherElements["deckSourceTimelineColumn-1"].exists)
        app.buttons["deckToggleSourceTimeline"].tap()
        seekSource(app, fraction: 0.075)
        XCTAssertTrue(app.buttons["deckRallyRow-R001"].isSelected)
        app.buttons["deckEditSelected"].tap()
        XCTAssertEqual(app.staticTexts["deckReviewCount"].label, "Editing rally 1")
        XCTAssertEqual(app.buttons["deckBrowse"].label, "Review rallies")
    }

    @MainActor func testManualListScrollingMovesOneDiamondWithoutSeekingTheVideo() {
        let app = launch(); defer { app.terminate() }
        app.buttons["deckBrowse"].tap()
        seekSource(app, fraction: 0)
        let playhead = app.staticTexts["deckPlayhead"].label
        let marked = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH %@ AND value == %@", "deckRallyRow-", "List center"))
        XCTAssertEqual(marked.count, 1)
        let before = marked.firstMatch.identifier
        app.scrollViews.firstMatch.swipeUp()
        XCTAssertEqual(app.staticTexts["deckPlayhead"].label, playhead)
        XCTAssertEqual(marked.count, 1)
        XCTAssertNotEqual(marked.firstMatch.identifier, before)
        let center = marked.firstMatch.identifier
        app.buttons["workspace-editor-tab"].tap()
        app.buttons["workspace-rallies-tab"].tap()
        XCTAssertEqual(marked.firstMatch.identifier, center)
        XCTAssertEqual(app.staticTexts["deckPlayhead"].label, playhead)
        seekSource(app, fraction: 0.775)
        XCTAssertEqual(app.buttons["deckRallyRow-R005"].value as? String, "List center")
        XCTAssertTrue(app.buttons["deckRallyRow-R005"].isSelected)
    }

    @MainActor func testOrdinaryTapOpensTheCurrentlySelectedRally() {
        let app = launch(); defer { app.terminate() }
        app.buttons["deckBrowse"].tap()
        let first = app.buttons["deckRallyRow-R001"]
        first.tap()
        XCTAssertTrue(first.isSelected)
        XCTAssertEqual(app.staticTexts["deckReviewCount"].label, "All rallies")
        first.tap()
        XCTAssertEqual(app.staticTexts["deckReviewCount"].label, "Editing rally 1")
        app.buttons["deckBrowse"].tap()
        XCTAssertEqual(app.buttons["deckBrowse"].label, "All rallies (6)")
        app.buttons["deckBrowse"].tap()
        first.doubleTap()
        XCTAssertEqual(app.staticTexts["deckReviewCount"].label, "Editing rally 1")
        app.buttons["deckBrowse"].tap()
        app.buttons["deckBrowse"].tap()
        seekSource(app, fraction: 0.775)
        app.buttons["deckRallyRow-R005"].tap()
        XCTAssertTrue(app.staticTexts["deckReviewCount"].label.hasPrefix("Reviewing "))
        XCTAssertEqual(app.buttons["deckBrowse"].label, "All rallies (6)", "Opening cannot make a review decision")
    }

}
