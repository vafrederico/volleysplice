import XCTest
import UIKit

final class EditorTutorialUITests: XCTestCase {
    override func setUpWithError() throws { continueAfterFailure = false }

    @MainActor func testDeferredScoreToggleQueuesOnceAndHidesDisabledDetails() {
        XCUIDevice.shared.orientation = .portrait
        let app = XCUIApplication()
        app.launchArguments = ["--parity-editor", "--parity-deferred-score"]
        app.launch()
        defer { app.terminate() }
        XCTAssertTrue(app.buttons["editorMore"].waitForExistence(timeout: 30), app.debugDescription)
        if app.buttons["tutorialSkip"].exists { app.buttons["tutorialSkip"].tap() }
        let toggle = app.switches["scoreTracking"]
        XCTAssertTrue(toggle.waitForExistence(timeout: 15), app.debugDescription)
        // This test targets score queueing, so normal scrolling to the scoreboard
        // is allowed. The separate tutorial regression never swipes to reveal it.
        for _ in 0..<8 where !toggle.isHittable { app.swipeUp() }
        XCTAssertTrue(toggle.isHittable)
        XCTAssertEqual(toggle.value as? String, "0")
        XCTAssertFalse(app.descendants(matching: .any).matching(identifier: "scoreMarkerList").firstMatch.exists)
        XCTAssertFalse(app.textFields["team1Name"].exists)
        XCTAssertFalse(app.buttons["prepareScores"].exists)
        toggle.tap()
        let enabled = XCTNSPredicateExpectation(predicate: NSPredicate(format: "value == %@", "1"), object: toggle)
        XCTAssertEqual(XCTWaiter.wait(for: [enabled], timeout: 15), .completed)
        XCTAssertTrue(app.buttons["processingQueueBar"].waitForExistence(timeout: 15))
        toggle.tap(); toggle.tap()
        XCTAssertFalse(app.alerts["Unable to complete"].exists)
        app.buttons["processingQueueBar"].tap()
        let name = "ios-parity-deferred-score.mp4"
        let rows = app.cells.containing(.staticText, identifier: name)
        let queued = XCTNSPredicateExpectation(predicate: NSPredicate { _, _ in
            rows.count == 1 && rows.firstMatch.staticTexts["Score preparation · Queued"].exists
        }, object: nil)
        XCTAssertEqual(XCTWaiter.wait(for: [queued], timeout: 15), .completed, app.debugDescription)
        XCTAssertEqual(rows.count, 1, "Retoggling must not enqueue a duplicate score job")
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = "deferred-score-queued-on-enable"; attachment.lifetime = .keepAlways; add(attachment)
        // Leave no synthetic pending work for subsequent launches/tests.
        let remove = rows.firstMatch.buttons.matching(NSPredicate(format: "identifier BEGINSWITH %@", "removeJob-")).firstMatch
        XCTAssertTrue(remove.exists); remove.tap()
        app.buttons["queueDone"].tap()
    }

    @MainActor func testPhoneReplayRevealsLaterEditorTargetsWithoutManualScrolling() throws {
        XCUIDevice.shared.orientation = .portrait
        let app = XCUIApplication()
        app.launchArguments = ["--parity-editor"]
        app.launch()
        defer { app.terminate() }
        try XCTSkipIf(app.frame.width >= 600, "This regression exercises the phone portrait scroll container.")
        XCTAssertTrue(app.buttons["editorMore"].waitForExistence(timeout: 30), app.debugDescription)
        if app.buttons["tutorialSkip"].exists {
            app.buttons["tutorialSkip"].tap()
            // A previously persisted later step can launch scrolled down. Relaunch
            // after dismissal restores the normal top; no test swipe reveals targets.
            app.terminate(); app.launch()
        }
        let more = app.buttons["editorMore"]
        XCTAssertTrue(more.waitForExistence(timeout: 30))
        XCTAssertTrue(more.isHittable)
        more.tap()
        let replay = app.buttons["replayTutorial"]
        XCTAssertTrue(replay.waitForExistence(timeout: 10)); replay.tap()

        let steps: [(String, String?)] = [
            ("Fine-tune only when you need to", nil),
            ("Save the finished video", nil),
            ("Add a scoreboard if you want one", nil),
            ("Check the score markers", nil),
            ("Watch the video", "editorVideo"),
            ("Start with what needs attention", "reviewCleanup"),
            ("Review the suggested clips", "gameTimeline"),
            ("Fix one clip", "focusedTimeline"),
            ("Add anything VolleySplice missed", "manualRange")
        ]
        for (title, identifier) in steps {
            let heading = app.staticTexts["tutorialTitle"]
            let titleExpectation = XCTNSPredicateExpectation(predicate: NSPredicate(format: "exists == true AND label == %@", title), object: heading)
            XCTAssertEqual(XCTWaiter.wait(for: [titleExpectation], timeout: 15), .completed, app.debugDescription)
            if let identifier { assertVisible(identifier, in: app) }
            let next = app.buttons["tutorialNext"]
            XCTAssertTrue(next.isEnabled); XCTAssertTrue(next.isHittable)
            next.tap()
        }
        let dismissed = XCTNSPredicateExpectation(predicate: NSPredicate(format: "exists == false"), object: app.staticTexts["tutorialTitle"])
        XCTAssertEqual(XCTWaiter.wait(for: [dismissed], timeout: 10), .completed)
    }

    @MainActor private func assertVisible(_ identifier: String, in app: XCUIApplication, file: StaticString = #filePath, line: UInt = #line) {
        let target = app.descendants(matching: .any).matching(identifier: identifier).firstMatch
        let visible = XCTNSPredicateExpectation(predicate: NSPredicate { _, _ in
            guard target.exists else { return false }
            let frame = target.frame
            return frame.width > 0 && frame.height > 0 && app.frame.insetBy(dx: 0, dy: 24).contains(frame)
        }, object: nil)
        XCTAssertEqual(XCTWaiter.wait(for: [visible], timeout: 15), .completed,
                       "Tutorial did not reveal \(identifier): \(target.debugDescription)", file: file, line: line)
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = "phone-tutorial-" + identifier; attachment.lifetime = .keepAlways; add(attachment)
    }
}
