import XCTest
import UIKit

/// Scoreboard visibility affects overview and focused timelines, including their
/// accessible marker actions, without deleting stored serves or team switches.
final class TimelineScoreVisibilityUITests: XCTestCase {
    override func setUpWithError() throws { continueAfterFailure = false }

    @MainActor func testScoreboardToggleHidesTimelineMarkersAndRestoresThem() {
        XCUIDevice.shared.orientation = .portrait
        let app = XCUIApplication()
        app.launchArguments = ["--parity-editor"]
        app.launch()
        defer { app.terminate(); XCUIDevice.shared.orientation = .portrait }
        if app.buttons["tutorialSkip"].waitForExistence(timeout: 5) { app.buttons["tutorialSkip"].tap() }
        XCTAssertTrue(app.buttons["reviewCleanup"].waitForExistence(timeout: 30))
        if app.buttons["tutorialSkip"].exists { app.buttons["tutorialSkip"].tap() }

        let serves = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH %@", "timelineServe-"))
        let switches = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH %@", "timelineSwitch-"))
        XCTAssertTrue(serves.firstMatch.waitForExistence(timeout: 15))
        XCTAssertTrue(switches.firstMatch.waitForExistence(timeout: 15))
        let originalServes = Set(serves.allElementsBoundByIndex.map(\.identifier))
        let originalSwitches = Set(switches.allElementsBoundByIndex.map(\.identifier))
        let focus = app.descendants(matching: .any).matching(identifier: "focusedTimeline").firstMatch
        XCTAssertTrue(focus.buttons.matching(NSPredicate(format: "identifier BEGINSWITH %@", "timelineServe-")).firstMatch.exists,
                      "Fixture must exercise focused as well as overview markers")

        let toggle = app.switches["scoreTracking"]
        reveal(toggle, in: app)
        XCTAssertEqual(toggle.value as? String, "1")
        toggle.tap()
        let hidden = XCTNSPredicateExpectation(predicate: NSPredicate { _, _ in serves.count == 0 && switches.count == 0 }, object: nil)
        XCTAssertEqual(XCTWaiter.wait(for: [hidden], timeout: 15), .completed)
        XCTAssertTrue(focus.exists, "Disabling scores must retain the rally timeline")

        reveal(toggle, in: app)
        toggle.tap()
        let restored = XCTNSPredicateExpectation(predicate: NSPredicate { _, _ in
            Set(serves.allElementsBoundByIndex.map(\.identifier)) == originalServes &&
            Set(switches.allElementsBoundByIndex.map(\.identifier)) == originalSwitches
        }, object: nil)
        XCTAssertEqual(XCTWaiter.wait(for: [restored], timeout: 15), .completed,
                       "Re-enabling scores must restore the same edited markers")
    }

    @MainActor private func reveal(_ element: XCUIElement, in app: XCUIApplication) {
        XCTAssertTrue(element.waitForExistence(timeout: 15))
        for _ in 0..<10 where !element.isHittable {
            let parent = app.scrollViews.containing(element.elementType, identifier: element.identifier)
                .allElementsBoundByIndex.filter { !$0.frame.intersection(app.frame).isEmpty }
                .min { $0.frame.width * $0.frame.height < $1.frame.width * $1.frame.height }
            guard let parent else { break }
            let y: CGFloat = element.frame.midY < parent.frame.midY ? 0.25 : 0.75
            parent.coordinate(withNormalizedOffset: CGVector(dx: 0.98, dy: y))
                .press(forDuration: 0.05, thenDragTo: parent.coordinate(withNormalizedOffset: CGVector(dx: 0.98, dy: 1 - y)))
        }
        XCTAssertTrue(element.isHittable)
    }
}
