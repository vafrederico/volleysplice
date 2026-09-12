import XCTest
import UIKit

/// Uses the full iPad export imported by ReferenceProjectImportTests, never a
/// synthetic replacement for its inference/timelines. Edits affect only this copy.
final class ReferenceProjectUITests: XCTestCase {
    override func setUpWithError() throws { continueAfterFailure = false }

    @MainActor func testInterfaceScaleChangesWholeEditorAndPersistsIntoSetup() throws {
        XCUIDevice.shared.orientation = .portrait
        let app = XCUIApplication(); app.launch()
        defer { app.terminate() }
        if app.buttons["tutorialSkip"].waitForExistence(timeout: 5) { app.buttons["tutorialSkip"].tap() }
        XCTAssertTrue(app.buttons["projectSelector"].waitForExistence(timeout: 20))
        app.buttons["projectSelector"].tap()
        let reference = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH 'project-project-' AND label CONTAINS %@", "tds6-reference.mp4")).firstMatch
        try XCTSkipUnless(reference.waitForExistence(timeout: 10), "Import full iPad reference into this simulator first")
        reference.tap()
        let settings = app.buttons["editorSettings"]
        XCTAssertTrue(settings.waitForExistence(timeout: 30)); settings.tap()
        let reset = app.buttons["uiScaleDefault"]
        XCTAssertTrue(reset.waitForExistence(timeout: 15))
        let savedScale = scaleState(app)
        if reset.isEnabled { reset.tap() }
        app.buttons["settingsDone"].tap()
        let originalWidth = settings.frame.width
        settings.tap()
        let slider = app.sliders["uiScale"]
        let originalSheetTextHeight = app.staticTexts["uiScalePercent"].frame.height
        dragScaleToEndpoint(slider, larger: false, app: app)
        XCTAssertEqual(app.staticTexts["uiScalePercent"].label, "60%")
        XCTAssertEqual(app.staticTexts["uiScalePercent"].frame.height, originalSheetTextHeight * 0.60, accuracy: 1)
        capture("ui-size-60-settings", app: app)
        app.buttons["settingsDone"].tap()
        XCTAssertEqual(settings.frame.width, originalWidth * 0.60, accuracy: 2)
        capture("ui-size-60-editor", app: app)
        let handle = app.descendants(matching: .any).matching(identifier: "resizeVideo").firstMatch
        reveal(handle, in: app)
        let video = app.descendants(matching: .any).matching(identifier: "editorVideo").firstMatch
        let originalHeight = video.frame.height
        let start = handle.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
        start.press(forDuration: 0.1, thenDragTo: start.withOffset(CGVector(dx: 0, dy: 24)))
        XCTAssertEqual(video.frame.height, originalHeight + 24, accuracy: 3,
                       "At 60%, the player must follow the physical drag, not move only 14.4 points")
        let resized = handle.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
        resized.press(forDuration: 0.1, thenDragTo: resized.withOffset(CGVector(dx: 0, dy: -24)))
        XCTAssertEqual(video.frame.height, originalHeight, accuracy: 3)
        // A changed display preference survives process termination and is shared
        // with setup, rather than being saved in this project's draft.
        app.terminate(); app.launch()
        let setupSettings = app.buttons["setupSettings"]
        XCTAssertTrue(setupSettings.waitForExistence(timeout: 20)); setupSettings.tap()
        XCTAssertTrue(app.staticTexts["uiScalePercent"].waitForExistence(timeout: 15))
        XCTAssertEqual(app.staticTexts["uiScalePercent"].label, "60%")
        capture("ui-size-60-setup-settings", app: app)
        dragScaleToEndpoint(app.sliders["uiScale"], larger: true, app: app)
        XCTAssertEqual(app.staticTexts["uiScalePercent"].label, "125%")
        XCTAssertEqual(app.staticTexts["uiScalePercent"].frame.height, originalSheetTextHeight * 1.25, accuracy: 1)
        capture("ui-size-125-setup-settings", app: app)
        app.buttons["uiScaleDefault"].tap()
        XCTAssertEqual(app.staticTexts["uiScalePercent"].label, "100%")
        XCTAssertFalse(app.buttons["uiScaleDefault"].isEnabled)
        restoreScale(savedScale, app: app)
        app.buttons["Done"].tap()
        XCTAssertTrue(setupSettings.isHittable)
    }

    @MainActor func testFullReferenceRotatesIntoLandscapeWorkspace() throws {
        XCUIDevice.shared.orientation = .portrait
        let app = XCUIApplication(); app.launch()
        defer { XCUIDevice.shared.orientation = .portrait; app.terminate() }
        if app.buttons["tutorialSkip"].waitForExistence(timeout: 5) { app.buttons["tutorialSkip"].tap() }
        XCTAssertTrue(app.buttons["projectSelector"].waitForExistence(timeout: 20))
        app.buttons["projectSelector"].tap()
        let reference = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH 'project-project-' AND label CONTAINS %@", "tds6-reference.mp4")).firstMatch
        try XCTSkipUnless(reference.waitForExistence(timeout: 10), "Import full iPad reference into this simulator first")
        reference.tap()
        let summary = app.descendants(matching: .any).matching(identifier: "exportSummary").firstMatch
        XCTAssertTrue(summary.waitForExistence(timeout: 30))
        XCTAssertEqual(summary.value as? String, "Compact workspace")
        app.buttons["editorSettings"].tap()
        XCTAssertTrue(app.buttons["settingsDone"].waitForExistence(timeout: 15))
        let savedScale = scaleState(app)
        if app.buttons["uiScaleDefault"].isEnabled { app.buttons["uiScaleDefault"].tap() }
        app.buttons["settingsDone"].tap()
        for orientation in [UIDeviceOrientation.landscapeLeft, .landscapeRight] {
            XCUIDevice.shared.orientation = orientation
            let wide = XCTNSPredicateExpectation(predicate: NSPredicate(format: "value == %@", "Wide workspace"), object: summary)
            XCTAssertEqual(XCTWaiter.wait(for: [wide], timeout: 20), .completed)
            XCTAssertGreaterThan(app.frame.width, app.frame.height)
            for identifier in ["projectSelector", "editorSettings", "editorMore", "reviewCleanup", "reviewClips", "reviewServes", "resizeLeftSidebar", "resizeRightSidebar"] {
                let element = app.descendants(matching: .any).matching(identifier: identifier).firstMatch
                XCTAssertTrue(element.isHittable, identifier)
                XCTAssertGreaterThanOrEqual(element.frame.minX, app.frame.minX, identifier)
                XCTAssertLessThanOrEqual(element.frame.maxX, app.frame.maxX, identifier)
            }
            // No scrolling: player, transport, both overview rows and the legend
            // must share the initial short phone viewport.
            capture(orientation == .landscapeLeft ? "landscape-left-layout" : "landscape-right-layout", app: app)
            for identifier in ["editorVideo", "playPause", "resizeVideo", "gameTimelineFirst", "gameTimelineSecond", "timelineLegend"] {
                let element = app.descendants(matching: .any).matching(identifier: identifier).firstMatch
                XCTAssertTrue(element.exists, identifier)
                XCTAssertGreaterThanOrEqual(element.frame.minY, app.frame.minY, identifier)
                XCTAssertLessThanOrEqual(element.frame.maxY, app.frame.maxY - 20, identifier)
            }
            XCTAssertTrue(app.buttons["playPause"].isHittable)
            XCTAssertFalse(app.sliders["sourceSeek"].exists)
            XCTAssertFalse(app.staticTexts["markerCount"].exists)
            XCTAssertFalse(app.staticTexts["awardedCount"].exists)
            let markerViewport = app.scrollViews["markerViewport"]
            XCTAssertTrue(markerViewport.exists)
            XCTAssertEqual(markerViewport.frame.height, 4 * 33 * 0.75, accuracy: 1,
                           "Phone marker viewport contains four fixed-height rows")
            let resize = app.descendants(matching: .any).matching(identifier: "resizeVideo").firstMatch
            let play = app.buttons["playPause"]
            XCTAssertEqual(resize.frame.midY, play.frame.midY, accuracy: 2)
            XCTAssertLessThanOrEqual(play.frame.height, 26)
            let next = app.buttons["nextRally"], previous = app.buttons["previousRally"]
            XCTAssertEqual(next.frame.midY, previous.frame.midY, accuracy: 1)
            XCTAssertLessThanOrEqual(next.frame.height, 26)
            XCTAssertTrue(next.isHittable)
            XCTAssertLessThanOrEqual(app.descendants(matching: .any).matching(identifier: "focusedTimeline").firstMatch.frame.height, 37)
            // This fixture runs on iPhone 17 (62pt horizontal safe insets).
            let headerLeft = app.frame.minX + 62
            let headerRight = app.frame.maxX - 62
            XCTAssertGreaterThanOrEqual(app.buttons["projectSelector"].frame.minX, headerLeft)
            XCTAssertLessThanOrEqual(app.buttons["editorMore"].frame.maxX, headerRight)
            let events = app.staticTexts["eventsHeading"]
            if orientation == .landscapeLeft { // camera left, Home edge right
                XCTAssertGreaterThan(next.frame.maxX, headerRight + 20)
                XCTAssertGreaterThanOrEqual(events.frame.minX, headerLeft)
            } else {
                XCTAssertLessThan(events.frame.minX, headerLeft - 20)
                XCTAssertLessThanOrEqual(next.frame.maxX, headerRight)
                XCTAssertTrue(events.isHittable)
            }

            let leftDivider = app.descendants(matching: .any).matching(identifier: "resizeLeftSidebar").firstMatch
            let rightDivider = app.descendants(matching: .any).matching(identifier: "resizeRightSidebar").firstMatch
            // Android's 75% landscape reference gives most width to the center.
            let centerWidth = rightDivider.frame.minX - leftDivider.frame.maxX
            XCTAssertGreaterThan(centerWidth, app.frame.width * 0.45)
            let player = app.descendants(matching: .any).matching(identifier: "editorVideo").firstMatch
            XCTAssertGreaterThan(player.frame.height, 110, "Fitting the timeline must leave a usable video preview")
            capture(orientation == .landscapeLeft ? "full-reference-landscape-left" : "full-reference-landscape-right", app: app)
            app.buttons["editorSettings"].tap()
            XCTAssertTrue(app.buttons["settingsDone"].waitForExistence(timeout: 15))
            XCTAssertEqual(app.staticTexts["uiScalePercent"].label, "75%")
            XCTAssertFalse(app.buttons["uiScaleDefault"].isEnabled)
            app.buttons["settingsDone"].tap()
        }
        XCUIDevice.shared.orientation = .portrait
        let compact = XCTNSPredicateExpectation(predicate: NSPredicate(format: "value == %@", "Compact workspace"), object: summary)
        XCTAssertEqual(XCTWaiter.wait(for: [compact], timeout: 20), .completed)
        XCTAssertTrue(app.buttons["projectSelector"].isHittable)
        XCTAssertTrue(app.buttons["reviewCleanup"].exists)
        app.buttons["editorSettings"].tap()
        XCTAssertTrue(app.buttons["settingsDone"].waitForExistence(timeout: 15))
        restoreScale(savedScale, app: app)
        app.buttons["settingsDone"].tap()
        capture("full-reference-portrait-after-rotation", app: app)
    }

    @MainActor func testFullReferenceEditorControlsAndAppearance() throws {
        XCUIDevice.shared.orientation = .portrait
        let app = XCUIApplication(); app.launch()
        defer { app.terminate() }
        if app.buttons["tutorialSkip"].waitForExistence(timeout: 5) { app.buttons["tutorialSkip"].tap() }
        let selector = app.buttons["projectSelector"]
        XCTAssertTrue(selector.waitForExistence(timeout: 20)); selector.tap()
        let reference = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH 'project-project-' AND label CONTAINS %@", "tds6-reference.mp4")).firstMatch
        try XCTSkipUnless(reference.waitForExistence(timeout: 10), "Import full iPad reference into this simulator first")
        reference.tap()
        XCTAssertTrue(app.buttons["reviewCleanup"].waitForExistence(timeout: 45), app.debugDescription)
        let summary = app.descendants(matching: .any).matching(identifier: "exportSummary").firstMatch
        XCTAssertTrue(["Wide workspace", "Compact workspace"].contains(summary.value as? String ?? ""))
        capture("full-reference-portrait", app: app)
        for identifier in ["reviewCleanup", "reviewClips", "reviewServes"] {
            let button = app.buttons[identifier]
            reveal(button, in: app)
            if button.isEnabled {
                button.tap()
                XCTAssertTrue(app.staticTexts["currentRallyRange"].exists)
                if identifier == "reviewServes" {
                    let side = app.descendants(matching: .any).matching(identifier: "selectedServeSide").firstMatch
                    XCTAssertTrue(side.waitForExistence(timeout: 15))
                    XCTAssertFalse(side.buttons["Review"].exists)
                    if side.value as? String == "review" {
                        XCTAssertFalse(side.buttons["Near"].isSelected)
                        XCTAssertFalse(side.buttons["Far"].isSelected)
                    }
                }
                capture("full-reference-" + identifier, app: app)
            }
        }
        let play = app.buttons["playPause"]
        reveal(play, in: app)
        let seek = app.sliders["sourceSeek"], speed = app.buttons["playbackRate"]
        XCTAssertLessThan(seek.frame.midY, play.frame.midY)
        XCTAssertEqual(speed.frame.midY, play.frame.midY, accuracy: 3)
        XCTAssertEqual(speed.frame.width, play.frame.width, accuracy: 3)
        XCTAssertTrue(app.descendants(matching: .any).matching(identifier: "gameTimelineFirst").firstMatch.exists)
        XCTAssertTrue(app.descendants(matching: .any).matching(identifier: "gameTimelineSecond").firstMatch.exists)
        capture("full-reference-phone-transport", app: app)
        reveal(app.buttons["editorSettings"], in: app); app.buttons["editorSettings"].tap()
        XCTAssertTrue(app.buttons["settingsDone"].waitForExistence(timeout: 15))
        let before = app.sliders["beforePadding"], after = app.sliders["afterPadding"], gap = app.sliders["joinGap"]
        XCTAssertTrue(before.exists); XCTAssertTrue(after.exists); XCTAssertTrue(gap.exists)
        capture("full-reference-padding-and-joins", app: app)
        before.adjust(toNormalizedSliderPosition: 0)
        before.adjust(toNormalizedSliderPosition: 0.2)
        app.buttons["settingsDone"].tap()
        let export = app.buttons["exportTab"]
        reveal(export, in: app); export.tap()
        let video = app.buttons["exportVideo"]
        reveal(video, in: app)
        capture("full-reference-export", app: app)
        reveal(app.buttons["reviewTab"], in: app); app.buttons["reviewTab"].tap()
    }

    @MainActor private func capture(_ name: String, app: XCUIApplication) {
        let shot = XCTAttachment(screenshot: app.screenshot()); shot.name = name
        shot.lifetime = .keepAlways; add(shot)
        let tree = XCTAttachment(string: app.debugDescription); tree.name = name + "-hierarchy"
        tree.lifetime = .keepAlways; add(tree)
    }
    @MainActor private func scaleState(_ app: XCUIApplication) -> (percent: Double, custom: Bool) {
        (Double(app.staticTexts["uiScalePercent"].label.replacingOccurrences(of: "%", with: "")) ?? 100, app.buttons["uiScaleDefault"].isEnabled)
    }
    @MainActor private func restoreScale(_ state: (percent: Double, custom: Bool), app: XCUIApplication) {
        if state.custom { dragScale(app.sliders["uiScale"], target: state.percent, app: app) }
        else if app.buttons["uiScaleDefault"].isEnabled { app.buttons["uiScaleDefault"].tap() }
        XCTAssertEqual(scaleState(app).percent, state.percent)
        XCTAssertEqual(scaleState(app).custom, state.custom)
    }
    @MainActor private func dragScaleToEndpoint(_ slider: XCUIElement, larger: Bool, app: XCUIApplication) {
        dragScale(slider, target: larger ? 125 : 60, app: app)
    }
    @MainActor private func dragScale(_ slider: XCUIElement, target: Double, app: XCUIApplication) {
        // The accessible value is the actual UI percentage (60...125), not the
        // track's normalized 0...100 percentage assumed by XCTest.adjust.
        // Drag the observed thumb; layout changes apply on release.
        let percent = Double(app.staticTexts["uiScalePercent"].label.replacingOccurrences(of: "%", with: "")) ?? 100
        let fraction = min(1, max(0, (percent - 60) / 65))
        XCTAssertTrue(slider.isHittable)
        // Ask this runtime's UIKit for thumb geometry. iOS 26's thumb inset is
        // not half the accessibility frame height, especially at 60% scale.
        let scale = percent / 100
        let probe = UISlider(frame: CGRect(x: 0, y: 0, width: slider.frame.width / scale, height: 31))
        probe.minimumValue = 0; probe.maximumValue = 1; probe.value = Float(fraction)
        let track = probe.trackRect(forBounds: probe.bounds)
        let thumb = probe.thumbRect(forBounds: probe.bounds, trackRect: track, value: probe.value)
        let start = slider.coordinate(withNormalizedOffset: CGVector(dx: thumb.midX / probe.bounds.width, dy: 0.5))
        let targetThumb = probe.thumbRect(forBounds: probe.bounds, trackRect: track, value: Float((target - 60) / 65))
        let end = target == 60 || target == 125
            ? app.coordinate(withNormalizedOffset: CGVector(dx: (target == 125
                ? min(app.frame.maxX - 8, slider.frame.maxX + 16)
                : max(app.frame.minX + 8, slider.frame.minX - 16)) / app.frame.width,
                dy: (slider.frame.midY - app.frame.minY) / app.frame.height))
            : slider.coordinate(withNormalizedOffset: CGVector(dx: targetThumb.midX / probe.bounds.width, dy: 0.5))
        start.press(forDuration: 0.2, thenDragTo: end)
    }
    @MainActor private func reveal(_ element: XCUIElement, in app: XCUIApplication) {
        XCTAssertTrue(element.waitForExistence(timeout: 15), app.debugDescription)
        for _ in 0..<12 {
            if element.isHittable { break }
            let parent = app.scrollViews.containing(element.elementType, identifier: element.identifier)
                .allElementsBoundByIndex.filter { !$0.frame.intersection(app.frame).isEmpty }
                .min { $0.frame.width * $0.frame.height < $1.frame.width * $1.frame.height }
            guard let parent else { break }
            let area = parent.frame.intersection(app.frame)
            let above = element.frame.midY < area.midY
            let top = max(0.15, (area.minY - parent.frame.minY + 30) / parent.frame.height)
            let bottom = min(0.85, (area.maxY - parent.frame.minY - 30) / parent.frame.height)
            parent.coordinate(withNormalizedOffset: CGVector(dx: 0.98, dy: above ? top : bottom))
                .press(forDuration: 0.05, thenDragTo: parent.coordinate(withNormalizedOffset: CGVector(dx: 0.98, dy: above ? bottom : top)))
        }
        XCTAssertTrue(element.isHittable, element.debugDescription)
    }
}
