import XCTest

/// Verifies folded (phone UI) and unfolded (iPad UI) layouts on foldable
/// displays such as iPhone Duo without assuming device-side orientation APIs
/// can change the simulator hinge posture in-process.
final class IPhoneDuoUITests: XCTestCase {
    override func setUpWithError() throws { continueAfterFailure = false }

    @MainActor private func launchFixture(stage: String, openSettings: Bool = false) -> XCUIApplication {
        let app = XCUIApplication()
        app.launchArguments += ["--parity-editor", "--parity-skip-tour", "--parity-stage=\(stage)"]
        if openSettings { app.launchArguments.append("--parity-settings") }
        app.launch()
        if !openSettings {
            XCTAssertTrue(app.buttons["editorSettings"].waitForExistence(timeout: 30), app.debugDescription)
        } else {
            XCTAssertTrue(app.navigationBars["Editor settings"].waitForExistence(timeout: 30), app.debugDescription)
        }
        return app
    }

    @MainActor private func capture(_ name: String, app: XCUIApplication) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = "iphone-duo-" + name
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    @MainActor private func assertScalePercent(_ expected: String, app: XCUIApplication) {
        let percent = app.staticTexts["uiScalePercent"]
        for _ in 0..<4 where !percent.exists {
            if app.collectionViews.firstMatch.exists { app.collectionViews.firstMatch.swipeUp() }
            else if app.tables.firstMatch.exists { app.tables.firstMatch.swipeUp() }
            else { app.swipeUp() }
        }
        XCTAssertTrue(percent.waitForExistence(timeout: 10), app.debugDescription)
        XCTAssertEqual(percent.label, expected, app.debugDescription)
    }

    @MainActor func testFoldedPortraitUsesPhoneInterface() throws {
        let rallies = launchFixture(stage: "rallies"); defer { rallies.terminate() }
        let frame = rallies.frame
        try XCTSkipUnless(frame.width < 560 && frame.height > frame.width,
                          "Folded portrait phone viewport required; got \(Int(frame.width))x\(Int(frame.height))")
        XCTAssertTrue(rallies.staticTexts["deckReviewCount"].waitForExistence(timeout: 15))
        XCTAssertFalse(rallies.descendants(matching: .any)["deckVideoDivider"].exists)
        XCTAssertFalse(rallies.descendants(matching: .any)["deckEditingToolsDisclosure"].exists)
        capture("folded-portrait-rallies", app: rallies)
        rallies.terminate()

        let editor = launchFixture(stage: "editor"); defer { editor.terminate() }
        let summary = editor.descendants(matching: .any).matching(identifier: "exportSummary").firstMatch
        XCTAssertTrue(summary.waitForExistence(timeout: 15), editor.debugDescription)
        XCTAssertEqual(summary.value as? String, "Compact workspace")
        XCTAssertFalse(editor.descendants(matching: .any)["eventsSidebar"].exists)
        XCTAssertFalse(editor.descendants(matching: .any)["awardedCount"].exists)
        XCTAssertFalse(editor.descendants(matching: .any)["markerCount"].exists)
        XCTAssertFalse(editor.buttons["nudgeStartEarlier"].exists)
        capture("folded-portrait-editor", app: editor)
        editor.terminate()

        let settings = launchFixture(stage: "editor", openSettings: true); defer { settings.terminate() }
        assertScalePercent("100%", app: settings)
    }

    @MainActor func testFoldedLandscapeUsesPhoneLandscapeInterface() throws {
        let rallies = launchFixture(stage: "rallies"); defer { rallies.terminate() }
        let frame = rallies.frame
        try XCTSkipUnless(frame.width > frame.height && frame.height < 560,
                          "Folded landscape phone viewport required; got \(Int(frame.width))x\(Int(frame.height))")
        XCTAssertTrue(rallies.staticTexts["deckReviewCount"].waitForExistence(timeout: 15))
        XCTAssertFalse(rallies.descendants(matching: .any)["deckVideoDivider"].exists)
        XCTAssertFalse(rallies.descendants(matching: .any)["deckEditingToolsDisclosure"].exists)
        capture("folded-landscape-rallies", app: rallies)
        rallies.terminate()

        let editor = launchFixture(stage: "editor"); defer { editor.terminate() }
        let summary = editor.descendants(matching: .any).matching(identifier: "exportSummary").firstMatch
        XCTAssertTrue(summary.waitForExistence(timeout: 15), editor.debugDescription)
        XCTAssertEqual(summary.value as? String, "Wide workspace")
        XCTAssertTrue(editor.descendants(matching: .any)["eventsSidebar"].exists)
        XCTAssertTrue(editor.descendants(matching: .any)["rallySidebar"].exists)
        XCTAssertFalse(editor.sliders["sourceSeek"].exists, "Phone landscape keeps the inline resize bar instead of sourceSeek")
        XCTAssertFalse(editor.descendants(matching: .any)["awardedCount"].exists)
        XCTAssertFalse(editor.descendants(matching: .any)["markerCount"].exists)
        capture("folded-landscape-editor", app: editor)
        editor.terminate()

        let settings = launchFixture(stage: "editor", openSettings: true); defer { settings.terminate() }
        assertScalePercent("75%", app: settings)
    }

    @MainActor func testUnfoldedLandscapeUsesIPadInterface() throws {
        let rallies = launchFixture(stage: "rallies"); defer { rallies.terminate() }
        let frame = rallies.frame
        try XCTSkipUnless(frame.width >= 840 && frame.height >= 560,
                          "Unfolded landscape / iPad viewport required; got \(Int(frame.width))x\(Int(frame.height))")
        XCTAssertTrue(rallies.staticTexts["deckReviewCount"].waitForExistence(timeout: 15))
        XCTAssertTrue(rallies.descendants(matching: .any)["deckVideoDivider"].waitForExistence(timeout: 10), rallies.debugDescription)
        XCTAssertTrue(rallies.descendants(matching: .any)["deckEditingToolsDisclosure"].exists, rallies.debugDescription)
        XCTAssertFalse(rallies.descendants(matching: .any)["deckEditingTools"].exists)
        capture("unfolded-landscape-rallies", app: rallies)
        rallies.terminate()

        let editor = launchFixture(stage: "editor"); defer { editor.terminate() }
        let summary = editor.descendants(matching: .any).matching(identifier: "exportSummary").firstMatch
        XCTAssertTrue(summary.waitForExistence(timeout: 15), editor.debugDescription)
        XCTAssertEqual(summary.value as? String, "Wide workspace")
        XCTAssertTrue(editor.descendants(matching: .any)["eventsSidebar"].exists)
        XCTAssertTrue(editor.descendants(matching: .any)["rallySidebar"].exists)
        XCTAssertTrue(editor.descendants(matching: .any)["resizeLeftSidebar"].exists)
        XCTAssertTrue(editor.descendants(matching: .any)["resizeRightSidebar"].exists)
        XCTAssertTrue(editor.sliders["sourceSeek"].exists, "Unfolded iPhone Duo uses iPad transport with sourceSeek")
        XCTAssertTrue(editor.descendants(matching: .any)["awardedCount"].exists)
        XCTAssertTrue(editor.descendants(matching: .any)["markerCount"].exists)
        capture("unfolded-landscape-editor", app: editor)
        editor.terminate()

        let export = launchFixture(stage: "export"); defer { export.terminate() }
        let video = export.buttons["exportVideo"]
        let project = export.buttons["exportProject"]
        XCTAssertTrue(video.waitForExistence(timeout: 15), export.debugDescription)
        XCTAssertTrue(project.exists, export.debugDescription)
        XCTAssertTrue(export.buttons["exportChapters"].exists, export.debugDescription)
        XCTAssertLessThan(video.frame.midX, project.frame.midX, "Unfolded iPhone Duo export uses multi-column iPad layout")
        capture("unfolded-landscape-export", app: export)
        export.terminate()

        let settings = launchFixture(stage: "editor", openSettings: true); defer { settings.terminate() }
        assertScalePercent("100%", app: settings)
    }

    @MainActor func testUnfoldedPortraitUsesIPadPortraitInterface() throws {
        let editor = launchFixture(stage: "editor"); defer { editor.terminate() }
        let frame = editor.frame
        try XCTSkipUnless(frame.height > frame.width && frame.width >= 560 && frame.width < 840,
                          "Unfolded portrait tablet viewport required; got \(Int(frame.width))x\(Int(frame.height))")
        let summary = editor.descendants(matching: .any).matching(identifier: "exportSummary").firstMatch
        XCTAssertTrue(summary.waitForExistence(timeout: 15), editor.debugDescription)
        XCTAssertEqual(summary.value as? String, "Compact workspace")
        XCTAssertTrue(editor.descendants(matching: .any)["awardedCount"].exists)
        XCTAssertTrue(editor.descendants(matching: .any)["markerCount"].exists)
        XCTAssertTrue(editor.buttons["nudgeStartEarlier"].exists)
        capture("unfolded-portrait-editor", app: editor)
        editor.terminate()

        let settings = launchFixture(stage: "editor", openSettings: true); defer { settings.terminate() }
        assertScalePercent("100%", app: settings)
    }
}
