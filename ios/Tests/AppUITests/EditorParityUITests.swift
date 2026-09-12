import XCTest
import UIKit

/// Interaction/layout checks only. --parity-editor installs a documented
/// synthetic draft over a repeated Android overlay fixture; it runs no inference.
final class EditorParityUITests: XCTestCase {
    @MainActor func testMoreMenuQueueAndConfirmedProjectDeletion() {
        let app = launchFixture(); defer { app.terminate() }
        app.buttons["editorMore"].tap()
        XCTAssertFalse(app.buttons["backToProjects"].exists)
        XCTAssertFalse(app.buttons["openSavedProjects"].exists)
        app.buttons["editorProcessingQueue"].tap()
        XCTAssertTrue(app.navigationBars["Processing queue"].waitForExistence(timeout: 15))
        app.buttons["queueDone"].tap()
        app.buttons["editorMore"].tap(); app.buttons["deleteCurrentProject"].tap()
        XCTAssertTrue(app.alerts["Delete current project?"].waitForExistence(timeout: 15))
        app.alerts.buttons["Cancel"].tap()
        XCTAssertTrue(app.buttons["reviewCleanup"].exists)
        app.buttons["editorMore"].tap(); app.buttons["deleteCurrentProject"].tap()
        app.alerts.buttons["Delete"].tap()
        XCTAssertTrue(app.buttons["chooseRecording"].waitForExistence(timeout: 15))
        XCTAssertFalse(app.buttons["reviewCleanup"].exists)
        XCTAssertEqual(app.state, .runningForeground)
        capture("deleted-project-returns-to-setup", app: app)
    }
    override func setUpWithError() throws { continueAfterFailure = false }

    @MainActor private func launchFixture(extra: [String] = []) -> XCUIApplication {
        XCUIDevice.shared.orientation = .portrait
        let app = XCUIApplication()
        app.launchArguments += ["--parity-editor"] + extra
        app.launch()
        let skip = app.buttons["tutorialSkip"]
        if skip.waitForExistence(timeout: 5) { skip.tap() }
        XCTAssertTrue(app.buttons["reviewCleanup"].waitForExistence(timeout: 30), app.debugDescription)
        if skip.exists { skip.tap() }
        return app
    }
    @MainActor private func wait(_ element: XCUIElement, predicate: String, _ arguments: Any..., file: StaticString = #filePath, line: UInt = #line) {
        let expectation = XCTNSPredicateExpectation(predicate: NSPredicate(format: predicate, argumentArray: arguments), object: element)
        XCTAssertEqual(XCTWaiter.wait(for: [expectation], timeout: 15), .completed, element.debugDescription, file: file, line: line)
    }
    /// Reveal an action through its observed scroll ancestor. Never used to make
    /// the reviewed marker visible: that remains an app-owned navigation assertion.
    @MainActor private func reveal(_ element: XCUIElement, in app: XCUIApplication) {
        XCTAssertTrue(element.waitForExistence(timeout: 15), app.debugDescription)
        for _ in 0..<12 where !element.isHittable {
            let match = element.identifier.isEmpty ? element.label : element.identifier
            let parents = app.scrollViews.containing(element.elementType, identifier: match).allElementsBoundByIndex
            let parent = parents.filter { !$0.frame.intersection(app.frame).isEmpty }.min {
                $0.frame.width * $0.frame.height < $1.frame.width * $1.frame.height
            }
            guard let parent else { break }
            let viewport = parent.frame.intersection(app.frame)
            // A fully visible but non-hittable control is an interaction bug,
            // not a reason to keep scrolling until a different state appears.
            if viewport.insetBy(dx: 1, dy: 24).contains(element.frame) { break }
            let above = element.frame.midY < viewport.midY
            // Keep the gesture in the container's trailing inset, outside nested
            // marker lists and the video/slider surface. Direction follows the
            // target's actual frame, allowing return to queues above a cut card.
            let top = max(0.15, (viewport.minY - parent.frame.minY + 30) / parent.frame.height)
            let bottom = min(0.85, (viewport.maxY - parent.frame.minY - 30) / parent.frame.height)
            guard bottom > top else { break }
            let from = parent.coordinate(withNormalizedOffset: CGVector(dx: 0.985, dy: above ? top : bottom))
            let to = parent.coordinate(withNormalizedOffset: CGVector(dx: 0.985, dy: above ? bottom : top))
            from.press(forDuration: 0.05, thenDragTo: to)
        }
        XCTAssertTrue(element.isHittable, element.debugDescription)
    }
    @MainActor private func capture(_ name: String, app: XCUIApplication) {
        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = "synthetic-editor-" + name; attachment.lifetime = .keepAlways; add(attachment)
    }

    @MainActor func testStartNewVideoFromEditorReturnsToUsableSetup() {
        let app = launchFixture(); defer { app.terminate() }
        let orientations: [UIDeviceOrientation] = app.frame.width > 600 ? [.landscapeLeft, .portrait] : [.portrait]
        for orientation in orientations {
            XCUIDevice.shared.orientation = orientation
            let selector = app.buttons["projectSelector"]
            XCTAssertTrue(selector.waitForExistence(timeout: 15))
            XCTAssertTrue(selector.isHittable)
            selector.tap()
            capture("project-menu-\(orientation.rawValue)", app: app)
            let startNew = app.buttons["startNewVideo"]
            XCTAssertTrue(startNew.waitForExistence(timeout: 15), app.debugDescription)
            startNew.tap()
            let choose = app.buttons["chooseRecording"]
            XCTAssertTrue(choose.waitForExistence(timeout: 15), app.debugDescription)
            XCTAssertEqual(app.state, .runningForeground)
            XCTAssertFalse(app.buttons["reviewCleanup"].exists)
            // Also exercise the freshly reset view, so a delayed teardown crash
            // cannot be mistaken for a successful navigation.
            choose.tap()
            XCTAssertTrue(app.buttons["Camera Roll"].waitForExistence(timeout: 15))
            // iPad renders the source menu as a popover without a Cancel row.
            // Exercise the Files route, then dismiss its native document picker.
            app.buttons["Files"].tap()
            let cancel = app.buttons["Cancel"].firstMatch
            XCTAssertTrue(cancel.waitForExistence(timeout: 30), app.debugDescription)
            cancel.tap()
            XCTAssertTrue(choose.waitForExistence(timeout: 15))
            XCTAssertTrue(choose.isHittable)
            capture("start-new-video-\(orientation.rawValue)", app: app)
            if orientation != orientations.last {
                // Launch a new fixture process for the other layout without
                // requiring the reset screen to retain a video selection.
                app.terminate(); app.launch()
                XCTAssertTrue(app.buttons["reviewCleanup"].waitForExistence(timeout: 30))
            }
        }
    }

    @MainActor func testStorageIsAccessibleFromEditorAndSetupSettings() {
        let app = launchFixture(); defer { app.terminate() }
        app.buttons["editorMore"].tap()
        app.buttons["manageStorage"].tap()
        XCTAssertTrue(app.buttons["appStorageDone"].waitForExistence(timeout: 20))
        XCTAssertTrue(app.descendants(matching: .any)["appStorageTotal"].exists)
        capture("storage-editor", app: app)
        app.buttons["appStorageDone"].tap()
        app.buttons["projectSelector"].tap(); app.buttons["startNewVideo"].tap()
        XCTAssertTrue(app.buttons["setupSettings"].waitForExistence(timeout: 15))
        app.buttons["setupSettings"].tap(); app.buttons["manageStorage"].tap()
        XCTAssertTrue(app.buttons["appStorageDone"].waitForExistence(timeout: 20))
        capture("storage-setup-settings", app: app)
        app.buttons["appStorageDone"].tap()
        XCTAssertTrue(app.navigationBars["Settings"].waitForExistence(timeout: 15))
    }

    @MainActor func testQueueEntryCanBeRemovedWithoutPausingEverything() {
        let app = launchFixture(extra: ["--parity-queue"]); defer { app.terminate() }
        app.buttons["processingQueueBar"].tap()
        let remove = app.buttons["removeJob-00000000-0000-4000-8000-000000000023"]
        XCTAssertTrue(remove.waitForExistence(timeout: 15))
        remove.tap()
        wait(remove, predicate: "exists == false")
        XCTAssertTrue(app.staticTexts["Queued analysis and exports appear here."].exists)
        capture("queue-removed", app: app)
        app.buttons["queueDone"].tap()
    }

    @MainActor func testCleanupDecisionsResolveUnreviewedSuggestions() {
        let app = launchFixture(); defer { app.terminate() }
        let queue = app.buttons["reviewCleanup"]
        // C-ignored is entirely outside reviewable time, leaving C01 and C02.
        wait(queue, predicate: "label == %@", "Review cleanup 2")
        reveal(queue, in: app); queue.tap()
        let selected = app.descendants(matching: .any).matching(identifier: "cleanupSelection").firstMatch
        wait(selected, predicate: "value == %@", "C01")
        let keep = app.buttons["keepRally"]
        reveal(keep, in: app)
        wait(keep, predicate: "value == %@", "automatically excluded")
        keep.tap()
        wait(queue, predicate: "label == %@", "Review cleanup 1")
        wait(keep, predicate: "value == %@", "included")
        capture("cleanup-kept", app: app)
        reveal(queue, in: app); queue.tap()
        wait(selected, predicate: "value == %@", "C02")
        let remove = app.buttons["removeRally"]
        reveal(remove, in: app); remove.tap()
        wait(queue, predicate: "label == %@ AND enabled == false", "Review cleanup 0")
        wait(keep, predicate: "value == %@", "removed")
        capture("cleanup-explicitly-removed", app: app)
    }

    @MainActor func testSettingsHaveNoUndoOrPerSuggestionList() {
        let app = launchFixture(); defer { app.terminate() }
        XCTAssertFalse(app.buttons["undoEdit"].exists)
        let settings = app.buttons["editorSettings"]
        reveal(settings, in: app); settings.tap()
        XCTAssertTrue(app.buttons["settingsDone"].waitForExistence(timeout: 15))
        XCTAssertFalse(app.descendants(matching: .any).matching(NSPredicate(format: "identifier BEGINSWITH %@", "suppression-")).firstMatch.exists)
        XCTAssertFalse(app.descendants(matching: .any).matching(identifier: "ignoreReason").firstMatch.exists)
        XCTAssertFalse(app.descendants(matching: .any).matching(identifier: "suppressionBehavior").firstMatch.exists)
        let policy = app.descendants(matching: .any).matching(identifier: "suppressionPolicy").firstMatch
        // Form rows are lazy and the policy can begin below the sheet viewport.
        for _ in 0..<4 where !policy.exists {
            if app.collectionViews.firstMatch.exists { app.collectionViews.firstMatch.swipeUp() }
            else if app.scrollViews.firstMatch.exists { app.scrollViews.firstMatch.swipeUp() }
            else { app.swipeUp() }
        }
        XCTAssertTrue(policy.exists, app.debugDescription)
        capture("settings", app: app)
        app.buttons["settingsDone"].tap()
        XCTAssertFalse(app.buttons["undoEdit"].exists)
    }

    @MainActor func testServeReviewSelectsAndRevealsMarker() {
        let app = launchFixture(); defer { app.terminate() }
        let queue = app.buttons["reviewServes"]
        wait(queue, predicate: "label == %@", "Review serves 1")
        reveal(queue, in: app); queue.tap()
        // S09 starts below the marker list's first 220pt viewport. This asserts
        // automatic reveal rather than manually scrolling to make it pass.
        let marker = app.buttons["serve-S09"]
        wait(marker, predicate: "exists == true AND hittable == true")
        let side = app.descendants(matching: .any).matching(identifier: "selectedServeSide").firstMatch
        XCTAssertTrue(side.waitForExistence(timeout: 15))
        // Desktop review must expose the correction, not only the highlighted
        // marker. Do not manually scroll the sidebar to make this pass.
        let summary = app.descendants(matching: .any).matching(identifier: "exportSummary").firstMatch
        let near = side.buttons["Near"]
        if (summary.value as? String) == "Wide workspace" {
            wait(near, predicate: "hittable == true")
            let ignored = app.buttons["ignorePreviousPoint"]
            wait(ignored, predicate: "hittable == true")
            XCTAssertLessThan(abs(near.frame.midY - ignored.frame.midY), 8)
            XCTAssertFalse(side.buttons["Review"].exists)
            XCTAssertFalse(near.isSelected)
            XCTAssertFalse(side.buttons["Far"].isSelected)
            XCTAssertFalse(app.buttons["removeServe"].exists)
        } else {
            XCTAssertFalse(side.buttons["Review"].exists)
            XCTAssertFalse(near.isSelected)
            XCTAssertFalse(side.buttons["Far"].isSelected)
        }
        capture("reviewed-serve-selected", app: app)
        reveal(near, in: app); near.tap()
        wait(queue, predicate: "label == %@ AND enabled == false", "Review serves 0")
    }

    @MainActor func testDesktopClipRegisterUsesKeptBoundsAndIndependentInclusion() throws {
        let app = launchFixture(); defer { app.terminate() }
        let summary = app.descendants(matching: .any).matching(identifier: "exportSummary").firstMatch
        try XCTSkipUnless((summary.value as? String) == "Wide workspace", "Desktop register contract")
        let row = app.buttons["clip-R001"]
        reveal(row, in: app)
        XCTAssertTrue(row.label.contains("0:00.50"), row.debugDescription)
        XCTAssertTrue(row.label.contains("0:03.00"), row.debugDescription)
        XCTAssertEqual(row.value as? String, "Ready")
        row.tap()
        let clock = app.staticTexts["playheadTime"]
        wait(clock, predicate: "label BEGINSWITH %@", "0:00 /")
        let range = app.staticTexts["currentRallyRange"]
        XCTAssertTrue(range.label.contains("0:00.50"))
        XCTAssertTrue(range.label.contains("0:03.00"))
        let cleanupBefore = app.buttons["reviewCleanup"].label
        let clipsBefore = app.buttons["reviewClips"].label
        let toggle = app.buttons["toggleClip-R001"]
        reveal(toggle, in: app)
        XCTAssertEqual(toggle.label, "On")
        toggle.tap()
        wait(toggle, predicate: "label == %@", "Off")
        wait(row, predicate: "value == %@", "Left out")
        XCTAssertEqual(app.buttons["reviewCleanup"].label, cleanupBefore)
        XCTAssertEqual(app.buttons["reviewClips"].label, clipsBefore)
        XCTAssertTrue(clock.label.hasPrefix("0:00 /"))
        toggle.tap()
        wait(toggle, predicate: "label == %@", "On")
        wait(row, predicate: "value == %@", "Ready")
        capture("desktop-ledger-kept-bounds-and-toggle", app: app)
    }

    @MainActor func testResizeHandlesFollowTouchAndReturnToOriginalPosition() throws {
        let app = launchFixture(); defer { app.terminate() }
        let summary = app.descendants(matching: .any).matching(identifier: "exportSummary").firstMatch
        try XCTSkipUnless((summary.value as? String) == "Wide workspace", "Desktop resizing")
        for identifier in ["resizeLeftSidebar", "resizeRightSidebar", "resizeVideo"] {
            let handle = app.descendants(matching: .any).matching(identifier: identifier).firstMatch
            reveal(handle, in: app)
            let original = handle.frame
            let vertical = identifier == "resizeVideo"
            let delta: CGFloat = vertical ? (original.midY > 600 ? -50 : 50)
                : identifier == "resizeLeftSidebar" ? (original.midX > 300 ? -40 : 40)
                : (app.frame.maxX - original.midX > 300 ? 40 : -40)
            let start = handle.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
            start.press(forDuration: 0.05, thenDragTo: start.withOffset(CGVector(dx: vertical ? 0 : delta, dy: vertical ? delta : 0)))
            let changed = handle.frame
            let movement = vertical ? changed.midY - original.midY : changed.midX - original.midX
            XCTAssertEqual(movement, delta, accuracy: 5, identifier)
            let end = handle.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
            end.press(forDuration: 0.05, thenDragTo: end.withOffset(CGVector(dx: vertical ? 0 : -movement, dy: vertical ? -movement : 0)))
            XCTAssertEqual(vertical ? handle.frame.midY : handle.frame.midX,
                           vertical ? original.midY : original.midX, accuracy: 5, identifier)
        }
        capture("resize-handles-follow-touch", app: app)
    }

    @MainActor func testDesktopTransportAndLegendInBothOrientations() throws {
        let app = launchFixture(); defer { app.terminate() }
        // P25 requests phone portrait. The VM's landscape framebuffer/activation
        // coordinates are unreliable; iPad retains its structural rotation check.
        let orientations: [UIDeviceOrientation] = app.frame.width > 600 ? [.landscapeLeft, .portrait] : [.portrait]
        for orientation in orientations {
            XCUIDevice.shared.orientation = orientation
            let play = app.buttons["playPause"]
            XCTAssertTrue(play.waitForExistence(timeout: 15))
            let seek = app.sliders["sourceSeek"]
            XCTAssertTrue(seek.exists)
            let summary = app.descendants(matching: .any).matching(identifier: "exportSummary").firstMatch
            if (summary.value as? String) == "Wide workspace" {
                let time = app.staticTexts["playheadTime"]
                XCTAssertLessThan(abs(seek.frame.midY - play.frame.midY), 8)
                XCTAssertLessThan(abs(time.frame.midY - play.frame.midY), 8)
            }
            XCTAssertTrue(app.descendants(matching: .any).matching(identifier: "timelineLegend").firstMatch.exists)
            XCTAssertFalse(app.buttons["undoEdit"].exists)
            reveal(play, in: app)
            capture(orientation == .portrait ? "portrait" : "landscape", app: app)
        }
    }

    @MainActor func testVideoExportAsksForDestinationBeforeWriting() {
        let app = launchFixture(); defer { app.terminate() }
        let exportStage = app.buttons["exportTab"]
        reveal(exportStage, in: app); exportStage.tap()
        let export = app.buttons["exportVideo"]
        reveal(export, in: app); export.tap()
        XCTAssertTrue(app.buttons["Camera Roll"].waitForExistence(timeout: 15), app.debugDescription)
        XCTAssertTrue(app.buttons["Files"].exists)
        capture("export-destinations", app: app)
        // Stop at destination choice: this is not a Photos permission, encoder,
        // file-provider, or decoded-media correctness test.
    }

    @MainActor func testCompactPortraitFollowsAndroidSectionOrder() throws {
        let app = launchFixture(); defer { app.terminate() }
        let summary = app.descendants(matching: .any).matching(identifier: "exportSummary").firstMatch
        try XCTSkipUnless((summary.value as? String) == "Compact workspace", "Compact portrait organization is exercised on iPhone; iPad portrait retains the Android desktop breakpoint.")

        let project = app.buttons["projectSelector"]
        let review = app.buttons["reviewTab"]
        let nowReviewing = app.staticTexts["NOW REVIEWING"]
        let finalVideo = app.staticTexts["YOUR FINAL VIDEO"]
        XCTAssertTrue(project.isHittable)
        XCTAssertTrue(app.buttons["editorSettings"].isHittable)
        XCTAssertTrue(review.isHittable)
        XCTAssertLessThan(project.frame.midY, review.frame.midY)
        XCTAssertLessThan(review.frame.midY, nowReviewing.frame.midY)
        XCTAssertLessThan(nowReviewing.frame.midY, finalVideo.frame.midY)
        XCTAssertTrue(app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH %@", "VIDEO EDITOR")).firstMatch.exists)
        capture("compact-portrait-header-and-summary", app: app)

        let export = app.buttons["exportVideo"]
        reveal(export, in: app)
        XCTAssertEqual(export.label, "Save final video")
        XCTAssertTrue(app.buttons["youtubeChaptersButton"].exists)
        XCTAssertTrue(app.buttons["projectOptions"].exists)
        capture("compact-portrait-final-video", app: app)

        let play = app.buttons["playPause"]
        reveal(play, in: app)
        let cleanup = app.buttons["reviewCleanup"]
        XCTAssertLessThan(play.frame.midY, cleanup.frame.midY)
        XCTAssertTrue(app.sliders["sourceSeek"].exists)
        XCTAssertFalse(app.buttons["undoEdit"].exists)
        capture("compact-portrait-player-and-queues", app: app)
    }
}
