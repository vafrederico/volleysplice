import XCTest
@testable import VolleyCore

final class GuidedTourTests: XCTestCase {
    func testAndroidSetupHandsOffToEditorInOrder() {
        XCTAssertEqual(GuidedTourStep.setupSource.next(), .setupWindow)
        XCTAssertEqual(GuidedTourStep.setupWindow.next(), .setupCreate)
        XCTAssertEqual(GuidedTourStep.setupCreate.next(), .editorSettings)
    }
    func testAndroidEditorFollowsControlsTopToBottom() {
        let steps: [GuidedTourStep] = [.editorSettings, .editorExport, .editorScoreToggle, .editorScorePanel,
                                     .editorVideo, .editorReviewQueues, .editorOverview, .editorFocus, .editorMarking]
        for pair in zip(steps, steps.dropFirst()) { XCTAssertEqual(pair.0.next(), pair.1) }
        XCTAssertNil(GuidedTourStep.editorMarking.next())
    }
    func testDisabledScoreTrackingExplainsToggleAndSkipsPanel() {
        XCTAssertEqual(GuidedTourStep.editorExport.next(scoreTrackingEnabled: false), .editorScoreToggle)
        XCTAssertEqual(GuidedTourStep.editorScoreToggle.next(scoreTrackingEnabled: false), .editorVideo)
        var progress = GuidedTourProgress(stored: GuidedTourStep.editorScorePanel.rawValue)
        progress.enter(stage: .editor, scoreTrackingEnabled: false)
        XCTAssertEqual(progress.current(stage: .editor), .editorVideo)
    }
    func testActualSourceAndEditorRequiredBeforeSetupAdvances() {
        var progress = GuidedTourProgress(); progress.enter(stage: .setup)
        progress.advance(stage: .setup, sourceReady: false)
        XCTAssertEqual(progress.current(stage: .setup), .setupSource)
        progress.advance(stage: .setup, sourceReady: true)
        progress.advance(stage: .setup, sourceReady: true)
        progress.advance(stage: .setup, sourceReady: true)
        XCTAssertEqual(progress.current(stage: .setup), .setupCreate)
        progress.enter(stage: .editor)
        XCTAssertEqual(progress.current(stage: .editor), .editorSettings)
    }
    func testDismissalAndCompletionSurviveStageChangesUntilReplay() throws {
        for value in ["done", "dismissed"] {
            var progress = GuidedTourProgress(stored: value)
            progress = try JSONDecoder().decode(GuidedTourProgress.self, from: JSONEncoder().encode(progress))
            progress.enter(stage: .editor)
            XCTAssertNil(progress.current(stage: .editor))
            progress.restart(stage: .editor)
            XCTAssertEqual(progress.current(stage: .editor), .editorSettings)
            progress.restart(stage: .setup)
            XCTAssertEqual(progress.current(stage: .setup), .setupSource)
        }
    }
}
