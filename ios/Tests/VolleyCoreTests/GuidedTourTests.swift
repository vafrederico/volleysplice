import XCTest
@testable import VolleyCore

final class GuidedTourTests: XCTestCase {
    func testSetupHandsOffToRalliesInOrder() {
        XCTAssertEqual(GuidedTourStep.setupSource.next(), .setupWindow)
        XCTAssertEqual(GuidedTourStep.setupWindow.next(), .setupCreate)
        XCTAssertEqual(GuidedTourStep.setupCreate.next(), .deckVideo)
    }
    func testCourtsideTourFollowsActualWorkflow() {
        let steps = GuidedTourStep.visibleSteps(stage: .editor, scoreTrackingEnabled: true)
        XCTAssertEqual(steps, [.deckVideo, .deckTrim, .deckTagging, .deckList, .deckAdd, .deckFullEditor, .deckExport])
        for pair in zip(steps, steps.dropFirst()) { XCTAssertEqual(pair.0.next(), pair.1) }
        XCTAssertNil(GuidedTourStep.deckExport.next())
        XCTAssertEqual(GuidedTourStep.deckFullEditor.target, "editor-video")
    }
    func testLegacyProgressMigratesButDismissalDoesNotRestart() {
        for step in [GuidedTourStep.editorSettings, .editorScorePanel, .editorMarking] {
            var progress = GuidedTourProgress(stored: step.rawValue)
            progress.enter(stage: .editor, scoreTrackingEnabled: false)
            XCTAssertEqual(progress.current(stage: .editor), .deckVideo)
        }
        XCTAssertEqual(GuidedTourStep.visibleSteps(stage: .editor, scoreTrackingEnabled: false),
                       GuidedTourStep.visibleSteps(stage: .editor, scoreTrackingEnabled: true))
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
        XCTAssertEqual(progress.current(stage: .editor), .deckVideo)
    }
    func testDismissalAndCompletionSurviveStageChangesUntilReplay() throws {
        for value in ["done", "dismissed"] {
            var progress = GuidedTourProgress(stored: value)
            progress = try JSONDecoder().decode(GuidedTourProgress.self, from: JSONEncoder().encode(progress))
            progress.enter(stage: .editor)
            XCTAssertNil(progress.current(stage: .editor))
            progress.restart(stage: .editor)
            XCTAssertEqual(progress.current(stage: .editor), .deckVideo)
            progress.restart(stage: .setup)
            XCTAssertEqual(progress.current(stage: .setup), .setupSource)
        }
    }
}
