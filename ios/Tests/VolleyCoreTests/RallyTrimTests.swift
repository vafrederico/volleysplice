import XCTest
@testable import VolleyCore

final class RallyTrimTests: XCTestCase {
    private let bounds = TimeRange(startMs: 0, endMs: 60_000)
    private func fixture() -> EditorDraft {
        var a = EditableCut(id: "a", coreStartMs: 10_000, coreEndMs: 15_000, confidence: 0.3)
        a.keepStartMs = 8_000; a.keepEndMs = 17_000
        var b = EditableCut(id: "b", coreStartMs: 22_000, coreEndMs: 25_000, confidence: 0.3)
        b.keepStartMs = 20_000; b.keepEndMs = 27_000
        var draft = EditorDraft(sourceRevision: "synthetic", cuts: [a, b])
        draft.reviewedCutIds = ["a", "b"]; draft.joinGapMs = 3_000
        return draft
    }
    private func trim(_ draft: EditorDraft, id: String, start: Bool, value: Int64, frame: Int64) throws -> RallyTrimResult {
        try XCTUnwrap(RallyReview.trimBoundary(draft, selectedCutIds: [id], isStart: start,
            valueMs: value, minimumDurationMs: 500, frameMs: frame, suggestions: [], bounds: bounds, durationMs: 60_000))
    }
    func testEndTrimKeepsAFrameInTheNewlyJoinedNextRally() throws {
        let result = try trim(fixture(), id: "a", start: false, value: 16_000, frame: 23_000)
        XCTAssertEqual(result.selectedGroup.cutIds, ["a", "b"])
        XCTAssertEqual(result.selectedGroup.keepEndMs, 27_000)
        XCTAssertEqual(result.frameMs, 23_000)
        XCTAssertEqual(result.presentation.groups.count, 1)
    }
    func testStartTrimKeepsAFrameInTheNewlyJoinedPreviousRally() throws {
        let result = try trim(fixture(), id: "b", start: true, value: 21_000, frame: 12_000)
        XCTAssertEqual(result.selectedGroup.cutIds, ["a", "b"])
        XCTAssertEqual(result.selectedGroup.keepStartMs, 8_000)
        XCTAssertEqual(result.frameMs, 12_000)
    }
    func testServeAndPendingReviewBoundariesStillPreventJoining() throws {
        var pending = fixture(); pending.reviewedCutIds.remove("b")
        let reviewResult = try trim(pending, id: "a", start: false, value: 16_000, frame: 23_000)
        XCTAssertEqual(reviewResult.presentation.groups.count, 2)
        XCTAssertEqual(reviewResult.frameMs, 18_000)
        var served = fixture()
        served.scoreTracking = try ScoreTracking(serveMarkers: [.init(id: "serve", timestampMs: 22_000, side: .near, rallyId: "b")]).jsonValue()
        let serveResult = try trim(served, id: "a", start: false, value: 16_000, frame: 23_000)
        XCTAssertEqual(serveResult.selectedGroup.cutIds, ["a"])
        XCTAssertEqual(serveResult.frameMs, 18_000)
    }
    func testFullEditorDetailKeepsShortCoreTargetsDistinct() {
        for (duration, padding): (Int64, Int64) in [(500, 10_000), (100, 2_000)] {
            let core = TimeRange(startMs: 25_000, endMs: 25_000 + duration)
            let kept = TimeRange(startMs: core.startMs - padding, endMs: core.endMs + padding)
            let scale = RallyTrimScale(core: core, bounds: bounds, width: 306)
            let handles = scale.handles(for: core)
            XCTAssertGreaterThanOrEqual(handles.endX - handles.startX, RallyTrimScale.minimumCenterGap)
            XCTAssertGreaterThan(scale.hiddenPadding(kept).beforeMs, 0)
            XCTAssertGreaterThan(scale.hiddenPadding(kept).afterMs, 0)
            XCTAssertEqual(scale.time(atX: scale.x(core.startMs)), core.startMs)
            XCTAssertEqual(scale.time(atX: scale.x(core.endMs)), core.endMs)
        }
    }
    func testDeckKeepsShortCoreHandlesAtTrueTimeEvenWithLargePadding() {
        for (duration, padding): (Int64, Int64) in [(500, 10_000), (100, 2_000)] {
            var cut = EditableCut(id: "short", coreStartMs: 25_000, coreEndMs: 25_000 + duration)
            cut.keepStartMs = cut.coreStartMs - padding; cut.keepEndMs = cut.coreEndMs + padding
            let rally = EditableRallyGroup(cuts: [cut])
            let core = TimeRange(startMs: cut.coreStartMs, endMs: cut.coreEndMs)
            let context = RallyReview.context(for: rally, rallies: [rally], effectiveIds: [cut.id], bounds: bounds)
            let scale = RallyTrimScale(core: core, bounds: bounds, width: 306, context: context)
            let handles = scale.handles(for: core, spreadForTouch: false)
            XCTAssertFalse(handles.spreadForTouch)
            XCTAssertLessThan(handles.endX - handles.startX, 10)
            XCTAssertEqual(handles.startX, scale.x(core.startMs), accuracy: 0.000001)
            XCTAssertEqual(handles.endX, scale.x(core.endMs), accuracy: 0.000001)
            XCTAssertEqual(scale.time(atX: handles.startX), core.startMs)
            XCTAssertEqual(scale.time(atX: handles.endX), core.endMs)
            XCTAssertLessThanOrEqual(context.startMs, cut.keepStartMs)
            XCTAssertGreaterThanOrEqual(context.endMs, cut.keepEndMs)
        }
    }
    func testNarrowTracksKeepEachTargetInsideItsOwnRegion() {
        let core = TimeRange(startMs: 10_000, endMs: 10_100)
        for width in [88.0, 120, 240, 306, 390] {
            let scale = RallyTrimScale(core: core, bounds: bounds, width: width)
            let handles = scale.handles(for: core)
            XCTAssertGreaterThanOrEqual(handles.startX - 22, -0.001)
            XCTAssertLessThanOrEqual(handles.endX + 22, width + 0.001)
            XCTAssertGreaterThanOrEqual(handles.endX - handles.startX, 44)
        }
    }
    func testFrozenScaleSpreadsHandlesAndKeepsExactTicksWhenCoreShrinks() {
        let scale = RallyTrimScale(core: .init(startMs: 10_000, endMs: 15_000), bounds: bounds, width: 306)
        let short = TimeRange(startMs: 14_900, endMs: 15_000)
        let handles = scale.handles(for: short)
        XCTAssertTrue(handles.spreadForTouch)
        XCTAssertEqual(handles.endX - handles.startX, 52, accuracy: 0.001)
        XCTAssertLessThan(scale.x(short.endMs) - scale.x(short.startMs), 10)
    }
    func testDragUsesActualEdgeAndFrozenScaleRatherThanTouchTarget() {
        let core = TimeRange(startMs: 10_000, endMs: 10_100)
        let scale = RallyTrimScale(core: core, bounds: bounds, width: 306)
        for start in [true, false] {
            let edge = start ? core.startMs : core.endMs
            let drag = RallyTrimDrag(scale: scale, initialValueMs: edge, isStart: start)
            XCTAssertEqual(drag.value(translationX: 0), edge)
            let translation = 50.0 / Double(scale.window.endMs - scale.window.startMs) * scale.usableWidth
            XCTAssertEqual(drag.value(translationX: translation), edge + 50)
            XCTAssertEqual(drag.scale, scale)
        }
    }
    func testFocusedInspectionRetainsSelectionButBrowserSourceSeekingFollows() {
        let presentation = EditorPresentation(draft: fixture(), suggestions: [], bounds: bounds, durationMs: 60_000)
        var navigation = RallyDeckNavigation(); navigation.selectedId = "a"
        navigation.sourceSeek(time: 23_000, rallies: presentation.groups)
        XCTAssertEqual(navigation.selectedId, "a")
        XCTAssertEqual(navigation.followRequest, 0)
        navigation.browsing = true
        navigation.sourceSeek(time: 23_000, rallies: presentation.groups)
        XCTAssertEqual(navigation.selectedId, "b")
        XCTAssertEqual(navigation.listCenterId, "b")
        XCTAssertEqual(navigation.followRequest, 1)
    }
    func testFocusedPlaybackRestoresItsEndBoundaryAfterTrimming() {
        let retained = TimeRange(startMs: 8_000, endMs: 27_000)
        let inside = RallyReview.focusedPlayback(frameMs: 23_000, retained: retained)
        XCTAssertEqual(inside.positionMs, 23_000); XCTAssertEqual(inside.endMs, 27_000)
        XCTAssertEqual(RallyReview.focusedPlayback(frameMs: 27_000, retained: retained).positionMs, 8_000)
        XCTAssertEqual(RallyReview.focusedPlayback(frameMs: 7_000, retained: retained).positionMs, 8_000)
    }
    func testSharedPlaybackRatesIncludeFastAndFractionalValues() throws {
        XCTAssertTrue(Set([Float(0.25), 0.5, 1, 2, 4, 8]).isSubset(of: Set(EditorPlaybackPolicy.rates)))
        XCTAssertEqual(Set(EditorPlaybackPolicy.rates).count, EditorPlaybackPolicy.rates.count)
        var draft = fixture(); draft.playbackRate = 8
        try draft.validate(durationMs: 60_000)
        XCTAssertEqual(try JSONDecoder().decode(EditorDraft.self, from: JSONEncoder().encode(draft)).playbackRate, 8)
    }
}
