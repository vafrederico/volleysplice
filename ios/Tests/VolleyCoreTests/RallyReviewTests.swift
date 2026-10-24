import XCTest
@testable import VolleyCore

final class RallyReviewTests: XCTestCase {
    private let bounds = TimeRange(startMs: 0, endMs: 60_000)
    private func cut(_ id: String, _ start: Int64, _ end: Int64, included: Bool = true) -> EditableCut {
        var value = EditableCut(id: id, coreStartMs: start, coreEndMs: end, confidence: 0.3)
        value.included = included
        return value
    }
    private func presentation(_ draft: EditorDraft, cleanup: [SuppressionRegion] = []) -> EditorPresentation {
        .init(draft: draft, suggestions: cleanup, bounds: bounds, durationMs: bounds.endMs)
    }
    func testQueueCoalescesRelatedWorkAndRetainsOrphanEvidence() throws {
        var draft = EditorDraft(sourceRevision: "synthetic", cuts: [cut("a", 10_000, 15_000)])
        draft.scoreTracking = try ScoreTracking(serveMarkers: [
            .init(id: "inside", timestampMs: 10_000, side: .review, rallyId: "a"),
            .init(id: "orphan", timestampMs: 40_000, side: .review)
        ]).jsonValue()
        let cleanup = [SuppressionRegion(id: "c", logicalId: "c", startMs: 11_000, endMs: 12_000, eligiblePolicyIds: ["aggressive"]),
            SuppressionRegion(id: "gap", logicalId: "gap", startMs: 30_000, endMs: 32_000, eligiblePolicyIds: ["aggressive"])]
        draft.suppressionScopeOverrides["c"] = "veto-region"
        let queue = RallyReview.queue(presentation: presentation(draft, cleanup: cleanup), ignored: [])
        XCTAssertEqual(queue.map(\.id), ["rally-a", "cleanup-gap", "serve-orphan"])
        XCTAssertTrue(queue[0].clipReview)
        XCTAssertEqual(queue[0].cleanupIds, ["c"])
        XCTAssertEqual(queue[0].serveIds, ["inside"])
    }
    func testFullyIgnoredEvidenceNeverEntersQueueButPartiallyVisibleWorkDoes() throws {
        var draft = EditorDraft(sourceRevision: "synthetic", cuts: [cut("a", 10_000, 15_000), cut("b", 30_000, 35_000)])
        draft.ignoredIntervals = [.init(id: "ignore", startMs: 10_000, endMs: 20_000), .init(id: "part", startMs: 30_000, endMs: 31_000)]
        draft.scoreTracking = try ScoreTracking(serveMarkers: [.init(id: "s", timestampMs: 12_000, side: .review)]).jsonValue()
        let queue = RallyReview.queue(presentation: presentation(draft), ignored: draft.ignoredIntervals)
        XCTAssertEqual(queue.map(\.rallyId), ["b"])
        XCTAssertEqual(EditorReview.firstReviewableTime(startMs: queue[0].startMs, endMs: queue[0].endMs, ignored: draft.ignoredIntervals), 31_000)
    }
    func testServeIdentityWinsOverOverlappingPadding() throws {
        var a = cut("a", 10_000, 15_000), b = cut("b", 16_000, 20_000)
        a.keepEndMs = 18_000; b.keepStartMs = 14_000
        var draft = EditorDraft(sourceRevision: "synthetic", cuts: [a, b])
        draft.reviewedCutIds = ["a", "b"]
        draft.scoreTracking = try ScoreTracking(serveMarkers: [.init(id: "s", timestampMs: 14_500, side: .review, rallyId: "b")]).jsonValue()
        let p = presentation(draft)
        XCTAssertEqual(RallyReview.queue(presentation: p, ignored: []).first?.rallyId, "b")
        XCTAssertTrue(RallyReview.serves(for: .init(cuts: [a]), in: p.score.serveMarkers).isEmpty)
    }
    func testSeekingIntoGapRetainsSelectionAndOnlyChangesNavigation() {
        let draft = EditorDraft(sourceRevision: "synthetic", cuts: [cut("a", 10_000, 15_000), cut("b", 30_000, 35_000)])
        let p = presentation(draft)
        var navigation = RallyDeckNavigation(); navigation.selectedId = "a"; navigation.browsing = true
        navigation.follow(time: 28_000, rallies: p.groups)
        XCTAssertEqual(navigation.selectedId, "a")
        XCTAssertEqual(navigation.listCenterId, "b")
        XCTAssertTrue(navigation.browsing)
        XCTAssertNil(RallyReview.at(28_000, in: p.groups))
        XCTAssertEqual(p.pendingConfidenceIds, ["a", "b"])
        navigation.follow(time: 32_000, rallies: p.groups)
        XCTAssertEqual(navigation.selectedId, "b")
    }
    func testVisitingReviewLeavesAllDraftDecisionsPending() {
        let draft = EditorDraft(sourceRevision: "synthetic", cuts: [cut("a", 10_000, 15_000), cut("b", 30_000, 35_000)])
        let p = presentation(draft), queue = RallyReview.queue(presentation: presentation(draft), ignored: [])
        var navigation = RallyDeckNavigation()
        navigation.visit(queue[0]); navigation.visit(queue[1]); navigation.visit(queue[0])
        XCTAssertEqual(p.pendingConfidenceIds, ["a", "b"])
        XCTAssertTrue(draft.reviewedCutIds.isEmpty)
        XCTAssertEqual(navigation.focus?.id, "rally-a")
    }
    func testNeighborsSkipRemovedAndFullyExcludedAndContextRetainsThem() {
        let groups = [cut("a", 3_000, 6_000), cut("removed", 10_000, 12_000, included: false),
                      cut("selected", 20_000, 23_000), cut("excluded", 30_000, 33_000), cut("z", 42_000, 45_000)].map { EditableRallyGroup(cuts: [$0]) }
        let effective: Set<String> = ["a", "selected", "z"]
        XCTAssertEqual(RallyReview.neighbor(of: groups[2], direction: -1, rallies: groups, effectiveIds: effective)?.id, "a")
        XCTAssertEqual(RallyReview.neighbor(of: groups[2], direction: 1, rallies: groups, effectiveIds: effective)?.id, "z")
        XCTAssertEqual(RallyReview.context(for: groups[2], rallies: groups, effectiveIds: effective, bounds: bounds), .init(startMs: 12_000, endMs: 31_000))
    }
    func testMissingRangeClipsToUnoccupiedGapAndCannotCrossIgnoredTime() {
        let occupied: [TimeRange] = [.init(startMs: 10_000, endMs: 15_000), .init(startMs: 19_000, endMs: 21_000), .init(startMs: 24_000, endMs: 28_000)]
        XCTAssertEqual(RallyReview.missingRange(at: 17_000, bounds: bounds, occupied: occupied), .init(startMs: 15_000, endMs: 19_000))
        XCTAssertNil(RallyReview.missingRange(at: 20_000, bounds: bounds, occupied: occupied))
        XCTAssertNil(RallyReview.missingRange(at: 60_000, bounds: bounds, occupied: occupied))
        XCTAssertNil(RallyReview.missingRange(at: 100, bounds: .init(startMs: 0, endMs: 400), occupied: []))
    }
    func testExplicitDecisionsResolveCleanupFragmentsAndClipButNotServe() throws {
        var draft = EditorDraft(sourceRevision: "synthetic", cuts: [cut("a", 10_000, 15_000)])
        draft.scoreTracking = try ScoreTracking(serveMarkers: [.init(id: "s", timestampMs: 10_000, side: .review, rallyId: "a")]).jsonValue()
        let cleanup = SuppressionRegion(id: "c", logicalId: "logical", startMs: 11_000, endMs: 12_000, eligiblePolicyIds: ["aggressive"])
        let kept = RallyReview.decide(draft, rally: .init(cuts: draft.cuts), cleanup: [cleanup], keep: true)
        XCTAssertEqual(kept.reviewedCutIds, ["a"])
        XCTAssertEqual(kept.suppressionDecisionOverrides["logical"], "keep")
        XCTAssertTrue(presentation(kept, cleanup: [cleanup]).reviewCuts.isEmpty)
        XCTAssertEqual(presentation(kept, cleanup: [cleanup]).reviewServes.map(\.id), ["s"])
        let removed = RallyReview.decide(kept, rally: .init(cuts: kept.cuts), cleanup: [cleanup], keep: false)
        XCTAssertFalse(removed.cuts[0].included)
        XCTAssertTrue(presentation(removed, cleanup: [cleanup]).intervals.isEmpty)
        XCTAssertEqual(removed.scoreTracking, kept.scoreTracking)
    }
    func testTrimKeepsInRangeFrameAndClampsOnlyExcludedFrameWithServeAlignment() throws {
        var draft = EditorDraft(sourceRevision: "synthetic", cuts: [cut("a", 10_000, 15_000)])
        draft.scoreTracking = try ScoreTracking(serveMarkers: [.init(id: "s", timestampMs: 10_000, side: .near, rallyId: "a")]).jsonValue()
        let trimmed = EditorMath.setCoreStart(draft, cutId: "a", valueMs: 11_000, minimumMs: 0)
        let range = TimeRange(startMs: trimmed.cuts[0].coreStartMs, endMs: trimmed.cuts[0].coreEndMs)
        XCTAssertEqual(RallyReview.frameAfterTrim(12_000, range: range), 12_000)
        XCTAssertEqual(RallyReview.frameAfterTrim(10_500, range: range), 11_000)
        XCTAssertEqual(RallyReview.frameAfterTrim(16_000, range: range), 15_000)
        XCTAssertEqual(try ScoreTracking.fromJSON(trimmed.scoreTracking, durationMs: 60_000).serveMarkers[0].timestampMs, 11_000)
        try trimmed.validate(durationMs: 60_000)
    }
    func testUnrelatedEndTrimRetainsFrameInStartPadding() throws {
        var padded = cut("a", 10_000, 15_000); padded.keepStartMs = 8_000; padded.keepEndMs = 17_000
        let draft = EditorDraft(sourceRevision: "synthetic", cuts: [padded])
        let trimmed = EditorMath.setCoreEnd(draft, cutId: "a", valueMs: 14_000, maximumMs: 60_000)
        let range = TimeRange(startMs: trimmed.cuts[0].keepStartMs, endMs: trimmed.cuts[0].keepEndMs)
        XCTAssertEqual(range, .init(startMs: 8_000, endMs: 16_000))
        XCTAssertEqual(RallyReview.frameAfterTrim(9_000, range: range), 9_000)
        XCTAssertEqual(RallyReview.frameAfterTrim(16_500, range: range), 16_000)
        try trimmed.validate(durationMs: 60_000)
    }
    func testHourLongGapsDoNotCompressFiveSecondRally() {
        let groups = [cut("a", 1_000, 6_000), cut("b", 3_600_000, 3_605_000), cut("c", 7_200_000, 7_205_000)].map { EditableRallyGroup(cuts: [$0]) }
        let window = RallyReview.context(for: groups[1], rallies: groups, effectiveIds: ["a", "b", "c"], bounds: .init(startMs: 0, endMs: 8_000_000))
        XCTAssertEqual(window, .init(startMs: 3_592_000, endMs: 3_613_000))
        XCTAssertGreaterThan(Double(5_000) / Double(window.endMs - window.startMs) * 240, 50)
        XCTAssertEqual(RallyReview.neighbor(of: groups[1], direction: -1, rallies: groups, effectiveIds: ["a", "b", "c"])?.id, "a")
        XCTAssertEqual(RallyReview.context(for: groups[1], rallies: groups, effectiveIds: ["b"], bounds: .init(startMs: 0, endMs: 8_000_000)), window)
    }
    func testAccessibleAddSourceSeekingCanLeaveExistingFootageWithoutCreatingARally() {
        let draft = EditorDraft(sourceRevision: "synthetic", cuts: [cut("a", 10_000, 15_000)])
        let occupied = draft.cuts.map { TimeRange(startMs: $0.keepStartMs, endMs: $0.keepEndMs) }
        var selection = MissingRallySelection(positionMs: 12_000, bounds: bounds, occupied: occupied)
        XCTAssertNil(selection.range)
        selection.seek(to: 20_000, bounds: bounds, occupied: occupied)
        XCTAssertEqual(selection.positionMs, 20_000)
        XCTAssertEqual(selection.range, .init(startMs: 17_000, endMs: 25_000))
        selection.seek(to: 11_000, bounds: bounds, occupied: occupied)
        XCTAssertNil(selection.range)
        XCTAssertEqual(draft.cuts.count, 1)
        XCTAssertTrue(draft.userTouchedCutIds.isEmpty)
    }
}
