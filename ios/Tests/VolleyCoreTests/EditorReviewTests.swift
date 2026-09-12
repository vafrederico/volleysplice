import XCTest
@testable import VolleyCore

final class EditorReviewTests: XCTestCase {
    private func region(_ id: String, _ start: Int64, _ end: Int64, logical: String? = nil) -> SuppressionRegion {
        .init(id: id, logicalId: logical ?? id, startMs: start, endMs: end, eligiblePolicyIds: ["aggressive"])
    }
    // Android EditorMathTest.nextSuppressionUsesThePlayheadWhenThePreviousSelectionIsStale.
    func testCleanupNavigationUsesPlayheadForStaleSelectionAndWraps() {
        let first = region("a", 10_000, 11_000), second = region("b", 20_000, 21_000)
        XCTAssertEqual(EditorReview.nextCleanup([second, first], positionMs: 5_000, selectedId: nil)?.id, "a")
        XCTAssertEqual(EditorReview.nextCleanup([first, second], positionMs: 15_000, selectedId: "a")?.id, "b")
        XCTAssertEqual(EditorReview.nextCleanup([first, second], positionMs: 8_000, selectedId: "a")?.id, "b")
        XCTAssertEqual(EditorReview.nextCleanup([first, second], positionMs: 25_000, selectedId: "b")?.id, "a")
    }
    func testReviewSkipsIgnoredUnionAndUsesHalfOpenEndpoints() {
        let ignored: [IgnoredSourceInterval] = [.init(id: "a", startMs: 1_000, endMs: 3_000),
            .init(id: "b", startMs: 2_000, endMs: 4_000), .init(id: "c", startMs: 5_000, endMs: 6_000)]
        XCTAssertEqual(EditorReview.firstReviewableTime(startMs: 1_500, endMs: 4_500, ignored: ignored), 4_000)
        XCTAssertNil(EditorReview.firstReviewableTime(startMs: 1_500, endMs: 4_000, ignored: ignored))
        XCTAssertEqual(EditorReview.firstReviewableTime(startMs: 4_000, endMs: 4_000, ignored: ignored), 4_000)
        XCTAssertNil(EditorReview.firstReviewableTime(startMs: 1_000, endMs: 1_000, ignored: ignored))
    }
    func testCleanupDecisionsResolveLogicalFragmentsAndPreservePartialRallyExport() throws {
        var draft = EditorDraft(sourceRevision: "fixture", cuts: [.init(id: "R001", coreStartMs: 1_000, coreEndMs: 10_000)])
        draft.beforePaddingMs = 0; draft.afterPaddingMs = 0
        draft.suppressionScopeOverrides["logical"] = "veto-region"
        let fragments = [region("a", 3_000, 4_000, logical: "logical"), region("b", 6_000, 7_000, logical: "logical")]
        XCTAssertEqual(EditorReview.pendingCleanup(fragments, draft: draft).count, 2)
        let removed = EditorReview.decideCleanup(draft, suggestion: fragments[0], keep: false, reviewedCutIds: ["R001"])
        XCTAssertTrue(removed.cuts[0].included)
        XCTAssertTrue(removed.reviewedCutIds.contains("R001"))
        XCTAssertTrue(EditorReview.pendingCleanup(fragments, draft: removed).isEmpty)
        let presentation = EditorPresentation(draft: removed, suggestions: fragments, bounds: .init(startMs: 0, endMs: 12_000), durationMs: 12_000)
        XCTAssertEqual(presentation.userRemovedSuggestionIds, ["a", "b"])
        XCTAssertEqual(presentation.intervals.map { [$0.startMs, $0.endMs] }, [[1_000, 3_000], [4_000, 6_000], [7_000, 10_000]])
        let kept = EditorReview.decideCleanup(removed, suggestion: fragments[0], keep: true, reviewedCutIds: [])
        XCTAssertEqual(EditorMath.totalFinalMs(EditorMath.finalIntervals(kept, suppression: fragments)), 9_000)
        XCTAssertTrue(kept.userTouchedCutIds.isEmpty)
    }
    func testCleanupQueueIgnoresInactivePolicyAndFullyIgnoredFragments() {
        var draft = EditorDraft(sourceRevision: "fixture")
        draft.ignoredIntervals = [.init(id: "I", startMs: 10_000, endMs: 11_000)]
        let regions = [region("ignored", 10_000, 11_000), region("visible", 20_000, 21_000)]
        XCTAssertEqual(EditorReview.pendingCleanup(EditorReview.activeSuggestions(regions, draft: draft), draft: draft).map(\.id), ["visible"])
        draft.selectedSuppressionPolicy = "none"
        XCTAssertTrue(EditorReview.activeSuggestions(regions, draft: draft).isEmpty)
    }
}
