import XCTest
@testable import VolleyCore

final class ChaptersTests: XCTestCase {
    private func cut(_ id: String, _ start: Int64, _ end: Int64) -> EditableCut {
        .init(id: id, coreStartMs: start, coreEndMs: end)
    }
    private func serve(_ id: String, _ time: Int64, _ side: ServingSide, _ rally: String) -> ServeMarker {
        .init(id: id, timestampMs: time, side: side, rallyId: rally)
    }
    func testSourceToConcatenatedOutputTimeline() {
        let chapters = YouTubeChapters.build(intervals: [.init(startMs: 8000, endMs: 17000, cutIds: ["R1"]),
            .init(startMs: 38000, endMs: 48000, cutIds: ["R2"])], cuts: [cut("R1", 10000, 15000), cut("R2", 40000, 46000)],
            scoreTracking: nil, options: YouTubeChapters.defaultOptions(hasScoreTracking: false, hasSideSwitches: false))
        XCTAssertEqual(YouTubeChapters.text(chapters, includeCredit: false), "0:02 Rally 1\n0:11 Rally 2")
    }
    func testScoreAndServingTeamDefaults() {
        let tracking = ScoreTracking(team1Name: "Falcons", team2Name: "Owls",
            serveMarkers: [serve("S1", 1100, .near, "R1"), serve("S2", 4100, .near, "R2")])
        let chapters = YouTubeChapters.build(intervals: [.init(startMs: 1000, endMs: 5000, cutIds: ["R1", "R2"])],
            cuts: [cut("R1", 1000, 2000), cut("R2", 4000, 5000)], scoreTracking: tracking,
            options: YouTubeChapters.defaultOptions(hasScoreTracking: true, hasSideSwitches: false))
        XCTAssertEqual(chapters.map(\.title), ["0–0 - Falcons serving", "1–0 - Falcons serving"])
    }
    func testJoinedCutsNeedSeparatingServeForDistinctChapters() {
        let intervals: [FinalCutInterval] = [.init(startMs: 8000, endMs: 24000, cutIds: ["R1", "R2"],
            joinedGaps: [.init(startMs: 15000, endMs: 17000)])]
        let cuts = [cut("R1", 8000, 15000), cut("R2", 17000, 24000)]
        let options = YouTubeChapters.defaultOptions(hasScoreTracking: false, hasSideSwitches: false)
        XCTAssertEqual(YouTubeChapters.text(YouTubeChapters.build(intervals: intervals, cuts: cuts, scoreTracking: nil, options: options), includeCredit: false), "0:00 Rally 1")
        let tracking = ScoreTracking(serveMarkers: [serve("S2", 17000, .far, "R2")])
        XCTAssertEqual(YouTubeChapters.text(YouTubeChapters.build(intervals: intervals, cuts: cuts, scoreTracking: tracking, options: options), includeCredit: false), "0:09 Rally 2")
        XCTAssertEqual(EditorMath.editableRallyGroups(cuts: cuts, intervals: intervals, serveMarkers: [], hardBoundaryCutIds: ["R2"]).count, 2)
    }
    func testRemovedSwitchAttachesToNextVisibleClipAndMergesWholeSecond() {
        let chapters = YouTubeChapters.build(intervals: [.init(startMs: 10000, endMs: 15000, cutIds: ["R1"])],
            cuts: [cut("R1", 10000, 12000)], scoreTracking: ScoreTracking(sideSwitchMarkers: [.init(id: "X", timestampMs: 8000)]),
            options: YouTubeChapters.defaultOptions(hasScoreTracking: false, hasSideSwitches: true))
        XCTAssertEqual(chapters.count, 1); XCTAssertEqual(chapters.first?.title, "Side switch 1")
        XCTAssertEqual(chapters.first?.sourceTimestampMs, 10000)
    }
    func testRedoScoreSwitchAndPaddedFinalTimeline() {
        let tracking = ScoreTracking(serveMarkers: [.init(id: "S1", timestampMs: 3623, side: .near),
            .init(id: "S2", timestampMs: 28000, side: .far, ignorePreviousPoint: true)],
            sideSwitchMarkers: [.init(id: "X1", timestampMs: 28000)])
        let cuts: [EditableCut] = [
            .init(id: "R1", coreStartMs: 5000, coreEndMs: 15000, keepStartMs: 3000, keepEndMs: 17000),
            .init(id: "R2", coreStartMs: 30000, coreEndMs: 42000, keepStartMs: 28000, keepEndMs: 44000),
            .init(id: "R3", coreStartMs: 60000, coreEndMs: 72000, keepStartMs: 58000, keepEndMs: 74000),
            .init(id: "R4", coreStartMs: 90000, coreEndMs: 105000, keepStartMs: 88000, keepEndMs: 107000)]
        let intervals = cuts.map { FinalCutInterval(startMs: $0.keepStartMs, endMs: $0.keepEndMs, cutIds: [$0.id]) }
        let chapters = YouTubeChapters.build(intervals: intervals, cuts: cuts, scoreTracking: tracking,
            options: YouTubeChapters.defaultOptions(hasScoreTracking: true, hasSideSwitches: true))
        XCTAssertEqual(YouTubeChapters.text(chapters, includeCredit: false), "0:00 0–0 - Team 1 serving - Re-do\n0:14 0–0 - Team 1 serving / Side switch 1")
    }
    func testCreditDefaultsAndLegacyOptionsDecode() throws {
        let legacy = Data(#"{"includeRallyNumber":true,"includeServeNumber":false,"includeScore":false,"includeServingTeam":false,"includeSideSwitches":false}"#.utf8)
        var options = try JSONDecoder().decode(YouTubeChapterOptions.self, from: legacy)
        XCTAssertTrue(options.includeCredit)
        XCTAssertTrue(YouTubeChapters.defaultOptions(hasScoreTracking: true, hasSideSwitches: true).includeCredit)
        options.includeCredit = false
        XCTAssertEqual(try JSONDecoder().decode(YouTubeChapterOptions.self, from: JSONEncoder().encode(options)), options)
        let chapters = [YouTubeChapter(kind: .rally, outputMs: 0, sourceTimestampMs: 0, title: "Rally 1")]
        XCTAssertEqual(YouTubeChapters.text(chapters), "Edited with https://volleysplice.com\n\n0:00 Rally 1")
        XCTAssertEqual(YouTubeChapters.text(chapters, includeCredit: false), "0:00 Rally 1")
        XCTAssertEqual(YouTubeChapters.text([]), "")
    }
    func testEnabledTrackingOmitsUnmarkedRalliesWithoutShiftingTimeOrNumber() {
        let cuts = [cut("R1", 10000, 20000), cut("R2", 40000, 50000)]
        let intervals = cuts.map { FinalCutInterval(startMs: $0.keepStartMs, endMs: $0.keepEndMs, cutIds: [$0.id]) }
        let options = YouTubeChapters.defaultOptions(hasScoreTracking: false, hasSideSwitches: false)
        var tracking = ScoreTracking(serveMarkers: [serve("S2", 42000, .far, "R2")])
        func build(_ score: ScoreTracking?) -> [YouTubeChapter] {
            YouTubeChapters.build(intervals: intervals, cuts: cuts, scoreTracking: score, options: options)
        }
        XCTAssertEqual(YouTubeChapters.text(build(tracking), includeCredit: false), "0:10 Rally 2")
        XCTAssertTrue(build(ScoreTracking()).isEmpty)
        tracking.enabled = false
        XCTAssertEqual(build(tracking).count, 2)
        XCTAssertEqual(build(nil).count, 2)
    }
    func testFormattingAndFilename() {
        XCTAssertEqual(YouTubeChapters.formatTimestamp(999), "0:00")
        XCTAssertEqual(YouTubeChapters.formatTimestamp(65999), "1:05")
        XCTAssertEqual(YouTubeChapters.formatTimestamp(3661999), "1:01:01")
        XCTAssertEqual(YouTubeChapters.filename("Friday Night!.mp4"), "Friday-Night-youtube-chapters.txt")
        XCTAssertEqual(YouTubeChapters.filename("!!.mp4"), "volleysplice-youtube-chapters.txt")
    }

    private func groups(_ draft: EditorDraft, pending: Set<String> = []) -> [EditableRallyGroup] {
        EditorMath.editableRallyGroups(cuts: draft.cuts, intervals: EditorMath.finalIntervals(draft),
            serveMarkers: [], hardBoundaryCutIds: pending)
    }
    func testExplicitSplitStaysIndependentForHighConfidenceReviewedAndManualRallies() throws {
        for origin in [CutOrigin.inferred, .manual] {
            var original = EditorDraft(sourceRevision: "split", cuts: [
                .init(id: "R001", coreStartMs: 10000, coreEndMs: 20000, keepStartMs: 8000,
                      keepEndMs: 22000, confidence: 1, origin: origin)])
            original.reviewedCutIds = ["R001"]
            let split = try XCTUnwrap(EditorMath.splitCut(original, cutId: "R001", positionMs: 15000,
                bounds: .init(startMs: 0, endMs: 40000)))
            XCTAssertEqual(groups(split).count, 2)
            XCTAssertEqual(groups(split, pending: Set(split.cuts.map(\.id))).count, 2)
            XCTAssertNotEqual(split.cuts[0].editableGroupId, split.cuts[1].editableGroupId)
            XCTAssertEqual(split.cuts[0].id, "R001")
            XCTAssertEqual(EditorMath.finalIntervals(split).map { [$0.startMs, $0.endMs] }, [[8000, 22000]])
            let removed = EditorMath.setIncluded(split, cutId: split.cuts[1].id, included: false)
            XCTAssertTrue(removed.cuts[0].included); XCTAssertFalse(removed.cuts[1].included)
            XCTAssertEqual(EditorMath.finalIntervals(removed).map { [$0.startMs, $0.endMs] }, [[8000, 17000]])
            let trimmed = EditorMath.setCoreStart(removed, cutId: split.cuts[1].id, valueMs: 16000, minimumMs: 0)
            XCTAssertEqual(trimmed.cuts[0], removed.cuts[0])
            XCTAssertEqual(groups(trimmed).count, 2)
            XCTAssertEqual(try JSONDecoder().decode(EditorDraft.self, from: JSONEncoder().encode(trimmed)), trimmed)
        }
    }
    func testSplitPartitionsWholeSelectedGroupAndRepeatedSplitDoesNotChangeSibling() throws {
        let bounds = TimeRange(startMs: 0, endMs: 50000)
        let original = EditorDraft(sourceRevision: "split", cuts: [cut("R001", 10000, 18000), cut("R009", 19000, 25000)])
        XCTAssertEqual(groups(original).count, 1)
        let split = try XCTUnwrap(EditorMath.splitCut(original, cutId: "R001", positionMs: 14000,
            bounds: bounds, groupCutIds: ["R001", "R009"]))
        XCTAssertEqual(groups(split).map { $0.cuts.count }, [1, 2])
        let repeated = try XCTUnwrap(EditorMath.splitCut(split, cutId: "R009", positionMs: 22000, bounds: bounds))
        XCTAssertEqual(groups(repeated).map { $0.cuts.count }, [1, 2, 1])
        XCTAssertEqual(repeated.cuts[0], split.cuts[0])
        let options = YouTubeChapters.defaultOptions(hasScoreTracking: false, hasSideSwitches: false)
        XCTAssertEqual(YouTubeChapters.build(intervals: EditorMath.finalIntervals(repeated), cuts: repeated.cuts,
            scoreTracking: nil, options: options).count, 3)
    }
    func testSameExplicitGroupIgnoresConnectivityConfidenceAndServeBoundaries() {
        var first = cut("R001", 1000, 2000), second = cut("R002", 10000, 11000)
        first.editableGroupId = "opaque"; second.editableGroupId = "opaque"
        second.origin = .manual; second.included = false
        let result = EditorMath.editableRallyGroups(cuts: [first, second], intervals: [],
            serveMarkers: [serve("S2", 10000, .near, "R002")], hardBoundaryCutIds: ["R002"])
        XCTAssertEqual(result.count, 1); XCTAssertEqual(result[0].cutIds, ["R001", "R002"])
        second.editableGroupId = nil
        XCTAssertEqual(EditorMath.editableRallyGroups(cuts: [first, second], intervals: [
            .init(startMs: 1000, endMs: 11000, cutIds: ["R001", "R002"])], serveMarkers: []).count, 2)
    }
    func testPausedExplicitSelectionSurvivesOverlappingPaddingAndPlaybackFollowsTime() throws {
        let original = EditorDraft(sourceRevision: "split", cuts: [.init(id: "R001", coreStartMs: 10000,
            coreEndMs: 20000, keepStartMs: 8000, keepEndMs: 22000)])
        let split = try XCTUnwrap(EditorMath.splitCut(original, cutId: "R001", positionMs: 15000,
            bounds: .init(startMs: 0, endMs: 40000)))
        let rallies = groups(split), right = rallies[1].id
        XCTAssertEqual(EditorMath.rallySelectionAtPlayhead(groups: rallies, selectedId: right,
            positionMs: rallies[1].keepStartMs, playing: false)?.id, right)
        XCTAssertEqual(EditorMath.rallySelectionAtPlayhead(groups: rallies, selectedId: right,
            positionMs: 9000, playing: true)?.id, rallies[0].id)
        let trimmed = EditorMath.setCoreStart(split, cutId: right, valueMs: 14000, minimumMs: 0)
        let overlapping = groups(trimmed)
        let selected = try XCTUnwrap(EditorMath.rallySelectionAtPlayhead(groups: overlapping, selectedId: right,
            positionMs: 14000, playing: false))
        XCTAssertEqual(selected.id, right)
        let removed = EditorMath.setIncluded(trimmed, cutId: selected.id, included: false)
        XCTAssertTrue(removed.cuts[0].included); XCTAssertFalse(removed.cuts[1].included)
    }
    func testSplitLimitsAndCompleteDraftIdNamespaceAreCheckedBeforeMutation() throws {
        let bounds = TimeRange(startMs: 0, endMs: 40000)
        var draft = EditorDraft(sourceRevision: "split", cuts: [cut("R001", 10000, 20000)])
        draft.ignoredIntervals = [.init(id: "R002", startMs: 1000, endMs: 2000)]
        let split = try XCTUnwrap(EditorMath.splitCut(draft, cutId: "R001", positionMs: 15000, bounds: bounds))
        XCTAssertEqual(split.cuts.map(\.id), ["R001", "R003"])
        for id in [String(repeating: "x", count: 257), "  ", String(repeating: "😀", count: 129)] {
            let old = EditorDraft(sourceRevision: "split", cuts: [cut(id, 10000, 20000)])
            XCTAssertNil(EditorMath.splitCut(old, cutId: id, positionMs: 15000, bounds: bounds))
            XCTAssertNil(old.cuts[0].editableGroupId)
        }
        for count in [2047, 2048] {
            let cuts = (0..<count).map { index -> EditableCut in
                var value = cut("C\(index)", 10000, 20000); value.editableGroupId = "g\(index)"; return value
            }
            let source = EditorDraft(sourceRevision: "limit", cuts: cuts)
            let result = EditorMath.splitCut(source, cutId: "C0", positionMs: 15000, bounds: bounds)
            XCTAssertEqual(result == nil, count == 2048)
            XCTAssertEqual(source.cuts.count, count)
            if let result { XCTAssertEqual(result.cuts.count, 2048); try result.validate(durationMs: 40000) }
        }
        for id in [String(repeating: "x", count: 256), String(repeating: "😀", count: 128)] {
            XCTAssertNotNil(EditorMath.splitCut(.init(sourceRevision: "boundary", cuts: [cut(id, 10000, 20000)]),
                cutId: id, positionMs: 15000, bounds: bounds))
        }
    }
    func testSplitIdAllocationHandlesLargeAndOverflowingNumericSuffixesWithoutCollisions() throws {
        for id in ["R4294967296", "R9223372036854775807", "R99999999999999999999999999999999999999"] {
            var draft = EditorDraft(sourceRevision: "wide-id", cuts: [cut("R001", 10000, 20000)])
            draft.ignoredIntervals = [.init(id: id, startMs: 1000, endMs: 2000)]
            let split = try XCTUnwrap(EditorMath.splitCut(draft, cutId: "R001", positionMs: 15000,
                bounds: .init(startMs: 0, endMs: 40000)))
            XCTAssertEqual(split.cuts[0].id, "R001")
            XCTAssertNotEqual(split.cuts[1].id, "R001"); XCTAssertNotEqual(split.cuts[1].id, id)
            XCTAssertEqual(split.ignoredIntervals, draft.ignoredIntervals)
            try split.validate(durationMs: 40000)
        }
        XCTAssertEqual(EditorMath.nextId(prefix: "R", ids: ["R001", "R4294967296"]), "R4294967297")
        XCTAssertEqual(EditorMath.nextId(prefix: "R", ids: ["R001", "R009"]), "R010")
    }
}
