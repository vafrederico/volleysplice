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
        XCTAssertEqual(YouTubeChapters.text(chapters), "0:02 Rally 1\n0:11 Rally 2")
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
        XCTAssertEqual(YouTubeChapters.text(YouTubeChapters.build(intervals: intervals, cuts: cuts, scoreTracking: nil, options: options)), "0:00 Rally 1")
        let tracking = ScoreTracking(serveMarkers: [serve("S2", 17000, .far, "R2")])
        XCTAssertEqual(YouTubeChapters.text(YouTubeChapters.build(intervals: intervals, cuts: cuts, scoreTracking: tracking, options: options)), "0:00 Rally 1\n0:09 Rally 2")
        XCTAssertEqual(EditorMath.editableRallyGroups(cuts: cuts, intervals: intervals, serveMarkers: [], hardBoundaryCutIds: ["R2"]).count, 2)
    }
    func testRemovedSwitchAttachesToNextVisibleClipAndMergesWholeSecond() {
        let chapters = YouTubeChapters.build(intervals: [.init(startMs: 10000, endMs: 15000, cutIds: ["R1"])],
            cuts: [cut("R1", 10000, 12000)], scoreTracking: ScoreTracking(sideSwitchMarkers: [.init(id: "X", timestampMs: 8000)]),
            options: YouTubeChapters.defaultOptions(hasScoreTracking: false, hasSideSwitches: true))
        XCTAssertEqual(chapters.count, 1); XCTAssertEqual(chapters.first?.title, "Rally 1 / Side switch 1")
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
        XCTAssertEqual(YouTubeChapters.text(chapters), "0:00 0–0 - Team 1 serving - Re-do\n0:14 0–0 - Team 1 serving / Side switch 1\n0:30 Rally 3\n0:46 Rally 4")
    }
    func testFormattingAndFilename() {
        XCTAssertEqual(YouTubeChapters.formatTimestamp(999), "0:00")
        XCTAssertEqual(YouTubeChapters.formatTimestamp(65999), "1:05")
        XCTAssertEqual(YouTubeChapters.formatTimestamp(3661999), "1:01:01")
        XCTAssertEqual(YouTubeChapters.filename("Friday Night!.mp4"), "Friday-Night-youtube-chapters.txt")
        XCTAssertEqual(YouTubeChapters.filename("!!.mp4"), "volleysplice-youtube-chapters.txt")
    }
}
