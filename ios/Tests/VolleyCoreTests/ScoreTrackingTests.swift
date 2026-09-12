import XCTest
@testable import VolleyCore

final class ScoreTrackingTests: XCTestCase {
    private func serve(_ id: String, _ time: Int64, _ side: ServingSide, rallyId: String? = nil) -> ServeMarker {
        .init(id: id, timestampMs: time, side: side, rallyId: rallyId)
    }
    private func output(_ verdict: ServingSideVerdict) -> ServingSideOutput {
        .init(candidates: [.init(id: "R001", anchor: 1, intervalEnd: 2, nearProbability: 0.9, side: .near, verdict: verdict)])
    }
    func testFirstServeEstablishesServerAndNextServesAwardPreviousPoints() {
        let tracking = ScoreTracking(serveMarkers: [serve("S1", 1000, .near), serve("S2", 2000, .near), serve("S3", 3000, .far)])
        let score = ScoreReducer.deriveAt(tracking)
        XCTAssertEqual(score.team1Score, 1); XCTAssertEqual(score.team2Score, 1)
        XCTAssertEqual(score.points.count, 2); XCTAssertEqual(score.servingTeamId, .team2)
        XCTAssertEqual(ScoreReducer.deriveAt(tracking, sourceTimestampMs: 1000).team1Score, 0)
        XCTAssertNil(ScoreReducer.deriveAt(tracking, sourceTimestampMs: 999).servingTeamId)
    }
    func testFinalRallyRemainsUnawardedUntilManualNextServe() throws {
        // Android ScoreReducerTest.finalRallyIsUnawardedUntilManualNextServeExists.
        let oneServe = ScoreTracking(serveMarkers: [serve("S1", 1_000, .near)])
        let unfinished = ScoreReducer.deriveAt(oneServe)
        XCTAssertEqual(unfinished.team1Score, 0)
        XCTAssertTrue(unfinished.points.isEmpty)
        let completed = ScoreReducer.addServe(oneServe, timestampMs: 5_000, side: .near)
        let manual = try XCTUnwrap(completed.serveMarkers.last)
        XCTAssertEqual(manual.origin, .manual)
        XCTAssertEqual(ScoreReducer.deriveAt(completed, sourceTimestampMs: 4_999).team1Score, 0)
        let score = ScoreReducer.deriveAt(completed)
        XCTAssertEqual(score.team1Score, 1)
        XCTAssertEqual(score.team2Score, 0)
        XCTAssertEqual(score.points.map(\.serveMarkerId), [manual.id])
        XCTAssertEqual(oneServe.serveMarkers.count, 1)
    }
    func testFirstLeadingPaddingStopsLookaheadAfterItsFirstServe() {
        // Exact Android firstLeadingPaddingStopsLookingAheadAfterCrossingItsServeMarker fixture.
        let ranges: [ScoreRallyRange] = [.init(coreStartMs: 5_000, coreEndMs: 12_000, keepStartMs: 3_000, keepEndMs: 14_000)]
        let merged: [ScoreMergedRange] = [.init(startMs: 3_000, endMs: 14_000)]
        let crossed = ScoreTracking(serveMarkers: [serve("S1", 2_500, .near), serve("S2", 20_000, .far)])
        for time in [Int64(3_000), 3_500, 4_500] {
            XCTAssertEqual(ScoreReducer.scoreBoundaryTimestamp(time, rallyRanges: ranges, tracking: crossed, mergedRanges: merged), time)
        }
        // Additional guardrail for Android's servesInsidePadding rule: a second
        // serve inside the same leading pad must not be revealed prematurely.
        let multiple = ScoreTracking(serveMarkers: [serve("S2", 4_500, .far), serve("S1", 3_500, .near)])
        let cases: [(Int64, Int64)] = [(2_999, 2_999), (3_000, 3_500), (3_500, 3_500), (4_000, 4_000), (4_500, 4_500), (5_000, 5_000)]
        for (time, expected) in cases {
            let boundary = ScoreReducer.scoreBoundaryTimestamp(time, rallyRanges: ranges, tracking: multiple, mergedRanges: merged)
            XCTAssertEqual(boundary, expected)
            XCTAssertEqual(ScoreReducer.deriveAt(multiple, sourceTimestampMs: boundary).team2Score, time < 4_500 ? 0 : 1)
        }
    }
    func testWireRejectsInvalidOptionalModelSideButAcceptsAbsentOrNull() throws {
        // Android ScoreReducerTest rejects the optional modelSide value "left".
        var json = try ScoreTracking(serveMarkers: [.init(id: "S001", timestampMs: 1_000, side: .near, modelSide: .near)]).jsonValue()
        var marker = try XCTUnwrap(json["serveMarkers"]?.array?.first)
        marker["modelSide"] = .string("left"); json["serveMarkers"] = .array([marker])
        XCTAssertThrowsError(try ScoreTracking.fromJSON(json, durationMs: 5_000))
        marker["modelSide"] = nil; json["serveMarkers"] = .array([marker])
        XCTAssertNil(try ScoreTracking.fromJSON(json, durationMs: 5_000).serveMarkers.first?.modelSide)
        marker["modelSide"] = .null; json["serveMarkers"] = .array([marker])
        XCTAssertNil(try ScoreTracking.fromJSON(json, durationMs: 5_000).serveMarkers.first?.modelSide)
    }
    func testReviewReplayAndSimultaneousSwitch() {
        var replay = serve("S3", 3000, .far); replay.ignorePreviousPoint = true
        let tracking = ScoreTracking(serveMarkers: [serve("S1", 1000, .near), serve("S2", 2000, .review), replay])
        let score = ScoreReducer.deriveAt(tracking)
        XCTAssertEqual(score.team1Score + score.team2Score, 0)
        XCTAssertEqual(score.reviewPointCount, 1); XCTAssertEqual(score.ignoredPointCount, 1)
        let switched = ScoreTracking(serveMarkers: [serve("S1", 1000, .near), serve("S2", 2000, .near)],
                                     sideSwitchMarkers: [.init(id: "X1", timestampMs: 2000)])
        XCTAssertEqual(ScoreReducer.deriveAt(switched).team2Score, 1)
        XCTAssertEqual(ScoreReducer.deriveAt(switched).team1Score, 0)
    }
    func testSourceOrderingAndSwitchDoesNotRetroactivelyChangeServerBetweenServes() {
        let tracking = ScoreTracking(serveMarkers: [serve("B", 1000, .far), serve("A", 1000, .near)],
                                     sideSwitchMarkers: [.init(id: "X", timestampMs: 2000)])
        let score = ScoreReducer.deriveAt(tracking, sourceTimestampMs: 2500)
        XCTAssertEqual(score.points.map(\.serveMarkerId), ["B"])
        XCTAssertEqual(score.team2Score, 1); XCTAssertEqual(score.servingTeamId, .team2)
    }
    func testIgnoredHalfOpenTimesAndExcludedRalliesDoNotDeleteStateOrSwitches() {
        let tracking = ScoreTracking(serveMarkers: [serve("S1", 1000, .near, rallyId: "R1"),
            serve("S2", 2000, .near, rallyId: "R2"), serve("S3", 3000, .near)],
            sideSwitchMarkers: [.init(id: "X1", timestampMs: 1500), .init(id: "X2", timestampMs: 3000, rallyIds: ["R2"])])
        let visible = ScoreReducer.visibleTracking(tracking, ignoredIntervals: [.init(id: "I", startMs: 1000, endMs: 2000)], excludedRallyIds: ["R2"])
        XCTAssertEqual(visible.serveMarkers.map(\.id), ["S3"])
        XCTAssertEqual(visible.sideSwitchMarkers.map(\.id), ["X2"])
        XCTAssertEqual(tracking.serveMarkers.count, 3)
    }
    func testLeadingPaddingAndInternalMergedGaps() {
        let tracking = ScoreTracking(serveMarkers: [serve("S1", 5000, .near), serve("S2", 18000, .far), serve("S3", 30000, .near)])
        let ranges: [ScoreRallyRange] = [.init(coreStartMs: 5000, coreEndMs: 12000, keepStartMs: 3000, keepEndMs: 14000),
                                       .init(coreStartMs: 20000, coreEndMs: 25000, keepStartMs: 16000, keepEndMs: 27000)]
        let cases: [(Int64, Int64)] = [(13000, 13000), (15000, 15000), (16000, 18000), (19000, 19000), (21000, 21000)]
        for (input, expected) in cases {
            XCTAssertEqual(ScoreReducer.scoreBoundaryTimestamp(input, rallyRanges: ranges, tracking: tracking,
                mergedRanges: [.init(startMs: 3000, endMs: 27000)]), expected)
        }
    }
    func testTombstonesPreventReseedingAndPreserveToggle() throws {
        let seeded = try ScoreReducer.seedModelMarkers(ScoreTracking(enabled: false), output: output(.near))
        XCTAssertFalse(seeded.enabled)
        let removed = ScoreReducer.removeServe(seeded, markerId: "serve-R001")
        let reseeded = try ScoreReducer.seedModelMarkers(removed, output: output(.near))
        XCTAssertEqual(reseeded.removedModelMarkerIds, ["serve-R001"]); XCTAssertTrue(reseeded.serveMarkers.isEmpty)
    }
    func testCorrectedNotServeCanonicalizesIdAndRetainsReplayAndManual() throws {
        let tracking = ScoreTracking(serveMarkers: [.init(id: "imported-id", timestampMs: 999, side: .far, origin: .model,
            modelSide: .near, ignorePreviousPoint: true, rallyId: "R001"), serve("S001", 4000, .near)])
        let reseeded = try ScoreReducer.seedModelMarkers(tracking, output: output(.notServe))
        let marker = try XCTUnwrap(reseeded.serveMarkers.first { $0.origin == .model })
        XCTAssertEqual(marker.id, "serve-R001"); XCTAssertEqual(marker.timestampMs, 1000)
        XCTAssertEqual(marker.side, .far); XCTAssertEqual(marker.modelSide, .review); XCTAssertTrue(marker.ignorePreviousPoint)
        XCTAssertTrue(reseeded.serveMarkers.contains { $0.id == "S001" })
        XCTAssertTrue(try ScoreReducer.seedModelMarkers(ScoreTracking(), output: output(.notServe)).serveMarkers.isEmpty)
    }
    func testSwitchSeedingDeletionAndDisableKeepManualMarkers() throws {
        let output = SideSwitchOutput(candidates: [.init(id: "boundary", timestamp: 3, probability: 0.91,
            kind: .adjacentRallyBoundary, sourceRangeIds: ["R1", "R2", "R1"])])
        let initial = ScoreTracking(sideSwitchMarkers: [.init(id: "X001", timestampMs: 1000)])
        let seeded = try ScoreReducer.seedModelMarkers(initial, output: nil, sideSwitchOutput: output)
        XCTAssertEqual(seeded.sideSwitchMarkers[1].rallyIds, ["R1", "R2"])
        XCTAssertEqual(seeded.sideSwitchMarkers[1].modelConfidence, 0.91)
        let disabled = try ScoreReducer.seedModelMarkers(seeded, output: nil, sideSwitchEnabled: false)
        XCTAssertEqual(disabled.sideSwitchMarkers.map(\.id), ["X001"])
        let deleted = ScoreReducer.removeSideSwitch(seeded, markerId: "switch-boundary")
        XCTAssertEqual(try ScoreReducer.seedModelMarkers(deleted, output: nil, sideSwitchOutput: output).sideSwitchMarkers.map(\.id), ["X001"])
    }
    func testWireRoundTripMigrationAndValidation() throws {
        let value = ScoreTracking(team1Name: "Falcons", serveMarkers: [.init(id: "S", timestampMs: 1250, side: .far,
            modelSide: .near, rallyId: "R1")], sideSwitchMarkers: [.init(id: "X", timestampMs: 2000)], removedModelMarkerIds: ["serve-R009"])
        let json = try value.jsonValue()
        XCTAssertEqual(json["serveMarkers"]?.array?.first?["timestamp"], .number(1.25))
        XCTAssertNil(json["serveMarkers"]?.array?.first?["timestampMs"])
        XCTAssertEqual(try ScoreTracking.fromJSON(json, durationMs: 5000), value)
        var legacy = try ScoreTracking().jsonValue(); legacy["version"] = .number(1); legacy["removedModelMarkerIds"] = nil
        XCTAssertEqual(try ScoreTracking.fromJSON(legacy).version, 3)
        var invalid = json; invalid["enabled"] = nil
        XCTAssertThrowsError(try ScoreTracking.fromJSON(invalid))
        invalid = json; invalid["removedModelMarkerIds"] = .array([.string("gone"), .string("gone")])
        XCTAssertThrowsError(try ScoreTracking.fromJSON(invalid))
        XCTAssertThrowsError(try ScoreTracking.fromJSON(json, durationMs: 1000))
        invalid = json; invalid["removedModelMarkerIds"] = .array([.string("S")])
        XCTAssertThrowsError(try ScoreTracking.fromJSON(invalid))
    }
    func testExportScoreKeepsNullsAndSourceSeconds() {
        let score = ScoreReducer.deriveAt(.init(serveMarkers: [serve("A", 1000, .near), serve("B", 2250, .review)])).jsonValue()
        XCTAssertEqual(score["servingTeamId"], .null)
        XCTAssertEqual(score["points"]?.array?.first?["timestamp"], .number(2.25))
        XCTAssertEqual(score["points"]?.array?.first?["winnerTeamId"], .null)
    }
}
