import Foundation
import XCTest
@testable import VolleyCore

final class ProjectContractTests: XCTestCase {
    func testNewDraftUsesAutomaticDefaultsWithScoreTrackingDisabled() throws {
        // Android EditorMathTest.newDraftUsesSimpleAutomaticDefaults and the
        // skipped-serving branch. Core newDraft has no initially-enabled input.
        let draft = EditorMath.newDraft(ranges: [], durationMs: 10_000, sourceRevision: "fixture")
        XCTAssertEqual(draft.selectedSuppressionPolicy, "aggressive")
        XCTAssertFalse(try ScoreTracking.fromJSON(draft.scoreTracking).enabled)
        XCTAssertTrue(draft.renderScoreOverlay)
        XCTAssertTrue(draft.renderScoreTimeline)
        XCTAssertTrue(draft.reviewedCutIds.isEmpty)
        XCTAssertEqual(draft.beforePaddingMs, 2_000)
        XCTAssertEqual(draft.afterPaddingMs, 2_000)
        XCTAssertEqual(draft.joinGapMs, 3_000)
        try draft.validate(durationMs: 10_000)
    }
    func testInvalidOrEmptySeedsAreFilteredBeforeAssigningContiguousIds() throws {
        // Android EditorMathTest.invalidOrEmptySeedRangesAreSkippedAndIdsRemainContiguous.
        let ranges: [EditableCut] = [
            .init(id: "outside", coreStartMs: 10_000, coreEndMs: 11_000, confidence: 0.2),
            .init(id: "valid", coreStartMs: 2_000, coreEndMs: 3_000, confidence: 0.8)
        ]
        let original = EditorMath.newDraft(ranges: ranges, durationMs: 10_000, sourceRevision: "fixture")
        XCTAssertEqual(original.cuts.map(\.id), ["R001"])
        XCTAssertEqual(original.cuts.first?.keepStartMs, 0)
        XCTAssertEqual(original.cuts.first?.keepEndMs, 5_000)
        // Zero-length and reversed seeds also consume no IDs between valid cuts.
        let extra: [EditableCut] = [.init(id: "empty", coreStartMs: 4_000, coreEndMs: 4_000),
            .init(id: "reversed", coreStartMs: 6_000, coreEndMs: 5_000),
            .init(id: "second", coreStartMs: 7_000, coreEndMs: 8_000)]
        let draft = EditorMath.newDraft(ranges: ranges + extra, durationMs: 10_000, sourceRevision: "fixture")
        XCTAssertEqual(draft.cuts.map(\.id), ["R001", "R002"])
        XCTAssertEqual(draft.cuts.map(\.coreStartMs), [2_000, 7_000])
        try draft.validate(durationMs: 10_000)
    }
    func testStrictJoinThresholdAndIgnoredGapAreNotRejoined() {
        var draft = EditorDraft(sourceRevision: "test", cuts: [
            .init(id: "R001", coreStartMs: 0, coreEndMs: 1_000),
            .init(id: "R002", coreStartMs: 3_999, coreEndMs: 5_000),
            .init(id: "R003", coreStartMs: 8_000, coreEndMs: 9_000)
        ])
        XCTAssertEqual(EditorMath.finalIntervals(draft).map(\.endMs), [5_000, 9_000])
        XCTAssertEqual(EditorMath.finalIntervals(draft)[0].joinedGaps, [.init(startMs: 1_000, endMs: 3_999)])
        draft.ignoredIntervals = [.init(id: "I001", startMs: 500, endMs: 750)]
        let result = EditorMath.finalIntervals(draft)
        XCTAssertEqual(result.map(\.startMs), [0, 750, 8_000])
        XCTAssertEqual(result.map(\.endMs), [500, 5_000, 9_000])
        XCTAssertEqual(result[0].cutIds, ["R001"])
    }
    func testZeroJoinStillMergesTouchingAndOverlappingRanges() {
        var draft = EditorDraft(sourceRevision: "test", cuts: [
            .init(id: "A", coreStartMs: 0, coreEndMs: 1_000),
            .init(id: "B", coreStartMs: 1_000, coreEndMs: 2_000),
            .init(id: "C", coreStartMs: 2_001, coreEndMs: 3_000)
        ])
        draft.joinGapMs = 0
        XCTAssertEqual(EditorMath.finalIntervals(draft).map(\.endMs), [2_000, 3_000])
    }
    func testPaddingClipsToGameAndNeverRepadsManualCuts() {
        var draft = EditorMath.newDraft(ranges: [.init(id: "x", coreStartMs: 500, coreEndMs: 9_500)],
            durationMs: 10_000, gameWindow: .init(startMs: 1_000, endMs: 9_000), sourceRevision: "test")
        XCTAssertEqual(draft.cuts[0].coreStartMs, 1_000)
        XCTAssertEqual(draft.cuts[0].coreEndMs, 9_000)
        XCTAssertEqual(draft.ignoredIntervals.map(\.reason), ["outside-game-window", "outside-game-window"])
        draft = EditorMath.addManual(draft, startMs: 4_000, endMs: 5_000, bounds: .init(startMs: 1_000, endMs: 9_000))
        let padded = EditorMath.applyPadding(draft, beforeMs: 20_000, afterMs: -1, durationMs: 10_000, gameWindow: .init(startMs: 1_000, endMs: 9_000))
        XCTAssertEqual(padded.beforePaddingMs, 10_000); XCTAssertEqual(padded.afterPaddingMs, 0)
        XCTAssertEqual(padded.cuts[1].keepStartMs, 4_000); XCTAssertEqual(padded.cuts[1].keepEndMs, 5_000)
    }
    func testManualCutSurvivesWholeRallySuppressionAndTouchOverridesDefault() {
        var draft = EditorDraft(sourceRevision: "test", cuts: [
            .init(id: "R001", coreStartMs: 1_000, coreEndMs: 5_000),
            .init(id: "M001", coreStartMs: 2_000, coreEndMs: 3_000, origin: .manual)
        ])
        let suggestion = SuppressionRegion(id: "s1", logicalId: "s", startMs: 2_000, endMs: 3_000, eligiblePolicyIds: ["aggressive"])
        XCTAssertEqual(EditorMath.finalIntervals(draft, suppression: [suggestion]).flatMap(\.cutIds), ["M001"])
        draft.userTouchedCutIds = ["R001"]
        XCTAssertEqual(EditorMath.finalIntervals(draft, suppression: [suggestion]).first?.startMs, 1_000)
        draft.suppressionDecisionOverrides = ["s": "suppress"]
        XCTAssertEqual(EditorMath.finalIntervals(draft, suppression: [suggestion]).flatMap(\.cutIds), ["M001"])
    }
    func testVetoRegionIsHardBarrierEvenWithPaddingAndShortJoin() {
        var draft = EditorDraft(sourceRevision: "test", cuts: [.init(id: "R001", coreStartMs: 2_000, coreEndMs: 8_000, keepStartMs: 0, keepEndMs: 10_000)])
        draft.suppressionScopeOverrides = ["s": "veto-region"]
        let suggestion = SuppressionRegion(id: "s1", logicalId: "s", startMs: 4_000, endMs: 5_000, eligiblePolicyIds: ["aggressive"])
        let result = EditorMath.finalIntervals(draft, suppression: [suggestion])
        XCTAssertEqual(result.map(\.startMs), [0, 5_000]); XCTAssertEqual(result.map(\.endMs), [4_000, 10_000])
    }
    func testVetoFragmentPaddingClipsToKnownGameBoundsAndIgnoredFragmentsKeepOneRallyIdentity() throws {
        var draft = EditorDraft(sourceRevision: "test", cuts: [.init(id: "R001", coreStartMs: 2_500, coreEndMs: 5_500, keepStartMs: 2_000, keepEndMs: 6_000)])
        draft.beforePaddingMs = 10_000; draft.afterPaddingMs = 10_000
        draft.suppressionScopeOverrides = ["s": "veto-region"]
        draft.ignoredIntervals = [.init(id: "I001", startMs: 4_900, endMs: 5_100)]
        let suggestion = SuppressionRegion(id: "s1", logicalId: "s", startMs: 3_000, endMs: 4_800, eligiblePolicyIds: ["aggressive"])
        let intervals = EditorMath.finalIntervals(draft, suppression: [suggestion], bounds: .init(startMs: 2_000, endMs: 6_000))
        XCTAssertEqual(intervals.map(\.startMs), [2_000, 4_800, 5_100])
        XCTAssertEqual(intervals.map(\.endMs), [3_000, 4_900, 6_000])
        XCTAssertEqual(intervals.map(\.cutIds), [["R001"], ["R001"], ["R001"]])
        XCTAssertTrue(intervals.allSatisfy { $0.joinedGaps.isEmpty })
        let groups = EditorMath.editableRallyGroups(cuts: draft.cuts, intervals: intervals, serveMarkers: [])
        XCTAssertEqual(groups.map(\.id), ["R001"])
        XCTAssertEqual(groups.flatMap(\.cuts).map(\.id), ["R001"])
        XCTAssertEqual(EditorMath.totalFinalMs(intervals), 2_000)
        try draft.validate(durationMs: 6_000)
    }
    func testCoreEditAndSplitPreservePaddingAndTrackUserChanges() throws {
        var draft = EditorDraft(sourceRevision: "test", cuts: [.init(id: "R001", coreStartMs: 2_000, coreEndMs: 8_000, keepStartMs: 1_000, keepEndMs: 9_000)])
        draft = EditorMath.setCoreRange(draft, cutId: "R001", startMs: 3_000, endMs: 7_000, bounds: .init(startMs: 0, endMs: 10_000))
        XCTAssertEqual(draft.cuts[0].keepStartMs, 2_000); XCTAssertEqual(draft.cuts[0].keepEndMs, 8_000)
        let split = try XCTUnwrap(EditorMath.splitCut(draft, cutId: "R001", positionMs: 5_000, bounds: .init(startMs: 0, endMs: 10_000)))
        XCTAssertEqual(split.cuts.map(\.id), ["R001", "R002"])
        XCTAssertEqual(split.userTouchedCutIds, ["R001", "R002"])
        XCTAssertNil(EditorMath.splitCut(draft, cutId: "R001", positionMs: 3_050, bounds: .init(startMs: 0, endMs: 10_000)))
    }
    func testIndependentCoreEdgeCannotPushOppositeEdgeAndOuterPaddingDoesNotChangeLabels() throws {
        let bounds = TimeRange(startMs: 0, endMs: 10_000)
        let original = EditorDraft(sourceRevision: "test", cuts: [.init(id: "R001", coreStartMs: 2_000, coreEndMs: 8_000, keepStartMs: 1_000, keepEndMs: 9_000)])
        let edge = EditorMath.setCoreStart(original, cutId: "R001", valueMs: 9_000, minimumMs: 0)
        XCTAssertEqual(edge.cuts[0].coreStartMs, 7_900); XCTAssertEqual(edge.cuts[0].coreEndMs, 8_000)
        XCTAssertEqual(edge.cuts[0].keepStartMs, 6_900); XCTAssertEqual(edge.cuts[0].keepEndMs, 9_000)
        var padded = EditorMath.setKeepBoundary(original, cutId: "R001", valueMs: 500, isStart: true, bounds: bounds)
        XCTAssertEqual(padded.cuts[0].coreStartMs, 2_000); XCTAssertEqual(padded.cuts[0].keepStartMs, 500)
        padded = EditorMath.resetExtraTime(padded, cutIds: ["R001"], bounds: bounds)
        XCTAssertEqual(padded.cuts[0].keepStartMs, 0); XCTAssertEqual(padded.cuts[0].keepEndMs, 10_000)
        try padded.validate(durationMs: 10_000)
    }
    func testManualOuterBoundaryMovesCoreAndDeletePreservesInferredCuts() throws {
        let original = EditorDraft(sourceRevision: "test", cuts: [
            .init(id: "R001", coreStartMs: 1_000, coreEndMs: 2_000),
            .init(id: "M001", coreStartMs: 4_000, coreEndMs: 6_000, origin: .manual)])
        var edited = EditorMath.setKeepBoundary(original, cutId: "M001", valueMs: 3_500, isStart: true, bounds: .init(startMs: 0, endMs: 10_000))
        XCTAssertEqual(edited.cuts[1].coreStartMs, 3_500); XCTAssertEqual(edited.cuts[1].keepStartMs, 3_500)
        edited.reviewedCutIds = ["M001"]
        edited = EditorMath.deleteManualCuts(edited, cutIds: ["R001", "M001"])
        XCTAssertEqual(edited.cuts, [original.cuts[0]])
        XCTAssertTrue(edited.reviewedCutIds.isEmpty); XCTAssertTrue(edited.userTouchedCutIds.isEmpty)
        try edited.validate(durationMs: 10_000)
    }
    func testNumericEncodingMatchesKnownLittleEndianBits() throws {
        let array = try FeedbackNumericArray([Float(1), Float(-2), Float(0.5)], shape: [1, 3])
        XCTAssertEqual(array.data, "AACAPwAAAMAAAAA/")
        XCTAssertEqual(try array.decoded(), [1, -2, 0.5])
        XCTAssertEqual(try FeedbackNumericArray([Double(0), Double(0.25)], shape: [2]).decoded(), [0, 0.25])
        var corrupt = array; corrupt.shape = [4]
        XCTAssertThrowsError(try corrupt.decoded())
        corrupt = array; corrupt.byteOrder = "big-endian"
        XCTAssertThrowsError(try corrupt.decoded())
        XCTAssertThrowsError(try FeedbackNumericArray([Float.nan], shape: [1]))
    }
    private func bundle() throws -> JSONValue {
        let cuts = [EditableCut(id: "R001", coreStartMs: 1_000, coreEndMs: 3_000, keepStartMs: 0, keepEndMs: 5_000, confidence: 0.8)]
        let draft = EditorDraft(sourceRevision: "golden", cuts: cuts)
        let analysis = FeedbackAnalysis(timestamps: [0, 0.25], baseFeatures: [1, 2, 3, 4], featureNames: ["one", "two"],
            rallyProbabilities: [0.2, 0.8], serveProbabilities: [0.1, 0.9], deadStateProbabilities: [0.7, 0.1])
        var inference = ProjectArchive.initialInference(ranges: cuts)
        inference["futureModelMetadata"] = .object(["value": .string("preserve-me")])
        return try ProjectArchive.create(source: ProjectArchive.source(projectId: "golden", name: "known.mp4", sizeBytes: 500,
            lastModifiedMs: 0, duration: 10, width: 1920, height: 1080), initialInference: inference, draft: draft, analysis: analysis,
            generatedAt: Date(timeIntervalSince1970: 0))
    }
    func testFeedbackRoundTripRetainsFeaturesInferenceUserEditsAndIgnoredTime() throws {
        let initial = try bundle()
        var project = try ProjectArchive.importBundle(initial, expectedFeatureNames: ["one", "two"])
        project.draft = EditorMath.setIncluded(project.draft, cutId: "R001", included: false)
        project.draft = EditorMath.addManual(project.draft, startMs: 4_000, endMs: 8_000, bounds: .init(startMs: 0, endMs: 10_000))
        project.draft = EditorMath.addIgnored(project.draft, startMs: 5_000, endMs: 6_000, durationMs: 10_000, reason: "replay")
        var updated = try XCTUnwrap(project.feedback)
        try ProjectArchive.updateCorrections(&updated, draft: project.draft)
        XCTAssertEqual(updated["features"], initial["features"])
        XCTAssertEqual(updated["initialInference"], initial["initialInference"])
        let imported = try ProjectArchive.importBundle(updated)
        XCTAssertEqual(imported.draft.cuts, project.draft.cuts)
        XCTAssertEqual(imported.draft.ignoredIntervals, project.draft.ignoredIntervals)
        XCTAssertEqual(updated["corrections"]?["labels"]?["falsePositives"]?.array?.count, 1)
        XCTAssertEqual(updated["finalExportIntervals"]?.array?.count, 2)
        XCTAssertEqual(try ProjectArchive.retainedAnalysis(updated)?.baseFeatures, [1, 2, 3, 4])
    }
    func testFeedbackRejectsWrongModelTimelineShapeAndDraftBounds() throws {
        let original = try bundle()
        var invalid = original; var inference = invalid["initialInference"]!
        inference["modelId"] = .string("different-model"); invalid["initialInference"] = inference
        XCTAssertThrowsError(try ProjectArchive.importBundle(invalid))
        invalid = original; var features = invalid["features"]!
        features["timestamps"] = try FeedbackNumericArray([Double(0), Double(0.5)], shape: [2]).json
        invalid["features"] = features
        XCTAssertThrowsError(try ProjectArchive.importBundle(invalid))
        var draft = try ProjectArchive.importBundle(original).draft
        draft.cuts[0].keepEndMs = 10_001
        XCTAssertThrowsError(try draft.validate(durationMs: 10_000))
        XCTAssertThrowsError(try ProjectArchive.importBundle(original, expectedFeatureNames: ["different"]))
    }
    func testAtomicLocalProjectRoundTrip() throws {
        let project = try ProjectArchive.importBundle(bundle())
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".json")
        defer { try? FileManager.default.removeItem(at: path) }
        try project.save(to: path)
        XCTAssertEqual(try ProjectDocument.load(from: path), project)
    }
    func testChapterPreferencesSurviveLocalSaveAndFeedbackWithoutChangingFeatures() throws {
        var feedback = try bundle()
        var project = try ProjectArchive.importBundle(feedback)
        XCTAssertNil(project.draft.chapterOptions)
        let legacyData = try JSONEncoder().encode(project)
        XCTAssertNil(try JSONDecoder().decode(ProjectDocument.self, from: legacyData).draft.chapterOptions)
        let selected = YouTubeChapterOptions(includeRallyNumber: false, includeServeNumber: true,
            includeScore: false, includeServingTeam: true, includeSideSwitches: false)
        project.draft.chapterOptions = selected
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".json")
        defer { try? FileManager.default.removeItem(at: path) }
        try project.save(to: path)
        let restored = try ProjectDocument.load(from: path)
        XCTAssertEqual(restored.draft.chapterOptions, selected)
        let originalFeatures = feedback["features"], originalInference = feedback["initialInference"]
        feedback["corrections"]?["ui"] = .object(["preferences": .object(["futurePreference": .string("retain")])])
        try ProjectArchive.updateCorrections(&feedback, draft: restored.draft)
        XCTAssertEqual(try ProjectArchive.importBundle(feedback).draft.chapterOptions, selected)
        XCTAssertEqual(feedback["corrections"]?["ui"]?["preferences"]?["futurePreference"]?.string, "retain")
        XCTAssertEqual(feedback["features"], originalFeatures)
        XCTAssertEqual(feedback["initialInference"], originalInference)
        feedback["corrections"]?["ui"]?["preferences"]?["chapterOptions"]?["includeScore"] = .string("invalid")
        XCTAssertThrowsError(try ProjectArchive.importBundle(feedback))
    }
    func testNonGridGameStartUsesGlobalSourceGrid() throws {
        var original = try bundle()
        var source = original["source"]!
        source["gameWindow"] = .object(["start": .number(0.1), "end": .number(10)])
        original["source"] = source
        var features = original["features"]!; var inference = original["initialInference"]!
        let times = try FeedbackNumericArray([Double(0.25), Double(0.5)], shape: [2]).json
        features["timestamps"] = times; inference["timestamps"] = times
        original["features"] = features; original["initialInference"] = inference
        XCTAssertEqual(try ProjectArchive.retainedAnalysis(original)?.timestamps, [0.25, 0.5])
        let invalid = try FeedbackNumericArray([Double(0.1), Double(0.35)], shape: [2]).json
        features["timestamps"] = invalid; inference["timestamps"] = invalid
        original["features"] = features; original["initialInference"] = inference
        XCTAssertThrowsError(try ProjectArchive.retainedAnalysis(original))
    }
    func testFeedbackRecomputesScoreAfterIgnoredServe() throws {
        var original = try bundle()
        var project = try ProjectArchive.importBundle(original)
        var score = ScoreTracking(enabled: true)
        score = ScoreReducer.addServe(score, timestampMs: 1_000, side: .near)
        score = ScoreReducer.addServe(score, timestampMs: 2_000, side: .near)
        project.draft.scoreTracking = try score.jsonValue()
        try ProjectArchive.updateCorrections(&original, draft: project.draft)
        XCTAssertEqual(original["corrections"]?["scoreTracking"]?["derivedFinalScore"]?["team1Score"]?.double, 1)
        project.draft = EditorMath.addIgnored(project.draft, startMs: 1_500, endMs: 2_500, durationMs: 10_000, reason: "replay")
        try ProjectArchive.updateCorrections(&original, draft: project.draft)
        XCTAssertEqual(original["corrections"]?["scoreTracking"]?["derivedFinalScore"]?["team1Score"]?.double, 0)
        XCTAssertEqual(original["corrections"]?["scoreTracking"]?["state"]?["serveMarkers"]?.array?.count, 2)
    }
    #if canImport(CryptoKit)
    func testSourceFingerprintIncludesLittleEndianLengthAndAcceptsRenamedCopy() throws {
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".mp4")
        defer { try? FileManager.default.removeItem(at: path) }
        try Data("abc".utf8).write(to: path)
        let fingerprint = try ProjectArchive.sampledFingerprint(url: path)
        XCTAssertEqual(fingerprint, "sampled-sha256-v1:ce91dc5eec0139adf091900d225971d6ad246a845bad791b5693a9d0d55dd391")
        let source = ProjectArchive.source(projectId: "test", name: "old-name.mp4", sizeBytes: 3, lastModifiedMs: 1,
            fingerprint: fingerprint, duration: 10, width: 10, height: 10)
        XCTAssertNoThrow(try ProjectArchive.verifySource(url: path, source: source))
        try Data("abd".utf8).write(to: path)
        XCTAssertThrowsError(try ProjectArchive.verifySource(url: path, source: source))
    }
    #endif
    /// Allows the serial lab runner to collect a real Swift-produced archive for Android/browser validation.
    func testWriteInteroperabilityFixtureWhenRequested() throws {
        guard let path = ProcessInfo.processInfo.environment["VOLLEYSPLICE_FEEDBACK_FIXTURE"] else { return }
        var fixture = try bundle()
        var features = fixture["features"]!
        features["names"] = .array(FeatureSchema.base.map(JSONValue.string))
        features["columns"] = .number(Double(FeatureSchema.base.count))
        features["values"] = try FeedbackNumericArray([Float](repeating: 0, count: 2 * FeatureSchema.base.count), shape: [2, FeatureSchema.base.count]).json
        fixture["features"] = features
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(fixture).write(to: URL(fileURLWithPath: path), options: .atomic)
    }
}
