import Foundation
import XCTest
@testable import VolleyCore

final class PortableReviewPreferencesTests: XCTestCase {
    private var cut: EditableCut { .init(id: "R001", coreStartMs: 1000, coreEndMs: 3000) }
    private func fixture(_ name: String) throws -> JSONValue {
        let root = ProcessInfo.processInfo.environment["VOLLEY_FIXTURES"].map { URL(fileURLWithPath: $0) } ??
            URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent().appendingPathComponent("Fixtures")
        return try XCTUnwrap(JSONDecoder().decode(JSONValue.self,
            from: Data(contentsOf: root.appendingPathComponent("portable-review-preferences.json")))[name])
    }
    private func bundle(ui: JSONValue? = nil, alias: JSONValue? = nil) throws -> JSONValue {
        let draft = EditorDraft(sourceRevision: "synthetic", cuts: [cut])
        let analysis = FeedbackAnalysis(timestamps: [0, 0.25], baseFeatures: [1, 2], featureNames: ["synthetic"],
            rallyProbabilities: [0.2, 0.8], serveProbabilities: [0.1, 0.9], deadStateProbabilities: [0.7, 0.1])
        var bundle = try ProjectArchive.create(source: ProjectArchive.source(projectId: "synthetic", name: "fixture.mp4",
            sizeBytes: 1, lastModifiedMs: 0, duration: 10, width: 100, height: 100),
            initialInference: ProjectArchive.initialInference(ranges: [cut]), draft: draft, analysis: analysis)
        bundle["corrections"]?["ui"] = ui
        bundle["editorPreferences"] = alias.map { .object(["renderScoreOnFinalVideo": $0]) }
        return bundle
    }
    private func ui(_ preferences: [String: JSONValue], version: JSONValue? = .number(1)) -> JSONValue {
        var value: JSONValue = .object(["preferences": .object(preferences)])
        value["schemaVersion"] = version
        return value
    }
    private func imported(_ preferences: [String: JSONValue]) throws -> EditorDraft {
        try ProjectArchive.importBundle(bundle(ui: ui(preferences))).draft
    }
    private func folder() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func legacyLocalProject() throws -> ProjectDocument {
        var project = try ProjectArchive.importBundle(bundle())
        project.id = "legacy-local"; project.sourceName = "local-fixture.mp4"
        project.sourceBookmark = Data([1, 2, 3]); project.persistenceRevision = "retained-generation"
        project.draft = EditorMath.addManual(project.draft, startMs: 4000, endMs: 6000,
            bounds: .init(startMs: 0, endMs: 10000))
        // The previous inclusion toggle marked both inferred and manual cuts reviewed.
        project.draft.reviewedCutIds = ["R001", "M001"]
        project.draft.userTouchedCutIds = ["M001"]
        project.draft.beforePaddingMs = 1000; project.draft.afterPaddingMs = 500
        project.draft.joinGapMs = 1234; project.draft.updatedAtMs = 123456
        project.draft.pendingManualStartMs = 7000; project.draft.pendingIgnoreStartMs = 8000
        project.draft.ignoreReason = "retained reason"; project.draft.finalPreviewEnabled = true
        project.draft.playbackRate = 4; project.draft.confidenceReviewThreshold = 0.77
        project.draft.ignoredIntervals = [.init(id: "I001", startMs: 8000, endMs: 9000)]
        project.draft.suppressionDecisionOverrides = ["cleanup": "keep"]
        project.draft.suppressionScopeOverrides = ["cleanup": "veto-region"]
        project.draft.renderScoreOverlay = false; project.draft.renderScoreTimeline = true
        project.draft.fadeScoreOverlay = true
        project.draft.chapterOptions = .init(includeRallyNumber: false, includeServeNumber: true,
            includeScore: true, includeServingTeam: false, includeSideSwitches: true, includeCredit: false)
        project.draft.scoreTracking = try ScoreTracking(enabled: false, team1Name: "Retained team").jsonValue()
        project.feedback?["localExtension"] = .array([.null, .object(["retained": .bool(true)])])
        return project
    }

    func testLegacyLocalManualReviewUpgradePreservesAllOtherMetadataAndDoesNotRewriteSource() throws {
        let directory = try folder(); defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("legacy.volleyproject.json")
        let legacy = try legacyLocalProject()
        let data = try JSONEncoder().encode(legacy)
        try data.write(to: url) // Exact old local v8 representation, without portableUi.
        XCTAssertNil(try JSONDecoder().decode(JSONValue.self, from: data)["draft"]?["portableUi"])
        var expected = legacy; expected.draft.reviewedCutIds = ["R001"]
        let loaded = try ProjectDocument.load(from: url)
        XCTAssertEqual(loaded, expected)
        XCTAssertEqual(try Data(contentsOf: url), data)
        XCTAssertEqual(try ProjectPersistence.read(from: url).project, expected)
        XCTAssertEqual(try ProjectPersistence.listMetadata(from: url).sourceName, legacy.sourceName)
        // The direct local-decoder path used by VolleyProjectCheck performs the same upgrade.
        let decoded = try JSONDecoder().decode(ProjectDocument.self, from: data).migratingLegacyLocalReviewIds()
        try decoded.validate(); XCTAssertEqual(decoded, expected)
        var output = try XCTUnwrap(loaded.feedback)
        try ProjectArchive.updateCorrections(&output, draft: loaded.draft)
        let reimported = try ProjectArchive.importBundle(output)
        XCTAssertEqual(reimported.draft.reviewedCutIds, ["R001"])
        XCTAssertEqual(reimported.draft.cuts, legacy.draft.cuts)
    }

    func testLegacyManualReviewUpgradeRunsAfterCheckpointRecoveryAndSurvivesPublication() throws {
        let directory = try folder(); defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("checkpoint.volleyproject.json")
        var valid = try legacyLocalProject(); valid.draft.reviewedCutIds = ["R001"]
        let base = try ProjectPersistence.publish(ProjectPersistence.stage(valid, to: url))
        var latest = base; latest.sourceName = "checkpoint-fixture.mp4"
        latest.sourceBookmark = Data([7, 8, 9]); latest.draft.updatedAtMs += 1
        latest.draft.chapterOptions?.includeCredit = true
        try ProjectPersistence.checkpoint(latest, to: url)
        let journalURL = ProjectPersistence.checkpointURL(for: url)
        var journal = try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: journalURL))
        var entries = try XCTUnwrap(journal["entries"]?.array)
        entries[0]["draft"]?["reviewedCutIds"] = .array([.string("R001"), .string("M001")])
        journal["entries"] = .array(entries)
        let oldJournal = try JSONEncoder().encode(journal)
        try oldJournal.write(to: journalURL)
        let baseBytes = try Data(contentsOf: url)
        let loaded = try ProjectDocument.load(from: url)
        XCTAssertEqual(loaded, latest)
        XCTAssertEqual(try Data(contentsOf: url), baseBytes)
        XCTAssertEqual(try Data(contentsOf: journalURL), oldJournal)
        try ProjectPersistence.checkpoint(loaded, to: url)
        let published = try ProjectPersistence.publish(ProjectPersistence.stage(loaded, to: url), current: loaded)
        XCTAssertEqual(try ProjectDocument.load(from: url), published)
        XCTAssertEqual(published.draft, latest.draft)
    }

    func testLegacyLocalMigrationDoesNotSanitizeUnknownIdsOrModernDrafts() throws {
        let directory = try folder(); defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("invalid-local.json")
        var legacy = try legacyLocalProject(); legacy.draft.reviewedCutIds.insert("missing")
        let migrated = legacy.migratingLegacyLocalReviewIds()
        XCTAssertEqual(migrated.draft.reviewedCutIds, ["R001", "missing"])
        try JSONEncoder().encode(legacy).write(to: url)
        XCTAssertThrowsError(try ProjectDocument.load(from: url))
        var modern = try legacyLocalProject(); modern.draft.portableUi = ui([:])
        XCTAssertEqual(modern.migratingLegacyLocalReviewIds(), modern)
        try JSONEncoder().encode(modern).write(to: url)
        XCTAssertThrowsError(try ProjectDocument.load(from: url))
        var otherVersion = try legacyLocalProject(); otherVersion.draft.version = 9
        XCTAssertEqual(otherVersion.migratingLegacyLocalReviewIds(), otherVersion)
        var otherSchema = try legacyLocalProject(); otherSchema.schema = "other-format"
        XCTAssertEqual(otherSchema.migratingLegacyLocalReviewIds(), otherSchema)
    }

    func testLocalUpgradeDoesNotWeakenWireManualReviewIdentityRejection() throws {
        var draft = try legacyLocalProject().draft; draft.reviewedCutIds = ["R001"]
        var input = try bundle()
        try ProjectArchive.updateCorrections(&input, draft: draft)
        for version in [nil, JSONValue.number(1), .number(2)] as [JSONValue?] {
            input["corrections"]?["ui"] = ui(["reviewedCutIds": .array([.string("M001")])], version: version)
            XCTAssertThrowsError(try ProjectArchive.importData(JSONEncoder().encode(input)))
        }
    }

    private func futureSplitProject() throws -> ProjectDocument {
        var input = try bundle()
        let original = try ProjectArchive.importBundle(input)
        let split = try XCTUnwrap(EditorMath.splitCut(original.draft, cutId: "R001", positionMs: 2000,
            bounds: original.gameWindow))
        XCTAssertEqual(split.cuts.map(\.id), ["R001", "R002"])
        try ProjectArchive.updateCorrections(&input, draft: split)
        var future = try fixture("futureUi")
        future["preferences"]?["reviewedCutIds"] = .array([.string("R002")])
        input["corrections"]?["ui"] = future
        return try ProjectArchive.importData(JSONEncoder().encode(input))
    }

    func testFutureResetKeepsLocalDataLosslessButRejectsEveryArchiveExportPathWithStaleReviewIds() throws {
        let directory = try folder(); defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("future-reset.volleyproject.json")
        var project = try futureSplitProject()
        let envelope = project.draft.portableUi
        project.draft = try ProjectArchive.restoredModelDraft(project)
        XCTAssertEqual(project.draft.cuts.map(\.id), ["R001"])
        XCTAssertEqual(project.draft.portableUi, envelope)
        try project.save(to: url)
        let reopened = try ProjectDocument.load(from: url)
        XCTAssertEqual(reopened, project)
        XCTAssertEqual(reopened.draft.portableUi, envelope)
        // Exercise both opaque early returns: retained bundle and draft-only envelope.
        for retained in [envelope, nil, ui([:])] as [JSONValue?] {
            var output = try XCTUnwrap(reopened.feedback)
            output["corrections"]?["ui"] = retained
            let before = output
            XCTAssertThrowsError(try ProjectArchive.updateCorrections(&output, draft: reopened.draft)) { error in
                XCTAssertTrue(error.localizedDescription.contains("Cannot export"))
                XCTAssertTrue(error.localizedDescription.contains("unchanged copy"))
                XCTAssertTrue(error.localizedDescription.contains("local project is preserved"))
            }
            XCTAssertEqual(output, before)
        }
        XCTAssertThrowsError(try PortableReviewPreferences.compose(reopened.draft))
        XCTAssertEqual(try ProjectDocument.load(from: url), reopened)
    }

    func testFutureReviewIdentityMustStillBeInferredOnExportAfterOriginChange() throws {
        var project = try futureSplitProject()
        let envelope = project.draft.portableUi
        project.draft.cuts[1].origin = .manual
        try project.validate() // Local state remains valid and the envelope stays opaque.
        var output = try XCTUnwrap(project.feedback)
        let before = output
        XCTAssertThrowsError(try ProjectArchive.updateCorrections(&output, draft: project.draft))
        XCTAssertEqual(output, before); XCTAssertEqual(project.draft.portableUi, envelope)
    }

    func testFutureResetWithCurrentReviewIdentityExportsAndReimportsWholeEnvelope() throws {
        var future = try fixture("futureUi")
        future["preferences"]?["reviewedCutIds"] = .array([.string("R001")])
        var project = try ProjectArchive.importData(JSONEncoder().encode(bundle(ui: future)))
        project.draft = try ProjectArchive.restoredModelDraft(project)
        var output = try XCTUnwrap(project.feedback)
        try ProjectArchive.updateCorrections(&output, draft: project.draft)
        let reimported = try ProjectArchive.importData(JSONEncoder().encode(output))
        XCTAssertEqual(output["corrections"]?["ui"], future)
        XCTAssertEqual(reimported.draft.portableUi, future)
        XCTAssertEqual(reimported.draft.cuts, project.draft.cuts)
    }

    func testSharedV1FixtureActivatesNativeSettingsAndRetainsNestedExtensions() throws {
        let ui = try fixture("portableUi"), project = try ProjectArchive.importBundle(bundle(ui: ui))
        let draft = project.draft
        XCTAssertEqual(draft.portableUi, ui)
        XCTAssertTrue(draft.finalPreviewEnabled); XCTAssertTrue(draft.renderScoreOverlay)
        XCTAssertTrue(draft.renderScoreTimeline); XCTAssertEqual(draft.fadeScoreOverlay, true)
        XCTAssertEqual(draft.playbackRate, 1.5); XCTAssertEqual(draft.confidenceReviewThreshold, 0.73)
        XCTAssertEqual(draft.reviewedCutIds, ["R001"])
        XCTAssertEqual(draft.chapterOptions?.includeScore, false)
        XCTAssertEqual(draft.chapterOptions?.includeCredit, false)
        let output = try PortableReviewPreferences.compose(draft)
        XCTAssertEqual(output, ui)
        XCTAssertEqual(output["preferences"]?["renderScoreOnPreviewVideo"], .bool(false))
        XCTAssertNil(PortableReviewPreferences.notice(draft))
    }

    func testFutureFixtureStaysWholeAcrossEditsAndDoesNotActivateChapterData() throws {
        var future = try fixture("futureUi")
        future["preferences"]?["renderScoreOnFinalVideo"] = .bool(true)
        future["preferences"]?["renderScoreTimeline"] = .bool(false)
        future["preferences"]?["fadeScoreOverlay"] = .bool(true)
        future["preferences"]?["cutPreviewEnabled"] = .bool(true)
        future["preferences"]?["reviewedCutIds"] = .array([.string("R001")])
        future["preferences"]?["chapterOptions"] = .object(["includeScore": .array([.null])])
        var project = try ProjectArchive.importBundle(bundle(ui: future, alias: .bool(false)))
        XCTAssertFalse(project.draft.renderScoreOverlay)
        XCTAssertTrue(project.draft.renderScoreTimeline)
        XCTAssertFalse(project.draft.finalPreviewEnabled)
        XCTAssertNil(project.draft.fadeScoreOverlay); XCTAssertNil(project.draft.chapterOptions)
        XCTAssertEqual(project.draft.playbackRate, 1); XCTAssertTrue(project.draft.reviewedCutIds.isEmpty)
        XCTAssertNotNil(PortableReviewPreferences.notice(project.draft))
        project.draft.playbackRate = 2; project.draft.fadeScoreOverlay = true
        project.draft.chapterOptions = .init(includeRallyNumber: false, includeServeNumber: true,
            includeScore: false, includeServingTeam: true, includeSideSwitches: false)
        var output = try XCTUnwrap(project.feedback)
        try ProjectArchive.updateCorrections(&output, draft: project.draft)
        XCTAssertEqual(output["corrections"]?["ui"], future)
        let again = try ProjectArchive.importData(JSONEncoder().encode(output))
        XCTAssertEqual(again.draft.portableUi, future); XCTAssertEqual(again.draft.playbackRate, 1)
        XCTAssertNil(again.draft.chapterOptions)
    }

    func testV1DefaultsDifferFromAbsentAndUnversionedIosDefaults() throws {
        let versioned = try ProjectArchive.importBundle(bundle(ui: .object(["schemaVersion": .number(1)]))).draft
        XCTAssertTrue(versioned.renderScoreOverlay); XCTAssertFalse(versioned.renderScoreTimeline)
        XCTAssertEqual(versioned.fadeScoreOverlay, false); XCTAssertEqual(versioned.playbackRate, 1)
        XCTAssertFalse(versioned.finalPreviewEnabled); XCTAssertTrue(versioned.reviewedCutIds.isEmpty)
        XCTAssertEqual(versioned.confidenceReviewThreshold, 0.7)
        for value in [nil, try fixture("legacyIosUi"), .object([:])] as [JSONValue?] {
            let legacy = try ProjectArchive.importBundle(bundle(ui: value)).draft
            XCTAssertTrue(legacy.renderScoreOverlay); XCTAssertTrue(legacy.renderScoreTimeline)
            XCTAssertNil(legacy.fadeScoreOverlay); XCTAssertEqual(legacy.playbackRate, 1)
        }
        let legacyChapters = try ProjectArchive.importBundle(bundle(ui: fixture("legacyIosUi"))).draft
        XCTAssertEqual(legacyChapters.chapterOptions?.includeServingTeam, false)
    }

    func testCanonicalFinalFlagWinsAliasAndPreviewNeverControlsNativeOutput() throws {
        for final in [true, false] {
            let draft = try ProjectArchive.importBundle(bundle(ui: ui([
                "renderScoreOnFinalVideo": .bool(final), "renderScoreOnPreviewVideo": .bool(!final)
            ]), alias: .bool(!final))).draft
            XCTAssertEqual(draft.renderScoreOverlay, final)
        }
        for value in [nil, ui([:]), ui([:], version: nil), try fixture("futureUi")] as [JSONValue?] {
            XCTAssertFalse(try ProjectArchive.importBundle(bundle(ui: value, alias: .bool(false))).draft.renderScoreOverlay)
        }
        XCTAssertTrue(try imported(["renderScoreOnPreviewVideo": .bool(false)]).renderScoreOverlay)
    }

    func testStrictAliasValidationEvenWhenCanonicalShadowsIt() throws {
        for invalid in [JSONValue.null, .string("false"), .number(0), .array([]), .object([:])] {
            XCTAssertThrowsError(try ProjectArchive.importBundle(bundle(ui: ui(["renderScoreOnFinalVideo": .bool(true)]), alias: invalid)))
        }
        for invalid in [JSONValue.null, .array([]), .bool(false)] {
            var input = try bundle(); input["editorPreferences"] = invalid
            XCTAssertThrowsError(try ProjectArchive.importBundle(input))
        }
    }

    func testEverySharedInvalidFixtureFailsBeforeImportPublicationIncludingFutureVersion() throws {
        for (index, invalid) in try XCTUnwrap(fixture("invalidUi").array).enumerated() {
            XCTAssertThrowsError(try ProjectArchive.importBundle(bundle(ui: invalid)), "invalid fixture \(index)")
            if case .object = invalid, invalid["schemaVersion"] == nil {
                var future = invalid; future["schemaVersion"] = .number(2)
                XCTAssertThrowsError(try ProjectArchive.importBundle(bundle(ui: future)), "future invalid fixture \(index)")
            }
        }
    }

    func testRatesKeepExactDoubleAcrossArchiveTransportAndHaveHonestMenuLabels() throws {
        for rate in [0.25, 0.3333333333333333, 1, 1.5, 1.2345678901234567, 8] {
            let draft = try imported(["playbackRate": .number(rate)])
            XCTAssertEqual(draft.playbackRate, rate)
            XCTAssertEqual(try PortableReviewPreferences.compose(draft)["preferences"]?["playbackRate"], .number(rate))
            XCTAssertEqual(PortableReviewPreferences.playbackRates(rate), Array(Set([1, 1.5, 2, 4, 8, rate])).sorted())
        }
        XCTAssertEqual(PortableReviewPreferences.playbackRateLabel(1.5), "1.5x")
        XCTAssertEqual(PortableReviewPreferences.playbackRateLabel(0.25), "0.25x")
        XCTAssertEqual(PortableReviewPreferences.playbackRateLabel(1), "1x")
        for rate in [0.249999, 8.000001, Double.nan, .infinity, -.infinity] {
            XCTAssertThrowsError(try imported(["playbackRate": .number(rate)]))
        }
    }

    func testThresholdRetainsExactImportedNumberUntilActiveFloatChanges() throws {
        let precise = 0.7312345678901234
        var draft = try imported(["confidenceReviewThreshold": .number(precise)])
        XCTAssertEqual(try PortableReviewPreferences.compose(draft)["preferences"]?["confidenceReviewThreshold"], .number(precise))
        draft.confidenceReviewThreshold = 0.8
        XCTAssertEqual(try PortableReviewPreferences.compose(draft)["preferences"]?["confidenceReviewThreshold"], .number(0.8))
        for threshold in [0.0, 1.0] { XCTAssertNoThrow(try imported(["confidenceReviewThreshold": .number(threshold)])) }
        for threshold in [-0.01, 1.01, Double.nan, .infinity] {
            XCTAssertThrowsError(try imported(["confidenceReviewThreshold": .number(threshold)]))
        }
    }

    func testChapterEditsMergeKnownKeysAndRetainAllUnknownNestedData() throws {
        let original = try fixture("portableUi")
        var draft = try ProjectArchive.importBundle(bundle(ui: original)).draft
        draft.chapterOptions?.includeScore = true; draft.chapterOptions?.includeCredit = true
        let composed = try PortableReviewPreferences.compose(draft)
        XCTAssertEqual(composed["preferences"]?["chapterOptions"]?["includeScore"], .bool(true))
        XCTAssertEqual(composed["preferences"]?["chapterOptions"]?["includeCredit"], .bool(true))
        XCTAssertEqual(composed["preferences"]?["chapterOptions"]?["futureFormatting"], original["preferences"]?["chapterOptions"]?["futureFormatting"])
        XCTAssertEqual(composed["preferences"]?["futurePreference"], original["preferences"]?["futurePreference"])
        XCTAssertEqual(composed["futurePanel"], original["futurePanel"])
        let partial = try imported(["chapterOptions": .object(["includeCredit": .bool(false), "extension": .null])])
        XCTAssertEqual(partial.chapterOptions?.includeCredit, false)
        XCTAssertEqual(try PortableReviewPreferences.compose(partial)["preferences"]?["chapterOptions"]?["extension"], .null)
        let extensionOnly = try imported(["chapterOptions": .object(["extension": .null])])
        XCTAssertNil(extensionOnly.chapterOptions) // Keep the native score-aware defaults dynamic.
        XCTAssertEqual(try PortableReviewPreferences.compose(extensionOnly)["preferences"]?["chapterOptions"]?["extension"], .null)
        XCTAssertThrowsError(try imported(["chapterOptions": .object(["includeScore": .number(1)])]))
    }

    func testActualArchiveExportAndReimportPreservePreferencesAndModelPayloads() throws {
        let original = try bundle(ui: fixture("portableUi"), alias: .bool(false))
        let first = try ProjectArchive.importData(JSONEncoder().encode(original))
        var output = try XCTUnwrap(first.feedback), draft = first.draft
        draft.renderScoreOverlay = false // Timeline/fade choices survive output being disabled.
        draft.playbackRate = 1.25
        try ProjectArchive.updateCorrections(&output, draft: draft)
        let again = try ProjectArchive.importData(JSONEncoder().encode(output))
        XCTAssertEqual(again.draft.playbackRate, 1.25); XCTAssertFalse(again.draft.renderScoreOverlay)
        XCTAssertTrue(again.draft.renderScoreTimeline); XCTAssertEqual(again.draft.fadeScoreOverlay, true)
        XCTAssertEqual(again.draft.reviewedCutIds, ["R001"])
        XCTAssertEqual(output["editorPreferences"]?["renderScoreOnFinalVideo"], .bool(false))
        XCTAssertEqual(again.draft.cuts, first.draft.cuts)
        for key in ["source", "features", "initialInference"] { XCTAssertEqual(output[key], original[key]) }
        var twice = output; try ProjectArchive.updateCorrections(&twice, draft: again.draft)
        XCTAssertEqual(twice["corrections"]?["ui"], output["corrections"]?["ui"])
    }

    func testRetainedBundleUpdatesMergeUnknownSiblingsRecursivelyAndNeverDowngradeFutureUi() throws {
        let initial = try bundle(ui: fixture("portableUi"))
        let project = try ProjectArchive.importBundle(initial)
        var updated = initial
        updated["corrections"]?["ui"]?["futurePanel"]?["extra"] = .null
        updated["corrections"]?["ui"]?["preferences"]?["futurePreference"]?["new"] = .array([.bool(true)])
        updated["corrections"]?["ui"]?["preferences"]?["chapterOptions"]?["futureFormatting"]?["new"] = .string("retained")
        try ProjectArchive.updateCorrections(&updated, draft: project.draft)
        XCTAssertEqual(updated["corrections"]?["ui"]?["futurePanel"]?["extra"], .null)
        XCTAssertEqual(updated["corrections"]?["ui"]?["preferences"]?["futurePreference"]?["new"], .array([.bool(true)]))
        XCTAssertEqual(updated["corrections"]?["ui"]?["preferences"]?["chapterOptions"]?["futureFormatting"]?["new"], .string("retained"))
        updated["corrections"]?["ui"] = try fixture("futureUi")
        try ProjectArchive.updateCorrections(&updated, draft: project.draft)
        XCTAssertEqual(updated["corrections"]?["ui"], try fixture("futureUi"))
    }

    func testLocalReopenCheckpointAndPreparedSnapshotPreserveActiveValuesAndOpaqueEnvelope() throws {
        let directory = try folder(); defer { try? FileManager.default.removeItem(at: directory) }
        for name in ["portableUi", "futureUi"] {
            let url = directory.appendingPathComponent(name + ".json")
            var project = try ProjectArchive.importBundle(bundle(ui: fixture(name)))
            project.draft = PortableReviewPreferences.onOpen(project.draft, deviceFade: false)
            project = try ProjectPersistence.publish(ProjectPersistence.stage(project, to: url))
            let prepared = try ProjectPersistence.stage(project, to: url)
            project.draft.playbackRate = 1.2345678901234567; project.draft.fadeScoreOverlay = true
            project.draft.renderScoreOverlay = false; project.draft.confidenceReviewThreshold = 0.8
            try ProjectPersistence.checkpoint(project, to: url)
            XCTAssertEqual(try ProjectDocument.load(from: url).draft, project.draft)
            let published = try ProjectPersistence.publish(prepared, current: project)
            let reopened = try ProjectDocument.load(from: url)
            XCTAssertEqual(reopened, published)
            XCTAssertEqual(PortableReviewPreferences.onOpen(reopened.draft, deviceFade: false), project.draft)
            XCTAssertEqual(reopened.draft.portableUi, try fixture(name))
            // Video queue snapshots use this same ProjectDocument Codable representation.
            let snapshot = try JSONDecoder().decode(ProjectDocument.self, from: JSONEncoder().encode(reopened))
            XCTAssertEqual(snapshot.draft, project.draft)
        }
    }

    func testLegacyVersion8DraftStillDecodesWithoutPortableUiAndWithFloatRate() throws {
        let project = try ProjectArchive.importBundle(bundle())
        var json = try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(project))
        json["draft"]?["portableUi"] = nil; json["draft"]?["fadeScoreOverlay"] = nil
        json["draft"]?["playbackRate"] = .number(Double(Float(1.5)))
        let legacy = try JSONDecoder().decode(ProjectDocument.self, from: JSONEncoder().encode(json))
        XCTAssertEqual(legacy.draft.version, 8); XCTAssertNil(legacy.draft.portableUi)
        XCTAssertEqual(legacy.draft.playbackRate, 1.5)
        XCTAssertEqual(PortableReviewPreferences.onOpen(legacy.draft, deviceFade: true).fadeScoreOverlay, true)
    }

    func testProjectFadeAlwaysWinsDevicePreferenceAfterOpenAndAfterEditing() throws {
        for fade in [false, true] {
            let draft = try imported(["fadeScoreOverlay": .bool(fade)])
            let opened = PortableReviewPreferences.onOpen(draft, deviceFade: !fade)
            XCTAssertEqual(opened.fadeScoreOverlay, fade)
            XCTAssertEqual(PortableReviewPreferences.onOpen(opened, deviceFade: !fade), opened)
        }
        let v1 = try imported([:])
        XCTAssertEqual(PortableReviewPreferences.onOpen(v1, deviceFade: true).fadeScoreOverlay, false)
        var legacy = try ProjectArchive.importBundle(bundle()).draft
        legacy = PortableReviewPreferences.onOpen(legacy, deviceFade: true)
        legacy.fadeScoreOverlay = false
        XCTAssertEqual(PortableReviewPreferences.onOpen(legacy, deviceFade: true).fadeScoreOverlay, false)
    }

    func testResetKeepsScoreFlagsAndUnknownExtensionsWithoutReactivatingStaleSupportedValues() throws {
        var envelope = try fixture("portableUi")
        envelope["preferences"]?["confidenceReviewThreshold"] = .number(0.70000000000001)
        var project = try ProjectArchive.importBundle(bundle(ui: envelope))
        project.draft.renderScoreOverlay = false; project.draft.renderScoreTimeline = true
        project.draft.playbackRate = 3.25; project.draft.confidenceReviewThreshold = 0.8
        let originalScore = project.draft.scoreTracking
        let reset = try ProjectArchive.restoredModelDraft(project)
        XCTAssertEqual(reset.playbackRate, 1); XCTAssertEqual(reset.confidenceReviewThreshold, 0.7)
        XCTAssertTrue(reset.reviewedCutIds.isEmpty); XCTAssertFalse(reset.finalPreviewEnabled)
        XCTAssertFalse(reset.renderScoreOverlay); XCTAssertTrue(reset.renderScoreTimeline)
        XCTAssertEqual(reset.fadeScoreOverlay, true); XCTAssertEqual(reset.scoreTracking, originalScore)
        XCTAssertNil(reset.chapterOptions)
        var output = try XCTUnwrap(project.feedback)
        try ProjectArchive.updateCorrections(&output, draft: reset)
        let again = try ProjectArchive.importBundle(output).draft
        XCTAssertEqual(again.playbackRate, 1); XCTAssertEqual(again.confidenceReviewThreshold, 0.7)
        XCTAssertEqual(again.portableUi?["preferences"]?["confidenceReviewThreshold"], .number(0.7))
        XCTAssertTrue(again.reviewedCutIds.isEmpty); XCTAssertFalse(again.renderScoreOverlay)
        XCTAssertNil(again.chapterOptions)
        XCTAssertEqual(again.portableUi?["preferences"]?["renderScoreOnPreviewVideo"], .bool(false))
        XCTAssertEqual(again.portableUi?["futurePanel"], project.draft.portableUi?["futurePanel"])
        XCTAssertEqual(again.portableUi?["preferences"]?["chapterOptions"]?["futureFormatting"],
            project.draft.portableUi?["preferences"]?["chapterOptions"]?["futureFormatting"])
        XCTAssertEqual(reset.portableUi?["preferences"]?["confidenceReviewThreshold"], nil)
        // A pre-extension local draft still retains its feedback's chapter extensions on reset.
        project.draft.portableUi = nil
        XCTAssertNotNil(try ProjectArchive.restoredModelDraft(project).portableUi?["preferences"]?["chapterOptions"]?["futureFormatting"])
    }

    func testResetPreservesFutureEnvelopeWhole() throws {
        var project = try ProjectArchive.importBundle(bundle(ui: fixture("futureUi")))
        project.draft.playbackRate = 4
        let reset = try ProjectArchive.restoredModelDraft(project)
        XCTAssertEqual(reset.portableUi, try fixture("futureUi"))
        XCTAssertEqual(try PortableReviewPreferences.compose(reset), try fixture("futureUi"))
    }

    func testReviewIdentitiesRejectMissingManualDuplicatesAndUtf16Overflow() throws {
        for ids in [[JSONValue.string("missing")], [.string("R001"), .string("R001")], [.string("")],
                    [.string(String(repeating: "😀", count: 129))], [.number(1)], [.null]] {
            XCTAssertThrowsError(try imported(["reviewedCutIds": .array(ids)]))
        }
        var input = try bundle(ui: ui(["reviewedCutIds": .array([.string("R001")])]))
        input["corrections"]?["correctedRanges"] = .array([.object([
            "id": .string("R001"), "origin": .string("manual"), "included": .bool(true),
            "coreStart": .number(1), "coreEnd": .number(3), "keepStart": .number(1), "keepEnd": .number(3), "confidence": .number(1)
        ])])
        XCTAssertThrowsError(try ProjectArchive.importBundle(input))
        let ids = (0..<2048).map { "R\($0)" }
        let cuts = ids.map { EditableCut(id: $0, coreStartMs: 1000, coreEndMs: 2000) }
        XCTAssertNoThrow(try PortableReviewPreferences.validate(ui(["reviewedCutIds": .array(ids.map(JSONValue.string))]), cuts: cuts))
        XCTAssertThrowsError(try PortableReviewPreferences.validate(ui(["reviewedCutIds": .array((ids + ["extra"]).map(JSONValue.string))]), cuts: cuts))
        let longId = String(repeating: "😀", count: 128)
        XCTAssertNoThrow(try PortableReviewPreferences.validate(ui(["reviewedCutIds": .array([.string(longId)])]),
            cuts: [.init(id: longId, coreStartMs: 1000, coreEndMs: 2000)]))
    }

    func testJsonSafetyDepthNodesStringsUnsafeKeysAndFiniteNumbers() throws {
        func validate(_ value: JSONValue) throws { try PortableReviewPreferences.validate(value, cuts: [cut]) }
        for key in ["__proto__", "constructor", "prototype"] {
            XCTAssertThrowsError(try validate(.object(["extension": .array([.object([key: .null])])])))
        }
        for number in [Double.nan, .infinity, -.infinity] {
            XCTAssertThrowsError(try validate(.object(["extension": .array([.number(number)])])))
        }
        var nested = JSONValue.null
        for _ in 0..<15 { nested = .array([nested]) }
        XCTAssertNoThrow(try validate(.object(["extension": nested])))
        XCTAssertThrowsError(try validate(.object(["extension": .array([nested]) ])))
        XCTAssertNoThrow(try validate(.object(["extension": .array(Array(repeating: .null, count: 4094))])))
        XCTAssertThrowsError(try validate(.object(["extension": .array(Array(repeating: .null, count: 4095))])))
        XCTAssertNoThrow(try validate(.object(["extension": .string(String(repeating: "😀", count: 8192))])))
        XCTAssertThrowsError(try validate(.object(["extension": .string(String(repeating: "😀", count: 8193))])))
    }

    func testCompactByteLimitAndExportCompositionOverflowFailWithoutTruncatingOrMutatingBundle() throws {
        var boundary: JSONValue = .object(["a": .string(String(repeating: "a", count: 16384)),
            "b": .string(String(repeating: "b", count: 16384)), "c": .string(String(repeating: "c", count: 16384)), "d": .string("")])
        let encoder = JSONEncoder(); encoder.outputFormatting = [.withoutEscapingSlashes]
        let remaining = 65536 - (try encoder.encode(boundary).count)
        boundary["d"] = .string(String(repeating: "d", count: remaining))
        XCTAssertEqual(try encoder.encode(boundary).count, 65536)
        XCTAssertNoThrow(try PortableReviewPreferences.validate(boundary, cuts: [cut]))
        let project = try ProjectArchive.importBundle(bundle(ui: boundary))
        var output = try XCTUnwrap(project.feedback)
        let original = output
        XCTAssertThrowsError(try ProjectArchive.updateCorrections(&output, draft: project.draft))
        XCTAssertEqual(output, original)
        boundary["d"] = .string(String(repeating: "d", count: remaining + 1))
        XCTAssertThrowsError(try ProjectArchive.importBundle(bundle(ui: boundary)))
    }

    func testSchemaVersionIsPositiveSafeIntegerAndKnownFieldsAreStrict() throws {
        for version in [JSONValue.number(0), .number(-1), .number(1.5), .number(9007199254740992), .bool(true), .null, .string("1")] {
            XCTAssertThrowsError(try ProjectArchive.importBundle(bundle(ui: ui([:], version: version))))
        }
        XCTAssertNoThrow(try ProjectArchive.importBundle(bundle(ui: ui([:], version: .number(9007199254740991)))))
        for key in ["cutPreviewEnabled", "renderScoreOnPreviewVideo", "renderScoreOnFinalVideo", "renderScoreTimeline", "fadeScoreOverlay"] {
            for invalid in [JSONValue.number(1), .string("true"), .null, .array([])] {
                XCTAssertThrowsError(try imported([key: invalid]))
            }
        }
    }

    func testLocalPersistenceRejectsUnsafeRetainedEnvelopesAndInvalidActiveSettings() throws {
        let directory = try folder(); defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("unsafe.json")
        let original = try ProjectArchive.importBundle(bundle(ui: fixture("portableUi")))
        for invalid in [JSONValue.null, .array([]), ui(["playbackRate": .number(0)]), .object(["constructor": .null])] {
            var project = original; project.draft.portableUi = invalid
            XCTAssertThrowsError(try ProjectPersistence.stage(project, to: url))
            try JSONEncoder().encode(project).write(to: url)
            if invalid != .null { XCTAssertThrowsError(try ProjectDocument.load(from: url)) }
            project = original; project.feedback?["corrections"]?["ui"] = invalid
            try JSONEncoder().encode(project).write(to: url)
            XCTAssertThrowsError(try ProjectDocument.load(from: url))
        }
        var invalid = original; invalid.draft.playbackRate = 9
        XCTAssertThrowsError(try invalid.validate())
        invalid = original; invalid.draft.confidenceReviewThreshold = .nan
        XCTAssertThrowsError(try invalid.validate())
        invalid = original; invalid.draft.reviewedCutIds = ["missing"]
        XCTAssertThrowsError(try invalid.validate())
    }
}
