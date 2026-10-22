package com.volleycut.nativeanalysis

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.nio.file.Files

class ExplicitRallyGroupsTest {
    private val project = NativeProject(
        id = "synthetic-split", source = ProjectSource("content://fixture/split", "synthetic.mp4", 1024, 100, "video/mp4"),
        media = AnalysisTypes.MediaInfo(40.0, 1920, 1080, 0, "video/avc", null),
        roi = AnalysisTypes.Roi(0.0, 0.0, 1.0, 1.0, "fixture"), status = ProjectStatus.READY,
        ranges = listOf(SeedRange(5_000, 20_000, .95f, ProductionEnsemble.BOTH_MODELS)),
        createdAtMs = 100, updatedAtMs = 100,
    )
    private val seed = requireNotNull(project.editorSeed())
    private fun draft(manual: Boolean = false) = EditorMath.newDraft(seed).let { d ->
        if (!manual) d else d.copy(cuts = d.cuts.map { it.copy(id = "M001", origin = CutOrigin.MANUAL,
            keepStartMs = it.coreStartMs, keepEndMs = it.coreEndMs) })
    }
    private fun groups(d: EditorDraft) = EditorMath.editableRallyGroups(d.cuts, EditorMath.finalIntervals(d),
        d.scoreTracking.serveMarkers.takeIf { d.scoreTracking.enabled }.orEmpty())
    private fun split(d: EditorDraft, id: String = d.cuts.first().id, time: Long = 12_000) =
        requireNotNull(EditorMath.splitCut(d, id, time, 0, seed.durationMs)).draft
    private fun geometry(d: EditorDraft) = EditorMath.finalIntervals(d).map { it.startMs to it.endMs }
    private fun bundle(d: EditorDraft, p: NativeProject = project) = ModelFeedbackExporter.createBundle(
        p, d, EditorMath.finalIntervals(d), null, null)
    private fun fixture(): JSONObject = generateSequence(File(System.getProperty("user.dir"))) { it.parentFile }
        .map { File(it, "tests/fixtures/explicit-rally-groups.json") }.first { it.isFile }.let { JSONObject(it.readText()) }

    @Test fun highConfidenceAndManualSplitRemainIndependentDespiteContinuousExport() {
        for (manual in listOf(false, true)) {
            val base = draft(manual)
            val result = split(base)
            assertEquals("Two independently editable halves", 2, groups(result).size)
            assertEquals(base.cuts.first().id, result.cuts.first().id)
            assertEquals(2, result.cuts.map { it.editableGroupId }.distinct().size)
            assertEquals(geometry(base), geometry(result))
            val rightId = result.cuts.last().id
            assertEquals(rightId, EditorMath.rallySelectionAtPlayhead(groups(result), rightId, 0, false))
            assertEquals(result.cuts.first().id,
                EditorMath.rallySelectionAtPlayhead(groups(result), result.cuts.first().id, 19_000, false))
            assertEquals(rightId, EditorMath.rallySelectionAtPlayhead(groups(result), result.cuts.first().id, 19_000, true))
            // Counterexample: the old grouping sees only output connectivity and regroups both.
            assertEquals(1, groups(result.copy(cuts = result.cuts.map { it.copy(editableGroupId = null) })).size)
        }
    }

    @Test fun reviewConfidenceScoreReseedTrimAndPaddingDoNotEraseExplicitIdentity() {
        val base = split(draft())
        val expected = groups(base).map { it.cutIds }
        val marker = ServeMarker("S001", 5_000, ServingSide.NEAR, ServeMarkerOrigin.MODEL, rallyId = "R001")
        for (enabled in listOf(false, true, false)) {
            var changed = base.copy(reviewedCutIds = base.cuts.mapTo(hashSetOf()) { it.id }, confidenceReviewThreshold = 0f,
                scoreTracking = ScoreReducer.seedModelMarkers(base.scoreTracking.copy(enabled = enabled,
                    serveMarkers = listOf(marker)), seed.servingSide, seed.sideSwitch, seed.sideSwitchEnabled))
            changed = EditorMath.reconcileTouchedCuts(EditorMath.setCoreEnd(changed, "R001", 10_000, 40_000), seed)
            for (padding in 0L..3L) {
                val padded = EditorMath.applyPadding(changed, padding * 1000, padding * 1000, 40_000)
                assertEquals(expected, groups(padded).map { it.cutIds })
                assertEquals(base.cuts.map { it.editableGroupId }, padded.cuts.map { it.editableGroupId })
            }
        }
    }

    @Test fun ordinaryMergedTrimDoesNotManufactureAnExplicitBoundary() {
        val base = draft().let { it.copy(cuts = listOf(it.cuts[0].copy(coreEndMs = 10_000, keepEndMs = 12_000),
            it.cuts[0].copy(id = "R009", coreStartMs = 11_000, keepStartMs = 9_000))) }
        val changed = EditorMath.reconcileTouchedCuts(EditorMath.setCoreStart(base, "R009", 12_000, 0), seed)
        assertTrue(changed.userTouchedCutIds.isNotEmpty())
        assertEquals(1, groups(changed).size)
        assertEquals(groups(changed).single().id,
            EditorMath.rallySelectionAtPlayhead(groups(changed), "missing", 19_000, false))
        assertTrue(changed.cuts.all { it.editableGroupId == null })
        val explicit = requireNotNull(EditorMath.splitCut(base, "R001", 8_000, 0, 40_000,
            groupCutIds = groups(base).single().cutIds)).draft
        assertEquals(listOf(setOf("R001"), setOf("R002", "R009")), groups(explicit).map { it.cutIds })
        val again = split(explicit, "R009", 15_000)
        assertEquals(setOf("R001", "R009", "R002", "R003"), again.cuts.mapTo(hashSetOf()) { it.id })
        assertEquals(3, groups(again).size)
        assertEquals(explicit.cuts.first().editableGroupId, again.cuts.first().editableGroupId)
    }

    @Test fun exportUnionAndTargetRemovalRespectAllPaddingAndShortGapCases() {
        for (manual in listOf(false, true)) for (padding in 0L..3L) {
            val base = EditorMath.applyPadding(draft(manual), padding * 1000, padding * 1000, 40_000)
            val result = split(base)
            assertEquals(geometry(base), geometry(result))
            for (removed in result.cuts) {
                val retained = result.cuts.single { it.id != removed.id }
                val changed = result.copy(cuts = result.cuts.map { if (it.id == removed.id) it.copy(included = false) else it })
                assertEquals(listOf(retained.keepStartMs to retained.keepEndMs), geometry(changed))
                assertEquals(2, groups(changed).size)
            }
        }
        val base = split(draft()).let { d -> d.copy(cuts = d.cuts + d.cuts.last().copy(id = "R099",
            coreStartMs = 25_000, keepStartMs = 25_000, coreEndMs = 28_000, keepEndMs = 28_000, editableGroupId = null)) }
        assertEquals(2, EditorMath.finalIntervals(base).size) // exactly 3 s remains a cut
        assertEquals(1, EditorMath.finalIntervals(base.copy(cuts = base.cuts.map {
            if (it.id == "R099") it.copy(keepStartMs = 24_999) else it
        })).size)
    }

    @Test fun persistenceRelinkAndArchiveRetainIdentityWhileFullResetRestoresModelAndOpaqueUi() {
        val directory = Files.createTempDirectory("explicit-split-test").toFile()
        try {
            for (ui in listOf(JSONObject().put("schemaVersion", 2).put("preferences", JSONObject()
                .put("chapterOptions", JSONObject().put("future", JSONArray().put(JSONObject.NULL)))),
                JSONObject().put("schemaVersion", 1).put("preferences", JSONObject()
                    .put("chapterOptions", JSONObject().put("future", JSONArray().put(JSONObject.NULL)))))) {
                val d = split(draft()).copy(portableUi = ui.toString())
                val store = EditorDraftStore(directory, seed)
                store.save(d)
                val reopened = requireNotNull(store.load())
                assertEquals(d.cuts, reopened.cuts)
                val relinkedSeed = seed.copy(sourceUri = "content://fixture/relinked")
                EditorDraftStore(directory, relinkedSeed).save(reopened.copy(sourceRevision = relinkedSeed.sourceRevision))
                assertEquals(d.cuts, requireNotNull(EditorDraftStore(directory, relinkedSeed).load()).cuts)
                val imported = ModelFeedbackImporter.parse(bundle(reopened).toString())
                assertEquals(reopened.cuts, imported.draft.cuts)
                assertEquals(2, groups(imported.draft).size)
                assertEquals(geometry(reopened), geometry(imported.draft))
                val reexport = bundle(imported.draft, imported.project)
                assertEquals(ui.getJSONObject("preferences").getJSONObject("chapterOptions").toString(),
                    reexport.getJSONObject("corrections").getJSONObject("ui").getJSONObject("preferences").getJSONObject("chapterOptions").toString())
                if (ui.getInt("schemaVersion") == 2) assertEquals(ui.toString(), reexport.getJSONObject("corrections").getJSONObject("ui").toString())
                val reset = PortableReviewPreferences.reset(seed, reopened)
                assertEquals(EditorMath.newDraft(seed).cuts, reset.cuts)
                store.save(reset)
                assertEquals(1, groups(requireNotNull(store.load())).size)
                val resetUi = PortableReviewPreferences.export(reset)
                assertEquals(ui.getJSONObject("preferences").getJSONObject("chapterOptions").toString(),
                    resetUi.getJSONObject("preferences").getJSONObject("chapterOptions").toString())
                if (ui.getInt("schemaVersion") == 2) assertEquals(ui.toString(), resetUi.toString())
            }
        } finally { directory.deleteRecursively() }
    }

    @Test fun fullResetSharedFixtureRestoresModelOrderAndGeometryAndRemovesManualSplitAdditions() {
        val scenario = fixture().getJSONObject("fullReset")
        val ranges = scenario.getJSONArray("modelRanges")
        val resetSeed = seed.copy(ranges = List(ranges.length()) { index -> ranges.getJSONObject(index).let {
            SeedRange((it.getDouble("start") * 1000).toLong(), (it.getDouble("end") * 1000).toLong(),
                it.getDouble("confidence").toFloat(), id = it.getString("id"))
        } })
        val input = bundle(draft()).apply {
            getJSONObject("corrections").put("correctedRanges", scenario.getJSONArray("correctedRanges"))
        }
        val edited = ModelFeedbackImporter.parse(input.toString()).draft.copy(
            sourceRevision = resetSeed.sourceRevision, reviewedCutIds = setOf("R001", "R002"))
        val reset = PortableReviewPreferences.reset(resetSeed, edited)
        assertEquals(EditorMath.newDraft(resetSeed).cuts, reset.cuts)
        assertEquals(listOf("R001", "R009"), reset.cuts.map { it.id })
        assertEquals(listOf(10_000L to 20_000L, 30_000L to 35_000L), reset.cuts.map { it.coreStartMs to it.coreEndMs })
        assertTrue(reset.reviewedCutIds.isEmpty())
        assertTrue(reset.cuts.all { it.editableGroupId == null && it.origin == CutOrigin.INFERRED && it.included })
    }

    @Test fun fixtureRoundTripsAndRemovedMembersLeaveNoStaleReferences() {
        val b = bundle(draft())
        b.getJSONObject("corrections").put("correctedRanges", fixture().getJSONArray("correctedRanges"))
        val imported = ModelFeedbackImporter.parse(b.toString())
        assertEquals(4, groups(imported.draft).size)
        val deleted = imported.draft.copy(cuts = imported.draft.cuts.filterNot { it.id == "M001" })
        val reexport = bundle(deleted, imported.project)
        assertEquals(3, groups(ModelFeedbackImporter.parse(reexport.toString()).draft).size)
        assertFalse(reexport.getJSONObject("corrections").getJSONArray("correctedRanges").toString().contains("M001"))
        val originalIds = deleted.cuts.map { it.id }
        assertEquals(originalIds, ModelFeedbackImporter.parse(reexport.toString()).draft.cuts.map { it.id })
    }

    @Test fun strictPresentMetadataBoundsOriginsAndDuplicateIds() {
        val invalid = fixture().getJSONArray("invalidGroupIds")
        val values = (0 until invalid.length()).map { invalid.get(it) } + "a".repeat(129)
        for (value in values) {
            val b = bundle(draft())
            b.getJSONObject("corrections").getJSONArray("correctedRanges").getJSONObject(0).put("editableGroupId", value)
            assertThrows(IllegalArgumentException::class.java) { ModelFeedbackImporter.parse(b.toString()) }
        }
        val b = bundle(split(draft()))
        val ranges = b.getJSONObject("corrections").getJSONArray("correctedRanges")
        ranges.getJSONObject(1).put("id", "R001")
        assertThrows(IllegalArgumentException::class.java) { ModelFeedbackImporter.parse(b.toString()) }
        ranges.getJSONObject(1).put("id", "R002").put("origin", "inferred")
        assertThrows(IllegalArgumentException::class.java) { ModelFeedbackImporter.parse(b.toString()) }
        val member = draft().cuts.single().copy(editableGroupId = "a".repeat(128))
        ExplicitRallyGroups.validate(List(2048) { member.copy(id = "C$it") })
        assertThrows(IllegalArgumentException::class.java) { ExplicitRallyGroups.validate(List(2049) { member.copy(id = "C$it") }) }
        assertThrows(IllegalArgumentException::class.java) { ExplicitRallyGroups.validate(listOf(member.copy(id = "x".repeat(257)))) }
    }

    @Test fun legacyDraftAbsenceAndDefaultsRemainUnchanged() {
        val directory = Files.createTempDirectory("explicit-legacy-test").toFile()
        try {
            val store = EditorDraftStore(directory, seed)
            store.save(draft())
            val file = File(directory, "${seed.sourceRevision}.json")
            val json = JSONObject(file.readText()).put("version", 9)
            assertFalse(json.getJSONArray("cuts").getJSONObject(0).has("editableGroupId"))
            file.writeText(json.toString())
            assertEquals(draft().cuts, requireNotNull(store.load()).cuts)
            assertEquals(EditorMath.newDraft(seed).cuts, PortableReviewPreferences.reset(seed, draft()).cuts)
        } finally { directory.deleteRecursively() }
    }
}
