package com.volleycut.nativeanalysis

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.nio.file.Files

class PortableReviewPreferencesTest {
    private val project = NativeProject(
        id = "synthetic-portable",
        source = ProjectSource("content://fixture/portable", "synthetic-match.mp4", 1024, 100, "video/mp4"),
        media = AnalysisTypes.MediaInfo(30.0, 1920, 1080, 0, "video/avc", null),
        roi = AnalysisTypes.Roi(0.0, 0.0, 1.0, 1.0, "fixture"),
        status = ProjectStatus.READY,
        ranges = listOf(SeedRange(5_000, 10_000, .8f, ProductionEnsemble.BOTH_MODELS)),
        createdAtMs = 100, updatedAtMs = 100,
    )
    private val seed = requireNotNull(project.editorSeed())
    private fun bundle(draft: EditorDraft = EditorMath.newDraft(seed), owner: NativeProject = project) =
        ModelFeedbackExporter.createBundle(owner, draft, EditorMath.finalIntervals(draft, owner.suppression), null, "sampled-sha256-v1:${"a".repeat(64)}")
    private fun fixture(): JSONObject {
        val file = generateSequence(File(System.getProperty("user.dir"))) { it.parentFile }
            .map { File(it, "tests/fixtures/portable-review-preferences.json") }.first { it.isFile }
        return JSONObject(file.readText())
    }
    private fun input(ui: Any? = null): JSONObject = bundle().apply {
        remove("editorPreferences")
        getJSONObject("corrections").remove("ui")
        if (ui != null) getJSONObject("corrections").put("ui", ui)
    }
    private fun parse(ui: Any) = ModelFeedbackImporter.parse(input(ui).toString())
    private fun wire(bundle: JSONObject) = bundle.getJSONObject("corrections").getJSONObject("ui")
    private fun assertJson(expected: Any, actual: Any) {
        when (expected) {
            is JSONObject -> {
                require(actual is JSONObject)
                assertEquals(expected.keys().asSequence().toSet(), actual.keys().asSequence().toSet())
                expected.keys().forEach { assertJson(expected.get(it), actual.get(it)) }
            }
            is JSONArray -> {
                require(actual is JSONArray)
                assertEquals(expected.length(), actual.length())
                repeat(expected.length()) { assertJson(expected.get(it), actual.get(it)) }
            }
            is Number -> { require(actual is Number); assertEquals(expected.toDouble(), actual.toDouble(), 0.0) }
            else -> assertEquals(expected, actual)
        }
    }

    @Test fun sharedFixtureActivatesFinalFlagAndPreservesIndependentWebPreviewThroughProductionRoundTrip() {
        val ui = fixture().getJSONObject("portableUi")
        val imported = parse(ui)
        val draft = imported.draft
        assertTrue(draft.finalPreviewEnabled)
        assertTrue(draft.renderScoreOverlay) // Preview is false; native output is final=true.
        assertTrue(draft.renderScoreTimeline)
        assertTrue(draft.fadeScoreOverlay)
        assertEquals(1.5, draft.playbackRate, 0.0)
        assertEquals(.73f, draft.confidenceReviewThreshold, 0f)
        assertEquals(setOf("R001"), draft.reviewedCutIds)
        val exported = bundle(draft, imported.project)
        assertJson(ui, wire(exported))
        val again = ModelFeedbackImporter.parse(exported.toString())
        assertEquals(draft, again.draft.copy(sourceRevision = draft.sourceRevision))
        assertEquals(EditorMath.finalIntervals(draft), EditorMath.finalIntervals(again.draft))
    }

    @Test fun legacyDefaultsAndV1DefaultsAreDistinctAndOptionalKnownValuesAreEmitted() {
        val legacy = ModelFeedbackImporter.parse(input().toString()).draft
        assertFalse(legacy.renderScoreOverlay)
        assertFalse(legacy.renderScoreTimeline)
        val v1 = parse(JSONObject().put("schemaVersion", 1)).draft
        assertTrue(v1.renderScoreOverlay)
        assertFalse(v1.renderScoreTimeline)
        assertFalse(v1.finalPreviewEnabled)
        assertFalse(v1.fadeScoreOverlay)
        assertTrue(v1.fadeScoreOverlayExplicit)
        assertEquals(1.0, v1.playbackRate, 0.0)
        assertEquals(.7f, v1.confidenceReviewThreshold, 0f)
        assertEquals(8, PortableReviewPreferences.export(v1).getJSONObject("preferences").length())
        val ios = parse(fixture().getJSONObject("legacyIosUi")).draft
        assertFalse(ios.renderScoreOverlay)
        assertFalse(ios.renderScoreTimeline)
        assertJson(fixture().getJSONObject("legacyIosUi").getJSONObject("preferences").getJSONObject("chapterOptions"),
            PortableReviewPreferences.export(ios).getJSONObject("preferences").getJSONObject("chapterOptions"))
    }

    @Test fun finalAliasIsStrictAndCanonicalWinsInBothDirections() {
        for (final in listOf(false, true)) {
            val json = input(JSONObject().put("schemaVersion", 1).put("preferences", JSONObject()
                .put("renderScoreOnPreviewVideo", !final).put("renderScoreOnFinalVideo", final)))
            json.put("editorPreferences", JSONObject().put("renderScoreOnFinalVideo", !final))
            val imported = ModelFeedbackImporter.parse(json.toString())
            assertEquals(final, imported.draft.renderScoreOverlay)
            assertEquals(final, bundle(imported.draft).getJSONObject("editorPreferences").getBoolean("renderScoreOnFinalVideo"))
            json.getJSONObject("corrections").getJSONObject("ui").getJSONObject("preferences").remove("renderScoreOnFinalVideo")
            assertEquals(!final, ModelFeedbackImporter.parse(json.toString()).draft.renderScoreOverlay)
            json.getJSONObject("corrections").getJSONObject("ui").getJSONObject("preferences").put("renderScoreOnFinalVideo", final)
            json.getJSONObject("editorPreferences").put("renderScoreOnFinalVideo", "true")
            assertThrows(IllegalArgumentException::class.java) { ModelFeedbackImporter.parse(json.toString()) }
        }
        val legacy = input().put("editorPreferences", JSONObject().put("renderScoreOnFinalVideo", true))
        assertTrue(ModelFeedbackImporter.parse(legacy.toString()).draft.renderScoreOverlay)
    }

    @Test fun reviewedIdsRespectCountLengthAndInferredIdentityBounds() {
        val cut = EditorMath.newDraft(seed).cuts.single()
        val cuts = List(2049) { cut.copy(id = "cut-$it") }
        fun ui(ids: List<String>) = JSONObject().put("preferences", JSONObject().put("reviewedCutIds", JSONArray(ids)))
        PortableReviewPreferences.validate(ui(cuts.take(2048).map { it.id }), cuts)
        assertThrows(IllegalArgumentException::class.java) {
            PortableReviewPreferences.validate(ui(cuts.map { it.id }), cuts)
        }
        val longest = cut.copy(id = "x".repeat(256))
        PortableReviewPreferences.validate(ui(listOf(longest.id)), listOf(longest))
        assertThrows(IllegalArgumentException::class.java) {
            PortableReviewPreferences.validate(ui(listOf("x".repeat(257))), listOf(longest.copy(id = "x".repeat(257))))
        }
    }

    @Test fun futureVersionRemainsOpaqueThroughEditsResetAndReexportWithoutActivationOrDowngrade() {
        val ui = fixture().getJSONObject("futureUi").put("preferences", JSONObject()
            .put("playbackRate", 1.5).put("cutPreviewEnabled", true).put("fadeScoreOverlay", true)
            .put("renderScoreTimeline", true).put("renderScoreOnFinalVideo", true)
            .put("reviewedCutIds", JSONArray().put("R001")).put("confidenceReviewThreshold", .2))
        val draft = parse(ui).draft
        assertEquals(1.0, draft.playbackRate, 0.0)
        assertFalse(draft.finalPreviewEnabled)
        assertFalse(draft.renderScoreOverlay)
        assertFalse(draft.renderScoreTimeline)
        assertFalse(draft.fadeScoreOverlay)
        assertTrue(draft.reviewedCutIds.isEmpty())
        assertEquals(.7f, draft.confidenceReviewThreshold, 0f)
        assertTrue(PortableReviewPreferences.notice(draft).contains("not applied"))
        assertJson(ui, wire(bundle(draft.copy(playbackRate = 4.0, renderScoreOverlay = false))))
        assertJson(ui, wire(bundle(PortableReviewPreferences.reset(seed, draft))))
        val alias = input(ui).put("editorPreferences", JSONObject().put("renderScoreOnFinalVideo", true))
        assertTrue(ModelFeedbackImporter.parse(alias.toString()).draft.renderScoreOverlay)
        assertThrows(IllegalArgumentException::class.java) { parse(JSONObject(ui.toString()).apply {
            getJSONObject("preferences").put("playbackRate", "1.5")
        }) }
    }

    @Test fun allSharedInvalidFragmentsFailProductionParse() {
        val invalid = fixture().getJSONArray("invalidUi")
        repeat(invalid.length()) { index ->
            assertThrows("fixture invalidUi[$index]", IllegalArgumentException::class.java) { parse(invalid.get(index)) }
        }
    }

    @Test fun reservedTypesVersionsAndIdentitiesCannotBeCoerced() {
        val bad = listOf(
            JSONObject().put("schemaVersion", 1.5), JSONObject().put("schemaVersion", 9_007_199_254_740_992.0),
            JSONObject().put("schemaVersion", JSONObject.NULL),
            JSONObject().put("preferences", JSONObject().put("playbackRate", "1.5")),
            JSONObject().put("preferences", JSONObject().put("confidenceReviewThreshold", -.1)),
            JSONObject().put("preferences", JSONObject().put("reviewedCutIds", JSONArray().put("missing"))),
            JSONObject().put("preferences", JSONObject().put("reviewedCutIds", JSONArray().put(1))),
            JSONObject().put("preferences", JSONObject().put("reviewedCutIds", JSONArray().put("a".repeat(257)))),
        )
        bad.forEach { assertThrows(IllegalArgumentException::class.java) { parse(it) } }
        val manual = input(JSONObject().put("preferences", JSONObject().put("reviewedCutIds", JSONArray().put("M001"))))
        manual.getJSONObject("corrections").getJSONArray("correctedRanges").put(JSONObject()
            .put("id", "M001").put("origin", "manual").put("coreStart", 20).put("coreEnd", 22)
            .put("keepStart", 20).put("keepEnd", 22))
        assertThrows(IllegalArgumentException::class.java) { ModelFeedbackImporter.parse(manual.toString()) }
    }

    @Test fun jsonSafetyBoundsHaveInclusiveLimitsAndApplyOnExportToo() {
        fun validate(ui: JSONObject) = PortableReviewPreferences.validate(ui, EditorMath.newDraft(seed).cuts)
        fun nested(depth: Int): JSONObject {
            var value: Any = 0
            repeat(depth) { value = JSONObject().put("x", value) }
            return value as JSONObject
        }
        validate(nested(16))
        assertThrows(IllegalArgumentException::class.java) { validate(nested(17)) }
        validate(JSONObject().put("x", JSONArray(List(4094) { 0 }))) // root + array + leaves
        assertThrows(IllegalArgumentException::class.java) { validate(JSONObject().put("x", JSONArray(List(4095) { 0 }))) }
        validate(JSONObject().put("x", "a".repeat(16_384)))
        assertThrows(IllegalArgumentException::class.java) { validate(JSONObject().put("x", "a".repeat(16_385))) }
        validate(JSONObject().put("x", "\uD83C\uDFD0".repeat(8192)))
        assertThrows(IllegalArgumentException::class.java) { validate(JSONObject().put("x", "\uD83C\uDFD0".repeat(8193))) }
        val exact = JSONObject().put("k".repeat(65_530), 0)
        assertEquals(65_536, exact.toString().toByteArray().size)
        validate(exact)
        assertThrows(IllegalArgumentException::class.java) { validate(JSONObject().put("k".repeat(65_531), 0)) }
        assertThrows(IllegalArgumentException::class.java) { validate(JSONObject().put("x", JSONArray(List(4) { "\u20ac".repeat(6000) }))) }
        for (key in listOf("__proto__", "constructor", "prototype")) {
            assertThrows(IllegalArgumentException::class.java) { validate(JSONObject().put("nested", JSONArray().put(JSONObject().put(key, 1)))) }
        }
        assertThrows(IllegalArgumentException::class.java) {
            PortableReviewPreferences.export(EditorMath.newDraft(seed).copy(portableUi = exact.toString()))
        }
    }

    @Test fun arbitraryRatesAndProbabilitiesKeepExactWireNumbersUntilChanged() {
        for (rate in listOf(.25, 1.5, 3.125, 8.0, .3333333333333333)) {
            val draft = parse(JSONObject().put("preferences", JSONObject().put("playbackRate", rate)
                .put("confidenceReviewThreshold", .731234567890123))).draft
            assertEquals(rate, draft.playbackRate, 0.0)
            val prefs = PortableReviewPreferences.export(draft).getJSONObject("preferences")
            assertEquals(rate, prefs.getDouble("playbackRate"), 0.0)
            assertEquals(.731234567890123, prefs.getDouble("confidenceReviewThreshold"), 0.0)
            assertTrue(PortableReviewPreferences.playbackRates(draft.playbackRate).contains(rate))
            assertEquals(2.0, PortableReviewPreferences.export(draft.copy(playbackRate = 2.0)).getJSONObject("preferences").getDouble("playbackRate"), 0.0)
        }
        assertEquals("1.5x", PortableReviewPreferences.playbackRateLabel(1.5))
        assertEquals("0.25x", PortableReviewPreferences.playbackRateLabel(.25))
    }

    @Test fun filePersistenceReopenRelinkAndResetKeepEnvelopeAndProjectFadePriority() {
        val directory = Files.createTempDirectory("portable-draft-test").toFile()
        try {
            val imported = parse(fixture().getJSONObject("portableUi"))
            val importedSeed = requireNotNull(imported.project.editorSeed())
            val store = EditorDraftStore(directory, importedSeed)
            store.save(imported.draft)
            val reopened = requireNotNull(EditorDraftStore(directory, importedSeed).load())
            assertEquals(imported.draft, reopened)
            assertTrue(PortableReviewPreferences.onOpen(reopened, false).fadeScoreOverlay)
            assertFalse(PortableReviewPreferences.onOpen(reopened.copy(fadeScoreOverlay = false), true).fadeScoreOverlay)
            assertTrue(PortableReviewPreferences.onOpen(EditorMath.newDraft(seed), true).fadeScoreOverlay)
            val relinked = reopened.copy(sourceRevision = seed.sourceRevision)
            EditorDraftStore(directory, seed).save(relinked)
            assertEquals(reopened.portableUi, requireNotNull(EditorDraftStore(directory, seed).load()).portableUi)
            val reset = PortableReviewPreferences.reset(importedSeed, reopened)
            val defaults = EditorMath.newDraft(importedSeed)
            assertEquals(defaults, reset.copy(portableUi = null, fadeScoreOverlayExplicit = false))
            store.save(reset)
            val restoredReset = requireNotNull(EditorDraftStore(directory, importedSeed).load())
            assertFalse(PortableReviewPreferences.onOpen(restoredReset, true).fadeScoreOverlay)
            val exported = PortableReviewPreferences.export(restoredReset)
            val prefs = exported.getJSONObject("preferences")
            assertTrue(prefs.getJSONArray("reviewedCutIds").length() == 0)
            assertEquals(1.0, prefs.getDouble("playbackRate"), 0.0)
            assertFalse(prefs.getBoolean("fadeScoreOverlay"))
            assertJson(fixture().getJSONObject("portableUi").getJSONObject("preferences").getJSONObject("chapterOptions"), prefs.getJSONObject("chapterOptions"))
            assertJson(fixture().getJSONObject("portableUi").getJSONObject("futurePanel"), exported.getJSONObject("futurePanel"))
            assertFalse(prefs.getBoolean("renderScoreOnPreviewVideo"))
            val future = parse(fixture().getJSONObject("futureUi")).draft
            val futureSeed = requireNotNull(parse(fixture().getJSONObject("futureUi")).project.editorSeed())
            val futureStore = EditorDraftStore(directory, futureSeed)
            futureStore.save(future.copy(sourceRevision = futureSeed.sourceRevision))
            val futureRestored = requireNotNull(futureStore.load())
            assertJson(fixture().getJSONObject("futureUi"), PortableReviewPreferences.export(futureRestored))
            assertEquals(1.0, futureRestored.playbackRate, 0.0)
        } finally { directory.deleteRecursively() }
    }

    @Test fun legacyDraftMigrationStillLoadsAndKeepsDeviceFallbackOnlyWhenFadeWasNeverStored() {
        val directory = Files.createTempDirectory("portable-legacy-test").toFile()
        try {
            val store = EditorDraftStore(directory, seed)
            store.save(EditorMath.newDraft(seed))
            val file = File(directory, "${seed.sourceRevision}.json")
            val json = JSONObject(file.readText()).put("version", 8)
            json.remove("portableUi"); json.remove("fadeScoreOverlayExplicit")
            json.put("fadeScoreOverlay", false)
            file.writeText(json.toString())
            assertFalse(PortableReviewPreferences.onOpen(requireNotNull(store.load()), true).fadeScoreOverlay)
            json.remove("fadeScoreOverlay"); file.writeText(json.toString())
            assertTrue(PortableReviewPreferences.onOpen(requireNotNull(store.load()), true).fadeScoreOverlay)
        } finally { directory.deleteRecursively() }
    }
}
