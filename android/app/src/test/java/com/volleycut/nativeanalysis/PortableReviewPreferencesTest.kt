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

    private val futureIdentityError = "Cannot export: newer project preferences refer to rallies changed or removed. " +
        "Open an unchanged copy, or export with an app that supports this preferences version. Your local project is preserved."

    private fun futureUi(id: String = "R001") = fixture().getJSONObject("futureUi").apply {
        getJSONObject("preferences").put("reviewedCutIds", JSONArray().put(id))
        getJSONObject("preferences").put("chapterOptions", fixture().getJSONObject("portableUi")
            .getJSONObject("preferences").getJSONObject("chapterOptions"))
        put("unknownNested", JSONObject().put("values", JSONArray().put(JSONObject.NULL)
            .put(JSONObject().put("nested", JSONArray().put(false).put(1.25)))))
    }

    private fun importedFutureWithCorrectedId(id: String = "R002"): ModelFeedbackImporter.Imported {
        val json = input(futureUi(id))
        json.getJSONObject("corrections").getJSONArray("correctedRanges").getJSONObject(0).put("id", id)
        return ModelFeedbackImporter.parse(json.toString())
    }

    @Test fun unchangedFutureEnvelopeRoundTripsWithoutActivationIncludingNestedExtensions() {
        val imported = importedFutureWithCorrectedId()
        val ui = JSONObject(requireNotNull(imported.draft.portableUi))
        assertEquals(listOf("R001"), EditorMath.newDraft(requireNotNull(imported.project.editorSeed())).cuts.map { it.id })
        assertEquals(listOf("R002"), imported.draft.cuts.map { it.id })
        assertTrue(imported.draft.reviewedCutIds.isEmpty())
        assertEquals(1.0, imported.draft.playbackRate, 0.0)
        val edited = imported.draft.copy(playbackRate = 3.125, finalPreviewEnabled = true, fadeScoreOverlay = true)
        val exported = bundle(edited, imported.project)
        assertJson(ui, wire(exported))
        val again = ModelFeedbackImporter.parse(exported.toString())
        assertJson(ui, JSONObject(requireNotNull(again.draft.portableUi)))
        assertEquals(1.0, again.draft.playbackRate, 0.0)
        assertFalse(again.draft.finalPreviewEnabled)
        assertFalse(again.draft.fadeScoreOverlay)
        assertTrue(again.draft.reviewedCutIds.isEmpty())
        assertJson(ui, wire(bundle(again.draft, again.project)))
    }

    @Test fun futureResetAndRepeatedResetRetainStaleIdsLocallyButRejectArchivePublication() {
        val imported = importedFutureWithCorrectedId()
        val importedSeed = requireNotNull(imported.project.editorSeed())
        val directory = Files.createTempDirectory("future-ui-reset-test").toFile()
        val store = EditorDraftStore(directory, importedSeed)
        try {
            store.save(imported.draft)
            var current = (store.load() as EditorDraftLoadResult.Restored).draft
            assertJson(JSONObject(requireNotNull(imported.draft.portableUi)), wire(bundle(current, imported.project)))
            repeat(2) {
                current = PortableReviewPreferences.reset(importedSeed, current)
                assertEquals(listOf("R001"), current.cuts.map { it.id })
                assertTrue(current.reviewedCutIds.isEmpty())
                assertEquals(imported.draft.portableUi, current.portableUi)
                store.save(current)
                val reopened = (EditorDraftStore(directory, importedSeed).load() as EditorDraftLoadResult.Restored).draft
                assertEquals(current, reopened)
                assertEquals(futureIdentityError, assertThrows(IllegalArgumentException::class.java) {
                    bundle(reopened, imported.project)
                }.message)
                assertEquals(current, (store.load() as EditorDraftLoadResult.Restored).draft)
                assertEquals(imported.draft.portableUi, reopened.portableUi)
                current = reopened
            }
        } finally {
            store.clear()
            Files.delete(File(directory, "editor-drafts").toPath())
            Files.delete(directory.toPath())
        }
    }

    @Test fun originalFailureControlRejectsTheArchivePublishedByTheOldFutureEarlyReturn() {
        val imported = importedFutureWithCorrectedId()
        val reset = PortableReviewPreferences.reset(requireNotNull(imported.project.editorSeed()), imported.draft)
        // Bypass the export guard to reconstruct exactly the old dangling UI publication.
        val previouslyPublished = bundle(reset.copy(portableUi = null), imported.project)
        previouslyPublished.getJSONObject("corrections").put("ui", JSONObject(requireNotNull(reset.portableUi)))
        assertEquals("R001", previouslyPublished.getJSONObject("corrections")
            .getJSONArray("correctedRanges").getJSONObject(0).getString("id"))
        assertEquals("R002", wire(previouslyPublished).getJSONObject("preferences")
            .getJSONArray("reviewedCutIds").getString(0))
        assertEquals("Invalid reviewed cut identity", assertThrows(IllegalArgumentException::class.java) {
            ModelFeedbackImporter.parse(previouslyPublished.toString())
        }.message)
    }

    @Test fun futureReviewedIdsRequireCurrentInferredCutsRatherThanInitialOrManualCuts() {
        val imported = parse(futureUi())
        val original = imported.draft
        val cut = original.cuts.single()
        for (cuts in listOf(emptyList(), listOf(cut.copy(id = "R002")), listOf(cut.copy(origin = CutOrigin.MANUAL)))) {
            val changed = original.copy(cuts = cuts)
            assertEquals(futureIdentityError, assertThrows(IllegalArgumentException::class.java) {
                bundle(changed, imported.project)
            }.message)
            assertEquals(original.portableUi, changed.portableUi)
        }
        // Exclusion keeps the inferred identity in correctedRanges and remains valid.
        val excluded = original.copy(cuts = listOf(cut.copy(included = false)))
        assertJson(futureUi(), wire(bundle(excluded, imported.project)))
        ModelFeedbackImporter.parse(bundle(excluded, imported.project).toString())
        val manualInput = input(futureUi()).apply {
            getJSONObject("corrections").getJSONArray("correctedRanges").getJSONObject(0).put("origin", "manual")
        }
        assertThrows(IllegalArgumentException::class.java) { ModelFeedbackImporter.parse(manualInput.toString()) }
    }

    @Test fun futureAbsentReviewedIdsAreAllowedButMalformedListsStillFailImportAndExport() {
        val absent = futureUi().apply { getJSONObject("preferences").remove("reviewedCutIds") }
        for (ui in listOf(absent, JSONObject().put("schemaVersion", 2),
            futureUi().apply { getJSONObject("preferences").put("reviewedCutIds", JSONArray()) })) {
            val imported = parse(ui)
            val reset = PortableReviewPreferences.reset(requireNotNull(imported.project.editorSeed()), imported.draft)
            assertJson(ui, wire(bundle(reset, imported.project)))
            ModelFeedbackImporter.parse(bundle(reset, imported.project).toString())
        }
        for (value in listOf(JSONObject.NULL, "R001", JSONObject(), JSONArray().put(1),
            JSONArray().put(""), JSONArray().put("R001").put("R001"))) {
            val ui = futureUi().apply { getJSONObject("preferences").put("reviewedCutIds", value) }
            assertThrows(IllegalArgumentException::class.java) { parse(ui) }
            assertThrows(IllegalArgumentException::class.java) {
                bundle(EditorMath.newDraft(seed).copy(portableUi = ui.toString()))
            }
        }
        assertThrows(IllegalArgumentException::class.java) { parse(futureUi("missing")) }
    }

    @Test fun archiveValidationFailureDoesNotOpenDestinationOrPublishAStaleReplacement() {
        val imported = importedFutureWithCorrectedId()
        val reset = PortableReviewPreferences.reset(requireNotNull(imported.project.editorSeed()), imported.draft)
        val directory = Files.createTempDirectory("future-ui-archive-test").toFile()
        val destination = File(directory, "project.json")
        var opened = 0
        fun publish(draft: EditorDraft) = runCatching {
            ModelFeedbackExporter.writeArchive(
                buildArchive = { bundle(draft, imported.project).toString() + "\n" },
                openDestination = { opened++; destination.outputStream() },
            )
        }
        try {
            // Validation failure cannot create a partial file, even on the first attempt.
            assertEquals(futureIdentityError, publish(reset).exceptionOrNull()?.message)
            assertEquals(0, opened)
            assertFalse(destination.exists())
            assertTrue(publish(imported.draft).isSuccess)
            assertEquals(1, opened)
            val previous = destination.readBytes()
            repeat(2) {
                assertEquals(futureIdentityError, publish(reset).exceptionOrNull()?.message)
                assertEquals(1, opened)
                assertArrayEquals(previous, destination.readBytes())
                assertEquals(imported.draft.portableUi, reset.portableUi)
                assertEquals(listOf("R001"), reset.cuts.map { it.id })
            }
            assertJson(JSONObject(requireNotNull(imported.draft.portableUi)),
                JSONObject(requireNotNull(ModelFeedbackImporter.parse(destination.readText()).draft.portableUi)))
            // A later valid attempt uses the current draft rather than a cached old archive.
            val changed = imported.draft.copy(cuts = imported.draft.cuts.map { it.copy(included = false) })
            assertTrue(publish(changed).isSuccess)
            assertEquals(2, opened)
            assertFalse(ModelFeedbackImporter.parse(destination.readText()).draft.cuts.single().included)
        } finally {
            Files.deleteIfExists(destination.toPath())
            Files.delete(directory.toPath())
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
            val reopened = (EditorDraftStore(directory, importedSeed).load() as EditorDraftLoadResult.Restored).draft
            assertEquals(imported.draft, reopened)
            assertTrue(PortableReviewPreferences.onOpen(reopened, false).fadeScoreOverlay)
            assertFalse(PortableReviewPreferences.onOpen(reopened.copy(fadeScoreOverlay = false), true).fadeScoreOverlay)
            assertTrue(PortableReviewPreferences.onOpen(EditorMath.newDraft(seed), true).fadeScoreOverlay)
            val relinked = reopened.copy(sourceRevision = seed.sourceRevision)
            EditorDraftStore(directory, seed).save(relinked)
            assertEquals(reopened.portableUi, (EditorDraftStore(directory, seed).load() as EditorDraftLoadResult.Restored).draft.portableUi)
            val reset = PortableReviewPreferences.reset(importedSeed, reopened)
            val defaults = EditorMath.newDraft(importedSeed)
            assertEquals(defaults, reset.copy(portableUi = null, fadeScoreOverlayExplicit = false))
            store.save(reset)
            val restoredReset = (EditorDraftStore(directory, importedSeed).load() as EditorDraftLoadResult.Restored).draft
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
            val futureRestored = (futureStore.load() as EditorDraftLoadResult.Restored).draft
            assertJson(fixture().getJSONObject("futureUi"), PortableReviewPreferences.export(futureRestored))
            assertEquals(1.0, futureRestored.playbackRate, 0.0)
        } finally { directory.deleteRecursively() }
    }

    @Test fun legacyDraftMigrationStillLoadsAndKeepsDeviceFallbackOnlyWhenFadeWasNeverStored() {
        val directory = Files.createTempDirectory("portable-legacy-test").toFile()
        try {
            val store = EditorDraftStore(directory, seed)
            store.save(EditorMath.newDraft(seed))
            val file = File(directory, "editor-drafts/${seed.sourceRevision}.json")
            val json = JSONObject(file.readText()).put("version", 8)
            json.remove("portableUi"); json.remove("fadeScoreOverlayExplicit")
            json.put("fadeScoreOverlay", false)
            file.writeText(json.toString())
            assertFalse(PortableReviewPreferences.onOpen((store.load() as EditorDraftLoadResult.Restored).draft, true).fadeScoreOverlay)
            json.remove("fadeScoreOverlay"); file.writeText(json.toString())
            assertTrue(PortableReviewPreferences.onOpen((store.load() as EditorDraftLoadResult.Restored).draft, true).fadeScoreOverlay)
        } finally { directory.deleteRecursively() }
    }
}
