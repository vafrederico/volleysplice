package com.volleycut.nativeanalysis

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/** Exercise the production archive and materializer, without source media or Android services. */
class ModelFeedbackCleanupRoundTripTest {
    @Test
    fun cachelessActiveCleanupDoesNotDoubleExportDuration() {
        val project = project()
        val draft = draft(project)
        val expected = listOf(FinalCutInterval(5_000, 10_000, listOf("R001")))
        assertEquals(expected, EditorMath.finalIntervals(draft, project.suppression))
        assertEquals(5_000L, expected.sumOf { it.endMs - it.startMs })

        val imported = ModelFeedbackImporter.parse(bundle(project, draft, null).toString())
        val actual = EditorMath.finalIntervals(imported.draft, imported.project.suppression)
        // AND-01 baseline: cache eviction loses cleanup and restores 10,000 ms here.
        assertEquals(5_000L, actual.sumOf { it.endMs - it.startMs })
        assertEquals(expected, actual)
        assertTrue(imported.draft.cuts.all { it.included })
        assertEquals(ProjectStatus.READY, imported.project.status)
        assertNull(imported.featureCache)
    }

    @Test
    fun bothScopesPreserveExactUnionThroughImportPersistenceAndReexportWithEitherCacheState() {
        val project = project()
        for (scope in SuppressionScope.entries) {
            val draft = draft(project, scope)
            val expected = if (scope == SuppressionScope.WHOLE_RALLY) {
                listOf(FinalCutInterval(5_000, 10_000, listOf("R001")))
            } else listOf(
                FinalCutInterval(5_000, 10_000, listOf("R001")),
                FinalCutInterval(20_000, 21_000, listOf("R002")),
                FinalCutInterval(23_000, 25_000, listOf("R002")),
            )
            for (cachePresent in listOf(false, true)) {
                for (reexportCachePresent in listOf(false, true)) {
                    assertRoundTrip(project, draft, expected, cachePresent, reexportCachePresent)
                }
            }
        }
    }

    @Test
    fun retainingCachelessSuggestionsDoesNotActivateHighlightOnlyCleanup() {
        val project = project()
        val logicalId = project.suppression!!.suggestions().single().logicalId()
        val kept = listOf(
            FinalCutInterval(5_000, 10_000, listOf("R001")),
            FinalCutInterval(20_000, 25_000, listOf("R002")),
        )
        for (scope in SuppressionScope.entries) {
            val active = draft(project, scope)
            val highlighted = active.copy(suppressionInitialBehavior = SuppressionInitialBehavior.HIGHLIGHT_ONLY)
            for (cachePresent in listOf(false, true)) {
                assertRoundTrip(project, highlighted, kept, cachePresent, !cachePresent)
                assertRoundTrip(project, highlighted.copy(
                    suppressionDecisionOverrides = mapOf(logicalId to SuppressionDecision.SUPPRESS),
                ), EditorMath.finalIntervals(active, project.suppression), cachePresent, !cachePresent)
            }
        }
    }

    @Test
    fun protectedEditsManualRangesAndExplicitDecisionsRemainEditableForBothScopes() {
        val project = project()
        val logicalId = project.suppression!!.suggestions().single().logicalId()
        for (scope in SuppressionScope.entries) {
            val base = draft(project, scope)
            val kept = listOf(
                FinalCutInterval(5_000, 10_000, listOf("R001")),
                FinalCutInterval(20_000, 25_000, listOf("R002")),
            )
            val protected = base.copy(userTouchedCutIds = setOf("R002"))
            val explicitKeep = base.copy(suppressionDecisionOverrides = mapOf(logicalId to SuppressionDecision.KEEP))
            val explicitSuppress = protected.copy(
                suppressionDecisionOverrides = mapOf(logicalId to SuppressionDecision.SUPPRESS),
            )
            val manual = base.copy(cuts = base.cuts + EditableCut(
                "M001", 22_000, 24_000, 22_000, 24_000, 1f, true, CutOrigin.MANUAL,
            ))
            val manualExpected = if (scope == SuppressionScope.WHOLE_RALLY) listOf(
                FinalCutInterval(5_000, 10_000, listOf("R001")),
                FinalCutInterval(22_000, 24_000, listOf("M001")),
            ) else listOf(
                FinalCutInterval(5_000, 10_000, listOf("R001")),
                FinalCutInterval(20_000, 21_000, listOf("R002")),
                FinalCutInterval(22_000, 25_000, listOf("M001", "R002")),
            )
            for (cachePresent in listOf(false, true)) {
                assertRoundTrip(project, protected, kept, cachePresent, cachePresent)
                assertRoundTrip(project, explicitKeep, kept, cachePresent, cachePresent)
                assertRoundTrip(project, explicitSuppress,
                    EditorMath.finalIntervals(base, project.suppression), cachePresent, cachePresent)
                assertRoundTrip(project, manual, manualExpected, cachePresent, cachePresent)
            }
        }
    }

    @Test
    fun vetoBarriersPaddingAndIgnoredTimeSurviveWithoutFlatteningCorrectedRanges() {
        val project = project()
        val padded = EditorMath.applyPadding(draft(project, SuppressionScope.VETO_REGION), 2_000, 2_000, 30_000)
            .copy(joinGapMs = 3_000, ignoredIntervals = listOf(IgnoredSourceInterval("I001", 7_000, 8_000, "break")))
        val expected = listOf(
            FinalCutInterval(3_000, 7_000, listOf("R001")),
            FinalCutInterval(8_000, 12_000, listOf("R001")),
            FinalCutInterval(18_000, 21_000, listOf("R002")),
            FinalCutInterval(23_000, 27_000, listOf("R002")),
        )
        for (cachePresent in listOf(false, true)) {
            assertRoundTrip(project, padded, expected, cachePresent, !cachePresent)
        }
    }

    @Test
    fun cacheWarningsAndFeatureReadinessRemainIndependentOfSavedCleanup() {
        val project = project()
        for (cachePresent in listOf(false, true)) {
            val exported = bundle(project, draft(project), analysis().takeIf { cachePresent })
            val warnings = exported.getJSONArray("warnings").let { values ->
                List(values.length()) { values.getString(it) }
            }
            assertEquals(!cachePresent, exported.isNull("features"))
            assertEquals(!cachePresent, warnings.any { it.contains("feature cache") })
            assertEquals(!cachePresent, warnings.any { it.contains("Suppression probability traces") })
            assertFalse(exported.getJSONObject("initialInference").isNull("suppression"))
            val cleanup = exported.getJSONObject("initialInference").getJSONObject("suppression")
            val expectedRows = if (cachePresent) analysis().timestamps.size else 0
            assertEquals(expectedRows, ModelFeedbackImporter.decodeNumeric(cleanup.getJSONObject("timestamps")).size)
            assertEquals(expectedRows, ModelFeedbackImporter.decodeNumeric(cleanup.getJSONObject("probabilities")).size)
            assertEquals(!cachePresent, cleanup.has("retainedProbabilities"))
            val imported = ModelFeedbackImporter.parse(exported.toString())
            assertEquals(cachePresent, imported.featureCache != null)
            assertEquals(ProjectStatus.READY, imported.project.status)
            assertNotNull(imported.project.editorSeed())
        }
    }

    @Test
    fun mismatchedCacheFailsRatherThanSilentlyDroppingCleanup() {
        val project = project().copy(suppression = project().suppression!!.let { value ->
            AnalysisTypes.SuppressionAnalysis(value.modelId(), value.artifactSha256(), value.weightsSha256(),
                value.decoderVersion(), floatArrayOf(.3f), value.decodedIntervals(), value.suggestions())
        })
        assertThrows(IllegalArgumentException::class.java) { bundle(project, draft(project), analysis()) }
        val cacheless = ModelFeedbackImporter.parse(bundle(project, draft(project), null).toString())
        assertEquals(listOf(FinalCutInterval(5_000, 10_000, listOf("R001"))),
            EditorMath.finalIntervals(cacheless.draft, cacheless.project.suppression))
    }

    @Test
    fun retainedProbabilityArchiveRejectsAmbiguityAndInvalidValues() {
        val project = project()
        for (invalid in listOf(
            ModelFeedbackExporter.encode(floatArrayOf(1.1f), intArrayOf(1)),
            ModelFeedbackExporter.encode(floatArrayOf(.3f), intArrayOf(1, 1)),
        )) {
            val exported = bundle(project, draft(project), null)
            exported.getJSONObject("initialInference").getJSONObject("suppression")
                .put("retainedProbabilities", invalid)
            assertThrows(IllegalArgumentException::class.java) { ModelFeedbackImporter.parse(exported.toString()) }
        }
        val ambiguous = bundle(project, draft(project), analysis())
        ambiguous.getJSONObject("initialInference").getJSONObject("suppression")
            .put("retainedProbabilities", ModelFeedbackExporter.encode(floatArrayOf(.3f), intArrayOf(1)))
        assertThrows(IllegalArgumentException::class.java) { ModelFeedbackImporter.parse(ambiguous.toString()) }
        // The optional extension is not required for schema v3 materialization-only archives.
        val old = bundle(project, draft(project), null)
        old.getJSONObject("initialInference").getJSONObject("suppression").remove("retainedProbabilities")
        val imported = ModelFeedbackImporter.parse(old.toString())
        assertEquals(listOf(FinalCutInterval(5_000, 10_000, listOf("R001"))),
            EditorMath.finalIntervals(imported.draft, imported.project.suppression))
        val reexported = bundle(imported.project, imported.draft, null)
        assertEquals(old.getJSONArray("finalExportIntervals").toString(),
            reexported.getJSONArray("finalExportIntervals").toString())
    }

    @Test
    fun neuralArchiveImportDoesNotReactivateLegacyCleanup() {
        val legacy = project()
        val legacyBundle = bundle(legacy, draft(legacy), analysis())
        for (modelId in RallyModels.OPTIONS.filter(RallyModels::isNeural)) {
            val neural = legacy.copy(modelId = modelId, ranges = legacy.ranges.map { it.copy(agreement = "neural") })
            val draft = draft(neural)
            val exported = bundle(neural, draft, analysis())
            assertTrue(exported.getJSONObject("initialInference").isNull("suppression"))
            // Older or external archives may contain auxiliary legacy cleanup data.
            exported.getJSONObject("initialInference").put("suppression",
                JSONObject(legacyBundle.getJSONObject("initialInference").getJSONObject("suppression").toString())
                    .put("timestamps", ModelFeedbackExporter.encode(doubleArrayOf(), intArrayOf(0))))
            val imported = ModelFeedbackImporter.parse(exported.toString())
            assertNull(imported.project.suppression)
            assertEquals(listOf(
                FinalCutInterval(5_000, 10_000, listOf("R001")),
                FinalCutInterval(20_000, 25_000, listOf("R002")),
            ), EditorMath.finalIntervals(imported.draft, imported.project.suppression))
        }
    }

    private fun assertRoundTrip(
        project: NativeProject,
        draft: EditorDraft,
        expected: List<FinalCutInterval>,
        cachePresent: Boolean,
        reexportCachePresent: Boolean,
    ) {
        assertEquals(expected, EditorMath.finalIntervals(draft, project.suppression))
        val exported = bundle(project, draft, analysis().takeIf { cachePresent })
        val imported = ModelFeedbackImporter.parse(exported.toString())
        val restored = requireNotNull(NativeProjectStore.decode(NativeProjectStore.encode(imported.project)))
        assertEquals(expected, EditorMath.finalIntervals(imported.draft, restored.suppression))
        assertEquals(EditorMath.materialize(draft, project.suppression),
            EditorMath.materialize(imported.draft, restored.suppression))
        assertEquals(draft.cuts, imported.draft.cuts)
        assertEquals(draft.ignoredIntervals, imported.draft.ignoredIntervals)
        assertEquals(draft.selectedSuppressionPolicy, imported.draft.selectedSuppressionPolicy)
        assertEquals(draft.suppressionDecisionOverrides, imported.draft.suppressionDecisionOverrides)
        assertEquals(draft.suppressionScopeOverrides, imported.draft.suppressionScopeOverrides)
        assertEquals(draft.userTouchedCutIds, imported.draft.userTouchedCutIds)
        assertEquals(draft.suppressionInitialBehavior, imported.draft.suppressionInitialBehavior)
        assertEquals(project.suppression!!.suggestions(), restored.suppression!!.suggestions())
        assertArrayEquals(project.suppression.probabilities(), restored.suppression.probabilities(), 0f)
        val reexported = bundle(restored, imported.draft,
            (imported.featureCache?.feedbackAnalysis() ?: analysis()).takeIf { reexportCachePresent })
        assertEquals(exported.getJSONArray("finalExportIntervals").toString(),
            reexported.getJSONArray("finalExportIntervals").toString())
        assertEquals(exported.getJSONArray("finalExportProvenance").toString(),
            reexported.getJSONArray("finalExportProvenance").toString())
        assertEquals(exported.getJSONObject("corrections").getJSONObject("suppression").toString(),
            reexported.getJSONObject("corrections").getJSONObject("suppression").toString())
        val again = ModelFeedbackImporter.parse(reexported.toString())
        assertEquals(expected, EditorMath.finalIntervals(again.draft, again.project.suppression))
        assertEquals(EditorMath.finalIntervals(draft.copy(selectedSuppressionPolicy = SuppressionPolicyEngine.Policy.NONE),
            project.suppression), EditorMath.finalIntervals(
            again.draft.copy(selectedSuppressionPolicy = SuppressionPolicyEngine.Policy.NONE), again.project.suppression))
        // Final output is evidence, never a replacement for the editable rally geometry.
        reexported.put("finalExportIntervals", JSONArray().put(JSONObject().put("start", 0).put("end", 30)))
        val changedEvidence = ModelFeedbackImporter.parse(reexported.toString())
        assertEquals(expected, EditorMath.finalIntervals(changedEvidence.draft, changedEvidence.project.suppression))
    }

    private fun bundle(project: NativeProject, draft: EditorDraft, analysis: ModelFeedbackAnalysis?) =
        ModelFeedbackExporter.createBundle(project, draft,
            EditorMath.finalIntervals(draft, project.suppression.takeUnless { RallyModels.isNeural(project.modelId) }),
            analysis, "sampled-sha256-v1:${"a".repeat(64)}")

    private fun draft(project: NativeProject, scope: SuppressionScope = SuppressionScope.WHOLE_RALLY) =
        EditorMath.applyPadding(EditorMath.newDraft(requireNotNull(project.editorSeed())), 0, 0, 30_000).copy(
            joinGapMs = 0,
            suppressionScopeOverrides = project.suppression!!.suggestions().associate { it.logicalId() to scope },
        )

    private fun analysis(): ModelFeedbackAnalysis {
        val times = AnalysisEngine.analysisTimes(30.0, 0.0, 30.0)
        return ModelFeedbackAnalysis(times, FloatArray(times.size * FeatureSchema.BASE.size),
            FloatArray(times.size) { .4f }, FloatArray(times.size) { .2f }, FloatArray(times.size) { .6f })
    }

    private fun project(): NativeProject {
        val allLabels = listOf(AnalysisTypes.Interval(5.0, 10.0, .8f), AnalysisTypes.Interval(20.0, 25.0, .8f))
        val previous = listOf(allLabels.first())
        return NativeProject(
            id = "synthetic-cleanup",
            source = ProjectSource("content://fixture/cleanup", "synthetic-match.mp4", 1_024, 100, "video/mp4"),
            media = AnalysisTypes.MediaInfo(30.0, 1_920, 1_080, 0, "video/avc", "audio/mp4a-latm"),
            roi = AnalysisTypes.Roi(.05, .1, .9, .8, "fixture"),
            status = ProjectStatus.READY,
            ranges = listOf(SeedRange(5_000, 10_000, .8f, ProductionEnsemble.BOTH_MODELS),
                SeedRange(20_000, 25_000, .8f, ProductionEnsemble.ALL_LABELS_V2_ONLY)),
            productionComponents = AnalysisTypes.ProductionComponents(allLabels, previous),
            suppression = SuppressionPolicyEngine.build(allLabels, previous,
                FloatArray(analysis().timestamps.size) { .3f }, listOf(AnalysisTypes.Interval(21.0, 23.0, .9f)), 30.0),
            createdAtMs = 100,
            updatedAtMs = 100,
        )
    }
}
