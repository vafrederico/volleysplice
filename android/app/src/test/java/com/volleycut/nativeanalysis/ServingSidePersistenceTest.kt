package com.volleycut.nativeanalysis

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.file.Files
import java.nio.file.Path
import java.time.Instant
import java.util.concurrent.TimeUnit

class ServingSidePersistenceTest {
    @Test
    fun cacheIdentityInvalidatesEveryInferenceInputExceptEditorIgnoredRanges() {
        val project = projectFixture()
        val identity = ServingSideCache.identity(project)
        val output = requireNotNull(project.servingSide)
        assertTrue(ServingSideCache.isReusable(identity, project, output))
        assertFalse(ServingSideCache.isReusable(identity, project.copy(
            source = project.source.copy(uri = "content://replacement"),
        ), output))
        assertFalse(ServingSideCache.isReusable(identity, project.copy(
            roi = project.roi.let { AnalysisTypes.Roi(it.x() + .01, it.y(), it.width(), it.height(), it.label()) },
        ), output))
        assertFalse(ServingSideCache.isReusable(identity, project.copy(
            analysisWindow = AnalysisTypes.AnalysisWindow(0.5, project.analysisWindow.end()),
        ), output))
        assertFalse(ServingSideCache.isReusable(identity, project.copy(
            ranges = listOf(project.ranges.single().copy(startMs = 1_100)),
        ), output))
        assertFalse(ServingSideCache.isReusable(identity, project, output.copy(modelId = "changed")))
        assertFalse(ServingSideCache.isReusable(identity, project, output.copy(featureVersion = "changed")))
        assertEquals(identity, requireNotNull(project.servingSideCacheIdentity))
        val decoded = requireNotNull(NativeProjectStore.decode(NativeProjectStore.encode(project)))
        assertEquals(identity, decoded.servingSideCacheIdentity)
        assertArrayEquals(output.rawFeatures, requireNotNull(decoded.servingSide).rawFeatures, 0.0)
        assertNotNull(NativeProjectStore.normalizeStored(project).servingSide)
        val missingIdentity = NativeProjectStore.normalizeStored(project.copy(
            servingSideCacheIdentity = null,
        ))
        assertEquals(ProjectStatus.READY, missingIdentity.status)
        assertEquals(project.ranges, missingIdentity.ranges)
        assertEquals(null, missingIdentity.servingSide)
        assertEquals(null, NativeProjectStore.normalizeStored(project.copy(
            source = project.source.copy(uri = "content://replacement"),
        )).servingSide)

        // Ignored ranges live only in the editor draft and are deliberately not
        // represented by the serving-side cache identity.
        val ignored = EditorMath.newDraft(requireNotNull(project.editorSeed())).copy(
            ignoredIntervals = listOf(IgnoredSourceInterval("I1", 3_000, 4_000, "camera-gap")),
        )
        assertEquals(identity, ServingSideCache.identity(project))
        assertEquals(1, ignored.ignoredIntervals.size)
    }

    @Test
    fun editListAndFeedbackV3RoundTripServingAndScoreStateWithoutInference() {
        val project = projectFixture()
        val seed = requireNotNull(project.editorSeed())
        val seeded = EditorMath.newDraft(seed)
        val corrected = seeded.scoreTracking.copy(
            team1Name = "Falcons",
            team2Name = "Waves",
            serveMarkers = seeded.scoreTracking.serveMarkers.map {
                it.copy(side = ServingSide.FAR, ignorePreviousPoint = true)
            } + ServeMarker("S001", 8_000, ServingSide.NEAR, ServeMarkerOrigin.MANUAL),
            removedModelMarkerIds = setOf("serve-R999"),
        )
        val draft = seeded.copy(
            updatedAtMs = 1_728_000_000_000,
            scoreTracking = corrected,
            renderScoreOverlay = true,
            selectedSuppressionPolicy = SuppressionPolicyEngine.Policy.BALANCED,
            suppressionDecisionOverrides = mapOf("suppression-fixture" to SuppressionDecision.KEEP),
            suppressionScopeOverrides = mapOf("suppression-fixture" to SuppressionScope.VETO_REGION),
        )
        val intervals = EditorMath.finalIntervals(draft, project.suppression)
        val editList = editListJson(seed, draft, intervals)
        assertTrue(editList.getBoolean("renderScoreOverlay"))
        assertEquals(2, editList.getJSONObject("scoreTracking").getInt("version"))
        assertEquals("Falcons", editList.getJSONObject("scoreTracking").getString("team1Name"))

        val times = project.productionServeOutputs.allLabelsV2().times()
        val analysis = ModelFeedbackAnalysis(
            timestamps = times,
            baseFeatures = FloatArray(times.size * FeatureSchema.BASE.size),
            rallyProbabilities = floatArrayOf(.8f, .7f),
            serveProbabilities = floatArrayOf(.9f, .1f),
            deadStateProbabilities = floatArrayOf(.1f, .2f),
        )
        val bundle = ModelFeedbackExporter.createBundle(
            project, draft, intervals, analysis,
            "sampled-sha256-v1:" + "a".repeat(64),
            Instant.parse("2026-08-22T00:00:00Z"),
        )
        assertEquals(3, bundle.getInt("schemaVersion"))
        assertEquals(
            "float64",
            bundle.getJSONObject("initialInference").getJSONObject("servingSide")
                .getJSONObject("features").getJSONObject("values").getString("dataType"),
        )
        assertEquals(
            2,
            bundle.getJSONObject("corrections").getJSONObject("scoreTracking")
                .getJSONObject("state").getInt("version"),
        )
        assertNotNull(bundle.getJSONObject("corrections").getJSONObject("suppression"))
        assertFalse(bundle.has("renderScoreOverlay"))
        assertBrowserAccepts(bundle.toString())

        val imported = ModelFeedbackImporter.parse(bundle.toString(), 1234)
        assertEquals(ProjectStatus.READY, imported.project.status)
        assertEquals(
            "sampled-sha256-v1:" + "a".repeat(64),
            imported.project.source.sampledFingerprint,
        )
        assertEquals("Falcons", imported.draft.scoreTracking.team1Name)
        assertEquals("Waves", imported.draft.scoreTracking.team2Name)
        assertEquals(SuppressionPolicyEngine.Policy.BALANCED, imported.draft.selectedSuppressionPolicy)
        assertEquals(
            SuppressionDecision.KEEP,
            imported.draft.suppressionDecisionOverrides["suppression-fixture"],
        )
        assertNotNull(imported.project.suppression)
        assertEquals(setOf("serve-R999"), imported.draft.scoreTracking.removedModelMarkerIds)
        assertFalse(imported.draft.renderScoreOverlay)
        assertEquals(ServingSide.FAR, imported.draft.scoreTracking.serveMarkers.first().side)
        assertTrue(imported.draft.scoreTracking.serveMarkers.first().ignorePreviousPoint)
        assertArrayEquals(
            requireNotNull(project.servingSide).rawFeatures,
            requireNotNull(imported.project.servingSide).rawFeatures,
            0.0,
        )
        assertArrayEquals(
            project.productionServeOutputs.previousProduction().probabilities(),
            imported.project.productionServeOutputs.previousProduction().probabilities(),
            0f,
        )
    }

    private fun assertBrowserAccepts(json: String) {
        val root = repositoryRoot()
        val temporary = Files.createTempFile("volleycut-android-feedback-", ".json")
        try {
            Files.writeString(temporary, json)
            val validator = root.resolve("prod/src/lib/model-feedback-validation.ts").toUri().toString()
            val script = """
                import { readFile } from 'node:fs/promises';
                import { parseModelFeedback } from '$validator';
                parseModelFeedback(JSON.parse(await readFile(process.argv[1], 'utf8')));
            """.trimIndent()
            val process = ProcessBuilder(
                "node", "--experimental-strip-types", "--input-type=module", "--eval", script,
                temporary.toAbsolutePath().toString(),
            ).redirectErrorStream(true).start()
            assertTrue("Browser validator timed out", process.waitFor(20, TimeUnit.SECONDS))
            val output = process.inputStream.bufferedReader().use { it.readText() }
            assertEquals("Browser rejected Android feedback:\n$output", 0, process.exitValue())
        } finally {
            Files.deleteIfExists(temporary)
        }
    }

    private fun repositoryRoot(): Path {
        var current: Path? = Path.of(System.getProperty("user.dir")).toAbsolutePath()
        while (current != null) {
            if (Files.exists(current.resolve("prod/src/lib/model-feedback-validation.ts"))) {
                return current
            }
            current = current.parent
        }
        error("Could not locate repository root")
    }

    private fun projectFixture(): NativeProject {
        val range = SeedRange(1_000, 5_000, .9f, ProductionEnsemble.BOTH_MODELS)
        val times = doubleArrayOf(1.0, 2.0)
        val all = AnalysisTypes.ProductionServeOutput(
            FeatureSchema.ALL_LABELS_V2_MODEL_ID,
            times,
            floatArrayOf(.9f, .1f),
            listOf(AnalysisTypes.Serve(1.0, .9f)),
        )
        val previous = AnalysisTypes.ProductionServeOutput(
            FeatureSchema.PREVIOUS_PRODUCTION_MODEL_ID,
            times,
            floatArrayOf(.86f, .2f),
            listOf(AnalysisTypes.Serve(1.0, .86f)),
        )
        val evidenceAll = ServingSideHeadEvidence(
            all.modelId(), .85, .9, 1.0, true, all.detections().single(),
        )
        val evidencePrevious = ServingSideHeadEvidence(
            previous.modelId(), .85, .86, 1.0, true, previous.detections().single(),
        )
        val serving = ServingSideOutput(
            rows = 1,
            rawFeatures = DoubleArray(237) { it / 237.0 },
            candidates = listOf(ServingSideCandidate(
                "R001", 1.0, 1.0, 5.0, ProductionEnsemble.BOTH_MODELS,
                .7, ServingSide.NEAR, ServingSideVerdict.NEAR,
                ServingSideDecisionSource.SERVE_HEAD, emptyList(), evidenceAll, evidencePrevious,
            )),
        )
        val project = NativeProject(
            id = "fixture",
            source = ProjectSource("content://fixture", "fixture.mp4", 10_000, 99, "video/mp4"),
            media = AnalysisTypes.MediaInfo(10.0, 1920, 1080, 0, "video/avc", "audio/mp4a-latm"),
            analysisWindow = AnalysisTypes.AnalysisWindow(0.0, 10.0),
            roi = AnalysisTypes.Roi(0.1, 0.2, 0.8, 0.7, "fixture"),
            status = ProjectStatus.READY,
            ranges = listOf(range),
            productionComponents = AnalysisTypes.ProductionComponents(
                listOf(AnalysisTypes.Interval(1.0, 5.0, .9f, ProductionEnsemble.BOTH_MODELS)),
                listOf(AnalysisTypes.Interval(1.0, 5.0, .8f, ProductionEnsemble.BOTH_MODELS)),
            ),
            productionServeOutputs = AnalysisTypes.ProductionServeOutputs(all, previous),
            servingSide = serving,
            suppression = AnalysisTypes.SuppressionAnalysis(
                FeatureSchema.SUPPRESSION_MODEL_ID,
                FeatureSchema.SUPPRESSION_ARTIFACT_SHA256,
                FeatureSchema.SUPPRESSION_WEIGHTS_SHA256,
                FeatureSchema.SUPPRESSION_DECODER_VERSION,
                floatArrayOf(.2f, .3f),
                listOf(AnalysisTypes.Interval(1.2, 1.5, .7f)),
                listOf(AnalysisTypes.SuppressionSuggestion(
                    "suppression-fixture", "suppression-fixture-1200-1500",
                    1_200, 1_500, .7f,
                    listOf("all-labels-v2:0001:1000:5000"),
                    listOf("conservative", "balanced", "aggressive"),
                )),
            ),
            createdAtMs = 100,
            updatedAtMs = 200,
        )
        return project.copy(servingSideCacheIdentity = ServingSideCache.identity(project))
    }
}
