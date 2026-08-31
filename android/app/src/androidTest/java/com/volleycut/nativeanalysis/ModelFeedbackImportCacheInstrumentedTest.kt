package com.volleycut.nativeanalysis

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.time.Instant

@RunWith(AndroidJUnit4::class)
class ModelFeedbackImportCacheInstrumentedTest {
    @Test
    fun completeFeedbackFeaturesHydrateNativeCacheAndSurviveReexport() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val times = AnalysisEngine.analysisTimes(2.0, 0.0, 2.0)
        val allLabelsServe = AnalysisTypes.ProductionServeOutput(
            FeatureSchema.ALL_LABELS_V2_MODEL_ID,
            times,
            FloatArray(times.size) { .2f },
            emptyList(),
        )
        val previousServe = AnalysisTypes.ProductionServeOutput(
            FeatureSchema.PREVIOUS_PRODUCTION_MODEL_ID,
            times,
            FloatArray(times.size) { .1f },
            emptyList(),
        )
        val sourceProject = NativeProject(
            id = "instrumented-feedback-source",
            source = ProjectSource(
                "content://instrumented/feedback-source",
                "feedback-source.mp4",
                1_024,
                123,
                "video/mp4",
            ),
            media = AnalysisTypes.MediaInfo(2.0, 1_920, 1_080, 0, "video/avc", "audio/mp4a-latm"),
            analysisWindow = AnalysisTypes.AnalysisWindow(0.0, 2.0),
            roi = AnalysisTypes.Roi(.05, .1, .9, .8, "fixture"),
            status = ProjectStatus.READY,
            ranges = listOf(SeedRange(250, 1_250, .8f, ProductionEnsemble.BOTH_MODELS)),
            productionServeOutputs = AnalysisTypes.ProductionServeOutputs(allLabelsServe, previousServe),
            suppression = AnalysisTypes.SuppressionAnalysis(
                FeatureSchema.SUPPRESSION_MODEL_ID,
                FeatureSchema.SUPPRESSION_ARTIFACT_SHA256,
                FeatureSchema.SUPPRESSION_WEIGHTS_SHA256,
                FeatureSchema.SUPPRESSION_DECODER_VERSION,
                FloatArray(times.size) { .3f },
                listOf(AnalysisTypes.Interval(.5, .75, .7f)),
                listOf(AnalysisTypes.SuppressionSuggestion(
                    "instrumented-suppression",
                    "instrumented-suppression-500-750",
                    500,
                    750,
                    .7f,
                    listOf("all-labels-v2:0001:250:1250"),
                    listOf("aggressive"),
                )),
            ),
            createdAtMs = 100,
            updatedAtMs = 100,
        )
        val sourceDraft = EditorMath.newDraft(checkNotNull(sourceProject.editorSeed())).copy(
            selectedSuppressionPolicy = SuppressionPolicyEngine.Policy.AGGRESSIVE,
        )
        val sourceAnalysis = ModelFeedbackAnalysis(
            timestamps = times,
            baseFeatures = FloatArray(times.size * FeatureSchema.BASE.size),
            rallyProbabilities = FloatArray(times.size) { .4f },
            serveProbabilities = FloatArray(times.size) { .2f },
            deadStateProbabilities = FloatArray(times.size) { .6f },
        )
        val sourceBundle = ModelFeedbackExporter.createBundle(
            sourceProject,
            sourceDraft,
            EditorMath.finalIntervals(sourceDraft, sourceProject.suppression),
            sourceAnalysis,
            "sampled-sha256-v1:${"a".repeat(64)}",
            Instant.parse("2026-08-30T00:00:00Z"),
        )

        var importedProject: NativeProject? = null
        try {
            importedProject = ModelFeedbackImporter.import(context, sourceBundle.toString())
            val retainedAnalysis = requireNotNull(
                ModelFeedbackExporter.loadAnalysis(context, importedProject),
            )
            assertArrayEquals(times, retainedAnalysis.timestamps, 0.0)
            assertArrayEquals(sourceAnalysis.baseFeatures, retainedAnalysis.baseFeatures, 0f)

            val importedSeed = checkNotNull(importedProject.editorSeed())
            val importedDraft = checkNotNull(EditorDraftStore(context, importedSeed).load())
            val reexported = ModelFeedbackExporter.createBundle(
                importedProject,
                importedDraft,
                EditorMath.finalIntervals(importedDraft, importedProject.suppression),
                retainedAnalysis,
                importedProject.source.sampledFingerprint,
                Instant.parse("2026-08-30T00:01:00Z"),
            )
            assertFalse(reexported.isNull("features"))
            assertFalse(reexported.getJSONObject("initialInference").isNull("suppression"))
            assertNotNull(reexported.getJSONObject("initialInference").getJSONObject("componentServeOutputs"))
            File(requireNotNull(context.getExternalFilesDir(null)), REEXPORT_FILENAME)
                .writeText(reexported.toString() + "\n")
        } finally {
            importedProject?.let { NativeProjectStore.delete(context, it) }
        }
    }

    private companion object {
        const val REEXPORT_FILENAME = "model-feedback-import-reexport.json"
    }
}
