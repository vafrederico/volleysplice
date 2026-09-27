package com.volleycut.nativeanalysis

import org.junit.Assert.assertEquals
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.time.Instant
import java.util.Base64

class ModelFeedbackExporterTest {
    @Test
    fun eachRallyModelKeepsItsIdentityThroughExportAndImport() {
        for (model in RallyModels.OPTIONS) {
            val original = project().let { value -> value.copy(
                modelId = model,
                neuralScores = if (RallyModels.isNeural(model)) NeuralRallyScores(model,
                    doubleArrayOf(2.0, 2.25), floatArrayOf(.1f, .2f, .3f, .4f, .5f, .6f, .7f, .8f)) else null,
                ranges = value.ranges.map { it.copy(agreement = if (RallyModels.isNeural(model)) "neural" else it.agreement) },
            ) }
            val draft = EditorDraft(sourceRevision = "fixture", updatedAtMs = original.updatedAtMs, cuts = emptyList())
            val bundle = ModelFeedbackExporter.createBundle(original, draft, emptyList(), null, null)
            val inference = bundle.getJSONObject("initialInference")
            assertEquals(model, inference.getString("modelId"))
            assertEquals(RallyModels.label(model), inference.getString("modelLabel"))
            assertEquals(if (RallyModels.isNeural(model)) "score-support" else "rally-and-score",
                inference.getString("componentsRole"))
            assertEquals(FeatureSchema.ALL_LABELS_V2_MODEL_ID,
                inference.getJSONObject("probabilityModelIds").getString("rally"))
            val imported = ModelFeedbackImporter.parse(bundle.toString(), 1_786_752_001_000)
            assertEquals(model, imported.project.modelId)
            assertEquals(ProjectStatus.READY, imported.project.status)
            assertEquals(original.ranges, imported.project.ranges)
            if (original.neuralScores != null) {
                // Preserve the four original heads even without an AV cache or source video.
                val restored = requireNotNull(NativeProjectStore.decode(NativeProjectStore.encode(imported.project)))
                val scores = requireNotNull(restored.neuralScores)
                assertArrayEquals(original.neuralScores.timestamps, scores.timestamps, 0.0)
                assertArrayEquals(original.neuralScores.probabilities, scores.probabilities, 0f)
                val again = ModelFeedbackExporter.createBundle(restored, draft, emptyList(), null, null)
                assertEquals(inference.getJSONObject("neuralScores").toString(),
                    again.getJSONObject("initialInference").getJSONObject("neuralScores").toString())
                assertFalse(again.toString().contains("embeddings"))
            } else assertTrue(inference.isNull("neuralScores"))
        }
    }

    @Test
    fun neuralScoreImportRejectsWrongIdentityShapeTimelineAndProbability() {
        val scores = NeuralRallyScores(RallyModels.F1, doubleArrayOf(2.0, 2.25),
            floatArrayOf(.1f, .2f, .3f, .4f, .5f, .6f, .7f, .8f))
        val invalid = listOf(
            scores.encode().put("modelId", RallyModels.RECALL),
            scores.encode().apply { getJSONObject("probabilities").put("shape", org.json.JSONArray(listOf(4, 2))) },
            scores.encode().put("timestamps", ModelFeedbackExporter.encode(doubleArrayOf(2.25, 2.0), intArrayOf(2))),
            scores.encode().put("probabilities", ModelFeedbackExporter.encode(FloatArray(8) { 1.1f }, intArrayOf(2, 4))),
        )
        for (json in invalid) {
            org.junit.Assert.assertThrows(IllegalArgumentException::class.java) {
                NeuralRallyScores.decode(json, RallyModels.F1, 2.0, 28.0)
            }
        }
    }

    @Test
    fun bundlePreservesInferenceAndClassifiesCorrectionsWithoutVideo() {
        val project = project()
        val inferredRemoved = cut("R001", 5_000, 8_000, CutOrigin.INFERRED, false)
        val inferredKept = cut("R002", 12_000, 15_000, CutOrigin.INFERRED, true)
        val manualKept = cut("M001", 18_000, 20_000, CutOrigin.MANUAL, true)
        val manualRemoved = cut("M002", 22_000, 23_000, CutOrigin.MANUAL, false)
        val draft = EditorDraft(
            sourceRevision = "fixture",
            updatedAtMs = 1_786_752_000_000,
            cuts = listOf(inferredRemoved, inferredKept, manualKept, manualRemoved),
            ignoredIntervals = listOf(IgnoredSourceInterval("I001", 24_000, 25_000, "break")),
            suppressionInitialBehavior = SuppressionInitialBehavior.HIGHLIGHT_ONLY,
        )

        val bundle = ModelFeedbackExporter.createBundle(
            project,
            draft,
            EditorMath.finalIntervals(draft),
            analysis = null,
            sampledFingerprint = "sampled-sha256-v1:${"a".repeat(64)}",
            generatedAt = Instant.parse("2026-08-15T02:00:00Z"),
        )

        assertEquals(MODEL_FEEDBACK_SCHEMA, bundle.getString("schema"))
        assertFalse(bundle.getJSONObject("source").getBoolean("videoBytesIncluded"))
        assertTrue(bundle.isNull("features"))
        val labels = bundle.getJSONObject("corrections").getJSONObject("labels")
        assertEquals("R001", labels.getJSONArray("falsePositives").getJSONObject(0).getString("id"))
        assertEquals("M001", labels.getJSONArray("falseNegatives").getJSONObject(0).getString("id"))
        assertEquals("R002", labels.getJSONArray("confirmedModelRanges").getJSONObject(0).getString("id"))
        assertEquals("M002", labels.getJSONArray("discardedManualRanges").getJSONObject(0).getString("id"))
        assertEquals(2, bundle.getJSONObject("initialInference").getJSONArray("ranges").length())
        assertEquals(
            "highlight-only",
            bundle.getJSONObject("corrections").getString("suppressionInitialBehavior"),
        )
        assertEquals(
            "whole-rally",
            bundle.getJSONObject("corrections").getString("defaultSuppressionScope"),
        )
        assertTrue(bundle.getJSONArray("warnings").getString(0).contains("feature cache"))

        val imported = ModelFeedbackImporter.parse(bundle.toString(), 1_786_752_001_000)
        assertNull(imported.featureCache)
        assertEquals(draft.cuts, imported.draft.cuts)
    }

    @Test
    fun numericArraysAreLittleEndianBase64() {
        val encoded = ModelFeedbackExporter.encode(floatArrayOf(1.25f, -2.5f), intArrayOf(1, 2))
        val bytes = Base64.getDecoder().decode(encoded.getString("data"))
        val floats = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()

        assertEquals("float32", encoded.getString("dataType"))
        assertEquals(1.25f, floats.get(0), 0f)
        assertEquals(-2.5f, floats.get(1), 0f)
    }

    @Test
    fun feedbackFilenameMatchesWebContract() {
        assertEquals(
            "Match-One.model-feedback.json",
            ModelFeedbackExporter.filename("Match One.mp4"),
        )
    }

    private fun project() = NativeProject(
        id = "project-1",
        source = ProjectSource(
            uri = "content://fixture/video",
            name = "Match One.mp4",
            size = 1_234,
            lastModified = 1_786_752_000_000,
            mimeType = "video/mp4",
        ),
        media = AnalysisTypes.MediaInfo(30.0, 1_920, 1_080, 0, "video/avc", "audio/mp4a-latm"),
        analysisWindow = AnalysisTypes.AnalysisWindow(2.0, 28.0),
        roi = AnalysisTypes.Roi(.03, .12, .94, .86, "fixture"),
        status = ProjectStatus.READY,
        ranges = listOf(
            SeedRange(5_000, 8_000, .8f, ProductionEnsemble.BOTH_MODELS),
            SeedRange(12_000, 15_000, .6f, ProductionEnsemble.ALL_LABELS_V2_ONLY),
        ),
        createdAtMs = 1_786_752_000_000,
        updatedAtMs = 1_786_752_000_000,
    )

    private fun cut(
        id: String,
        startMs: Long,
        endMs: Long,
        origin: CutOrigin,
        included: Boolean,
    ) = EditableCut(
        id = id,
        coreStartMs = startMs,
        coreEndMs = endMs,
        keepStartMs = startMs,
        keepEndMs = endMs,
        confidence = 1f,
        included = included,
        origin = origin,
        agreement = if (origin == CutOrigin.INFERRED) ProductionEnsemble.BOTH_MODELS else null,
    )
}
