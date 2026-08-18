package com.volleycut.nativeanalysis

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.time.Instant
import java.util.Base64

class ModelFeedbackExporterTest {
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
