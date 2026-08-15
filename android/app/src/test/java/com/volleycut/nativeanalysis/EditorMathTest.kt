package com.volleycut.nativeanalysis

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class EditorMathTest {
    @Test
    fun finalIntervalsMergeTouchingCutsAndSubtractIgnoredTime() {
        val draft = EditorDraft(
            sourceRevision = "fixture",
            updatedAtMs = 0,
            cuts = listOf(
                cut("R001", 1_000, 3_000),
                cut("R002", 3_000, 6_000),
                cut("R003", 8_000, 9_000),
            ),
            ignoredIntervals = listOf(
                IgnoredSourceInterval("I001", 2_000, 4_000, "camera-gap"),
                IgnoredSourceInterval("I002", 5_000, 8_500, "camera-gap"),
            ),
        )

        assertEquals(
            listOf(
                FinalCutInterval(1_000, 2_000, listOf("R001")),
                FinalCutInterval(4_000, 5_000, listOf("R002")),
                FinalCutInterval(8_500, 9_000, listOf("R003")),
            ),
            EditorMath.finalIntervals(draft),
        )
    }

    @Test
    fun globalPaddingChangesOnlyInferredCuts() {
        val draft = EditorDraft(
            sourceRevision = "fixture",
            updatedAtMs = 0,
            cuts = listOf(
                cut("R001", 5_000, 6_000),
                cut("M001", 8_000, 9_000, CutOrigin.MANUAL),
            ),
        )

        val padded = EditorMath.applyPadding(draft, 3_000, 1_500, 20_000)
        assertEquals(2_000, padded.cuts[0].keepStartMs)
        assertEquals(7_500, padded.cuts[0].keepEndMs)
        assertEquals(8_000, padded.cuts[1].keepStartMs)
        assertEquals(9_000, padded.cuts[1].keepEndMs)
    }

    @Test
    fun fullyIgnoredCutIsNotEffective() {
        val draft = EditorDraft(
            sourceRevision = "fixture",
            updatedAtMs = 0,
            cuts = listOf(cut("R001", 1_000, 2_000), cut("R002", 3_000, 4_000)),
            ignoredIntervals = listOf(IgnoredSourceInterval("I001", 900, 2_100, "gap")),
        )
        val ids = EditorMath.effectiveKeptIds(draft)
        assertFalse("R001" in ids)
        assertTrue("R002" in ids)
    }

    @Test
    fun invalidOrEmptySeedRangesAreSkippedAndIdsRemainContiguous() {
        val draft = EditorMath.newDraft(EditorSeed(
            sourceUri = "content://fixture/video",
            displayName = "fixture.mp4",
            durationMs = 10_000,
            width = 1_920,
            height = 1_080,
            rotation = 0,
            ranges = listOf(
                SeedRange(10_000, 11_000, .2f),
                SeedRange(2_000, 3_000, .8f),
            ),
        ))

        assertEquals(listOf("R001"), draft.cuts.map { it.id })
        assertEquals(0, draft.cuts.single().keepStartMs)
        assertEquals(5_000, draft.cuts.single().keepEndMs)
    }

    @Test
    fun sourceRevisionChangesWhenInferenceConfidenceChanges() {
        fun seed(confidence: Float) = EditorSeed(
            sourceUri = "content://fixture/video",
            displayName = "fixture.mp4",
            durationMs = 10_000,
            width = 1_920,
            height = 1_080,
            rotation = 0,
            ranges = listOf(SeedRange(2_000, 3_000, confidence)),
        )

        assertFalse(seed(.7f).sourceRevision == seed(.8f).sourceRevision)
    }

    private fun cut(
        id: String,
        start: Long,
        end: Long,
        origin: CutOrigin = CutOrigin.INFERRED,
    ) = EditableCut(
        id = id,
        coreStartMs = start,
        coreEndMs = end,
        keepStartMs = start,
        keepEndMs = end,
        confidence = 1f,
        included = true,
        origin = origin,
    )
}
