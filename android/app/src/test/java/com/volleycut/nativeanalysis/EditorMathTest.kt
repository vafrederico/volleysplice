package com.volleycut.nativeanalysis

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
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

    @Test
    fun sourceRevisionChangesWhenModelAgreementChanges() {
        fun seed(agreement: String) = EditorSeed(
            sourceUri = "content://fixture/video",
            displayName = "fixture.mp4",
            durationMs = 10_000,
            width = 1_920,
            height = 1_080,
            rotation = 0,
            ranges = listOf(SeedRange(2_000, 3_000, .8f, agreement)),
        )

        assertNotEquals(
            seed(ProductionEnsemble.BOTH_MODELS).sourceRevision,
            seed(ProductionEnsemble.ALL_LABELS_V2_ONLY).sourceRevision,
        )
    }

    @Test
    fun markedGameWindowClampsPaddingAndIgnoresOutsideSource() {
        val seed = EditorSeed(
            sourceUri = "content://fixture/video",
            displayName = "fixture.mp4",
            durationMs = 20_000,
            width = 1_920,
            height = 1_080,
            rotation = 0,
            ranges = listOf(SeedRange(5_500, 14_500, .8f)),
            gameStartMs = 5_000,
            gameEndMs = 15_000,
        )

        val draft = EditorMath.newDraft(seed)
        assertEquals(5_000, draft.cuts.single().keepStartMs)
        assertEquals(15_000, draft.cuts.single().keepEndMs)
        assertEquals(
            listOf(
                IgnoredSourceInterval("G001", 0, 5_000, "outside-game-window"),
                IgnoredSourceInterval("G002", 15_000, 20_000, "outside-game-window"),
            ),
            draft.ignoredIntervals,
        )
    }

    @Test
    fun finalIntervalsJoinOnlyPositiveGapsStrictlyBelowThreshold() {
        val first = cut("R001", 0, 5_000)
        val second = cut("R002", 7_500, 10_000)
        val draft = EditorDraft(
            sourceRevision = "fixture",
            updatedAtMs = 0,
            cuts = listOf(first, second),
        )

        assertEquals(
            listOf(FinalCutInterval(
                0,
                10_000,
                listOf("R001", "R002"),
                listOf(JoinedGap(5_000, 7_500)),
            )),
            EditorMath.finalIntervals(draft),
        )
        assertEquals(2, EditorMath.finalIntervals(
            draft.copy(cuts = listOf(first, second.copy(
                coreStartMs = 8_000,
                keepStartMs = 8_000,
            ))),
        ).size)
        assertEquals(2, EditorMath.finalIntervals(draft.copy(joinGapMs = 0)).size)
    }

    @Test
    fun ignoredTimeSplitsJoinedOutputAndClipsItsGapMarkers() {
        val draft = EditorDraft(
            sourceRevision = "fixture",
            updatedAtMs = 0,
            cuts = listOf(cut("R001", 0, 5_000), cut("R002", 7_500, 10_000)),
            ignoredIntervals = listOf(IgnoredSourceInterval("I001", 6_000, 7_000, "break")),
        )

        assertEquals(
            listOf(
                FinalCutInterval(0, 6_000, listOf("R001"), listOf(JoinedGap(5_000, 6_000))),
                FinalCutInterval(7_000, 10_000, listOf("R002"), listOf(JoinedGap(7_000, 7_500))),
            ),
            EditorMath.finalIntervals(draft),
        )
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
