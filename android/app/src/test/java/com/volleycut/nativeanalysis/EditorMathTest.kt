package com.volleycut.nativeanalysis

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class EditorMathTest {
    @Test
    fun newDraftUsesSimpleAutomaticDefaults() {
        val draft = EditorMath.newDraft(EditorSeed(
            sourceUri = "content://fixture/video",
            displayName = "fixture.mp4",
            durationMs = 10_000,
            width = 1_920,
            height = 1_080,
            rotation = 0,
            ranges = emptyList(),
        ))

        assertEquals(SuppressionPolicyEngine.Policy.AGGRESSIVE, draft.selectedSuppressionPolicy)
        assertFalse(draft.scoreTracking.enabled)
        assertTrue(draft.renderScoreOverlay)
        assertTrue(draft.renderScoreTimeline)
        assertTrue(draft.reviewedCutIds.isEmpty())
    }

    @Test
    fun scoreTrackingStartsDisabledWhenServingSideWasSkipped() {
        val draft = EditorMath.newDraft(EditorSeed(
            sourceUri = "content://fixture/video",
            displayName = "fixture.mp4",
            durationMs = 10_000,
            width = 1_920,
            height = 1_080,
            rotation = 0,
            ranges = emptyList(),
            scoreTrackingInitiallyEnabled = false,
        ))

        assertFalse(draft.scoreTracking.enabled)
    }

    @Test
    fun newProjectEnablesScoreAndPointTimelineRenderingWithScoreMarkers() {
        val draft = EditorMath.newDraft(EditorSeed(
            sourceUri = "content://fixture/video",
            displayName = "fixture.mp4",
            durationMs = 10_000,
            width = 1_920,
            height = 1_080,
            rotation = 0,
            ranges = emptyList(),
            scoreTrackingInitiallyEnabled = true,
        ))

        assertTrue(draft.scoreTracking.enabled)
        assertTrue(draft.renderScoreOverlay)
        assertTrue(draft.renderScoreTimeline)
    }

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
    fun correctingCoreEdgesPreservesPerCutPaddingAndSurvivesGlobalPaddingChanges() {
        val base = EditorDraft(
            sourceRevision = "fixture",
            updatedAtMs = 0,
            beforePaddingMs = 2_000,
            afterPaddingMs = 2_000,
            cuts = listOf(cut("R001", 10_000, 30_000).copy(
                keepStartMs = 8_000,
                keepEndMs = 32_000,
            )),
        )

        val trimmed = EditorMath.setCoreRange(base, "R001", 12_000, 20_000, 0, 40_000)
        assertEquals(12_000, trimmed.cuts.single().coreStartMs)
        assertEquals(20_000, trimmed.cuts.single().coreEndMs)
        assertEquals(10_000, trimmed.cuts.single().keepStartMs)
        assertEquals(22_000, trimmed.cuts.single().keepEndMs)

        val repadded = EditorMath.applyPadding(trimmed, 3_000, 1_000, 40_000)
        assertEquals(9_000, repadded.cuts.single().keepStartMs)
        assertEquals(21_000, repadded.cuts.single().keepEndMs)
    }

    @Test
    fun splitCreatesIndependentlyEditablePartsWithTheirExistingPadding() {
        val base = EditorDraft(
            sourceRevision = "fixture",
            updatedAtMs = 0,
            joinGapMs = 0,
            cuts = listOf(cut("R001", 10_000, 30_000).copy(
                keepStartMs = 8_000,
                keepEndMs = 32_000,
            )),
        )

        val split = EditorMath.splitCut(base, "R001", 20_000, 0, 40_000)!!
        assertEquals(listOf("R001", "R002"), split.draft.cuts.map { it.id })
        assertEquals(10_000, split.draft.cuts[0].coreStartMs)
        assertEquals(20_000, split.draft.cuts[0].coreEndMs)
        assertEquals(8_000, split.draft.cuts[0].keepStartMs)
        assertEquals(22_000, split.draft.cuts[0].keepEndMs)
        assertEquals(20_000, split.newCut.coreStartMs)
        assertEquals(30_000, split.newCut.coreEndMs)
        assertEquals(18_000, split.newCut.keepStartMs)
        assertEquals(32_000, split.newCut.keepEndMs)

        val shortenedLeft = EditorMath.setCoreEnd(split.draft, "R001", 15_000, 40_000)
        val separated = EditorMath.setCoreStart(shortenedLeft, "R002", 25_000, 0)
        assertEquals(
            listOf(
                FinalCutInterval(8_000, 17_000, listOf("R001")),
                FinalCutInterval(23_000, 32_000, listOf("R002")),
            ),
            EditorMath.finalIntervals(separated),
        )
    }

    @Test
    fun splitRequiresRoomForBothParts() {
        val draft = EditorDraft(
            sourceRevision = "fixture",
            updatedAtMs = 0,
            cuts = listOf(cut("R001", 10_000, 11_000)),
        )

        assertEquals(null, EditorMath.splitCut(draft, "R001", 10_050, 0, 20_000))
        assertEquals(null, EditorMath.splitCut(draft, "R001", 10_950, 0, 20_000))
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
    fun overlappingPaddingProducesOneContinuousIntervalWithoutASyntheticGap() {
        val first = cut("R010", 146_125, 160_375).copy(
            keepStartMs = 144_125,
            keepEndMs = 162_375,
        )
        val second = cut("R011", 162_375, 174_375).copy(
            keepStartMs = 160_375,
            keepEndMs = 176_375,
        )
        val draft = EditorDraft(
            sourceRevision = "fixture",
            updatedAtMs = 0,
            cuts = listOf(first, second),
        )

        assertEquals(
            listOf(FinalCutInterval(144_125, 176_375, listOf("R010", "R011"))),
            EditorMath.finalIntervals(draft),
        )
    }

    @Test
    fun joinedAndOverlappingCutsShareOneEditableRallyUntilAServeSeparatesThem() {
        val first = cut("R010", 10_000, 20_000).copy(keepStartMs = 8_000, keepEndMs = 22_000)
        val overlapping = cut("R011", 22_000, 30_000).copy(keepStartMs = 20_000, keepEndMs = 32_000)
        val joined = cut("R012", 34_000, 40_000).copy(keepStartMs = 34_000, keepEndMs = 42_000)
        val intervals = listOf(FinalCutInterval(
            8_000,
            42_000,
            listOf("R010", "R011", "R012"),
            listOf(JoinedGap(32_000, 34_000)),
        ))

        assertEquals(
            listOf(setOf("R010", "R011", "R012")),
            EditorMath.editableRallyGroups(listOf(first, overlapping, joined), intervals, emptyList())
                .map { it.cutIds },
        )
        assertEquals(
            listOf(setOf("R010"), setOf("R011", "R012")),
            EditorMath.editableRallyGroups(
                listOf(first, overlapping, joined),
                intervals,
                listOf(ServeMarker(
                    "S011",
                    22_000,
                    ServingSide.NEAR,
                    ServeMarkerOrigin.MODEL,
                    rallyId = "R011",
                )),
            ).map { it.cutIds },
        )
    }

    @Test
    fun pendingConfidenceReviewIsAnEditableBoundaryUntilKept() {
        val first = cut("R010", 10_000, 20_000).copy(keepStartMs = 8_000, keepEndMs = 22_000)
        val needsReview = cut("R011", 22_000, 30_000).copy(
            keepStartMs = 20_000,
            keepEndMs = 32_000,
            confidence = .4f,
        )
        val last = cut("R012", 32_000, 40_000).copy(keepStartMs = 30_000, keepEndMs = 42_000)
        val cuts = listOf(first, needsReview, last)
        val intervals = EditorMath.finalIntervals(EditorDraft(
            sourceRevision = "fixture",
            updatedAtMs = 0,
            cuts = cuts,
        ))

        assertEquals(
            listOf(setOf("R010"), setOf("R011"), setOf("R012")),
            EditorMath.editableRallyGroups(
                cuts,
                intervals,
                emptyList(),
                hardBoundaryCutIds = setOf("R011"),
            ).map { it.cutIds },
        )
        assertEquals(
            listOf(setOf("R010", "R011", "R012")),
            EditorMath.editableRallyGroups(cuts, intervals, emptyList()).map { it.cutIds },
        )
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

    @Test
    fun untouchedSuggestionsDefaultToWholeRallyWhileTouchedAndExplicitChoicesWin() {
        val suppression = suppression(12_000, 18_000)
        val base = EditorDraft(
            sourceRevision = "fixture",
            updatedAtMs = 0,
            beforePaddingMs = 0,
            afterPaddingMs = 0,
            joinGapMs = 0,
            selectedSuppressionPolicy = SuppressionPolicyEngine.Policy.AGGRESSIVE,
            cuts = listOf(cut("R001", 10_000, 20_000)),
        )

        assertTrue(EditorMath.finalIntervals(base, suppression).isEmpty())
        assertEquals(
            listOf(FinalCutInterval(10_000, 20_000, listOf("R001"))),
            EditorMath.finalIntervals(
                base.copy(suppressionInitialBehavior = SuppressionInitialBehavior.HIGHLIGHT_ONLY),
                suppression,
            ),
        )
        assertEquals(
            listOf(FinalCutInterval(10_000, 20_000, listOf("R001"))),
            EditorMath.finalIntervals(base.copy(userTouchedCutIds = setOf("R001")), suppression),
        )
        assertEquals(
            0,
            EditorMath.finalIntervals(base.copy(
                userTouchedCutIds = setOf("R001"),
                suppressionDecisionOverrides = mapOf("S-fixture" to SuppressionDecision.SUPPRESS),
            ), suppression).size,
        )
        assertEquals(
            1,
            EditorMath.finalIntervals(base.copy(
                suppressionDecisionOverrides = mapOf("S-fixture" to SuppressionDecision.KEEP),
            ), suppression).size,
        )
        assertEquals(
            listOf(
                FinalCutInterval(10_000, 12_000, listOf("R001")),
                FinalCutInterval(18_000, 20_000, listOf("R001")),
            ),
            EditorMath.finalIntervals(base.copy(
                suppressionScopeOverrides = mapOf("S-fixture" to SuppressionScope.VETO_REGION),
            ), suppression),
        )
    }

    @Test
    fun suppressionRemainsAHardBoundaryAfterPaddingAndShortGapJoining() {
        val draft = EditorDraft(
            sourceRevision = "fixture",
            updatedAtMs = 0,
            beforePaddingMs = 2_000,
            afterPaddingMs = 2_000,
            joinGapMs = 3_000,
            selectedSuppressionPolicy = SuppressionPolicyEngine.Policy.AGGRESSIVE,
            suppressionScopeOverrides = mapOf("S-fixture" to SuppressionScope.VETO_REGION),
            cuts = listOf(
                cut("R001", 10_000, 20_000).copy(keepStartMs = 8_000, keepEndMs = 22_000),
            ),
        )

        assertEquals(
            listOf(
                FinalCutInterval(8_000, 12_000, listOf("R001")),
                FinalCutInterval(13_000, 22_000, listOf("R001")),
            ),
            EditorMath.finalIntervals(draft, suppression(12_000, 13_000)),
        )
    }

    @Test
    fun nextSuppressionUsesThePlayheadWhenThePreviousSelectionIsStale() {
        val first = suppression(10_000, 11_000).suggestions().single()
        val second = suppression(20_000, 21_000).suggestions().single()
        val suggestions = listOf(first, second)

        assertEquals(first.fragmentId(), EditorMath.nextSuppressionSuggestion(
            suggestions, 5_000, null,
        )?.fragmentId())
        assertEquals(second.fragmentId(), EditorMath.nextSuppressionSuggestion(
            suggestions, 15_000, first.fragmentId(),
        )?.fragmentId())
        assertEquals(second.fragmentId(), EditorMath.nextSuppressionSuggestion(
            suggestions, 8_000, first.fragmentId(),
        )?.fragmentId())
        assertEquals(first.fragmentId(), EditorMath.nextSuppressionSuggestion(
            suggestions, 25_000, second.fragmentId(),
        )?.fragmentId())
    }

    @Test
    fun reviewableTimeSkipsIgnoredFootageAndRejectsFullyIgnoredRanges() {
        val ignored = listOf(
            IgnoredSourceInterval("I001", 1_000, 3_000, "gap"),
            IgnoredSourceInterval("I002", 5_000, 6_000, "gap"),
        )

        assertEquals(3_000L, EditorMath.firstReviewableTime(1_500, 4_000, ignored))
        assertEquals(null, EditorMath.firstReviewableTime(1_500, 2_500, ignored))
        assertEquals(4_000L, EditorMath.firstReviewableTime(4_000, 7_000, ignored))
    }

    @Test
    fun ignoredTimestampUsesHalfOpenIntervalBoundaries() {
        val ignored = listOf(IgnoredSourceInterval("I001", 1_000, 3_000, "gap"))

        assertEquals(true, EditorMath.isTimestampIgnored(1_000, ignored))
        assertEquals(true, EditorMath.isTimestampIgnored(2_999, ignored))
        assertEquals(false, EditorMath.isTimestampIgnored(3_000, ignored))
    }

    @Test
    fun manualRangesAreUnionedBackAfterAutomaticSuppression() {
        val draft = EditorDraft(
            sourceRevision = "fixture",
            updatedAtMs = 0,
            beforePaddingMs = 0,
            afterPaddingMs = 0,
            joinGapMs = 0,
            selectedSuppressionPolicy = SuppressionPolicyEngine.Policy.AGGRESSIVE,
            suppressionScopeOverrides = mapOf("S-fixture" to SuppressionScope.VETO_REGION),
            cuts = listOf(
                cut("R001", 10_000, 20_000),
                cut("M001", 11_000, 19_000, CutOrigin.MANUAL),
            ),
        )

        assertEquals(
            listOf(FinalCutInterval(10_000, 20_000, listOf("R001", "M001"))),
            EditorMath.finalIntervals(draft, suppression(12_000, 18_000)),
        )
    }

    private fun suppression(startMs: Long, endMs: Long) = AnalysisTypes.SuppressionAnalysis(
        FeatureSchema.SUPPRESSION_MODEL_ID,
        FeatureSchema.SUPPRESSION_ARTIFACT_SHA256,
        FeatureSchema.SUPPRESSION_WEIGHTS_SHA256,
        FeatureSchema.SUPPRESSION_DECODER_VERSION,
        floatArrayOf(),
        listOf(AnalysisTypes.Interval(startMs / 1_000.0, endMs / 1_000.0, .9f)),
        listOf(AnalysisTypes.SuppressionSuggestion(
            "S-fixture",
            "S-fixture:$startMs:$endMs",
            startMs,
            endMs,
            .9f,
            listOf("all-labels-v2:0001"),
            listOf("aggressive"),
        )),
    )

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
