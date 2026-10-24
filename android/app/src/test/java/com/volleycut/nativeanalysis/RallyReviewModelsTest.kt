package com.volleycut.nativeanalysis

import org.junit.Assert.*
import org.junit.Test

class RallyReviewModelsTest {
    private fun rally(id: String, start: Long, end: Long) = EditableRallyGroup(listOf(
        EditableCut(id, start, end, (start - 2_000).coerceAtLeast(0), end + 2_000, .5f, true, CutOrigin.INFERRED),
    ))

    @Test
    fun combinesClipCleanupAndServeDecisionsForTheSameRally() {
        val rally = rally("R1", 5_000, 10_000)
        val queue = rallyReviewQueue(listOf(rally), setOf("R1"),
            listOf(RallyReviewCleanup("cleanup", 6_000, 8_000)),
            listOf(ServeMarker("serve", 5_000, ServingSide.REVIEW, ServeMarkerOrigin.MODEL, rallyId = "R1")), emptyList())
        assertEquals(1, queue.size)
        assertTrue(queue.single().clipReview)
        assertEquals(setOf("cleanup"), queue.single().cleanupIds)
        assertEquals(setOf("serve"), queue.single().serveIds)
        assertEquals("R1", queue.single().rallyId)
    }

    @Test
    fun keepsOrphanEvidenceAndOrdersItChronologically() {
        val queue = rallyReviewQueue(listOf(rally("R1", 10_000, 15_000)), setOf("R1"),
            listOf(RallyReviewCleanup("cleanup", 1_000, 2_000)),
            listOf(ServeMarker("serve", 20_000, ServingSide.REVIEW, ServeMarkerOrigin.MANUAL)), emptyList())
        assertEquals(listOf("cleanup-cleanup", "rally-R1", "serve-serve"), queue.map { it.key })
        assertNull(queue.first().rallyId)
        assertNull(queue.last().rallyId)
    }

    @Test
    fun ignoredEvidenceAndResolvedServesDoNotCreateReviewStops() {
        val queue = rallyReviewQueue(listOf(rally("R1", 5_000, 7_000)), setOf("R1"),
            listOf(RallyReviewCleanup("cleanup", 4_000, 6_000)), listOf(
                ServeMarker("review", 5_000, ServingSide.REVIEW, ServeMarkerOrigin.MANUAL),
                ServeMarker("resolved", 15_000, ServingSide.NEAR, ServeMarkerOrigin.MANUAL),
            ), listOf(IgnoredSourceInterval("ignored", 3_000, 9_000, "non-game")))
        assertTrue(queue.isEmpty())
    }

    @Test
    fun serveIdentityWinsOverOverlappingPadding() {
        val rallies = listOf(rally("R1", 5_000, 8_000), rally("R2", 9_000, 12_000))
        val marker = ServeMarker("serve", 8_000, ServingSide.REVIEW, ServeMarkerOrigin.MODEL, rallyId = "R2")
        assertEquals("R2", rallyReviewQueue(rallies, emptySet(), emptyList(), listOf(marker), emptyList()).single().rallyId)
        assertEquals(listOf(marker), rallyServeMarkers(rallies[1], listOf(marker)))
        assertTrue(rallyServeMarkers(rallies[0], listOf(marker)).isEmpty())
    }

    @Test
    fun gapNavigationNeverSelectsAPrecedingRally() {
        val rallies = listOf(rally("R1", 5_000, 8_000), rally("R2", 15_000, 18_000))
        assertEquals("R1", rallyAtTime(rallies, 7_000)?.id)
        assertNull(rallyAtTime(rallies, 12_000))
        assertNull(rallyAtTime(rallies, 20_000))
    }

    @Test
    fun retainsMultipleServingDecisionsAtOneStop() {
        val rally = rally("R1", 5_000, 12_000)
        val markers = listOf(5_000L, 10_000L).mapIndexed { index, time ->
            ServeMarker("serve-$index", time, ServingSide.REVIEW, ServeMarkerOrigin.MANUAL)
        }
        assertEquals(setOf("serve-0", "serve-1"),
            rallyReviewQueue(listOf(rally), emptySet(), emptyList(), markers, emptyList()).single().serveIds)
    }

    @Test
    fun directBoundsKeepHalfASecondAndRespectTheSourceWindow() {
        assertEquals(1_000L, rallyBoundaryValue("start", -1_000, 2_000, 4_000, 1_000, 8_000))
        assertEquals(3_500L, rallyBoundaryValue("start", 6_000, 2_000, 4_000, 1_000, 8_000))
        assertEquals(2_500L, rallyBoundaryValue("end", 1_000, 2_000, 4_000, 1_000, 8_000))
        assertEquals(8_000L, rallyBoundaryValue("end", 9_000, 2_000, 4_000, 1_000, 8_000))
    }

    @Test
    fun missingCandidateIsLimitedToTheRealGapAndIgnoredFootage() {
        val occupied = listOf(2_000L to 5_000L, 12_000L to 15_000L, 18_000L to 20_000L)
        assertNull(missingRallyRange(3_000, 0, 20_000, occupied))
        assertEquals(MissingRallyRange(6_000, 12_000), missingRallyRange(9_000, 0, 20_000, occupied))
        assertEquals(MissingRallyRange(15_000, 18_000), missingRallyRange(17_000, 0, 20_000, occupied))
        assertNull(missingRallyRange(5_200, 0, 20_000, occupied + (5_400L to 12_000L)))
        assertEquals(MissingRallyRange(0, 2_000), missingRallyRange(0, 0, 20_000, occupied))
    }

    @Test
    fun gapBrowsingUsesDistanceToIntervalsWithoutSelectingOne() {
        val rallies = listOf(rally("R1", 5_000, 8_000), rally("R2", 15_000, 18_000))
        assertEquals(1, nearestRallyIndex(rallies, 12_000))
        assertNull(rallyAtTime(rallies, 12_000))
        assertEquals(-1, nearestRallyIndex(emptyList(), 13_000))
    }

    @Test
    fun focusedContextSkipsRemovedAndSuppressedRalliesOnBothSides() {
        val previous = rally("R1", 5_000, 8_000)
        val removed = rally("R2", 15_000, 18_000).let { r -> r.copy(cuts = r.cuts.map { it.copy(included = false) }) }
        val selected = rally("R3", 25_000, 30_000)
        val suppressed = rally("R4", 35_000, 40_000)
        val next = rally("R5", 50_000, 56_000)
        val context = rallyTimelineContext(listOf(next, removed, selected, previous, suppressed), selected,
            setOf("R1", "R2", "R3", "R5"), 0, 90_000)
        assertEquals("R1", context.previous?.id)
        assertEquals("R5", context.next?.id)
        assertEquals(DetailWindow(1_000, 60_000), context.window)
    }

    @Test
    fun focusedContextClipsToTheGameAndIgnoresRalliesOutsideIt() {
        val selected = rally("R2", 25_000, 30_000)
        val context = rallyTimelineContext(listOf(rally("R1", 5_000, 8_000), selected, rally("R3", 50_000, 56_000)),
            selected, setOf("R1", "R2", "R3"), 11_000, 55_000)
        assertNull(context.previous)
        assertEquals("R3", context.next?.id)
        assertEquals(DetailWindow(17_000, 55_000), context.window)
    }

    @Test
    fun longBreaksKeepTheirActualDurationAndRemovedNeighboursAreSkipped() {
        val previous = rally("R1", 5_000, 8_000)
        val removed = rally("R2", 1_790_000, 1_795_000).let { r ->
            r.copy(cuts = r.cuts.map { it.copy(included = false) })
        }
        val selected = rally("R3", 1_800_000, 1_805_000)
        val next = rally("R4", 3_600_000, 3_605_000)
        val context = rallyTimelineContext(listOf(previous, removed, selected, next), selected,
            setOf("R1", "R2", "R3", "R4"), 0, 3_700_000)
        assertEquals(previous, context.previous)
        assertEquals(next, context.next)
        assertEquals(DetailWindow(1_000, 3_609_000), context.window)
        val scale = RallyTimelineScale(context.window)
        val duration = context.window.endMs - context.window.startMs
        assertEquals(5_000f / duration, scale.fraction(selected.coreEndMs) - scale.fraction(selected.coreStartMs), .000001f)
        assertEquals(1_795_000f / duration, scale.fraction(next.coreStartMs) - scale.fraction(selected.coreEndMs), .000001f)
    }

    @Test
    fun shortRalliesKeepTheirSurroundingSourceContext() {
        val selected = rally("R2", 25_000, 25_500)
        val context = rallyTimelineContext(listOf(rally("R1", 1_000, 4_000), selected,
            rally("R3", 60_000, 65_000)), selected, setOf("R1", "R2", "R3"), 0, 70_000)
        assertEquals(DetailWindow(0, 69_000), context.window)
    }

    @Test
    fun shortCorePaddingIsShownInFullWithoutZoomingAwayItsContext() {
        listOf(500L to 10_000L, 100L to 2_000L).forEach { (duration, padding) ->
            val selected = rally("R1", 25_000, 25_000 + duration).let { r ->
                r.copy(cuts = r.cuts.map { it.copy(keepStartMs = 25_000 - padding, keepEndMs = 25_000 + duration + padding) })
            }
            val context = rallyTimelineContext(listOf(selected), selected, setOf("R1"), 0, 60_000)
            assertTrue(context.window.startMs <= selected.keepStartMs)
            assertTrue(context.window.endMs >= selected.keepEndMs)
            assertTrue(context.window.startMs <= selected.coreStartMs - 8_000)
            assertTrue(context.window.endMs >= selected.coreEndMs + 8_000)
        }
    }

    @Test
    fun equalDistancesAlwaysMeanEqualTimeInRalliesAndGaps() {
        listOf(100L, 500L, 5_000L).forEach { duration ->
            val selected = rally("R2", 25_000, 25_000 + duration)
            val rallies = listOf(rally("R1", 1_000, 4_000), selected, rally("R3", 60_000, 65_000))
            val scale = RallyTimelineScale(rallyTimelineContext(rallies, selected, setOf("R1", "R2", "R3"), 0, 70_000).window)
            (0..100).forEach { step ->
                val fraction = step / 100f
                val expectedTime = scale.startMs + step * (scale.endMs - scale.startMs) / 100
                assertTrue("Uniform spacing at $fraction", kotlin.math.abs(expectedTime - scale.time(fraction)) <= 1)
                val time = scale.time(fraction)
                assertTrue("Source-time round trip at $time", kotlin.math.abs(scale.time(scale.fraction(time)) - time) <= 2)
            }
            val second = 1_000f / (scale.endMs - scale.startMs)
            listOf(2_000L, 12_000L, 25_000L, 60_000L).forEach { time ->
                assertEquals("One second has the same width in every part of the timeline", second,
                    scale.fraction(time + 1_000) - scale.fraction(time), .000001f)
            }
            assertEquals(scale.startMs, scale.time(-1f))
            assertEquals(scale.endMs, scale.time(2f))
        }
    }

    private fun mergeFixture(): EditorDraft = EditorDraft(sourceRevision = "synthetic", updatedAtMs = 0,
        cuts = rally("A", 10_000, 15_000).cuts + rally("B", 22_000, 25_000).cuts,
        reviewedCutIds = setOf("A", "B"), joinGapMs = 3_000)

    @Test
    fun endingTrimResolvesTheNewMergedGroupBeforeClamping() {
        val initial = mergeFixture()
        assertEquals(2, rallyGroupsForDraft(initial).size)
        val updated = EditorMath.setCoreEnd(initial, "A", 16_000, 60_000)
        val selected = rallyGroupsForDraft(updated).single { "A" in it.cutIds }
        assertEquals(setOf("A", "B"), selected.cutIds)
        assertEquals(23_000L, 23_000L.coerceIn(selected.keepStartMs, selected.keepEndMs))
    }

    @Test
    fun startingTrimResolvesTheNewMergedGroupBeforeClamping() {
        val updated = EditorMath.setCoreStart(mergeFixture(), "B", 21_000, 0)
        val selected = rallyGroupsForDraft(updated).single { "B" in it.cutIds }
        assertEquals(setOf("A", "B"), selected.cutIds)
        assertEquals(12_000L, 12_000L.coerceIn(selected.keepStartMs, selected.keepEndMs))
    }

    @Test
    fun serveAndPendingReviewBoundariesStillSeparateTrimGroups() {
        val joined = EditorMath.setCoreEnd(mergeFixture(), "A", 16_000, 60_000)
        assertEquals(2, rallyGroupsForDraft(joined.copy(reviewedCutIds = emptySet())).size)
        val separated = joined.copy(scoreTracking = ScoreTracking(enabled = true, serveMarkers = listOf(
            ServeMarker("serve", 22_000, ServingSide.NEAR, ServeMarkerOrigin.MANUAL, rallyId = "B"))))
        val selected = rallyGroupsForDraft(separated).single { "A" in it.cutIds }
        assertEquals(setOf("A"), selected.cutIds)
        assertEquals(18_000L, 23_000L.coerceIn(selected.keepStartMs, selected.keepEndMs))
    }

    @Test
    fun aRemovedSelectionRemainsEditableWithoutRetainedNeighbours() {
        val selected = rally("R1", 25_000, 30_000).let { r -> r.copy(cuts = r.cuts.map { it.copy(included = false) }) }
        val context = rallyTimelineContext(listOf(selected), selected, emptySet(), 20_000, 35_000)
        assertNull(context.previous)
        assertNull(context.next)
        assertEquals(DetailWindow(20_000, 35_000), context.window)
    }
}
