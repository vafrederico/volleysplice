package com.volleycut.nativeanalysis

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RallyStartMarkerTest {
    private val seed = EditorSeed(
        "content://fixture/video", "fixture.mp4", 40_000, 640, 480, 0,
        listOf(SeedRange(10_000, 20_000, .9f), SeedRange(25_000, 30_000, .9f)),
    )
    private val linked = ServeMarker(
        "serve-R001", 10_000, ServingSide.FAR, ServeMarkerOrigin.MODEL,
        modelSide = ServingSide.NEAR, ignorePreviousPoint = true, rallyId = "R001",
    )
    private val independent = ServeMarker("S001", 10_000, ServingSide.NEAR, ServeMarkerOrigin.MANUAL)
    private val other = linked.copy(id = "serve-R002", rallyId = "R002", timestampMs = 25_000)
    private val base = EditorMath.newDraft(seed).copy(scoreTracking = ScoreTracking(
        enabled = false,
        serveMarkers = listOf(linked, independent, other),
        sideSwitchMarkers = listOf(SideSwitchMarker("X001", 10_000)),
        removedModelMarkerIds = setOf("serve-R003"),
    ))

    @Test fun startMovesBothDirectionsWithPaddingAndPreservesMarkerCorrections() {
        val forward = EditorMath.setCoreStart(base, "R001", 12_000, 0)
        assertEquals(10_000L, forward.cuts.first().keepStartMs)
        assertEquals(linked.copy(timestampMs = 12_000), forward.scoreTracking.serveMarkers.single { it.id == linked.id })
        val backward = EditorMath.setCoreStart(forward, "R001", 8_000, 0)
        assertEquals(6_000L, backward.cuts.first().keepStartMs)
        assertEquals(linked.copy(timestampMs = 8_000), backward.scoreTracking.serveMarkers.single { it.id == linked.id })
        assertEquals(independent, backward.scoreTracking.serveMarkers.single { it.id == independent.id })
        assertEquals(other, backward.scoreTracking.serveMarkers.single { it.id == other.id })
        assertEquals(base.scoreTracking.sideSwitchMarkers, backward.scoreTracking.sideSwitchMarkers)
        assertEquals(base.scoreTracking.removedModelMarkerIds, backward.scoreTracking.removedModelMarkerIds)
        assertEquals(false, backward.scoreTracking.enabled)
    }

    @Test fun markerUsesClampedCoreStartAndRangeEditsAlsoMoveIt() {
        val atMinimum = EditorMath.setCoreStart(base, "R001", -1_000, 1_000)
        assertEquals(1_000L, atMinimum.scoreTracking.serveMarkers.single { it.id == linked.id }.timestampMs)
        val atMaximum = EditorMath.setCoreStart(base, "R001", 50_000, 0)
        assertEquals(19_900L, atMaximum.scoreTracking.serveMarkers.single { it.id == linked.id }.timestampMs)
        val range = EditorMath.setCoreRange(base, "R001", 13_000, 18_000, 0, 40_000)
        assertEquals(13_000L, range.scoreTracking.serveMarkers.single { it.id == linked.id }.timestampMs)
    }

    @Test fun paddingEndEditsAndSplitDoNotMoveOrDuplicateServes() {
        assertEquals(base.scoreTracking, EditorMath.applyPadding(base, 3_000, 1_000, 40_000).scoreTracking)
        assertEquals(base.scoreTracking, EditorMath.setCoreEnd(base, "R001", 18_000, 40_000).scoreTracking)
        val split = checkNotNull(EditorMath.splitCut(base, "R001", 15_000, 0, 40_000))
        val right = EditorMath.setCoreStart(split.draft, split.newCut.id, 16_000, 0)
        assertEquals(base.scoreTracking, right.scoreTracking)
    }

    @Test fun reconciliationRepairsLegacyAndReseededMarkersAndHandlesManualRallies() {
        val stale = base.copy(cuts = base.cuts.map {
            if (it.id == "R001") it.copy(coreStartMs = 12_000, keepStartMs = 10_000) else it
        })
        val repaired = EditorMath.reconcileTouchedCuts(stale, seed)
        assertEquals(12_000L, repaired.scoreTracking.serveMarkers.single { it.id == linked.id }.timestampMs)
        assertTrue("R001" in repaired.userTouchedCutIds)
        assertEquals(repaired, EditorMath.reconcileTouchedCuts(repaired, seed))
        val manual = base.copy(
            cuts = listOf(base.cuts.first().copy(id = "M001", origin = CutOrigin.MANUAL)),
            scoreTracking = base.scoreTracking.copy(serveMarkers = listOf(independent.copy(rallyId = "M001"))),
        )
        val edited = manual.copy(cuts = manual.cuts.map { it.copy(coreStartMs = 9_000, keepStartMs = 9_000) })
        assertEquals(9_000L, EditorMath.reconcileTouchedCuts(edited, seed).scoreTracking.serveMarkers.single().timestampMs)
    }

    @Test fun removedMarkerStaysRemoved() {
        val removed = base.copy(scoreTracking = ScoreReducer.removeServe(base.scoreTracking, linked.id))
        val moved = EditorMath.setCoreStart(removed, "R001", 12_000, 0)
        assertEquals(removed.scoreTracking, moved.scoreTracking)
    }
}
