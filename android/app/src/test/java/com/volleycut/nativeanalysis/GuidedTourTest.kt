package com.volleycut.nativeanalysis

import org.junit.Assert.*
import org.junit.Test

class GuidedTourTest {
    @Test
    fun setupHandsOffToRallyReview() {
        assertEquals(GuidedTourStep.SETUP_WINDOW, nextGuidedTourStep(GuidedTourStep.SETUP_SOURCE))
        assertEquals(GuidedTourStep.SETUP_CREATE, nextGuidedTourStep(GuidedTourStep.SETUP_WINDOW))
        assertEquals(GuidedTourStep.RALLIES_REVIEW, nextGuidedTourStep(GuidedTourStep.SETUP_CREATE))
    }

    @Test
    fun rallyTourCoversReviewBrowsingCreationAndBothOtherTabs() {
        val visited = mutableListOf<GuidedTourStep>()
        var step: GuidedTourStep? = GuidedTourStep.RALLIES_REVIEW
        repeat(20) {
            step?.let { current -> visited += current; step = nextGuidedTourStep(current) }
        }
        assertEquals(12, visited.size)
        assertEquals(visited.size, visited.distinct().size)
        assertTrue(visited.containsAll(listOf(GuidedTourStep.RALLIES_TAG, GuidedTourStep.RALLIES_TIMELINE,
            GuidedTourStep.RALLIES_NAVIGATION, GuidedTourStep.RALLIES_ALL, GuidedTourStep.RALLIES_LIST,
            GuidedTourStep.RALLIES_SOURCE, GuidedTourStep.RALLIES_MISSING, GuidedTourStep.RALLIES_ADD)))
        assertEquals(GuidedTourStep.EDITOR_OVERVIEW, visited[9])
        assertEquals(GuidedTourStep.EDITOR_EXPORT, visited[10])
        assertEquals(GuidedTourStep.EDITOR_SETTINGS, visited.last())
        assertNull(step)
    }

    @Test
    fun emptyProjectStillTeachesBrowsingAndAddingWithoutAbsentRallyControls() {
        assertEquals(GuidedTourStep.RALLIES_ALL,
            nextGuidedTourStep(GuidedTourStep.RALLIES_REVIEW, hasRallies = false))
        assertEquals(GuidedTourStep.RALLIES_ADD,
            nextGuidedTourStep(GuidedTourStep.RALLIES_MISSING, hasRallies = false))
    }

    @Test
    fun scoreDisabledStillTeachesKeepAndRemove() {
        assertEquals(GuidedTourStep.RALLIES_TAG,
            nextGuidedTourStep(GuidedTourStep.RALLIES_REVIEW, scoreTrackingEnabled = false))
    }

    @Test
    fun eachTutorialSurfaceMatchesTheRealWorkspace() {
        assertEquals(GuidedTourWorkspace.RALLY, GuidedTourStep.RALLIES_TAG.workspace)
        assertEquals(GuidedTourWorkspace.RALLY_LIST, GuidedTourStep.RALLIES_SOURCE.workspace)
        assertEquals(GuidedTourWorkspace.ADD_RALLY, GuidedTourStep.RALLIES_ADD.workspace)
        assertEquals(GuidedTourWorkspace.EDITOR, GuidedTourStep.EDITOR_OVERVIEW.workspace)
        assertEquals(GuidedTourWorkspace.EXPORT, GuidedTourStep.EDITOR_EXPORT.workspace)
        assertEquals(GuidedTourWorkspace.RALLY, GuidedTourStep.EDITOR_SETTINGS.workspace)
    }
}
