package com.volleycut.nativeanalysis

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class GuidedTourTest {
    @Test
    fun setupHandsOffToEditorInOrder() {
        assertEquals(
            GuidedTourStep.SETUP_WINDOW,
            nextGuidedTourStep(GuidedTourStep.SETUP_SOURCE),
        )
        assertEquals(
            GuidedTourStep.SETUP_CREATE,
            nextGuidedTourStep(GuidedTourStep.SETUP_WINDOW),
        )
        assertEquals(
            GuidedTourStep.EDITOR_HEADER,
            nextGuidedTourStep(GuidedTourStep.SETUP_CREATE),
        )
    }

    @Test
    fun editorFinishesAfterExport() {
        assertEquals(
            GuidedTourStep.EDITOR_EXPORT,
            nextGuidedTourStep(GuidedTourStep.EDITOR_CUTS),
        )
        assertNull(nextGuidedTourStep(GuidedTourStep.EDITOR_EXPORT))
    }

    @Test
    fun disabledScoreTrackingSkipsAllScoreTourSteps() {
        assertEquals(
            GuidedTourStep.EDITOR_SUPPRESSION,
            nextGuidedTourStep(GuidedTourStep.EDITOR_OUTPUT, scoreTrackingEnabled = false),
        )
        assertEquals(
            GuidedTourStep.EDITOR_SCORE_TOGGLE,
            nextGuidedTourStep(GuidedTourStep.EDITOR_OUTPUT, scoreTrackingEnabled = true),
        )
    }
}
