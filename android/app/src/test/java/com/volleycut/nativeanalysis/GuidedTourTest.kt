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
            GuidedTourStep.EDITOR_SETTINGS,
            nextGuidedTourStep(GuidedTourStep.SETUP_CREATE),
        )
    }

    @Test
    fun editorFollowsTheControlsTopToBottom() {
        assertEquals(
            GuidedTourStep.EDITOR_EXPORT,
            nextGuidedTourStep(GuidedTourStep.EDITOR_SETTINGS),
        )
        assertEquals(
            GuidedTourStep.EDITOR_SCORE_TOGGLE,
            nextGuidedTourStep(GuidedTourStep.EDITOR_EXPORT),
        )
        assertEquals(
            GuidedTourStep.EDITOR_SCORE_PANEL,
            nextGuidedTourStep(GuidedTourStep.EDITOR_SCORE_TOGGLE),
        )
        assertEquals(
            GuidedTourStep.EDITOR_VIDEO,
            nextGuidedTourStep(GuidedTourStep.EDITOR_SCORE_PANEL),
        )
        assertEquals(
            GuidedTourStep.EDITOR_REVIEW_QUEUES,
            nextGuidedTourStep(GuidedTourStep.EDITOR_VIDEO),
        )
        assertEquals(
            GuidedTourStep.EDITOR_OVERVIEW,
            nextGuidedTourStep(GuidedTourStep.EDITOR_REVIEW_QUEUES),
        )
        assertEquals(
            GuidedTourStep.EDITOR_FOCUS,
            nextGuidedTourStep(GuidedTourStep.EDITOR_OVERVIEW),
        )
        assertEquals(
            GuidedTourStep.EDITOR_MARKING,
            nextGuidedTourStep(GuidedTourStep.EDITOR_FOCUS),
        )
        assertNull(nextGuidedTourStep(GuidedTourStep.EDITOR_MARKING))
    }

    @Test
    fun disabledScoreTrackingStillExplainsToggleAndSkipsPanel() {
        assertEquals(
            GuidedTourStep.EDITOR_SCORE_TOGGLE,
            nextGuidedTourStep(GuidedTourStep.EDITOR_EXPORT, scoreTrackingEnabled = false),
        )
        assertEquals(
            GuidedTourStep.EDITOR_VIDEO,
            nextGuidedTourStep(GuidedTourStep.EDITOR_SCORE_TOGGLE, scoreTrackingEnabled = false),
        )
    }
}
