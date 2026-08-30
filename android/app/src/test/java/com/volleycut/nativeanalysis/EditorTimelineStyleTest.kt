package com.volleycut.nativeanalysis

import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class EditorTimelineStyleTest {
    @Test
    fun retainedPaddingAndJoinedGapsRemainVisibleAgainstTheTimelineTrack() {
        assertNotEquals(TimelineTrackColor, TimelineKeptPaddingColor)
        assertNotEquals(TimelineTrackColor, TimelineJoinedGapColor)
    }

    @Test
    fun lowConfidenceTimelineStyleClearsAfterTheCutIsReviewed() {
        assertTrue(timelineNeedsReview(true, "rally-1", emptySet()))
        assertFalse(timelineNeedsReview(true, "rally-1", setOf("rally-1")))
        assertFalse(timelineNeedsReview(false, "rally-1", emptySet()))
    }
}
