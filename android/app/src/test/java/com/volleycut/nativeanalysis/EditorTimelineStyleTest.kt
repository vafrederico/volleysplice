package com.volleycut.nativeanalysis

import org.junit.Assert.assertNotEquals
import org.junit.Test

class EditorTimelineStyleTest {
    @Test
    fun retainedPaddingAndJoinedGapsRemainVisibleAgainstTheTimelineTrack() {
        assertNotEquals(TimelineTrackColor, TimelineKeptPaddingColor)
        assertNotEquals(TimelineTrackColor, TimelineJoinedGapColor)
    }
}
