package com.volleycut.nativeanalysis

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ProcessingTimeoutTrackerTest {
    @Test
    fun timeoutHistoryRoundTripsAndPreservesOptionalFields() {
        val events = listOf(
            ProcessingTimeoutEvent(
                sequence = 3,
                operation = MediaProcessingOperation.INFERENCE,
                projectId = "project-3",
                sourceName = "game.mp4",
                occurredAtMs = 1_700_000_000_000,
                detail = "Inference stopped",
            ),
            ProcessingTimeoutEvent(
                sequence = 4,
                operation = MediaProcessingOperation.EXPORT,
                projectId = null,
                sourceName = null,
                occurredAtMs = 1_700_000_001_000,
                detail = "Export stopped",
            ),
        )

        assertEquals(events, ProcessingTimeoutTracker.decodeHistory(
            ProcessingTimeoutTracker.encodeHistory(events),
        ))
    }

    @Test
    fun malformedHistoryIsIgnored() {
        assertTrue(ProcessingTimeoutTracker.decodeHistory("not-json").isEmpty())
        assertTrue(ProcessingTimeoutTracker.decodeHistory("[]").isEmpty())
    }
}
