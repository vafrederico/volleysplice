package com.volleycut.nativeanalysis

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class EditorProjectStoreTest {
    private val seed = EditorSeed(
        sourceUri = "content://recordings/match.mp4",
        displayName = "match.mp4",
        durationMs = 12_500,
        width = 1920,
        height = 1080,
        rotation = 0,
        ranges = listOf(SeedRange(1_000, 4_500, .82f, ProductionEnsemble.BOTH_MODELS)),
    )

    @Test
    fun recoveryGeometryRoundTripsWithoutChangingTheDraftIdentity() {
        for (roi in listOf(
            AnalysisTypes.Roi(0.0, 0.0, 1.0, 1.0, "Full frame"),
            AnalysisTypes.Roi(.03, .12, .94, .86, "Saved crop"),
        )) {
            val withGeometry = seed.copy(analysisRoi = roi)
            val restored = requireNotNull(EditorProjectStore.decode(EditorProjectStore.encode(withGeometry)))

            assertEquals(roi, restored.analysisRoi)
            assertEquals(seed.ranges, restored.ranges)
            assertEquals(seed.sourceRevision, restored.sourceRevision)
        }
    }

    @Test
    fun legacyAndUnknownRecoveryRecordsDoNotAcquireTodaysDefaultRoi() {
        val legacy = EditorProjectStore.encode(seed).apply {
            put("version", 5)
            remove("analysisRoi")
        }
        val restored = requireNotNull(EditorProjectStore.decode(legacy))
        val savedAgain = requireNotNull(EditorProjectStore.decode(EditorProjectStore.encode(restored)))

        assertNull(restored.analysisRoi)
        assertNull(savedAgain.analysisRoi)
        assertEquals(seed.ranges, savedAgain.ranges)
        assertEquals(seed.sourceRevision, savedAgain.sourceRevision)
    }

    @Test
    fun invalidRecoveryGeometryIsUnknownAndKeepsSavedRallies() {
        val invalid = EditorProjectStore.encode(seed.copy(
            analysisRoi = AnalysisTypes.Roi(0.0, 0.0, 1.5, 1.0, "Invalid crop"),
        ))
        val restored = requireNotNull(EditorProjectStore.decode(invalid))

        assertNull(restored.analysisRoi)
        assertEquals(seed.ranges, restored.ranges)

        invalid.getJSONObject("analysisRoi").remove("width")
        val incomplete = requireNotNull(EditorProjectStore.decode(invalid))
        assertNull(incomplete.analysisRoi)
        assertEquals(seed.ranges, incomplete.ranges)
    }
}
