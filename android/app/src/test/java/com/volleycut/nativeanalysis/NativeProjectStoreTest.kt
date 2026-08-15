package com.volleycut.nativeanalysis

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

class NativeProjectStoreTest {
    private val source = ProjectSource(
        uri = "content://recordings/match.mp4",
        name = "match.mp4",
        size = 4_200_000_000,
        lastModified = 1_700_000_000,
        mimeType = "video/mp4",
    )

    @Test
    fun projectIdIsStableAndSourceSpecific() {
        val first = NativeProjectStore.projectId(source, 1_105.817)
        val second = NativeProjectStore.projectId(source, 1_105.817)
        val changed = NativeProjectStore.projectId(source.copy(size = source.size + 1), 1_105.817)

        assertEquals(first, second)
        assertNotEquals(first, changed)
        assertEquals(true, first.startsWith("project-"))
    }

    @Test
    fun projectIdNormalizesProviderTimestampPrecision() {
        val rounded = source.copy(lastModified = 1_786_690_482_000)
        val precise = source.copy(lastModified = 1_786_690_482_790)

        assertEquals(
            NativeProjectStore.projectId(rounded, 1_105.817),
            NativeProjectStore.projectId(precise, 1_105.817),
        )
    }

    @Test
    fun readyProjectRoundTripsFinalizedInference() {
        val project = NativeProject(
            id = NativeProjectStore.projectId(source, 12.5),
            source = source,
            media = AnalysisTypes.MediaInfo(12.5, 1920, 1080, 0, "video/avc", "audio/mp4a-latm"),
            roi = AnalysisTypes.Roi(.03, .12, .94, .86, "Indoor camera default"),
            status = ProjectStatus.READY,
            ranges = listOf(SeedRange(1_000, 4_500, .82f)),
            createdAtMs = 100,
            updatedAtMs = 200,
        )

        val restored = NativeProjectStore.decode(NativeProjectStore.encode(project))

        assertNotNull(restored)
        assertEquals(project, restored)
        assertEquals(project.ranges, restored?.editorSeed()?.ranges)
    }
}
