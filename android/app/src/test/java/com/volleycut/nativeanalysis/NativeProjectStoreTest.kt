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
    fun projectIdIncludesNonDefaultGameWindowButPreservesFullVideoIdentity() {
        val legacyFull = NativeProjectStore.projectId(source, 1_105.817)
        val explicitFull = NativeProjectStore.projectId(
            source,
            1_105.817,
            AnalysisTypes.AnalysisWindow(0.0, 1_105.817),
        )
        val markedWindow = NativeProjectStore.projectId(
            source,
            1_105.817,
            AnalysisTypes.AnalysisWindow(42.5, 1_000.0),
        )

        assertEquals(legacyFull, explicitFull)
        assertNotEquals(legacyFull, markedWindow)
    }

    @Test
    fun readyProjectRoundTripsFinalizedInference() {
        val project = NativeProject(
            id = NativeProjectStore.projectId(source, 12.5),
            source = source,
            media = AnalysisTypes.MediaInfo(12.5, 1920, 1080, 0, "video/avc", "audio/mp4a-latm"),
            roi = AnalysisTypes.Roi(.03, .12, .94, .86, "Indoor camera default"),
            status = ProjectStatus.READY,
            ranges = listOf(SeedRange(
                1_000,
                4_500,
                .82f,
                ProductionEnsemble.BOTH_MODELS,
            )),
            createdAtMs = 100,
            updatedAtMs = 200,
        )

        val restored = NativeProjectStore.decode(NativeProjectStore.encode(project))

        assertNotNull(restored)
        assertEquals(project, restored)
        assertEquals(project.ranges, restored?.editorSeed()?.ranges)
    }

    @Test
    fun markedGameWindowRoundTripsAndSeedsEditorBounds() {
        val media = AnalysisTypes.MediaInfo(120.0, 1920, 1080, 0, "video/avc", "audio/mp4a-latm")
        val project = NativeProject(
            id = NativeProjectStore.projectId(
                source,
                media.durationSeconds(),
                AnalysisTypes.AnalysisWindow(10.0, 100.0),
            ),
            source = source,
            media = media,
            analysisWindow = AnalysisTypes.AnalysisWindow(10.0, 100.0),
            roi = AnalysisTypes.Roi(.03, .12, .94, .86, "Indoor camera default"),
            status = ProjectStatus.READY,
            ranges = listOf(SeedRange(20_000, 30_000, .9f, ProductionEnsemble.BOTH_MODELS)),
            createdAtMs = 100,
            updatedAtMs = 200,
        )

        val restored = requireNotNull(NativeProjectStore.decode(NativeProjectStore.encode(project)))
        assertEquals(project.analysisWindow, restored.analysisWindow)
        assertEquals(10_000L, restored.editorSeed()?.gameStartMs)
        assertEquals(100_000L, restored.editorSeed()?.gameEndMs)
    }

    @Test
    fun legacyProjectWithoutWindowDefaultsToFullVideo() {
        val project = NativeProject(
            id = NativeProjectStore.projectId(source, 12.5),
            source = source,
            media = AnalysisTypes.MediaInfo(12.5, 1920, 1080, 0, "video/avc", "audio/mp4a-latm"),
            roi = AnalysisTypes.Roi(.03, .12, .94, .86, "Indoor camera default"),
            status = ProjectStatus.QUEUED,
            createdAtMs = 100,
            updatedAtMs = 200,
        )
        val legacyJson = NativeProjectStore.encode(project).apply { remove("analysisWindow") }

        val restored = requireNotNull(NativeProjectStore.decode(legacyJson))
        assertEquals(AnalysisTypes.AnalysisWindow(0.0, 12.5), restored.analysisWindow)
        assertEquals(project.id, restored.id)
    }

    @Test
    fun staleSingleModelInferenceIsQueuedWithoutDiscardingFeatureIdentity() {
        val project = NativeProject(
            id = NativeProjectStore.projectId(source, 12.5),
            source = source,
            media = AnalysisTypes.MediaInfo(12.5, 1920, 1080, 0, "video/avc", "audio/mp4a-latm"),
            roi = AnalysisTypes.Roi(.03, .12, .94, .86, "Indoor camera default"),
            status = ProjectStatus.READY,
            ranges = listOf(SeedRange(1_000, 4_500, .82f)),
            modelId = FeatureSchema.PREVIOUS_PRODUCTION_MODEL_ID,
            createdAtMs = 100,
            updatedAtMs = 200,
        )

        val normalized = NativeProjectStore.normalizeStored(project)

        assertEquals(ProjectStatus.QUEUED, normalized.status)
        assertEquals(FeatureSchema.MODEL_ID, normalized.modelId)
        assertEquals(emptyList<SeedRange>(), normalized.ranges)
        assertEquals(project.id, normalized.id)
        assertEquals(project.source, normalized.source)
        assertEquals(project.roi, normalized.roi)
    }

    @Test
    fun currentEnsembleWithoutAgreementProvenanceIsQueued() {
        val project = NativeProject(
            id = NativeProjectStore.projectId(source, 12.5),
            source = source,
            media = AnalysisTypes.MediaInfo(12.5, 1920, 1080, 0, "video/avc", "audio/mp4a-latm"),
            roi = AnalysisTypes.Roi(.03, .12, .94, .86, "Indoor camera default"),
            status = ProjectStatus.READY,
            ranges = listOf(SeedRange(1_000, 4_500, .49f)),
            createdAtMs = 100,
            updatedAtMs = 200,
        )

        assertEquals(ProjectStatus.QUEUED, NativeProjectStore.normalizeStored(project).status)
    }

    @Test
    fun replacementMatchingStableMediaIdentityIsAcceptedAfterMoveOrRename() {
        val project = readyProject()
        val replacement = source.copy(
            uri = "content://new-location/renamed.mp4",
            name = "renamed.mp4",
            lastModified = source.lastModified + 500_000,
        )

        assertEquals(
            null,
            NativeProjectStore.replacementMismatch(project, replacement, project.media),
        )
    }

    @Test
    fun replacementWithDifferentContentIdentityIsRejected() {
        val project = readyProject()

        assertEquals(
            "That file has a different byte size from the original recording.",
            NativeProjectStore.replacementMismatch(
                project,
                source.copy(size = source.size + 1),
                project.media,
            ),
        )
        assertEquals(
            "That file has a different duration from the original recording.",
            NativeProjectStore.replacementMismatch(
                project,
                source,
                AnalysisTypes.MediaInfo(13.0, 1920, 1080, 0, "video/avc", "audio/mp4a-latm"),
            ),
        )
    }

    @Test
    fun retainedCacheSourceRoundTripsAfterRelink() {
        val project = readyProject().copy(
            source = source.copy(uri = "content://new-location/match.mp4"),
            featureCacheSource = source,
        )

        val restored = requireNotNull(NativeProjectStore.decode(NativeProjectStore.encode(project)))

        assertEquals(project.source, restored.source)
        assertEquals(source, restored.featureCacheSource)
    }

    private fun readyProject() = NativeProject(
        id = NativeProjectStore.projectId(source, 12.5),
        source = source,
        media = AnalysisTypes.MediaInfo(12.5, 1920, 1080, 0, "video/avc", "audio/mp4a-latm"),
        roi = AnalysisTypes.Roi(.03, .12, .94, .86, "Indoor camera default"),
        status = ProjectStatus.READY,
        ranges = listOf(SeedRange(1_000, 4_500, .82f, ProductionEnsemble.BOTH_MODELS)),
        createdAtMs = 100,
        updatedAtMs = 200,
    )
}
