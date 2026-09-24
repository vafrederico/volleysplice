package com.volleycut.nativeanalysis

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
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
    fun newAnalysisUsesFullFrameForArbitraryAndFormerCameraProfileNames() {
        val media = AnalysisTypes.MediaInfo(12.5, 1920, 1080, 0, "video/avc", "audio/mp4a-latm")
        for (filename in listOf(
            "match.mp4", "recording-044.mp4", "beach-source-01.mp4",
            "beach-source-02.mp4", "indoor-source-07.mp4", "",
        )) {
            val inferred = AnalysisEngine.inferRoi(filename)
            val project = NativeProjectStore.newQueued(source.copy(name = filename), media)
            assertEquals(AnalysisTypes.Roi(0.0, 0.0, 1.0, 1.0, "Full frame"), inferred)
            assertEquals(inferred, project.roi)
        }
    }

    @Test
    fun newFullFrameAnalysisCannotReuseOrOverwriteAnOlderCroppedProject() {
        val saved = readyProject()
        val fullFrame = NativeProjectStore.newQueued(saved.source, saved.media)
        val explicitCrop = NativeProjectStore.newQueued(saved.source, saved.media, saved.roi)

        assertNotEquals(saved.id, fullFrame.id)
        assertNotEquals(explicitCrop.id, fullFrame.id)
        assertFalse(NativeProjectStore.matchesAnalysis(saved, fullFrame))
        assertTrue(NativeProjectStore.matchesAnalysis(saved, explicitCrop))
        assertTrue(NativeProjectStore.matchesAnalysis(
            fullFrame.copy(roi = AnalysisTypes.Roi(0.0, 0.0, 1.0, 1.0, "Imported full frame")),
            fullFrame,
        ))

        val restored = requireNotNull(NativeProjectStore.decode(NativeProjectStore.encode(saved)))
        assertEquals(saved.id, restored.id)
        assertEquals(saved.roi, restored.roi)
        assertEquals(saved.ranges, restored.ranges)
        assertEquals(saved.roi, NativeProjectStore.normalizeStored(restored).roi)
    }

    @Test
    fun analysisMatchingStillSeparatesWindowsWithTheSameFullFrameRoi() {
        val media = AnalysisTypes.MediaInfo(120.0, 1920, 1080, 0, "video/avc", "audio/mp4a-latm")
        val whole = NativeProjectStore.newQueued(source, media)
        val window = NativeProjectStore.newQueued(
            source, media, requestedWindow = AnalysisTypes.AnalysisWindow(10.0, 100.0),
        )

        assertNotEquals(whole.id, window.id)
        assertFalse(NativeProjectStore.matchesAnalysis(whole, window))
    }

    @Test
    fun legacyRecoveryRetainsReviewedOutputWithoutClaimingCurrentAnalysisGeometry() {
        val saved = readyProject()
        val seed = requireNotNull(saved.editorSeed()).copy(
            analysisRoi = null,
            ranges = saved.ranges.map { it.copy(agreement = null) },
        )
        val recovered = NativeProjectStore.fromSeed(seed, saved.source, saved.media)
        val fresh = NativeProjectStore.newQueued(saved.source, saved.media)
        val restored = requireNotNull(NativeProjectStore.decode(NativeProjectStore.encode(recovered)))
        val normalized = NativeProjectStore.normalizeStored(restored)

        assertFalse(normalized.analysisRoiKnown)
        assertEquals(ProjectStatus.READY, normalized.status)
        assertEquals(seed.ranges, normalized.ranges)
        assertEquals(seed.sourceRevision, normalized.editorSeed()?.sourceRevision)
        assertEquals(null, normalized.editorSeed()?.analysisRoi)
        assertNotEquals(fresh.id, normalized.id)
        assertFalse(NativeProjectStore.matchesAnalysis(normalized, fresh))
        assertFalse(NativeProjectStore.matchesAnalysis(fresh, normalized))
    }

    @Test
    fun newRecoverySeedsPreserveCroppedAndFullFrameGeometry() {
        val saved = readyProject()
        for (roi in listOf(saved.roi, AnalysisEngine.inferRoi(saved.source.name))) {
            val seed = requireNotNull(saved.copy(roi = roi).editorSeed())
            val recovered = NativeProjectStore.fromSeed(seed, saved.source, saved.media)
            val candidate = NativeProjectStore.newQueued(saved.source, saved.media, roi)

            assertTrue(recovered.analysisRoiKnown)
            assertEquals(roi, recovered.roi)
            assertEquals(candidate.id, recovered.id)
            assertTrue(NativeProjectStore.matchesAnalysis(recovered, candidate))
            assertEquals(seed.ranges, recovered.ranges)
        }
    }

    @Test
    fun existingNativeProjectGeometryRemainsKnownWithoutTheNewProvenanceField() {
        val saved = readyProject()
        val legacy = NativeProjectStore.encode(saved).apply {
            put("version", 8)
            remove("analysisRoiKnown")
        }
        val restored = requireNotNull(NativeProjectStore.decode(legacy))

        assertTrue(restored.analysisRoiKnown)
        assertEquals(saved.roi, restored.roi)
        assertTrue(NativeProjectStore.matchesAnalysis(
            restored, NativeProjectStore.newQueued(saved.source, saved.media, saved.roi),
        ))
    }

    @Test
    fun teamSwitchInferenceIsOptInAndRoundTripsWhenEnabled() {
        val media = AnalysisTypes.MediaInfo(12.5, 1920, 1080, 0, "video/avc", "audio/mp4a-latm")
        val roi = AnalysisTypes.Roi(.03, .12, .94, .86, "Indoor camera default")

        val defaultProject = NativeProjectStore.newQueued(source, media, roi)
        val enabledProject = NativeProjectStore.newQueued(
            source, media, roi, sideSwitchEnabled = true,
        )

        assertEquals(false, defaultProject.sideSwitchEnabled)
        assertEquals(true, enabledProject.sideSwitchEnabled)
        assertEquals(
            true,
            NativeProjectStore.decode(NativeProjectStore.encode(enabledProject))?.sideSwitchEnabled,
        )
    }

    @Test
    fun legacyTeamSwitchProjectsInferOptInOnlyFromExistingOutput() {
        val withoutOutput = NativeProjectStore.encode(readyProject()).apply {
            remove("sideSwitchEnabled")
        }
        val withOutput = NativeProjectStore.encode(readyProject().copy(
            sideSwitch = SideSwitchOutput(
                rows = 0,
                features = doubleArrayOf(),
                candidates = emptyList(),
            ),
        )).apply {
            remove("sideSwitchEnabled")
        }

        assertEquals(false, NativeProjectStore.decode(withoutOutput)?.sideSwitchEnabled)
        assertEquals(true, NativeProjectStore.decode(withOutput)?.sideSwitchEnabled)
    }

    @Test
    fun queuedProjectCanExplicitlySkipServingSideAnalysis() {
        val project = NativeProjectStore.newQueued(
            source,
            AnalysisTypes.MediaInfo(12.5, 1920, 1080, 0, "video/avc", "audio/mp4a-latm"),
            AnalysisTypes.Roi(.03, .12, .94, .86, "Indoor camera default"),
            analyzeServingSide = false,
        )

        assertEquals(ServingSideAnalysisStatus.DISABLED, project.servingSideStatus)
        assertEquals(false, project.copy(status = ProjectStatus.READY).editorSeed()?.scoreTrackingInitiallyEnabled)
        assertEquals(
            ServingSideAnalysisStatus.DISABLED,
            NativeProjectStore.decode(NativeProjectStore.encode(project))?.servingSideStatus,
        )
    }

    @Test
    fun legacyProjectWithoutServingSideStateRemainsEligibleForDeferredAnalysis() {
        val project = readyProject()
        val legacyJson = NativeProjectStore.encode(project).apply {
            put("version", 3)
            remove("servingSideStatus")
        }

        val restored = requireNotNull(NativeProjectStore.decode(legacyJson))

        assertEquals(ServingSideAnalysisStatus.NOT_RUN, restored.servingSideStatus)
        assertEquals(true, restored.editorSeed()?.scoreTrackingInitiallyEnabled)
    }

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
    fun projectAnalysisMeasurementsRoundTripAndLegacyProjectsDefaultEmpty() {
        val measurement = AnalysisRunMeasurements(
            kind = AnalysisRunKind.PROJECT,
            completedAtMs = 1_700_000_000_000,
            succeeded = true,
            totalMilliseconds = 12_345.5,
            stageMilliseconds = linkedMapOf(
                "video_decode_and_features" to 8_000.0,
                "audio_decode_and_features" to 2_000.0,
            ),
            profileMilliseconds = linkedMapOf("video/opencv_feature_call" to 1_250.25),
            counters = linkedMapOf("sample_rows" to 500L),
        )
        val project = readyProject().copy(analysisMeasurements = listOf(measurement))

        val encoded = NativeProjectStore.encode(project)
        val restored = requireNotNull(NativeProjectStore.decode(encoded))
        assertEquals(listOf(measurement), restored.analysisMeasurements)

        encoded.remove("analysisMeasurements")
        assertEquals(
            emptyList<AnalysisRunMeasurements>(),
            NativeProjectStore.decode(encoded)?.analysisMeasurements,
        )
    }

    @Test
    fun newerMeasurementReplacesOnlyTheSameRunKind() {
        val projectRun = AnalysisRunMeasurements(
            AnalysisRunKind.PROJECT, 100L, true, totalMilliseconds = 10.0,
            stageMilliseconds = mapOf("video" to 8.0),
        )
        val oldScoreRun = AnalysisRunMeasurements(
            AnalysisRunKind.SCORE_SPECIALISTS, 200L, true, totalMilliseconds = 20.0,
            stageMilliseconds = mapOf("serving_side" to 4.0),
        )
        val newScoreRun = oldScoreRun.copy(completedAtMs = 300L, totalMilliseconds = 15.0)

        assertEquals(
            listOf(projectRun, newScoreRun),
            replaceAnalysisMeasurement(listOf(projectRun, oldScoreRun), newScoreRun),
        )
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

    @Test
    fun successfulVideoExportMetadataRoundTripsAndLegacyProjectsRemainUnsaved() {
        val exported = readyProject().copy(
            lastExportedVideoName = "Championship-final.mp4",
            lastExportedAtMs = 1_725_000_000_000,
        )

        val restored = requireNotNull(NativeProjectStore.decode(NativeProjectStore.encode(exported)))
        assertEquals("Championship-final.mp4", restored.lastExportedVideoName)
        assertEquals(1_725_000_000_000, restored.lastExportedAtMs)

        val legacy = NativeProjectStore.encode(readyProject()).apply {
            put("version", 7)
            remove("lastExportedVideoName")
            remove("lastExportedAtMs")
        }
        val restoredLegacy = requireNotNull(NativeProjectStore.decode(legacy))
        assertEquals(null, restoredLegacy.lastExportedVideoName)
        assertEquals(null, restoredLegacy.lastExportedAtMs)
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
