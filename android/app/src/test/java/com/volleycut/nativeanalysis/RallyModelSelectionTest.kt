package com.volleycut.nativeanalysis

import org.junit.Assert.*
import org.junit.Test

class RallyModelSelectionTest {
    private val source = ProjectSource("content://test/video", "synthetic.mp4", 1200, 10, "video/mp4")
    private val media = AnalysisTypes.MediaInfo(12.0, 320, 180, 0, "video/avc", null)

    @Test fun pairedPreparationIsOptInAndDoesNotChangeVariantIdentity() {
        val single = NativeProjectStore.newQueued(source, media)
        val paired = NativeProjectStore.newQueued(source, media, prepareBothVariants = true)
        assertFalse(single.prepareBothVariants)
        assertEquals(single.id, paired.id)
        assertTrue(requireNotNull(NativeProjectStore.decode(NativeProjectStore.encode(paired))).prepareBothVariants)
        assertFalse(NativeProjectStore.newQueued(source, media, rallyModelId = FeatureSchema.MODEL_ID,
            prepareBothVariants = true).prepareBothVariants)
        val older = NativeProjectStore.encode(single).apply { remove("prepareBothVariants"); put("version", 11) }
        assertFalse(requireNotNull(NativeProjectStore.decode(older)).prepareBothVariants)
    }

    @Test fun missingOrStaleAgreementNeverChangesTheSelectedNeuralBundle() {
        for (model in listOf(RallyModels.RECALL, RallyModels.F1)) {
            for (agreement in listOf(null, ProductionEnsemble.BOTH_MODELS)) {
                val saved = NativeProjectStore.newQueued(source, media, rallyModelId = model).copy(
                    status = ProjectStatus.READY, ranges = listOf(SeedRange(1000, 3000, .7f, agreement)),
                )
                val restored = NativeProjectStore.normalizeStored(requireNotNull(
                    NativeProjectStore.decode(NativeProjectStore.encode(saved))))
                assertEquals(ProjectStatus.QUEUED, restored.status)
                assertEquals(model, restored.modelId)
                assertEquals(saved.id, restored.id)
                assertTrue(restored.ranges.isEmpty())
                assertEquals(model, NativeProjectStore.normalizeStored(restored).modelId)
            }
        }
    }

    @Test fun unavailableVariantRequiresExplicitSelectionAndPreservesSavedOutput() {
        val saved = NativeProjectStore.newQueued(source, media).copy(
            status = ProjectStatus.READY, modelId = "unavailable-future-model",
            ranges = listOf(SeedRange(1000, 3000, .7f, "neural")),
        )
        val restored = NativeProjectStore.normalizeStored(saved)
        assertEquals(ProjectStatus.ERROR, restored.status)
        assertEquals(saved.modelId, restored.modelId)
        assertEquals(saved.ranges, restored.ranges)
        assertTrue(restored.error.orEmpty().contains("unavailable"))
    }

    @Test fun selectionIsDefaultedAndNeverReusesAnotherModelsReviewedProject() {
        val recall = NativeProjectStore.newQueued(source, media)
        val f1 = NativeProjectStore.newQueued(source, media, rallyModelId = RallyModels.F1)
        val ensemble = NativeProjectStore.newQueued(source, media, rallyModelId = FeatureSchema.MODEL_ID)
        assertEquals(RallyModels.RECALL, recall.modelId)
        assertEquals(3, setOf(recall.id, f1.id, ensemble.id).size)
        assertFalse(NativeProjectStore.matchesAnalysis(recall, f1))
        assertFalse(NativeProjectStore.matchesAnalysis(recall, ensemble))
        for (project in listOf(recall, f1)) {
            val saved = project.copy(status = ProjectStatus.READY,
                ranges = listOf(SeedRange(1000, 3000, .7f, "neural")))
            val restored = NativeProjectStore.normalizeStored(requireNotNull(
                NativeProjectStore.decode(NativeProjectStore.encode(saved))))
            assertEquals(ProjectStatus.READY, restored.status)
            assertEquals(project.modelId, restored.modelId)
            val seed = requireNotNull(EditorProjectStore.decode(EditorProjectStore.encode(requireNotNull(restored.editorSeed()))))
            assertEquals(project.modelId, seed.rallyModelId)
            val recovered = NativeProjectStore.fromSeed(seed, source, media)
            assertEquals(project.id, recovered.id)
            assertTrue(NativeProjectStore.matchesAnalysis(recovered, project))
        }
    }
}
