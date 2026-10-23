package com.volleycut.nativeanalysis

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.security.MessageDigest

@RunWith(AndroidJUnit4::class)
class SideSwitchFeedbackInstrumentedTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext

    @Test
    fun sharedWebFixtureSurvivesDiskReopenMetadataRelinkAndReset() {
        val selectedBefore = NativeProjectStore.selectedId(context)
        try {
            for (enabled in listOf(true, false)) for (archived in listOf(true, false)) {
                val bundle = fixture()
                bundle.getJSONObject("source").put("generateSideSwitchMarkers", enabled)
                if (archived) bundle.getJSONObject("initialInference").getJSONObject("sideSwitch")
                    .put("modelId", "synthetic-archived-model").put("modelFingerprint", "synthetic-archived-fingerprint")
                val imported = ModelFeedbackImporter.import(context, bundle.toString())
                val oldSeed = requireNotNull(imported.editorSeed())
                var project = imported
                try {
                    project = requireNotNull(NativeProjectStore.get(context, imported.id))
                    assertEquals(imported.sideSwitch, project.sideSwitch)
                    assertEquals(enabled, project.sideSwitchEnabled)
                    assertEquals(ServingSideAnalysisStatus.READY, project.servingSideStatus)
                    val seed = requireNotNull(project.editorSeed())
                    val draft = (EditorDraftStore(context, seed).load() as EditorDraftLoadResult.Restored).draft
                    assertReview(draft)
                    assertEquals(project.sideSwitch, requireNotNull(EditorProjectStore.load(context)).sideSwitch)

                    // Exercises production metadata migration without claiming a physical provider check.
                    project = NativeProjectStore.relink(context, project, project.source.copy(
                        uri = "content://synthetic/relinked/${project.id}", name = "synthetic-relinked.mp4",
                    ), project.media)
                    project = requireNotNull(NativeProjectStore.get(context, project.id))
                    assertEquals(enabled, project.sideSwitchEnabled)
                    assertEquals(imported.sideSwitch, project.sideSwitch)
                    val relinkedSeed = requireNotNull(project.editorSeed())
                    val restored = (EditorDraftStore(context, relinkedSeed).load() as EditorDraftLoadResult.Restored).draft
                    assertReview(restored)
                    val reexport = ModelFeedbackExporter.createBundle(
                        project, restored, EditorMath.finalIntervals(restored), null, null,
                    )
                    val reparsed = ModelFeedbackImporter.parse(reexport.toString())
                    assertEquals(project.sideSwitch, reparsed.project.sideSwitch)
                    assertEquals(enabled, reparsed.project.sideSwitchEnabled)
                    assertReview(reparsed.draft)

                    val reset = EditorMath.newDraft(relinkedSeed)
                    EditorDraftStore(context, relinkedSeed).save(reset)
                    val resetFromDisk = (EditorDraftStore(context, relinkedSeed).load() as EditorDraftLoadResult.Restored).draft
                    assertTrue(resetFromDisk.scoreTracking.removedModelMarkerIds.isEmpty())
                    assertEquals(if (enabled) listOf(25_000L, 45_000L) else emptyList<Long>(),
                        resetFromDisk.scoreTracking.sideSwitchMarkers.map { it.timestampMs })
                    assertEquals(imported.sideSwitch, requireNotNull(NativeProjectStore.get(context, project.id)).sideSwitch)
                    assertEquals(enabled, requireNotNull(NativeProjectStore.get(context, project.id)).sideSwitchEnabled)
                } finally {
                    EditorDraftStore(context, oldSeed).clear()
                    NativeProjectStore.delete(context, project)
                }
            }
        } finally {
            NativeProjectStore.setSelectedId(context, selectedBefore)
        }
    }

    @Test
    fun malformedOutputFailsBeforeAnyProjectDraftSelectionOrCachePublication() {
        val original = ModelFeedbackImporter.parse(fixture().toString())
        val times = AnalysisEngine.analysisTimes(60.0, 0.0, 60.0)
        val bundle = ModelFeedbackExporter.createBundle(
            original.project, original.draft, EditorMath.finalIntervals(original.draft),
            ModelFeedbackAnalysis(times, FloatArray(times.size * FeatureSchema.BASE.size),
                FloatArray(times.size), FloatArray(times.size), FloatArray(times.size)), null,
        )
        // The same bundle is otherwise valid and includes hydratable base features.
        assertNotNull(ModelFeedbackImporter.parse(bundle.toString()).featureCache)
        bundle.getJSONObject("initialInference").getJSONObject("sideSwitch")
            .getJSONObject("features").getJSONObject("values").put("data", "invalid")
        val filesBefore = filesSnapshot()
        val selectedBefore = NativeProjectStore.selectedId(context)
        assertThrows(Exception::class.java) { ModelFeedbackImporter.import(context, bundle.toString()) }
        assertEquals(filesBefore, filesSnapshot())
        assertEquals(selectedBefore, NativeProjectStore.selectedId(context))
    }

    private fun fixture() = JSONObject(instrumentation.context.assets.open("side-switch-web-project.json")
        .bufferedReader().use { it.readText() })

    private fun assertReview(draft: EditorDraft) {
        assertEquals(listOf("switch-boundary-1", "manual-switch"), draft.scoreTracking.sideSwitchMarkers.map { it.id })
        assertEquals(listOf(35_000L, 40_000L), draft.scoreTracking.sideSwitchMarkers.map { it.timestampMs })
        assertEquals(setOf("switch-internal-2"), draft.scoreTracking.removedModelMarkerIds)
    }

    private fun filesSnapshot() = context.filesDir.walkTopDown().filter { it.isFile }.associate {
        it.relativeTo(context.filesDir).path to MessageDigest.getInstance("SHA-256").digest(it.readBytes()).toList()
    }
}
