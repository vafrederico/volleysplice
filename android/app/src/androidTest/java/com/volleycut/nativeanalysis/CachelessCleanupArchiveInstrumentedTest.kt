package com.volleycut.nativeanalysis

import android.app.Activity
import android.app.Instrumentation
import android.content.Intent
import android.net.Uri
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.test.core.app.ActivityScenario
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import java.io.File

/** Production import, disk restore, Activity recreation and project-save callback; no provider/codec claim. */
class CachelessCleanupArchiveInstrumentedTest {
    @get:Rule val compose = createEmptyComposeRule()

    @Test fun cachelessCleanupAndHighlightOnlySurviveEditorRecreationAndProjectSave() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val selectedBefore = NativeProjectStore.selectedId(context)
        val destination = File(context.cacheDir, "synthetic-cacheless-reexport.json")
        val monitor = object : Instrumentation.ActivityMonitor() {
            override fun onStartActivity(intent: Intent): Instrumentation.ActivityResult? =
                if (intent.action == Intent.ACTION_CREATE_DOCUMENT)
                    Instrumentation.ActivityResult(Activity.RESULT_OK, Intent().setData(Uri.fromFile(destination))) else null
        }
        instrumentation.addMonitor(monitor)
        GuidedTourStore.write(context, "done")
        try {
            for (scope in SuppressionScope.entries) for (behavior in SuppressionInitialBehavior.entries) {
                destination.delete()
                val original = project()
                val seed = requireNotNull(original.editorSeed())
                val logicalId = original.suppression!!.suggestions().single().logicalId()
                val draft = EditorMath.applyPadding(EditorMath.newDraft(seed), 0, 0, 30_000).copy(
                    joinGapMs = 0, suppressionInitialBehavior = behavior,
                    suppressionScopeOverrides = mapOf(logicalId to scope),
                )
                val expected = EditorMath.finalIntervals(draft, original.suppression)
                val expectedDuration = if (behavior == SuppressionInitialBehavior.HIGHLIGHT_ONLY) 10_000L
                    else if (scope == SuppressionScope.WHOLE_RALLY) 5_000L else 8_000L
                assertEquals(expectedDuration, expected.sumOf { it.endMs - it.startMs })
                val archive = ModelFeedbackExporter.createBundle(original, draft, expected, null, null)
                val imported = ModelFeedbackImporter.import(context, archive.toString())
                val restoredSeed = requireNotNull(imported.editorSeed())
                val store = EditorDraftStore(context, restoredSeed)
                fun restored() = (store.load() as EditorDraftLoadResult.Restored).draft
                try {
                    assertNull(ModelFeedbackExporter.loadAnalysis(context, imported))
                    assertEquals(expected, EditorMath.finalIntervals(restored(), imported.suppression))
                    ActivityScenario.launch<EditorActivity>(Intent(context, EditorActivity::class.java)).use { scenario ->
                        compose.waitForIdle()
                        if (compose.onAllNodesWithText("Not now").fetchSemanticsNodes().isNotEmpty())
                            compose.onNodeWithText("Not now").performClick()
                        compose.onNodeWithText("Project options").performScrollTo().assertExists()
                        scenario.recreate()
                        compose.waitForIdle()
                        assertEquals(behavior, restored().suppressionInitialBehavior)
                        assertEquals(scope, EditorMath.suggestionEffectiveScope(restored(), imported.suppression!!.suggestions().single()))
                        assertEquals(expected, EditorMath.finalIntervals(restored(), imported.suppression))
                        compose.onNodeWithText("Project options").performScrollTo().performClick()
                        compose.onNodeWithText("Save project").performClick()
                        compose.waitUntil(10_000) { destination.exists() && runCatching {
                            ModelFeedbackImporter.parse(destination.readText())
                        }.isSuccess }
                        val again = ModelFeedbackImporter.parse(destination.readText())
                        assertEquals(expected, EditorMath.finalIntervals(again.draft, again.project.suppression))
                        assertEquals(behavior, again.draft.suppressionInitialBehavior)
                        assertEquals(draft.cuts, again.draft.cuts)
                        assertNull(again.featureCache)
                    }
                } finally {
                    EditorDraftSessions().close(store).get()
                    NativeProjectStore.delete(context, imported)
                }
            }
        } finally {
            instrumentation.removeMonitor(monitor)
            NativeProjectStore.setSelectedId(context, selectedBefore)
            destination.delete()
        }
    }

    private fun project(): NativeProject {
        val all = listOf(AnalysisTypes.Interval(5.0, 10.0, .8f), AnalysisTypes.Interval(20.0, 25.0, .8f))
        val previous = listOf(all.first())
        val rows = AnalysisEngine.analysisTimes(30.0, 0.0, 30.0).size
        return NativeProject(
            id = "synthetic-cacheless-qualification",
            source = ProjectSource("content://synthetic/cacheless", "synthetic-cleanup.mp4", 1_024, 100, "video/mp4"),
            media = AnalysisTypes.MediaInfo(30.0, 1920, 1080, 0, "video/avc", "audio/mp4a-latm"),
            roi = AnalysisTypes.Roi(0.0, 0.0, 1.0, 1.0, "synthetic"), status = ProjectStatus.READY,
            ranges = listOf(SeedRange(5_000, 10_000, .8f, ProductionEnsemble.BOTH_MODELS),
                SeedRange(20_000, 25_000, .8f, ProductionEnsemble.ALL_LABELS_V2_ONLY)),
            productionComponents = AnalysisTypes.ProductionComponents(all, previous),
            suppression = SuppressionPolicyEngine.build(all, previous, FloatArray(rows) { .3f },
                listOf(AnalysisTypes.Interval(21.0, 23.0, .9f)), 30.0),
            createdAtMs = 100, updatedAtMs = 100,
        )
    }
}
