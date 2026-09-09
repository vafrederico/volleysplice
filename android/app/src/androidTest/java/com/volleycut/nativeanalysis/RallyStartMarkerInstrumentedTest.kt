package com.volleycut.nativeanalysis

import android.content.Intent
import android.net.Uri
import android.util.Base64
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.test.click
import androidx.compose.ui.test.junit4.createEmptyComposeRule
import androidx.compose.ui.test.onAllNodesWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performTouchInput
import androidx.test.core.app.ActivityScenario
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.UiDevice
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import java.io.File

/** Exercises the real editor, its save path, and training-feedback serialization. */
class RallyStartMarkerInstrumentedTest {
    @get:Rule val compose = createEmptyComposeRule()

    @Test fun movingStartForwardMovesServeAndSurvivesReopen() = moveStart(.3f, "forward")
    @Test fun movingStartBackwardMovesServeAndSurvivesReopen() = moveStart(.25f, "backward")

    private fun moveStart(fraction: Float, direction: String) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val video = File(context.filesDir, "rally-start-marker-fixture.mp4")
        video.writeBytes(Base64.decode(
            instrumentation.context.assets.open("overlay-fixture.mp4.b64")
                .bufferedReader().use { it.readText() }, Base64.DEFAULT,
        ))
        val evidence = ServingSideHeadEvidence("fixture", .5, .9, .5, true, null)
        val seedInput = EditorSeed(
            Uri.fromFile(video).toString(), video.name, 1_000, 320, 240, 0,
            listOf(SeedRange(500, 900, .9f, ProductionEnsemble.BOTH_MODELS)),
            servingSide = ServingSideOutput(
                rows = 1, rawFeatures = DoubleArray(237),
                candidates = listOf(ServingSideCandidate(
                    "R001", .5, .5, .9, ProductionEnsemble.BOTH_MODELS,
                    .9, ServingSide.NEAR, ServingSideVerdict.NEAR,
                    ServingSideDecisionSource.SERVE_HEAD, emptyList(),
                    evidence.copy(modelId = FeatureSchema.ALL_LABELS_V2_MODEL_ID),
                    evidence.copy(modelId = FeatureSchema.PREVIOUS_PRODUCTION_MODEL_ID),
                )),
            ),
            scoreTrackingInitiallyEnabled = true,
        )
        val project = NativeProjectStore.fromSeed(context, seedInput)
        NativeProjectStore.save(context, project)
        NativeProjectStore.setSelectedId(context, project.id)
        GuidedTourStore.write(context, "done")
        val seed = checkNotNull(project.editorSeed())
        val store = EditorDraftStore(context, seed)
        store.save(EditorMath.newDraft(seed))
        val rawFile = File(context.filesDir, "editor-drafts/${seed.sourceRevision}.json")
        val device = UiDevice.getInstance(instrumentation)
        device.wakeUp()
        ActivityScenario.launch<EditorActivity>(Intent(context, EditorActivity::class.java)).use { scenario ->
            compose.waitForIdle()
            compose.onAllNodesWithContentDescription("Game timeline")[if (direction == "forward") 1 else 0]
                .performScrollTo().performTouchInput {
                    // Below the marker icons: ordinary seek, without selecting a marker.
                    click(Offset(width * fraction, height - 2f))
                }
            // Seeking alone must leave the annotations unchanged.
            val beforeEdit = JSONObject(rawFile.readText())
            assertEquals(500L, beforeEdit.getJSONArray("cuts").getJSONObject(0).getLong("coreStartMs"))
            assertEquals(500L, beforeEdit.getJSONObject("scoreTracking")
                .getJSONArray("serveMarkers").getJSONObject(0).getLong("timestampMs"))
            compose.onNodeWithText("Set rally start here").performScrollTo().performClick()
            compose.waitUntil(5_000) {
                JSONObject(rawFile.readText()).getJSONArray("cuts")
                    .getJSONObject(0).getLong("coreStartMs") != 500L
            }
            val saved = JSONObject(rawFile.readText())
            val start = saved.getJSONArray("cuts").getJSONObject(0).getLong("coreStartMs")
            assertTrue(if (direction == "forward") start > 500 else start < 500)
            val marker = saved.getJSONObject("scoreTracking").getJSONArray("serveMarkers")
                .getJSONObject(0).getLong("timestampMs")
            val evidenceDir = File(context.getExternalFilesDir(null), "rally-start-marker").apply { mkdirs() }
            File(evidenceDir, "$direction.json").writeText(saved.toString(2))
            device.takeScreenshot(File(evidenceDir, "$direction.png"))
            assertEquals("Saved serve must follow the $direction rally-start edit", start, marker)

            scenario.recreate()
            compose.waitForIdle()
            val restored = checkNotNull(store.load())
            assertEquals(start, restored.scoreTracking.serveMarkers.single().timestampMs)
            val bundle = ModelFeedbackExporter.createBundle(
                project, restored, EditorMath.finalIntervals(restored),
                ModelFeedbackAnalysis(
                    timestamps = doubleArrayOf(0.0, .25, .5, .75),
                    baseFeatures = FloatArray(4 * FeatureSchema.BASE.size),
                    rallyProbabilities = FloatArray(4) { .9f },
                    serveProbabilities = FloatArray(4) { .9f },
                    deadStateProbabilities = FloatArray(4),
                ), null,
            )
            File(evidenceDir, "$direction-feedback.json").writeText(bundle.toString(2))
            val exportedMarker = bundle.getJSONObject("corrections")
                .getJSONObject("scoreTracking").getJSONObject("state")
                .getJSONArray("serveMarkers").getJSONObject(0).getDouble("timestamp")
            assertEquals(start / 1_000.0, exportedMarker, .000001)
            assertEquals(start / 1_000.0, bundle.getJSONObject("corrections")
                .getJSONArray("correctedRanges").getJSONObject(0).getDouble("coreStart"), .000001)
            val imported = ModelFeedbackImporter.parse(bundle.toString())
            assertEquals(start, imported.draft.scoreTracking.serveMarkers.single().timestampMs)
            assertEquals(.5, project.servingSide!!.candidates.single().anchor, .000001)

            val snapshot = ScoreExportSnapshot(
                render = true, scoreTracking = restored.scoreTracking,
                ignoredIntervals = restored.ignoredIntervals, excludedRallyIds = emptySet(),
                rallyRanges = restored.cuts.map {
                    ScoreRallyRange(it.coreStartMs, it.coreEndMs, it.keepStartMs, it.keepEndMs)
                },
            )
            val encodedSnapshot = ScoreExportSnapshotJson.encode(snapshot)
            File(evidenceDir, "$direction-video-score-snapshot.json").writeText(encodedSnapshot)
            assertEquals(start, checkNotNull(ScoreExportSnapshotJson.decode(encodedSnapshot, seed.durationMs))
                .scoreTracking.serveMarkers.single().timestampMs)

            // Older app versions saved the edited cut alongside the original marker.
            store.save(restored.copy(scoreTracking = restored.scoreTracking.copy(
                serveMarkers = restored.scoreTracking.serveMarkers.map { it.copy(timestampMs = 500) },
            )))
            assertEquals(start, checkNotNull(store.load()).scoreTracking.serveMarkers.single().timestampMs)
        }
    }
}
