package com.volleycut.nativeanalysis

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.media3.common.PlaybackParameters
import androidx.media3.exoplayer.ExoPlayer
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import java.io.File

/** Synthetic UI and real Android draft I/O; no media, inference, or video encoding. */
class PortableReviewPreferencesInstrumentedTest {
    @get:Rule val compose = createAndroidComposeRule<androidx.activity.ComponentActivity>()

    @Test fun importedRateIsVisibleSelectableAndAppliedWithoutIntegerRounding() {
        val seed = seed()
        val cuts = EditorMath.newDraft(seed).cuts
        val ui = JSONObject().put("schemaVersion", 1).put("preferences", JSONObject()
            .put("playbackRate", .75).put("fadeScoreOverlay", true)
            .put("renderScoreOnPreviewVideo", false).put("renderScoreOnFinalVideo", true)
            .put("reviewedCutIds", JSONArray().put("R001"))
            .put("chapterOptions", JSONObject().put("future", JSONArray().put(JSONObject.NULL))))
        val draft = PortableReviewPreferences.import(EditorMath.newDraft(seed).copy(renderScoreOverlay = false),
            JSONObject().put("ui", ui), JSONObject())
        assertEquals(cuts, draft.cuts)
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val store = EditorDraftStore(context, seed)
        try {
            store.save(draft)
            val reopened = requireNotNull(EditorDraftStore(context, seed).load())
            assertTrue(PortableReviewPreferences.onOpen(reopened, false).fadeScoreOverlay)
            assertEquals(ui.toString(), reopened.portableUi)
            assertTrue(reopened.renderScoreOverlay)
            var rate by mutableStateOf(reopened.playbackRate)
            lateinit var player: ExoPlayer
            compose.runOnUiThread {
                player = ExoPlayer.Builder(context).build()
                player.playbackParameters = PlaybackParameters(rate.toFloat())
                compose.activity.setTurnScreenOn(true)
                compose.activity.setShowWhenLocked(true)
            }
            try {
                compose.setContent {
                    MaterialTheme {
                        PlayerControls(false, 0, 30_000, rate, onToggle = {}, onRate = {
                            rate = it
                            player.playbackParameters = PlaybackParameters(it.toFloat())
                        })
                    }
                }
                compose.onNodeWithText("Speed · 0.75x").performClick()
                compose.onNodeWithText("0.75x").assertExists()
                compose.onNodeWithText("1.5x").performClick()
                compose.onNodeWithText("Speed · 1.5x").assertExists()
                compose.runOnIdle {
                    assertEquals(1.5f, player.playbackParameters.speed, 0f)
                    store.save(reopened.copy(playbackRate = rate))
                    assertEquals(1.5, requireNotNull(store.load()).playbackRate, 0.0)
                }
                val reset = PortableReviewPreferences.reset(seed, reopened)
                store.save(reset)
                val resetReopened = requireNotNull(store.load())
                assertFalse(PortableReviewPreferences.onOpen(resetReopened, true).fadeScoreOverlay)
                assertTrue(PortableReviewPreferences.export(resetReopened).getJSONObject("preferences").has("chapterOptions"))
            } finally { compose.runOnUiThread { player.release() } }
        } finally { store.clear() }
    }

    @Test fun futureExportFailureDisplaysActionableDialogAndPreservesDraftAndDestination() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val originalSeed = seed().copy(displayName = "synthetic-future-export.mp4")
        val project = NativeProject(
            id = "synthetic-future-export", source = ProjectSource(originalSeed.sourceUri, originalSeed.displayName,
                1024, 100, "video/mp4"), media = AnalysisTypes.MediaInfo(30.0, 1920, 1080, 0, "video/avc", null),
            roi = AnalysisTypes.Roi(0.0, 0.0, 1.0, 1.0, "fixture"), status = ProjectStatus.READY,
            ranges = originalSeed.ranges, createdAtMs = 100, updatedAtMs = 100,
        )
        val ui = JSONObject().put("schemaVersion", 2).put("preferences", JSONObject()
            .put("reviewedCutIds", JSONArray().put("R002")).put("playbackRate", 1.5)
            .put("futureControl", JSONObject().put("nested", JSONArray().put(JSONObject.NULL))))
        val corrected = EditorMath.newDraft(originalSeed).copy(
            cuts = EditorMath.newDraft(originalSeed).cuts.map { it.copy(id = "R002") }, portableUi = ui.toString(),
        )
        val reset = PortableReviewPreferences.reset(originalSeed, corrected)
        val directory = File(context.cacheDir, "future-export-test-${System.nanoTime()}").apply { mkdirs() }
        val destination = File(directory, "project.json").apply { writeText("previous archive") }
        val store = EditorDraftStore(directory, originalSeed)
        var opened = false
        var error by mutableStateOf<String?>(null)
        try {
            store.save(reset)
            compose.setContent {
                MaterialTheme {
                    Button(onClick = {
                        runCatching {
                            ModelFeedbackExporter.writeArchive(
                                buildArchive = { ModelFeedbackExporter.createBundle(project, reset,
                                    EditorMath.finalIntervals(reset), null, null).toString() },
                                openDestination = { opened = true; destination.outputStream() },
                            )
                        }.onFailure { error = it.message }
                    }) { Text("Save project") }
                    error?.let { ProjectSaveErrorDialog(it, onDismiss = { error = null }) }
                }
            }
            repeat(2) {
                compose.onNodeWithText("Save project").performClick()
                compose.onNodeWithText("Could not save project").assertIsDisplayed()
                compose.onNodeWithText("Cannot export: newer project preferences refer to rallies changed or removed. " +
                    "Open an unchanged copy, or export with an app that supports this preferences version. Your local project is preserved.")
                    .assertIsDisplayed()
                compose.runOnIdle {
                    assertFalse(opened)
                    assertEquals("previous archive", destination.readText())
                    assertEquals(reset, requireNotNull(store.load()))
                    assertEquals(ui.toString(), requireNotNull(store.load()).portableUi)
                }
                compose.onNodeWithText("Close").performClick()
                compose.onNodeWithText("Could not save project").assertDoesNotExist()
            }
        } finally {
            store.clear()
            check(destination.delete())
            check(directory.delete())
        }
    }

    private fun seed() = EditorSeed(
        sourceUri = "content://fixture/portable-ui", displayName = "synthetic-ui.mp4",
        durationMs = 30_000, width = 1920, height = 1080, rotation = 0,
        ranges = listOf(SeedRange(5_000, 10_000, .8f, ProductionEnsemble.BOTH_MODELS)),
    )
}
