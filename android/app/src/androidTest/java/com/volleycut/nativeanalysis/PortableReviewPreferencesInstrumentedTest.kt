package com.volleycut.nativeanalysis

import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.test.junit4.createAndroidComposeRule
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

    private fun seed() = EditorSeed(
        sourceUri = "content://fixture/portable-ui", displayName = "synthetic-ui.mp4",
        durationMs = 30_000, width = 1920, height = 1080, rotation = 0,
        ranges = listOf(SeedRange(5_000, 10_000, .8f, ProductionEnsemble.BOTH_MODELS)),
    )
}
