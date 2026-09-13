package com.volleycut.nativeanalysis

import androidx.compose.material3.MaterialTheme
import androidx.compose.ui.test.assertIsOn
import androidx.compose.ui.test.assertIsOff
import androidx.compose.ui.test.assertTextEquals
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test

class YouTubeChaptersUiInstrumentedTest {
    @get:Rule
    val compose = createAndroidComposeRule<androidx.activity.ComponentActivity>()

    @Test
    fun creditToggleControlsPreviewCopyAndSavedTextAndSkipsUnmarkedRallies() {
        var copied = ""
        var saved = ""
        var filename = ""
        compose.setContent {
            MaterialTheme {
                YouTubeChaptersDialog(
                    sourceFilename = "Match.mp4",
                    intervals = listOf(FinalCutInterval(10_000, 20_000, listOf("R1")), FinalCutInterval(40_000, 50_000, listOf("R2"))),
                    cuts = listOf(
                        EditableCut("R1", 10_000, 20_000, 10_000, 20_000, 1f, true, CutOrigin.INFERRED),
                        EditableCut("R2", 40_000, 50_000, 40_000, 50_000, 1f, true, CutOrigin.INFERRED),
                    ),
                    scoreTracking = ScoreTracking(serveMarkers = listOf(
                        ServeMarker("S2", 42_000, ServingSide.NEAR, ServeMarkerOrigin.MANUAL, rallyId = "R2"),
                    )),
                    hasSideSwitches = false,
                    status = null, onClearStatus = {}, onDismiss = {},
                    onCopy = { copied = it }, onSaveTextFile = { text, name -> saved = text; filename = name },
                )
            }
        }
        val plain = "0:10 0\u20130 - Team 1 serving"
        val credited = "Edited with https://volleysplice.com\n\n$plain"
        val credit = compose.onNodeWithTag("chapter-option-Include VolleySplice credit")
        credit.performScrollTo().assertIsOn()
        compose.onNodeWithTag("youtube-chapters-preview").performScrollTo().assertTextEquals(credited)
        compose.onNodeWithTag("youtube-chapters-copy").performClick()
        compose.runOnIdle { assertEquals(credited, copied) }
        compose.onNodeWithTag("youtube-chapters-save-file").performClick()
        compose.runOnIdle { assertEquals(credited, saved); assertEquals("Match-youtube-chapters.txt", filename) }
        credit.performScrollTo().performClick().assertIsOff()
        compose.onNodeWithTag("youtube-chapters-preview").performScrollTo().assertTextEquals(plain)
        compose.onNodeWithTag("youtube-chapters-copy").performClick()
        compose.onNodeWithTag("youtube-chapters-save-file").performClick()
        compose.runOnIdle { assertEquals(plain, copied); assertEquals(plain, saved) }
    }
}
