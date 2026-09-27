package com.volleycut.nativeanalysis

import android.content.pm.ActivityInfo
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.semantics.SemanticsProperties
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test

class SettingsProgressInstrumentedTest {
    @get:Rule val compose = createAndroidComposeRule<androidx.activity.ComponentActivity>()

    @Test fun settingsLabelRemainsReachableInLandscapeAndToggleWorks() {
        compose.activity.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
        val checked = mutableStateOf(false)
        compose.setContent {
            MaterialTheme {
                AppSettingsDialog(checked.value, { checked.value = it }, {})
            }
        }
        compose.onNodeWithText("Show analysis measurements", useUnmergedTree = true)
            .performScrollTo().assertIsDisplayed()
        compose.onNodeWithContentDescription("Show analysis measurements").performClick()
        compose.runOnIdle { assertTrue(checked.value) }
        compose.onNodeWithText("Close").assertIsDisplayed()
    }

    @Test fun preparationShowsActivityWithoutAnInventedPercentage() {
        val tracker = InferenceProgressTracker(true, false, false)
        val steps = mutableStateOf(tracker.update("opening", 0.0, "Opening the video"))
        compose.setContent { MaterialTheme { InferenceProgressMeasurementsPanel(steps.value) } }
        compose.onNodeWithText("Preparing video").assertIsDisplayed()
        compose.onNodeWithText("0%", substring = true).assertDoesNotExist()
        compose.runOnIdle { steps.value = tracker.update("video-indexing", .6, "Reading timestamps") }
        compose.onNodeWithText("Timestamp scan: 60%", substring = true).assertIsDisplayed()
        compose.onNodeWithText("Preparing video").assertIsDisplayed()
        compose.runOnIdle { steps.value = tracker.update("video", .25, "Scanning frames") }
        compose.onNodeWithText("Scanning video").assertIsDisplayed()
        compose.onNodeWithText("25%", substring = true).assertIsDisplayed()
        compose.onNodeWithText("Preparing video").assertDoesNotExist()
    }

    @Test fun elapsedTimeKeepsMovingWithoutWorkerCallbacksAndStopsWhenComplete() {
        compose.mainClock.autoAdvance = false
        val tracker = InferenceProgressTracker(true, false, false, nanoTime = { 0L })
        val steps = mutableStateOf(tracker.update("opening", 0.0, "Checking analysis files"))
        compose.setContent { MaterialTheme { InferenceProgressMeasurementsPanel(steps.value) } }
        fun elapsedText() = compose.onNodeWithText("Preparing ·", substring = true)
            .fetchSemanticsNode().config[SemanticsProperties.Text].joinToString()
        compose.mainClock.advanceTimeBy(32)
        val initial = elapsedText()
        compose.mainClock.advanceTimeBy(2_000)
        assertNotEquals(initial, elapsedText())
        compose.runOnIdle { steps.value = tracker.update("complete", 1.0, "Ready") }
        compose.mainClock.advanceTimeBy(32)
        fun totalText() = compose.onNodeWithText("Total elapsed", substring = true)
            .fetchSemanticsNode().config[SemanticsProperties.Text].joinToString()
        val completed = totalText()
        compose.mainClock.advanceTimeBy(2_000)
        org.junit.Assert.assertEquals(completed, totalText())
    }
}
