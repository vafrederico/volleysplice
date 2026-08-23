package com.volleycut.nativeanalysis

import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.test.assertCountEquals
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onAllNodesWithContentDescription
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.unit.dp
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.Until
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import kotlin.math.abs

@Suppress("DEPRECATION")
class ScoreTrackingUiInstrumentedTest {
    @get:Rule
    val compose = createAndroidComposeRule<androidx.activity.ComponentActivity>()

    @Test
    fun scorePanelUsesOneHeaderAndHidesDetailsWhenDisabled() {
        var enabled by mutableStateOf(false)
        var servingSideStatus by mutableStateOf(ServingSideAnalysisStatus.NOT_RUN)
        var servingSideProgress by mutableStateOf<Float?>(null)
        val tracking = ScoreTracking()
        val device = UiDevice.getInstance(InstrumentationRegistry.getInstrumentation())
        compose.runOnUiThread {
            compose.activity.setTurnScreenOn(true)
            compose.activity.setShowWhenLocked(true)
            compose.activity.setContent {
                MaterialTheme {
                    ScoreTrackingPanel(
                        enabled = enabled,
                        tracking = tracking,
                        visibleTracking = tracking,
                        visibleScore = ScoreReducer.deriveAt(tracking),
                        selectedMarkerId = null,
                        manualServingSide = ServingSide.NEAR,
                        currentTimestampMs = 0,
                        servingSideStatus = servingSideStatus,
                        servingSideError = null,
                        servingSideProgress = servingSideProgress,
                        servingSideProgressDetail = "Sampling serving-side windows",
                        onEnabledChange = { enabled = it },
                        onTracking = {},
                        onSelect = { _, _ -> },
                        onManualServingSide = {},
                    )
                }
            }
        }
        compose.waitForIdle()
        assertTrue(device.hasObject(By.desc("Score tracking controls")))
        assertFalse(device.hasObject(By.desc("Score marker list")))
        compose.runOnIdle {
            enabled = true
            servingSideStatus = ServingSideAnalysisStatus.ANALYZING
            servingSideProgress = 0.42f
        }
        compose.waitForIdle()
        assertTrue(device.wait(Until.hasObject(By.desc("Score marker list")), 2_000))
        assertTrue(device.wait(Until.hasObject(By.text("42%")), 2_000))
    }

    @Test
    fun markerLineKeepsNormalTimelineTapOutsideItsIconZone() {
        var selectedId: String? = null
        var seekTimestamp: Long? = null
        val device = UiDevice.getInstance(InstrumentationRegistry.getInstrumentation())
        compose.runOnUiThread {
            compose.activity.setTurnScreenOn(true)
            compose.activity.setShowWhenLocked(true)
            compose.activity.setContent {
                MaterialTheme {
                    Box(Modifier.padding(top = 80.dp).width(300.dp)) {
                        WholeTimeline(
                            windowStartMs = 0,
                            windowEndMs = 10_000,
                            cuts = emptyList(),
                            joinedGaps = emptyList(),
                            ignored = emptyList(),
                            suggestions = emptyList(),
                            appliedSuggestionIds = emptySet(),
                            selectedSuggestionId = null,
                            selectedId = null,
                            effectiveIds = emptySet(),
                            confidenceThreshold = .5f,
                            playheadMs = 0,
                            serveMarkers = listOf(ServeMarker(
                                "S001", 5_000, ServingSide.NEAR, ServeMarkerOrigin.MANUAL,
                            )),
                            sideSwitchMarkers = emptyList(),
                            onMarkerSelect = { id, _ -> selectedId = id },
                            onSeek = { timestamp, _, _ -> seekTimestamp = timestamp },
                        )
                    }
                }
            }
        }
        compose.waitForIdle()
        var timelineCenterX = 0
        var timelineBottomY = 0
        var timelineTopY = 0
        compose.runOnIdle {
            val content = compose.activity.findViewById<android.view.View>(android.R.id.content)
            val location = IntArray(2)
            content.getLocationOnScreen(location)
            val density = compose.activity.resources.displayMetrics.density
            timelineCenterX = location[0] + (150 * density).toInt()
            timelineTopY = location[1] + (80 * density).toInt()
            timelineBottomY = timelineTopY + (34 * density).toInt()
        }
        assertTrue(device.click(timelineCenterX, timelineBottomY - 2))
        compose.waitUntil(2_000) { seekTimestamp != null }
        assertTrue(abs(checkNotNull(seekTimestamp) - 5_000L) <= 20L)
        assertEquals(null, selectedId)
        seekTimestamp = null
        assertTrue(device.click(timelineCenterX, timelineTopY + 3))
        compose.waitUntil(2_000) { selectedId != null }
        assertEquals("S001", selectedId)
        assertEquals(null, seekTimestamp)
    }

    @Test
    fun reviewPredictionsAreHighlightedAndCanBeSelectedFromStatus() {
        var selectedId by mutableStateOf<String?>(null)
        val tracking = ScoreTracking(
            serveMarkers = listOf(
                ServeMarker("S001", 1_000, ServingSide.REVIEW, ServeMarkerOrigin.MODEL),
                ServeMarker("S002", 2_000, ServingSide.NEAR, ServeMarkerOrigin.MODEL),
                ServeMarker("S003", 3_000, ServingSide.REVIEW, ServeMarkerOrigin.MODEL),
            ),
        )
        compose.setContent {
            MaterialTheme {
                ScoreTrackingPanel(
                    enabled = true,
                    tracking = tracking,
                    visibleTracking = tracking,
                    visibleScore = ScoreReducer.deriveAt(tracking),
                    selectedMarkerId = selectedId,
                    manualServingSide = ServingSide.NEAR,
                    currentTimestampMs = 0,
                    servingSideStatus = ServingSideAnalysisStatus.READY,
                    servingSideError = null,
                    servingSideProgress = 1f,
                    servingSideProgressDetail = null,
                    onEnabledChange = {},
                    onTracking = {},
                    onSelect = { id, _ -> selectedId = id },
                    onManualServingSide = {},
                )
            }
        }

        compose.onAllNodesWithText("Review").assertCountEquals(0)
        compose.onAllNodesWithContentDescription("Score marker needs review").assertCountEquals(2)
        compose.onNodeWithText("Review next · 2").performClick()
        compose.runOnIdle { assertEquals("S001", selectedId) }
        compose.onNodeWithText("Review next · 2").performClick()
        compose.runOnIdle { assertEquals("S003", selectedId) }
    }
}
