package com.volleycut.nativeanalysis

import org.junit.Assert.assertEquals
import org.junit.Test

class EditorLayoutModeTest {
    @Test
    fun uiScaleUsesFivePercentStepsAndStaysWithinSupportedRange() {
        assertEquals(0.75f, normalizeUiScale(0.5f))
        assertEquals(0.8f, normalizeUiScale(0.81f))
        assertEquals(1f, normalizeUiScale(0.99f))
        assertEquals(1.25f, normalizeUiScale(1.5f))
    }

    @Test
    fun compactLayoutRemainsDefaultForPhones() {
        assertEquals(EditorLayoutMode.COMPACT, editorLayoutMode(widthDp = 412, heightDp = 915))
        assertEquals(EditorLayoutMode.COMPACT, editorLayoutMode(widthDp = 839, heightDp = 600))
    }

    @Test
    fun desktopLayoutActivatesForWideUsableWindows() {
        assertEquals(EditorLayoutMode.DESKTOP, editorLayoutMode(widthDp = 840, heightDp = 360))
        assertEquals(EditorLayoutMode.DESKTOP, editorLayoutMode(widthDp = 1280, heightDp = 800))
    }

    @Test
    fun veryShortWideWindowFallsBackToCompactLayout() {
        assertEquals(EditorLayoutMode.COMPACT, editorLayoutMode(widthDp = 1280, heightDp = 359))
    }

    @Test
    fun desktopPlayerResizeUsesDisplayDensityAndClampsTheHeight() {
        assertEquals(370f, resizedDesktopPlayerHeight(320f, dragDeltaPx = 100f, density = 2f))
        assertEquals(
            DESKTOP_PLAYER_MIN_HEIGHT_DP,
            resizedDesktopPlayerHeight(200f, dragDeltaPx = -1000f, density = 2f),
        )
        assertEquals(
            DESKTOP_PLAYER_MAX_HEIGHT_DP,
            resizedDesktopPlayerHeight(600f, dragDeltaPx = 2000f, density = 2f),
        )
    }

    @Test
    fun compactAndSetupPlayerResizeUsesItsOwnHeightBounds() {
        assertEquals(
            226f,
            resizedPlayerHeight(
                currentHeightDp = 176f,
                dragDeltaPx = 100f,
                density = 2f,
                minHeightDp = 140f,
                maxHeightDp = 720f,
            ),
        )
        assertEquals(
            COMPACT_PLAYER_MIN_HEIGHT_DP,
            resizedPlayerHeight(
                currentHeightDp = 176f,
                dragDeltaPx = -1000f,
                density = 2f,
                minHeightDp = 140f,
                maxHeightDp = 720f,
            ),
        )
        assertEquals(
            SETUP_PLAYER_DESKTOP_MAX_HEIGHT_DP,
            resizedPlayerHeight(
                currentHeightDp = 600f,
                dragDeltaPx = 2000f,
                density = 2f,
                minHeightDp = 140f,
                maxHeightDp = 1200f,
            ),
        )
    }

    @Test
    fun desktopSidebarResizeUsesDisplayDensityAndClampsTheWidth() {
        assertEquals(
            342f,
            resizedDesktopSidebarWidth(292f, dragDeltaPx = 100f, density = 2f, 180f, 520f),
        )
        assertEquals(
            DESKTOP_LEFT_PANE_MIN_WIDTH_DP,
            resizedDesktopSidebarWidth(292f, dragDeltaPx = -1000f, density = 2f, 180f, 520f),
        )
        assertEquals(
            DESKTOP_RIGHT_PANE_MAX_WIDTH_DP,
            resizedDesktopSidebarWidth(292f, dragDeltaPx = 1000f, density = 2f, 220f, 560f),
        )
    }
}
