package com.volleycut.nativeanalysis

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.roundToInt

class ScoreOverlayTest {
    @Test
    fun scoresUseAtLeastTwoDigitsAndKeepLargerValues() {
        assertEquals("00", ScoreOverlay.formatScore(0))
        assertEquals("09", ScoreOverlay.formatScore(9))
        assertEquals("10", ScoreOverlay.formatScore(10))
        assertEquals("123", ScoreOverlay.formatScore(123))
    }

    @Test
    fun layoutMatchesResponsiveContractAcrossSd1080pAnd4k() {
        listOf(640 to 480, 1920 to 1080, 3840 to 2160).forEach { (width, height) ->
            val snapshot = ScoreOverlaySnapshot(
                "A very long first team name", 12, "12",
                "A much longer second team name", 9, "09",
            )
            val layout = ScoreOverlay.layout(width, height, snapshot) { text, font ->
                text.length * font * .55f
            }
            assertTrue(layout.height in 36..76)
            assertTrue(layout.width <= maxOf((layout.height * 2.25).roundToInt() * 2 + layout.scoreWidth * 2, (width * .96).toInt()))
            assertTrue(layout.team1Width > 0 && layout.team2Width > 0)
            assertEquals(if (minOf(width, height) >= 720) 2 else 1, layout.borderWidth)
        }
        assertEquals(0xffd9342b, ScoreOverlay.TEAM_1_COLOR)
        assertEquals(0xff2367c9, ScoreOverlay.TEAM_2_COLOR)
    }

    @Test
    fun layoutUsesBrowserCeilingForExactIntegerTextMeasurements() {
        val snapshot = ScoreOverlaySnapshot("Team 1", 0, "00", "Team 2", 0, "00")
        val layout = ScoreOverlay.layout(1920, 1080, snapshot) { _, _ -> 200f }
        assertEquals(234, layout.team1Width)
        assertEquals(234, layout.team2Width)
        assertEquals(648, layout.width)
    }

    @Test
    fun sourceTimeMappingAddsEachRetainedClipStart() {
        assertEquals(42_000, scoreOverlaySourceTimestampMs(42_000, 0))
        assertEquals(43_250, scoreOverlaySourceTimestampMs(42_000, 1_250_999))
    }

    @Test
    fun displayGeometrySwapsOnlyQuarterTurnInputs() {
        assertEquals(1920 to 1080, scoreOverlayDisplaySize(1920, 1080, 0))
        assertEquals(1080 to 1920, scoreOverlayDisplaySize(1920, 1080, 90))
        assertEquals(1920 to 1080, scoreOverlayDisplaySize(1920, 1080, 180))
        assertEquals(1080 to 1920, scoreOverlayDisplaySize(1920, 1080, 270))
    }
}
