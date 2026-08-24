package com.volleycut.nativeanalysis

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
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
    fun snapshotUsesTheUpcomingServerThroughoutPreServePadding() {
        val tracking = ScoreTracking(
            team1Name = "Falcons",
            team2Name = "Waves",
            serveMarkers = listOf(
                ServeMarker("S1", 1_000, ServingSide.NEAR, ServeMarkerOrigin.MANUAL),
                ServeMarker("S2", 5_000, ServingSide.FAR, ServeMarkerOrigin.MANUAL),
            ),
        )
        val prepared = ScoreOverlay.prepare(
            tracking,
            emptyList(),
            emptySet(),
            listOf(
                ScoreRallyRange(1_000, 3_000, 900, 4_000),
                ScoreRallyRange(5_000, 7_000, 4_000, 8_000),
            ),
        )

        assertEquals(ScoreTeamId.TEAM_1, ScoreOverlay.snapshot(prepared, 3_999).servingTeamId)
        assertEquals(ScoreTeamId.TEAM_2, ScoreOverlay.snapshot(prepared, 4_000).servingTeamId)
        assertEquals(ScoreTeamId.TEAM_2, ScoreOverlay.snapshot(prepared, 4_999).servingTeamId)
        assertEquals(ScoreTeamId.TEAM_2, ScoreOverlay.snapshot(prepared, 5_000).servingTeamId)
        assertEquals("Falcons 🏐", ScoreOverlay.formatTeamLabel("Falcons", true))
        assertEquals("Falcons", ScoreOverlay.formatTeamLabel("Falcons", false))
    }

    @Test
    fun pointTimelineRevealsAtLeadingPaddingHoldsAndFades() {
        val tracking = ScoreTracking(
            serveMarkers = listOf(
                ServeMarker("S1", 1_000, ServingSide.NEAR, ServeMarkerOrigin.MANUAL),
                ServeMarker("S2", 5_000, ServingSide.NEAR, ServeMarkerOrigin.MANUAL),
                ServeMarker("S3", 10_000, ServingSide.FAR, ServeMarkerOrigin.MANUAL),
            ),
        )
        val prepared = ScoreOverlay.prepare(
            tracking,
            emptyList(),
            emptySet(),
            listOf(
                ScoreRallyRange(1_000, 3_000, 500, 3_500),
                ScoreRallyRange(5_000, 8_000, 4_000, 8_500),
                ScoreRallyRange(10_000, 13_000, 9_000, 13_500),
            ),
        )

        assertEquals(
            listOf(ScorePointTimelineEntry("S2", ScoreTeamId.TEAM_1, 1)),
            ScoreOverlay.pointTimelineSnapshot(prepared, 4_000).points,
        )
        assertEquals(0f, ScoreOverlay.pointTimelineSnapshot(prepared, 4_000).opacity, .0001f)
        assertEquals(.5f, ScoreOverlay.pointTimelineSnapshot(prepared, 4_125).opacity, .0001f)
        assertEquals(1f, ScoreOverlay.pointTimelineSnapshot(prepared, 4_250).opacity, .0001f)
        assertEquals(1f, ScoreOverlay.pointTimelineSnapshot(prepared, 6_250).opacity, .0001f)
        assertEquals(.5f, ScoreOverlay.pointTimelineSnapshot(prepared, 6_425).opacity, .0001f)
        assertEquals(0f, ScoreOverlay.pointTimelineSnapshot(prepared, 6_600).opacity, .0001f)
        assertEquals(
            listOf(
                ScorePointTimelineEntry("S2", ScoreTeamId.TEAM_1, 1),
                ScorePointTimelineEntry("S3", ScoreTeamId.TEAM_2, 1),
            ),
            ScoreOverlay.pointTimelineSnapshot(prepared, 9_250).points,
        )
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

        val longPoints = List(38) { index ->
            ScorePointTimelineEntry(
                "S${index + 1}",
                if (index % 2 == 0) ScoreTeamId.TEAM_1 else ScoreTeamId.TEAM_2,
                index / 2 + 1,
            )
        }
        val visiblePoints = ScoreOverlay.visiblePointTimelineEntries(1920, layout, longPoints)
        val timeline = ScoreOverlay.pointTimelineLayout(1920, 1080, layout, visiblePoints.size)
        assertEquals(layout.width.toFloat(), timeline.startX, .0001f)
        assertTrue(timeline.team1CenterY < timeline.team2CenterY)
        assertTrue(visiblePoints.size < longPoints.size)
        assertEquals("S${39 - visiblePoints.size}", visiblePoints.first().serveMarkerId)
        assertEquals("S38", visiblePoints.last().serveMarkerId)
        assertTrue(timeline.columnSpacing * visiblePoints.size <= 1920 - layout.width)
        assertTrue(timeline.circleRadius > 0)
        assertTrue(timeline.lineWidth >= 2)

        val shortTimeline = ScoreOverlay.pointTimelineLayout(1920, 1080, layout, 8)
        assertEquals(layout.height * .58f, shortTimeline.columnSpacing, .0001f)
        assertTrue(shortTimeline.circleRadius > layout.height * .17f)
    }

    @Test
    fun sourceTimeMappingUsesPresentationTimeLocalToEachRetainedClip() {
        assertEquals(42_000, scoreOverlaySourceTimestampMs(42_000, 0, 0))
        assertEquals(43_250, scoreOverlaySourceTimestampMs(42_000, 0, 1_250_999))
        assertEquals(90_000, scoreOverlaySourceTimestampMs(90_000, 8_000_000, 8_000_000))
        assertEquals(91_250, scoreOverlaySourceTimestampMs(90_000, 8_000_000, 9_250_999))
    }

    @Test
    fun displayGeometrySwapsOnlyQuarterTurnInputs() {
        assertEquals(1920 to 1080, scoreOverlayDisplaySize(1920, 1080, 0))
        assertEquals(1080 to 1920, scoreOverlayDisplaySize(1920, 1080, 90))
        assertEquals(1920 to 1080, scoreOverlayDisplaySize(1920, 1080, 180))
        assertEquals(1080 to 1920, scoreOverlayDisplaySize(1920, 1080, 270))
    }

    @Test
    fun exportSnapshotRoundTripFreezesScoreFiltersAndRallyGeometry() {
        val original = ScoreExportSnapshot(
            render = true,
            renderPointTimeline = false,
            scoreTracking = ScoreTracking(team1Name = "Falcons", team2Name = "Wolves"),
            ignoredIntervals = listOf(IgnoredSourceInterval("I1", 1_000, 2_000, "timeout")),
            excludedRallyIds = setOf("R002"),
            rallyRanges = listOf(ScoreRallyRange(3_000, 4_000, 2_500, 4_500)),
            mergedRanges = listOf(ScoreMergedRange(2_500, 4_500)),
        )

        val restored = ScoreExportSnapshotJson.decode(
            ScoreExportSnapshotJson.encode(original),
            10_000,
        )

        assertNotNull(restored)
        assertEquals(original.render, restored?.render)
        assertEquals(original.renderPointTimeline, restored?.renderPointTimeline)
        assertEquals(original.scoreTracking, restored?.scoreTracking)
        assertEquals(original.ignoredIntervals, restored?.ignoredIntervals)
        assertEquals(original.excludedRallyIds, restored?.excludedRallyIds)
        assertEquals(original.rallyRanges, restored?.rallyRanges)
        assertEquals(original.mergedRanges, restored?.mergedRanges)
    }
}
