package com.volleycut.nativeanalysis

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class YouTubeChaptersTest {
    @Test
    fun mapsSourceRalliesOntoConcatenatedOutputTimeline() {
        val chapters = YouTubeChapters.build(
            listOf(
                FinalCutInterval(8_000, 17_000, listOf("R001")),
                FinalCutInterval(38_000, 48_000, listOf("R002")),
            ),
            listOf(cut("R001", 10_000, 15_000), cut("R002", 40_000, 46_000)),
            null,
            YouTubeChapters.defaultOptions(false, false),
        )
        assertEquals("0:02 Rally 1\n0:11 Rally 2", YouTubeChapters.text(chapters, false))
    }

    @Test
    fun defaultsToScoreAndServingTeamWhenServeMarkersExist() {
        val tracking = ScoreTracking(
            team1Name = "Falcons",
            team2Name = "Owls",
            serveMarkers = listOf(
                serve("S001", 1_100, ServingSide.NEAR, "R001"),
                serve("S002", 4_100, ServingSide.NEAR, "R002"),
            ),
        )
        val chapters = YouTubeChapters.build(
            listOf(FinalCutInterval(1_000, 5_000, listOf("R001", "R002"))),
            listOf(cut("R001", 1_000, 2_000), cut("R002", 4_000, 5_000)),
            tracking,
            YouTubeChapters.defaultOptions(true, false),
        )
        assertEquals(listOf("0–0 - Falcons serving", "1–0 - Falcons serving"), chapters.map { it.title })
    }

    @Test
    fun joinedCutsWithoutASeparatingServeProduceOneRallyChapter() {
        val intervals = listOf(FinalCutInterval(
            8_000,
            24_000,
            listOf("R001", "R002"),
            listOf(JoinedGap(15_000, 17_000)),
        ))
        val cuts = listOf(cut("R001", 8_000, 15_000), cut("R002", 17_000, 24_000))

        assertEquals(
            "0:00 Rally 1",
            YouTubeChapters.text(YouTubeChapters.build(
                intervals,
                cuts,
                null,
                YouTubeChapters.defaultOptions(false, false),
            ), false),
        )
        assertEquals(
            "0:09 Rally 2",
            YouTubeChapters.text(YouTubeChapters.build(
                intervals,
                cuts,
                ScoreTracking(serveMarkers = listOf(
                    serve("S002", 17_000, ServingSide.FAR, "R002"),
                )),
                YouTubeChapterOptions(true, false, false, false, false),
            ), false),
        )
    }

    @Test
    fun attachesRemovedSideSwitchToNextVisibleClipAndMergesSameSecond() {
        val chapters = YouTubeChapters.build(
            listOf(FinalCutInterval(10_000, 15_000, listOf("R001"))),
            listOf(cut("R001", 10_000, 12_000)),
            ScoreTracking(sideSwitchMarkers = listOf(SideSwitchMarker("X001", 8_000))),
            YouTubeChapters.defaultOptions(false, true),
        )
        assertEquals(1, chapters.size)
        assertEquals("Side switch 1", chapters.single().title)
    }

    @Test
    fun labelsRedoAndSanitizesFilename() {
        val tracking = ScoreTracking(serveMarkers = listOf(
            serve("S001", 1_000, ServingSide.NEAR, "R001"),
            serve("S002", 4_000, ServingSide.FAR, "R002").copy(ignorePreviousPoint = true),
        ))
        val chapters = YouTubeChapters.build(
            listOf(FinalCutInterval(0, 6_000, listOf("R001", "R002"))),
            listOf(cut("R001", 1_000, 2_000), cut("R002", 4_000, 5_000)),
            tracking,
            YouTubeChapterOptions(true, false, false, false, false),
        )
        assertTrue("Re-do" in chapters.first().title)
        assertEquals("Friday-Night-youtube-chapters.txt", YouTubeChapters.filename("Friday Night!.mp4"))
    }

    @Test
    fun tracksRedoScoreAndSideSwitchOnFinalVideoTimeline() {
        val tracking = ScoreTracking(
            serveMarkers = listOf(
                ServeMarker("S001", 3_623, ServingSide.NEAR, ServeMarkerOrigin.MANUAL),
                ServeMarker("S002", 28_000, ServingSide.FAR, ServeMarkerOrigin.MANUAL)
                    .copy(ignorePreviousPoint = true),
            ),
            sideSwitchMarkers = listOf(SideSwitchMarker("X001", 28_000)),
        )
        val chapters = YouTubeChapters.build(
            listOf(
                FinalCutInterval(3_000, 17_000, listOf("R001")),
                FinalCutInterval(28_000, 44_000, listOf("R002")),
                FinalCutInterval(58_000, 74_000, listOf("R003")),
                FinalCutInterval(88_000, 107_000, listOf("R004")),
            ),
            listOf(
                EditableCut("R001", 5_000, 15_000, 3_000, 17_000, 1f, true, CutOrigin.INFERRED),
                EditableCut("R002", 30_000, 42_000, 28_000, 44_000, 1f, true, CutOrigin.INFERRED),
                EditableCut("R003", 60_000, 72_000, 58_000, 74_000, 1f, true, CutOrigin.INFERRED),
                EditableCut("R004", 90_000, 105_000, 88_000, 107_000, 1f, true, CutOrigin.INFERRED),
            ),
            tracking,
            YouTubeChapters.defaultOptions(true, true),
        )

        assertEquals(
            "0:00 0–0 - Team 1 serving - Re-do\n" +
                "0:14 0–0 - Team 1 serving / Side switch 1",
            YouTubeChapters.text(chapters, false),
        )
    }

    @Test
    fun formatsLongTimestampsLikeYouTube() {
        assertEquals("0:00", YouTubeChapters.formatTimestamp(999))
        assertEquals("1:05", YouTubeChapters.formatTimestamp(65_999))
        assertEquals("1:01:01", YouTubeChapters.formatTimestamp(3_661_999))
    }

    @Test
    fun creditDefaultsOnAndCanBeOmittedWithoutCreatingAnEmptyExport() {
        assertTrue(YouTubeChapters.defaultOptions(false, false).includeCredit)
        assertTrue(YouTubeChapters.defaultOptions(true, true).includeCredit)
        val chapters = listOf(YouTubeChapter(YouTubeChapter.Kind.RALLY, 0, 0, "Rally 1"))
        assertEquals("Edited with https://volleysplice.com\n\n0:00 Rally 1", YouTubeChapters.text(chapters))
        assertEquals("0:00 Rally 1", YouTubeChapters.text(chapters, false))
        assertEquals("", YouTubeChapters.text(emptyList()))
    }

    @Test
    fun enabledTrackingOmitsUnmarkedRalliesAndPreservesOutputTimeAndNumber() {
        val cuts = listOf(cut("R1", 10_000, 20_000), cut("R2", 40_000, 50_000))
        val intervals = cuts.map { FinalCutInterval(it.keepStartMs, it.keepEndMs, listOf(it.id)) }
        val options = YouTubeChapters.defaultOptions(false, false)
        val tracking = ScoreTracking(serveMarkers = listOf(serve("S2", 42_000, ServingSide.FAR, "R2")))
        fun build(score: ScoreTracking?) = YouTubeChapters.build(intervals, cuts, score, options)
        assertEquals("0:10 Rally 2", YouTubeChapters.text(build(tracking), false))
        assertTrue(build(ScoreTracking()).isEmpty())
        assertEquals(2, build(tracking.copy(enabled = false)).size)
        assertEquals(2, build(null).size)
    }

    private fun cut(id: String, startMs: Long, endMs: Long) = EditableCut(
        id, startMs, endMs, startMs, endMs, 1f, true, CutOrigin.INFERRED,
    )

    private fun serve(id: String, timestampMs: Long, side: ServingSide, rallyId: String) = ServeMarker(
        id, timestampMs, side, ServeMarkerOrigin.MANUAL, rallyId = rallyId,
    )
}
