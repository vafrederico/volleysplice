package com.volleycut.nativeanalysis

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ScoreReducerTest {
    @Test
    fun firstServeEstablishesServerAndLaterServesAwardPreviousRallies() {
        val tracking = tracking(
            serve("S1", 1_000, ServingSide.NEAR),
            serve("S2", 2_000, ServingSide.NEAR),
            serve("S3", 3_000, ServingSide.FAR),
        )
        val score = ScoreReducer.deriveAt(tracking)
        assertEquals(1, score.team1Score)
        assertEquals(1, score.team2Score)
        assertEquals(2, score.points.size)
        assertEquals(ScoreTeamId.TEAM_2, score.servingTeamId)
    }

    @Test
    fun reviewAndReplayRecordUnawardedPoints() {
        val tracking = tracking(
            serve("S1", 1_000, ServingSide.NEAR),
            serve("S2", 2_000, ServingSide.REVIEW),
            serve("S3", 3_000, ServingSide.FAR).copy(ignorePreviousPoint = true),
        )
        val score = ScoreReducer.deriveAt(tracking)
        assertEquals(0, score.team1Score)
        assertEquals(0, score.team2Score)
        assertEquals(1, score.reviewPointCount)
        assertEquals(1, score.ignoredPointCount)
    }

    @Test
    fun equalTimestampSideSwitchAppliesToServe() {
        val tracking = tracking(
            serve("S1", 1_000, ServingSide.NEAR),
            serve("S2", 2_000, ServingSide.NEAR),
        ).copy(sideSwitchMarkers = listOf(SideSwitchMarker("X1", 2_000)))
        val score = ScoreReducer.deriveAt(tracking)
        assertEquals(0, score.team1Score)
        assertEquals(1, score.team2Score)
    }

    @Test
    fun ignoredTimeAndExcludedRalliesFilterWithoutMutatingState() {
        val tracking = tracking(
            serve("S1", 1_000, ServingSide.NEAR, "R001"),
            serve("S2", 2_000, ServingSide.NEAR, "R002"),
            serve("S3", 3_000, ServingSide.NEAR),
        )
        val visible = ScoreReducer.visibleTracking(
            tracking,
            listOf(IgnoredSourceInterval("I1", 2_900, 3_100, "gap")),
            setOf("R002"),
        )
        assertEquals(listOf("S1"), visible.serveMarkers.map { it.id })
        assertEquals(3, tracking.serveMarkers.size)
    }

    @Test
    fun finalRallyIsUnawardedUntilManualNextServeExists() {
        val oneServe = tracking(serve("S1", 1_000, ServingSide.NEAR))
        assertEquals(0, ScoreReducer.deriveAt(oneServe).team1Score)
        val withManual = ScoreReducer.addServe(oneServe, 5_000, ServingSide.NEAR)
        assertEquals(1, ScoreReducer.deriveAt(withManual).team1Score)
    }

    @Test
    fun deadTimeKeepsPreviousServeAndLeadingPaddingUsesNextServeBoundary() {
        val tracking = tracking(
            serve("S1", 1_000, ServingSide.NEAR),
            serve("S2", 5_000, ServingSide.FAR),
        )
        val ranges = listOf(ScoreRallyRange(5_000, 8_000, 4_000, 9_000))
        assertEquals(3_000, ScoreReducer.scoreBoundaryTimestamp(3_000, ranges, tracking))
        assertEquals(5_000, ScoreReducer.scoreBoundaryTimestamp(4_500, ranges, tracking))
        assertEquals(6_000, ScoreReducer.scoreBoundaryTimestamp(6_000, ranges, tracking))
        assertEquals(8_500, ScoreReducer.scoreBoundaryTimestamp(8_500, ranges, tracking))
    }

    @Test
    fun mergedRallyFragmentsDoNotAdvanceInsideAnInternalGapWithoutAServe() {
        val tracking = tracking(
            serve("S1", 5_000, ServingSide.NEAR),
            serve("S2", 20_000, ServingSide.FAR),
        )
        val ranges = listOf(
            ScoreRallyRange(5_000, 12_000, 3_000, 14_000),
            ScoreRallyRange(20_000, 25_000, 16_000, 27_000),
        )
        val merged = listOf(ScoreMergedRange(3_000, 27_000))

        assertEquals(13_000, ScoreReducer.scoreBoundaryTimestamp(13_000, ranges, tracking, merged))
        assertEquals(15_000, ScoreReducer.scoreBoundaryTimestamp(15_000, ranges, tracking, merged))
        assertEquals(20_000, ScoreReducer.scoreBoundaryTimestamp(19_000, ranges, tracking, merged))
        assertEquals(20_000, ScoreReducer.scoreBoundaryTimestamp(20_000, ranges, tracking, merged))
    }

    @Test
    fun upcomingServeActivatesOnlyAfterEnteringItsLeadingPadding() {
        val tracking = tracking(
            serve("S1", 5_000, ServingSide.NEAR),
            serve("S2", 18_000, ServingSide.FAR),
            serve("S3", 30_000, ServingSide.NEAR),
        )
        val ranges = listOf(
            ScoreRallyRange(5_000, 12_000, 3_000, 14_000),
            ScoreRallyRange(20_000, 25_000, 16_000, 27_000),
        )

        assertEquals(
            15_000,
            ScoreReducer.scoreBoundaryTimestamp(
                15_000,
                ranges,
                tracking,
                listOf(ScoreMergedRange(3_000, 27_000)),
            ),
        )
        assertEquals(
            18_000,
            ScoreReducer.scoreBoundaryTimestamp(
                16_000,
                ranges,
                tracking,
                listOf(ScoreMergedRange(3_000, 27_000)),
            ),
        )
        assertEquals(
            19_000,
            ScoreReducer.scoreBoundaryTimestamp(
                19_000,
                ranges,
                tracking,
                listOf(ScoreMergedRange(3_000, 27_000)),
            ),
        )
    }

    @Test
    fun firstLeadingPaddingStopsLookingAheadAfterCrossingItsServeMarker() {
        val tracking = tracking(
            serve("S1", 2_500, ServingSide.NEAR),
            serve("S2", 20_000, ServingSide.FAR),
        )
        val ranges = listOf(ScoreRallyRange(5_000, 12_000, 3_000, 14_000))
        val merged = listOf(ScoreMergedRange(3_000, 14_000))

        assertEquals(3_000, ScoreReducer.scoreBoundaryTimestamp(3_000, ranges, tracking, merged))
        assertEquals(3_500, ScoreReducer.scoreBoundaryTimestamp(3_500, ranges, tracking, merged))
        assertEquals(4_500, ScoreReducer.scoreBoundaryTimestamp(4_500, ranges, tracking, merged))
    }

    @Test
    fun removingModelMarkerCreatesTombstoneAndReseedingDoesNotResurrectIt() {
        val output = output(ServingSideVerdict.NEAR)
        val seeded = ScoreReducer.seedModelMarkers(ScoreTracking(), output)
        val removed = ScoreReducer.removeServe(seeded, "serve-R001")
        val reseeded = ScoreReducer.seedModelMarkers(removed, output)
        assertTrue("serve-R001" in reseeded.removedModelMarkerIds)
        assertFalse(reseeded.serveMarkers.any { it.id == "serve-R001" })
    }

    @Test
    fun inferredSwitchesSeedEditableMarkersAndDeletionCreatesATombstone() {
        val output = SideSwitchOutput(
            rows = 1,
            features = DoubleArray(SIDE_SWITCH_FEATURE_COLUMNS),
            candidates = listOf(SideSwitchPrediction(
                "switch:boundary:R001:R002",
                3.0,
                .91,
                SideSwitchCandidateKind.ADJACENT_RALLY_BOUNDARY,
                listOf("R001", "R002"),
            )),
        )
        val seeded = ScoreReducer.seedModelMarkers(ScoreTracking(), null, output)
        val marker = seeded.sideSwitchMarkers.single()
        assertEquals("switch-switch:boundary:R001:R002", marker.id)
        assertEquals(ServeMarkerOrigin.MODEL, marker.origin)
        assertEquals(.91, marker.modelConfidence ?: 0.0, 0.0)
        val removed = ScoreReducer.removeSideSwitch(seeded, marker.id)
        val reseeded = ScoreReducer.seedModelMarkers(removed, null, output)
        assertTrue(marker.id in reseeded.removedModelMarkerIds)
        assertTrue(reseeded.sideSwitchMarkers.isEmpty())
    }

    @Test
    fun disabledSwitchInferenceRemovesModelMarkersButPreservesManualMarkers() {
        val tracking = ScoreTracking(sideSwitchMarkers = listOf(
            SideSwitchMarker("manual", 1_000),
            SideSwitchMarker(
                "switch-model", 2_000, origin = ServeMarkerOrigin.MODEL,
                modelEventId = "model",
            ),
        ))

        val seeded = ScoreReducer.seedModelMarkers(
            tracking, null, null, sideSwitchEnabled = false,
        )

        assertEquals(listOf("manual"), seeded.sideSwitchMarkers.map { it.id })
    }

    @Test
    fun ignoredIntervalsHideSwitchesButExcludedRalliesDoNot() {
        val tracking = ScoreTracking(sideSwitchMarkers = listOf(
            SideSwitchMarker("ignored", 1_500, rallyIds = listOf("R001", "R002")),
            SideSwitchMarker("excluded", 3_000, rallyIds = listOf("R003", "R004")),
            SideSwitchMarker("visible", 4_500, rallyIds = listOf("R005", "R006")),
        ))

        val visible = ScoreReducer.visibleTracking(
            tracking,
            listOf(IgnoredSourceInterval("ignored-1", 1_000, 2_000, "camera obstruction")),
            setOf("R004"),
        )

        assertEquals(listOf("excluded", "visible"), visible.sideSwitchMarkers.map { it.id })
        assertEquals(3, tracking.sideSwitchMarkers.size)
    }

    @Test
    fun deferredServingResultsDoNotOverrideTheCurrentScoreToggle() {
        val seeded = ScoreReducer.seedModelMarkers(
            ScoreTracking(enabled = false),
            output(ServingSideVerdict.NEAR),
        )

        assertFalse(seeded.enabled)
        assertTrue(seeded.serveMarkers.isNotEmpty())
    }

    @Test
    fun reseedingPreservesCorrectionsReplayFlagsManualMarkersAndCorrectedNotServe() {
        val initial = ScoreReducer.seedModelMarkers(ScoreTracking(), output(ServingSideVerdict.NEAR))
        val corrected = initial.copy(
            serveMarkers = initial.serveMarkers.map {
                it.copy(side = ServingSide.FAR, ignorePreviousPoint = true)
            } + serve("S001", 4_000, ServingSide.NEAR),
        )
        val notServe = output(ServingSideVerdict.NOT_SERVE)
        val reseeded = ScoreReducer.seedModelMarkers(corrected, notServe)
        val model = reseeded.serveMarkers.single { it.origin == ServeMarkerOrigin.MODEL }
        assertEquals(ServingSide.FAR, model.side)
        assertEquals(ServingSide.REVIEW, model.modelSide)
        assertTrue(model.ignorePreviousPoint)
        assertTrue(reseeded.serveMarkers.any { it.id == "S001" })
    }

    @Test
    fun reseedingCanonicalizesImportedModelMarkerIdsLikeBrowser() {
        val imported = ScoreTracking(serveMarkers = listOf(
            ServeMarker(
                id = "imported-model-id",
                timestampMs = 1_000,
                side = ServingSide.FAR,
                origin = ServeMarkerOrigin.MODEL,
                modelSide = ServingSide.NEAR,
                ignorePreviousPoint = true,
                rallyId = "R001",
            ),
        ))
        val reseeded = ScoreReducer.seedModelMarkers(imported, output(ServingSideVerdict.NEAR))
        val marker = reseeded.serveMarkers.single()
        assertEquals("serve-R001", marker.id)
        assertEquals(ServingSide.FAR, marker.side)
        assertEquals(ServingSide.NEAR, marker.modelSide)
        assertTrue(marker.ignorePreviousPoint)
    }

    @Test
    fun scoreWireRoundTripsAndVersionOneDraftMigratesToCurrentVersion() {
        val value = tracking(
            ServeMarker(
                "M1", 1_250, ServingSide.FAR, ServeMarkerOrigin.MANUAL,
                modelSide = ServingSide.NEAR, rallyId = "R001",
            ),
        ).copy(
            team1Name = "Falcons",
            sideSwitchMarkers = listOf(SideSwitchMarker("X1", 2_000)),
            removedModelMarkerIds = setOf("serve-R009"),
        )
        assertEquals(value, ScoreTrackingJson.decodeWire(ScoreTrackingJson.encodeWire(value), 5_000))

        val old = JSONObject().apply {
            put("version", 1)
            put("enabled", true)
            put("team1Name", "Team 1")
            put("team2Name", "Team 2")
            put("serveMarkers", JSONArray())
            put("sideSwitchMarkers", JSONArray())
        }
        val migrated = ScoreTrackingJson.decode(old, 5_000)
        assertEquals(SCORE_TRACKING_SCHEMA_VERSION, migrated?.version)
        assertTrue(migrated?.removedModelMarkerIds?.isEmpty() == true)
    }

    @Test
    fun scoreWireRejectsMissingRequiredFieldsAndInvalidOptionalModelSide() {
        val valid = ScoreTrackingJson.encodeWire(ScoreTracking())
        assertEquals(null, ScoreTrackingJson.decodeWire(
            org.json.JSONObject(valid.toString()).apply { remove("enabled") },
            5_000,
        ))
        val markerState = ScoreTracking(serveMarkers = listOf(
            serve("S001", 1_000, ServingSide.NEAR).copy(modelSide = ServingSide.NEAR),
        ))
        val invalidSide = ScoreTrackingJson.encodeWire(markerState).apply {
            getJSONArray("serveMarkers").getJSONObject(0).put("modelSide", "left")
        }
        assertEquals(null, ScoreTrackingJson.decodeWire(invalidSide, 5_000))
        val duplicateTombstones = ScoreTrackingJson.encodeWire(ScoreTracking()).apply {
            put("removedModelMarkerIds", JSONArray(listOf("serve-R001", "serve-R001")))
        }
        assertEquals(null, ScoreTrackingJson.decodeWire(duplicateTombstones, 5_000))
    }

    private fun tracking(vararg serves: ServeMarker) = ScoreTracking(serveMarkers = serves.toList())
    private fun serve(id: String, time: Long, side: ServingSide, rallyId: String? = null) =
        ServeMarker(id, time, side, ServeMarkerOrigin.MANUAL, rallyId = rallyId)

    private fun output(verdict: ServingSideVerdict) = ServingSideOutput(
        rows = 1,
        rawFeatures = DoubleArray(237),
        candidates = listOf(ServingSideCandidate(
            "R001", 1.0, 1.0, 2.0, ProductionEnsemble.BOTH_MODELS,
            .9, ServingSide.NEAR, verdict, ServingSideDecisionSource.SERVE_HEAD,
            emptyList(), evidence(), evidence(),
        )),
    )

    private fun evidence() = ServingSideHeadEvidence("model", .85, .9, 1.0, true, null)
}
