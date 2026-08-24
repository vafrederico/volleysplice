package com.volleycut.nativeanalysis

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.roundToInt

internal const val SCORE_TRACKING_SCHEMA_VERSION = 3

internal enum class ScoreTeamId(val wireName: String) {
    TEAM_1("team-1"), TEAM_2("team-2");

    companion object {
        fun fromWireName(value: String) = entries.firstOrNull { it.wireName == value }
    }
}

internal enum class ServeMarkerOrigin(val wireName: String) {
    MODEL("model"), MANUAL("manual");

    companion object {
        fun fromWireName(value: String) = entries.firstOrNull { it.wireName == value }
    }
}

internal data class ServeMarker(
    val id: String,
    val timestampMs: Long,
    val side: ServingSide,
    val origin: ServeMarkerOrigin,
    val modelSide: ServingSide? = null,
    val ignorePreviousPoint: Boolean = false,
    val rallyId: String? = null,
)

internal data class SideSwitchMarker(
    val id: String,
    val timestampMs: Long,
    val origin: ServeMarkerOrigin = ServeMarkerOrigin.MANUAL,
    val modelConfidence: Double? = null,
    val modelEventId: String? = null,
    val rallyIds: List<String> = emptyList(),
)

internal data class ScoreTracking(
    val version: Int = SCORE_TRACKING_SCHEMA_VERSION,
    val enabled: Boolean = true,
    val team1Name: String = "Team 1",
    val team2Name: String = "Team 2",
    val serveMarkers: List<ServeMarker> = emptyList(),
    val sideSwitchMarkers: List<SideSwitchMarker> = emptyList(),
    val removedModelMarkerIds: Set<String> = emptySet(),
)

internal enum class ScorePointStatus(val wireName: String) {
    COUNTED("counted"), IGNORED("ignored"), REVIEW("review")
}

internal data class DerivedScorePoint(
    val serveMarkerId: String,
    val timestampMs: Long,
    val servingSide: ServingSide,
    val winnerTeamId: ScoreTeamId?,
    val status: ScorePointStatus,
    val team1ScoreAfter: Int,
    val team2ScoreAfter: Int,
)

internal data class DerivedScore(
    val team1Score: Int,
    val team2Score: Int,
    val servingTeamId: ScoreTeamId?,
    val servingSide: ServingSide?,
    val points: List<DerivedScorePoint>,
    val ignoredPointCount: Int,
    val reviewPointCount: Int,
)

internal data class ScoreRallyRange(
    val coreStartMs: Long,
    val coreEndMs: Long,
    val keepStartMs: Long,
    val keepEndMs: Long,
)

internal data class ScoreMergedRange(val startMs: Long, val endMs: Long)

internal object ScoreReducer {
    private val serveOrder = compareBy<ServeMarker> { it.timestampMs }.thenBy { it.id }
    private val switchOrder = compareBy<SideSwitchMarker> { it.timestampMs }.thenBy { it.id }

    fun seedModelMarkers(
        current: ScoreTracking,
        output: ServingSideOutput?,
        sideSwitchOutput: SideSwitchOutput? = null,
        sideSwitchEnabled: Boolean = true,
    ): ScoreTracking {
        if (output == null && sideSwitchOutput == null && sideSwitchEnabled) return current
        val existingByRally = current.serveMarkers
            .filter { it.rallyId != null }
            .associateBy { it.rallyId }
        var tracking = if (output != null) current.copy(
            serveMarkers = current.serveMarkers.filter { it.origin == ServeMarkerOrigin.MANUAL },
        ) else current
        output?.candidates?.forEach { candidate ->
            val id = "serve-${candidate.id}"
            val existing = existingByRally[candidate.id]
            val wasCorrected = existing?.modelSide != null && existing.side != existing.modelSide
            if (candidate.verdict == ServingSideVerdict.NOT_SERVE && !wasCorrected) {
                return@forEach
            }
            val modelSide = when (candidate.verdict) {
                ServingSideVerdict.NEAR, ServingSideVerdict.FAR -> candidate.side
                ServingSideVerdict.REVIEW -> ServingSide.REVIEW
                ServingSideVerdict.NOT_SERVE -> ServingSide.REVIEW
            }
            val used = tracking.serveMarkers.any { it.id == id } ||
                tracking.sideSwitchMarkers.any { it.id == id } ||
                id in tracking.removedModelMarkerIds
            if (used) return@forEach
            tracking = tracking.copy(serveMarkers = (tracking.serveMarkers + ServeMarker(
                id = id,
                timestampMs = secondsToMs(candidate.anchor),
                side = if (wasCorrected) checkNotNull(existing).side else modelSide,
                origin = ServeMarkerOrigin.MODEL,
                modelSide = modelSide,
                ignorePreviousPoint = existing?.ignorePreviousPoint == true,
                rallyId = candidate.id,
            )).sortedWith(serveOrder))
        }
        if (!sideSwitchEnabled) {
            tracking = tracking.copy(
                sideSwitchMarkers = tracking.sideSwitchMarkers.filter {
                    it.origin == ServeMarkerOrigin.MANUAL
                },
            )
        } else if (sideSwitchOutput != null) {
            tracking = tracking.copy(
                sideSwitchMarkers = tracking.sideSwitchMarkers.filter {
                    it.origin == ServeMarkerOrigin.MANUAL
                },
            )
            sideSwitchOutput.candidates.forEach { candidate ->
                val id = "switch-${candidate.id}"
                val used = tracking.serveMarkers.any { it.id == id } ||
                    tracking.sideSwitchMarkers.any { it.id == id } ||
                    id in tracking.removedModelMarkerIds
                if (!used) {
                    tracking = tracking.copy(
                        sideSwitchMarkers = (tracking.sideSwitchMarkers + SideSwitchMarker(
                            id = id,
                            timestampMs = secondsToMs(candidate.timestamp),
                            origin = ServeMarkerOrigin.MODEL,
                            modelConfidence = candidate.probability,
                            modelEventId = candidate.id,
                            rallyIds = candidate.sourceRangeIds.distinct(),
                        )).sortedWith(switchOrder),
                    )
                }
            }
        }
        return tracking
    }

    fun visibleTracking(
        tracking: ScoreTracking,
        ignoredIntervals: List<IgnoredSourceInterval>,
        excludedRallyIds: Set<String>,
    ): ScoreTracking = tracking.copy(
        serveMarkers = tracking.serveMarkers.filter { marker ->
            ignoredIntervals.none { marker.timestampMs >= it.startMs && marker.timestampMs < it.endMs } &&
                (marker.rallyId == null || marker.rallyId !in excludedRallyIds)
        },
        sideSwitchMarkers = tracking.sideSwitchMarkers.filter { marker ->
            ignoredIntervals.none { marker.timestampMs >= it.startMs && marker.timestampMs < it.endMs } &&
                marker.rallyIds.none { it in excludedRallyIds }
        },
    )

    fun teamForServingSide(side: ServingSide, sideSwitchCount: Int): ScoreTeamId? {
        if (side == ServingSide.REVIEW) return null
        val switched = abs(sideSwitchCount) % 2 == 1
        return when (side) {
            ServingSide.NEAR -> if (switched) ScoreTeamId.TEAM_2 else ScoreTeamId.TEAM_1
            ServingSide.FAR -> if (switched) ScoreTeamId.TEAM_1 else ScoreTeamId.TEAM_2
            ServingSide.REVIEW -> null
        }
    }

    fun deriveAt(tracking: ScoreTracking, sourceTimestampMs: Long = Long.MAX_VALUE): DerivedScore {
        val serves = tracking.serveMarkers.sortedWith(serveOrder)
            .filter { it.timestampMs <= sourceTimestampMs.coerceAtLeast(0) }
        val switches = tracking.sideSwitchMarkers.sortedWith(switchOrder)
        var switchIndex = 0
        var team1Score = 0
        var team2Score = 0
        var servingTeam: ScoreTeamId? = null
        var servingSide: ServingSide? = null
        var ignored = 0
        var review = 0
        val points = mutableListOf<DerivedScorePoint>()
        serves.forEachIndexed { index, serve ->
            while (switchIndex < switches.size && switches[switchIndex].timestampMs <= serve.timestampMs) {
                switchIndex++
            }
            servingSide = serve.side
            servingTeam = teamForServingSide(serve.side, switchIndex)
            if (index == 0) return@forEachIndexed
            val status = when {
                serve.ignorePreviousPoint -> ScorePointStatus.IGNORED.also { ignored++ }
                servingTeam == null -> ScorePointStatus.REVIEW.also { review++ }
                else -> ScorePointStatus.COUNTED.also {
                    if (servingTeam == ScoreTeamId.TEAM_1) team1Score++ else team2Score++
                }
            }
            points += DerivedScorePoint(
                serve.id, serve.timestampMs, serve.side, servingTeam, status,
                team1Score, team2Score,
            )
        }
        return DerivedScore(
            team1Score, team2Score, servingTeam, servingSide, points, ignored, review,
        )
    }

    fun scoreBoundaryTimestamp(
        playbackTimestampMs: Long,
        rallyRanges: List<ScoreRallyRange>,
        tracking: ScoreTracking,
        mergedRanges: List<ScoreMergedRange> = emptyList(),
    ): Long {
        val timestamp = playbackTimestampMs.coerceAtLeast(0)
        val serves = tracking.serveMarkers.sortedWith(serveOrder)
        fun nextServeTimestamp() = serves.firstOrNull { it.timestampMs >= timestamp }
            ?.timestampMs ?: timestamp
        fun paddingBoundaryTimestamp(startMs: Long, endMs: Long): Long {
            val paddingServes = serves.filter {
                it.timestampMs >= startMs && it.timestampMs < endMs
            }
            if (paddingServes.isEmpty()) return nextServeTimestamp()
            return if (paddingServes.any { it.timestampMs <= timestamp }) {
                timestamp
            } else paddingServes.first().timestampMs
        }
        val mergedRange = mergedRanges.firstOrNull {
            timestamp >= it.startMs && timestamp < it.endMs
        }
        if (mergedRange != null) {
            val mergedRallies = rallyRanges.filter {
                it.keepStartMs < mergedRange.endMs && mergedRange.startMs < it.keepEndMs
            }.sortedWith(compareBy<ScoreRallyRange> { it.coreStartMs }.thenBy { it.coreEndMs })
            if (mergedRallies.isEmpty()) return timestamp
            if (timestamp < mergedRallies.first().coreStartMs) {
                val previousMergedEnd = mergedRanges.fold(0L) { latest, range ->
                    if (range.endMs <= mergedRange.startMs) maxOf(latest, range.endMs) else latest
                }
                return paddingBoundaryTimestamp(
                    previousMergedEnd,
                    mergedRallies.first().coreStartMs,
                )
            }
            for (index in 1 until mergedRallies.size) {
                val previous = mergedRallies[index - 1]
                val next = mergedRallies[index]
                if (timestamp < previous.coreEndMs) return timestamp
                if (timestamp < next.coreStartMs) {
                    val bridgeServes = serves.filter {
                        it.timestampMs >= previous.coreEndMs && it.timestampMs < next.coreStartMs
                    }
                    if (bridgeServes.isEmpty()) return timestamp
                    return if (bridgeServes.any { it.timestampMs <= timestamp }) {
                        timestamp
                    } else bridgeServes.first().timestampMs
                }
            }
            return timestamp
        }
        val insideRally = rallyRanges.any { timestamp >= it.keepStartMs && timestamp < it.keepEndMs }
        val leadingPaddingRange = rallyRanges.firstOrNull {
            timestamp >= it.keepStartMs && timestamp < it.coreStartMs
        }
        if (insideRally && leadingPaddingRange == null) return timestamp
        if (leadingPaddingRange != null) {
            return paddingBoundaryTimestamp(
                leadingPaddingRange.keepStartMs,
                leadingPaddingRange.coreStartMs,
            )
        }
        return nextServeTimestamp()
    }

    fun nextMarkerId(prefix: String, tracking: ScoreTracking): String {
        val used = buildSet {
            addAll(tracking.serveMarkers.map { it.id })
            addAll(tracking.sideSwitchMarkers.map { it.id })
            addAll(tracking.removedModelMarkerIds)
        }
        for (index in 1 until 10_000) {
            val candidate = prefix + index.toString().padStart(3, '0')
            if (candidate !in used) return candidate
        }
        return prefix + System.currentTimeMillis()
    }

    fun addServe(tracking: ScoreTracking, timestampMs: Long, side: ServingSide): ScoreTracking {
        if (timestampMs < 0) return tracking
        val marker = ServeMarker(
            nextMarkerId("S", tracking), timestampMs, side, ServeMarkerOrigin.MANUAL,
        )
        return tracking.copy(serveMarkers = (tracking.serveMarkers + marker).sortedWith(serveOrder))
    }

    fun addSideSwitch(tracking: ScoreTracking, timestampMs: Long): ScoreTracking {
        if (timestampMs < 0) return tracking
        val marker = SideSwitchMarker(nextMarkerId("X", tracking), timestampMs)
        return tracking.copy(
            sideSwitchMarkers = (tracking.sideSwitchMarkers + marker).sortedWith(switchOrder),
        )
    }

    fun removeServe(tracking: ScoreTracking, markerId: String): ScoreTracking {
        val removed = tracking.serveMarkers.firstOrNull { it.id == markerId }
        return tracking.copy(
            serveMarkers = tracking.serveMarkers.filterNot { it.id == markerId },
            removedModelMarkerIds = if (removed?.origin == ServeMarkerOrigin.MODEL) {
                tracking.removedModelMarkerIds + markerId
            } else tracking.removedModelMarkerIds,
        )
    }

    fun removeSideSwitch(tracking: ScoreTracking, markerId: String): ScoreTracking {
        val removed = tracking.sideSwitchMarkers.firstOrNull { it.id == markerId }
        return tracking.copy(
            sideSwitchMarkers = tracking.sideSwitchMarkers.filterNot { it.id == markerId },
            removedModelMarkerIds = if (removed?.origin == ServeMarkerOrigin.MODEL) {
                tracking.removedModelMarkerIds + markerId
            } else tracking.removedModelMarkerIds,
        )
    }
}

internal data class PreparedScoreOverlay(
    val tracking: ScoreTracking,
    val rallyRanges: List<ScoreRallyRange>,
    val mergedRanges: List<ScoreMergedRange>,
)

internal data class ScoreOverlaySnapshot(
    val team1Name: String,
    val team1Score: Int,
    val team1ScoreLabel: String,
    val team2Name: String,
    val team2Score: Int,
    val team2ScoreLabel: String,
    val servingTeamId: ScoreTeamId? = null,
)

internal data class ScoreOverlayLayout(
    val width: Int,
    val height: Int,
    val team1Width: Int,
    val team2Width: Int,
    val scoreWidth: Int,
    val borderWidth: Int,
    val radius: Int,
    val fontSize: Int,
    val horizontalPadding: Int,
)

internal object ScoreOverlay {
    const val BORDER_COLOR = 0xff000000
    const val TEAM_1_COLOR = 0xffd9342b
    const val TEAM_2_COLOR = 0xff2367c9
    const val TEAM_TEXT_COLOR = 0xffffffff
    const val SCORE_BACKGROUND_COLOR = 0xffffffff
    const val SCORE_TEXT_COLOR = 0xff000000

    fun prepare(
        scoreTracking: ScoreTracking,
        ignoredIntervals: List<IgnoredSourceInterval>,
        excludedRallyIds: Set<String>,
        rallyRanges: List<ScoreRallyRange>,
        mergedRanges: List<ScoreMergedRange> = emptyList(),
    ) = PreparedScoreOverlay(
        ScoreReducer.visibleTracking(scoreTracking, ignoredIntervals, excludedRallyIds),
        rallyRanges,
        mergedRanges,
    )

    fun snapshot(prepared: PreparedScoreOverlay, sourceTimestampMs: Long): ScoreOverlaySnapshot {
        val boundary = ScoreReducer.scoreBoundaryTimestamp(
            sourceTimestampMs, prepared.rallyRanges, prepared.tracking, prepared.mergedRanges,
        )
        val score = ScoreReducer.deriveAt(prepared.tracking, boundary)
        return ScoreOverlaySnapshot(
            prepared.tracking.team1Name, score.team1Score, formatScore(score.team1Score),
            prepared.tracking.team2Name, score.team2Score, formatScore(score.team2Score),
            score.servingTeamId,
        )
    }

    fun formatScore(score: Int) = max(0, score).toString().padStart(2, '0')

    fun formatTeamLabel(name: String, isServing: Boolean) = if (isServing) "$name 🏐" else name

    fun layout(
        videoWidth: Int,
        videoHeight: Int,
        snapshot: ScoreOverlaySnapshot,
        measureText: (String, Int) -> Float,
    ): ScoreOverlayLayout {
        val shortestEdge = max(1, minOf(videoWidth, videoHeight))
        val height = (shortestEdge * 0.064).coerceIn(36.0, 76.0).roundToInt()
        val borderWidth = if (shortestEdge >= 720) 2 else 1
        val fontSize = (height * 0.39).roundToInt()
        val horizontalPadding = (height * 0.24).roundToInt()
        val scoreWidth = (height * 1.3).roundToInt()
        val minimumTeamWidth = (height * 2.25).roundToInt()
        fun measured(name: String, teamId: ScoreTeamId) = max(
            minimumTeamWidth,
            ceil(measureText(formatTeamLabel(name, snapshot.servingTeamId == teamId), fontSize)).toInt() +
                horizontalPadding * 2,
        )
        val desiredTeam1Width = measured(snapshot.team1Name, ScoreTeamId.TEAM_1)
        val desiredTeam2Width = measured(snapshot.team2Name, ScoreTeamId.TEAM_2)
        val maximumOverlayWidth = max(
            minimumTeamWidth * 2 + scoreWidth * 2,
            floor(videoWidth * 0.96).toInt(),
        )
        val availableTeamWidth = maximumOverlayWidth - scoreWidth * 2
        var team1Width = desiredTeam1Width
        var team2Width = desiredTeam2Width
        if (desiredTeam1Width + desiredTeam2Width > availableTeamWidth) {
            val flexible1 = desiredTeam1Width - minimumTeamWidth
            val flexible2 = desiredTeam2Width - minimumTeamWidth
            val flexible = flexible1 + flexible2
            val availableFlexible = max(0, availableTeamWidth - minimumTeamWidth * 2)
            val scale = if (flexible > 0) minOf(1.0, availableFlexible.toDouble() / flexible) else 0.0
            team1Width = (minimumTeamWidth + flexible1 * scale).roundToInt()
            team2Width = (minimumTeamWidth + flexible2 * scale).roundToInt()
        }
        return ScoreOverlayLayout(
            team1Width + scoreWidth + team2Width + scoreWidth,
            height, team1Width, team2Width, scoreWidth, borderWidth,
            (height * 0.24).roundToInt(), fontSize, horizontalPadding,
        )
    }
}

internal fun scoreOverlayDisplaySize(width: Int, height: Int, rotation: Int): Pair<Int, Int> =
    if (Math.floorMod(rotation, 180) == 90) height to width else width to height
