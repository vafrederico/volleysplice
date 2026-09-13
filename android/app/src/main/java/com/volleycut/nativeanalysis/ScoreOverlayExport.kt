package com.volleycut.nativeanalysis

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PorterDuff
import androidx.annotation.OptIn
import androidx.media3.common.util.UnstableApi
import androidx.media3.effect.CanvasOverlay
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.roundToInt

internal data class ScoreExportSnapshot(
    val render: Boolean,
    val renderPointTimeline: Boolean = true,
    val fadeScoreOverlay: Boolean = false,
    val scoreTracking: ScoreTracking,
    val ignoredIntervals: List<IgnoredSourceInterval>,
    val excludedRallyIds: Set<String>,
    val rallyRanges: List<ScoreRallyRange>,
    val mergedRanges: List<ScoreMergedRange> = emptyList(),
) {
    val prepared: PreparedScoreOverlay by lazy {
        ScoreOverlay.prepare(
            scoreTracking,
            ignoredIntervals,
            excludedRallyIds,
            rallyRanges,
            mergedRanges,
        )
    }
}

internal fun scoreOverlaySourceTimestampMs(
    sourceStartMs: Long,
    compositionStartUs: Long,
    presentationTimeUs: Long,
): Long = sourceStartMs + ((presentationTimeUs - compositionStartUs) / 1_000L).coerceAtLeast(0)

internal object ScoreExportSnapshotJson {
    fun encode(value: ScoreExportSnapshot) = JSONObject().apply {
        put("render", value.render)
        put("renderPointTimeline", value.renderPointTimeline)
        put("fadeScoreOverlay", value.fadeScoreOverlay)
        put("scoreTracking", ScoreTrackingJson.encode(value.scoreTracking))
        put("ignoredIntervals", JSONArray().apply {
            value.ignoredIntervals.forEach { put(JSONObject().apply {
                put("id", it.id)
                put("startMs", it.startMs)
                put("endMs", it.endMs)
                put("reason", it.reason)
            }) }
        })
        put("excludedRallyIds", JSONArray(value.excludedRallyIds.sorted()))
        put("rallyRanges", JSONArray().apply {
            value.rallyRanges.forEach { put(JSONObject().apply {
                put("coreStartMs", it.coreStartMs)
                put("coreEndMs", it.coreEndMs)
                put("keepStartMs", it.keepStartMs)
                put("keepEndMs", it.keepEndMs)
            }) }
        })
        put("mergedRanges", JSONArray().apply {
            value.mergedRanges.forEach { put(JSONObject().apply {
                put("startMs", it.startMs)
                put("endMs", it.endMs)
            }) }
        })
    }.toString()

    fun decode(value: String, durationMs: Long): ScoreExportSnapshot? = runCatching {
        val json = JSONObject(value)
        val tracking = ScoreTrackingJson.decode(json.getJSONObject("scoreTracking"), durationMs)
            ?: return null
        val ignored = json.getJSONArray("ignoredIntervals")
        val excluded = json.getJSONArray("excludedRallyIds")
        val ranges = json.getJSONArray("rallyRanges")
        val merged = json.optJSONArray("mergedRanges") ?: JSONArray()
        ScoreExportSnapshot(
            render = json.optBoolean("render") && tracking.enabled,
            renderPointTimeline = json.optBoolean("renderPointTimeline", true),
            fadeScoreOverlay = json.optBoolean("fadeScoreOverlay", false),
            scoreTracking = tracking,
            ignoredIntervals = List(ignored.length()) { index -> ignored.getJSONObject(index).let {
                IgnoredSourceInterval(
                    it.getString("id"), it.getLong("startMs"), it.getLong("endMs"),
                    it.getString("reason"),
                )
            } },
            excludedRallyIds = buildSet { repeat(excluded.length()) { add(excluded.getString(it)) } },
            rallyRanges = List(ranges.length()) { index -> ranges.getJSONObject(index).let {
                ScoreRallyRange(
                    it.getLong("coreStartMs"), it.getLong("coreEndMs"),
                    it.getLong("keepStartMs"), it.getLong("keepEndMs"),
                )
            } },
            mergedRanges = List(merged.length()) { index -> merged.getJSONObject(index).let {
                ScoreMergedRange(it.getLong("startMs"), it.getLong("endMs"))
            } },
        )
    }.getOrNull()
}

@OptIn(UnstableApi::class)
internal class ScoreCanvasOverlay(
    private val snapshot: ScoreExportSnapshot,
    private val sourceStartMs: Long,
    private val compositionStartUs: Long,
) : CanvasOverlay(true) {
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)

    internal fun sourceTimestampMs(presentationTimeUs: Long): Long =
        scoreOverlaySourceTimestampMs(sourceStartMs, compositionStartUs, presentationTimeUs)

    override fun onDraw(canvas: Canvas, presentationTimeUs: Long) {
        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
        if (!snapshot.render) return
        val sourceTimeMs = sourceTimestampMs(presentationTimeUs)
        val score = ScoreOverlay.snapshot(snapshot.prepared, sourceTimeMs)
        val timeline = ScoreOverlay.pointTimelineSnapshot(snapshot.prepared, sourceTimeMs)
        paint.typeface = android.graphics.Typeface.create(android.graphics.Typeface.DEFAULT, android.graphics.Typeface.BOLD)
        val layout = ScoreOverlay.layout(canvas.width, canvas.height, score) { text, fontSize ->
            paint.textSize = fontSize.toFloat()
            paint.measureText(text)
        }
        val inset = layout.borderWidth / 2f
        val scoreLayer = canvas.saveLayerAlpha(
            0f, 0f, layout.width.toFloat(), layout.height.toFloat(),
            ((if (snapshot.fadeScoreOverlay) timeline.opacity else 1f) * 255).roundToInt(),
        )
        val path = bottomRightRoundedPath(
            layout.width.toFloat(), layout.height.toFloat(), layout.radius.toFloat(), inset,
        )
        canvas.save()
        canvas.clipPath(path)
        var left = 0f
        drawCell(canvas, left, layout.team1Width, layout.height, ScoreOverlay.TEAM_1_COLOR)
        drawText(
            canvas,
            ScoreOverlay.formatTeamLabel(
                score.team1Name,
                score.servingTeamId == ScoreTeamId.TEAM_1,
            ),
            left,
            layout.team1Width,
            layout,
            Color.WHITE,
        )
        left += layout.team1Width
        drawCell(canvas, left, layout.scoreWidth, layout.height, ScoreOverlay.SCORE_BACKGROUND_COLOR)
        drawText(canvas, score.team1ScoreLabel, left, layout.scoreWidth, layout, Color.BLACK)
        left += layout.scoreWidth
        drawCell(canvas, left, layout.team2Width, layout.height, ScoreOverlay.TEAM_2_COLOR)
        drawText(
            canvas,
            ScoreOverlay.formatTeamLabel(
                score.team2Name,
                score.servingTeamId == ScoreTeamId.TEAM_2,
            ),
            left,
            layout.team2Width,
            layout,
            Color.WHITE,
        )
        left += layout.team2Width
        drawCell(canvas, left, layout.scoreWidth, layout.height, ScoreOverlay.SCORE_BACKGROUND_COLOR)
        drawText(canvas, score.team2ScoreLabel, left, layout.scoreWidth, layout, Color.BLACK)
        canvas.restore()
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = layout.borderWidth.toFloat()
        paint.color = Color.BLACK
        var divider = layout.team1Width.toFloat()
        canvas.drawLine(divider, 0f, divider, layout.height.toFloat(), paint)
        divider += layout.scoreWidth
        canvas.drawLine(divider, 0f, divider, layout.height.toFloat(), paint)
        divider += layout.team2Width
        canvas.drawLine(divider, 0f, divider, layout.height.toFloat(), paint)
        canvas.drawPath(path, paint)
        paint.style = Paint.Style.FILL
        canvas.restoreToCount(scoreLayer)
        if (snapshot.renderPointTimeline) {
            drawPointTimeline(canvas, layout, timeline)
        }
    }

    private fun drawPointTimeline(
        canvas: Canvas,
        scoreLayout: ScoreOverlayLayout,
        timeline: ScorePointTimelineSnapshot,
    ) {
        if (timeline.points.isEmpty() || timeline.opacity <= 0f) return
        val visiblePoints = ScoreOverlay.visiblePointTimelineEntries(
            canvas.width,
            scoreLayout,
            timeline.points,
        )
        if (visiblePoints.isEmpty()) return
        val layout = ScoreOverlay.pointTimelineLayout(
            canvas.width,
            canvas.height,
            scoreLayout,
            visiblePoints.size,
        )
        if (layout.columnSpacing <= 0f || layout.circleRadius <= 0f) return
        paint.alpha = (timeline.opacity * 255).roundToInt().coerceIn(0, 255)
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = layout.lineWidth
        listOf(
            Triple(ScoreTeamId.TEAM_1, layout.team1CenterY, ScoreOverlay.TEAM_1_COLOR),
            Triple(ScoreTeamId.TEAM_2, layout.team2CenterY, ScoreOverlay.TEAM_2_COLOR),
        ).forEach { (teamId, y, color) ->
            val lastPointIndex = visiblePoints.indexOfLast { it.winnerTeamId == teamId }
            if (lastPointIndex < 0) return@forEach
            val lastX = layout.startX + layout.columnSpacing * (lastPointIndex + .5f)
            paint.color = color.toInt()
            canvas.drawLine(layout.startX, y, lastX, y, paint)
        }
        paint.textAlign = Paint.Align.CENTER
        paint.textSize = layout.fontSize
        visiblePoints.forEachIndexed { index, point ->
            val x = layout.startX + layout.columnSpacing * (index + .5f)
            val y = if (point.winnerTeamId == ScoreTeamId.TEAM_1) {
                layout.team1CenterY
            } else layout.team2CenterY
            paint.style = Paint.Style.FILL
            paint.color = if (point.winnerTeamId == ScoreTeamId.TEAM_1) {
                ScoreOverlay.TEAM_1_COLOR.toInt()
            } else ScoreOverlay.TEAM_2_COLOR.toInt()
            canvas.drawCircle(x, y, layout.circleRadius, paint)
            paint.style = Paint.Style.STROKE
            paint.strokeWidth = layout.lineWidth
            paint.color = ScoreOverlay.BORDER_COLOR.toInt()
            canvas.drawCircle(x, y, layout.circleRadius, paint)
            paint.style = Paint.Style.FILL
            paint.color = ScoreOverlay.POINT_TEXT_COLOR.toInt()
            val baseline = y - (paint.ascent() + paint.descent()) / 2f
            canvas.drawText(point.teamPointNumber.toString(), x, baseline, paint)
        }
        paint.alpha = 255
    }

    private fun drawCell(canvas: Canvas, left: Float, width: Int, height: Int, color: Long) {
        paint.style = Paint.Style.FILL
        paint.color = color.toInt()
        canvas.drawRect(left, 0f, left + width, height.toFloat(), paint)
    }

    private fun drawText(
        canvas: Canvas,
        raw: String,
        left: Float,
        width: Int,
        layout: ScoreOverlayLayout,
        color: Int,
    ) {
        paint.color = color
        paint.textSize = layout.fontSize.toFloat()
        paint.textAlign = Paint.Align.CENTER
        val maximum = (width - layout.horizontalPadding * 2).coerceAtLeast(1).toFloat()
        val text = ellipsize(raw, maximum)
        val baseline = layout.height / 2f - (paint.ascent() + paint.descent()) / 2f
        canvas.save()
        canvas.clipRect(left, 0f, left + width, layout.height.toFloat())
        canvas.drawText(text, left + width / 2f, baseline, paint)
        canvas.restore()
    }

    private fun ellipsize(value: String, maximum: Float): String {
        if (paint.measureText(value) <= maximum) return value
        val suffix = "\u2026"
        var end = value.length
        while (end > 0 && paint.measureText(value.substring(0, end) + suffix) > maximum) end--
        return value.substring(0, end) + suffix
    }

    private fun bottomRightRoundedPath(
        width: Float,
        height: Float,
        radius: Float,
        inset: Float,
    ) = Path().apply {
        val right = width - inset
        val bottom = height - inset
        moveTo(inset, inset)
        lineTo(right, inset)
        lineTo(right, bottom - radius)
        quadTo(right, bottom, right - radius, bottom)
        lineTo(inset, bottom)
        close()
    }
}
