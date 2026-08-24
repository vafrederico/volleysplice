package com.volleycut.nativeanalysis

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.util.Base64
import androidx.annotation.OptIn
import androidx.media3.common.MediaItem
import androidx.media3.common.MimeTypes
import androidx.media3.common.util.UnstableApi
import androidx.media3.effect.OverlayEffect
import androidx.media3.transformer.Composition
import androidx.media3.transformer.EditedMediaItem
import androidx.media3.transformer.EditedMediaItemSequence
import androidx.media3.transformer.ExportException
import androidx.media3.transformer.ExportResult
import androidx.media3.transformer.Effects
import androidx.media3.transformer.Transformer
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

@RunWith(AndroidJUnit4::class)
class ScoreOverlayInstrumentedTest {
    @Test
    fun exportCanvasDrawsTeamAndScoreCellsAtTheVideoTopLeft() {
        val snapshot = ScoreExportSnapshot(
            render = true,
            scoreTracking = ScoreTracking(team1Name = "Falcons", team2Name = "Wolves"),
            ignoredIntervals = emptyList(),
            excludedRallyIds = emptySet(),
            rallyRanges = emptyList(),
        )
        val bitmap = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)

        ScoreCanvasOverlay(snapshot, sourceStartMs = 0, compositionStartUs = 0)
            .onDraw(canvas, 0)

        assertEquals(ScoreOverlay.TEAM_1_COLOR.toInt(), bitmap.getPixel(10, 5))
        assertEquals(ScoreOverlay.SCORE_BACKGROUND_COLOR.toInt(), bitmap.getPixel(90, 5))
        assertEquals(ScoreOverlay.TEAM_2_COLOR.toInt(), bitmap.getPixel(138, 5))
        assertEquals(Color.TRANSPARENT, bitmap.getPixel(500, 100))
        bitmap.recycle()
    }

    @Test
    fun exportCanvasDrawsTheTimedPointTimelineBesideTheScoreboard() {
        val snapshot = ScoreExportSnapshot(
            render = true,
            scoreTracking = ScoreTracking(
                team1Name = "Falcons",
                team2Name = "Wolves",
                serveMarkers = listOf(
                    ServeMarker("S1", 1_000, ServingSide.NEAR, ServeMarkerOrigin.MANUAL),
                    ServeMarker("S2", 5_000, ServingSide.NEAR, ServeMarkerOrigin.MANUAL),
                ),
            ),
            ignoredIntervals = emptyList(),
            excludedRallyIds = emptySet(),
            rallyRanges = listOf(ScoreRallyRange(5_000, 7_000, 4_000, 8_000)),
        )
        val bitmap = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)

        ScoreCanvasOverlay(snapshot, sourceStartMs = 0, compositionStartUs = 0)
            .onDraw(canvas, 4_250_000)

        val score = ScoreOverlay.snapshot(snapshot.prepared, 4_250)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG)
        val scoreLayout = ScoreOverlay.layout(640, 480, score) { text, fontSize ->
            paint.textSize = fontSize.toFloat()
            paint.measureText(text)
        }
        val timeline = ScoreOverlay.pointTimelineLayout(640, 480, scoreLayout, 1)
        val sampleX = (timeline.startX + timeline.columnSpacing * .5f + timeline.circleRadius * .65f).toInt()
        val sampleY = timeline.team1CenterY.toInt()
        assertColorNear(ScoreOverlay.TEAM_1_COLOR.toInt(), bitmap.getPixel(sampleX, sampleY))
        bitmap.recycle()
    }

    @Test
    fun exportCanvasOmitsPointTimelineWhenItsToggleIsOff() {
        val snapshot = ScoreExportSnapshot(
            render = true,
            renderPointTimeline = false,
            scoreTracking = ScoreTracking(
                serveMarkers = listOf(
                    ServeMarker("S1", 1_000, ServingSide.NEAR, ServeMarkerOrigin.MANUAL),
                    ServeMarker("S2", 5_000, ServingSide.NEAR, ServeMarkerOrigin.MANUAL),
                ),
            ),
            ignoredIntervals = emptyList(),
            excludedRallyIds = emptySet(),
            rallyRanges = listOf(ScoreRallyRange(5_000, 7_000, 4_000, 8_000)),
        )
        val bitmap = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        ScoreCanvasOverlay(snapshot, sourceStartMs = 0, compositionStartUs = 0)
            .onDraw(canvas, 4_250_000)

        val score = ScoreOverlay.snapshot(snapshot.prepared, 4_250)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG)
        val scoreLayout = ScoreOverlay.layout(640, 480, score) { text, fontSize ->
            paint.textSize = fontSize.toFloat()
            paint.measureText(text)
        }
        val timeline = ScoreOverlay.pointTimelineLayout(640, 480, scoreLayout, 1)
        val sampleX = (timeline.startX + timeline.columnSpacing * .5f + timeline.circleRadius * .65f).toInt()
        assertEquals(Color.TRANSPARENT, bitmap.getPixel(sampleX, timeline.team1CenterY.toInt()))
        bitmap.recycle()
    }

    @OptIn(UnstableApi::class)
    @Test
    fun transformerBakesTheOverlayIntoEncodedVideoPixels() {
        val frame = encodeOverlayFixture("overlay-fixture.mp4.b64")
        assertEquals(320, frame.width)
        assertEquals(240, frame.height)
        assertOverlayPixels(frame)
        frame.recycle()
    }

    @OptIn(UnstableApi::class)
    @Test
    fun transformerBakesRotationBeforeDrawingTheOverlay() {
        listOf(
            Triple("overlay-fixture-90.mp4.b64", 240, 320),
            Triple("overlay-fixture-180.mp4.b64", 320, 240),
            Triple("overlay-fixture-270.mp4.b64", 240, 320),
        ).forEach { (assetName, width, height) ->
            val frame = encodeOverlayFixture(assetName)
            assertEquals(assetName, width, frame.width)
            assertEquals(assetName, height, frame.height)
            assertOverlayPixels(frame)
            frame.recycle()
        }
    }

    @OptIn(UnstableApi::class)
    private fun encodeOverlayFixture(assetName: String): Bitmap {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val encodedFixture = instrumentation.context.assets
            .open(assetName)
            .bufferedReader()
            .use { it.readText() }
        val fixtureId = assetName.removeSuffix(".mp4.b64")
        val input = context.cacheDir.resolve("$fixtureId-input.mp4")
        val output = context.cacheDir.resolve("$fixtureId-output.mp4")
        input.writeBytes(Base64.decode(encodedFixture.trim(), Base64.DEFAULT))
        output.delete()
        val snapshot = ScoreExportSnapshot(
            render = true,
            scoreTracking = ScoreTracking(team1Name = "Falcons", team2Name = "Wolves"),
            ignoredIntervals = emptyList(),
            excludedRallyIds = emptySet(),
            rallyRanges = emptyList(),
        )
        val edited = EditedMediaItem.Builder(MediaItem.fromUri(Uri.fromFile(input)))
            .setDurationUs(1_000_000)
            .setEffects(Effects(
                emptyList(),
                listOf(OverlayEffect(listOf(ScoreCanvasOverlay(snapshot, 0, 0)))),
            ))
            .build()
        val composition = Composition.Builder(
            EditedMediaItemSequence.withAudioAndVideoFrom(listOf(edited)),
        ).build()
        val completed = CountDownLatch(1)
        val failure = AtomicReference<ExportException?>(null)

        instrumentation.runOnMainSync {
            Transformer.Builder(context)
                .setVideoMimeType(MimeTypes.VIDEO_H264)
                .addListener(object : Transformer.Listener {
                    override fun onCompleted(composition: Composition, exportResult: ExportResult) {
                        completed.countDown()
                    }

                    override fun onError(
                        composition: Composition,
                        exportResult: ExportResult,
                        exportException: ExportException,
                    ) {
                        failure.set(exportException)
                        completed.countDown()
                    }
                })
                .build()
                .start(composition, output.absolutePath)
        }

        assertTrue("Timed out waiting for Transformer", completed.await(30, TimeUnit.SECONDS))
        assertNull(failure.get()?.message, failure.get())
        assertTrue(output.length() > 0L)
        val retriever = MediaMetadataRetriever()
        retriever.setDataSource(output.absolutePath)
        val frame = checkNotNull(
            retriever.getFrameAtTime(500_000, MediaMetadataRetriever.OPTION_CLOSEST),
        )
        retriever.release()
        input.delete()
        output.delete()
        return frame
    }

    private fun assertOverlayPixels(frame: Bitmap) {
        assertColorNear(ScoreOverlay.TEAM_1_COLOR.toInt(), frame.getPixel(10, 5))
        assertColorNear(ScoreOverlay.SCORE_BACKGROUND_COLOR.toInt(), frame.getPixel(90, 5))
        assertColorNear(ScoreOverlay.TEAM_2_COLOR.toInt(), frame.getPixel(138, 5))
    }

    private fun assertColorNear(expected: Int, actual: Int) {
        val distance = maxOf(
            kotlin.math.abs(Color.red(expected) - Color.red(actual)),
            kotlin.math.abs(Color.green(expected) - Color.green(actual)),
            kotlin.math.abs(Color.blue(expected) - Color.blue(actual)),
        )
        assertTrue(
            "Expected #${Integer.toHexString(expected)}, got #${Integer.toHexString(actual)}",
            distance <= 48,
        )
    }
}
