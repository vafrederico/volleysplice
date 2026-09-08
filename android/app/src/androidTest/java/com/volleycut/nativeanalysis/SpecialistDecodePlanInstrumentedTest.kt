package com.volleycut.nativeanalysis

import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.ceil
import kotlin.math.roundToLong

/** Fast, opt-in analysis of real sync-frame preroll for the shared specialist schedule. */
@RunWith(AndroidJUnit4::class)
class SpecialistDecodePlanInstrumentedTest {
    @Test
    fun compareGapThresholdsAgainstSourceSyncFrames() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val arguments = InstrumentationRegistry.getArguments()
        assumeTrue(
            "Set -e specialist_decode_plan true to inspect the stored source",
            arguments.getString("specialist_decode_plan") == "true",
        )
        val projectId = arguments.getString("specialist_plan_project_id") ?: "project-mqmfyy"
        val project = NativeProjectStore.get(context, projectId) ?: error("Project $projectId was not found")
        val ranges = project.ranges.map {
            AnalysisTypes.Interval(
                it.startMs / 1_000.0, it.endMs / 1_000.0, it.confidence, it.agreement,
            )
        }
        val state = ProductionStateLoader.loadIfMissing(context, project)
        val serving = ServingSideInference.framePlan(ranges, project.media.durationSeconds())
        val switching = SideSwitchInference.framePlan(
            context,
            SideSwitchAnalysisInput(ranges, project.productionComponents, state),
            project.media.durationSeconds(),
        )
        val requested = (serving.requestedTimes.asList() + switching.requestedTimes.asList())
            .distinct().sorted()
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(context, Uri.parse(project.source.uri), null)
            val track = NativeVideoDecoder.findTrack(extractor, "video/")
            require(track >= 0) { "The source has no video track" }
            extractor.selectTrack(track)
            val format = extractor.getTrackFormat(track)
            val frameRate = format.getInteger(MediaFormat.KEY_FRAME_RATE).coerceAtLeast(1)
            val results = JSONArray()
            listOf(0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, Double.POSITIVE_INFINITY)
                .forEach { threshold ->
                    val segments = mutableListOf<MutableList<Double>>()
                    requested.forEach { time ->
                        val current = segments.lastOrNull()
                        if (current == null || time - current.last() > threshold) {
                            segments += mutableListOf(time)
                        } else current += time
                    }
                    var decodedSpanSeconds = 0.0
                    segments.forEach { segment ->
                        extractor.seekTo(
                            (segment.first() * 1_000_000).roundToLong(),
                            MediaExtractor.SEEK_TO_PREVIOUS_SYNC,
                        )
                        val syncStart = (extractor.sampleTime.coerceAtLeast(0L)) / 1_000_000.0
                        decodedSpanSeconds += (segment.last() - syncStart).coerceAtLeast(0.0)
                    }
                    results.put(JSONObject().apply {
                        put("maximumGapSeconds", if (threshold.isFinite()) threshold else "infinity")
                        put("segments", segments.size)
                        put("syncAwareDecodedSpanSeconds", decodedSpanSeconds)
                        put("estimatedDecodedFrames", ceil(decodedSpanSeconds * frameRate).toLong())
                    })
                }
            Log.i(TAG, JSONObject().apply {
                put("projectId", project.id)
                put("frameRate", frameRate)
                put("requestedFrames", requested.size)
                put("thresholds", results)
            }.toString())
        } finally {
            extractor.release()
        }
    }

    companion object {
        private const val TAG = "VolleySpliceSpecialistPlan"
    }
}
