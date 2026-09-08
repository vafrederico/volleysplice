package com.volleycut.nativeanalysis

import android.net.Uri
import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import java.util.concurrent.atomic.AtomicInteger

/** Opt-in Pixel 10 end-to-end test for shared decode, both score models, persistence, and markers. */
@RunWith(AndroidJUnit4::class)
class SideSwitchPipelineInstrumentedTest {
    @Test
    fun inferPersistAndSeedTeamSwitchMarkers() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val arguments = InstrumentationRegistry.getArguments()
        assumeTrue(
            "Set -e side_switch_pipeline true to run the long Pixel pipeline test",
            arguments.getString("side_switch_pipeline") == "true",
        )
        assertTrue("This pipeline must be verified on Pixel 10", Build.MODEL.startsWith("Pixel 10"))
        val requestedId = arguments.getString("side_switch_project_id")
        val project = (if (requestedId.isNullOrBlank()) {
            NativeProjectStore.list(context).firstOrNull {
                it.status == ProjectStatus.READY && it.ranges.size >= 2
            }
        } else NativeProjectStore.get(context, requestedId))
            ?: error("No ready project with at least two rally ranges was found")
        val ranges = project.ranges.map {
            AnalysisTypes.Interval(
                it.startMs / 1_000.0, it.endMs / 1_000.0, it.confidence, it.agreement,
            )
        }
        val states = ProductionStateLoader.loadIfMissing(context, project)
        val lastProgress = AtomicInteger(-1)
        val started = SystemClock.elapsedRealtimeNanos()
        val output = ScoreSpecialistInference.run(
            context,
            Uri.parse(project.source.uri),
            project.media,
            project.roi,
            ranges,
            project.productionServeOutputs,
            states,
            project.productionComponents,
            object : AnalysisTypes.ProgressListener {
                override fun onProgress(stage: String, fraction: Double, detail: String) {
                    val percent = (fraction.coerceIn(0.0, 1.0) * 100).toInt()
                    if (percent / 5 > lastProgress.get() / 5) {
                        lastProgress.set(percent)
                        Log.i(TAG, "stage=$stage progress=$percent detail=$detail")
                    }
                }

                override fun onPerformance(stats: AnalysisTypes.PerformanceStats) = Unit
            },
            { false },
        )
        val elapsedMs = (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000.0
        val sideSwitch = checkNotNull(output.sideSwitch)
        assertEquals(project.ranges.size, output.servingSide.rows)
        assertTrue(output.servingSide.rawFeatures.all(Double::isFinite))
        assertTrue(sideSwitch.rows >= project.ranges.size - 1)
        assertTrue(SideSwitchJson.validOutput(sideSwitch))
        project.servingSide?.let {
            assertArrayEquals(it.rawFeatures, output.servingSide.rawFeatures, 0.0)
        }
        project.sideSwitch?.let {
            assertArrayEquals(it.features, sideSwitch.features, 0.0)
        }

        val seeded = ScoreReducer.seedModelMarkers(
            ScoreTracking(), output.servingSide, sideSwitch,
        )
        assertEquals(
            sideSwitch.candidates.size,
            seeded.sideSwitchMarkers.count { it.origin == ServeMarkerOrigin.MODEL },
        )
        assertTrue(seeded.sideSwitchMarkers.all { it.modelEventId != null })

        if (arguments.getString("persist_side_switch") == "true") {
            val completed = NativeProjectStore.completeServingSide(
                context, project.id, output.servingSide, sideSwitch, states,
            )
            assertNotNull(completed)
            val seed = checkNotNull(completed?.editorSeed())
            val draft = EditorDraftStore(context, seed).load() ?: EditorMath.newDraft(seed)
            assertEquals(
                sideSwitch.candidates.size,
                draft.scoreTracking.sideSwitchMarkers.count {
                    it.origin == ServeMarkerOrigin.MODEL
                },
            )
        }

        Log.i(TAG, JSONObject().apply {
            put("device", Build.MODEL)
            put("projectId", project.id)
            put("durationSeconds", project.media.durationSeconds())
            put("rallyCount", project.ranges.size)
            put("sideSwitchFeatureRows", sideSwitch.rows)
            put("predictedSwitches", sideSwitch.candidates.size)
            put("servingSideRows", output.servingSide.rows)
            put("elapsedMilliseconds", elapsedMs)
            put("servingRequestedFrames", output.profile.servingRequestedFrames)
            put("sideSwitchRequestedFrames", output.profile.sideSwitchRequestedFrames)
            put("decoderSegments", output.profile.sharedDecode.segments())
            put("sharedRequestedFrames", output.profile.sharedDecode.requestedFrames())
            put("decoderOutputFrames", output.profile.sharedDecode.decoderOutputFrames())
            put("convertedOutputFrames", output.profile.sharedDecode.convertedOutputFrames())
            put("decodeSetupMilliseconds", output.profile.sharedDecode.setupMilliseconds())
            put("decodeConversionMilliseconds", output.profile.sharedDecode.conversionMilliseconds())
            put("decodeWallMilliseconds", output.profile.sharedDecode.wallMilliseconds())
            put("planningMilliseconds", output.profile.planningMilliseconds)
            put("servingEvaluationMilliseconds", output.profile.servingEvaluationMilliseconds)
            put("sideSwitchEvaluationMilliseconds", output.profile.sideSwitchEvaluationMilliseconds)
            put("persisted", arguments.getString("persist_side_switch") == "true")
        }.toString())
    }

    companion object {
        private const val TAG = "VolleySpliceSwitchE2E"

        @JvmStatic
        @BeforeClass
        fun loadOpenCv() {
            check(OpenCVLoader.initLocal()) { "OpenCV could not initialize" }
        }
    }
}
