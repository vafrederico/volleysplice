package com.volleycut.nativeanalysis

import android.net.Uri
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader

/** Opt-in regression against an existing source and its actual production frame schedule. */
@RunWith(AndroidJUnit4::class)
class PrecedingServingFramesInstrumentedTest {
    @Test fun storedProjectFramesMatchThePrecedingPlan() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val args = InstrumentationRegistry.getArguments()
        val projectId = args.getString("preceding_frames_project")
        assumeTrue("Set preceding_frames_project to a stored project", !projectId.isNullOrBlank())
        val project = checkNotNull(NativeProjectStore.get(context, checkNotNull(projectId)))
        assertTrue(OpenCVLoader.initLocal())
        val ranges = project.ranges.map {
            AnalysisTypes.Interval(it.startMs / 1000.0, it.endMs / 1000.0, it.confidence, it.agreement)
        }
        val plan = ServingSideInference.framePlan(ranges, project.media.durationSeconds())
        assertTrue(plan.requestedTimes.isNotEmpty())
        val decoded = SpecialistFrameDecoder(context).decodePrecedingServing(
            Uri.parse(project.source.uri), project.media, project.roi, plan.requestedTimes,
            doubleArrayOf(), AnalysisTypes.ProgressListener { _, _, _ -> }, { false },
        )
        assertEquals(plan.requestedTimes.size, decoded.servingGray().size)
        decoded.servingGray().values.forEach { assertEquals(SERVING_SIDE_WIDTH * SERVING_SIDE_HEIGHT, it.size) }
        Log.i("PrecedingServingFrames", "All requested source frames decoded: ${decoded.stats()}")
    }
}
