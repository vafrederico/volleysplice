package com.volleycut.nativeanalysis

import android.os.Build
import android.os.Debug
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.function.BooleanSupplier

/** Device-only profiling for the production serving-side decode and feature path. */
@RunWith(AndroidJUnit4::class)
class ServingSidePipelineBenchmarkInstrumentedTest {
    @Test
    fun profileProductionProject() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val arguments = InstrumentationRegistry.getArguments()
        assumeTrue(
            "Set -e serving_side_benchmark true to run the long device benchmark",
            arguments.getString("serving_side_benchmark") == "true",
        )
        val strategy = arguments.getString("serving_side_strategy") ?: "gap-aware"
        require(strategy in setOf("gap-aware", "sequential", "candidate-seeks")) {
            "Unknown serving-side benchmark strategy: $strategy"
        }
        val requestedProjectId = arguments.getString("serving_side_project_id")
        val project = if (requestedProjectId.isNullOrBlank()) {
            NativeProjectStore.list(context).firstOrNull {
                it.status == ProjectStatus.READY && it.ranges.isNotEmpty()
            }
        } else {
            NativeProjectStore.get(context, requestedProjectId)
        } ?: error("No ready project with serving-side candidates was found")
        val candidates = ServingSideModelRunner.candidates(project.ranges)
        assertTrue("The benchmark needs serving-side candidates", candidates.isNotEmpty())
        val duration = project.media.durationSeconds()
        val candidatePlans = candidates.map { candidate ->
            (COURT_FLOW_OFFSETS_SECONDS.asList() + FLIGHT_OFFSETS_SECONDS.asList())
                .map { ServingSideInference.clampedTimestamp(candidate.start, it, duration) }
                .distinct()
                .sorted()
                .toDoubleArray()
        }
        val requestedTimes = candidatePlans.flatMap(DoubleArray::asList)
            .distinct()
            .sorted()
            .toDoubleArray()
        val decoder = if (strategy == "sequential") {
            ServingSideFrameDecoder(context, Double.POSITIVE_INFINITY)
        } else ServingSideFrameDecoder(context)
        val progress = AnalysisTypes.ProgressListener { _, _, _ -> }
        val cancelled = BooleanSupplier { false }

        // Keep native initialization and first-use OpenCV setup outside the measured sections.
        val staticFrame = ByteArray(SERVING_SIDE_WIDTH * SERVING_SIDE_HEIGHT) { 73 }
        ServingSideFeatureExtractor.extract(
            Array(COURT_FLOW_OFFSETS_SECONDS.size) { staticFrame },
            Array(FLIGHT_OFFSETS_SECONDS.size) { staticFrame },
        )
        System.gc()
        val decodeStarted = SystemClock.elapsedRealtimeNanos()
        val sampled = linkedMapOf<Double, ByteArray>()
        when (strategy) {
            "sequential", "gap-aware" -> sampled.putAll(
                decoder.decode(
                    android.net.Uri.parse(project.source.uri),
                    project.media,
                    project.roi,
                    requestedTimes,
                    progress,
                    cancelled,
                ),
            )
            "candidate-seeks" -> candidatePlans.forEach { plan ->
                sampled.putAll(
                    decoder.decode(
                        android.net.Uri.parse(project.source.uri),
                        project.media,
                        project.roi,
                        plan,
                        progress,
                        cancelled,
                    ),
                )
            }
        }
        val decodeMilliseconds = elapsedMilliseconds(decodeStarted)
        assertEquals(requestedTimes.size, sampled.size)
        Log.i(
            TAG,
            "stage=decode-complete strategy=$strategy milliseconds=$decodeMilliseconds " +
                "frames=${sampled.size}",
        )

        val raw = DoubleArray(candidates.size * SERVING_SIDE_FEATURE_COLUMNS)
        val featureStarted = SystemClock.elapsedRealtimeNanos()
        candidates.forEachIndexed { row, candidate ->
            fun frames(offsets: DoubleArray) = offsets.map { offset ->
                val timestamp = ServingSideInference.clampedTimestamp(candidate.start, offset, duration)
                sampled[timestamp] ?: error("Missing serving-side frame at $timestamp")
            }.toTypedArray()
            ServingSideFeatureExtractor.extract(
                frames(COURT_FLOW_OFFSETS_SECONDS),
                frames(FLIGHT_OFFSETS_SECONDS),
            ).copyInto(raw, row * SERVING_SIDE_FEATURE_COLUMNS)
            if ((row + 1) % 10 == 0 || row + 1 == candidates.size) {
                Log.i(TAG, "stage=features strategy=$strategy candidates=${row + 1}/${candidates.size}")
            }
        }
        val featureMilliseconds = elapsedMilliseconds(featureStarted)
        assertTrue(raw.all(Double::isFinite))

        val firstRequested = requestedTimes.first()
        val lastRequested = requestedTimes.last()
        val output = JSONObject().apply {
            put("strategy", strategy)
            put("device", Build.MODEL)
            put("androidRelease", Build.VERSION.RELEASE)
            put("androidSdk", Build.VERSION.SDK_INT)
            put("projectId", project.id)
            put("sourceName", project.source.name)
            put("durationSeconds", duration)
            put("width", project.media.width())
            put("height", project.media.height())
            put("videoMime", project.media.videoMime())
            put("candidateCount", candidates.size)
            put("candidateRequestedFrames", candidatePlans.sumOf(DoubleArray::size))
            put("distinctRequestedFrames", requestedTimes.size)
            put("firstRequestedSecond", firstRequested)
            put("lastRequestedSecond", lastRequested)
            put("sequentialSpanSeconds", lastRequested - firstRequested)
            val plannedSessions = when (strategy) {
                "sequential" -> 1
                "gap-aware" -> SpecialistFrameDecoder.plannedSegmentCount(requestedTimes)
                else -> candidatePlans.sumOf(SpecialistFrameDecoder::plannedSegmentCount)
            }
            put("decoderSessions", plannedSessions)
            put("extractorSeekCalls", plannedSessions)
            put("decodeMilliseconds", decodeMilliseconds)
            put("featureMilliseconds", featureMilliseconds)
            put("totalMilliseconds", decodeMilliseconds + featureMilliseconds)
            put("frameSha256", frameHash(sampled, requestedTimes))
            put("rawFeatureSha256", doubleHash(raw))
            put("javaHeapUsedBytes", Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory())
            put("nativeHeapAllocatedBytes", Debug.getNativeHeapAllocatedSize())
        }
        Log.i(TAG, output.toString())
    }

    private fun frameHash(
        frames: Map<Double, ByteArray>,
        requestedTimes: DoubleArray,
    ): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val timestamp = ByteBuffer.allocate(java.lang.Double.BYTES).order(ByteOrder.LITTLE_ENDIAN)
        requestedTimes.forEach { time ->
            timestamp.clear()
            timestamp.putDouble(time)
            digest.update(timestamp.array())
            digest.update(frames[time] ?: error("Missing frame at $time"))
        }
        return digest.digest().toHex()
    }

    private fun doubleHash(values: DoubleArray): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val value = ByteBuffer.allocate(java.lang.Double.BYTES).order(ByteOrder.LITTLE_ENDIAN)
        values.forEach {
            value.clear()
            value.putDouble(it)
            digest.update(value.array())
        }
        return digest.digest().toHex()
    }

    private fun ByteArray.toHex(): String = joinToString("") { "%02x".format(it.toInt() and 0xff) }

    private fun elapsedMilliseconds(started: Long): Double =
        (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000.0

    companion object {
        private const val TAG = "VolleyCutServingBench"

        @JvmStatic
        @BeforeClass
        fun loadOpenCv() {
            check(OpenCVLoader.initLocal()) { "OpenCV could not initialize" }
        }
    }
}
