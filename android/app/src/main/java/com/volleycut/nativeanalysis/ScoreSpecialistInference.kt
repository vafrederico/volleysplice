package com.volleycut.nativeanalysis

import android.content.Context
import android.net.Uri
import android.util.Log

/** Runs one shared gap-aware decode for both score-tracking specialists. */
internal object ScoreSpecialistInference {
    internal data class Profile(
        val servingRequestedFrames: Int,
        val sideSwitchRequestedFrames: Int,
        val sharedDecode: SpecialistFrameDecoder.Stats,
        val planningMilliseconds: Double,
        val servingEvaluationMilliseconds: Double,
        val sideSwitchEvaluationMilliseconds: Double,
        val totalMilliseconds: Double,
    ) {
        fun measurementProfile(): Map<String, Double> = linkedMapOf(
            "score/planning" to planningMilliseconds,
            "score/decode_setup" to sharedDecode.setupMilliseconds(),
            "score/requested_frame_conversion" to sharedDecode.conversionMilliseconds(),
            "score/shared_decode_wall" to sharedDecode.wallMilliseconds(),
            "score/serving_side_evaluation" to servingEvaluationMilliseconds,
            "score/side_switch_evaluation" to sideSwitchEvaluationMilliseconds,
        ).filterValues { it > 0.0 }

        fun measurementCounters(): Map<String, Long> = linkedMapOf(
            "serving_requested_frames" to servingRequestedFrames.toLong(),
            "side_switch_requested_frames" to sideSwitchRequestedFrames.toLong(),
            "shared_requested_frames" to sharedDecode.requestedFrames().toLong(),
            "score_decoder_segments" to sharedDecode.segments().toLong(),
            "score_decoder_output_frames" to sharedDecode.decoderOutputFrames(),
            "score_converted_frames" to sharedDecode.convertedOutputFrames(),
        )
    }

    internal data class Result(
        val servingSide: ServingSideOutput,
        val sideSwitch: SideSwitchOutput?,
        val profile: Profile,
    )

    fun run(
        context: Context,
        uri: Uri,
        media: AnalysisTypes.MediaInfo,
        roi: AnalysisTypes.Roi,
        ranges: List<AnalysisTypes.Interval>,
        serveOutputs: AnalysisTypes.ProductionServeOutputs,
        stateOutputs: AnalysisTypes.ProductionStateOutputs,
        productionComponents: AnalysisTypes.ProductionComponents,
        progress: AnalysisTypes.ProgressListener,
        cancelled: () -> Boolean,
        includeSideSwitch: Boolean = true,
    ): Result {
        val started = System.nanoTime()
        val duration = media.durationSeconds()
        val servingPlan = ServingSideInference.framePlan(ranges, duration)
        val switchInput = if (includeSideSwitch) {
            SideSwitchAnalysisInput(ranges, productionComponents, stateOutputs)
        } else null
        val switchPlan = switchInput?.let { SideSwitchInference.framePlan(context, it, duration) }
        val planningMilliseconds = elapsedMilliseconds(started)
        progress.onProgress(
            "score-specialists",
            0.0,
            if (includeSideSwitch) "Loading serving-side and team-switch models"
            else "Loading serving-side model; automatic team switches are off",
        )
        val sampled = SpecialistFrameDecoder(context).decode(
            uri,
            media,
            roi,
            servingPlan.requestedTimes,
            switchPlan?.requestedTimes ?: doubleArrayOf(),
            progress,
            cancelled,
        )
        val servingStarted = System.nanoTime()
        val serving = ServingSideInference.evaluate(
            context,
            duration,
            servingPlan.candidates,
            serveOutputs,
            sampled.servingGray(),
            progress,
            cancelled,
        )
        val servingEvaluationMilliseconds = elapsedMilliseconds(servingStarted)
        val sideSwitchStarted = System.nanoTime()
        val switches = if (switchInput != null && switchPlan != null) {
            SideSwitchInference.evaluate(
                duration,
                switchInput,
                switchPlan,
                sampled.sideSwitchBgr(),
                progress,
                cancelled,
            )
        } else {
            progress.onProgress("side-switch", 1.0, "Automatic team-switch markers disabled")
            null
        }
        val sideSwitchEvaluationMilliseconds = elapsedMilliseconds(sideSwitchStarted)
        val result = Result(
            serving,
            switches,
            Profile(
                servingRequestedFrames = servingPlan.requestedTimes.size,
                sideSwitchRequestedFrames = switchPlan?.requestedTimes?.size ?: 0,
                sharedDecode = sampled.stats(),
                planningMilliseconds = planningMilliseconds,
                servingEvaluationMilliseconds = servingEvaluationMilliseconds,
                sideSwitchEvaluationMilliseconds = sideSwitchEvaluationMilliseconds,
                totalMilliseconds = elapsedMilliseconds(started),
            ),
        )
        Log.i(TAG, result.profile.toString())
        return result
    }

    private fun elapsedMilliseconds(started: Long): Double =
        (System.nanoTime() - started) / 1_000_000.0

    private const val TAG = "VolleyCutScoreProfile"
}
