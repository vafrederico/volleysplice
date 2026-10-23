package com.volleycut.nativeanalysis

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.net.Uri
import android.os.SystemClock
import android.util.Base64
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.opencv.android.OpenCVLoader
import java.io.File

/** Real decoder/encoder/heads, with actual progress-panel rendering and notification checkpoints.
 * This does not qualify service lifecycle, physical codecs, accuracy, or long footage. */
class NeuralProgressQualificationInstrumentedTest {
    @get:Rule val compose = createAndroidComposeRule<androidx.activity.ComponentActivity>()

    @Test fun uncachedCachedAndSwitchDisabledNeuralProgressNeverRegresses() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        assertTrue(OpenCVLoader.initLocal())
        val input = File(context.cacheDir, "synthetic-neural-progress.mp4")
        input.writeBytes(Base64.decode(instrumentation.context.assets.open("draft-recovery-fixture.mp4.b64")
            .use { it.readBytes() }, Base64.DEFAULT))
        val media = AnalysisEngine(context).probe(Uri.fromFile(input))
        val roi = AnalysisTypes.Roi(0.0, 0.0, 1.0, 1.0, "Synthetic full frame")
        val ranges = listOf(AnalysisTypes.Interval(.1, .4, .9f), AnalysisTypes.Interval(.6, .9, .9f))
        fun state(id: String) = AnalysisTypes.ProductionStateOutput(id, doubleArrayOf(0.0, .25, .5, .75, 1.0), FloatArray(5) { .8f }, FloatArray(5) { .2f })
        val states = AnalysisTypes.ProductionStateOutputs(state(FeatureSchema.ALL_LABELS_V2_MODEL_ID), state(FeatureSchema.PREVIOUS_PRODUCTION_MODEL_ID))
        fun serves(id: String) = AnalysisTypes.ProductionServeOutput(id, doubleArrayOf(0.0, .25, .5, .75, 1.0), FloatArray(5) { .5f }, emptyList())
        val serveOutputs = AnalysisTypes.ProductionServeOutputs(serves(FeatureSchema.ALL_LABELS_V2_MODEL_ID), serves(FeatureSchema.PREVIOUS_PRODUCTION_MODEL_ID))
        val steps = mutableStateOf(emptyList<InferenceStepMeasurement>())
        compose.setContent { MaterialTheme { InferenceProgressMeasurementsPanel(steps.value) } }
        val notifications = context.getSystemService(NotificationManager::class.java)
        val channel = "synthetic-progress-qualification"
        notifications.createNotificationChannel(NotificationChannel(channel, "Synthetic progress test", NotificationManager.IMPORTANCE_LOW))
        val receipts = JSONArray()
        try {
            for (model in listOf(RallyModels.F1, RallyModels.RECALL)) {
                checkNotNull(DistilledRallyModels.open(context, model, media, roi, { false }, false)).use { pipeline ->
                    for ((case, switches) in listOf("uncached" to true, "cached" to true, "switches-off" to false)) {
                        val tracker = InferenceProgressTracker(false, true, switches)
                        val callbacks = JSONArray()
                        var previousOverall = -1.0
                        var previousServe = -1.0
                        var sawContext = false
                        var sawHead = false
                        val output = ScoreSpecialistInference.run(
                            context, Uri.fromFile(input), media, roi, ranges,
                            serveOutputs,
                            states,
                            AnalysisTypes.ProductionComponents.empty(),
                            object : AnalysisTypes.ProgressListener {
                                override fun onProgress(stage: String, fraction: Double, detail: String) {
                                    val overall = servingSideOverallProgress(stage, fraction)
                                    val mapped = projectCreationNotificationStage(stage, fraction, true)
                                    val snapshot = tracker.update(stage, fraction, detail)
                                    val serve = snapshot.single { it.id == "serving-side" }
                                    assertTrue("$model/$case overall regressed at $stage: $previousOverall -> $overall", overall + 1e-9 >= previousOverall)
                                    assertTrue("$model/$case serve UI regressed at $stage", serve.fraction + 1e-9 >= previousServe)
                                    assertEquals((overall * 100).toInt(), mapped.progressPercent)
                                    if (stage == "serving-side-features") {
                                        if (fraction > .5) sawContext = true
                                        assertNotEquals("Preliminary features must not complete the serve step", InferenceStepStatus.COMPLETE, serve.status)
                                    }
                                    if (stage == "serving-side") sawHead = true
                                    val notificationCheckpoint = listOf(.65, .77, 1.0).any {
                                        previousOverall < it && overall + 1e-9 >= it
                                    }
                                    previousOverall = overall
                                    previousServe = serve.fraction
                                    instrumentation.runOnMainSync { steps.value = snapshot }
                                    instrumentation.waitForIdleSync()
                                    compose.onNodeWithText("Finding serve markers").assertIsDisplayed()
                                    if (notificationCheckpoint) {
                                        // Avoid Android's package notification-rate limiter; the real service
                                        // also throttles updates rather than posting every decoder callback.
                                        SystemClock.sleep(550)
                                        val notification = Notification.Builder(context, channel)
                                            .setSmallIcon(android.R.drawable.ic_media_play).setContentTitle(mapped.title)
                                            .setProgress(100, mapped.progressPercent, false).build()
                                        notifications.notify("synthetic-progress", 901, notification)
                                        compose.waitUntil(3_000) { notifications.activeNotifications.any {
                                            it.tag == "synthetic-progress" &&
                                                it.notification.extras.getInt(Notification.EXTRA_PROGRESS) == mapped.progressPercent
                                        } }
                                        val posted = notifications.activeNotifications.single { it.tag == "synthetic-progress" }
                                        assertEquals(mapped.progressPercent, posted.notification.extras.getInt(Notification.EXTRA_PROGRESS))
                                    }
                                    callbacks.put(JSONObject().put("stage", stage).put("fraction", fraction)
                                        .put("overall", overall).put("serveUi", serve.fraction)
                                        .put("notificationPercent", mapped.progressPercent)
                                        .put("notificationCheckpoint", notificationCheckpoint))
                                }
                            },
                            { false }, switches, model, pipeline, listOf("N001", "N002"),
                        )
                        assertTrue("Neural head must execute", sawHead)
                        assertEquals(case == "uncached", sawContext)
                        assertEquals(2, output.servingSide.rows)
                        assertEquals(NeuralServingContract.identity(model).first, output.servingSide.modelId)
                        assertEquals(switches, output.sideSwitch != null)
                        assertEquals(1.0, previousOverall, 0.0)
                        assertEquals(InferenceStepStatus.COMPLETE, tracker.snapshot().single { it.id == "serving-side" }.status)
                        assertEquals(switches, tracker.snapshot().any { it.id == "side-switch" })
                        receipts.put(JSONObject().put("model", model).put("case", case).put("callbacks", callbacks))
                    }
                }
            }
            File(context.cacheDir, "synthetic-neural-progress-receipt.json").writeText(receipts.toString(2))
        } finally {
            notifications.cancel("synthetic-progress", 901)
            notifications.deleteNotificationChannel(channel)
            input.delete()
        }
    }
}
