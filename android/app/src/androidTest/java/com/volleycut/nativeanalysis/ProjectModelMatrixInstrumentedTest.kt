package com.volleycut.nativeanalysis

import android.content.Intent
import android.os.SystemClock
import android.view.WindowManager
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.UUID

/** Full production-worker reruns, with private inputs supplied only at runtime. */
@RunWith(AndroidJUnit4::class)
class ProjectModelMatrixInstrumentedTest {
    @Test fun rerunStoredSourcesAcrossModels() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val args = InstrumentationRegistry.getArguments()
        val hdrId = args.getString("matrix_hdr_project")
        val sdrId = args.getString("matrix_sdr_project")
        assumeTrue("Supply matrix_hdr_project and matrix_sdr_project", !hdrId.isNullOrBlank() && !sdrId.isNullOrBlank())
        val hdr = checkNotNull(NativeProjectStore.get(context, checkNotNull(hdrId)))
        val sdr = checkNotNull(NativeProjectStore.get(context, checkNotNull(sdrId)))
        assertTrue(NativeProjectStore.sourceAvailable(context, hdr))
        assertTrue(NativeProjectStore.sourceAvailable(context, sdr))
        assertFalse("Finish existing analyses before starting a matrix", NativeProjectStore.list(context).any {
            it.status == ProjectStatus.QUEUED || it.status == ProjectStatus.ANALYZING ||
                it.servingSideStatus == ServingSideAnalysisStatus.QUEUED ||
                it.servingSideStatus == ServingSideAnalysisStatus.ANALYZING
        })
        val originalSelection = NativeProjectStore.selectedId(context)
        val originalFiles = listOf(hdr, sdr).associate { project ->
            val file = File(context.filesDir, "native-projects/${project.id}.json")
            file to file.readBytes()
        }
        val runId = UUID.randomUUID().toString()
        val output = File(checkNotNull(context.getExternalFilesDir("model-matrix-validation")), runId)
        output.mkdirs()
        val cases = listOf(
            Triple("hdr-coverage", hdr, RallyModels.RECALL),
            Triple("hdr-legacy", hdr, FeatureSchema.MODEL_ID),
            Triple("sdr-balanced", sdr, RallyModels.F1),
        )
        val queued = cases.map { (label, source, model) ->
            label to NativeProjectStore.newQueued(
                source.source, source.media, source.roi, source.analysisWindow,
                analyzeServingSide = true, sideSwitchEnabled = source.sideSwitchEnabled,
                rallyModelId = model,
            ).copy(id = "validation-$label-$runId")
        }
        val started = SystemClock.elapsedRealtime()
        val copiedResults = mutableSetOf<String>()
        fun writeReceipt(): List<NativeProject> {
            val states = queued.map { (_, project) -> checkNotNull(NativeProjectStore.get(context, project.id)) }
            val receipt = JSONObject().put("runId", runId).put("elapsedMs", SystemClock.elapsedRealtime() - started)
                .put("cacheMode", "use").put("cases", JSONArray(states.mapIndexed { index, project ->
                    val candidates = project.servingSide?.candidates.orEmpty()
                    JSONObject().put("label", queued[index].first).put("projectId", project.id)
                        .put("modelId", project.modelId).put("status", project.status.wireName)
                        .put("servingStatus", project.servingSideStatus.wireName)
                        .put("rallies", project.ranges.size).put("candidates", candidates.size)
                        .put("verdicts", JSONObject(candidates.groupingBy { it.verdict.wireName }.eachCount()))
                        .put("servingModelId", project.servingSide?.modelId ?: JSONObject.NULL)
                        .put("error", project.error ?: project.servingSideError ?: project.sideSwitchError ?: JSONObject.NULL)
                }))
            File(output, "receipt.json").writeText(receipt.toString(2))
            states.forEachIndexed { index, project ->
                if ((project.status == ProjectStatus.READY || project.status == ProjectStatus.ERROR)
                    && copiedResults.add(project.id)) {
                    File(output, "${queued[index].first}.json").writeBytes(
                        File(context.filesDir, "native-projects/${project.id}.json").readBytes(),
                    )
                }
            }
            return states
        }
        try {
            // Display the existing ready editor while the serialized worker runs.
            // Keep this test activity awake without changing system sleep settings.
            NativeProjectStore.setSelectedId(context, hdr.id)
            val activity = instrumentation.startActivitySync(Intent(context, EditorActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK))
            instrumentation.waitForIdleSync()
            instrumentation.runOnMainSync { activity.window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON) }
            queued.forEach { (_, project) -> NativeProjectStore.save(context, project) }
            writeReceipt()
            queued.forEach { (_, project) -> ProjectAnalysisService.enqueue(context, project.id) }
            val deadline = started + 60 * 60_000L
            while (true) {
                val states = writeReceipt()
                if (states.all { it.status == ProjectStatus.READY || it.status == ProjectStatus.ERROR }) break
                check(SystemClock.elapsedRealtime() < deadline) { "Matrix still running; its production queue was retained" }
                SystemClock.sleep(2_000)
            }
            writeReceipt().forEach { project ->
                assertEquals(project.id, ProjectStatus.READY, project.status)
                assertNull(project.error)
                assertEquals(project.id, ServingSideAnalysisStatus.READY, project.servingSideStatus)
                assertNull(project.servingSideError)
                assertNull(project.sideSwitchError)
                assertTrue(project.ranges.isNotEmpty())
                val serving = checkNotNull(project.servingSide)
                assertTrue(serving.candidates.isNotEmpty())
                assertEquals(serving.rows, serving.candidates.size)
                val expected = if (RallyModels.isNeural(project.modelId)) {
                    assertNotNull(project.neuralScores)
                    NeuralServingContract.identity(project.modelId).first
                } else SERVING_SIDE_MODEL_ID
                assertEquals(expected, serving.modelId)
            }
            originalFiles.forEach { (file, bytes) -> assertArrayEquals(file.name, bytes, file.readBytes()) }
        } finally {
            NativeProjectStore.setSelectedId(context, originalSelection)
        }
    }
}
