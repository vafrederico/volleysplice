package com.volleycut.nativeanalysis

import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.json.JSONObject
import org.json.JSONArray
import org.opencv.android.OpenCVLoader
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicBoolean

@RunWith(AndroidJUnit4::class)
class PairedNeuralIntegrationTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val roi = AnalysisTypes.Roi(0.0, 0.0, 1.0, 1.0, "Full frame")
    private val progress = AnalysisTypes.ProgressListener { stage, _, detail ->
        android.util.Log.i("PairedValidation", "$stage: $detail")
    }
    private fun hash(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { stream ->
            val bytes = ByteArray(65536)
            while (true) { val n = stream.read(bytes); if (n < 0) break; digest.update(bytes, 0, n) }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
    private fun neural(neural: NeuralRallyPipeline): JSONObject {
        val target = neural.encoderTarget()
        val tokens = File(target.root(), target.id() + "-tokens.f32")
        return JSONObject(neural.report.toString()).put("embeddingSha256", hash(tokens))
            .put("embeddingBytes", tokens.length())
    }
    private fun result(model: String, value: AnalysisTypes.AnalysisResult) = JSONObject().apply {
        put("modelId", model)
        put("rallies", JSONArray().apply { value.ranges().forEach { put(JSONObject()
            .put("start", it.start()).put("end", it.end()).put("confidence", it.confidence().toDouble())) } })
        put("stageMilliseconds", JSONObject(value.stageMilliseconds()))
        put("profileMilliseconds", JSONObject(value.profileMilliseconds()))
        put("audioSha256", value.audioFeatureSha256())
        put("sampleRows", value.sampleRows())
        put("totalMilliseconds", value.totalMilliseconds())
        put("pssKilobytes", value.pssKilobytes())
        put("javaHeapUsedBytes", value.javaHeapUsedBytes())
        put("servingSide", value.servingSide()?.let(ServingSideJson::encodeOutput) ?: JSONObject.NULL)
        put("sideSwitch", value.sideSwitch()?.let(SideSwitchJson::encodeOutput) ?: JSONObject.NULL)
        put("suppressionApplied", value.suppression() != null)
        put("cache", JSONObject().put("visualHit", value.featureCache().visualHit())
            .put("audioHit", value.featureCache().audioHit()).put("contextHit", value.featureCache().contextHit()))
    }
    private fun single(uri: Uri, model: String, window: AnalysisTypes.AnalysisWindow,
                       cache: NativeFeatureCache.Mode, scores: Boolean): JSONObject {
        val media = AnalysisEngine(context).probe(uri)
        val cancelled = AtomicBoolean(false)
        val began = System.nanoTime()
        return requireNotNull(DistilledRallyModels.open(context, model, media, roi, cancelled::get)).use {
            val output = AnalysisEngine(context, it).analyze(uri, true, Int.MAX_VALUE,
                AnalysisTypes.VideoDecoderOptions.defaults(), cache, window, cancelled, progress, scores, scores)
            val ready = (System.nanoTime() - began) / 1e6
            result(model, output).put("allReadyMs", ready).put("neural", neural(it))
        }
    }

    @Test fun pairMatchesSeparateModelsAndPersistsBothWithoutChangingExistingEdits() {
        assertTrue(OpenCVLoader.initLocal())
        val fixture = DistilledRallyIntegrationTest().fixture(90)
        val uri = Uri.fromFile(fixture)
        try {
            val window = AnalysisTypes.AnalysisWindow(3.2, 5.5)
            val media = AnalysisEngine(context).probe(uri)
            val expected = listOf(RallyModels.RECALL, RallyModels.F1).associateWith {
                single(uri, it, window, NativeFeatureCache.Mode.BYPASS, false)
            }
            val tensors = mutableMapOf<String, JSONObject>()
            val output = PairedNeuralAnalysis.run(context, uri, RallyModels.RECALL, media, roi, window,
                NativeFeatureCache.Mode.BYPASS, AtomicBoolean(false), progress, false, false) {
                tensors[it.primaryId] = neural(it.primary)
                tensors[it.alternateId] = neural(it.alternate)
            }
            assertTrue(output.report.getBoolean("sharedImagePreparation"))
            assertEquals(output.report.getDouble("allReadyMs").toLong(), output.primary.totalMilliseconds())
            assertEquals(output.report.getJSONObject("primaryStageMilliseconds").getLong("score_specialists") +
                output.report.getLong("alternateScoreMs"), output.primary.stageMilliseconds().getValue("score_specialists"))
            for ((id, actual) in listOf(RallyModels.RECALL to output.primary, RallyModels.F1 to output.alternate)) {
                assertEquals(expected.getValue(id).getJSONArray("rallies").toString(), result(id, actual).getJSONArray("rallies").toString())
                assertEquals(expected.getValue(id).getJSONObject("neural").getString("embeddingSha256"), tensors.getValue(id).getString("embeddingSha256"))
                for (key in listOf("quality", "sampleTimestamps")) assertEquals(
                    expected.getValue(id).getJSONObject("neural").getJSONObject("video").getJSONArray(key).toString(),
                    tensors.getValue(id).getJSONObject("video").getJSONArray(key).toString())
            }
            val project = NativeProjectStore.newQueued(NativeProjectStore.source(context, uri, fixture.name),
                media, roi, window, false, false, RallyModels.RECALL, true)
            NativeProjectStore.save(context, project)
            val companion = requireNotNull(NativeProjectStore.completeAlternate(context, project, output))
            val saved = requireNotNull(NativeProjectStore.complete(context, project.id, output.primary))
            try {
                assertTrue(requireNotNull(NativeProjectStore.get(context, project.id)).prepareBothVariants)
                assertEquals(AnalysisRunKind.PAIRED_COMPANION,
                    requireNotNull(NativeProjectStore.get(context, companion.id)).analysisMeasurements.single().kind)
                assertEquals(2, listOf(saved, companion).map { it.id }.toSet().size)
                for (value in listOf(saved, companion)) assertEquals(ProjectStatus.READY,
                    requireNotNull(NativeProjectStore.get(context, value.id)).status)
                val edited = companion.copy(ranges = listOf(SeedRange(3300, 4000, 1f, "neural")))
                NativeProjectStore.save(context, edited)
                assertEquals(edited.ranges, requireNotNull(NativeProjectStore.completeAlternate(context, project, output)).ranges)
            } finally { NativeProjectStore.delete(context, saved); NativeProjectStore.delete(context, companion) }
            assertTrue(context.cacheDir.resolve("neural-analysis").listFiles().orEmpty().isEmpty())
        } finally { fixture.delete() }
    }

    @Test fun pairCancelsDuringVideoAndDeletesBothWorkspaces() {
        assertTrue(OpenCVLoader.initLocal())
        val fixture = DistilledRallyIntegrationTest().fixture(0)
        try {
            val uri = Uri.fromFile(fixture)
            val media = AnalysisEngine(context).probe(uri)
            val stop = AtomicBoolean(false)
            try {
                PairedNeuralAnalysis.run(context, uri, RallyModels.RECALL, media, roi,
                    AnalysisTypes.AnalysisWindow(0.0, 7.0), NativeFeatureCache.Mode.BYPASS, stop,
                    AnalysisTypes.ProgressListener { stage, fraction, _ -> if (stage == "video" && fraction > 0) stop.set(true) }, false, false)
                fail("Cancellation must stop the paired run")
            } catch (expected: java.io.IOException) { assertTrue(stop.get()) }
            assertTrue(context.cacheDir.resolve("neural-analysis").listFiles().orEmpty().isEmpty())
        } finally { fixture.delete() }
    }

    /** External source URI and window are provided by the private ledger runner. */
    @Test fun indexedVideoBenchmark() {
        val args = InstrumentationRegistry.getArguments()
        assumeTrue(args.containsKey("neuralVideoUri"))
        assertTrue(OpenCVLoader.initLocal())
        val uri = Uri.parse(requireNotNull(args.getString("neuralVideoUri")))
        val media = AnalysisEngine(context).probe(uri)
        val end = args.getString("neuralVideoSeconds")?.toDouble() ?: media.durationSeconds()
        val window = AnalysisTypes.AnalysisWindow(0.0, end)
        val mode = args.getString("pairedMode") ?: "paired"
        val model = args.getString("neuralRallyModel") ?: RallyModels.RECALL
        val cache = NativeFeatureCache.Mode.fromWireName(args.getString("neuralCacheMode") ?: "bypass")
        val scores = args.getString("neuralScores") != "false"
        val json = if (mode == "single") single(uri, model, window, cache, scores) else {
            require(mode == "paired")
            val tensors = mutableMapOf<String, JSONObject>()
            val output = PairedNeuralAnalysis.run(context, uri, model, media, roi, window, cache,
                AtomicBoolean(false), progress, scores, scores) {
                tensors[it.primaryId] = neural(it.primary); tensors[it.alternateId] = neural(it.alternate)
            }
            output.report.apply {
                put("results", JSONArray().put(result(model, output.primary).put("neural", tensors[model]))
                    .put(result(output.alternateModelId, output.alternate).put("neural", tensors[output.alternateModelId])))
                val project = NativeProjectStore.newQueued(NativeProjectStore.source(context, uri, "indexed-fixture.mp4"),
                    media, roi, window, scores, scores, model, true)
                NativeProjectStore.save(context, project)
                val persistStart = System.nanoTime()
                val companion = requireNotNull(NativeProjectStore.completeAlternate(context, project, output))
                val primary = requireNotNull(NativeProjectStore.complete(context, project.id, output.primary))
                put("persistBothMs", (System.nanoTime() - persistStart) / 1e6)
                val switchTimes = JSONArray()
                repeat(3) {
                    for (value in listOf(primary, companion)) {
                        val started = System.nanoTime()
                        val restored = requireNotNull(NativeProjectStore.get(context, value.id))
                        requireNotNull(restored.editorSeed())
                        switchTimes.put((System.nanoTime() - started) / 1e6)
                        assertEquals(ProjectStatus.READY, restored.status)
                        assertEquals(value.ranges, restored.ranges)
                        if (scores) assertEquals(ServingSideAnalysisStatus.READY, restored.servingSideStatus)
                    }
                }
                put("savedResultLoadMs", switchTimes)
                put("savedResultLoadScope", "Local project read, validation and editor seed creation; excludes UI rendering")
                NativeProjectStore.delete(context, primary); NativeProjectStore.delete(context, companion)
            }
        }
        json.put("durationSeconds", end).put("requestedMode", mode).put("cacheMode", cache.wireName())
        val folder = context.filesDir.resolve("integration-validation").also { it.mkdirs() }
        folder.resolve("paired-production-result.json").writeText(json.toString())
        assertTrue(context.cacheDir.resolve("neural-analysis").listFiles().orEmpty().isEmpty())
    }
}
