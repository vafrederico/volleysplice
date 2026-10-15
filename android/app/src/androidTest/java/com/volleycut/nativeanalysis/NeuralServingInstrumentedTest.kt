package com.volleycut.nativeanalysis

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.net.Uri
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicBoolean

/** Private recording/feedback paths are opt-in runtime arguments, never bundled. */
@RunWith(AndroidJUnit4::class)
class NeuralServingInstrumentedTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext
    private val args = InstrumentationRegistry.getArguments()
    private val progress = AnalysisTypes.ProgressListener { stage, fraction, detail ->
        if (fraction >= 1 || fraction == 0.0) Log.i("NeuralServingCheck", "$stage: $detail")
    }
    @Test fun pairedHeadsAndOptionalRecording() {
        // Ask Android to prepare the app's scoped external directory before
        // reading runtime inputs copied there through ADB.
        val external = checkNotNull(context.getExternalFilesDir("serving-validation"))
        external.mkdirs()
        assertTrue(OpenCVLoader.initLocal())
        val golden = JSONObject(instrumentation.context.assets.open("neural-serving-golden.json").use { it.readBytes().toString(Charsets.UTF_8) })
        val rows = golden.getInt("rows")
        val windows = FloatArray(rows * 13 * 3843) { (it % 101 - 50) / 100f }
        repeat(rows) { row -> repeat(13) { frame ->
            val base = (row * 13 + frame) * 3843
            windows[base + 3840] = 1f; windows[base + 3841] = (frame - 6) / 6f; windows[base + 3842] = 0f
        } }
        val scalars = FloatArray(rows * 249) { (it % 19 - 9) / 9f }
        val env = OrtEnvironment.getEnvironment()
        for (model in listOf(RallyModels.F1, RallyModels.RECALL)) {
            val directory = if (model == RallyModels.F1) "f1" else "recall"
            val bytes = context.assets.open("serving-models/$directory/head.onnx").use { it.readBytes() }
            OrtSession.SessionOptions().use { options -> env.createSession(bytes, options).use { session ->
                val actual = NeuralServingRuntime.infer(session, windows, scalars, rows)
                val reference = golden.getJSONObject("variants").getJSONObject(RallyModels.variant(model)).getJSONArray("logits")
                assertEquals(reference.length(), actual.size)
                actual.indices.forEach { assertEquals(reference.getDouble(it), actual[it].toDouble(), 1e-5) }
            } }
        }
        Log.i("NeuralServingCheck", "Both heads passed independent ONNX golden")
        for ((argument, model) in listOf("balancedFeedback" to RallyModels.F1, "coverageFeedback" to RallyModels.RECALL)) {
            args.getString(argument)?.let { path ->
                val imported = ModelFeedbackImporter.parse(File(path).readText(), System.currentTimeMillis())
                assertEquals(model, imported.project.modelId)
                assertEquals(imported.project.ranges.size, imported.project.servingSide?.candidates?.size)
                assertEquals(ServingSideModelRunner.candidates(imported.project.ranges).map { it.id },
                    imported.project.servingSide?.candidates?.map { it.id })
                Log.i("NeuralServingCheck", "$model feedback import retained model range identities")
            }
        }
        if (args.getString("importFeedbackOnly") == "true") return
        val source = args.getString("servingSource") ?: return
        val uri = Uri.fromFile(File(source).also { assertTrue("Runtime input is unreadable; app external directory: $external", it.canRead()) })
        val media = AnalysisEngine(context).probe(uri)
        val embeddingCache = args.getString("servingEmbeddingCache")?.let { File(it).also(File::mkdirs) }
        val sourceHash = embeddingCache?.let { hash(File(source)) }
        val roi = AnalysisTypes.Roi(0.0, 0.0, 1.0, 1.0, "Full frame")
        val cancelled = AtomicBoolean(false)
        val receipts = JSONArray()
        for (model in listOf(RallyModels.F1, RallyModels.RECALL)) {
            val began = System.nanoTime()
            val fresh = if (args.getString("skipFresh") == "true") null else checkNotNull(DistilledRallyModels.open(context,model,media,roi,cancelled::get)).use { neural ->
                AnalysisEngine(context,neural).analyze(uri,true,Int.MAX_VALUE,AnalysisTypes.VideoDecoderOptions.defaults(),
                    NativeFeatureCache.Mode.USE,AnalysisTypes.AnalysisWindow(0.0,minOf(30.0,media.durationSeconds())),
                    AnalysisTypes.AnalysisStages.all(),AnalysisTypes.AudioDecoderMode.AUTO,cancelled,progress,true,false)
            }
            if (fresh != null) {
                assertNull(fresh.servingSideError())
                assertEquals(NeuralServingContract.identity(model).first, fresh.servingSide()?.modelId)
                assertEquals(fresh.ranges().size, fresh.servingSide()?.rows)
            }
            val feedbackPath = args.getString(if (model == RallyModels.F1) "balancedFeedback" else "coverageFeedback") ?: continue
            val feedback = JSONObject(File(feedbackPath).readText())
            assertEquals(model,feedback.getJSONObject("initialInference").getString("modelId"))
            val saved = ModelFeedbackImporter.decodeServingSide(feedback.getJSONObject("initialInference").getJSONObject("servingSide"))
            val candidates = saved.candidates.map { ServingSideModelRunner.CandidateInterval(it.id,it.anchor,it.intervalEnd,it.agreement) }
            val times = ServingSideInference.requestedTimestamps(candidates,media.durationSeconds())
            val decoded = SpecialistFrameDecoder(context).decodePrecedingServing(uri,media,roi,times,doubleArrayOf(),progress,cancelled::get)
            val raw = DoubleArray(saved.rows * 237)
            candidates.forEachIndexed { row,candidate ->
                fun frames(offsets: DoubleArray) = offsets.map { offset ->
                    checkNotNull(decoded.servingGray()[ServingSideInference.clampedTimestamp(candidate.start,offset,media.durationSeconds())])
                }.toTypedArray()
                ServingSideFeatureExtractor.extract(frames(COURT_FLOW_OFFSETS_SECONDS),frames(FLIGHT_OFFSETS_SECONDS)).copyInto(raw,row*237)
            }
            val output = checkNotNull(DistilledRallyModels.open(context,model,media,roi,cancelled::get,false)).use { neural ->
                val needed = NeuralServingContract.embeddingTimes(candidates.map { it.start }.toDoubleArray(), media.durationSeconds())
                val tensor = embeddingCache?.let { File(it,"$model.f32") }
                val metadata = embeddingCache?.let { File(it,"$model.json") }
                val encoder = JSONObject(context.assets.open("rally-models/manifest.json").use { it.readBytes().toString(Charsets.UTF_8) })
                    .getJSONObject("variants").getJSONObject(RallyModels.variant(model)).getJSONObject("files").getJSONObject("encoder").getString("sha256")
                if (tensor != null && metadata != null && (tensor.exists() || metadata.exists())) {
                    val descriptor = JSONObject(metadata.readText())
                    assertEquals(model, descriptor.getString("modelId")); assertEquals(sourceHash, descriptor.getString("sourceSha256"))
                    assertEquals(encoder, descriptor.getString("encoderSha256")); assertEquals(hash(tensor), descriptor.getString("tensorSha256"))
                    assertEquals(needed.size * 3840L * 4, tensor.length())
                    val times = descriptor.getJSONArray("times"); assertEquals(needed.size,times.length())
                    needed.indices.forEach { assertEquals(needed[it],times.getDouble(it),0.0) }
                    // Test-only reuse of actual, source/hash-qualified Android encoder output.
                    NeuralRallyPipeline::class.java.getDeclaredField("servingEmbedding").also { it.isAccessible=true }
                        .set(neural,NeuralRallyPipeline.Embedding(needed,NeuralRallyPipeline.readFloats(tensor)))
                    Log.i("NeuralServingCheck", "$model reused verified native image embeddings")
                }
                val native = NeuralServingRuntime.run(context,uri,media,roi,model,saved.copy(rawFeatures=raw),neural,progress,cancelled::get)
                val replay = NeuralServingRuntime.run(context,uri,media,roi,model,saved,neural,progress,cancelled::get)
                if (tensor != null && metadata != null && !tensor.exists()) {
                    val embedding = neural.servingEmbeddings(uri,roi,media.durationSeconds(),needed,progress)
                    NeuralRallyPipeline.writeFloats(tensor,FloatArray(needed.size*3840) { embedding.tokens().get(it) })
                    metadata.writeText(JSONObject().put("modelId",model).put("sourceSha256",sourceHash).put("encoderSha256",encoder)
                        .put("tensorSha256",hash(tensor)).put("times",JSONArray(needed)).toString(2))
                }
                JSONObject().put("modelId",model).put("seconds",(System.nanoTime()-began)/1e9).put("pipelineCandidates",fresh?.ranges()?.size ?: JSONObject.NULL)
                    .put("native",ServingSideJson.encodeOutput(native)).put("replay",ServingSideJson.encodeOutput(replay))
            }
            receipts.put(output)
            val folder = checkNotNull(context.getExternalFilesDir("serving-validation")); folder.mkdirs()
            File(folder,"serving-validation-output.json").writeText(receipts.toString(2))
            Log.i("NeuralServingCheck", "$model full native/replay complete")
        }
    }
    private fun hash(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { input ->
            val buffer = ByteArray(1024*1024)
            while (true) { val count=input.read(buffer); if(count<0) break; digest.update(buffer,0,count) }
        }
        return digest.digest().joinToString("") { "%02x".format(it.toInt() and 255) }
    }
}
