package com.volleycut.nativeanalysis

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.net.Uri
import org.json.JSONObject
import java.nio.FloatBuffer

internal object NeuralServingRuntime {
    fun run(context: Context, uri: Uri, media: AnalysisTypes.MediaInfo, roi: AnalysisTypes.Roi,
            modelId: String, specialist: ServingSideOutput, pipeline: NeuralRallyPipeline?,
            progress: AnalysisTypes.ProgressListener, cancelled: () -> Boolean): ServingSideOutput {
        val rally = JSONObject(context.assets.open("rally-models/manifest.json").use { it.readBytes().toString(Charsets.UTF_8) })
            .getJSONObject("variants").getJSONObject(RallyModels.variant(modelId))
        val bundle = NeuralServingContract.bundle(context.assets.open("serving-models/manifest.json").use { it.readBytes() }, modelId, rally)
        val output = specialist.copy(modelId = bundle.id, modelFingerprint = bundle.fingerprint)
        if (output.candidates.isEmpty()) return output
        val active = pipeline ?: DistilledRallyModels.open(context, modelId, media, roi, cancelled, false)
        try {
            require(active.modelId() == modelId) { "Serving embeddings belong to a different encoder" }
            val needed = NeuralServingContract.embeddingTimes(output.candidates.map { it.anchor }.toDoubleArray(), media.durationSeconds())
            val embedding = active.servingEmbeddings(uri, roi, media.durationSeconds(), needed, progress)
            val bytes = context.assets.open("serving-models/${bundle.directory}/head.onnx").use { it.readBytes() }
            require(bytes.size.toLong() == bundle.size && NeuralServingContract.hash(bytes) == bundle.sha256) { "Serving model integrity check failed" }
            val features = NeuralServingContract.features(output.rawFeatures, output.rows)
            val env = OrtEnvironment.getEnvironment()
            val candidates = output.candidates.toMutableList()
            OrtSession.SessionOptions().use { options ->
                options.setIntraOpNumThreads(4); options.setInterOpNumThreads(1)
                env.createSession(bytes, options).use { session ->
                    for (start in candidates.indices step 8) {
                        check(!cancelled() && !Thread.currentThread().isInterrupted) { "Analysis cancelled" }
                        val count = minOf(8, candidates.size - start)
                        val windows = FloatArray(count * 13 * 3843)
                        repeat(count) { row ->
                            NeuralServingContract.window(embedding.times(), embedding.tokens(), candidates[start + row].anchor, media.durationSeconds())
                                .copyInto(windows, row * 13 * 3843)
                        }
                        val logits = infer(session, windows, features.copyOfRange(start * 249, (start + count) * 249), count)
                        repeat(count) { row -> candidates[start + row] = NeuralServingContract.apply(candidates[start + row],
                            NeuralServingContract.decision(logits.copyOfRange(row * 6, row * 6 + 6), bundle)) }
                        progress.onProgress("serving-side", (start + count).toDouble() / candidates.size, "Checking serve sides: ${start + count}/${candidates.size}")
                    }
                }
            }
            return output.copy(candidates = candidates)
        } finally { if (pipeline == null) active.close() }
    }
    internal fun infer(session: OrtSession, windows: FloatArray, features: FloatArray, rows: Int): FloatArray {
        val env = OrtEnvironment.getEnvironment()
        OnnxTensor.createTensor(env, FloatBuffer.wrap(windows), longArrayOf(rows.toLong(), 13, 3843)).use { images ->
            OnnxTensor.createTensor(env, FloatBuffer.wrap(features), longArrayOf(rows.toLong(), 249)).use { scalars ->
                session.run(mapOf("windows" to images, "features" to scalars)).use { result ->
                    val tensor = result[0] as OnnxTensor
                    require(tensor.info.shape.contentEquals(longArrayOf(rows.toLong(), 3, 2))) { "Serving output shape changed" }
                    return FloatArray(rows * 6).also { tensor.floatBuffer.get(it); require(it.all(Float::isFinite)) }
                }
            }
        }
    }
}
