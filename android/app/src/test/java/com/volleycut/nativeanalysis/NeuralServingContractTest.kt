package com.volleycut.nativeanalysis

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.nio.FloatBuffer
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs
import kotlin.math.ln

class NeuralServingContractTest {
    private fun root(): Path = generateSequence(Path.of(System.getProperty("user.dir")).toAbsolutePath()) { it.parent }
        .first { Files.exists(it.resolve("models/serving-side/android-manifest.json")) }
    private fun bundle(model: String): NeuralServingContract.Bundle {
        val rally = JSONObject(Files.readString(root().resolve("models/distilled-large/android-manifest.json")))
            .getJSONObject("variants").getJSONObject(RallyModels.variant(model))
        return NeuralServingContract.bundle(Files.readAllBytes(root().resolve("models/serving-side/android-manifest.json")), model, rally)
    }
    @Test fun consensusUsesMatchedEncoderCutoffsAndVetoesDisagreement() {
        val coverage = bundle(RallyModels.RECALL); val balanced = bundle(RallyModels.F1)
        fun logits(far: Double, near: Double) = FloatArray(6) { i -> (if (i % 2 == 0) ln(far / (1 - far)) else ln(near / (1 - near))).toFloat() }
        assertTrue(NeuralServingContract.decision(logits(.01, .46), coverage).accepted)
        assertFalse(NeuralServingContract.decision(logits(.01, .46), balanced).accepted)
        val strong = logits(.01, .7)
        assertTrue(NeuralServingContract.decision(strong, balanced).accepted)
        strong[4] = logits(.7, .01)[0]; strong[5] = logits(.7, .01)[1]
        assertFalse(NeuralServingContract.decision(strong, balanced).accepted)
        assertThrows(IllegalArgumentException::class.java) { NeuralServingContract.decision(logits(.9, .9), coverage) }
        assertEquals(NeuralServingContract.identity(RallyModels.RECALL).first, coverage.id)
        assertEquals(NeuralServingContract.identity(RallyModels.F1).second, balanced.fingerprint)
    }
    @Test fun manifestRejectsTamperingAndWrongEncoder() {
        val bytes = Files.readAllBytes(root().resolve("models/serving-side/android-manifest.json"))
        val rally = JSONObject(Files.readString(root().resolve("models/distilled-large/android-manifest.json")))
            .getJSONObject("variants").getJSONObject("high-f1")
        assertThrows(IllegalArgumentException::class.java) { NeuralServingContract.bundle(bytes + byteArrayOf(32),RallyModels.F1,rally) }
        rally.getJSONObject("files").getJSONObject("encoder").put("sha256","wrong")
        assertThrows(IllegalArgumentException::class.java) { NeuralServingContract.bundle(bytes,RallyModels.F1,rally) }
    }
    @Test fun finiteRecordingRanksAndUnrankedMotionQuality() {
        val raw = DoubleArray(474) { if (it < 237) -2.0 else 4.0 }
        val features = NeuralServingContract.features(raw, 2)
        assertArrayEquals(FloatArray(237), features.copyOfRange(0, 237), 0f)
        assertArrayEquals(FloatArray(237) { 1f }, features.copyOfRange(249, 486), 0f)
        assertArrayEquals(floatArrayOf(-2f,-2f,-2f,2f,-2f,-2f,-2f,2f,-2f,-2f,-2f,2f), features.copyOfRange(237,249), 0f)
        assertThrows(IllegalArgumentException::class.java) { NeuralServingContract.features(DoubleArray(237) { Double.NaN }, 1) }
    }
    @Test fun sparseWindowsMatchIndependentNearestSearchAndVideoMasks() {
        val duration = 10.2; val times = DoubleArray(21) { it / 2.0 }; val tokens = FloatArray(21 * 3840) { (it / 3840).toFloat() }
        for (anchor in doubleArrayOf(0.0,.1,.25,1.25,3.13,4.25,7.999,10.1)) {
            val actual = NeuralServingContract.window(times, FloatBuffer.wrap(tokens), anchor, duration)
            val sparse = NeuralServingContract.embeddingTimes(doubleArrayOf(anchor), duration)
            val sparseTokens = FloatArray(sparse.size * 3840) { (sparse[it / 3840] * 2).toFloat() }
            assertArrayEquals(actual, NeuralServingContract.window(sparse, FloatBuffer.wrap(sparseTokens), anchor, duration), 0f)
            repeat(13) { frame ->
                val target = anchor - 3 + frame / 2.0; val nearest = times.minBy { abs(it - target) }
                val valid = target >= 0 && target < duration && abs(nearest - target) <= .250001
                assertEquals(if (valid) (nearest * 2).toFloat() else 0f, actual[frame * 3843], 0f)
                assertEquals(if (valid) 1f else 0f, actual[frame * 3843 + 3840], 0f)
            }
        }
    }
    @Test fun servingPersistenceRejectsStaleAndCrossEncoderOutputWithoutClearingGateReview() {
        val evidence = ServingSideHeadEvidence(FeatureSchema.ALL_LABELS_V2_MODEL_ID, .85, .9, 1.0, true, null)
        val candidate = ServingSideCandidate("R001",1.0,1.0,3.0,"neural",.5,ServingSide.FAR,
            ServingSideVerdict.REVIEW,ServingSideDecisionSource.NEURAL_RALLY_RECOVERY,
            listOf(ServingSideReviewReason.SIDE_SCORE,ServingSideReviewReason.NEURAL_RALLY_RECOVERY),evidence,evidence.copy(modelId=FeatureSchema.PREVIOUS_PRODUCTION_MODEL_ID))
        val updated = NeuralServingContract.apply(candidate, NeuralServingContract.Decision(.99,ServingSide.NEAR,true))
        assertEquals(ServingSideVerdict.REVIEW,updated.verdict)
        assertEquals(listOf(ServingSideReviewReason.NEURAL_RALLY_RECOVERY),updated.reviewReasons)
        val identity = NeuralServingContract.identity(RallyModels.RECALL)
        val output = ServingSideOutput(identity.first,identity.second,rows=1,rawFeatures=DoubleArray(237),candidates=listOf(updated))
        val ranges = listOf(SeedRange(1000,3000,1f,"neural"))
        assertTrue(output.isReusableFor(ranges,RallyModels.RECALL))
        assertFalse(output.isReusableFor(ranges,RallyModels.F1))
        assertFalse(output.isReusableFor(ranges))
        assertNotNull(ServingSideJson.decodeOutput(ServingSideJson.encodeOutput(output)))
        assertFalse(output.copy(modelFingerprint="stale").isReusableFor(ranges,RallyModels.RECALL))
    }
}
