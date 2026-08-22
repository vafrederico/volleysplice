package com.volleycut.nativeanalysis

import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayInputStream
import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest

class ServingSideModelsTest {
    @Test
    fun generatedFeatureNamesMatchFrozenRuntimeInExactOrder() {
        val names = ServingSideFeatureNames.all()
        assertEquals(82, ServingSideFeatureNames.courtFlow().size)
        assertEquals(155, ServingSideFeatureNames.flight().size)
        assertEquals(237, names.size)
        val json = runtimeJson()
        val assetNames = json.getJSONArray("featureNames")
        assertEquals(names, List(assetNames.length()) { assetNames.getString(it) })
    }

    @Test
    fun runtimeAssetHasCanonicalHashAndRejectsMalformedPolicy() {
        val raw = Files.readAllBytes(runtimePath())
        val canonical = raw.toString(Charsets.UTF_8).replace("\r\n", "\n").toByteArray()
        val hash = MessageDigest.getInstance("SHA-256").digest(canonical)
            .joinToString("") { "%02x".format(it.toInt() and 0xff) }
        assertEquals(SERVING_SIDE_RUNTIME_SHA256, hash)
        val runtime = ServingSideRuntimeParser.read(ByteArrayInputStream(raw))
        assertEquals(237, runtime.featureNames.size)

        val wrongIdentity = JSONObject(runtimeJson().toString()).put("featureVersion", "wrong")
        assertThrows(IllegalArgumentException::class.java) { ServingSideRuntimeParser.parse(wrongIdentity) }
        val wrongScale = JSONObject(runtimeJson().toString())
        wrongScale.getJSONObject("model").getJSONArray("scale").put(0, 0.0)
        assertThrows(IllegalArgumentException::class.java) { ServingSideRuntimeParser.parse(wrongScale) }
    }

    @Test
    fun tiedRanksMatchStableMidrankContractIncludingSingleton() {
        assertArrayEquals(
            doubleArrayOf(.25, 0.0, .25, 1.0, 1.0, .5),
            ServingSideModelRunner.tiedPercentileRanks(
                doubleArrayOf(2.0, 1.0, 2.0, 3.0, 4.0, 2.0), 3, 2,
            ),
            0.0,
        )
        assertArrayEquals(
            doubleArrayOf(.5, .5, .5),
            ServingSideModelRunner.tiedPercentileRanks(doubleArrayOf(8.0, 9.0, 10.0), 1, 3),
            0.0,
        )
    }

    @Test
    fun frozenHalfRankLogisticScoreMatchesBrowser() {
        val runtime = ServingSideRuntimeParser.parse(runtimeJson())
        assertEquals(
            0.5428339645988896,
            ServingSideModelRunner.nearProbability(DoubleArray(237) { .5 }, runtime),
            1e-14,
        )
    }

    @Test
    fun serveEvidenceIncludesOneSecondBoundaryUsesFirstPeakAndNearestFallback() {
        val output = AnalysisTypes.ProductionServeOutput(
            "model", doubleArrayOf(3.0, 4.0, 5.0, 7.0),
            floatArrayOf(.9f, .2f, .9f, 1f),
            listOf(AnalysisTypes.Serve(4.4, .91f), AnalysisTypes.Serve(6.0, .99f)),
        )
        val evidence = ServingSideModelRunner.serveHeadEvidence(output, 4.0)
        assertEquals(3.0, evidence.peakTime, 0.0)
        assertTrue(evidence.crossesThreshold)
        assertEquals(4.4, evidence.nearestDetection?.time() ?: error("missing detection"), 0.0)

        val fallback = ServingSideModelRunner.serveHeadEvidence(output, 10.0, windowSeconds = .1)
        assertEquals(7.0, fallback.peakTime, 0.0)
        assertTrue(fallback.crossesThreshold)
    }

    @Test
    fun bothModelRallyRecoversServeButRequiresReview() {
        val runtime = ServingSideRuntimeParser.parse(runtimeJson())
        val output = AnalysisTypes.ProductionServeOutput(
            "model", doubleArrayOf(1.0), floatArrayOf(.2f), emptyList(),
        )
        val verdict = ServingSideModelRunner.verdict(
            ServingSideModelRunner.CandidateInterval("R001", 1.0, 2.0, ProductionEnsemble.BOTH_MODELS),
            DoubleArray(237) { .5 }, output, output, runtime,
        )
        assertEquals(ServingSideDecisionSource.PRODUCTION_RALLY_RECOVERY, verdict.serveDecisionSource)
        assertEquals(ServingSideVerdict.REVIEW, verdict.verdict)
        assertTrue(ServingSideReviewReason.PRODUCTION_RALLY_RECOVERY in verdict.reviewReasons)

        val missed = ServingSideModelRunner.verdict(
            ServingSideModelRunner.CandidateInterval(
                "R002", 1.0, 2.0, ProductionEnsemble.ALL_LABELS_V2_ONLY,
            ),
            DoubleArray(237) { .5 }, output, output, runtime,
        )
        assertEquals(ServingSideVerdict.NOT_SERVE, missed.verdict)
        assertEquals(ServingSideDecisionSource.NONE, missed.serveDecisionSource)
    }

    @Test
    fun framePlanClampsSourceStartAndEndAndDeduplicatesSharedSamples() {
        val candidates = listOf(
            ServingSideModelRunner.CandidateInterval("R001", .1, 1.0, null),
            ServingSideModelRunner.CandidateInterval("R002", 9.9, 10.0, null),
        )
        val requested = ServingSideInference.requestedTimestamps(candidates, 10.0)
        assertEquals(0.0, requested.first(), 0.0)
        assertEquals(9.99, requested.last(), 0.0)
        assertEquals(requested.size, requested.distinct().size)
        assertEquals(0.0, ServingSideInference.clampedTimestamp(.1, -1.25, 10.0), 0.0)
        assertEquals(9.99, ServingSideInference.clampedTimestamp(9.9, 1.75, 10.0), 0.0)
    }

    private fun runtimeJson() = JSONObject(Files.readString(runtimePath()))

    private fun runtimePath(): Path = repositoryRoot()
        .resolve("android/app/src/main/assets/$SERVING_SIDE_RUNTIME_ASSET")

    private fun repositoryRoot(): Path {
        var current: Path? = Path.of(System.getProperty("user.dir")).toAbsolutePath()
        while (current != null) {
            if (Files.exists(current.resolve("android/app/src/main/assets/$SERVING_SIDE_RUNTIME_ASSET"))) {
                return current
            }
            current = current.parent
        }
        error("Could not locate repository root")
    }
}
