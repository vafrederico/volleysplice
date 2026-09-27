package com.volleycut.nativeanalysis

import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest

class SideSwitchModelsTest {
    @Test
    fun runtimeIdentityFeatureOrderAndCanonicalHashStayFrozen() {
        val raw = Files.readAllBytes(runtimePath())
        val canonical = raw.toString(Charsets.UTF_8).replace("\r\n", "\n").toByteArray()
        val hash = MessageDigest.getInstance("SHA-256").digest(canonical)
            .joinToString("") { "%02x".format(it.toInt() and 0xff) }
        assertEquals("ab4197545fb916a37ee6ac1d69e74ddfc0123c09039cdfa88c4ef378dd3e27fc", hash)
        val json = JSONObject(raw.toString(Charsets.UTF_8))
        val names = json.getJSONObject("classifier").getJSONArray("featureNames")
        assertEquals(SIDE_SWITCH_FEATURE_NAMES, List(names.length()) { names.getString(it) })
        val runtime = SideSwitchRuntimeParser.parse(json)
        assertEquals(.39884973953581804, runtime.threshold, 0.0)

        val wrong = JSONObject(json.toString()).put("fingerprint", "changed")
        assertThrows(IllegalArgumentException::class.java) { SideSwitchRuntimeParser.parse(wrong) }
    }

    @Test
    fun classifierMatchesFrozenBrowserScores() {
        val runtime = runtime()
        assertEquals(
            .0002845953292207428,
            SideSwitchModelRunner.predict(runtime, DoubleArray(34)).single(),
            1e-16,
        )
        assertEquals(
            .9966496738167576,
            SideSwitchModelRunner.predict(runtime, DoubleArray(34) { .5 }).single(),
            1e-15,
        )
    }

    @Test
    fun candidateGeneratorUsesBoundariesAndScoreRankedDeadPeaks() {
        val times = DoubleArray(31) { it.toDouble() }
        val dead = FloatArray(times.size)
        dead[5] = .99f
        dead[6] = 1f
        dead[25] = .99f
        val input = input(times, dead)
        val candidates = SideSwitchModelRunner.generateCandidates(input, runtime())
        assertEquals(
            listOf(
                SideSwitchCandidateKind.INTERNAL_DEAD_STATE_PEAK,
                SideSwitchCandidateKind.ADJACENT_RALLY_BOUNDARY,
                SideSwitchCandidateKind.INTERNAL_DEAD_STATE_PEAK,
            ),
            candidates.map { it.kind },
        )
        assertEquals(listOf(6.0, 15.0, 25.0), candidates.map { it.transitionTime })
        assertEquals("switch:boundary:R001:R002", candidates[1].id)
        assertArrayEquals(
            doubleArrayOf(
                .2, 1.8083333333333333, 3.4166666666666665, 5.025,
                6.633333333333333, 8.241666666666667, 9.85,
            ),
            SideSwitchModelRunner.sampleTimes(0.0, 10.0),
            1e-12,
        )
    }

    @Test
    fun stateFeaturesAndOrdinalDecoderMatchFrozenContract() {
        val times = DoubleArray(31) { it.toDouble() }
        val input = input(times, FloatArray(times.size))
        val candidates = SideSwitchModelRunner.generateCandidates(input, runtime())
        val boundary = candidates.single()
        val state = SideSwitchModelRunner.stateFeatures(boundary, input)
        assertArrayEquals(
            doubleArrayOf(2.0, 2.0, 2.0, .4, 0.0, .4, .4, .8, .8, 10.0),
            state,
            1e-7,
        )

        val proposals = (0..3).map { index -> boundary.copy(
            id = "switch-$index", transitionTime = index.toDouble(),
        ) }
        assertEquals(
            listOf(0, 2),
            SideSwitchModelRunner.decode(runtime(), proposals, doubleArrayOf(.9, .89, .88, .87)),
        )
    }

    @Test
    fun stateAndModelOutputRoundTripCompactBinaryPayloads() {
        val times = doubleArrayOf(1.0, 2.0)
        val states = stateOutputs(times, floatArrayOf(.2f, .3f), floatArrayOf(.8f, .7f))
        assertEquals(states, SideSwitchJson.decodeStateOutputs(SideSwitchJson.encodeStateOutputs(states)))
        val output = SideSwitchOutput(
            rows = 1,
            features = DoubleArray(34) { it / 34.0 },
            candidates = listOf(SideSwitchPrediction(
                "switch:event", 1.5, .9,
                SideSwitchCandidateKind.ADJACENT_RALLY_BOUNDARY,
                listOf("R001", "R002"),
            )),
        )
        assertEquals(output, SideSwitchJson.decodeOutput(SideSwitchJson.encodeOutput(output)))
        assertTrue(SideSwitchJson.validOutput(output))
    }

    @Test
    fun sharedDecoderFreezesFiveSecondPositiveGapRule() {
        assertEquals(5.0, SpecialistFrameDecoder.MAX_SEQUENTIAL_GAP_SECONDS, 0.0)
        assertEquals(
            2,
            SpecialistFrameDecoder.plannedSegmentCount(doubleArrayOf(0.0, 1.0, 6.0, 11.0001)),
        )
    }

    @Test
    fun sharedDecoderDoesNotConvertEveryFrameBeforeOrdinaryTargets() {
        assertFalse(SpecialistFrameDecoder.shouldConvertOutput(
            9_500_000, 10_000_000, 60_000_000, false,
        ))
        assertTrue(SpecialistFrameDecoder.shouldConvertOutput(
            10_000_000, 10_000_000, 60_000_000, false,
        ))
        assertTrue(SpecialistFrameDecoder.shouldConvertOutput(
            59_000_000, 59_500_000, 60_000_000, false,
        ))
    }

    @Test
    fun terminalFrameRetainsFormatsForTheNextRequestBeforeEmptyEos() {
        // Last image satisfies request 0; request 1 is 6.6ms beyond the last
        // real PTS. Its different pixel format must be prepared before release.
        assertEquals(2, SpecialistFrameDecoder.terminalConversionEnd(
            1, 2, 1_060_999_856, 1_061_006_489, 1_061_016_489,
        ))
        assertEquals(1, SpecialistFrameDecoder.terminalConversionEnd(
            1, 2, 10_000_000, 10_010_000, 60_000_000,
        ))
        assertEquals(1, SpecialistFrameDecoder.terminalConversionEnd(
            1, 2, 58_000_000, 59_990_000, 60_000_000,
        ))
        assertEquals(2, SpecialistFrameDecoder.terminalConversionEnd(
            2, 2, 59_999_000, 59_990_000, 60_000_000,
        ))
    }

    private fun input(times: DoubleArray, dead: FloatArray): SideSwitchAnalysisInput {
        val ranges = listOf(
            AnalysisTypes.Interval(0.0, 10.0, .9f, ProductionEnsemble.BOTH_MODELS),
            AnalysisTypes.Interval(20.0, 30.0, .9f, ProductionEnsemble.BOTH_MODELS),
        )
        return SideSwitchAnalysisInput(
            ranges,
            AnalysisTypes.ProductionComponents(ranges, ranges),
            stateOutputs(times, FloatArray(times.size) { .4f }, dead.map { maxOf(it, .8f) }.toFloatArray()),
        )
    }

    private fun stateOutputs(
        times: DoubleArray,
        rally: FloatArray,
        dead: FloatArray,
    ) = AnalysisTypes.ProductionStateOutputs(
        AnalysisTypes.ProductionStateOutput(
            FeatureSchema.ALL_LABELS_V2_MODEL_ID, times.clone(), rally.clone(), dead.clone(),
        ),
        AnalysisTypes.ProductionStateOutput(
            FeatureSchema.PREVIOUS_PRODUCTION_MODEL_ID, times.clone(), rally.clone(), dead.clone(),
        ),
    )

    private fun runtime() = SideSwitchRuntimeParser.parse(JSONObject(Files.readString(runtimePath())))

    private fun runtimePath(): Path = repositoryRoot()
        .resolve("android/app/src/main/assets/$SIDE_SWITCH_RUNTIME_ASSET")

    private fun repositoryRoot(): Path {
        var current: Path? = Path.of(System.getProperty("user.dir")).toAbsolutePath()
        while (current != null) {
            if (Files.exists(current.resolve("android/app/src/main/assets/$SIDE_SWITCH_RUNTIME_ASSET"))) {
                return current
            }
            current = current.parent
        }
        error("Could not locate repository root")
    }
}
