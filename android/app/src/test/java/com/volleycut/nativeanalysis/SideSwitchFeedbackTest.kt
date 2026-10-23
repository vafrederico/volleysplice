package com.volleycut.nativeanalysis

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Base64

class SideSwitchFeedbackTest {
    @Test
    fun reviewedWebProducerRoundTripsEvidencePreferenceCorrectionsAndReset() {
        for (enabled in listOf(true, false)) {
            val bundle = fixture().apply { getJSONObject("source").put("generateSideSwitchMarkers", enabled) }
            val imported = ModelFeedbackImporter.parse(bundle.toString())
            val output = requireNotNull(imported.project.sideSwitch)
            assertEquals(3, output.rows) // All proposals, including the unselected row.
            assertEquals(listOf(25.0, 45.0), output.candidates.map { it.timestamp })
            assertEquals(SideSwitchCandidateKind.entries.toList(), output.candidates.map { it.kind })
            assertArrayEquals(DoubleArray(102) { (it - 50) / 8.0 }, output.features, 0.0)
            assertEquals(enabled, imported.project.sideSwitchEnabled)
            assertEquals(ServingSideAnalysisStatus.READY, imported.project.servingSideStatus)
            assertReview(imported.draft.scoreTracking)

            val stored = NativeProjectStore.normalizeStored(requireNotNull(
                NativeProjectStore.decode(NativeProjectStore.encode(imported.project)),
            ))
            assertEquals(output, stored.sideSwitch)
            assertEquals(enabled, stored.sideSwitchEnabled)
            assertEquals(ServingSideAnalysisStatus.READY, stored.servingSideStatus)
            val seed = requireNotNull(EditorProjectStore.decode(EditorProjectStore.encode(requireNotNull(stored.editorSeed()))))
            assertEquals(output, seed.sideSwitch)
            assertEquals(enabled, seed.sideSwitchEnabled)
            assertEquals(stored.feedbackWarnings, seed.feedbackWarnings)
            assertEquals(stored.feedbackWarnings, NativeProjectStore.fromSeed(seed, stored.source, stored.media).feedbackWarnings)
            assertReview(ScoreReducer.seedModelMarkers(imported.draft.scoreTracking, seed.servingSide, seed.sideSwitch, enabled))

            val reexport = export(stored, imported.draft)
            assertEquals(enabled, reexport.getJSONObject("source").getBoolean("generateSideSwitchMarkers"))
            assertEquals(output, ModelFeedbackImporter.parse(reexport.toString()).project.sideSwitch)
            assertReview(ModelFeedbackImporter.parse(reexport.toString()).draft.scoreTracking)
            assertTrue(reexport.getJSONArray("warnings").toString().contains("Synthetic retained warning"))
            assertEquals(reexport.getJSONArray("warnings").toString(),
                export(ModelFeedbackImporter.parse(reexport.toString()).project, imported.draft).getJSONArray("warnings").toString())

            // Consumed by the reviewed web production validator in the integration check.
            File("build/side-switch-contract").apply { mkdirs() }
                .resolve("android-$enabled.json").writeText(reexport.toString())
            val reset = EditorMath.newDraft(seed).scoreTracking
            assertTrue(reset.removedModelMarkerIds.isEmpty())
            assertEquals(if (enabled) listOf(25_000L, 45_000L) else emptyList<Long>(), reset.sideSwitchMarkers.map { it.timestampMs })
            assertTrue(reset.sideSwitchMarkers.all { it.origin == ServeMarkerOrigin.MODEL })
            assertEquals(output, stored.sideSwitch)
        }
    }

    @Test
    fun absentLegacyPreferenceDefaultsTrueAndMissingOutputDiffersFromHealthyZeroRows() {
        for (absent in listOf(true, false)) {
            val bundle = fixture()
            bundle.getJSONObject("source").remove("generateSideSwitchMarkers")
            val inference = bundle.getJSONObject("initialInference")
            if (absent) inference.remove("sideSwitch") else inference.put("sideSwitch", JSONObject.NULL)
            val imported = ModelFeedbackImporter.parse(bundle.toString())
            assertTrue(imported.project.sideSwitchEnabled)
            assertNull(imported.project.sideSwitch)
            assertEquals(ServingSideAnalysisStatus.NOT_RUN, imported.project.servingSideStatus)
            assertReview(imported.draft.scoreTracking)
            assertTrue(EditorMath.newDraft(requireNotNull(imported.project.editorSeed())).scoreTracking.sideSwitchMarkers.isEmpty())
        }
        for (enabled in listOf(true, false)) {
            val bundle = fixture()
            bundle.getJSONObject("source").put("generateSideSwitchMarkers", enabled)
            val side = side(bundle)
            side.put("candidates", JSONArray())
            side.getJSONObject("features").put("rows", 0)
                .put("values", ModelFeedbackExporter.encode(doubleArrayOf(), intArrayOf(0, 34)))
            val imported = ModelFeedbackImporter.parse(bundle.toString())
            assertEquals(SideSwitchModelRunner.emptyOutput(), imported.project.sideSwitch)
            val stored = NativeProjectStore.normalizeStored(requireNotNull(NativeProjectStore.decode(NativeProjectStore.encode(imported.project))))
            assertEquals(ServingSideAnalysisStatus.READY, stored.servingSideStatus)
            assertEquals(imported.project.sideSwitch, ModelFeedbackImporter.parse(export(stored, imported.draft).toString()).project.sideSwitch)
            assertReview(imported.draft.scoreTracking)
        }
    }

    @Test
    fun archivedIdentityRemainsEvidenceButCannotSatisfyCurrentInferenceRequest() {
        val bundle = fixture()
        side(bundle).put("modelId", "synthetic-archived-switch-model").put("modelFingerprint", "synthetic-archived-fingerprint")
        val imported = ModelFeedbackImporter.parse(bundle.toString())
        val output = requireNotNull(imported.project.sideSwitch)
        val stored = NativeProjectStore.normalizeStored(requireNotNull(NativeProjectStore.decode(NativeProjectStore.encode(imported.project))))
        assertEquals(output, stored.sideSwitch)
        assertEquals(output, ModelFeedbackImporter.parse(export(stored, imported.draft).toString()).project.sideSwitch)
        assertFalse(SideSwitchJson.isCurrentOutput(output))
        assertFalse(NativeProjectStore.matchesAnalysis(stored, stored.copy(status = ProjectStatus.QUEUED)))
        val current = stored.copy(sideSwitch = output.copy(modelId = SIDE_SWITCH_MODEL_ID, modelFingerprint = SIDE_SWITCH_MODEL_FINGERPRINT))
        assertTrue(NativeProjectStore.matchesAnalysis(current, current.copy(status = ProjectStatus.QUEUED)))
        // An explicit new request may enable a current cached detector or prepare missing output.
        assertTrue(NativeProjectStore.matchesAnalysis(current.copy(sideSwitchEnabled = false), current))
        assertTrue(NativeProjectStore.matchesAnalysis(current.copy(sideSwitch = null), current))
        assertReview(imported.draft.scoreTracking)
    }

    @Test
    fun malformedSideSwitchAndPreferenceFailInProductionParser() {
        val mutations = listOf<(JSONObject) -> Unit>(
            { it.getJSONObject("source").put("generateSideSwitchMarkers", "false") },
            { it.getJSONObject("source").put("generateSideSwitchMarkers", JSONObject.NULL) },
            { it.getJSONObject("initialInference").put("sideSwitch", false) },
            { side(it).put("modelId", "") },
            { side(it).put("modelFingerprint", 7) },
            { side(it).put("featureVersion", "unsupported") },
            { side(it).put("candidateContract", "unsupported") },
            { side(it).getJSONObject("features").put("rows", -1) },
            { side(it).getJSONObject("features").put("rows", 3.5) },
            { side(it).getJSONObject("features").put("rows", "3") },
            { side(it).getJSONObject("features").put("rows", Int.MAX_VALUE) },
            { side(it).getJSONObject("features").put("columns", 33) },
            { values(it).put("shape", JSONArray(listOf(34, 3))) },
            { values(it).put("shape", JSONArray(listOf(3, 34, 1))) },
            { values(it).put("encoding", "raw") },
            { values(it).put("byteOrder", "big-endian") },
            { values(it).put("dataType", "float32") },
            { values(it).put("data", "AAAA") },
            { values(it).put("data", values(it).getString("data") + "\n") },
            { values(it).put("data", "!") },
            { values(it).put("data", encodedNonFinite(Double.NaN)) },
            { values(it).put("data", encodedNonFinite(Double.POSITIVE_INFINITY)) },
            { candidate(it).put("timestamp", -0.01) },
            { candidate(it).put("timestamp", 60.01) },
            { candidate(it).put("timestamp", "25") },
            { candidate(it).put("probability", 1.01) },
            { candidate(it).put("probability", "0.9") },
            { candidate(it).put("kind", "unsupported") },
            { candidate(it).put("id", "") },
            { candidate(it).put("sourceRangeIds", JSONArray(listOf(1))) },
            { candidate(it).put("sourceRangeIds", JSONArray(listOf(""))) },
            { side(it).getJSONArray("candidates").put(candidate(it)) },
            { side(it).put("candidates", JSONArray(List(4) { index -> JSONObject(candidate(it).toString()).put("id", "event-$index") })) },
        )
        mutations.forEachIndexed { index, mutate ->
            val bundle = fixture().also(mutate)
            assertThrows("Malformed case $index", Exception::class.java) { ModelFeedbackImporter.parse(bundle.toString()) }
        }
    }

    private fun assertReview(score: ScoreTracking) {
        assertEquals(listOf("switch-boundary-1", "manual-switch"), score.sideSwitchMarkers.map { it.id })
        assertEquals(listOf(35_000L, 40_000L), score.sideSwitchMarkers.map { it.timestampMs })
        assertEquals(setOf("switch-internal-2"), score.removedModelMarkerIds)
        assertEquals("boundary-1", score.sideSwitchMarkers.first().modelEventId)
        assertEquals(.95, requireNotNull(score.sideSwitchMarkers.first().modelConfidence), 0.0)
    }

    private fun fixture() = JSONObject(requireNotNull(javaClass.getResource("/side-switch-web-project.json")).readText())
    private fun side(bundle: JSONObject) = bundle.getJSONObject("initialInference").getJSONObject("sideSwitch")
    private fun values(bundle: JSONObject) = side(bundle).getJSONObject("features").getJSONObject("values")
    private fun candidate(bundle: JSONObject) = side(bundle).getJSONArray("candidates").getJSONObject(0)
    private fun export(project: NativeProject, draft: EditorDraft) = ModelFeedbackExporter.createBundle(
        project, draft, EditorMath.finalIntervals(draft), null, project.source.sampledFingerprint,
    )
    private fun encodedNonFinite(value: Double) = Base64.getEncoder().encodeToString(
        ByteBuffer.allocate(102 * 8).order(ByteOrder.LITTLE_ENDIAN).putDouble(value).array(),
    )
}
