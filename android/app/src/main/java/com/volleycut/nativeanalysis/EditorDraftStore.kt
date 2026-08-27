package com.volleycut.nativeanalysis

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption

internal class EditorDraftStore(context: Context, private val seed: EditorSeed) {
    private val directory = File(context.filesDir, "editor-drafts")
    private val target = File(directory, "${seed.sourceRevision}.json")

    fun load(): EditorDraft? = runCatching {
        if (!target.isFile) return null
        decode(JSONObject(target.readText()))?.takeIf { it.sourceRevision == seed.sourceRevision }
    }.onFailure { Log.w(TAG, "Could not restore editor draft", it) }.getOrNull()

    @Synchronized
    fun save(draft: EditorDraft) {
        runCatching {
            directory.mkdirs()
            val temporary = File(directory, "${target.name}.tmp")
            temporary.writeText(encode(draft).toString())
            try {
                Files.move(
                    temporary.toPath(), target.toPath(),
                    StandardCopyOption.REPLACE_EXISTING,
                    StandardCopyOption.ATOMIC_MOVE,
                )
            } catch (_: Exception) {
                Files.move(temporary.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING)
            }
        }.onFailure { Log.e(TAG, "Could not persist editor draft", it) }
    }

    fun clear() {
        if (target.exists() && !target.delete()) Log.w(TAG, "Could not delete ${target.name}")
    }

    private fun encode(draft: EditorDraft) = JSONObject().apply {
        put("version", draft.version)
        put("sourceRevision", draft.sourceRevision)
        put("updatedAtMs", draft.updatedAtMs)
        put("beforePaddingMs", draft.beforePaddingMs)
        put("afterPaddingMs", draft.afterPaddingMs)
        put("joinGapMs", draft.joinGapMs)
        put("pendingManualStartMs", draft.pendingManualStartMs ?: JSONObject.NULL)
        put("pendingIgnoreStartMs", draft.pendingIgnoreStartMs ?: JSONObject.NULL)
        put("ignoreReason", draft.ignoreReason)
        put("finalPreviewEnabled", draft.finalPreviewEnabled)
        put("playbackRate", draft.playbackRate.toDouble())
        put("confidenceReviewThreshold", draft.confidenceReviewThreshold.toDouble())
        put("reviewedCutIds", JSONArray(draft.reviewedCutIds.sorted()))
        put("selectedSuppressionPolicy", draft.selectedSuppressionPolicy.wireName)
        put("suppressionInitialBehavior", draft.suppressionInitialBehavior.wireName)
        put("suppressionDecisionOverrides", JSONObject().apply {
            draft.suppressionDecisionOverrides.toSortedMap().forEach { (id, decision) ->
                put(id, decision.wireName)
            }
        })
        put("suppressionScopeOverrides", JSONObject().apply {
            draft.suppressionScopeOverrides.toSortedMap().forEach { (id, scope) ->
                put(id, scope.wireName)
            }
        })
        put("userTouchedCutIds", JSONArray(draft.userTouchedCutIds.sorted()))
        put("suppressionContractVersion", draft.suppressionContractVersion)
        put("scoreTracking", ScoreTrackingJson.encode(draft.scoreTracking))
        put("renderScoreOverlay", draft.renderScoreOverlay)
        put("renderScoreTimeline", draft.renderScoreTimeline)
        put("cuts", JSONArray().apply {
            draft.cuts.forEach { cut -> put(JSONObject().apply {
                put("id", cut.id)
                put("coreStartMs", cut.coreStartMs)
                put("coreEndMs", cut.coreEndMs)
                put("keepStartMs", cut.keepStartMs)
                put("keepEndMs", cut.keepEndMs)
                put("confidence", cut.confidence.toDouble())
                put("included", cut.included)
                put("origin", cut.origin.name)
                put("agreement", cut.agreement ?: JSONObject.NULL)
            }) }
        })
        put("ignoredIntervals", JSONArray().apply {
            draft.ignoredIntervals.forEach { interval -> put(JSONObject().apply {
                put("id", interval.id)
                put("startMs", interval.startMs)
                put("endMs", interval.endMs)
                put("reason", interval.reason)
            }) }
        })
    }

    private fun decode(json: JSONObject): EditorDraft? {
        val persistedVersion = json.optInt("version")
        if (persistedVersion !in 1..EDITOR_DRAFT_VERSION) return null
        val cutsJson = json.optJSONArray("cuts") ?: return null
        val cuts = buildList {
            for (index in 0 until cutsJson.length()) {
                val item = cutsJson.getJSONObject(index)
                add(EditableCut(
                    id = item.getString("id"),
                    coreStartMs = item.getLong("coreStartMs"),
                    coreEndMs = item.getLong("coreEndMs"),
                    keepStartMs = item.getLong("keepStartMs"),
                    keepEndMs = item.getLong("keepEndMs"),
                    confidence = item.getDouble("confidence").toFloat(),
                    included = item.getBoolean("included"),
                    origin = CutOrigin.valueOf(item.getString("origin")),
                    agreement = item.optNullableString("agreement"),
                ))
            }
        }
        val ignoredJson = json.optJSONArray("ignoredIntervals") ?: JSONArray()
        val ignored = buildList {
            for (index in 0 until ignoredJson.length()) {
                val item = ignoredJson.getJSONObject(index)
                add(IgnoredSourceInterval(
                    id = item.getString("id"),
                    startMs = item.getLong("startMs"),
                    endMs = item.getLong("endMs"),
                    reason = item.getString("reason"),
                ))
            }
        }
        val overridesJson = json.optJSONObject("suppressionDecisionOverrides") ?: JSONObject()
        val overrides = buildMap {
            overridesJson.keys().forEach { id ->
                SuppressionDecision.fromWireName(overridesJson.optString(id))?.let { put(id, it) }
            }
        }
        val scopeOverridesJson = json.optJSONObject("suppressionScopeOverrides") ?: JSONObject()
        val scopeOverrides = buildMap {
            scopeOverridesJson.keys().forEach { id ->
                SuppressionScope.fromWireName(scopeOverridesJson.optString(id))
                    .takeIf { it != SuppressionScope.WHOLE_RALLY }
                    ?.let { put(id, it) }
            }
        }
        val touchedJson = json.optJSONArray("userTouchedCutIds") ?: JSONArray()
        val reviewedJson = json.optJSONArray("reviewedCutIds") ?: JSONArray()
        val restoredScore = if (persistedVersion >= 6) {
            json.optJSONObject("scoreTracking")?.let { ScoreTrackingJson.decode(it, seed.durationMs) }
        } else null
        val scoreTracking = ScoreReducer.seedModelMarkers(
            restoredScore ?: ScoreTracking(enabled = seed.scoreTrackingInitiallyEnabled),
            seed.servingSide,
            seed.sideSwitch,
            seed.sideSwitchEnabled,
        )
        val draft = EditorDraft(
            sourceRevision = json.getString("sourceRevision"),
            updatedAtMs = json.optLong("updatedAtMs"),
            beforePaddingMs = json.optLong("beforePaddingMs", DEFAULT_BEFORE_PADDING_MS),
            afterPaddingMs = json.optLong("afterPaddingMs", DEFAULT_AFTER_PADDING_MS),
            joinGapMs = if (persistedVersion >= 2) {
                json.optLong("joinGapMs", DEFAULT_JOIN_GAP_MS)
            } else DEFAULT_JOIN_GAP_MS,
            pendingManualStartMs = json.optNullableLong("pendingManualStartMs"),
            pendingIgnoreStartMs = json.optNullableLong("pendingIgnoreStartMs"),
            ignoreReason = json.optString("ignoreReason", "non-game-content"),
            finalPreviewEnabled = json.optBoolean("finalPreviewEnabled"),
            playbackRate = json.optDouble("playbackRate", 1.0).toFloat(),
            confidenceReviewThreshold = json.optDouble("confidenceReviewThreshold", .7).toFloat(),
            reviewedCutIds = buildSet {
                for (index in 0 until reviewedJson.length()) add(reviewedJson.getString(index))
            },
            cuts = cuts,
            ignoredIntervals = ignored,
            selectedSuppressionPolicy = SuppressionPolicyEngine.Policy.fromWireName(
                json.optString("selectedSuppressionPolicy", "none"),
            ),
            suppressionInitialBehavior = SuppressionInitialBehavior.fromWireName(
                json.optString("suppressionInitialBehavior", "disable-initially"),
            ),
            suppressionDecisionOverrides = overrides,
            suppressionScopeOverrides = scopeOverrides,
            userTouchedCutIds = buildSet {
                for (index in 0 until touchedJson.length()) add(touchedJson.getString(index))
            },
            suppressionContractVersion = json.optString(
                "suppressionContractVersion",
                FeatureSchema.SUPPRESSION_POLICY_CONTRACT_VERSION,
            ),
            scoreTracking = scoreTracking,
            renderScoreOverlay = if (persistedVersion >= 6) {
                json.optBoolean("renderScoreOverlay")
            } else false,
            renderScoreTimeline = if (persistedVersion >= 7) {
                json.optBoolean("renderScoreTimeline")
            } else if (persistedVersion >= 6) {
                json.optBoolean("renderScoreOverlay")
            } else false,
        )
        return EditorMath.reconcileTouchedCuts(draft, seed).takeIf { validate(it) }
    }

    private fun validate(draft: EditorDraft): Boolean =
        draft.beforePaddingMs in 0..MAX_PADDING_MS &&
            draft.afterPaddingMs in 0..MAX_PADDING_MS &&
            draft.joinGapMs in 0..MAX_JOIN_GAP_MS &&
            draft.playbackRate in setOf(1f, 2f, 4f, 8f) &&
            draft.confidenceReviewThreshold in 0f..1f &&
            draft.suppressionContractVersion == FeatureSchema.SUPPRESSION_POLICY_CONTRACT_VERSION &&
            ScoreTrackingJson.validate(draft.scoreTracking, seed.durationMs) &&
            draft.userTouchedCutIds.all { id -> draft.cuts.any { it.id == id } } &&
            draft.reviewedCutIds.all { id -> draft.cuts.any { it.id == id } } &&
            draft.cuts.all { cut ->
                cut.keepStartMs in seed.gameStartMs..cut.coreStartMs &&
                    cut.coreStartMs < cut.coreEndMs &&
                    cut.coreEndMs <= cut.keepEndMs &&
                    cut.keepEndMs <= seed.gameEndMs &&
                    (cut.agreement == null || ProductionEnsemble.isValidAgreement(cut.agreement))
            } &&
            draft.ignoredIntervals.all { it.startMs in 0 until it.endMs && it.endMs <= seed.durationMs }

    private fun JSONObject.optNullableLong(key: String): Long? =
        if (!has(key) || isNull(key)) null else getLong(key)

    private fun JSONObject.optNullableString(key: String): String? =
        if (!has(key) || isNull(key)) null else getString(key)

    companion object {
        private const val TAG = "VolleyCutEditor"

        fun relink(context: Context, oldSeed: EditorSeed, newSeed: EditorSeed) {
            val oldDraft = EditorDraftStore(context, oldSeed).load() ?: return
            EditorDraftStore(context, newSeed).save(oldDraft.copy(
                sourceRevision = newSeed.sourceRevision,
                updatedAtMs = System.currentTimeMillis(),
            ))
        }
    }
}
