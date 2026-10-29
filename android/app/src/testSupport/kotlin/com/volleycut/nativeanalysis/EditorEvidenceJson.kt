package com.volleycut.nativeanalysis

import org.json.JSONArray
import org.json.JSONObject

/** Detailed edit evidence shared by JVM and device tests. */
internal fun editListJson(seed: EditorSeed, draft: EditorDraft, intervals: List<FinalCutInterval>) =
    JSONObject().apply {
        put("schemaVersion", 2)
        put("method", "android-editor-v3-suppression")
        put("modelId", seed.rallyModelId)
        put("modelLabel", RallyModels.label(seed.rallyModelId))
        put("sourceName", seed.displayName)
        put("sourceUri", seed.sourceUri)
        put("sourceDuration", seed.durationMs / 1_000.0)
        put("gameStartSeconds", seed.gameStartMs / 1_000.0)
        put("gameEndSeconds", seed.gameEndMs / 1_000.0)
        put("sourceRevision", seed.sourceRevision)
        put("beforePaddingSeconds", draft.beforePaddingMs / 1_000.0)
        put("afterPaddingSeconds", draft.afterPaddingMs / 1_000.0)
        put("joinGapSeconds", draft.joinGapMs / 1_000.0)
        put("outputDuration", EditorMath.totalFinalMs(intervals) / 1_000.0)
        put("policyContractVersion", draft.suppressionContractVersion)
        put("selectedSuppressionPolicy", draft.selectedSuppressionPolicy.wireName)
        put("recordedSuppressionPolicy", draft.selectedSuppressionPolicy.recordedPolicy)
        put("suppressionInitialBehavior", draft.suppressionInitialBehavior.wireName)
        put("defaultSuppressionScope", SuppressionScope.WHOLE_RALLY.wireName)
        put("suppression", seed.suppression?.let { analysis -> JSONObject().apply {
            put("modelId", analysis.modelId())
            put("artifactSha256", analysis.artifactSha256())
            put("weightsSha256", analysis.weightsSha256())
            put("decoderVersion", analysis.decoderVersion())
            put("productionComponents", JSONObject().apply {
                put("allLabelsV2", JSONArray().apply {
                    seed.productionComponents.allLabelsV2().forEach { interval ->
                        put(JSONObject().apply {
                            put("start", interval.start())
                            put("end", interval.end())
                            put("confidence", interval.confidence().toDouble())
                        })
                    }
                })
                put("previousProduction", JSONArray().apply {
                    seed.productionComponents.previousProduction().forEach { interval ->
                        put(JSONObject().apply {
                            put("start", interval.start())
                            put("end", interval.end())
                            put("confidence", interval.confidence().toDouble())
                        })
                    }
                })
            })
            put("suggestions", JSONArray().apply {
                analysis.suggestions().forEach { suggestion -> put(JSONObject().apply {
                    put("logicalId", suggestion.logicalId())
                    put("fragmentId", suggestion.fragmentId())
                    put("start", suggestion.startMs() / 1_000.0)
                    put("end", suggestion.endMs() / 1_000.0)
                    put("score", suggestion.score().toDouble())
                    put("sourceProductionIds", JSONArray(suggestion.sourceProductionIds()))
                    put("eligiblePolicyIds", JSONArray(suggestion.eligiblePolicyIds()))
                    put("active", suggestion.eligiblePolicyIds().contains(draft.selectedSuppressionPolicy.wireName))
                    put("explicitDecision", draft.suppressionDecisionOverrides[suggestion.logicalId()]?.wireName ?: JSONObject.NULL)
                    put("effectiveDecision", EditorMath.suggestionEffectiveDecision(draft, suggestion).wireName)
                    put("effectiveScope", EditorMath.suggestionEffectiveScope(draft, suggestion).wireName)
                }) }
            })
            put("decisionOverrides", JSONObject().apply {
                draft.suppressionDecisionOverrides.toSortedMap().forEach { (id, decision) ->
                    put(id, decision.wireName)
                }
            })
            put("scopeOverrides", JSONObject().apply {
                draft.suppressionScopeOverrides.toSortedMap().forEach { (id, scope) ->
                    put(id, scope.wireName)
                }
            })
            put("userTouchedCutIds", JSONArray(draft.userTouchedCutIds.sorted()))
        }} ?: JSONObject.NULL)
        put("ranges", JSONArray().apply {
            intervals.forEach { range -> put(JSONObject().apply {
                put("start", range.startMs / 1_000.0)
                put("end", range.endMs / 1_000.0)
                put("cutIds", JSONArray(range.cutIds))
                if (range.joinedGaps.isNotEmpty()) put("joinedGaps", JSONArray().apply {
                    range.joinedGaps.forEach { gap -> put(JSONObject().apply {
                        put("start", gap.startMs / 1_000.0)
                        put("end", gap.endMs / 1_000.0)
                    }) }
                })
            }) }
        })
        put("cuts", JSONArray().apply {
            draft.cuts.forEach { cut -> put(JSONObject().apply {
                put("id", cut.id)
                put("coreStart", cut.coreStartMs / 1_000.0)
                put("coreEnd", cut.coreEndMs / 1_000.0)
                put("keepStart", cut.keepStartMs / 1_000.0)
                put("keepEnd", cut.keepEndMs / 1_000.0)
                put("confidence", cut.confidence.toDouble())
                put("included", cut.included)
                put("origin", cut.origin.name.lowercase())
                cut.editableGroupId?.let { put("editableGroupId", it) }
                put("agreement", cut.agreement ?: JSONObject.NULL)
            }) }
        })
        put("ignoredIntervals", JSONArray().apply {
            draft.ignoredIntervals.forEach { interval -> put(JSONObject().apply {
                put("id", interval.id)
                put("start", interval.startMs / 1_000.0)
                put("end", interval.endMs / 1_000.0)
                put("reason", interval.reason)
            }) }
        })
        put("materializationProvenance", JSONArray().apply {
            EditorMath.materialize(draft, seed.suppression).provenance.forEach { segment ->
                put(JSONObject().apply {
                    put("start", segment.startMs / 1_000.0)
                    put("end", segment.endMs / 1_000.0)
                    put("kind", segment.kind)
                    put("cutIds", JSONArray(segment.cutIds))
                    put("suggestionIds", JSONArray(segment.suggestionIds))
                })
            }
        })
        put("scoreTracking", ScoreTrackingJson.encodeWire(draft.scoreTracking))
        put("renderScoreOverlay", draft.renderScoreOverlay)
        put("renderScoreTimeline", draft.renderScoreTimeline)
        put("fadeScoreOverlay", draft.fadeScoreOverlay)
    }
