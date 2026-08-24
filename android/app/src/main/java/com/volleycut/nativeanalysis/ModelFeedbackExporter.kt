package com.volleycut.nativeanalysis

import android.content.Context
import android.net.Uri
import org.json.JSONArray
import org.json.JSONObject
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.time.Instant
import java.util.Base64

internal const val MODEL_FEEDBACK_SCHEMA = "volleycut-model-feedback"
internal const val MODEL_FEEDBACK_SCHEMA_VERSION = 3

internal data class ModelFeedbackAnalysis(
    val timestamps: DoubleArray,
    val baseFeatures: FloatArray,
    val rallyProbabilities: FloatArray,
    val serveProbabilities: FloatArray,
    val deadStateProbabilities: FloatArray,
)

/** Builds the same raw-video-free training-feedback contract as the production web editor. */
internal object ModelFeedbackExporter {
    private const val SAMPLE_BYTES = 1024 * 1024
    private const val RUNTIME_VARIANT = "native-android-dsp-v1"

    fun loadAnalysis(context: Context, project: NativeProject): ModelFeedbackAnalysis? {
        val requestedTimes = AnalysisEngine.analysisTimes(
            project.media.durationSeconds(),
            project.analysisWindow.start(),
            project.analysisWindow.end(),
        )
        val cacheSource = project.featureCacheSource ?: project.source
        val cache = NativeFeatureCache.openWithSourceMetadata(
            context,
            Uri.parse(cacheSource.uri),
            cacheSource.name,
            cacheSource.size,
            cacheSource.lastModified,
            project.media,
            project.roi,
            FeatureSchema.FULL_SOURCE_FRAME_LIMIT,
            requestedTimes.size,
            project.analysisWindow,
        )
        val visual = cache.loadVisual()
        if (!visual.complete() || visual.rows() == 0 || !cache.timesMatch(visual, requestedTimes)) {
            return null
        }
        val rows = visual.rows()
        val audio = cache.loadAudio(rows) ?: return null
        val contextual = cache.loadContext(rows) ?: return null
        val temporal = FeatureMath.temporalVisualFeatures(visual.values(), rows)
        val base = combineBase(visual.values(), temporal, audio, rows)
        val probabilities = ModelRunner(context, FeatureSchema.ALL_LABELS_V2_MODEL_ID)
            .runProfiled(visual.times(), contextual, visual.analyzedDurationSeconds())
        return ModelFeedbackAnalysis(
            timestamps = visual.times(),
            baseFeatures = base,
            rallyProbabilities = probabilities.rallyProbabilities(),
            serveProbabilities = probabilities.serveProbabilities(),
            deadStateProbabilities = probabilities.deadStateProbabilities(),
        )
    }

    fun createBundle(
        project: NativeProject,
        draft: EditorDraft,
        finalIntervals: List<FinalCutInterval>,
        analysis: ModelFeedbackAnalysis?,
        sampledFingerprint: String?,
        generatedAt: Instant = Instant.now(),
    ): JSONObject {
        if (analysis != null) {
            val rows = analysis.timestamps.size
            require(analysis.baseFeatures.size == rows * FeatureSchema.BASE.size)
            require(analysis.rallyProbabilities.size == rows)
            require(analysis.serveProbabilities.size == rows)
            require(analysis.deadStateProbabilities.size == rows)
        }
        val warnings = JSONArray()
        if (analysis == null) {
            warnings.put(
                "Base features and probability traces are unavailable because the retained feature cache is missing; inference ranges and corrections are still included.",
            )
        }
        if (sampledFingerprint == null) {
            warnings.put("The source fingerprint could not be calculated; pair this bundle by source metadata.")
        }
        if (project.servingSide == null) {
            warnings.put(
                project.servingSideError
                    ?: "Serving-side features and initial verdicts are unavailable; score-marker corrections are still included.",
            )
        }
        if (project.sideSwitchEnabled && project.sideSwitch == null) {
            warnings.put(
                project.sideSwitchError
                    ?: "Team-switch features and inferred markers are unavailable; manual switch corrections are still included.",
            )
        }

        val inferredCuts = draft.cuts.filter { it.origin == CutOrigin.INFERRED }
        val manualCuts = draft.cuts.filter { it.origin == CutOrigin.MANUAL }
        val inferenceRows = analysis?.timestamps?.size ?: 0
        val updatedAtMs = draft.updatedAtMs.takeIf { it > 0 } ?: project.updatedAtMs
        val effectiveIds = EditorMath.effectiveKeptIds(draft, project.suppression)
        val excludedRallyIds = draft.cuts.filter {
            it.origin == CutOrigin.INFERRED && (!it.included || it.id !in effectiveIds)
        }.map { it.id }.sorted()
        val visibleScoring = ScoreReducer.visibleTracking(
            draft.scoreTracking,
            draft.ignoredIntervals,
            excludedRallyIds.toSet(),
        )

        return JSONObject().apply {
            put("schema", MODEL_FEEDBACK_SCHEMA)
            put("schemaVersion", MODEL_FEEDBACK_SCHEMA_VERSION)
            put("generatedAt", generatedAt.toString())
            put("source", JSONObject().apply {
                put("projectId", project.id)
                put("analysisId", "${project.id}-${project.modelId}-native-source")
                put("timelineCoordinates", "seconds-from-start-of-source")
                put("file", JSONObject().apply {
                    put("name", project.source.name)
                    put("sizeBytes", project.source.size.coerceAtLeast(0))
                    put("lastModifiedMs", lastModifiedMs(project.source.lastModified).coerceAtLeast(0))
                    put("mimeType", project.source.mimeType.ifBlank { project.media.videoMime() })
                    put("sampledFingerprint", sampledFingerprint ?: JSONObject.NULL)
                })
                put("media", JSONObject().apply {
                    put("duration", project.media.durationSeconds())
                    put("mimeType", project.source.mimeType.ifBlank { project.media.videoMime() })
                    put("width", project.media.width())
                    put("height", project.media.height())
                    put("rotation", project.media.rotation())
                    put("videoCodec", project.media.videoMime())
                    put("videoCodecString", JSONObject.NULL)
                    put("canDecodeVideo", true)
                    put("hasAudio", !project.media.audioMime().isNullOrBlank())
                    put("audioCodec", project.media.audioMime()?.takeIf(String::isNotBlank) ?: JSONObject.NULL)
                    put("sampleRate", JSONObject.NULL)
                    put("channels", JSONObject.NULL)
                    put("canDecodeAudio", !project.media.audioMime().isNullOrBlank())
                })
                put("gameWindow", JSONObject().apply {
                    put("start", project.analysisWindow.start())
                    put("end", project.analysisWindow.end())
                })
                put("featureRoi", JSONObject().apply {
                    put("x", project.roi.x())
                    put("y", project.roi.y())
                    put("width", project.roi.width())
                    put("height", project.roi.height())
                })
                put("runtimeVariant", RUNTIME_VARIANT)
                put("videoBytesIncluded", false)
            })
            put("features", analysis?.let { retained -> JSONObject().apply {
                put("analysisFps", FeatureSchema.ANALYSIS_FPS)
                put("rows", retained.timestamps.size)
                put("columns", FeatureSchema.BASE.size)
                put("names", JSONArray(FeatureSchema.BASE))
                put("timestamps", encode(retained.timestamps, intArrayOf(retained.timestamps.size)))
                put("values", encode(
                    retained.baseFeatures,
                    intArrayOf(retained.timestamps.size, FeatureSchema.BASE.size),
                ))
            }} ?: JSONObject.NULL)
            put("initialInference", JSONObject().apply {
                put("modelId", project.modelId)
                put("components", JSONArray().apply {
                    put(modelComponent(
                        FeatureSchema.ALL_LABELS_V2_MODEL_ID,
                        FeatureSchema.ALL_LABELS_V2_BUNDLE_SHA256,
                    ))
                    put(modelComponent(
                        FeatureSchema.PREVIOUS_PRODUCTION_MODEL_ID,
                        FeatureSchema.PREVIOUS_PRODUCTION_BUNDLE_SHA256,
                    ))
                })
                put("ensembleAlgorithmVersion", FeatureSchema.ENSEMBLE_ALGORITHM_VERSION)
                put("ranges", JSONArray().apply {
                    project.ranges.forEachIndexed { index, range -> put(JSONObject().apply {
                        put("id", "R${(index + 1).toString().padStart(3, '0')}")
                        put("start", range.startMs / 1_000.0)
                        put("end", range.endMs / 1_000.0)
                        put("confidence", range.confidence.toDouble())
                        put("included", true)
                        range.agreement?.let { put("agreement", it) }
                    }) }
                })
                put("productionComponents", JSONObject().apply {
                    put("allLabelsV2", analysisIntervals(
                        project.productionComponents.allLabelsV2(),
                        "all-labels-v2",
                    ))
                    put("previousProduction", analysisIntervals(
                        project.productionComponents.previousProduction(),
                        "previous-production",
                    ))
                })
                put("suppression", if (analysis != null) project.suppression
                    ?.takeIf { it.probabilities().size == inferenceRows }
                    ?.let { suppression -> JSONObject().apply {
                    put("modelId", suppression.modelId())
                    put("artifactSha256", suppression.artifactSha256())
                    put("weightsSha256", suppression.weightsSha256())
                    put("decoderVersion", suppression.decoderVersion())
                    put("policyContractVersion", 1)
                    put("identicalPolicyResults", identicalSuppressionPolicies(suppression))
                    put("timestamps", encode(analysis.timestamps, intArrayOf(inferenceRows)))
                    put("probabilities", encode(
                        suppression.probabilities(),
                        intArrayOf(suppression.probabilities().size),
                    ))
                    put("decodedIntervals", analysisIntervals(
                        suppression.decodedIntervals(),
                        "suppression",
                    ))
                    put("suggestions", JSONArray().apply {
                        suppression.suggestions().forEach { suggestion -> put(JSONObject().apply {
                            put("id", suggestion.fragmentId())
                            put("logicalId", suggestion.logicalId())
                            // Android's native suppression model predates the event
                            // field; its logical ID is the stable event grouping.
                            put("suppressionEventId", suggestion.logicalId())
                            put("start", suggestion.startMs() / 1_000.0)
                            put("end", suggestion.endMs() / 1_000.0)
                            put("score", suggestion.score().toDouble())
                            put("sourceProductionIds", JSONArray(suggestion.sourceProductionIds()))
                            put("eligiblePolicyIds", JSONArray(suggestion.eligiblePolicyIds()))
                        }) }
                    })
                }} ?: JSONObject.NULL else JSONObject.NULL)
                put("probabilityModelId", FeatureSchema.ALL_LABELS_V2_MODEL_ID)
                put("timestamps", encode(
                    analysis?.timestamps ?: doubleArrayOf(),
                    intArrayOf(inferenceRows),
                ))
                put("probabilities", JSONObject().apply {
                    put("rally", encode(
                        analysis?.rallyProbabilities ?: floatArrayOf(),
                        intArrayOf(inferenceRows),
                    ))
                    put("serve", encode(
                        analysis?.serveProbabilities ?: floatArrayOf(),
                        intArrayOf(inferenceRows),
                    ))
                    put("deadState", encode(
                        analysis?.deadStateProbabilities ?: floatArrayOf(),
                        intArrayOf(inferenceRows),
                    ))
                })
                if (analysis != null) {
                    componentServeOutputs(
                        project.productionServeOutputs,
                        analysis.timestamps,
                    )?.let {
                        put("componentServeOutputs", it)
                    }
                }
                put("servingSide", project.servingSide?.let(::servingSideFeedback) ?: JSONObject.NULL)
                put("sideSwitch", project.sideSwitch?.let(::sideSwitchFeedback) ?: JSONObject.NULL)
            })
            put("corrections", JSONObject().apply {
                put("updatedAt", Instant.ofEpochMilli(updatedAtMs).toString())
                put("beforePaddingSeconds", draft.beforePaddingMs / 1_000.0)
                put("afterPaddingSeconds", draft.afterPaddingMs / 1_000.0)
                put("joinGapSeconds", draft.joinGapMs / 1_000.0)
                put("selectedSuppressionPolicy", draft.selectedSuppressionPolicy.wireName)
                put("recordedSuppressionPolicy", draft.selectedSuppressionPolicy.recordedPolicy)
                put("suppressionInitialBehavior", draft.suppressionInitialBehavior.wireName)
                put("defaultSuppressionScope", SuppressionScope.WHOLE_RALLY.wireName)
                put("suppressionContractVersion", draft.suppressionContractVersion)
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
                put("suppression", suppressionCorrections(project.suppression, draft))
                put("correctedRanges", JSONArray().apply {
                    draft.cuts.forEach { put(correctedRange(it)) }
                })
                put("ignoredIntervals", JSONArray().apply {
                    draft.ignoredIntervals.forEach { interval -> put(JSONObject().apply {
                        put("id", interval.id)
                        put("start", interval.startMs / 1_000.0)
                        put("end", interval.endMs / 1_000.0)
                        put("reason", interval.reason)
                    }) }
                })
                put("scoreTracking", JSONObject().apply {
                    put("state", ScoreTrackingJson.encodeWire(draft.scoreTracking))
                    put("excludedRallyIds", JSONArray(excludedRallyIds))
                    put("derivedFinalScore", derivedScoreJson(ScoreReducer.deriveAt(visibleScoring)))
                })
                put("labels", JSONObject().apply {
                    put("falsePositives", feedbackRanges(inferredCuts.filterNot { it.included }))
                    put("falseNegatives", feedbackRanges(manualCuts.filter { it.included }))
                    put("confirmedModelRanges", feedbackRanges(inferredCuts.filter { it.included }))
                    put("discardedManualRanges", feedbackRanges(manualCuts.filterNot { it.included }))
                })
            })
            put("finalExportIntervals", JSONArray().apply {
                finalIntervals.forEach { interval -> put(JSONObject().apply {
                    put("start", interval.startMs / 1_000.0)
                    put("end", interval.endMs / 1_000.0)
                    put("cutIds", JSONArray(interval.cutIds))
                    if (interval.joinedGaps.isNotEmpty()) put("joinedGaps", JSONArray().apply {
                        interval.joinedGaps.forEach { gap -> put(JSONObject().apply {
                            put("start", gap.startMs / 1_000.0)
                            put("end", gap.endMs / 1_000.0)
                        }) }
                    })
                }) }
            })
            put("finalExportProvenance", JSONArray().apply {
                EditorMath.materialize(draft, project.suppression).provenance.forEach { segment ->
                    put(JSONObject().apply {
                        put("start", segment.startMs / 1_000.0)
                        put("end", segment.endMs / 1_000.0)
                        put("kind", segment.kind)
                        put("cutIds", JSONArray(segment.cutIds))
                        put("suggestionIds", JSONArray(segment.suggestionIds))
                    })
                }
            })
            put("warnings", warnings)
        }
    }

    fun sourceFingerprint(context: Context, project: NativeProject): String? = runCatching {
        val uri = Uri.parse(project.source.uri)
        context.contentResolver.openFileDescriptor(uri, "r")?.use { descriptor ->
            val size = project.source.size.takeIf { it >= 0 } ?: descriptor.statSize
            require(size >= 0) { "Source size is unavailable" }
            FileInputStream(descriptor.fileDescriptor).use { input ->
                val digest = MessageDigest.getInstance("SHA-256")
                digest.update(ByteBuffer.allocate(Long.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN).putLong(size).array())
                if (size <= SAMPLE_BYTES * 2L) {
                    updateDigest(digest, input, size)
                } else {
                    updateDigest(digest, input, SAMPLE_BYTES.toLong())
                    input.channel.position(size - SAMPLE_BYTES)
                    updateDigest(digest, input, SAMPLE_BYTES.toLong())
                }
                "sampled-sha256-v1:" + digest.digest().joinToString("") { "%02x".format(it.toInt() and 0xff) }
            }
        } ?: error("The source cannot be opened")
    }.getOrNull()

    fun filename(sourceName: String): String {
        val safe = sourceName.substringBeforeLast('.')
            .replace(Regex("[^A-Za-z0-9._-]+"), "-")
            .trim('-')
        return "${safe.ifBlank { "volleycut" }}.model-feedback.json"
    }

    internal fun encode(values: FloatArray, shape: IntArray): JSONObject =
        encodedArray("float32", shape, ByteBuffer.allocate(values.size * Float.SIZE_BYTES).apply {
            order(ByteOrder.LITTLE_ENDIAN)
            asFloatBuffer().put(values)
        }.array())

    internal fun encode(values: DoubleArray, shape: IntArray): JSONObject =
        encodedArray("float64", shape, ByteBuffer.allocate(values.size * Double.SIZE_BYTES).apply {
            order(ByteOrder.LITTLE_ENDIAN)
            asDoubleBuffer().put(values)
        }.array())

    private fun encodedArray(dataType: String, shape: IntArray, bytes: ByteArray): JSONObject {
        require(shape.isNotEmpty() && shape.all { it >= 0 }) { "Invalid numeric-array shape" }
        val elementBytes = if (dataType == "float32") Float.SIZE_BYTES else Double.SIZE_BYTES
        require(shape.fold(1L) { product, size -> product * size } * elementBytes == bytes.size.toLong()) {
            "Numeric-array shape does not match its data"
        }
        return JSONObject().apply {
            put("encoding", "base64")
            put("byteOrder", "little-endian")
            put("dataType", dataType)
            put("shape", JSONArray(shape.toList()))
            put("data", Base64.getEncoder().encodeToString(bytes))
        }
    }

    private fun combineBase(
        visual: FloatArray,
        temporal: FloatArray,
        audio: FloatArray,
        rows: Int,
    ): FloatArray {
        val frameColumns = FeatureSchema.FRAME.size
        val temporalColumns = FeatureSchema.TEMPORAL.size
        val audioColumns = FeatureSchema.AUDIO.size
        require(visual.size == rows * frameColumns)
        require(temporal.size == rows * temporalColumns)
        require(audio.size == rows * audioColumns)
        return FloatArray(rows * FeatureSchema.BASE.size).also { output ->
            repeat(rows) { row ->
                val target = row * FeatureSchema.BASE.size
                visual.copyInto(output, target, row * frameColumns, (row + 1) * frameColumns)
                temporal.copyInto(
                    output,
                    target + frameColumns,
                    row * temporalColumns,
                    (row + 1) * temporalColumns,
                )
                audio.copyInto(
                    output,
                    target + frameColumns + temporalColumns,
                    row * audioColumns,
                    (row + 1) * audioColumns,
                )
            }
        }
    }

    private fun correctedRange(cut: EditableCut) = JSONObject().apply {
        put("id", cut.id)
        put("coreStart", cut.coreStartMs / 1_000.0)
        put("coreEnd", cut.coreEndMs / 1_000.0)
        put("keepStart", cut.keepStartMs / 1_000.0)
        put("keepEnd", cut.keepEndMs / 1_000.0)
        put("confidence", cut.confidence.toDouble())
        put("included", cut.included)
        put("origin", if (cut.origin == CutOrigin.INFERRED) "cached-label" else "manual")
        cut.agreement?.let { put("agreement", it) }
    }

    private fun feedbackRanges(cuts: List<EditableCut>) = JSONArray().apply {
        cuts.forEach { cut -> put(JSONObject().apply {
            put("id", cut.id)
            put("start", cut.coreStartMs / 1_000.0)
            put("end", cut.coreEndMs / 1_000.0)
            put("confidence", cut.confidence.toDouble())
            cut.agreement?.let { put("agreement", it) }
        }) }
    }

    private fun modelComponent(modelId: String, sha256: String) = JSONObject().apply {
        put("modelId", modelId)
        put("bundleSha256", sha256)
    }

    private fun componentServeOutputs(
        value: AnalysisTypes.ProductionServeOutputs,
        inferenceTimes: DoubleArray,
    ): JSONObject? {
        val outputs = listOf(value.allLabelsV2(), value.previousProduction())
        if (outputs.any {
                it.times().size != inferenceTimes.size ||
                    it.probabilities().size != inferenceTimes.size ||
                    !it.times().contentEquals(inferenceTimes)
            }
        ) {
            return null
        }
        fun output(item: AnalysisTypes.ProductionServeOutput) = JSONObject().apply {
            put("modelId", item.modelId())
            put("probabilities", encode(item.probabilities(), intArrayOf(item.probabilities().size)))
            put("detections", JSONArray().apply {
                item.detections().forEach { detection -> put(JSONObject().apply {
                    put("time", detection.time())
                    put("confidence", detection.confidence().toDouble())
                }) }
            })
        }
        return JSONObject().apply {
            put("allLabelsV2", output(value.allLabelsV2()))
            put("previousProduction", output(value.previousProduction()))
        }
    }

    private fun identicalSuppressionPolicies(value: AnalysisTypes.SuppressionAnalysis): Boolean {
        fun signature(policy: String) = value.suggestions()
            .filter { policy in it.eligiblePolicyIds() }
            .joinToString("\u0000") { it.fragmentId() }
        val conservative = signature("conservative")
        val balanced = signature("balanced")
        val aggressive = signature("aggressive")
        return conservative == balanced && balanced == aggressive
    }

    private fun suppressionCorrections(
        suppression: AnalysisTypes.SuppressionAnalysis?,
        draft: EditorDraft,
    ) = JSONObject().apply {
        put("selectedPolicy", draft.selectedSuppressionPolicy.wireName)
        put("decisionOverrides", JSONObject().apply {
            draft.suppressionDecisionOverrides.toSortedMap().forEach { (id, decision) ->
                put(id, decision.wireName)
            }
        })
        put("defaultSuppressionScope", SuppressionScope.WHOLE_RALLY.wireName)
        put("suppressionScopeOverrides", JSONObject().apply {
            draft.suppressionScopeOverrides.toSortedMap().forEach { (id, scope) ->
                put(id, scope.wireName)
            }
        })
        put("userTouchedCutIds", JSONArray(draft.userTouchedCutIds.sorted()))
        put("decisions", JSONArray().apply {
            suppression?.suggestions()?.forEach { suggestion ->
                val active = draft.selectedSuppressionPolicy != SuppressionPolicyEngine.Policy.NONE &&
                    draft.selectedSuppressionPolicy.wireName in suggestion.eligiblePolicyIds()
                val explicit = draft.suppressionDecisionOverrides[suggestion.logicalId()]
                val effective = EditorMath.suggestionEffectiveDecision(draft, suggestion)
                val touched = draft.cuts.any { cut ->
                    cut.origin == CutOrigin.INFERRED && cut.id in draft.userTouchedCutIds &&
                        cut.coreStartMs < suggestion.endMs() && suggestion.startMs() < cut.coreEndMs
                }
                val state = when {
                    !active -> "dormant"
                    effective == SuppressionDecision.SUPPRESS -> "suppressed"
                    explicit == SuppressionDecision.KEEP -> "kept"
                    touched -> "edited-kept"
                    else -> "kept"
                }
                put(JSONObject().apply {
                    put("suggestionId", suggestion.fragmentId())
                    put("logicalId", suggestion.logicalId())
                    put("state", state)
                    put("scope", EditorMath.suggestionEffectiveScope(draft, suggestion).wireName)
                })
            }
        })
    }

    private fun servingSideFeedback(value: ServingSideOutput) = JSONObject().apply {
        put("modelId", value.modelId)
        put("modelFingerprint", value.modelFingerprint)
        put("featureVersion", value.featureVersion)
        put("anchorContract", value.anchorContract)
        put("features", JSONObject().apply {
            put("rows", value.rows)
            put("columns", value.columns)
            put("values", encode(value.rawFeatures, intArrayOf(value.rows, value.columns)))
        })
        put("candidates", JSONArray().apply {
            value.candidates.forEach { candidate -> put(JSONObject().apply {
                put("id", candidate.id)
                put("anchor", candidate.anchor)
                put("interval", JSONObject().apply {
                    put("start", candidate.intervalStart)
                    put("end", candidate.intervalEnd)
                    candidate.agreement?.let { put("agreement", it) }
                })
                put("nearProbability", candidate.nearProbability)
                put("side", candidate.side.wireName)
                put("verdict", candidate.verdict.wireName)
                put("serveDecisionSource", candidate.serveDecisionSource.wireName)
                put("reviewReasons", JSONArray(candidate.reviewReasons.map { it.wireName }))
                put("serveEvidence", JSONObject().apply {
                    put("allLabelsV2", evidenceJson(candidate.allLabelsV2Evidence))
                    put("previousProduction", evidenceJson(candidate.previousProductionEvidence))
                })
            }) }
        })
    }

    private fun sideSwitchFeedback(value: SideSwitchOutput) = JSONObject().apply {
        put("modelId", value.modelId)
        put("modelFingerprint", value.modelFingerprint)
        put("featureVersion", value.featureVersion)
        put("candidateContract", value.candidateContract)
        put("features", JSONObject().apply {
            put("rows", value.rows)
            put("columns", value.columns)
            put("values", encode(value.features, intArrayOf(value.rows, value.columns)))
        })
        put("candidates", JSONArray().apply {
            value.candidates.forEach { candidate -> put(JSONObject().apply {
                put("id", candidate.id)
                put("timestamp", candidate.timestamp)
                put("probability", candidate.probability)
                put("kind", candidate.kind.wireName)
                put("sourceRangeIds", JSONArray(candidate.sourceRangeIds))
            }) }
        })
    }

    private fun evidenceJson(value: ServingSideHeadEvidence) = JSONObject().apply {
        put("modelId", value.modelId)
        put("threshold", value.threshold)
        put("peakProbability", value.peakProbability)
        put("peakTime", value.peakTime)
        put("crossesThreshold", value.crossesThreshold)
        put("nearestDetection", value.nearestDetection?.let { JSONObject().apply {
            put("time", it.time())
            put("confidence", it.confidence().toDouble())
        }} ?: JSONObject.NULL)
    }

    private fun derivedScoreJson(value: DerivedScore) = JSONObject().apply {
        put("team1Score", value.team1Score)
        put("team2Score", value.team2Score)
        put("servingTeamId", value.servingTeamId?.wireName ?: JSONObject.NULL)
        put("servingSide", value.servingSide?.wireName ?: JSONObject.NULL)
        put("ignoredPointCount", value.ignoredPointCount)
        put("reviewPointCount", value.reviewPointCount)
        put("points", JSONArray().apply {
            value.points.forEach { point -> put(JSONObject().apply {
                put("serveMarkerId", point.serveMarkerId)
                put("timestamp", point.timestampMs / 1_000.0)
                put("servingSide", point.servingSide.wireName)
                put("winnerTeamId", point.winnerTeamId?.wireName ?: JSONObject.NULL)
                put("status", point.status.wireName)
                put("team1ScoreAfter", point.team1ScoreAfter)
                put("team2ScoreAfter", point.team2ScoreAfter)
            }) }
        })
    }

    private fun analysisIntervals(values: List<AnalysisTypes.Interval>, prefix: String) = JSONArray().apply {
        values.forEachIndexed { index, interval -> put(JSONObject().apply {
            put(
                "id",
                "$prefix:${(index + 1).toString().padStart(4, '0')}:" +
                    "${secondsToMs(interval.start())}:${secondsToMs(interval.end())}",
            )
            put("start", interval.start())
            put("end", interval.end())
            put("confidence", interval.confidence().toDouble())
            interval.agreement()?.let { put("agreement", it) }
        }) }
    }

    private fun lastModifiedMs(value: Long): Long =
        if (value in 0 until 10_000_000_000L) value * 1_000 else value

    private fun updateDigest(digest: MessageDigest, input: FileInputStream, requested: Long) {
        var remaining = requested
        val buffer = ByteArray(64 * 1024)
        while (remaining > 0) {
            val count = input.read(buffer, 0, minOf(buffer.size.toLong(), remaining).toInt())
            check(count > 0) { "Source ended before its declared size" }
            digest.update(buffer, 0, count)
            remaining -= count
        }
    }
}
