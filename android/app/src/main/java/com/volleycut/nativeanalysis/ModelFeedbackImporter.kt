package com.volleycut.nativeanalysis

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.time.Instant
import java.util.Base64
import java.util.UUID
import kotlin.math.abs
import kotlin.math.max

internal object ModelFeedbackImporter {
    internal data class ImportedFeatureCache(
        val timestamps: DoubleArray,
        val baseFeatures: FloatArray,
        val visualFeatures: FloatArray,
        val audioFeatures: FloatArray,
        val contextualFeatures: FloatArray,
        val rallyProbabilities: FloatArray,
        val serveProbabilities: FloatArray,
        val deadStateProbabilities: FloatArray,
    ) {
        fun feedbackAnalysis() = ModelFeedbackAnalysis(
            timestamps = timestamps,
            baseFeatures = baseFeatures,
            rallyProbabilities = rallyProbabilities,
            serveProbabilities = serveProbabilities,
            deadStateProbabilities = deadStateProbabilities,
        )
    }

    internal data class Imported(
        val project: NativeProject,
        val draft: EditorDraft,
        val featureCache: ImportedFeatureCache?,
    )

    fun import(context: Context, text: String): NativeProject {
        val imported = parse(text)
        imported.featureCache?.let { hydrateFeatureCache(context, imported.project, it) }
        NativeProjectStore.save(context, imported.project)
        val seed = checkNotNull(imported.project.editorSeed())
        EditorDraftStore(context, seed).save(imported.draft)
        EditorProjectStore.save(context, seed)
        NativeProjectStore.setSelectedId(context, imported.project.id)
        return imported.project
    }

    internal fun parse(text: String, nowMs: Long = System.currentTimeMillis()): Imported {
        val bundle = JSONObject(text)
        require(bundle.optString("schema") == MODEL_FEEDBACK_SCHEMA &&
            bundle.optInt("schemaVersion") == MODEL_FEEDBACK_SCHEMA_VERSION
        ) { "Choose a VolleyCut model-feedback schema v3 JSON file" }
        val source = bundle.getJSONObject("source")
        require(source.getString("timelineCoordinates") == "seconds-from-start-of-source" &&
            !source.getBoolean("videoBytesIncluded")
        ) { "Feedback must use the original source timeline without embedded video" }
        val file = source.getJSONObject("file")
        val mediaJson = source.getJSONObject("media")
        val window = source.getJSONObject("gameWindow")
        val roiJson = source.getJSONObject("featureRoi")
        val duration = mediaJson.getDouble("duration")
        require(duration.isFinite() && duration > 0 &&
            mediaJson.getInt("width") > 0 && mediaJson.getInt("height") > 0 &&
            file.getLong("sizeBytes") >= 0 && file.getLong("lastModifiedMs") >= 0
        ) { "Feedback source metadata is invalid" }
        if (!file.isNull("sampledFingerprint")) {
            require(file.getString("sampledFingerprint").matches(
                Regex("sampled-sha256-v1:[0-9a-f]{64}"),
            )) { "Feedback source fingerprint is invalid" }
        }
        val windowStart = window.getDouble("start")
        val windowEnd = window.getDouble("end")
        require(windowStart.isFinite() && windowEnd.isFinite() &&
            windowStart >= 0 && windowEnd > windowStart && windowEnd <= duration
        ) { "Feedback game window is invalid" }
        val roiValues = listOf(
            roiJson.getDouble("x"), roiJson.getDouble("y"),
            roiJson.getDouble("width"), roiJson.getDouble("height"),
        )
        require(roiValues.all(Double::isFinite) && roiValues[0] >= 0 && roiValues[1] >= 0 &&
            roiValues[2] > 0 && roiValues[3] > 0 &&
            roiValues[0] + roiValues[2] <= 1.000001 &&
            roiValues[1] + roiValues[3] <= 1.000001
        ) { "Feedback ROI is invalid" }
        val durationMs = secondsToMs(duration)
        val inference = bundle.getJSONObject("initialInference")
        require(inference.getString("modelId") == FeatureSchema.MODEL_ID &&
            inference.getString("ensembleAlgorithmVersion") == FeatureSchema.ENSEMBLE_ALGORITHM_VERSION
        ) { "Feedback uses a different production ensemble" }
        val components = inference.getJSONArray("components")
        require(components.length() == 2)
        requireModelComponent(
            components.getJSONObject(0),
            FeatureSchema.ALL_LABELS_V2_MODEL_ID,
            FeatureSchema.ALL_LABELS_V2_BUNDLE_SHA256,
        )
        requireModelComponent(
            components.getJSONObject(1),
            FeatureSchema.PREVIOUS_PRODUCTION_MODEL_ID,
            FeatureSchema.PREVIOUS_PRODUCTION_BUNDLE_SHA256,
        )
        val rangesJson = inference.getJSONArray("ranges")
        val ranges = List(rangesJson.length()) { index -> rangesJson.getJSONObject(index).let {
            val start = it.getDouble("start")
            val end = it.getDouble("end")
            val confidence = it.getDouble("confidence")
            val agreement = if (!it.has("agreement") || it.isNull("agreement")) {
                null
            } else it.getString("agreement")
            require(start.isFinite() && end.isFinite() && start >= 0 && end > start &&
                end <= duration && confidence in 0.0..1.0 &&
                (agreement == null || ProductionEnsemble.isValidAgreement(agreement))
            ) { "Feedback contains an invalid production range" }
            SeedRange(secondsToMs(start), secondsToMs(end), confidence.toFloat(), agreement)
        } }
        val serveOutputs = inference.optJSONObject("componentServeOutputs")?.let { outputs ->
            val times = decodeNumeric(inference.getJSONObject("timestamps"))
            require(times.all { it in 0.0..duration } &&
                times.indices.drop(1).all { times[it] > times[it - 1] }
            ) { "Feedback inference timestamps are invalid" }
            AnalysisTypes.ProductionServeOutputs(
                decodeServeOutput(outputs.getJSONObject("allLabelsV2"), times, duration),
                decodeServeOutput(outputs.getJSONObject("previousProduction"), times, duration),
            ).also {
                require(it.allLabelsV2().modelId() == FeatureSchema.ALL_LABELS_V2_MODEL_ID)
                require(it.previousProduction().modelId() == FeatureSchema.PREVIOUS_PRODUCTION_MODEL_ID)
            }
        } ?: AnalysisTypes.ProductionServeOutputs.empty()
        val servingSide = inference.optJSONObject("servingSide")?.let(::decodeServingSide)
        val sideSwitchEnabled = inference.optJSONObject("sideSwitch") != null
        val suppression = inference.optJSONObject("suppression")?.let(::decodeSuppression)
        val id = "import-${UUID.randomUUID()}"
        val projectSource = ProjectSource(
            uri = "content://com.volleycut.nativeanalysis.import/$id",
            name = file.optString("name", "imported-recording.mp4"),
            size = file.optLong("sizeBytes", -1),
            lastModified = file.optLong("lastModifiedMs", -1),
            mimeType = file.optString("mimeType"),
            sampledFingerprint = if (file.isNull("sampledFingerprint")) null
                else file.optString("sampledFingerprint").takeIf(String::isNotBlank),
        )
        val projectWithoutCacheIdentity = NativeProject(
            id = id,
            source = projectSource,
            media = AnalysisTypes.MediaInfo(
                duration, mediaJson.optInt("width"), mediaJson.optInt("height"),
                mediaJson.optInt("rotation"), mediaJson.optString("videoCodec"),
                if (mediaJson.isNull("audioCodec")) null else mediaJson.optString("audioCodec"),
            ),
            analysisWindow = AnalysisTypes.AnalysisWindow(windowStart, windowEnd),
            roi = AnalysisTypes.Roi(
                roiJson.getDouble("x"), roiJson.getDouble("y"),
                roiJson.getDouble("width"), roiJson.getDouble("height"), "Imported feedback ROI",
            ),
            status = ProjectStatus.READY,
            ranges = ranges,
            productionComponents = inference.optJSONObject("productionComponents")?.let {
                AnalysisTypes.ProductionComponents(
                    decodeIntervals(it.optJSONArray("allLabelsV2") ?: JSONArray()),
                    decodeIntervals(it.optJSONArray("previousProduction") ?: JSONArray()),
                )
            } ?: AnalysisTypes.ProductionComponents.empty(),
            productionServeOutputs = serveOutputs,
            servingSide = servingSide,
            sideSwitchEnabled = sideSwitchEnabled,
            servingSideStatus = if (servingSide == null) {
                ServingSideAnalysisStatus.NOT_RUN
            } else ServingSideAnalysisStatus.READY,
            servingSideError = if (servingSide == null) "Imported bundle has no serving-side output" else null,
            suppression = suppression,
            modelId = inference.optString("modelId", FeatureSchema.MODEL_ID),
            createdAtMs = nowMs,
            updatedAtMs = nowMs,
        )
        val project = projectWithoutCacheIdentity.copy(
            servingSideCacheIdentity = servingSide?.let {
                ServingSideCache.identity(projectWithoutCacheIdentity)
            },
        )
        require(servingSide == null || servingSide.isReusableFor(ranges)) {
            "Serving-side candidates do not match the imported production ranges"
        }
        val seed = checkNotNull(project.editorSeed())
        val corrections = bundle.getJSONObject("corrections")
        val draft = decodeDraft(corrections, seed, durationMs)
        val featureCache = decodeFeatureCache(bundle.optJSONObject("features"), inference, project)
        return Imported(project, draft, featureCache)
    }

    private fun decodeFeatureCache(
        features: JSONObject?,
        inference: JSONObject,
        project: NativeProject,
    ): ImportedFeatureCache? {
        if (features == null) return null
        val rows = features.getInt("rows")
        val columns = features.getInt("columns")
        require(features.getInt("analysisFps") == FeatureSchema.ANALYSIS_FPS &&
            rows > 0 && columns == FeatureSchema.BASE.size
        ) { "Feedback features use an incompatible native feature schema" }
        val names = features.getJSONArray("names")
        require(names.length() == FeatureSchema.BASE.size &&
            (0 until FeatureSchema.BASE.size).all { names.getString(it) == FeatureSchema.BASE[it] }
        ) { "Feedback feature names do not match the native feature schema" }

        val timestamps = decodeNumeric(
            features.getJSONObject("timestamps"),
            intArrayOf(rows),
        )
        val baseFeatures = decodeNumeric(
            features.getJSONObject("values"),
            intArrayOf(rows, columns),
        ).toFiniteFloats("Feedback base features")
        val inferenceTimestamps = decodeNumeric(
            inference.getJSONObject("timestamps"),
            intArrayOf(rows),
        )
        require(timestamps.contentEquals(inferenceTimestamps)) {
            "Feedback feature and inference timestamps do not match"
        }
        val expectedTimes = AnalysisEngine.analysisTimes(
            project.media.durationSeconds(),
            project.analysisWindow.start(),
            project.analysisWindow.end(),
        )
        require(rows <= expectedTimes.size && timestamps.indices.all { index ->
            abs(timestamps[index] - expectedTimes[index]) <= 1e-9
        }) { "Feedback timestamps do not match the native analysis timeline" }

        val probabilities = inference.getJSONObject("probabilities")
        val rallyProbabilities = decodeNumeric(
            probabilities.getJSONObject("rally"),
            intArrayOf(rows),
        ).toProbabilityFloats("Feedback rally probabilities")
        val serveProbabilities = decodeNumeric(
            probabilities.getJSONObject("serve"),
            intArrayOf(rows),
        ).toProbabilityFloats("Feedback serve probabilities")
        val deadStateProbabilities = decodeNumeric(
            probabilities.getJSONObject("deadState"),
            intArrayOf(rows),
        ).toProbabilityFloats("Feedback dead-state probabilities")
        inference.optJSONObject("suppression")?.let { suppression ->
            require(decodeNumeric(
                suppression.getJSONObject("timestamps"),
                intArrayOf(rows),
            ).contentEquals(timestamps)) {
                "Feedback suppression timestamps do not match its features"
            }
        }

        val frameColumns = FeatureSchema.FRAME.size
        val temporalColumns = FeatureSchema.TEMPORAL.size
        val audioColumns = FeatureSchema.AUDIO.size
        val visual = FloatArray(rows * frameColumns)
        val audio = FloatArray(rows * audioColumns)
        repeat(rows) { row ->
            val source = row * columns
            baseFeatures.copyInto(
                visual,
                row * frameColumns,
                source,
                source + frameColumns,
            )
            baseFeatures.copyInto(
                audio,
                row * audioColumns,
                source + frameColumns + temporalColumns,
                source + columns,
            )
        }
        val rebuiltTemporal = FeatureMath.temporalVisualFeatures(visual, rows)
        require((0 until rows).all { row ->
            (0 until temporalColumns).all { column ->
                val imported = baseFeatures[row * columns + frameColumns + column]
                val rebuilt = rebuiltTemporal[row * temporalColumns + column]
                abs(imported - rebuilt) <= 1e-5f * max(1f, max(abs(imported), abs(rebuilt)))
            }
        }) { "Feedback temporal features cannot be reconstructed by the native feature contract" }
        val contextual = FeatureMath.contextualize(timestamps, baseFeatures, FeatureSchema.BASE)
        return ImportedFeatureCache(
            timestamps = timestamps,
            baseFeatures = baseFeatures,
            visualFeatures = visual,
            audioFeatures = audio,
            contextualFeatures = contextual,
            rallyProbabilities = rallyProbabilities,
            serveProbabilities = serveProbabilities,
            deadStateProbabilities = deadStateProbabilities,
        )
    }

    private fun hydrateFeatureCache(
        context: Context,
        project: NativeProject,
        imported: ImportedFeatureCache,
    ) {
        val source = project.source
        fun clear() = NativeFeatureCache.clearEntryWithSourceMetadata(
            context,
            android.net.Uri.parse(source.uri),
            source.name,
            source.size,
            source.lastModified,
            project.media,
            project.roi,
            FeatureSchema.FULL_SOURCE_FRAME_LIMIT,
            project.analysisWindow,
        )
        try {
            val cache = NativeFeatureCache.openWithSourceMetadata(
                context,
                android.net.Uri.parse(source.uri),
                source.name,
                source.size,
                source.lastModified,
                project.media,
                project.roi,
                FeatureSchema.FULL_SOURCE_FRAME_LIMIT,
                imported.timestamps.size,
                project.analysisWindow,
            )
            val loaded = cache.loadVisual()
            require(loaded.rows() == 0) { "Imported feature cache identity is not unique" }
            val writer = cache.newVisualWriter(loaded)
            val frameColumns = FeatureSchema.FRAME.size
            val row = FloatArray(frameColumns)
            imported.timestamps.indices.forEach { index ->
                imported.visualFeatures.copyInto(
                    row,
                    0,
                    index * frameColumns,
                    (index + 1) * frameColumns,
                )
                writer.append(index, imported.timestamps[index], row)
            }
            writer.finish(AnalysisTypes.VideoFeatures(
                imported.visualFeatures,
                imported.timestamps,
                project.analysisWindow.end(),
                false,
                "model-feedback-import",
                false,
                0,
                0,
                0,
                0.0,
                0.0,
                0.0,
                emptyMap(),
            ))
            cache.storeAudio(imported.audioFeatures, imported.timestamps.size)
            cache.storeContext(imported.contextualFeatures, imported.timestamps.size)

            val retainedVisual = cache.loadVisual()
            val retainedAudio = cache.loadAudio(imported.timestamps.size)
            val retainedContext = cache.loadContext(imported.timestamps.size)
            val stats = cache.stats()
            require(stats.complete() && stats.failure().isBlank() &&
                retainedVisual.complete() &&
                retainedVisual.times().contentEquals(imported.timestamps) &&
                retainedVisual.values().contentEquals(imported.visualFeatures) &&
                retainedAudio?.contentEquals(imported.audioFeatures) == true &&
                retainedContext?.contentEquals(imported.contextualFeatures) == true
            ) { "Imported features could not be retained in the native feature cache" }
        } catch (error: Exception) {
            clear()
            throw IllegalArgumentException(
                "Could not retain the imported native feature cache: ${error.message}",
                error,
            )
        }
    }

    private fun decodeDraft(corrections: JSONObject, seed: EditorSeed, durationMs: Long): EditorDraft {
        val ranges = corrections.getJSONArray("correctedRanges")
        val ignored = corrections.optJSONArray("ignoredIntervals") ?: JSONArray()
        val score = requireNotNull(ScoreTrackingJson.decodeWire(
            corrections.getJSONObject("scoreTracking").getJSONObject("state"),
            durationMs,
        )) { "Feedback contains invalid score tracking state" }
        val suppression = corrections.optJSONObject("suppression")
        val decisionOverrides = suppression?.optJSONObject("decisionOverrides")?.let { values ->
            buildMap {
                values.keys().forEach { id ->
                    put(id, requireNotNull(SuppressionDecision.fromWireName(values.getString(id))) {
                        "Invalid suppression decision for $id"
                    })
                }
            }
        }.orEmpty()
        val scopeOverrides = suppression?.optJSONObject("suppressionScopeOverrides")?.let { values ->
            buildMap {
                values.keys().forEach { id ->
                    val raw = values.getString(id)
                    require(raw in SuppressionScope.entries.map { it.wireName }) {
                        "Invalid suppression scope for $id"
                    }
                    put(id, SuppressionScope.fromWireName(raw))
                }
            }
        }.orEmpty()
        val touchedJson = suppression?.optJSONArray("userTouchedCutIds") ?: JSONArray()
        val selectedPolicyName = suppression?.optString("selectedPolicy", "none") ?: "none"
        require(selectedPolicyName in SuppressionPolicyEngine.Policy.entries.map { it.wireName }) {
            "Invalid suppression policy"
        }
        val draft = EditorDraft(
            sourceRevision = seed.sourceRevision,
            updatedAtMs = runCatching {
                Instant.parse(corrections.getString("updatedAt")).toEpochMilli()
            }.getOrDefault(System.currentTimeMillis()),
            beforePaddingMs = secondsToMs(corrections.optDouble("beforePaddingSeconds", 2.0)),
            afterPaddingMs = secondsToMs(corrections.optDouble("afterPaddingSeconds", 2.0)),
            joinGapMs = secondsToMs(corrections.optDouble("joinGapSeconds", 3.0)),
            cuts = List(ranges.length()) { index -> ranges.getJSONObject(index).let {
                val origin = it.optString("origin")
                require(origin == "cached-label" || origin == "manual") {
                    "Invalid corrected-range origin"
                }
                EditableCut(
                    it.getString("id"), secondsToMs(it.getDouble("coreStart")),
                    secondsToMs(it.getDouble("coreEnd")), secondsToMs(it.getDouble("keepStart")),
                    secondsToMs(it.getDouble("keepEnd")), it.optDouble("confidence", 1.0).toFloat(),
                    it.optBoolean("included", true),
                    if (origin == "manual") CutOrigin.MANUAL else CutOrigin.INFERRED,
                    if (it.isNull("agreement")) null else it.optString("agreement"),
                )
            } },
            ignoredIntervals = List(ignored.length()) { index -> ignored.getJSONObject(index).let {
                IgnoredSourceInterval(
                    it.getString("id"), secondsToMs(it.getDouble("start")),
                    secondsToMs(it.getDouble("end")), it.getString("reason"),
                )
            } },
            selectedSuppressionPolicy = SuppressionPolicyEngine.Policy.fromWireName(
                selectedPolicyName,
            ),
            suppressionDecisionOverrides = decisionOverrides,
            suppressionScopeOverrides = scopeOverrides,
            userTouchedCutIds = buildSet {
                repeat(touchedJson.length()) { add(touchedJson.getString(it)) }
            },
            scoreTracking = ScoreReducer.seedModelMarkers(
                score,
                seed.servingSide,
                seed.sideSwitch,
                seed.sideSwitchEnabled,
            ),
            renderScoreOverlay = false,
            renderScoreTimeline = false,
        )
        require(draft.cuts.map { it.id }.distinct().size == draft.cuts.size &&
            draft.beforePaddingMs in 0..MAX_PADDING_MS &&
            draft.afterPaddingMs in 0..MAX_PADDING_MS &&
            draft.joinGapMs in 0..MAX_JOIN_GAP_MS &&
            draft.cuts.all {
                it.id.isNotBlank() && it.keepStartMs in 0..it.coreStartMs &&
                    it.coreStartMs < it.coreEndMs && it.coreEndMs <= it.keepEndMs &&
                    it.keepEndMs <= durationMs
            } && draft.ignoredIntervals.all {
                it.id.isNotBlank() && it.startMs in 0 until it.endMs && it.endMs <= durationMs
            } && draft.userTouchedCutIds.all { touched -> draft.cuts.any { it.id == touched } }
        ) { "Feedback corrections are not compatible with this editor version" }
        return draft
    }

    private fun decodeServeOutput(
        json: JSONObject,
        times: DoubleArray,
        duration: Double,
    ): AnalysisTypes.ProductionServeOutput {
        val probabilities = decodeNumeric(json.getJSONObject("probabilities")).map(Double::toFloat).toFloatArray()
        val detections = json.optJSONArray("detections") ?: JSONArray()
        require(probabilities.size == times.size && probabilities.all { it in 0f..1f })
        return AnalysisTypes.ProductionServeOutput(
            json.getString("modelId"), times.clone(), probabilities,
            List(detections.length()) { index -> detections.getJSONObject(index).let {
                val time = it.getDouble("time")
                val confidence = it.getDouble("confidence")
                require(time.isFinite() && time in 0.0..duration &&
                    confidence.isFinite() && confidence in 0.0..1.0
                ) { "Feedback contains an invalid serve detection" }
                AnalysisTypes.Serve(time, confidence.toFloat())
            } },
        )
    }

    private fun requireModelComponent(json: JSONObject, modelId: String, sha256: String) {
        require(json.getString("modelId") == modelId &&
            json.getString("bundleSha256") == sha256
        ) { "Feedback uses a different production component" }
    }

    private fun decodeServingSide(json: JSONObject): ServingSideOutput {
        val features = json.getJSONObject("features")
        val rows = features.getInt("rows")
        val columns = features.getInt("columns")
        val candidates = json.getJSONArray("candidates")
        return ServingSideOutput(
            json.getString("modelId"), json.getString("modelFingerprint"),
            json.getString("featureVersion"), json.getString("anchorContract"),
            rows, columns, decodeNumeric(features.getJSONObject("values")),
            List(candidates.length()) { index -> candidates.getJSONObject(index).let { item ->
                val interval = item.getJSONObject("interval")
                val evidence = item.getJSONObject("serveEvidence")
                val reasons = item.getJSONArray("reviewReasons")
                ServingSideCandidate(
                    item.getString("id"), item.getDouble("anchor"),
                    interval.getDouble("start"), interval.getDouble("end"),
                    if (interval.isNull("agreement")) null else interval.optString("agreement"),
                    item.getDouble("nearProbability"),
                    ServingSide.fromWireName(item.getString("side")) ?: error("Invalid side"),
                    ServingSideVerdict.fromWireName(item.getString("verdict")) ?: error("Invalid verdict"),
                    ServingSideDecisionSource.fromWireName(item.getString("serveDecisionSource"))
                        ?: error("Invalid decision source"),
                    List(reasons.length()) { reason ->
                        ServingSideReviewReason.fromWireName(reasons.getString(reason))
                            ?: error("Invalid review reason")
                    },
                    decodeEvidence(evidence.getJSONObject("allLabelsV2")),
                    decodeEvidence(evidence.getJSONObject("previousProduction")),
                )
            } },
        ).also { output ->
            require(output.modelId == SERVING_SIDE_MODEL_ID &&
                output.modelFingerprint == SERVING_SIDE_MODEL_FINGERPRINT &&
                output.featureVersion == SERVING_SIDE_FEATURE_VERSION &&
                output.anchorContract == SERVING_SIDE_ANCHOR_CONTRACT &&
                output.columns == SERVING_SIDE_FEATURE_COLUMNS &&
                output.rows == output.candidates.size &&
                output.rawFeatures.size == output.rows * output.columns &&
                output.candidates.all {
                    it.side != ServingSide.REVIEW && it.nearProbability in 0.0..1.0 &&
                        it.anchor == it.intervalStart && it.intervalEnd > it.intervalStart &&
                        it.allLabelsV2Evidence.modelId == FeatureSchema.ALL_LABELS_V2_MODEL_ID &&
                        it.previousProductionEvidence.modelId == FeatureSchema.PREVIOUS_PRODUCTION_MODEL_ID
                }
            ) { "Feedback serving-side output is incompatible" }
        }
    }

    private fun decodeEvidence(json: JSONObject) = ServingSideHeadEvidence(
        json.getString("modelId"), json.getDouble("threshold"),
        json.getDouble("peakProbability"), json.getDouble("peakTime"),
        json.getBoolean("crossesThreshold"), json.optJSONObject("nearestDetection")?.let {
            AnalysisTypes.Serve(it.getDouble("time"), it.getDouble("confidence").toFloat())
        },
    )

    private fun decodeSuppression(json: JSONObject): AnalysisTypes.SuppressionAnalysis {
        require(json.getString("modelId") == FeatureSchema.SUPPRESSION_MODEL_ID)
        require(json.getString("artifactSha256") == FeatureSchema.SUPPRESSION_ARTIFACT_SHA256)
        require(json.getString("weightsSha256") == FeatureSchema.SUPPRESSION_WEIGHTS_SHA256)
        require(json.getString("decoderVersion") == FeatureSchema.SUPPRESSION_DECODER_VERSION)
        require(json.optInt("policyContractVersion") == 1)
        val timestamps = decodeNumeric(json.getJSONObject("timestamps"))
        val probabilities = decodeNumeric(json.getJSONObject("probabilities"))
            .map(Double::toFloat).toFloatArray()
        require(probabilities.size == timestamps.size)
        val suggestions = json.getJSONArray("suggestions")
        return AnalysisTypes.SuppressionAnalysis(
            json.getString("modelId"), json.getString("artifactSha256"),
            json.getString("weightsSha256"), json.getString("decoderVersion"),
            probabilities,
            decodeIntervals(json.getJSONArray("decodedIntervals")),
            List(suggestions.length()) { index -> suggestions.getJSONObject(index).let { item ->
                AnalysisTypes.SuppressionSuggestion(
                    item.getString("logicalId"), item.getString("id"),
                    secondsToMs(item.getDouble("start")), secondsToMs(item.getDouble("end")),
                    item.getDouble("score").toFloat(),
                    item.getJSONArray("sourceProductionIds").let { ids ->
                        List(ids.length()) { ids.getString(it) }
                    },
                    item.getJSONArray("eligiblePolicyIds").let { ids ->
                        List(ids.length()) { ids.getString(it) }
                    },
                )
            } },
        )
    }

    private fun decodeIntervals(json: JSONArray) = List(json.length()) { index -> json.getJSONObject(index).let {
        AnalysisTypes.Interval(
            it.getDouble("start"), it.getDouble("end"), it.optDouble("confidence", 1.0).toFloat(),
            if (it.isNull("agreement")) null else it.optString("agreement"),
        )
    } }

    private fun decodeNumeric(json: JSONObject, expectedShape: IntArray? = null): DoubleArray {
        require(json.optString("encoding") == "base64" &&
            json.optString("byteOrder") == "little-endian"
        )
        val shapeJson = json.getJSONArray("shape")
        require(shapeJson.length() > 0)
        val shape = IntArray(shapeJson.length()) { index ->
            shapeJson.getInt(index).also { require(it >= 0) }
        }
        expectedShape?.let { require(shape.contentEquals(it)) }
        val elements = shape.fold(1L) { product, size ->
            require(size == 0 || product <= Long.MAX_VALUE / size)
            product * size
        }
        val bytesPerElement = when (json.getString("dataType")) {
            "float64" -> Double.SIZE_BYTES
            "float32" -> Float.SIZE_BYTES
            else -> error("Unsupported numeric array type")
        }
        val bytes = Base64.getDecoder().decode(json.getString("data"))
        require(elements <= Int.MAX_VALUE && elements * bytesPerElement ==
            bytes.size.toLong()
        ) { "Numeric array shape does not match its data" }
        return when (json.getString("dataType")) {
            "float64" -> ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer().let {
                DoubleArray(it.remaining()).also(it::get)
            }
            "float32" -> ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().let {
                FloatArray(it.remaining()).also(it::get).map(Float::toDouble).toDoubleArray()
            }
            else -> error("Unsupported numeric array type")
        }.also { values -> require(values.size.toLong() == elements && values.all(Double::isFinite)) }
    }

    private fun DoubleArray.toFiniteFloats(label: String) = FloatArray(size) { index ->
        this[index].toFloat().also { require(it.isFinite()) { "$label contain an out-of-range value" } }
    }

    private fun DoubleArray.toProbabilityFloats(label: String) = toFiniteFloats(label).also { values ->
        require(values.all { it in 0f..1f }) { "$label must be between zero and one" }
    }
}
