import Foundation

enum ProjectSession {
    static func create(sourceURL: URL, result: AnalysisResult) throws -> ProjectDocument {
        let window = TimeRange(startMs: ms(result.start), endMs: ms(result.end))
        let cuts = result.intervals.enumerated().map { index, interval in
            EditableCut(id: String(format: "R%03d", index + 1), coreStartMs: ms(interval.start), coreEndMs: ms(interval.end),
                        confidence: interval.confidence, agreement: interval.agreement)
        }
        var draft = EditorMath.newDraft(ranges: cuts, durationMs: ms(result.media.duration), gameWindow: window, sourceRevision: result.cacheIdentity)
        if let serving = result.servingSide {
            draft.scoreTracking = try ScoreReducer.seedModelMarkers(ScoreTracking(), output: serving, sideSwitchOutput: result.sideSwitch).jsonValue()
        }
        var project = ProjectDocument(sourceName: sourceURL.lastPathComponent, durationMs: ms(result.media.duration), gameWindow: window, draft: draft)
        let properties = try sourceURL.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
        let source = ProjectArchive.source(projectId: project.id, name: sourceURL.lastPathComponent,
            sizeBytes: Int64(properties.fileSize ?? 0), lastModifiedMs: Int64((properties.contentModificationDate?.timeIntervalSince1970 ?? 0) * 1000),
            fingerprint: try ProjectArchive.sampledFingerprint(url: sourceURL), duration: result.media.duration,
            width: result.media.width, height: result.media.height, rotation: result.media.rotation,
            videoCodec: result.media.videoCodec ?? "unknown", audioCodec: result.media.audioCodec,
            gameWindow: window, roi: [result.roi.x, result.roi.y, result.roi.width, result.roi.height])
        var inference = ProjectArchive.initialInference(ranges: draft.cuts)
        inference["productionComponents"] = .object([
            "allLabelsV2": component(result.allLabels.intervals, prefix: "all-labels-v2"),
            "previousProduction": component(result.previous.intervals, prefix: "previous-production")])
        if let serving = result.servingSide { inference["servingSide"] = try servingFeedback(serving) }
        if let switches = result.sideSwitch { inference["sideSwitch"] = try switchFeedback(switches) }
        if let suppression = result.suppression {
            inference["suppression"] = .object([
                "modelId": .string(suppression.modelId), "artifactSha256": .string(suppression.artifactSha256),
                "weightsSha256": .string(suppression.weightsSha256), "decoderVersion": .string(suppression.decoderVersion),
                "policyContractVersion": .number(1),
                "identicalPolicyResults": .bool(Set(suppression.suggestions.map { $0.eligiblePolicyIds.joined(separator: ",") }).count <= 1 && suppression.suggestions.allSatisfy { $0.eligiblePolicyIds.count == 3 }),
                "timestamps": try FeedbackNumericArray(result.times, shape: [result.times.count]).json,
                "probabilities": try FeedbackNumericArray(suppression.probabilities, shape: [result.times.count]).json,
                "decodedIntervals": component(suppression.decodedIntervals, prefix: "suppression"),
                "suggestions": .array(suppression.suggestions.map { suggestion in .object([
                    "id": .string(suggestion.fragmentId), "logicalId": .string(suggestion.logicalId),
                    "suppressionEventId": .string(suggestion.logicalId), "start": .number(Double(suggestion.startMs) / 1000),
                    "end": .number(Double(suggestion.endMs) / 1000), "score": .number(Double(suggestion.score)),
                    "sourceProductionIds": .array(suggestion.sourceProductionIds.map(JSONValue.string)),
                    "eligiblePolicyIds": .array(suggestion.eligiblePolicyIds.map(JSONValue.string))
                ]) })])
        }
        let analysis = FeedbackAnalysis(timestamps: result.times, baseFeatures: result.base, featureNames: FeatureSchema.base,
            rallyProbabilities: result.allLabels.rallyProbabilities, serveProbabilities: result.allLabels.serveProbabilities,
            deadStateProbabilities: result.allLabels.deadStateProbabilities)
        project.feedback = try ProjectArchive.create(source: source, initialInference: inference, draft: draft, analysis: analysis)
        if let error = result.scoreError { project.feedback?["scorePreparationError"] = .string(error) }
        return project
    }
    static func switchFeedback(_ output: SideSwitchOutput) throws -> JSONValue {
        .object(["modelId": .string(output.modelId), "modelFingerprint": .string(output.modelFingerprint),
            "featureVersion": .string(output.featureVersion), "candidateContract": .string(output.candidateContract),
            "features": .object(["rows": .number(Double(output.rows)), "columns": .number(Double(output.columns)),
                                  "values": try FeedbackNumericArray(output.features, shape: [output.rows, output.columns]).json]),
            "candidates": .array(output.candidates.map { value in
                .object(["id": .string(value.id), "timestamp": .number(value.timestamp), "probability": .number(value.probability),
                         "kind": .string(value.kind.rawValue), "sourceRangeIds": .array(value.sourceRangeIds.map(JSONValue.string))])
            })])
    }
    static func servingFeedback(_ output: ServingSideOutput) throws -> JSONValue {
        .object([
            "modelId": .string(output.modelId), "modelFingerprint": .string(output.modelFingerprint),
            "featureVersion": .string(output.featureVersion), "anchorContract": .string(output.anchorContract),
            "features": .object(["rows": .number(Double(output.rows)), "columns": .number(Double(output.columns)),
                                  "values": try FeedbackNumericArray(output.rawFeatures, shape: [output.rows, output.columns]).json]),
            "candidates": .array(output.candidates.map { candidate in
                var interval: JSONValue = .object(["start": .number(candidate.intervalStart), "end": .number(candidate.intervalEnd)])
                if let agreement = candidate.agreement { interval["agreement"] = .string(agreement) }
                return .object(["id": .string(candidate.id), "anchor": .number(candidate.anchor), "interval": interval,
                    "nearProbability": .number(candidate.nearProbability), "side": .string(candidate.side.rawValue),
                    "verdict": .string(candidate.verdict.rawValue), "serveDecisionSource": .string(candidate.serveDecisionSource.rawValue),
                    "reviewReasons": .array(candidate.reviewReasons.map { .string($0.rawValue) }),
                    "serveEvidence": .object(["allLabelsV2": candidate.allLabelsV2Evidence ?? .null,
                                              "previousProduction": candidate.previousProductionEvidence ?? .null])])
            })])
    }
    static func component(_ intervals: [Interval], prefix: String) -> JSONValue {
        .array(intervals.enumerated().map { index, interval in
            var value: JSONValue = .object([
                "id": .string(String(format: "%@:%04d:%lld:%lld", prefix, index + 1, ms(interval.start), ms(interval.end))),
                "start": .number(interval.start), "end": .number(interval.end), "confidence": .number(Double(interval.confidence))])
            if let agreement = interval.agreement { value["agreement"] = .string(agreement) }
            return value
        })
    }
    static func ms(_ seconds: Double) -> Int64 { Int64(floor(seconds * 1000 + 0.5)) }
}
