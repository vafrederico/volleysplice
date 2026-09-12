import Foundation
#if canImport(CryptoKit)
import CryptoKit
#endif

/// Shared web/Android little-endian row-major numeric wire representation.
public struct FeedbackNumericArray: Codable, Equatable, Sendable {
    public var encoding = "base64"
    public var byteOrder = "little-endian"
    public var dataType: String
    public var shape: [Int]
    public var data: String
    public init(_ values: [Float], shape: [Int]) throws {
        self.dataType = "float32"; self.shape = shape
        var bytes = Data(capacity: values.count * 4)
        for value in values {
            guard value.isFinite else { throw ProjectError.invalid("Non-finite feature value") }
            var bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { bytes.append(contentsOf: $0) }
        }
        self.data = bytes.base64EncodedString(); _ = try decoded()
    }
    public init(_ values: [Double], shape: [Int]) throws {
        self.dataType = "float64"; self.shape = shape
        var bytes = Data(capacity: values.count * 8)
        for value in values {
            guard value.isFinite else { throw ProjectError.invalid("Non-finite timestamp") }
            var bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { bytes.append(contentsOf: $0) }
        }
        self.data = bytes.base64EncodedString(); _ = try decoded()
    }
    public func decoded(expectedShape: [Int]? = nil, expectedType: String? = nil) throws -> [Double] {
        guard encoding == "base64", byteOrder == "little-endian", ["float32", "float64"].contains(dataType),
              !shape.isEmpty, shape.count <= 4, shape.allSatisfy({ $0 >= 0 && $0 <= 25_000_000 }),
              expectedShape == nil || shape == expectedShape, expectedType == nil || dataType == expectedType else {
            throw ProjectError.invalid("Invalid numeric-array encoding, shape, or type")
        }
        var count = 1
        for dimension in shape {
            let product = count.multipliedReportingOverflow(by: dimension)
            guard !product.overflow, product.partialValue <= 25_000_000 else { throw ProjectError.invalid("Numeric array exceeds size limit") }
            count = product.partialValue
        }
        let stride = dataType == "float32" ? 4 : 8
        guard data.utf8.count <= ((count * stride + 2) / 3) * 4,
              let bytes = Data(base64Encoded: data), bytes.count == count * stride else {
            throw ProjectError.invalid("Numeric-array shape does not match its data")
        }
        var output = [Double](); output.reserveCapacity(count)
        for offset in Swift.stride(from: 0, to: bytes.count, by: stride) {
            var bits: UInt64 = 0
            for index in 0..<stride { bits |= UInt64(bytes[offset + index]) << (index * 8) }
            let value = stride == 4 ? Double(Float(bitPattern: UInt32(bits))) : Double(bitPattern: bits)
            guard value.isFinite else { throw ProjectError.invalid("Numeric array contains non-finite values") }
            output.append(value)
        }
        return output
    }
    public var json: JSONValue { get throws { try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(self)) } }
    public static func decode(_ value: JSONValue, shape: [Int]? = nil, type: String? = nil) throws -> [Double] {
        try JSONDecoder().decode(Self.self, from: JSONEncoder().encode(value)).decoded(expectedShape: shape, expectedType: type)
    }
}

public struct FeedbackAnalysis: Sendable {
    public var timestamps: [Double]
    public var baseFeatures: [Float]
    public var featureNames: [String]
    public var rallyProbabilities: [Float]
    public var serveProbabilities: [Float]
    public var deadStateProbabilities: [Float]
    public init(timestamps: [Double], baseFeatures: [Float], featureNames: [String], rallyProbabilities: [Float],
                serveProbabilities: [Float], deadStateProbabilities: [Float]) {
        self.timestamps = timestamps; self.baseFeatures = baseFeatures; self.featureNames = featureNames
        self.rallyProbabilities = rallyProbabilities; self.serveProbabilities = serveProbabilities
        self.deadStateProbabilities = deadStateProbabilities
    }
}

public enum ProjectArchive {
    public static let ensembleAlgorithmVersion = "overlap-union-disagreement-v1"
    public static let componentIds = ["model-1ca43e38eefc", "model-9c92b8e9333f"]
    public static let componentHashes = ["d2c2c11e8fed8b6c6ad77d244b613e81d5bab101939a8f57be5166b45ebca78f",
                                         "d8cc42f70bc10576a5e03251b05981ceeee1a61a15c61cc5dfb68dd631e6f90d"]
    public static let modelId = "ensemble-" + ensembleAlgorithmVersion + "-" + componentHashes.joined(separator: "-")
    static func required(_ value: JSONValue?, _ field: String) throws -> JSONValue {
        guard let value = value else { throw ProjectError.invalid("Missing \(field)") }; return value
    }
    static func number(_ value: JSONValue?, _ field: String) throws -> Double {
        guard let v = value?.double, v.isFinite else { throw ProjectError.invalid("Invalid \(field)") }; return v
    }
    static func ms(_ value: JSONValue?, _ field: String) throws -> Int64 {
        let v = try number(value, field)
        guard v >= 0, v < Double(Int64.max / 1_000) else { throw ProjectError.invalid("Invalid source timestamp") }
        return Int64((v * 1_000).rounded())
    }
    static func strings(_ values: [String]) -> JSONValue { .array(values.map(JSONValue.string)) }
    static func dictionary(_ values: [String: String]) -> JSONValue { .object(values.mapValues(JSONValue.string)) }
    static func stringDictionary(_ value: JSONValue?) throws -> [String: String] {
        guard let value else { return [:] }
        guard case .object(let values) = value else { throw ProjectError.invalid("Invalid correction map") }
        return try values.mapValues { value in
            guard let text = value.string else { throw ProjectError.invalid("Correction map values must be strings") }
            return text
        }
    }
    static func validateNumericPayloads(_ value: JSONValue) throws {
        switch value {
        case .object(let values):
            if values["encoding"]?.string == "base64" { _ = try FeedbackNumericArray.decode(value) }
            else { for child in values.values { try validateNumericPayloads(child) } }
        case .array(let values): for child in values { try validateNumericPayloads(child) }
        default: break
        }
    }
    static func wireRange(_ cut: EditableCut) -> JSONValue {
        var result: JSONValue = .object(["id": .string(cut.id), "start": .number(Double(cut.coreStartMs) / 1_000),
            "end": .number(Double(cut.coreEndMs) / 1_000), "confidence": .number(Double(cut.confidence))])
        if let agreement = cut.agreement { result["agreement"] = .string(agreement) }; return result
    }
    public static func source(projectId: String, name: String, sizeBytes: Int64, lastModifiedMs: Int64,
                              fingerprint: String? = nil, duration: Double, width: Int, height: Int, rotation: Int = 0,
                              videoCodec: String = "video/avc", audioCodec: String? = nil,
                              gameWindow: TimeRange? = nil, roi: [Double] = [0, 0, 1, 1]) -> JSONValue {
        let safeDurationMs = duration.isFinite && duration > 0 && duration < Double(Int64.max / 1_000) ? Int64((duration * 1_000).rounded()) : 0
        let window = gameWindow ?? TimeRange(startMs: 0, endMs: safeDurationMs)
        return .object([
            "projectId": .string(projectId), "analysisId": .string(projectId + "-" + modelId + "-native-source"),
            "timelineCoordinates": .string("seconds-from-start-of-source"), "videoBytesIncluded": .bool(false),
            "runtimeVariant": .string("native-ios-dsp-v1"),
            "file": .object(["name": .string(name), "sizeBytes": .number(Double(sizeBytes)),
                "lastModifiedMs": .number(Double(lastModifiedMs)), "mimeType": .string("video/mp4"),
                "sampledFingerprint": fingerprint.map(JSONValue.string) ?? .null]),
            "media": .object(["duration": .number(duration), "mimeType": .string("video/mp4"),
                "width": .number(Double(width)), "height": .number(Double(height)), "rotation": .number(Double(rotation)),
                "videoCodec": .string(videoCodec), "videoCodecString": .null, "canDecodeVideo": .bool(true),
                "hasAudio": .bool(audioCodec != nil), "audioCodec": audioCodec.map(JSONValue.string) ?? .null,
                "sampleRate": .null, "channels": .null, "canDecodeAudio": .bool(audioCodec != nil)]),
            "gameWindow": .object(["start": .number(Double(window.startMs) / 1_000), "end": .number(Double(window.endMs) / 1_000)]),
            "featureRoi": .object(["x": .number(roi.count == 4 ? roi[0] : 0), "y": .number(roi.count == 4 ? roi[1] : 0),
                "width": .number(roi.count == 4 ? roi[2] : 1), "height": .number(roi.count == 4 ? roi[3] : 1)])
        ])
    }
    public static func initialInference(ranges: [EditableCut]) -> JSONValue {
        .object(["modelId": .string(modelId), "ensembleAlgorithmVersion": .string(ensembleAlgorithmVersion),
            "components": .array(zip(componentIds, componentHashes).map { .object(["modelId": .string($0), "bundleSha256": .string($1)]) }),
            "ranges": .array(ranges.map { cut in var value = wireRange(cut); value["included"] = .bool(true); return value }),
            "probabilityModelId": .string(componentIds[0]), "productionComponents": .object(["allLabelsV2": .array([]), "previousProduction": .array([])]),
            "suppression": .null, "servingSide": .null, "sideSwitch": .null])
    }
    /// Optional inference fields remain unchanged, including raw component ranges, suppression, and serving-side matrices.
    public static func create(source: JSONValue, initialInference: JSONValue, draft: EditorDraft,
                              analysis: FeedbackAnalysis?, derivedScore: JSONValue? = nil,
                              generatedAt: Date = Date()) throws -> JSONValue {
        let durationMs = try ms(source["media"]?["duration"], "source duration")
        try draft.validate(durationMs: durationMs)
        var inference = initialInference
        let rows = analysis?.timestamps.count ?? 0
        let times = analysis?.timestamps ?? []
        inference["timestamps"] = try FeedbackNumericArray(times, shape: [rows]).json
        inference["probabilities"] = .object([
            "rally": try FeedbackNumericArray(analysis?.rallyProbabilities ?? [Float](), shape: [rows]).json,
            "serve": try FeedbackNumericArray(analysis?.serveProbabilities ?? [Float](), shape: [rows]).json,
            "deadState": try FeedbackNumericArray(analysis?.deadStateProbabilities ?? [Float](), shape: [rows]).json])
        let features: JSONValue
        if let analysis {
            features = .object(["analysisFps": .number(4), "rows": .number(Double(rows)),
                "columns": .number(Double(analysis.featureNames.count)), "names": strings(analysis.featureNames),
                "timestamps": try FeedbackNumericArray(times, shape: [rows]).json,
                "values": try FeedbackNumericArray(analysis.baseFeatures, shape: [rows, analysis.featureNames.count]).json])
        } else {
            features = .null
            inference["suppression"] = .null
            inference["componentServeOutputs"] = nil
        }
        var warnings: [JSONValue] = []
        if analysis == nil { warnings.append(.string("Retained base features and probability traces are unavailable; inference and corrections are included.")) }
        if source["file"]?["sampledFingerprint"]?.string == nil { warnings.append(.string("The source fingerprint is unavailable; pair this bundle by source metadata.")) }
        if inference["servingSide"] == .null || inference["servingSide"] == nil { warnings.append(.string("Serving-side features and initial verdicts are unavailable; score-marker corrections are included.")) }
        var bundle: JSONValue = .object(["schema": .string("volleycut-model-feedback"), "schemaVersion": .number(3),
            "generatedAt": .string(ISO8601DateFormatter().string(from: generatedAt)), "source": source,
            "initialInference": inference, "features": features,
            "warnings": .array(warnings)])
        try updateCorrections(&bundle, draft: draft, derivedScore: derivedScore)
        _ = try importBundle(bundle)
        return bundle
    }
    public static func suppressionRegions(_ bundle: JSONValue) throws -> [SuppressionRegion] {
        let duration = try ms(bundle["source"]?["media"]?["duration"], "source duration")
        return try (bundle["initialInference"]?["suppression"]?["suggestions"]?.array ?? []).map { region in
            guard let id = region["id"]?.string, let logicalId = region["logicalId"]?.string else { throw ProjectError.invalid("Invalid suppression suggestion identity") }
            let result = SuppressionRegion(id: id, logicalId: logicalId, startMs: try ms(region["start"], "suppression start"),
                                     endMs: try ms(region["end"], "suppression end"),
                                     eligiblePolicyIds: region["eligiblePolicyIds"]?.array?.compactMap(\.string) ?? [])
            guard !id.isEmpty, !logicalId.isEmpty, result.startMs < result.endMs, result.endMs <= duration,
                  result.eligiblePolicyIds.allSatisfy({ ["conservative", "balanced", "aggressive"].contains($0) }) else { throw ProjectError.invalid("Invalid suppression suggestion range or policies") }
            return result
        }
    }
    /// Call on the retained imported bundle to preserve all initial model outputs and binary arrays.
    public static func updateCorrections(_ bundle: inout JSONValue, draft: EditorDraft, derivedScore: JSONValue? = nil) throws {
        try draft.validate(durationMs: try ms(bundle["source"]?["media"]?["duration"], "source duration"))
        let suggestions = try suppressionRegions(bundle)
        let analysisBounds = TimeRange(startMs: try ms(bundle["source"]?["gameWindow"]?["start"], "game start"),
                                       endMs: try ms(bundle["source"]?["gameWindow"]?["end"], "game end"))
        let intervals = EditorMath.finalIntervals(draft, suppression: suggestions, bounds: analysisBounds)
        let keptIds = Set(intervals.flatMap(\.cutIds))
        let excluded = draft.cuts.filter { $0.origin == .inferred && (!$0.included || !keptIds.contains($0.id)) }.map(\.id).sorted()
        let score = try ScoreTracking.fromJSON(draft.scoreTracking, durationMs: try ms(bundle["source"]?["media"]?["duration"], "source duration"))
        let visibleScore = ScoreReducer.visibleTracking(score, ignoredIntervals: draft.ignoredIntervals, excludedRallyIds: Set(excluded))
        let computedScore = ScoreReducer.deriveAt(visibleScore).jsonValue()
        var corrections = bundle["corrections"] ?? .object([:])
        // Additive UI extension: preserve unrelated preferences from imported bundles.
        var ui = corrections["ui"] ?? .object([:])
        var preferences = ui["preferences"] ?? .object([:])
        preferences["chapterOptions"] = try draft.chapterOptions.map {
            try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode($0))
        }
        ui["preferences"] = preferences; corrections["ui"] = ui
        corrections["updatedAt"] = .string(ISO8601DateFormatter().string(from: Date(timeIntervalSince1970: Double(draft.updatedAtMs) / 1_000)))
        corrections["beforePaddingSeconds"] = .number(Double(draft.beforePaddingMs) / 1_000)
        corrections["afterPaddingSeconds"] = .number(Double(draft.afterPaddingMs) / 1_000)
        corrections["joinGapSeconds"] = .number(Double(draft.joinGapMs) / 1_000)
        corrections["selectedSuppressionPolicy"] = .string(draft.selectedSuppressionPolicy)
        corrections["recordedSuppressionPolicy"] = .string([
            "none": "none", "conservative": "zero-non-exempt-misses", "balanced": "aggressive-intermediate", "aggressive": "raw-connected"
        ][draft.selectedSuppressionPolicy] ?? "none")
        corrections["suppressionInitialBehavior"] = .string(draft.suppressionInitialBehavior)
        corrections["defaultSuppressionScope"] = .string("whole-rally")
        corrections["suppressionContractVersion"] = .string(draft.suppressionContractVersion)
        corrections["suppressionDecisionOverrides"] = dictionary(draft.suppressionDecisionOverrides)
        corrections["suppressionScopeOverrides"] = dictionary(draft.suppressionScopeOverrides)
        corrections["userTouchedCutIds"] = strings(draft.userTouchedCutIds.sorted())
        corrections["suppression"] = .object(["selectedPolicy": .string(draft.selectedSuppressionPolicy),
            "decisionOverrides": dictionary(draft.suppressionDecisionOverrides), "defaultSuppressionScope": .string("whole-rally"),
            "suppressionScopeOverrides": dictionary(draft.suppressionScopeOverrides), "userTouchedCutIds": strings(draft.userTouchedCutIds.sorted()),
            "decisions": .array(suggestions.map { suggestion in
                let active = draft.selectedSuppressionPolicy != "none" && suggestion.eligiblePolicyIds.contains(draft.selectedSuppressionPolicy)
                let decision = EditorMath.effectiveDecision(draft, suggestion: suggestion)
                let touched = draft.cuts.contains { $0.origin == .inferred && draft.userTouchedCutIds.contains($0.id) && $0.coreStartMs < suggestion.endMs && suggestion.startMs < $0.coreEndMs }
                let state = !active ? "dormant" : decision == "suppress" ? "suppressed" :
                    draft.suppressionDecisionOverrides[suggestion.logicalId] == "keep" ? "kept" : touched ? "edited-kept" : "kept"
                return .object(["suggestionId": .string(suggestion.id), "logicalId": .string(suggestion.logicalId),
                    "state": .string(state),
                    "scope": .string(draft.suppressionScopeOverrides[suggestion.logicalId] ?? "whole-rally")])
            })])
        corrections["correctedRanges"] = .array(draft.cuts.map { cut in
            var value: JSONValue = .object(["id": .string(cut.id), "coreStart": .number(Double(cut.coreStartMs) / 1_000),
                "coreEnd": .number(Double(cut.coreEndMs) / 1_000), "keepStart": .number(Double(cut.keepStartMs) / 1_000),
                "keepEnd": .number(Double(cut.keepEndMs) / 1_000), "confidence": .number(Double(cut.confidence)),
                "included": .bool(cut.included), "origin": .string(cut.origin.rawValue)])
            if let agreement = cut.agreement { value["agreement"] = .string(agreement) }; return value
        })
        corrections["ignoredIntervals"] = .array(draft.ignoredIntervals.map { .object(["id": .string($0.id),
            "start": .number(Double($0.startMs) / 1_000), "end": .number(Double($0.endMs) / 1_000), "reason": .string($0.reason)]) })
        corrections["scoreTracking"] = .object(["state": try score.jsonValue(), "excludedRallyIds": strings(excluded), "derivedFinalScore": derivedScore ?? computedScore])
        corrections["labels"] = .object([
            "falsePositives": .array(draft.cuts.filter { $0.origin == .inferred && !$0.included }.map(wireRange)),
            "falseNegatives": .array(draft.cuts.filter { $0.origin == .manual && $0.included }.map(wireRange)),
            "confirmedModelRanges": .array(draft.cuts.filter { $0.origin == .inferred && $0.included }.map(wireRange)),
            "discardedManualRanges": .array(draft.cuts.filter { $0.origin == .manual && !$0.included }.map(wireRange))])
        bundle["corrections"] = corrections
        bundle["finalExportIntervals"] = .array(intervals.map { .object(["start": .number(Double($0.startMs) / 1_000),
            "end": .number(Double($0.endMs) / 1_000), "cutIds": strings($0.cutIds),
            "joinedGaps": .array($0.joinedGaps.map { .object(["start": .number(Double($0.startMs) / 1_000), "end": .number(Double($0.endMs) / 1_000)]) })]) })
        // Rebuilt from final intervals so removed or ignored time cannot leak into the provenance.
        bundle["finalExportProvenance"] = .array(intervals.flatMap { interval in
            draft.cuts.filter { interval.cutIds.contains($0.id) }.flatMap { cut -> [JSONValue] in
                let pieces: [(Int64, Int64, String)] = cut.origin == .manual ? [(cut.keepStartMs, cut.keepEndMs, "manual")] : [
                    (cut.keepStartMs, cut.coreStartMs, "padding"), (cut.coreStartMs, cut.coreEndMs, "inferred-core"), (cut.coreEndMs, cut.keepEndMs, "padding")]
                return pieces.compactMap { a, b, kind in
                    let start = max(interval.startMs, a); let end = min(interval.endMs, b)
                    guard end > start else { return nil }
                    return .object(["start": .number(Double(start) / 1_000), "end": .number(Double(end) / 1_000),
                        "kind": .string(kind), "cutIds": strings([cut.id]), "suggestionIds": .array([])])
                }
            } + interval.joinedGaps.map { .object(["start": .number(Double($0.startMs) / 1_000), "end": .number(Double($0.endMs) / 1_000),
                "kind": .string("joined-gap"), "cutIds": strings(interval.cutIds), "suggestionIds": .array([])]) }
        })
        var provenance = bundle["finalExportProvenance"]?.array ?? []
        for suggestion in suggestions where draft.selectedSuppressionPolicy != "none" && suggestion.eligiblePolicyIds.contains(draft.selectedSuppressionPolicy) && EditorMath.effectiveDecision(draft, suggestion: suggestion) == "suppress" {
            let scope = draft.suppressionScopeOverrides[suggestion.logicalId] ?? "whole-rally"
            if scope == "whole-rally" {
                for cut in draft.cuts where cut.origin == .inferred && cut.coreStartMs < suggestion.endMs && suggestion.startMs < cut.coreEndMs {
                    provenance.append(.object(["start": .number(Double(cut.keepStartMs) / 1_000), "end": .number(Double(cut.keepEndMs) / 1_000),
                        "kind": .string("suppression-whole-rally"), "cutIds": strings([cut.id]), "suggestionIds": strings([suggestion.logicalId])]))
                }
            } else {
                provenance.append(.object(["start": .number(Double(suggestion.startMs) / 1_000), "end": .number(Double(suggestion.endMs) / 1_000),
                    "kind": .string("suppression-veto-region"), "cutIds": .array([]), "suggestionIds": strings([suggestion.logicalId])]))
            }
        }
        bundle["finalExportProvenance"] = .array(provenance)
    }
    public static func importData(_ data: Data, expectedFeatureNames: [String]? = nil) throws -> ProjectDocument {
        try importBundle(JSONDecoder().decode(JSONValue.self, from: data), expectedFeatureNames: expectedFeatureNames)
    }
    public static func importBundle(_ bundle: JSONValue, expectedFeatureNames: [String]? = nil) throws -> ProjectDocument {
        guard bundle["schema"]?.string == "volleycut-model-feedback", bundle["schemaVersion"]?.double == 3 else { throw ProjectError.invalid("Choose a VolleySplice model-feedback schema v3 file") }
        let source = try required(bundle["source"], "source")
        guard source["timelineCoordinates"]?.string == "seconds-from-start-of-source", source["videoBytesIncluded"]?.bool == false else { throw ProjectError.invalid("Feedback must use the original source timeline without embedded video") }
        let duration = try number(source["media"]?["duration"], "duration")
        let start = try number(source["gameWindow"]?["start"], "game start")
        let end = try number(source["gameWindow"]?["end"], "game end")
        guard duration > 0, start >= 0, end > start, end <= duration,
              try number(source["media"]?["width"], "width") > 0, try number(source["media"]?["height"], "height") > 0,
              try number(source["file"]?["sizeBytes"], "size") >= 0, try number(source["file"]?["lastModifiedMs"], "modified") >= 0 else { throw ProjectError.invalid("Invalid source metadata or game window") }
        if let fingerprint = source["file"]?["sampledFingerprint"]?.string {
            guard fingerprint.range(of: "^sampled-sha256-v1:[0-9a-f]{64}$", options: .regularExpression) != nil else { throw ProjectError.invalid("Invalid sampled source fingerprint") }
        }
        let roi = try ["x", "y", "width", "height"].map { try number(source["featureRoi"]?[$0], "ROI \($0)") }
        guard roi[0] >= 0, roi[1] >= 0, roi[2] > 0, roi[3] > 0, roi[0] + roi[2] <= 1.000001, roi[1] + roi[3] <= 1.000001 else { throw ProjectError.invalid("Invalid source ROI") }
        let inference = try required(bundle["initialInference"], "initialInference")
        try validateNumericPayloads(inference)
        guard inference["modelId"]?.string == modelId, inference["ensembleAlgorithmVersion"]?.string == ensembleAlgorithmVersion,
              let components = inference["components"]?.array, components.count == 2,
              (0..<2).allSatisfy({ components[$0]["modelId"]?.string == componentIds[$0] && components[$0]["bundleSha256"]?.string == componentHashes[$0] }) else { throw ProjectError.invalid("Feedback uses a different production ensemble") }
        guard let initialRanges = inference["ranges"]?.array else { throw ProjectError.invalid("Missing initial inference ranges") }
        for range in initialRanges {
            let a = try number(range["start"], "range start"); let b = try number(range["end"], "range end")
            let c = try number(range["confidence"], "range confidence")
            let agreement = range["agreement"]?.string
            guard a >= 0, b > a, b <= duration, (0...1).contains(c),
                  agreement == nil || ["both-models", "all-labels-v2-only", "previous-production-only"].contains(agreement!) else { throw ProjectError.invalid("Invalid initial inference range") }
        }
        _ = try suppressionRegions(bundle)
        _ = try retainedAnalysis(bundle, expectedFeatureNames: expectedFeatureNames)
        let corrections = try required(bundle["corrections"], "corrections")
        guard let cuts = corrections["correctedRanges"]?.array else { throw ProjectError.invalid("Missing corrected ranges") }
        var draft = EditorDraft(sourceRevision: source["analysisId"]?.string ?? UUID().uuidString)
        if let options = corrections["ui"]?["preferences"]?["chapterOptions"], options != .null {
            draft.chapterOptions = try JSONDecoder().decode(YouTubeChapterOptions.self, from: JSONEncoder().encode(options))
        }
        draft.cuts = try cuts.map { cut in
            guard let id = cut["id"]?.string, let origin = cut["origin"]?.string.flatMap(CutOrigin.init(rawValue:)), let included = cut["included"]?.bool else { throw ProjectError.invalid("Invalid corrected range identity or origin") }
            return EditableCut(id: id, coreStartMs: try ms(cut["coreStart"], "core start"), coreEndMs: try ms(cut["coreEnd"], "core end"),
                               keepStartMs: try ms(cut["keepStart"], "keep start"), keepEndMs: try ms(cut["keepEnd"], "keep end"),
                               confidence: Float(try number(cut["confidence"], "confidence")), included: included, origin: origin, agreement: cut["agreement"]?.string)
        }
        draft.beforePaddingMs = try ms(corrections["beforePaddingSeconds"] ?? .number(2), "before padding")
        draft.afterPaddingMs = try ms(corrections["afterPaddingSeconds"] ?? .number(2), "after padding")
        draft.joinGapMs = try ms(corrections["joinGapSeconds"] ?? .number(3), "join gap")
        draft.ignoredIntervals = try (corrections["ignoredIntervals"]?.array ?? []).map { value in
            guard let id = value["id"]?.string, let reason = value["reason"]?.string else { throw ProjectError.invalid("Invalid ignored interval") }
            return .init(id: id, startMs: try ms(value["start"], "ignored start"), endMs: try ms(value["end"], "ignored end"), reason: reason)
        }
        let suppression = corrections["suppression"]
        draft.selectedSuppressionPolicy = suppression?["selectedPolicy"]?.string ?? "none"
        draft.suppressionInitialBehavior = corrections["suppressionInitialBehavior"]?.string ?? "disable-initially"
        draft.suppressionDecisionOverrides = try stringDictionary(suppression?["decisionOverrides"])
        draft.suppressionScopeOverrides = try stringDictionary(suppression?["suppressionScopeOverrides"])
        draft.userTouchedCutIds = Set(suppression?["userTouchedCutIds"]?.array?.compactMap(\.string) ?? [])
        draft.scoreTracking = try required(corrections["scoreTracking"]?["state"], "score tracking state")
        draft.scoreTracking = try ScoreTracking.fromJSON(draft.scoreTracking, durationMs: ms(.number(duration), "duration")).jsonValue()
        if let updated = corrections["updatedAt"]?.string, let date = ISO8601DateFormatter().date(from: updated) { draft.updatedAtMs = Int64((date.timeIntervalSince1970 * 1_000).rounded()) }
        draft = EditorMath.alignRallyServeMarkers(draft)
        let document = ProjectDocument(sourceName: source["file"]?["name"]?.string ?? "imported-recording.mp4",
            durationMs: try ms(.number(duration), "duration"), gameWindow: .init(startMs: try ms(.number(start), "start"), endMs: try ms(.number(end), "end")),
            draft: draft, feedback: bundle)
        try document.validate(); return document
    }
    public static func retainedAnalysis(_ bundle: JSONValue, expectedFeatureNames: [String]? = nil) throws -> FeedbackAnalysis? {
        let inference = try required(bundle["initialInference"], "initial inference")
        let encodedTimes = try required(inference["timestamps"], "inference timestamps")
        guard encodedTimes["shape"]?.array?.count == 1 else { throw ProjectError.invalid("Inference timestamps must have one dimension") }
        let timestamps = try FeedbackNumericArray.decode(encodedTimes, type: "float64")
        let duration = try number(bundle["source"]?["media"]?["duration"], "duration")
        guard timestamps.allSatisfy({ $0 >= 0 && $0 <= duration }), zip(timestamps, timestamps.dropFirst()).allSatisfy({ $0 < $1 }) else { throw ProjectError.invalid("Invalid inference timeline") }
        let probabilities = try ["rally", "serve", "deadState"].map { name -> [Float] in
            let values = try FeedbackNumericArray.decode(required(inference["probabilities"]?[name], name), shape: [timestamps.count], type: "float32")
            guard values.allSatisfy({ (0...1).contains($0) }) else { throw ProjectError.invalid("Invalid probability trace") }
            return values.map(Float.init)
        }
        guard let features = bundle["features"], features != .null else { return nil }
        let rows = try number(features["rows"], "rows"); let columns = try number(features["columns"], "columns")
        guard rows == Double(timestamps.count), rows > 0, columns > 0, columns <= 10_000, columns.rounded() == columns,
              features["analysisFps"]?.double == 4, let namesJSON = features["names"]?.array else { throw ProjectError.invalid("Invalid feature schema") }
        let names = namesJSON.compactMap(\.string)
        guard names.count == Int(columns), expectedFeatureNames == nil || names == expectedFeatureNames else { throw ProjectError.invalid("Feature names differ from native schema") }
        let featureTimes = try FeedbackNumericArray.decode(required(features["timestamps"], "feature timestamps"), shape: [timestamps.count], type: "float64")
        let start = try number(bundle["source"]?["gameWindow"]?["start"], "game start")
        let end = try number(bundle["source"]?["gameWindow"]?["end"], "game end")
        let firstIndex = max(0, ceil(start * 4 - 1e-9))
        guard featureTimes == timestamps, timestamps.enumerated().allSatisfy({ abs($0.element - ((firstIndex + Double($0.offset)) / 4)) <= 1e-9 && $0.element < end - 1e-9 }) else { throw ProjectError.invalid("Feature timestamps do not match source analysis window") }
        let values = try FeedbackNumericArray.decode(required(features["values"], "feature values"), shape: [timestamps.count, Int(columns)], type: "float32").map(Float.init)
        return FeedbackAnalysis(timestamps: timestamps, baseFeatures: values, featureNames: names,
            rallyProbabilities: probabilities[0], serveProbabilities: probabilities[1], deadStateProbabilities: probabilities[2])
    }
    #if canImport(CryptoKit)
    public static func verifySource(url: URL, source: JSONValue) throws {
        let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize
        guard let size, Double(size) == source["file"]?["sizeBytes"]?.double else { throw ProjectError.invalid("Selected video size does not match this project") }
        if let expected = source["file"]?["sampledFingerprint"]?.string {
            guard try sampledFingerprint(url: url) == expected else { throw ProjectError.invalid("Selected video fingerprint does not match this project") }
        } else {
            guard url.lastPathComponent == source["file"]?["name"]?.string else { throw ProjectError.invalid("Select the original source filename; this project has no fingerprint") }
            let modified = try url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate
            if let expected = source["file"]?["lastModifiedMs"]?.double, expected > 0 {
                guard let modified, abs(modified.timeIntervalSince1970 * 1_000 - expected) < 1_000 else { throw ProjectError.invalid("Source modification date does not match; a fingerprint is needed to reconnect a renamed or modified copy") }
            }
        }
    }
    public static func sampledFingerprint(url: URL) throws -> String {
        let file = try FileHandle(forReadingFrom: url); defer { try? file.close() }
        let size = try file.seekToEnd(); try file.seek(toOffset: 0)
        var digest = SHA256(); var littleSize = size.littleEndian
        withUnsafeBytes(of: &littleSize) { digest.update(data: Data($0)) }
        func consume(_ length: Int) throws {
            var remaining = length
            while remaining > 0 {
                let bytes = try file.read(upToCount: min(65_536, remaining)) ?? Data()
                guard !bytes.isEmpty else { throw ProjectError.invalid("Source ended before its declared size") }
                digest.update(data: bytes); remaining -= bytes.count
            }
        }
        if size <= 2_097_152 { try consume(Int(size)) }
        else { try consume(1_048_576); try file.seek(toOffset: size - 1_048_576); try consume(1_048_576) }
        return "sampled-sha256-v1:" + digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
    #endif
}
