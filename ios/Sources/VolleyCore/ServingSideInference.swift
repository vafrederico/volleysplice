import CryptoKit
import Foundation

public struct ProductionServeOutput: Codable, Equatable, Sendable {
    public var modelId: String, times: [Double], probabilities: [Float], detections: [Serve]
    public init(modelId: String, times: [Double], probabilities: [Float], detections: [Serve] = []) {
        self.modelId = modelId; self.times = times; self.probabilities = probabilities; self.detections = detections
    }
}
public struct ServingSideHeadEvidence: Codable, Equatable, Sendable {
    public let modelId: String, threshold: Double, peakProbability: Double, peakTime: Double
    public let crossesThreshold: Bool, nearestDetection: Serve?
    public var json: JSONValue {
        let detection: JSONValue = nearestDetection.map { .object(["time": .number($0.time), "confidence": .number(Double($0.confidence))]) } ?? .null
        return .object(["modelId": .string(modelId), "threshold": .number(threshold), "peakProbability": .number(peakProbability),
                        "peakTime": .number(peakTime), "crossesThreshold": .bool(crossesThreshold), "nearestDetection": detection])
    }
}

/// Pure SERVSIDE237-FLIGHT inference. Features remain Float64 from extraction/import through
/// recording-wide tied ranks and logistic accumulation; production serve heads remain Float32.
public struct ServingSideInference: Sendable {
    public static let modelId = "serving-side-fixed-flight-v3"
    public static let fingerprint = "85bc3325fbd43abba6ba3726ac091dc6a3a1eafc68d63d979cb2a7450eb49e06"
    public static let runtimeAsset = "serving-side-85bc3325fbd4.json"
    public static let runtimeSha256 = "14f18bf0b0f326ccd7ef4b3d614a96a53dd9675df61813fd375677489d0e5a7c"
    public static let featureVersion = "SERVSIDE237-FLIGHT"
    public static let anchorContract = "merged-production-interval-start-v1"
    public static let columns = 237
    public static let sideThreshold = 0.4783744762021848
    public static let farReviewThreshold = 0.3121748736511044
    public static let nearReviewThreshold = 0.5028396703865513
    public static let serveHeadThreshold = 0.85
    public static let serveHeadWindowSeconds = 1.0
    public static let courtFlowOffsets = [-1.25, -0.75, -0.35, -0.10, 0.10, 0.30, 0.55, 0.85]
    public static let flightOffsets = [-0.15, 0.05, 0.20, 0.35, 0.55, 0.80, 1.10, 1.40, 1.75]
    public static let courtStatistics = ["flowMean", "flowP90", "activeFraction", "largestComponentFraction", "componentCountDensity", "centroidX", "centroidY", "flowX", "flowY"]
    public static let flightGlobalStatistics = ["energyMean", "activeFraction", "centroidX", "centroidY", "spreadX", "spreadY", "entropy", "largestComponentFraction", "flowX", "flowY", "divergence", "bottomMinusTop", "smallComponentEnergyFraction", "smallComponentCentroidY", "smallComponentFlowY"]
    public static let flightTrajectoryStatistics = ["centroidY", "spreadY", "entropy", "flowY", "divergence", "bottomMinusTop", "smallComponentEnergyFraction", "smallComponentCentroidY", "smallComponentFlowY"]
    public static let courtFeatureNames: [String] = {
        var result: [String] = []
        for phase in ["pre", "contact", "post"] {
            for zone in ["near", "far"] { for statistic in courtStatistics { result.append("v2:\(phase):\(zone):\(statistic)") } }
            for statistic in courtStatistics.prefix(5) { result.append("v2:\(phase):nearMinusFar:\(statistic)") }
        }
        for zone in ["near", "far"] {
            for statistic in ["flowMean", "activeFraction", "largestComponentFraction"] { result.append("v2:contactMinusPre:\(zone):\(statistic)") }
            for statistic in ["flowMean", "activeFraction"] { result.append("v2:postMinusContact:\(zone):\(statistic)") }
        }
        for statistic in ["flowMean", "activeFraction", "largestComponentFraction"] { result.append("v2:contactDelta:nearMinusFar:\(statistic)") }
        return result
    }()
    public static let flightFeatureNames: [String] = {
        var result: [String] = []
        for phase in ["launch", "early", "late"] {
            for row in 0..<4 { for column in 0..<6 { result.append("flight:\(phase):grid:r\(row):c\(column):energy") } }
            for row in 0..<4 { result.append("flight:\(phase):row:r\(row):flowY") }
            for statistic in flightGlobalStatistics { result.append("flight:\(phase):\(statistic)") }
        }
        for transition in ["launchToEarly", "earlyToLate"] {
            for statistic in flightTrajectoryStatistics { result.append("flight:trajectory:\(transition):\(statistic)") }
            for row in 0..<4 { result.append("flight:trajectory:\(transition):row:r\(row):energy") }
        }
        return result
    }()
    public static let featureNames = courtFeatureNames + flightFeatureNames

    public struct CandidateInterval: Codable, Equatable, Sendable {
        public let id: String, start: Double, end: Double, agreement: String?
        public init(id: String, start: Double, end: Double, agreement: String? = nil) {
            self.id = id; self.start = start; self.end = end; self.agreement = agreement
        }
    }
    public struct FramePlan: Sendable { public let candidates: [CandidateInterval], requestedTimes: [Double] }
    private struct Model: Decodable, Sendable {
        let family: String, impute: [Double], mean: [Double], scale: [Double], weights: [Double]
        let bias: Double, l2: Double, threshold: Double
    }
    private struct Payload: Decodable {
        struct Resize: Decodable { let width: Int, height: Int }
        struct Grid: Decodable { let rows: Int, columns: Int }
        struct Review: Decodable { let farUpperExclusive: Double, nearLowerInclusive: Double }
        struct Gate: Decodable {
            let serveHeadThreshold: Double, serveHeadWindowSeconds: Double
            let rallyRecoveryAgreement: String, rallyRecoveryRequiresReview: Bool
        }
        let schemaVersion: Int, kind: String, modelId: String, fingerprint: String, featureVersion: String
        let courtFlowOffsetsSeconds: [Double], flightOffsetsSeconds: [Double], featureNames: [String]
        let resize: Resize, flightGrid: Grid, model: Model, sideThreshold: Double, reviewBand: Review, gate: Gate
    }
    private let model: Model
    public init(data: Data) throws {
        guard let text = String(data: data, encoding: .utf8) else { throw AnalysisError.invalid("Invalid serving-side asset encoding") }
        let canonical = Data(text.replacingOccurrences(of: "\r\n", with: "\n").utf8)
        let hash = SHA256.hash(data: canonical).map { String(format: "%02x", $0) }.joined()
        guard hash == Self.runtimeSha256 else { throw AnalysisError.invalid("Frozen serving-side asset checksum mismatch") }
        let payload = try JSONDecoder().decode(Payload.self, from: canonical)
        guard payload.schemaVersion == 1, payload.kind == "volleycut-serving-side-fixed-flight-runtime-v1",
              payload.modelId == Self.modelId, payload.fingerprint == Self.fingerprint, payload.featureVersion == Self.featureVersion,
              payload.courtFlowOffsetsSeconds == Self.courtFlowOffsets, payload.flightOffsetsSeconds == Self.flightOffsets,
              payload.featureNames == Self.featureNames, payload.resize.width == 192, payload.resize.height == 108,
              payload.flightGrid.rows == 4, payload.flightGrid.columns == 6, payload.model.family == "class-balanced-logistic",
              payload.model.threshold == Self.sideThreshold, payload.sideThreshold == Self.sideThreshold,
              payload.reviewBand.farUpperExclusive == Self.farReviewThreshold, payload.reviewBand.nearLowerInclusive == Self.nearReviewThreshold,
              payload.gate.serveHeadThreshold == Self.serveHeadThreshold, payload.gate.serveHeadWindowSeconds == Self.serveHeadWindowSeconds,
              payload.gate.rallyRecoveryAgreement == ProductionEnsemble.bothModels, payload.gate.rallyRecoveryRequiresReview else {
            throw AnalysisError.invalid("Serving-side geometry, feature signature or gate changed")
        }
        for vector in [payload.model.impute, payload.model.mean, payload.model.scale, payload.model.weights] {
            guard vector.count == Self.columns, vector.allSatisfy(\.isFinite) else { throw AnalysisError.invalid("Invalid serving-side model vector") }
        }
        guard payload.model.scale.allSatisfy({ $0 > 0 }), payload.model.bias.isFinite, payload.model.l2.isFinite else {
            throw AnalysisError.invalid("Invalid serving-side scale/bias")
        }
        model = payload.model
    }

    public static func candidates(ranges: [Interval]) throws -> [CandidateInterval] {
        var result: [CandidateInterval] = []
        for (index, range) in ranges.enumerated() {
            guard range.start.isFinite, range.end.isFinite, abs(range.start) < Double(Int64.max / 2000),
                  abs(range.end) < Double(Int64.max / 2000) else { throw AnalysisError.invalid("Invalid serving-side anchor") }
            let start = floor(range.start * 1000 + 0.5) / 1000, end = floor(range.end * 1000 + 0.5) / 1000
            if start < 0 || end <= start { continue }
            result.append(.init(id: String(format: "R%03d", index + 1), start: start, end: end, agreement: range.agreement))
        }
        return result.sorted { ($0.start, $0.end, $0.id) < ($1.start, $1.end, $1.id) }
    }
    public static func clampedTimestamp(anchor: Double, offset: Double, duration: Double) -> Double {
        min(max(anchor + offset, 0), max(duration - 0.01, 0))
    }
    public static func requestedTimestamps(candidates: [CandidateInterval], duration: Double) throws -> [Double] {
        guard duration.isFinite, duration >= 0 else { throw AnalysisError.invalid("Invalid serving-side duration") }
        try validateCandidates(candidates)
        var times: Set<Double> = []
        for candidate in candidates { for offset in courtFlowOffsets + flightOffsets {
            times.insert(clampedTimestamp(anchor: candidate.start, offset: offset, duration: duration))
        } }
        return times.sorted()
    }
    public static func framePlan(ranges: [Interval], duration: Double) throws -> FramePlan {
        let candidates = try candidates(ranges: ranges)
        return FramePlan(candidates: candidates, requestedTimes: try requestedTimestamps(candidates: candidates, duration: duration))
    }
    /// ScoreSpecialistInference's shared decode schedule can also include optional side-switch timestamps.
    public static func sharedRequestedTimestamps(serving: [Double], sideSwitch: [Double] = []) throws -> [Double] {
        guard (serving + sideSwitch).allSatisfy({ $0.isFinite && $0 >= 0 }) else { throw AnalysisError.invalid("Invalid specialist timestamp") }
        return Array(Set(serving + sideSwitch)).sorted()
    }
    private static func validateCandidates(_ candidates: [CandidateInterval]) throws {
        guard Set(candidates.map(\.id)).count == candidates.count, candidates.allSatisfy({ !$0.id.isEmpty &&
            $0.start.isFinite && $0.start >= 0 && $0.end.isFinite && $0.end > $0.start &&
            ($0.agreement == nil || ProductionEnsemble.isValidAgreement($0.agreement)) }) else {
            throw AnalysisError.invalid("Invalid serving-side candidate identity or interval")
        }
    }
    public static func tiedPercentileRanks(_ values: [Double], rows: Int, columns: Int) throws -> [Double] {
        guard rows >= 0, columns >= 0, (columns == 0 ? values.isEmpty : values.count / columns == rows && values.count % columns == 0),
              values.allSatisfy(\.isFinite) else { throw AnalysisError.invalid("Invalid Float64 serving-side feature matrix") }
        if rows == 0 { return [] }; if rows == 1 { return [Double](repeating: 0.5, count: columns) }
        var result = [Double](repeating: 0, count: values.count)
        for column in 0..<columns {
            let order = (0..<rows).sorted { left, right in
                let a = values[left * columns + column], b = values[right * columns + column]
                return a == b ? left < right : a < b
            }
            var start = 0
            while start < rows {
                var end = start + 1
                while end < rows && values[order[end] * columns + column] == values[order[start] * columns + column] { end += 1 }
                let rank = (Double(start + end - 1) / 2) / Double(rows - 1)
                for position in start..<end { result[order[position] * columns + column] = rank }
                start = end
            }
        }
        return result
    }
    public func nearProbability(ranked: [Double]) throws -> Double {
        guard ranked.count == Self.columns else { throw AnalysisError.invalid("Serving-side ranked width mismatch") }
        var logit = model.bias
        for i in ranked.indices {
            let filled = ranked[i].isFinite ? ranked[i] : model.impute[i]
            logit += (filled - model.mean[i]) / model.scale[i] * model.weights[i]
        }
        return 1 / (1 + exp(-max(-30, min(30, logit))))
    }
    public static func serveHeadEvidence(_ output: ProductionServeOutput, anchor: Double,
                                          threshold: Double = serveHeadThreshold, windowSeconds: Double = serveHeadWindowSeconds) throws -> ServingSideHeadEvidence {
        guard anchor.isFinite, anchor >= 0, threshold.isFinite, (0...1).contains(threshold), windowSeconds.isFinite, windowSeconds >= 0,
              !output.times.isEmpty, output.times.count == output.probabilities.count,
              output.times.allSatisfy({ $0.isFinite && $0 >= 0 }), output.probabilities.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 }),
              output.detections.allSatisfy({ $0.time.isFinite && $0.time >= 0 && $0.confidence.isFinite && $0.confidence >= 0 && $0.confidence <= 1 }) else {
            throw AnalysisError.invalid("Serving-side serve evidence is not aligned")
        }
        var nearest = 0, nearestDistance = Double.infinity, selected: [Int] = []
        for i in output.times.indices {
            let distance = abs(output.times[i] - anchor)
            if distance < nearestDistance { nearestDistance = distance; nearest = i }
            if distance <= windowSeconds + 1e-9 { selected.append(i) }
        }
        if selected.isEmpty { selected.append(nearest) }
        var peak = selected[0]
        for i in selected.dropFirst() where output.probabilities[i] > output.probabilities[peak] { peak = i }
        var detection: Serve?
        for candidate in output.detections {
            if detection == nil || abs(candidate.time - anchor) < abs(detection!.time - anchor) { detection = candidate }
        }
        return ServingSideHeadEvidence(modelId: output.modelId, threshold: threshold, peakProbability: Double(output.probabilities[peak]),
            peakTime: output.times[peak], crossesThreshold: Double(output.probabilities[peak]) >= threshold, nearestDetection: detection)
    }
    public static func decision(candidate: CandidateInterval, nearProbability: Double,
                                 allLabelsV2Evidence: ServingSideHeadEvidence, previousProductionEvidence: ServingSideHeadEvidence) throws -> ServingSideCandidate {
        try validateCandidates([candidate])
        guard nearProbability.isFinite, (0...1).contains(nearProbability) else { throw AnalysisError.invalid("Invalid near-side probability") }
        let side: ServingSide = nearProbability >= sideThreshold ? .near : .far
        var reasons: [ServingSideReviewReason] = []
        let source: ServingSideDecisionSource
        if allLabelsV2Evidence.crossesThreshold || previousProductionEvidence.crossesThreshold { source = .serveHead }
        else if candidate.agreement == ProductionEnsemble.bothModels { source = .productionRallyRecovery; reasons.append(.productionRallyRecovery) }
        else { source = .none }
        if nearProbability >= farReviewThreshold && nearProbability < nearReviewThreshold { reasons.insert(.sideScore, at: 0) }
        let verdict: ServingSideVerdict = source == .none ? .notServe : !reasons.isEmpty ? .review : side == .near ? .near : .far
        return ServingSideCandidate(id: candidate.id, anchor: candidate.start, intervalEnd: candidate.end, agreement: candidate.agreement,
            nearProbability: nearProbability, side: side, verdict: verdict, serveDecisionSource: source, reviewReasons: reasons,
            allLabelsV2Evidence: allLabelsV2Evidence.json, previousProductionEvidence: previousProductionEvidence.json)
    }
    public func evaluate(candidates: [CandidateInterval], rawFeatures: [Double], allLabelsV2: ProductionServeOutput,
                          previousProduction: ProductionServeOutput) throws -> ServingSideOutput {
        try Self.validateCandidates(candidates)
        let ranked = try Self.tiedPercentileRanks(rawFeatures, rows: candidates.count, columns: Self.columns)
        if candidates.isEmpty { return ServingSideOutput(candidates: []) }
        guard allLabelsV2.modelId == "model-1ca43e38eefc", previousProduction.modelId == "model-9c92b8e9333f",
              allLabelsV2.times == previousProduction.times,
              zip(allLabelsV2.times, allLabelsV2.times.dropFirst()).allSatisfy({ $0 < $1 }) else {
            throw AnalysisError.invalid("Serving-side requires aligned outputs from both production model IDs")
        }
        var verdicts: [ServingSideCandidate] = []
        for (row, candidate) in candidates.enumerated() {
            try Task.checkCancellation()
            let allEvidence = try Self.serveHeadEvidence(allLabelsV2, anchor: candidate.start)
            let previousEvidence = try Self.serveHeadEvidence(previousProduction, anchor: candidate.start)
            let probability = try nearProbability(ranked: Array(ranked[(row * Self.columns)..<((row + 1) * Self.columns)]))
            verdicts.append(try Self.decision(candidate: candidate, nearProbability: probability,
                allLabelsV2Evidence: allEvidence, previousProductionEvidence: previousEvidence))
        }
        return ServingSideOutput(candidates: verdicts, rawFeatures: rawFeatures)
    }
    public func evaluate(candidates: [CandidateInterval], rawFloat64LE: Data, allLabelsV2: ProductionServeOutput,
                          previousProduction: ProductionServeOutput) throws -> ServingSideOutput {
        guard rawFloat64LE.count % 8 == 0, rawFloat64LE.count / 8 == candidates.count * Self.columns else {
            throw AnalysisError.invalid("Serving-side Float64 byte matrix shape mismatch")
        }
        let values: [Double] = rawFloat64LE.withUnsafeBytes { bytes in
            (0..<(bytes.count / 8)).map { Double(bitPattern: UInt64(littleEndian: bytes.loadUnaligned(fromByteOffset: $0 * 8, as: UInt64.self))) }
        }
        return try evaluate(candidates: candidates, rawFeatures: values, allLabelsV2: allLabelsV2, previousProduction: previousProduction)
    }
    public static func isReusable(_ output: ServingSideOutput, ranges: [Interval]) -> Bool {
        guard output.modelId == modelId, output.modelFingerprint == fingerprint, output.featureVersion == featureVersion,
              output.anchorContract == anchorContract, output.columns == columns, output.rows == output.candidates.count,
              output.rawFeatures.count == output.rows * columns, output.rawFeatures.allSatisfy(\.isFinite),
              let expected = try? candidates(ranges: ranges), expected.count == output.candidates.count else { return false }
        return zip(expected, output.candidates).allSatisfy { left, right in
            left.id == right.id && left.start == right.anchor && left.start == right.intervalStart && left.end == right.intervalEnd && left.agreement == right.agreement
        }
    }
}
