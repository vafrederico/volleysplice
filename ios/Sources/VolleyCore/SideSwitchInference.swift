import CryptoKit
import Foundation

public struct ProductionStateOutput: Codable, Equatable, Sendable {
    public var modelId: String, times: [Double], rallyProbabilities: [Float], deadStateProbabilities: [Float]
    public init(modelId: String, times: [Double], rallyProbabilities: [Float], deadStateProbabilities: [Float]) {
        self.modelId = modelId; self.times = times; self.rallyProbabilities = rallyProbabilities; self.deadStateProbabilities = deadStateProbabilities
    }
}
public struct SideSwitchAnalysisInput: Sendable {
    public let intervals: [Interval], allLabelsV2Components: [Interval], previousProductionComponents: [Interval]
    public let allLabelsV2State: ProductionStateOutput, previousProductionState: ProductionStateOutput
    public init(intervals: [Interval], allLabelsV2Components: [Interval], previousProductionComponents: [Interval],
                allLabelsV2State: ProductionStateOutput, previousProductionState: ProductionStateOutput) {
        self.intervals = intervals; self.allLabelsV2Components = allLabelsV2Components; self.previousProductionComponents = previousProductionComponents
        self.allLabelsV2State = allLabelsV2State; self.previousProductionState = previousProductionState
    }
}
public struct SideSwitchCandidateProposal: Codable, Equatable, Sendable {
    public var id: String, kind: SideSwitchCandidateKind, gapStart: Double, gapEnd: Double, transitionTime: Double, generatorScore: Double
    public var sourceRangeIds: [String], beforeStart: Double, beforeEnd: Double, afterStart: Double, afterEnd: Double
}
public struct SideSwitchInference: Sendable {
    public static let runtimeAsset = "side-switch-c2570481c30d.json"
    public static let runtimeSha256 = "ab4197545fb916a37ee6ac1d69e74ddfc0123c09039cdfa88c4ef378dd3e27fc"
    public static let featureNames = [
        "v4BroadSameAssignmentCost", "v4TightSameAssignmentCost", "v4MeanSwapMargin", "v4GlobalAppearanceChange",
        "v4MaximumCameraShift", "v4MinimumAlignmentResponse", "playerSameAssignmentCost", "playerSwappedAssignmentCost",
        "playerSwapMargin", "playerOrientationFlipEvidence", "minimumPlayerSideSeparation", "playerSideSeparationChange",
        "beforePlayerPaletteInstability", "afterPlayerPaletteInstability", "playerGlobalAppearanceChange", "minimumProposalCoverage",
        "proposalCoverageChange", "minimumProposalCount", "proposalCountChange", "minimumNearSupport", "minimumFarSupport",
        "sideSupportImbalanceChange", "productionBeforeSupportCount", "productionAfterSupportCount", "productionMinimumAdjacentSupportCount",
        "productionMinimumAdjacentRallyPeak", "productionGapLiveFraction", "productionGapMeanRallyScore", "productionGapPeakRallyScore",
        "productionGapMeanDeadStateScore", "productionGapPeakDeadStateScore", "productionGapDurationSeconds",
        "candidateIsInternalDeadStatePeak", "candidateGeneratorScore"
    ]
    public struct FramePlan: Codable, Equatable, Sendable {
        public let candidates: [SideSwitchCandidateProposal], requestedTimes: [Double], calibrationTimes: [Double]
    }
    private struct Classifier: Decodable, Sendable {
        let featureNames: [String], impute: [Double], mean: [Double], scale: [Double], weights: [Double], bias: Double, threshold: Double
    }
    private struct Generator: Decodable, Sendable {
        let internalPeakThreshold: Double, internalPeakMinimumSeparationSeconds: Double
        let internalPeakRangeEdgeExclusionSeconds: Double, internalPeakProposalHalfWidthSeconds: Double
    }
    private struct Policy: Decodable, Sendable {
        let minimumCandidateIndexSeparation: Int, minimumTimeSeparationSeconds: Double, freePredictionsPerRecording: Int
        let countPenaltyLogitPerExcessPrediction: Double
    }
    private struct Runtime: Decodable, Sendable {
        let schemaVersion: Int, modelId: String, fingerprint: String, featureVersion: String
        let classifier: Classifier, candidateGenerator: Generator, decoder: Policy
    }
    private let runtime: Runtime
    public init(data: Data) throws {
        guard let string = String(data: data, encoding: .utf8) else { throw AnalysisError.invalid("Invalid side-switch asset encoding") }
        let canonical = Data(string.replacingOccurrences(of: "\r\n", with: "\n").utf8)
        guard SHA256.hash(data: canonical).map({ String(format: "%02x", $0) }).joined() == Self.runtimeSha256 else {
            throw AnalysisError.invalid("Frozen side-switch asset checksum mismatch")
        }
        let value = try JSONDecoder().decode(Runtime.self, from: canonical)
        guard value.schemaVersion == 1, value.modelId == "side-switch-hard-negative-mining-v1/union34-top2-x2",
              value.fingerprint == "sha256:c2570481c30dec62f56ac3284cd4028763ffd8e72a9ad07e9908fec10df387e3",
              value.featureVersion == "SIDE-SWITCH-UNION34-V1", value.classifier.featureNames == Self.featureNames,
              value.classifier.scale.allSatisfy({ $0 > 0 }), value.classifier.bias.isFinite,
              (0...1).contains(value.classifier.threshold) else { throw AnalysisError.invalid("Invalid side-switch runtime contract") }
        for vector in [value.classifier.impute, value.classifier.mean, value.classifier.scale, value.classifier.weights] {
            guard vector.count == 34, vector.allSatisfy(\.isFinite) else { throw AnalysisError.invalid("Side-switch vector mismatch") }
        }
        runtime = value
    }
    private static func ranges(_ source: [Interval]) -> [(id: String, start: Double, end: Double)] {
        let valid = source.filter { $0.start.isFinite && $0.end.isFinite && $0.end > $0.start }.sorted { ($0.start, $0.end) < ($1.start, $1.end) }
        return valid.enumerated().map { (String(format: "R%03d", $0.offset + 1), $0.element.start, $0.element.end) }
    }
    public static func sampleTimes(start: Double, end: Double) -> [Double] {
        let duration = end - start
        var first = start + min(0.2, duration * 0.08), last = end - min(0.15, duration * 0.08)
        if last <= first { first = start + duration * 0.2; last = start + duration * 0.8 }
        return (0..<7).map { first + (last - first) * Double($0) / 6 }
    }
    public static func calibrationTimes(intervals: [Interval]) -> [Double] {
        ranges(intervals).prefix(7).flatMap { sampleTimes(start: $0.start, end: $0.end) }
    }
    public static func candidateSampleTimes(_ candidate: SideSwitchCandidateProposal) -> [Double] {
        sampleTimes(start: candidate.beforeStart, end: candidate.beforeEnd) + sampleTimes(start: candidate.afterStart, end: candidate.afterEnd)
    }
    public func framePlan(input: SideSwitchAnalysisInput, duration: Double) throws -> FramePlan {
        guard duration.isFinite, duration >= 0 else { throw AnalysisError.invalid("Invalid side-switch duration") }
        let candidates = try generateCandidates(input: input)
        if candidates.isEmpty { return FramePlan(candidates: [], requestedTimes: [], calibrationTimes: []) }
        let calibration = Self.calibrationTimes(intervals: input.intervals).map { min(max($0, 0), max(duration - 0.01, 0)) }
        let times = calibration + candidates.flatMap(Self.candidateSampleTimes)
        return FramePlan(candidates: candidates, requestedTimes: Array(Set(times.map { min(max($0, 0), max(duration - 0.01, 0)) })).sorted(), calibrationTimes: calibration)
    }
    public static func validateStateOutputs(_ input: SideSwitchAnalysisInput) throws {
        for state in [input.allLabelsV2State, input.previousProductionState] {
            guard !state.times.isEmpty, state.times.count == state.rallyProbabilities.count, state.times.count == state.deadStateProbabilities.count,
                  state.times.allSatisfy(\.isFinite), zip(state.times, state.times.dropFirst()).allSatisfy({ $0 < $1 }),
                  state.rallyProbabilities.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 }),
                  state.deadStateProbabilities.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 }) else {
                throw AnalysisError.invalid("Side-switch requires complete aligned production state traces")
            }
        }
        guard input.allLabelsV2State.times == input.previousProductionState.times else { throw AnalysisError.invalid("Production state timestamps differ") }
    }
    public func generateCandidates(input: SideSwitchAnalysisInput) throws -> [SideSwitchCandidateProposal] {
        try Self.validateStateOutputs(input)
        let ranges = Self.ranges(input.intervals), state = input.allLabelsV2State, config = runtime.candidateGenerator
        var candidates: [SideSwitchCandidateProposal] = []
        if ranges.count > 1 { for index in 0..<(ranges.count - 1) {
            let before = ranges[index], after = ranges[index + 1]
            guard after.start >= before.end else { throw AnalysisError.invalid("Side-switch production ranges overlap") }
            candidates.append(.init(id: "switch:boundary:\(before.id):\(after.id)", kind: .adjacentRallyBoundary,
                gapStart: before.end, gapEnd: after.start, transitionTime: (before.end + after.start) / 2, generatorScore: 0,
                sourceRangeIds: [before.id, after.id], beforeStart: before.start, beforeEnd: before.end, afterStart: after.start, afterEnd: after.end))
        } }
        for range in ranges {
            let eligible = state.times.indices.filter { i in
                state.times[i] >= range.start + config.internalPeakRangeEdgeExclusionSeconds &&
                state.times[i] <= range.end - config.internalPeakRangeEdgeExclusionSeconds && Double(state.deadStateProbabilities[i]) >= config.internalPeakThreshold
            }.sorted { a, b in
                if state.deadStateProbabilities[a] != state.deadStateProbabilities[b] { return state.deadStateProbabilities[a] > state.deadStateProbabilities[b] }
                return (state.times[a], a) < (state.times[b], b)
            }
            var selected: [Int] = []
            for i in eligible where selected.allSatisfy({ abs(state.times[i] - state.times[$0]) >= config.internalPeakMinimumSeparationSeconds }) { selected.append(i) }
            for i in selected.sorted(by: { state.times[$0] < state.times[$1] }) {
                let time = state.times[i]
                guard abs(time) < Double(Int64.max / 2000) else { throw AnalysisError.invalid("Invalid dead-peak timestamp") }
                candidates.append(.init(id: "switch:internal-dead-peak:\(range.id):\(Int64(floor(time * 1000 + 0.5)))", kind: .internalDeadStatePeak,
                    gapStart: max(range.start, time - config.internalPeakProposalHalfWidthSeconds), gapEnd: min(range.end, time + config.internalPeakProposalHalfWidthSeconds),
                    transitionTime: time, generatorScore: Double(state.deadStateProbabilities[i]), sourceRangeIds: [range.id],
                    beforeStart: time - 4, beforeEnd: time - 1, afterStart: time + 1, afterEnd: time + 4))
            }
        }
        return candidates.sorted { ($0.transitionTime, $0.kind.rawValue, $0.id) < ($1.transitionTime, $1.kind.rawValue, $1.id) }
    }
    private static func window(_ state: ProductionStateOutput, values: [Float], start: Double, end: Double) -> [Double] {
        let selected = state.times.indices.filter { state.times[$0] >= start && state.times[$0] < end }.map { Double(values[$0]) }
        if !selected.isEmpty { return selected }
        let center = (start + end) / 2
        var nearest = 0
        for i in state.times.indices where abs(state.times[i] - center) < abs(state.times[nearest] - center) { nearest = i }
        return [Double(values[nearest])]
    }
    public static func stateFeatures(candidate: SideSwitchCandidateProposal, input: SideSwitchAnalysisInput) throws -> [Double] {
        try validateStateOutputs(input)
        let named = ranges(input.intervals)
        guard let before = named.first(where: { $0.id == candidate.sourceRangeIds.first }),
              let after = named.first(where: { $0.id == candidate.sourceRangeIds.last }) else { throw AnalysisError.invalid("Side-switch candidate source range missing") }
        func evidence(_ range: (id: String, start: Double, end: Double)) -> (count: Double, peak: Double) {
            var peaks: [Double] = []
            for (ranges, state) in [(input.allLabelsV2Components, input.allLabelsV2State), (input.previousProductionComponents, input.previousProductionState)] {
                if ranges.contains(where: { min(range.end, $0.end) > max(range.start, $0.start) }) {
                    peaks.append(window(state, values: state.rallyProbabilities, start: range.start, end: range.end).max() ?? 0)
                }
            }
            return (Double(peaks.count), peaks.min() ?? 0)
        }
        let a = evidence(before), b = evidence(after), all = input.allLabelsV2State, previous = input.previousProductionState
        let rally = all.times.indices.map { max(all.rallyProbabilities[$0], previous.rallyProbabilities[$0]) }
        let dead = all.times.indices.map { max(all.deadStateProbabilities[$0], previous.deadStateProbabilities[$0]) }
        let rw = window(all, values: rally, start: candidate.gapStart, end: candidate.gapEnd)
        let dw = window(all, values: dead, start: candidate.gapStart, end: candidate.gapEnd)
        let clipped: [(Double, Double)] = (input.allLabelsV2Components + input.previousProductionComponents).compactMap {
            let start = max(candidate.gapStart, $0.start), end = min(candidate.gapEnd, $0.end)
            return end > start ? (start, end) : nil
        }.sorted { ($0.0, $0.1) < ($1.0, $1.1) }
        var total = 0.0, span: (Double, Double)?
        for item in clipped {
            if let old = span {
                if item.0 <= old.1 { span = (old.0, max(old.1, item.1)) }
                else { total += old.1 - old.0; span = item }
            } else { span = item }
        }
        if let span { total += span.1 - span.0 }
        return [a.count, b.count, min(a.count, b.count), min(a.peak, b.peak), total / max(candidate.gapEnd - candidate.gapStart, 1e-6),
            rw.reduce(0, +) / Double(rw.count), rw.max() ?? 0, dw.reduce(0, +) / Double(dw.count), dw.max() ?? 0, candidate.gapEnd - candidate.gapStart]
    }
    public func predict(features: [Double]) throws -> [Double] {
        guard features.count % 34 == 0 else { throw AnalysisError.invalid("Side-switch matrix width mismatch") }
        let model = runtime.classifier
        return (0..<(features.count / 34)).map { row in
            var score = model.bias
            for column in 0..<34 {
                let raw = features[row * 34 + column], value = raw.isFinite ? raw : model.impute[column]
                score += ((value - model.mean[column]) / model.scale[column]) * model.weights[column]
            }
            return 1 / (1 + exp(-max(-30, min(30, score))))
        }
    }
    public func decode(candidates: [SideSwitchCandidateProposal], probabilities: [Double]) throws -> [Int] {
        guard candidates.count == probabilities.count, probabilities.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 }) else {
            throw AnalysisError.invalid("Side-switch probability dimensions or values invalid")
        }
        let chronological = candidates.indices.sorted { (candidates[$0].transitionTime, candidates[$0].id) < (candidates[$1].transitionTime, candidates[$1].id) }
        var ordinal = [Int](repeating: 0, count: candidates.count)
        for (i, candidate) in chronological.enumerated() { ordinal[candidate] = i }
        let ranked = candidates.indices.sorted { a, b in
            if probabilities[a] != probabilities[b] { return probabilities[a] > probabilities[b] }
            return (candidates[a].transitionTime, candidates[a].id) < (candidates[b].transitionTime, candidates[b].id)
        }
        func logit(_ p: Double) -> Double { let clipped = min(max(p, 1e-9), 1 - 1e-9); return log(clipped / (1 - clipped)) }
        let policy = runtime.decoder, threshold = logit(runtime.classifier.threshold)
        var selected: [Int] = []
        for i in ranked {
            if selected.contains(where: { (policy.minimumCandidateIndexSeparation > 0 && abs(ordinal[i] - ordinal[$0]) < policy.minimumCandidateIndexSeparation) ||
                (policy.minimumTimeSeparationSeconds > 0 && abs(candidates[i].transitionTime - candidates[$0].transitionTime) < policy.minimumTimeSeparationSeconds) }) { continue }
            let excess = max(0, selected.count + 1 - policy.freePredictionsPerRecording)
            if logit(probabilities[i]) - threshold - policy.countPenaltyLogitPerExcessPrediction * Double(excess) >= -1e-12 { selected.append(i) }
        }
        return selected.sorted { (candidates[$0].transitionTime, candidates[$0].id) < (candidates[$1].transitionTime, candidates[$1].id) }
    }
    public func evaluate(input: SideSwitchAnalysisInput, plan: FramePlan, visualFeatures: [Double]) throws -> SideSwitchOutput {
        guard visualFeatures.count == plan.candidates.count * 22, visualFeatures.allSatisfy(\.isFinite) else { throw AnalysisError.invalid("Side-switch visual bank must contain 22 finite columns") }
        var features: [Double] = []
        for (row, candidate) in plan.candidates.enumerated() {
            try Task.checkCancellation()
            features += visualFeatures[(row * 22)..<((row + 1) * 22)]
            features += try Self.stateFeatures(candidate: candidate, input: input)
            features += [candidate.kind == .internalDeadStatePeak ? 1 : 0, candidate.generatorScore]
        }
        let probabilities = try predict(features: features), selected = try decode(candidates: plan.candidates, probabilities: probabilities)
        let predictions: [SideSwitchPrediction] = selected.map { index in
            let c = plan.candidates[index]
            return .init(id: c.id, timestamp: c.transitionTime, probability: probabilities[index], kind: c.kind, sourceRangeIds: c.sourceRangeIds)
        }
        var output = SideSwitchOutput(candidates: predictions, features: features)
        output.rows = plan.candidates.count // rows includes unselected proposals, unlike candidates (selected markers).
        return output
    }
}
