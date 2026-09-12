import Foundation

/// Frozen overlap-safe specialist, including Android's upper-median cadence and division smoothing.
public struct SuppressionModelRunner: Sendable {
    public static let modelId = "suppression-overlap-exclusion-retrained"
    public static let artifactSha256 = "39eddf58163901930434ea422a802686ae921ae1e8fe23c20a3c12e5f453da93"
    public static let weightsSha256 = "a943749b69c98a1bc926f8efc9fe60c67c1226534fe09a892c632519217aa3bb"
    public static let decoderVersion = "held-production-suppression-decoder-v1"
    public static let assetSha256 = "02274d0f17b89cd54ea24da7d1665a6475e48dc0f554d06092e9ef892332d4f1"

    public struct Result: Codable, Sendable {
        public let probabilities: [Float]
        public let decodedIntervals: [Interval]
    }
    private struct Head: Decodable, Sendable { let mean: [Float], scale: [Float], weights: [Float], bias: Float }
    private struct Bundle: Decodable, Sendable {
        let schemaVersion: Int, modelId: String, artifactSha256: String, weightsSha256: String, decoderVersion: String
        let analysisFps: Double, featureNames: [String], head: Head, decoder: [String: Double]
    }
    private let bundle: Bundle
    public init(data: Data, expectedFeatureNames: [String] = FeatureSchema.contextualNames) throws {
        let value = try JSONDecoder().decode(Bundle.self, from: data)
        guard value.schemaVersion == 1, value.modelId == Self.modelId,
              value.artifactSha256 == Self.artifactSha256, value.weightsSha256 == Self.weightsSha256,
              value.decoderVersion == Self.decoderVersion, value.featureNames == expectedFeatureNames,
              !expectedFeatureNames.isEmpty, value.analysisFps.isFinite, value.analysisFps > 0 else {
            throw AnalysisError.invalid("Suppression identity or feature signature does not match the product contract")
        }
        let held: [String: Double] = ["smoothing_seconds": 1, "enter_threshold": 0.75, "exit_threshold": 0.65,
            "min_live_seconds": 0.5, "bridge_gap_seconds": 0.5, "short_event_min_seconds": 0.25, "short_event_threshold": 0.9]
        guard held.allSatisfy({ value.decoder[$0.key] == $0.value }) else {
            throw AnalysisError.invalid("Suppression decoder differs from the held production decoder")
        }
        let head = value.head
        guard head.mean.count == expectedFeatureNames.count, head.scale.count == expectedFeatureNames.count,
              head.weights.count == expectedFeatureNames.count, head.mean.allSatisfy(\.isFinite), head.weights.allSatisfy(\.isFinite),
              head.scale.allSatisfy({ $0.isFinite && $0 > 0 }), head.bias.isFinite else {
            throw AnalysisError.invalid("Invalid suppression model vectors")
        }
        bundle = value
    }
    public func run(times: [Double], contextual: [Float], duration: Double) throws -> Result {
        guard duration.isFinite, duration >= 0 else { throw AnalysisError.invalid("Invalid suppression duration") }
        if times.isEmpty {
            guard contextual.isEmpty else { throw AnalysisError.invalid("Suppression matrix shape mismatch") }
            return Result(probabilities: [], decodedIntervals: [])
        }
        try FeatureMath.validate(times: times, values: contextual, columns: bundle.featureNames.count)
        guard times.last! <= duration else { throw AnalysisError.invalid("Suppression sample exceeds duration") }
        let head = bundle.head, width = head.weights.count
        let probabilities: [Float] = times.indices.map { row in
            var logit = head.bias
            for column in 0..<width {
                let normalized = (contextual[row * width + column] - head.mean[column]) / head.scale[column]
                logit += normalized * head.weights[column]
            }
            return Float(1 / (1 + exp(-Double(max(-30, min(30, logit))))))
        }
        return Result(probabilities: probabilities, decodedIntervals: decode(times, probabilities, duration))
    }
    private func runs(_ values: [Bool], _ expected: Bool) -> [Range<Int>] {
        var result: [Range<Int>] = [], start: Int?
        for i in values.indices {
            if values[i] == expected && start == nil { start = i }
            if values[i] != expected, let lower = start { result.append(lower..<i); start = nil }
        }
        if let start { result.append(start..<values.count) }
        return result
    }
    private func decode(_ times: [Double], _ probabilities: [Float], _ duration: Double) -> [Interval] {
        var fps = bundle.analysisFps
        if times.count > 1 {
            let differences = zip(times, times.dropFirst()).map { $1 - $0 }.sorted()
            let median = differences[differences.count / 2]
            if median.isFinite && median > 0 { fps = 1 / median }
        }
        func samples(_ seconds: Double) -> Int {
            let count = (seconds * fps).rounded(.toNearestOrEven)
            return count >= Double(Int.max) ? Int.max : Int(count)
        }
        let window = min(probabilities.count, max(1, samples(1)))
        var smoothed = probabilities
        if window > 1 {
            for output in probabilities.indices {
                var sum: Float = 0
                for kernel in 0..<window {
                    let source = max(0, min(probabilities.count - 1, output + kernel - window / 2))
                    sum += probabilities[source] / Float(window)
                }
                smoothed[output] = sum
            }
        }
        var active = false
        var live = smoothed.map { probability in
            if !active && Double(probability) >= 0.75 { active = true }
            else if active && Double(probability) < 0.65 { active = false }
            return active
        }
        let bridge = max(0, samples(0.5))
        for run in runs(live, false) where run.lowerBound > 0 && run.upperBound < live.count && run.count <= bridge {
            for i in run { live[i] = true }
        }
        let minimum = max(1, samples(0.5)), shortMinimum = max(1, samples(0.25))
        for run in runs(live, true) {
            if run.count < minimum && !(run.count >= shortMinimum && Double(smoothed[run].max()!) >= 0.9) {
                for i in run { live[i] = false }
            }
        }
        return runs(live, true).compactMap { run in
            let start = max(0, times[run.lowerBound] - 0.5 / fps), end = min(duration, times[run.upperBound - 1] + 0.5 / fps)
            guard end > start else { return nil }
            var score: Float = 0; for i in run { score += smoothed[i] }
            return Interval(start: start, end: end, confidence: score / Float(run.count))
        }
    }
}
