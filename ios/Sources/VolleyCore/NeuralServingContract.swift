import CryptoKit
import Foundation

public struct NeuralServingBundle: Codable, Equatable, Sendable {
    public struct Seed: Codable, Equatable, Sendable {
        public let seed: Int, epoch: Int
        public let weightsSha256: String
        public let farCutoff: Double, nearCutoff: Double
    }
    public let id: String, fingerprint: String, rallyModelId: String, directory: String
    public let encoderSha256: String, encoderWeightsSha256: String
    public let file: NeuralRallyAsset
    public let seeds: [Seed]
}

/// Frozen production catalog, shared with the browser. A head must match its encoder.
public struct NeuralServingManifest: Sendable {
    public static let sha256 = "f71e00655efb89af2f8f5e231b673937588319c99027beb8efc016d637c682eb"
    public let variants: [String: NeuralServingBundle]
    public static func identity(for model: RallyModel) -> (id: String, fingerprint: String) {
        switch model {
        case .maximumCoverage: return ("serving-side-centered-recall-v1", "ff535f821460f2565e9db4df1ed3b36b7f1f3b0790ee87f05ff9aae5adc6ee76")
        case .balanced: return ("serving-side-centered-f1-v1", "14472e860198753ebfedd185a315ba7c68f36d124d741481b9fe5146ead0f7d1")
        case .legacy: return (ServingSideInference.modelId, ServingSideInference.fingerprint)
        }
    }
    public init(data: Data) throws {
        let canonical = Data(String(decoding: data, as: UTF8.self).replacingOccurrences(of: "\r\n", with: "\n").utf8)
        guard SHA256.hash(data: canonical).map({ String(format: "%02x", $0) }).joined() == Self.sha256 else {
            throw AnalysisError.invalid("The serving-side catalog differs from the frozen production models")
        }
        struct Payload: Decodable { let variants: [String: NeuralServingBundle] }
        variants = try JSONDecoder().decode(Payload.self, from: canonical).variants
    }
    public func bundle(for rally: NeuralRallyBundle) throws -> NeuralServingBundle {
        guard let value = variants.values.first(where: { $0.rallyModelId == rally.id }),
              value.encoderSha256 == rally.files.encoder.sha256,
              value.encoderWeightsSha256 == rally.encoderWeightsSha256 else {
            throw AnalysisError.invalid("The serving-side head does not match the selected rally encoder")
        }
        return value
    }
}

public enum NeuralServingContract {
    public struct Decision: Equatable, Sendable {
        public let nearProbability: Double, side: ServingSide, accepted: Bool
    }
    public static func decision(logits: [Float], bundle: NeuralServingBundle) throws -> Decision {
        guard logits.count == 6, logits.allSatisfy(\.isFinite), bundle.seeds.count == 3 else {
            throw AnalysisError.invalid("The serving-side model returned invalid scores")
        }
        var routes: [Double] = [], decisions: [ServingSide?] = []
        for seed in 0..<3 {
            func sigmoid(_ x: Float) -> Double { 1 / (1 + exp(-max(-35, min(35, Double(x))))) }
            let far = sigmoid(logits[seed * 2]), near = sigmoid(logits[seed * 2 + 1]), mass = far + near
            guard mass > 0, mass <= 1 + 1e-12 else { throw AnalysisError.invalid("Serving-side evidence exceeds its available mass") }
            let route = near / mass, side: ServingSide = route >= 0.5 ? .near : .far
            let confidence = min(mass, 1) * (side == .near ? route : 1 - route)
            let cutoff = side == .near ? bundle.seeds[seed].nearCutoff : bundle.seeds[seed].farCutoff
            routes.append(route); decisions.append(confidence > 0 && confidence >= cutoff ? side : nil)
        }
        let near = routes.reduce(0, +) / 3
        return Decision(nearProbability: near, side: near >= 0.5 ? .near : .far,
                        accepted: decisions[0] != nil && decisions.allSatisfy { $0 == decisions[0] })
    }
    public static func apply(_ decision: Decision, to candidate: ServingSideCandidate) -> ServingSideCandidate {
        var value = candidate
        value.nearProbability = decision.nearProbability; value.side = decision.side
        value.reviewReasons.removeAll { $0 == .sideScore }
        if !decision.accepted { value.reviewReasons.insert(.sideScore, at: 0) }
        value.verdict = candidate.verdict == .notServe ? .notServe : !value.reviewReasons.isEmpty ? .review : decision.side == .near ? .near : .far
        return value
    }
    /// Float64 recording-wide ranks, followed by 12 unranked phase quality values.
    public static func features(raw: [Double], rows: Int) throws -> [Float] {
        let ranked = try ServingSideInference.tiedPercentileRanks(raw, rows: rows, columns: 237)
        let columns = ["launch", "early", "late"].flatMap { phase in
            ["energyMean", "activeFraction", "smallComponentEnergyFraction", "flowY"].map {
                ServingSideInference.featureNames.firstIndex(of: "flight:\(phase):\($0)")!
            }
        }
        var result = [Float](repeating: 0, count: rows * 249)
        for row in 0..<rows {
            for column in 0..<237 { result[row * 249 + column] = Float(ranked[row * 237 + column]) }
            for column in 0..<12 {
                let value = raw[row * 237 + columns[column]]
                result[row * 249 + 237 + column] = Float(column % 4 == 3 ? abs(value) : value)
            }
        }
        guard result.allSatisfy(\.isFinite) else { throw AnalysisError.invalid("Non-finite neural serving-side features") }
        return result
    }
    public static func embeddingTimes(anchors: [Double], duration: Double) throws -> [Double] {
        guard duration.isFinite, duration > 0, anchors.allSatisfy({ $0.isFinite && $0 >= 0 && $0 < duration }) else {
            throw AnalysisError.invalid("Invalid serving-side anchor")
        }
        var ticks = Set<Double>()
        for anchor in anchors { for frame in 0..<13 {
            let requested = anchor - 3 + Double(frame) * 0.5
            if requested < 0 || requested >= duration { continue }
            let tick = ceil(requested * 2 - 0.5)
            if tick >= 0 && tick / 2 < duration { ticks.insert(tick / 2) }
        } }
        return ticks.sorted()
    }
    /// Centered global half-second grid. Nearest ties choose the earlier image.
    public static func window(times: [Double], tokens: [Float], anchor: Double, duration: Double) throws -> [Float] {
        guard anchor.isFinite, anchor >= 0, anchor < duration, !times.isEmpty,
              tokens.count == times.count * 3840, times.allSatisfy(\.isFinite),
              zip(times, times.dropFirst()).allSatisfy({ $0 < $1 }) else { throw AnalysisError.invalid("Invalid serving-side image sequence") }
        var values = [Float](repeating: 0, count: 13 * 3843)
        for frame in 0..<13 {
            let offset = -3 + Double(frame) * 0.5, requested = anchor + offset, destination = frame * 3843
            values[destination + 3841] = Float(offset / 3)
            if requested < 0 || requested >= duration { continue }
            var low = 0, high = times.count
            while low < high { let mid = (low + high) / 2; if times[mid] < requested { low = mid + 1 } else { high = mid } }
            let later = min(low, times.count - 1), earlier = max(0, later - 1)
            let chosen = abs(times[earlier] - requested) <= abs(times[later] - requested) ? earlier : later
            if abs(times[chosen] - requested) > 0.250001 { continue }
            values.replaceSubrange(destination..<(destination + 3840), with: tokens[(chosen * 3840)..<((chosen + 1) * 3840)])
            values[destination + 3840] = 1; values[destination + 3842] = Float((times[chosen] - requested) / 0.25)
        }
        guard values[6 * 3843 + 3840] == 1 else { throw AnalysisError.invalid("The serving anchor has no qualified image sample") }
        return values
    }
}
