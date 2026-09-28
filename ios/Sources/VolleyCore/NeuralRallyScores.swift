import Foundation

/// Retained temporal outputs only. Image embeddings stay in the device cache.
public struct NeuralRallyScores: Codable, Equatable, Sendable {
    public static let heads = ["live", "serve", "end", "keep"]
    public let modelId: String
    public let times: [Double]
    public let probabilities: [Float]
    public init(modelId: String, times: [Double], probabilities: [Float]) throws {
        self.modelId = modelId; self.times = times; self.probabilities = probabilities
        try validate()
    }
    public func validate() throws {
        guard RallyModel(rawValue: modelId)?.isNeural == true, !times.isEmpty,
              times.allSatisfy({ $0.isFinite && $0 >= 0 }),
              zip(times, times.dropFirst()).allSatisfy({ $0 < $1 }),
              probabilities.count == times.count * Self.heads.count,
              probabilities.allSatisfy({ $0.isFinite && (0...1).contains($0) }) else {
            throw ProjectError.invalid("Invalid neural model score traces")
        }
    }
    private enum CodingKeys: String, CodingKey { case modelId, times, probabilities }
    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(modelId: values.decode(String.self, forKey: .modelId),
                      times: values.decode([Double].self, forKey: .times),
                      probabilities: values.decode([Float].self, forKey: .probabilities))
    }
}
