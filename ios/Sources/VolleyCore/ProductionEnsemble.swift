import Foundation

/// overlap-union-disagreement-v1. Touching intervals remain separate; overlaps are transitive.
public enum ProductionEnsemble {
    public static let bothModels = "both-models"
    public static let allLabelsV2Only = "all-labels-v2-only"
    public static let previousProductionOnly = "previous-production-only"

    public static func isValidAgreement(_ value: String?) -> Bool {
        value == bothModels || isDisagreement(value)
    }
    public static func isDisagreement(_ value: String?) -> Bool {
        value == allLabelsV2Only || value == previousProductionOnly
    }

    public static func merge(allLabelsV2: [Interval], previousProduction: [Interval]) -> [Interval] {
        let allCandidates: [(Interval, Bool)] = allLabelsV2.map { ($0, true) }
        let previousCandidates: [(Interval, Bool)] = previousProduction.map { ($0, false) }
        let valid: [(Interval, Bool)] = (allCandidates + previousCandidates).filter {
            $0.0.start.isFinite && $0.0.end.isFinite && $0.0.end > $0.0.start
        }
        let candidates = valid.sorted { left, right in
            left.0.start == right.0.start ? left.0.end < right.0.end : left.0.start < right.0.start
        }
        var clusters: [[(Interval, Bool)]] = [], currentEnd = -Double.infinity
        for candidate in candidates {
            if clusters.isEmpty || candidate.0.start >= currentEnd {
                clusters.append([candidate]); currentEnd = candidate.0.end
            } else {
                clusters[clusters.count - 1].append(candidate)
                currentEnd = max(currentEnd, candidate.0.end)
            }
        }
        return clusters.map { cluster in
            let all = cluster.filter { $0.1 }.map { $0.0.confidence }.max()
            let previous = cluster.filter { !$0.1 }.map { $0.0.confidence }.max()
            let agreement: String, confidence: Float
            if let all, let previous { agreement = bothModels; confidence = (all + previous) / 2 }
            else if let all { agreement = allLabelsV2Only; confidence = min(0.49, all * 0.6) }
            else { agreement = previousProductionOnly; confidence = min(0.49, previous! * 0.6) }
            return Interval(start: cluster.map { $0.0.start }.min()!, end: cluster.map { $0.0.end }.max()!, confidence: confidence, agreement: agreement)
        }
    }
}
