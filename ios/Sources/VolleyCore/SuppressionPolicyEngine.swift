import CryptoKit
import Foundation

public struct SuppressionSuggestion: Codable, Equatable, Sendable {
    public var logicalId: String, fragmentId: String
    public var startMs: Int64, endMs: Int64
    public var score: Float
    public var sourceProductionIds: [String], eligiblePolicyIds: [String]
    public init(logicalId: String, fragmentId: String, startMs: Int64, endMs: Int64, score: Float,
                sourceProductionIds: [String], eligiblePolicyIds: [String]) {
        self.logicalId = logicalId; self.fragmentId = fragmentId; self.startMs = startMs; self.endMs = endMs
        self.score = score; self.sourceProductionIds = sourceProductionIds; self.eligiblePolicyIds = eligiblePolicyIds
    }
    /// Preserves logical identity across policy fragments for the existing draft override algebra.
    public var region: SuppressionRegion {
        .init(id: fragmentId, logicalId: logicalId, startMs: startMs, endMs: endMs, eligiblePolicyIds: eligiblePolicyIds)
    }
}
public struct SuppressionAnalysis: Codable, Equatable, Sendable {
    public let modelId: String, artifactSha256: String, weightsSha256: String, decoderVersion: String
    public let probabilities: [Float], decodedIntervals: [Interval], suggestions: [SuppressionSuggestion]
}

/// Android suppression-policy-v1 millisecond algebra. Suggestions describe eligible regions;
/// EditorMath applies a draft's explicit decisions and whole-rally/veto-region scope.
public enum SuppressionPolicyEngine {
    public static let contractVersion = "suppression-policy-v1"
    public enum Policy: String, Codable, Sendable, CaseIterable {
        case none, conservative, balanced, aggressive
        public static func fromWireName(_ value: String) -> Policy { Policy(rawValue: value) ?? .none }
        public var label: String {
            switch self { case .none: "No suppression"; case .conservative: "Conservative"; case .balanced: "Balanced"; case .aggressive: "Aggressive" }
        }
        public var recordedPolicy: String {
            switch self { case .none: "none"; case .conservative: "zero-non-exempt-misses"; case .balanced: "aggressive-intermediate"; case .aggressive: "raw-connected" }
        }
        fileprivate var padding: Int64 { self == .conservative ? 2000 : self == .balanced ? 1500 : 0 }
        fileprivate var join: Int64 { self == .conservative || self == .balanced ? 500 : 0 }
    }
    private struct Tagged {
        let id: String, start: Int64, end: Int64, allLabels: Bool
        var paddedStart: Int64, paddedEnd: Int64
    }
    private struct Span { let start: Int64, end: Int64 }
    private static let policies: [Policy] = [.conservative, .balanced, .aggressive]
    private static func milliseconds(_ seconds: Double) -> Int64 {
        // Java Math.round ties towards positive infinity. Callers validate representability first.
        Int64(floor(seconds * 1000 + 0.5))
    }
    private static func overlaps(_ a: Int64, _ b: Int64, _ c: Int64, _ d: Int64) -> Bool { a < d && c < b }

    public static func build(allLabelsV2: [Interval], previousProduction: [Interval], probabilities: [Float],
                             decoded: [Interval], duration: Double) throws -> SuppressionAnalysis {
        guard duration.isFinite, duration >= 0, duration < Double(Int64.max / 2000), probabilities.allSatisfy(\.isFinite),
              (allLabelsV2 + previousProduction + decoded).allSatisfy({ $0.start.isFinite && $0.end.isFinite &&
                  abs($0.start) < Double(Int64.max / 2000) && abs($0.end) < Double(Int64.max / 2000) && $0.confidence.isFinite }) else {
            throw AnalysisError.invalid("Invalid suppression policy intervals")
        }
        let durationMs = milliseconds(duration)
        var raw: [Tagged] = []
        for (source, allLabels, prefix) in [(allLabelsV2, true, "all-labels-v2"), (previousProduction, false, "previous-production")] {
            for (index, interval) in source.enumerated() {
                let start = max(0, milliseconds(interval.start)), end = min(durationMs, milliseconds(interval.end))
                if end <= start { continue }
                let id = prefix + ":" + String(format: "%04d", index + 1) + ":\(start):\(end)"
                raw.append(Tagged(id: id, start: start, end: end, allLabels: allLabels, paddedStart: start, paddedEnd: end))
            }
        }
        let policySpans: [(Span, Policy)] = policies.flatMap { policy in
            eligible(raw, policy, durationMs).map { ($0, policy) }
        }
        var suggestions: [SuppressionSuggestion] = []
        for event in decoded {
            let eventStart = max(0, milliseconds(event.start)), eventEnd = min(durationMs, milliseconds(event.end))
            if eventEnd <= eventStart { continue }
            let sources = raw.filter { overlaps(eventStart, eventEnd, $0.start, $0.end) }
            if sources.isEmpty { continue }
            let canonical = "\(eventStart):\(eventEnd):" + sources.map(\.id).sorted().joined(separator: ",")
            let logicalId = "S-" + SHA256.hash(data: Data(canonical.utf8)).prefix(10).map { String(format: "%02x", $0) }.joined()
            var boundaries: Set<Int64> = [eventStart, eventEnd]
            for (span, _) in policySpans {
                let start = max(eventStart, span.start), end = min(eventEnd, span.end)
                if end > start { boundaries.insert(start); boundaries.insert(end) }
            }
            let ordered = boundaries.sorted()
            for index in 0..<(ordered.count - 1) {
                let start = ordered[index], end = ordered[index + 1], midpoint = start + (end - start) / 2
                let eligiblePolicies: [Policy] = policies.filter { policy in
                    policySpans.contains { span, candidatePolicy in
                        candidatePolicy == policy && midpoint >= span.start && midpoint < span.end
                    }
                }
                let eligibleIds: [String] = eligiblePolicies.map(\.rawValue)
                if eligibleIds.isEmpty { continue }
                let ids = sources.filter { overlaps(start, end, $0.start, $0.end) }.map(\.id).sorted()
                if ids.isEmpty { continue }
                if let previous = suggestions.last, previous.logicalId == logicalId, previous.endMs == start, previous.eligiblePolicyIds == eligibleIds {
                    suggestions[suggestions.count - 1] = SuppressionSuggestion(logicalId: logicalId, fragmentId: "\(logicalId):\(previous.startMs):\(end)",
                        startMs: previous.startMs, endMs: end, score: max(previous.score, event.confidence),
                        sourceProductionIds: Array(Set(previous.sourceProductionIds + ids)).sorted(), eligiblePolicyIds: eligibleIds)
                } else {
                    suggestions.append(SuppressionSuggestion(logicalId: logicalId, fragmentId: "\(logicalId):\(start):\(end)",
                        startMs: start, endMs: end, score: event.confidence, sourceProductionIds: ids, eligiblePolicyIds: eligibleIds))
                }
            }
        }
        return SuppressionAnalysis(modelId: SuppressionModelRunner.modelId, artifactSha256: SuppressionModelRunner.artifactSha256,
            weightsSha256: SuppressionModelRunner.weightsSha256, decoderVersion: SuppressionModelRunner.decoderVersion,
            probabilities: probabilities, decodedIntervals: decoded, suggestions: suggestions)
    }

    public static func active(_ analysis: SuppressionAnalysis?, policy: Policy) -> [SuppressionSuggestion] {
        guard let analysis, policy != .none else { return [] }
        return analysis.suggestions.filter { $0.eligiblePolicyIds.contains(policy.rawValue) }.sorted { ($0.startMs, $0.endMs) < ($1.startMs, $1.endMs) }
    }

    private static func eligible(_ raw: [Tagged], _ policy: Policy, _ duration: Int64) -> [Span] {
        let padded = raw.map { item in
            var item = item; item.paddedStart = max(0, item.start - policy.padding); item.paddedEnd = min(duration, item.end + policy.padding); return item
        }.sorted { ($0.paddedStart, $0.paddedEnd) < ($1.paddedStart, $1.paddedEnd) }
        var components: [[Tagged]] = [], componentEnd: Int64 = 0
        for item in padded {
            let gap = item.paddedStart - componentEnd
            if components.isEmpty || (gap > 0 && gap >= policy.join) {
                components.append([]); componentEnd = item.paddedEnd
            } else { componentEnd = max(componentEnd, item.paddedEnd) }
            components[components.count - 1].append(item)
        }
        let singleSourceComponents: [[Tagged]] = components.filter { Set($0.map(\.allLabels)).count == 1 }
        let eligibleItems: [Tagged] = singleSourceComponents.flatMap { $0 }
        let spans: [Span] = eligibleItems.map { Span(start: $0.start, end: $0.end) }
        let source: [Span] = spans.sorted { ($0.start, $0.end) < ($1.start, $1.end) }
        var merged: [Span] = []
        for item in source {
            if let previous = merged.last, item.start <= previous.end {
                merged[merged.count - 1] = Span(start: previous.start, end: max(previous.end, item.end))
            } else { merged.append(item) }
        }
        return merged
    }
}
