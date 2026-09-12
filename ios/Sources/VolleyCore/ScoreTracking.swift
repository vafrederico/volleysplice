import Foundation

public enum ServingSide: String, Codable, Sendable { case near, far, review }
public enum ServingSideVerdict: String, Codable, Sendable { case near, far, review, notServe = "not-serve" }
public enum ServingSideDecisionSource: String, Codable, Sendable {
    case serveHead = "serve-head", productionRallyRecovery = "production-rally-recovery", none
}
public enum ServingSideReviewReason: String, Codable, Sendable { case sideScore = "side-score", productionRallyRecovery = "production-rally-recovery" }
public enum ScoreTeamId: String, Codable, Sendable { case team1 = "team-1", team2 = "team-2" }
public enum ServeMarkerOrigin: String, Codable, Sendable { case model, manual }
public enum ScorePointStatus: String, Codable, Sendable { case counted, ignored, review }
public enum SideSwitchCandidateKind: String, Codable, Sendable {
    case adjacentRallyBoundary = "adjacent-rally-boundary", internalDeadStatePeak = "internal-dead-state-peak"
}

public struct ServingSideCandidate: Codable, Equatable, Sendable {
    public var id: String
    public var anchor: Double
    public var intervalStart: Double
    public var intervalEnd: Double
    public var agreement: String?
    public var nearProbability: Double
    public var side: ServingSide
    public var verdict: ServingSideVerdict
    public var serveDecisionSource: ServingSideDecisionSource
    public var reviewReasons: [ServingSideReviewReason]
    // Retains the complete specialist evidence without running specialist inference.
    public var allLabelsV2Evidence: JSONValue?
    public var previousProductionEvidence: JSONValue?
    public init(id: String, anchor: Double, intervalStart: Double? = nil, intervalEnd: Double,
                agreement: String? = nil, nearProbability: Double = 0.5, side: ServingSide,
                verdict: ServingSideVerdict, serveDecisionSource: ServingSideDecisionSource = .none,
                reviewReasons: [ServingSideReviewReason] = [], allLabelsV2Evidence: JSONValue? = nil,
                previousProductionEvidence: JSONValue? = nil) {
        self.id = id; self.anchor = anchor; self.intervalStart = intervalStart ?? anchor; self.intervalEnd = intervalEnd
        self.agreement = agreement; self.nearProbability = nearProbability; self.side = side; self.verdict = verdict
        self.serveDecisionSource = serveDecisionSource; self.reviewReasons = reviewReasons
        self.allLabelsV2Evidence = allLabelsV2Evidence; self.previousProductionEvidence = previousProductionEvidence
    }
}
public struct ServingSideOutput: Codable, Equatable, Sendable {
    public var modelId = "serving-side-fixed-flight-v3"
    public var modelFingerprint = "85bc3325fbd43abba6ba3726ac091dc6a3a1eafc68d63d979cb2a7450eb49e06"
    public var featureVersion = "SERVSIDE237-FLIGHT"
    public var anchorContract = "merged-production-interval-start-v1"
    public var rows: Int
    public var columns = 237
    public var rawFeatures: [Double]
    public var candidates: [ServingSideCandidate]
    public init(candidates: [ServingSideCandidate], rawFeatures: [Double] = []) {
        self.candidates = candidates; self.rows = candidates.count; self.rawFeatures = rawFeatures
    }
}
public struct SideSwitchPrediction: Codable, Equatable, Sendable {
    public var id: String
    public var timestamp: Double
    public var probability: Double
    public var kind: SideSwitchCandidateKind
    public var sourceRangeIds: [String]
    public init(id: String, timestamp: Double, probability: Double, kind: SideSwitchCandidateKind,
                sourceRangeIds: [String] = []) {
        self.id = id; self.timestamp = timestamp; self.probability = probability
        self.kind = kind; self.sourceRangeIds = sourceRangeIds
    }
}
public struct SideSwitchOutput: Codable, Equatable, Sendable {
    public var modelId = "side-switch-hard-negative-mining-v1/union34-top2-x2"
    public var modelFingerprint = "sha256:c2570481c30dec62f56ac3284cd4028763ffd8e72a9ad07e9908fec10df387e3"
    public var featureVersion = "SIDE-SWITCH-UNION34-V1"
    public var candidateContract = "range-boundaries-dead-peaks-v1"
    public var rows: Int
    public var columns = 34
    public var features: [Double]
    public var candidates: [SideSwitchPrediction]
    public init(candidates: [SideSwitchPrediction], features: [Double] = []) {
        self.candidates = candidates; self.rows = candidates.count; self.features = features
    }
}

private func scoreMilliseconds(_ seconds: Double) throws -> Int64 {
    guard seconds.isFinite, seconds >= 0, seconds < Double(Int64.max) / 1000 - 1 else {
        throw ProjectError.invalid("Invalid score source timestamp")
    }
    return Int64(floor(seconds * 1000 + 0.5))
}
public struct ServeMarker: Codable, Equatable, Identifiable, Sendable {
    public var id: String
    public var timestampMs: Int64
    public var side: ServingSide
    public var origin: ServeMarkerOrigin
    public var modelSide: ServingSide?
    public var ignorePreviousPoint: Bool
    public var rallyId: String?
    public init(id: String, timestampMs: Int64, side: ServingSide, origin: ServeMarkerOrigin = .manual,
                modelSide: ServingSide? = nil, ignorePreviousPoint: Bool = false, rallyId: String? = nil) {
        self.id = id; self.timestampMs = timestampMs; self.side = side; self.origin = origin
        self.modelSide = modelSide; self.ignorePreviousPoint = ignorePreviousPoint; self.rallyId = rallyId
    }
    enum CodingKeys: String, CodingKey { case id, timestamp, side, origin, modelSide, ignorePreviousPoint, rallyId }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        timestampMs = try scoreMilliseconds(c.decode(Double.self, forKey: .timestamp))
        side = try c.decode(ServingSide.self, forKey: .side); origin = try c.decode(ServeMarkerOrigin.self, forKey: .origin)
        modelSide = try c.decodeIfPresent(ServingSide.self, forKey: .modelSide)
        ignorePreviousPoint = try c.decode(Bool.self, forKey: .ignorePreviousPoint)
        rallyId = try c.decodeIfPresent(String.self, forKey: .rallyId)
    }
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id); try c.encode(Double(timestampMs) / 1000, forKey: .timestamp)
        try c.encode(side, forKey: .side); try c.encode(origin, forKey: .origin)
        try c.encodeIfPresent(modelSide, forKey: .modelSide); try c.encode(ignorePreviousPoint, forKey: .ignorePreviousPoint)
        try c.encodeIfPresent(rallyId, forKey: .rallyId)
    }
}
public struct SideSwitchMarker: Codable, Equatable, Identifiable, Sendable {
    public var id: String
    public var timestampMs: Int64
    public var origin: ServeMarkerOrigin
    public var modelConfidence: Double?
    public var modelEventId: String?
    public var rallyIds: [String]
    public init(id: String, timestampMs: Int64, origin: ServeMarkerOrigin = .manual,
                modelConfidence: Double? = nil, modelEventId: String? = nil, rallyIds: [String] = []) {
        self.id = id; self.timestampMs = timestampMs; self.origin = origin; self.modelConfidence = modelConfidence
        self.modelEventId = modelEventId; self.rallyIds = rallyIds
    }
    enum CodingKeys: String, CodingKey { case id, timestamp, origin, modelConfidence, modelEventId, rallyIds }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id); timestampMs = try scoreMilliseconds(c.decode(Double.self, forKey: .timestamp))
        origin = try c.decodeIfPresent(ServeMarkerOrigin.self, forKey: .origin) ?? .manual
        modelConfidence = try c.decodeIfPresent(Double.self, forKey: .modelConfidence)
        modelEventId = try c.decodeIfPresent(String.self, forKey: .modelEventId)
        rallyIds = try c.decodeIfPresent([String].self, forKey: .rallyIds) ?? []
    }
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id); try c.encode(Double(timestampMs) / 1000, forKey: .timestamp)
        try c.encode(origin, forKey: .origin); try c.encodeIfPresent(modelConfidence, forKey: .modelConfidence)
        try c.encodeIfPresent(modelEventId, forKey: .modelEventId)
        if !rallyIds.isEmpty { try c.encode(rallyIds, forKey: .rallyIds) }
    }
}
public struct ScoreTracking: Codable, Equatable, Sendable {
    public var version = 3
    public var enabled: Bool
    public var team1Name: String
    public var team2Name: String
    public var serveMarkers: [ServeMarker]
    public var sideSwitchMarkers: [SideSwitchMarker]
    public var removedModelMarkerIds: Set<String>
    public init(enabled: Bool = true, team1Name: String = "Team 1", team2Name: String = "Team 2",
                serveMarkers: [ServeMarker] = [], sideSwitchMarkers: [SideSwitchMarker] = [],
                removedModelMarkerIds: Set<String> = []) {
        self.enabled = enabled; self.team1Name = team1Name; self.team2Name = team2Name
        self.serveMarkers = serveMarkers; self.sideSwitchMarkers = sideSwitchMarkers; self.removedModelMarkerIds = removedModelMarkerIds
    }
    enum CodingKeys: String, CodingKey { case version, enabled, team1Name, team2Name, serveMarkers, sideSwitchMarkers, removedModelMarkerIds }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        guard (1...3).contains(try c.decode(Int.self, forKey: .version)) else { throw ProjectError.invalid("Unsupported score schema") }
        enabled = try c.decode(Bool.self, forKey: .enabled)
        team1Name = try c.decode(String.self, forKey: .team1Name); team2Name = try c.decode(String.self, forKey: .team2Name)
        serveMarkers = try c.decode([ServeMarker].self, forKey: .serveMarkers)
        sideSwitchMarkers = try c.decode([SideSwitchMarker].self, forKey: .sideSwitchMarkers)
        let removed = try c.decodeIfPresent([String].self, forKey: .removedModelMarkerIds) ?? []
        guard Set(removed).count == removed.count else { throw ProjectError.invalid("Duplicate removed score marker IDs") }
        removedModelMarkerIds = Set(removed)
        try validate()
    }
    public func encode(to encoder: Encoder) throws {
        try validate()
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(version, forKey: .version); try c.encode(enabled, forKey: .enabled)
        try c.encode(team1Name, forKey: .team1Name); try c.encode(team2Name, forKey: .team2Name)
        try c.encode(serveMarkers, forKey: .serveMarkers); try c.encode(sideSwitchMarkers, forKey: .sideSwitchMarkers)
        try c.encode(removedModelMarkerIds.sorted(), forKey: .removedModelMarkerIds)
    }
    public func validate(durationMs: Int64 = .max) throws {
        func blank(_ text: String) -> Bool { text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        let ids = serveMarkers.map(\.id) + sideSwitchMarkers.map(\.id)
        guard version == 3, durationMs >= 0, !blank(team1Name), !blank(team2Name), Set(ids).count == ids.count,
              Set(ids).isDisjoint(with: removedModelMarkerIds), !removedModelMarkerIds.contains(where: blank),
              serveMarkers.allSatisfy({ !blank($0.id) && (0...durationMs).contains($0.timestampMs) && !($0.rallyId.map(blank) ?? false) }),
              sideSwitchMarkers.allSatisfy({ !blank($0.id) && (0...durationMs).contains($0.timestampMs) &&
                  ($0.modelConfidence.map { $0.isFinite && (0...1).contains($0) } ?? true) &&
                  !($0.modelEventId.map(blank) ?? false) && !$0.rallyIds.contains(where: blank) && Set($0.rallyIds).count == $0.rallyIds.count }) else {
            throw ProjectError.invalid("Invalid score tracking state, source bounds or marker IDs")
        }
    }
    public static func fromJSON(_ value: JSONValue, durationMs: Int64 = .max) throws -> Self {
        let result = try JSONDecoder().decode(Self.self, from: JSONEncoder().encode(value))
        try result.validate(durationMs: durationMs); return result
    }
    public func jsonValue() throws -> JSONValue { try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(self)) }
}

public struct DerivedScorePoint: Equatable, Sendable {
    public var serveMarkerId: String
    public var timestampMs: Int64
    public var servingSide: ServingSide
    public var winnerTeamId: ScoreTeamId?
    public var status: ScorePointStatus
    public var team1ScoreAfter: Int
    public var team2ScoreAfter: Int
}
public struct DerivedScore: Equatable, Sendable {
    public var team1Score: Int
    public var team2Score: Int
    public var servingTeamId: ScoreTeamId?
    public var servingSide: ServingSide?
    public var points: [DerivedScorePoint]
    public var ignoredPointCount: Int
    public var reviewPointCount: Int
    public func jsonValue() -> JSONValue {
        .object(["team1Score": .number(Double(team1Score)), "team2Score": .number(Double(team2Score)),
                 "servingTeamId": servingTeamId.map { .string($0.rawValue) } ?? .null,
                 "servingSide": servingSide.map { .string($0.rawValue) } ?? .null,
                 "ignoredPointCount": .number(Double(ignoredPointCount)), "reviewPointCount": .number(Double(reviewPointCount)),
                 "points": .array(points.map { .object([
                    "serveMarkerId": .string($0.serveMarkerId), "timestamp": .number(Double($0.timestampMs) / 1000),
                    "servingSide": .string($0.servingSide.rawValue), "winnerTeamId": $0.winnerTeamId.map { .string($0.rawValue) } ?? .null,
                    "status": .string($0.status.rawValue), "team1ScoreAfter": .number(Double($0.team1ScoreAfter)),
                    "team2ScoreAfter": .number(Double($0.team2ScoreAfter))]) })])
    }
}
public struct ScoreRallyRange: Equatable, Sendable {
    public var coreStartMs: Int64, coreEndMs: Int64, keepStartMs: Int64, keepEndMs: Int64
    public init(coreStartMs: Int64, coreEndMs: Int64, keepStartMs: Int64, keepEndMs: Int64) {
        self.coreStartMs = coreStartMs; self.coreEndMs = coreEndMs; self.keepStartMs = keepStartMs; self.keepEndMs = keepEndMs
    }
}
public typealias ScoreMergedRange = TimeRange

public enum ScoreReducer {
    static func serveOrder(_ a: ServeMarker, _ b: ServeMarker) -> Bool { (a.timestampMs, a.id) < (b.timestampMs, b.id) }
    static func switchOrder(_ a: SideSwitchMarker, _ b: SideSwitchMarker) -> Bool { (a.timestampMs, a.id) < (b.timestampMs, b.id) }
    public static func seedModelMarkers(_ current: ScoreTracking, output: ServingSideOutput?,
                                       sideSwitchOutput: SideSwitchOutput? = nil, sideSwitchEnabled: Bool = true) throws -> ScoreTracking {
        if output == nil && sideSwitchOutput == nil && sideSwitchEnabled { return current }
        var existingByRally: [String: ServeMarker] = [:]
        for marker in current.serveMarkers { if let id = marker.rallyId { existingByRally[id] = marker } }
        var tracking = current
        if output != nil { tracking.serveMarkers.removeAll { $0.origin != .manual } }
        for candidate in output?.candidates ?? [] {
            let id = "serve-\(candidate.id)", existing = existingByRally[candidate.id]
            let corrected = existing?.modelSide != nil && existing?.side != existing?.modelSide
            if candidate.verdict == .notServe && !corrected { continue }
            let modelSide: ServingSide = (candidate.verdict == .near || candidate.verdict == .far) ? candidate.side : .review
            if usedIds(tracking).contains(id) { continue }
            tracking.serveMarkers.append(.init(id: id, timestampMs: try scoreMilliseconds(candidate.anchor),
                side: corrected ? existing!.side : modelSide, origin: .model, modelSide: modelSide,
                ignorePreviousPoint: existing?.ignorePreviousPoint ?? false, rallyId: candidate.id))
            tracking.serveMarkers.sort(by: serveOrder)
        }
        if !sideSwitchEnabled { tracking.sideSwitchMarkers.removeAll { $0.origin != .manual } }
        else if let sideSwitchOutput {
            tracking.sideSwitchMarkers.removeAll { $0.origin != .manual }
            for candidate in sideSwitchOutput.candidates {
                let id = "switch-\(candidate.id)"
                if usedIds(tracking).contains(id) { continue }
                var rallyIds: [String] = []
                for source in candidate.sourceRangeIds where !rallyIds.contains(source) { rallyIds.append(source) }
                tracking.sideSwitchMarkers.append(.init(id: id, timestampMs: try scoreMilliseconds(candidate.timestamp),
                    origin: .model, modelConfidence: candidate.probability, modelEventId: candidate.id, rallyIds: rallyIds))
                tracking.sideSwitchMarkers.sort(by: switchOrder)
            }
        }
        try tracking.validate(); return tracking
    }
    public static func visibleTracking(_ tracking: ScoreTracking, ignoredIntervals: [IgnoredSourceInterval], excludedRallyIds: Set<String>) -> ScoreTracking {
        var value = tracking
        value.serveMarkers.removeAll { marker in ignoredIntervals.contains { marker.timestampMs >= $0.startMs && marker.timestampMs < $0.endMs }
            || (marker.rallyId.map { excludedRallyIds.contains($0) } ?? false) }
        value.sideSwitchMarkers.removeAll { marker in ignoredIntervals.contains { marker.timestampMs >= $0.startMs && marker.timestampMs < $0.endMs } }
        return value
    }
    public static func teamForServingSide(_ side: ServingSide, sideSwitchCount: Int) -> ScoreTeamId? {
        let switched = sideSwitchCount % 2 != 0
        switch side { case .near: return switched ? .team2 : .team1; case .far: return switched ? .team1 : .team2; case .review: return nil }
    }
    public static func deriveAt(_ tracking: ScoreTracking, sourceTimestampMs: Int64 = .max) -> DerivedScore {
        let serves = tracking.serveMarkers.sorted(by: serveOrder).filter { $0.timestampMs <= max(0, sourceTimestampMs) }
        let switches = tracking.sideSwitchMarkers.sorted(by: switchOrder)
        var switchIndex = 0
        var score = DerivedScore(team1Score: 0, team2Score: 0, points: [], ignoredPointCount: 0, reviewPointCount: 0)
        for (index, serve) in serves.enumerated() {
            while switchIndex < switches.count && switches[switchIndex].timestampMs <= serve.timestampMs { switchIndex += 1 }
            score.servingSide = serve.side; score.servingTeamId = teamForServingSide(serve.side, sideSwitchCount: switchIndex)
            if index == 0 { continue }
            let status: ScorePointStatus
            if serve.ignorePreviousPoint { status = .ignored; score.ignoredPointCount += 1 }
            else if score.servingTeamId == nil { status = .review; score.reviewPointCount += 1 }
            else { status = .counted; if score.servingTeamId == .team1 { score.team1Score += 1 } else { score.team2Score += 1 } }
            score.points.append(.init(serveMarkerId: serve.id, timestampMs: serve.timestampMs, servingSide: serve.side,
                                      winnerTeamId: score.servingTeamId, status: status,
                                      team1ScoreAfter: score.team1Score, team2ScoreAfter: score.team2Score))
        }
        return score
    }
    public static func scoreBoundaryTimestamp(_ playbackTimestampMs: Int64, rallyRanges: [ScoreRallyRange],
                                               tracking: ScoreTracking, mergedRanges: [ScoreMergedRange] = []) -> Int64 {
        let timestamp = max(0, playbackTimestampMs), serves = tracking.serveMarkers.sorted(by: serveOrder)
        guard let range = rallyRanges.filter({ timestamp >= $0.keepStartMs && timestamp < $0.coreStartMs }).min(by: { $0.coreStartMs < $1.coreStartMs }) else { return timestamp }
        let inPadding = serves.filter { $0.timestampMs >= range.keepStartMs && $0.timestampMs < range.coreStartMs }
        if inPadding.contains(where: { $0.timestampMs <= timestamp }) { return timestamp }
        return inPadding.first?.timestampMs ?? serves.first { $0.timestampMs >= timestamp && $0.timestampMs < range.coreEndMs }?.timestampMs ?? timestamp
    }
    private static func usedIds(_ tracking: ScoreTracking) -> Set<String> {
        Set(tracking.serveMarkers.map(\.id) + tracking.sideSwitchMarkers.map(\.id)).union(tracking.removedModelMarkerIds)
    }
    public static func nextMarkerId(prefix: String, tracking: ScoreTracking) -> String {
        let used = usedIds(tracking)
        for index in 1..<10_000 { let id = String(format: "%@%03d", prefix, index); if !used.contains(id) { return id } }
        return prefix + String(Int64(Date().timeIntervalSince1970 * 1000))
    }
    public static func addServe(_ tracking: ScoreTracking, timestampMs: Int64, side: ServingSide) -> ScoreTracking {
        guard timestampMs >= 0 else { return tracking }; var result = tracking
        result.serveMarkers.append(.init(id: nextMarkerId(prefix: "S", tracking: tracking), timestampMs: timestampMs, side: side))
        result.serveMarkers.sort(by: serveOrder); return result
    }
    public static func addSideSwitch(_ tracking: ScoreTracking, timestampMs: Int64) -> ScoreTracking {
        guard timestampMs >= 0 else { return tracking }; var result = tracking
        result.sideSwitchMarkers.append(.init(id: nextMarkerId(prefix: "X", tracking: tracking), timestampMs: timestampMs))
        result.sideSwitchMarkers.sort(by: switchOrder); return result
    }
    public static func removeServe(_ tracking: ScoreTracking, markerId: String) -> ScoreTracking {
        var result = tracking
        if tracking.serveMarkers.first(where: { $0.id == markerId })?.origin == .model { result.removedModelMarkerIds.insert(markerId) }
        result.serveMarkers.removeAll { $0.id == markerId }; return result
    }
    public static func removeSideSwitch(_ tracking: ScoreTracking, markerId: String) -> ScoreTracking {
        var result = tracking
        if tracking.sideSwitchMarkers.first(where: { $0.id == markerId })?.origin == .model { result.removedModelMarkerIds.insert(markerId) }
        result.sideSwitchMarkers.removeAll { $0.id == markerId }; return result
    }
}
