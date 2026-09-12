import Foundation

/// Losslessly retains optional model and scoring payloads across project saves.
public enum JSONValue: Codable, Equatable, Sendable {
    case object([String: JSONValue]), array([JSONValue]), string(String), number(Double), bool(Bool), null
    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let v = try? c.decode(Bool.self) { self = .bool(v) }
        else if let v = try? c.decode(Double.self) { self = .number(v) }
        else if let v = try? c.decode(String.self) { self = .string(v) }
        else if let v = try? c.decode([JSONValue].self) { self = .array(v) }
        else { self = .object(try c.decode([String: JSONValue].self)) }
    }
    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .object(let v): try c.encode(v)
        case .array(let v): try c.encode(v)
        case .string(let v): try c.encode(v)
        case .number(let v): try c.encode(v)
        case .bool(let v): try c.encode(v)
        case .null: try c.encodeNil()
        }
    }
    public subscript(_ key: String) -> JSONValue? {
        get { if case .object(let o) = self { return o[key] }; return nil }
        set { if case .object(var o) = self { o[key] = newValue; self = .object(o) } }
    }
    public var double: Double? { if case .number(let v) = self { return v }; return nil }
    public var string: String? { if case .string(let v) = self { return v }; return nil }
    public var array: [JSONValue]? { if case .array(let v) = self { return v }; return nil }
    public var bool: Bool? { if case .bool(let v) = self { return v }; return nil }
}

public enum ProjectError: Error, LocalizedError {
    case invalid(String)
    public var errorDescription: String? { switch self { case .invalid(let message): return message } }
}

public struct TimeRange: Codable, Equatable, Sendable {
    public var startMs: Int64
    public var endMs: Int64
    public init(startMs: Int64, endMs: Int64) { self.startMs = startMs; self.endMs = endMs }
}
public enum CutOrigin: String, Codable, Sendable { case inferred = "cached-label", manual }
public struct EditableCut: Codable, Equatable, Identifiable, Sendable {
    public var id: String
    public var coreStartMs: Int64
    public var coreEndMs: Int64
    public var keepStartMs: Int64
    public var keepEndMs: Int64
    public var confidence: Float
    public var included: Bool
    public var origin: CutOrigin
    public var agreement: String?
    public init(id: String, coreStartMs: Int64, coreEndMs: Int64, keepStartMs: Int64? = nil,
                keepEndMs: Int64? = nil, confidence: Float = 1, included: Bool = true,
                origin: CutOrigin = .inferred, agreement: String? = nil) {
        self.id = id; self.coreStartMs = coreStartMs; self.coreEndMs = coreEndMs
        self.keepStartMs = keepStartMs ?? coreStartMs; self.keepEndMs = keepEndMs ?? coreEndMs
        self.confidence = confidence; self.included = included; self.origin = origin; self.agreement = agreement
    }
}
public struct IgnoredSourceInterval: Codable, Equatable, Identifiable, Sendable {
    public var id: String
    public var startMs: Int64
    public var endMs: Int64
    public var reason: String
    public init(id: String, startMs: Int64, endMs: Int64, reason: String = "non-game-content") {
        self.id = id; self.startMs = startMs; self.endMs = endMs; self.reason = reason
    }
}
public struct EditorDraft: Codable, Equatable, Sendable {
    public var version = 8
    public var sourceRevision: String
    public var updatedAtMs: Int64 = 0
    public var beforePaddingMs: Int64 = 2_000
    public var afterPaddingMs: Int64 = 2_000
    public var joinGapMs: Int64 = 3_000
    public var pendingManualStartMs: Int64?
    public var pendingIgnoreStartMs: Int64?
    public var ignoreReason = "non-game-content"
    public var finalPreviewEnabled = false
    public var playbackRate: Float = 1
    public var confidenceReviewThreshold: Float = 0.7
    public var reviewedCutIds: Set<String> = []
    public var cuts: [EditableCut]
    public var ignoredIntervals: [IgnoredSourceInterval] = []
    public var selectedSuppressionPolicy = "aggressive"
    public var suppressionInitialBehavior = "disable-initially"
    public var suppressionDecisionOverrides: [String: String] = [:]
    public var suppressionScopeOverrides: [String: String] = [:]
    public var userTouchedCutIds: Set<String> = []
    public var suppressionContractVersion = "suppression-policy-v1"
    public var scoreTracking: JSONValue = .object([
        "version": .number(1), "enabled": .bool(false), "team1Name": .string("Team 1"),
        "team2Name": .string("Team 2"), "serveMarkers": .array([]), "sideSwitchMarkers": .array([]),
        "removedModelMarkerIds": .array([])
    ])
    public var renderScoreOverlay = true
    public var renderScoreTimeline = true
    /// Nil keeps the score-aware defaults for projects saved before chapter preferences existed.
    public var chapterOptions: YouTubeChapterOptions?
    public init(sourceRevision: String, cuts: [EditableCut] = []) { self.sourceRevision = sourceRevision; self.cuts = cuts }
    public func validate(durationMs: Int64) throws {
        guard durationMs > 0, version == 8, (0...10_000).contains(beforePaddingMs),
              (0...10_000).contains(afterPaddingMs), (0...10_000).contains(joinGapMs),
              Set(cuts.map(\.id)).count == cuts.count,
              cuts.allSatisfy({ !$0.id.isEmpty && $0.keepStartMs >= 0 && $0.keepStartMs <= $0.coreStartMs &&
                  $0.coreStartMs < $0.coreEndMs && $0.coreEndMs <= $0.keepEndMs && $0.keepEndMs <= durationMs &&
                  $0.confidence.isFinite && (0...1).contains($0.confidence) &&
                  ($0.agreement == nil || ["both-models", "all-labels-v2-only", "previous-production-only"].contains($0.agreement!)) }),
              Set(ignoredIntervals.map(\.id)).count == ignoredIntervals.count,
              ignoredIntervals.allSatisfy({ !$0.id.isEmpty && $0.startMs >= 0 && $0.endMs > $0.startMs && $0.endMs <= durationMs }),
              userTouchedCutIds.isSubset(of: Set(cuts.map(\.id))),
              ["none", "conservative", "balanced", "aggressive"].contains(selectedSuppressionPolicy),
              ["highlight-only", "disable-initially"].contains(suppressionInitialBehavior),
              suppressionDecisionOverrides.values.allSatisfy({ ["keep", "suppress"].contains($0) }),
              suppressionScopeOverrides.values.allSatisfy({ ["whole-rally", "veto-region"].contains($0) }) else {
            throw ProjectError.invalid("Invalid editor draft or source-time bounds")
        }
        _ = try ScoreTracking.fromJSON(scoreTracking, durationMs: durationMs)
    }
}

/// Local schema is separate from the interoperable feedback v3 schema.
/// Raw media is referenced by a security-scoped bookmark, never embedded in feedback.
public struct ProjectDocument: Codable, Equatable, Sendable {
    public var schema = "volleycut-ios-project"
    public var schemaVersion = 1
    public var id: String
    public var sourceName: String
    public var durationMs: Int64
    public var gameWindow: TimeRange
    public var sourceBookmark: Data?
    public var draft: EditorDraft
    public var feedback: JSONValue?
    /// Local persistence generation; never copied into feedback interchange.
    public var persistenceRevision: String?
    public init(id: String = UUID().uuidString, sourceName: String, durationMs: Int64,
                gameWindow: TimeRange? = nil, draft: EditorDraft, sourceBookmark: Data? = nil, feedback: JSONValue? = nil) {
        self.id = id; self.sourceName = sourceName; self.durationMs = durationMs
        self.gameWindow = gameWindow ?? TimeRange(startMs: 0, endMs: durationMs)
        self.draft = draft; self.sourceBookmark = sourceBookmark; self.feedback = feedback
    }
    public func validate() throws {
        guard schema == "volleycut-ios-project", schemaVersion == 1, !id.isEmpty,
              gameWindow.startMs >= 0, gameWindow.endMs <= durationMs,
              gameWindow.endMs > gameWindow.startMs else { throw ProjectError.invalid("Invalid project schema or game window") }
        try draft.validate(durationMs: durationMs)
    }
    public func save(to url: URL) throws {
        try validate()
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(self).write(to: url, options: .atomic)
    }
    public static func load(from url: URL) throws -> Self {
        try ProjectPersistence.read(from: url).project
    }
}
