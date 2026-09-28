import Foundation

/// A selection identifies a complete encoder, temporal head, scaler and decoder bundle.
/// Existing projects retain their saved selection; this default applies to new analyses.
public enum RallyModel: String, Codable, CaseIterable, Sendable {
    case balanced = "distilled-large-f1-v1"
    case maximumCoverage = "distilled-large-recall-v1"
    case legacy = "ensemble"

    public static let `default`: RallyModel = .maximumCoverage
    public var isNeural: Bool { self != .legacy }
    public var modelId: String { self == .legacy ? ProjectArchive.modelId : rawValue }
    public var displayName: String {
        switch self {
        case .balanced: return "Balanced · BETA"
        case .maximumCoverage: return "Maximum coverage · BETA"
        case .legacy: return "Legacy model"
        }
    }
    public var detail: String {
        switch self {
        case .balanced: return "Balances retained play and extra footage with tighter cuts."
        case .maximumCoverage: return "Keeps more possible play, with more extra footage to review."
        case .legacy: return "Uses the previous production detector."
        }
    }
    public var variantKey: String? {
        switch self {
        case .balanced: return "high-f1"
        case .maximumCoverage: return "high-recall"
        case .legacy: return nil
        }
    }
    public var assetDirectory: String? {
        switch self {
        case .balanced: return "f1"
        case .maximumCoverage: return "recall"
        case .legacy: return nil
        }
    }
    public init?(modelId: String) {
        if modelId == ProjectArchive.modelId { self = .legacy }
        else if let model = Self(rawValue: modelId) { self = model }
        else { return nil }
    }
}
