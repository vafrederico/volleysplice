import Foundation

public struct ScorePointTimelineEntry: Equatable, Sendable {
    public let serveMarkerId: String
    public let winnerTeamId: ScoreTeamId
    public let teamPointNumber: Int
}

public struct ScoreOverlaySnapshot: Equatable, Sendable {
    public let team1Name: String
    public let team2Name: String
    public let score: DerivedScore
    public let points: [ScorePointTimelineEntry]
    public let revealTimestampMs: Int64?
    public let opacity: Float
    public let effectiveTimestampMs: Int64
    public let currentServeId: String?
}

/// Prepare when the draft changes, then binary-search on each playback/frame tick.
/// Inputs are already filtered with ScoreReducer.visibleTracking, as in Android.
public struct PreparedScoreOverlay: Sendable {
    private struct State: Sendable {
        let startMs: Int64
        let score: DerivedScore
        let points: [ScorePointTimelineEntry]
        let revealMs: Int64?
        let boundaryMs: Int64
        let currentServeId: String?
    }
    public let tracking: ScoreTracking
    public let changeTimestamps: [Int64]
    private let states: [State]

    public init(tracking: ScoreTracking, rallyRanges: [ScoreRallyRange], mergedRanges: [ScoreMergedRange] = []) {
        self.tracking = tracking
        let changes = Set([Int64(0)] + tracking.serveMarkers.map(\.timestampMs) +
            rallyRanges.flatMap { [$0.keepStartMs, $0.coreStartMs, $0.coreEndMs, $0.keepEndMs] }).filter { $0 >= 0 }.sorted()
        changeTimestamps = changes
        states = changes.map { timestamp in
            let boundary = ScoreReducer.scoreBoundaryTimestamp(timestamp, rallyRanges: rallyRanges, tracking: tracking, mergedRanges: mergedRanges)
            let score = ScoreReducer.deriveAt(tracking, sourceTimestampMs: boundary)
            let counted = score.points.filter { $0.status == .counted && $0.winnerTeamId != nil }
            return State(startMs: timestamp, score: score, points: counted.map {
                .init(serveMarkerId: $0.serveMarkerId, winnerTeamId: $0.winnerTeamId!,
                      teamPointNumber: $0.winnerTeamId == .team1 ? $0.team1ScoreAfter : $0.team2ScoreAfter)
            }, revealMs: counted.last.map { ExportTimeline.pointRevealTimestamp($0.timestampMs, tracking: tracking, rallyRanges: rallyRanges) },
                boundaryMs: boundary, currentServeId: tracking.serveMarkers.filter { $0.timestampMs <= boundary }.max {
                    ($0.timestampMs, $0.id) < ($1.timestampMs, $1.id)
                }?.id)
        }
    }

    public func snapshot(at sourceTimestampMs: Int64) -> ScoreOverlaySnapshot {
        let timestamp = max(0, sourceTimestampMs)
        var lower = 0, upper = states.count
        while lower < upper {
            let middle = (lower + upper) / 2
            if states[middle].startMs <= timestamp { lower = middle + 1 } else { upper = middle }
        }
        let state = states[max(0, lower - 1)]
        return .init(team1Name: tracking.team1Name, team2Name: tracking.team2Name, score: state.score,
            points: state.points, revealTimestampMs: state.revealMs,
            opacity: state.revealMs.map { ExportTimeline.pointTimelineOpacity(sourceTimestampMs: sourceTimestampMs, revealTimestampMs: $0) } ?? 0,
            effectiveTimestampMs: state.boundaryMs > state.startMs ? state.boundaryMs : timestamp,
            currentServeId: state.currentServeId)
    }
}

public struct ScoreOverlayLayout: Equatable, Sendable {
    public let width: Int, height: Int, team1Width: Int, team2Width: Int, scoreWidth: Int
    public let borderWidth: Int, radius: Int, fontSize: Int, horizontalPadding: Int
}

public struct ScorePointTimelineLayout: Equatable, Sendable {
    public let startX: Float, team1CenterY: Float, team2CenterY: Float
    public let columnSpacing: Float, circleRadius: Float, lineWidth: Float, fontSize: Float
}

/// Geometry, palette and labels from Android ScoreOverlay (ScoreTrackingModels.kt).
public enum ScoreOverlay {
    public static let team1RGB: UInt32 = 0xd9342b
    public static let team2RGB: UInt32 = 0x2367c9
    public static func formatScore(_ value: Int) -> String {
        let text = String(max(0, value)); return text.count < 2 ? "0" + text : text
    }
    public static func formatTeamLabel(_ name: String, serving: Bool) -> String { serving ? name + " \u{1F3D0}" : name }
    public static func height(videoWidth: Int, videoHeight: Int) -> Int {
        Int(min(76, max(36, Double(max(1, min(videoWidth, videoHeight))) * 0.064)).rounded())
    }
    public static func layout(videoWidth: Int, videoHeight: Int, snapshot: ScoreOverlaySnapshot,
                              measureText: (String, Int) -> Double) -> ScoreOverlayLayout {
        let height = height(videoWidth: videoWidth, videoHeight: videoHeight)
        let border = min(videoWidth, videoHeight) >= 720 ? 2 : 1
        let font = Int((Double(height) * 0.39).rounded()), padding = Int((Double(height) * 0.24).rounded())
        let scoreWidth = Int((Double(height) * 1.3).rounded()), minimum = Int((Double(height) * 2.25).rounded())
        func measured(_ name: String, _ team: ScoreTeamId) -> Int {
            max(minimum, Int(ceil(measureText(formatTeamLabel(name, serving: snapshot.score.servingTeamId == team), font))) + padding * 2)
        }
        var first = measured(snapshot.team1Name, .team1), second = measured(snapshot.team2Name, .team2)
        let available = max(minimum * 2 + scoreWidth * 2, Int(floor(Double(videoWidth) * 0.96))) - scoreWidth * 2
        if first + second > available {
            let flexible = first + second - minimum * 2
            let scale = flexible > 0 ? min(1, Double(max(0, available - minimum * 2)) / Double(flexible)) : 0
            first = Int((Double(minimum) + Double(first - minimum) * scale).rounded())
            second = Int((Double(minimum) + Double(second - minimum) * scale).rounded())
        }
        return .init(width: first + second + scoreWidth * 2, height: height, team1Width: first, team2Width: second,
                     scoreWidth: scoreWidth, borderWidth: border, radius: padding, fontSize: font, horizontalPadding: padding)
    }
    public static func visiblePoints(videoWidth: Int, layout: ScoreOverlayLayout, points: [ScorePointTimelineEntry]) -> [ScorePointTimelineEntry] {
        let available = max(0, videoWidth - layout.width)
        guard available > 0 else { return [] }
        return Array(points.suffix(max(1, Int(Float(available) / (Float(layout.height) * 0.58)))))
    }
    public static func pointLayout(videoWidth: Int, videoHeight: Int, score: ScoreOverlayLayout, pointCount: Int) -> ScorePointTimelineLayout {
        let available = Float(max(0, videoWidth - score.width)), height = Float(score.height)
        let spacing: Float = pointCount > 0 && available > 0 ? min(height * 0.58, available) : 0
        return .init(startX: Float(score.width), team1CenterY: height * 0.28, team2CenterY: height * 0.72,
                     columnSpacing: spacing, circleRadius: min(height * 0.18, spacing * 0.36, Float(videoHeight) * 0.04),
                     lineWidth: max(2, Float(score.borderWidth) * 1.5), fontSize: max(1, min(height * 0.18, spacing * 0.52)))
    }
}
