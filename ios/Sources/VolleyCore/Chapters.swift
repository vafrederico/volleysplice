import Foundation

public struct EditableRallyGroup: Equatable, Sendable {
    public var cuts: [EditableCut]
    public init(cuts: [EditableCut]) { precondition(!cuts.isEmpty); self.cuts = cuts }
    public var id: String { cuts[0].id }
    public var cutIds: Set<String> { Set(cuts.map(\.id)) }
    public var keepStartMs: Int64 { cuts.map(\.keepStartMs).min()! }
    public var keepEndMs: Int64 { cuts.map(\.keepEndMs).max()! }
    public var coreStartMs: Int64 { cuts.map(\.coreStartMs).min()! }
    public var coreEndMs: Int64 { cuts.map(\.coreEndMs).max()! }
    public func asEditableCut() -> EditableCut {
        var cut = cuts[0]
        cut.coreStartMs = coreStartMs; cut.coreEndMs = coreEndMs
        cut.keepStartMs = keepStartMs; cut.keepEndMs = keepEndMs
        cut.confidence = cuts.map(\.confidence).min()!; cut.included = cuts.allSatisfy(\.included)
        cut.origin = cuts.allSatisfy { $0.origin == .manual } ? .manual : .inferred
        return cut
    }
}
extension EditorMath {
    /// A serve on the next cut splits otherwise joined fragments into distinct rallies.
    public static func editableRallyGroups(cuts: [EditableCut], intervals: [FinalCutInterval],
                                          serveMarkers: [ServeMarker], hardBoundaryCutIds: Set<String> = []) -> [EditableRallyGroup] {
        var indexes: [String: Set<Int>] = [:]
        for cut in cuts { indexes[cut.id] = Set(intervals.indices.filter { intervals[$0].cutIds.contains(cut.id) }) }
        var groups: [EditableRallyGroup] = []
        for cut in cuts.sorted(by: { ($0.keepStartMs, $0.keepEndMs) < ($1.keepStartMs, $1.keepEndMs) }) {
            guard let previous = groups.last else { groups.append(.init(cuts: [cut])); continue }
            let previousIndexes = previous.cutIds.reduce(into: Set<Int>()) { $0.formUnion(indexes[$1] ?? []) }
            let sharesInterval = !previousIndexes.isDisjoint(with: indexes[cut.id] ?? [])
            let separatedByServe = serveMarkers.contains {
                $0.rallyId == cut.id || ($0.rallyId == nil && $0.timestampMs >= cut.keepStartMs && $0.timestampMs < cut.coreEndMs)
            }
            let hardBoundary = hardBoundaryCutIds.contains(cut.id) || !previous.cutIds.isDisjoint(with: hardBoundaryCutIds)
            if !sharesInterval || separatedByServe || hardBoundary { groups.append(.init(cuts: [cut])) }
            else { groups[groups.count - 1].cuts.append(cut) }
        }
        return groups
    }
}

public struct YouTubeChapterOptions: Codable, Equatable, Sendable {
    public var includeRallyNumber: Bool
    public var includeServeNumber: Bool
    public var includeScore: Bool
    public var includeServingTeam: Bool
    public var includeSideSwitches: Bool
    public var includeCredit: Bool
    public init(includeRallyNumber: Bool, includeServeNumber: Bool, includeScore: Bool,
                includeServingTeam: Bool, includeSideSwitches: Bool, includeCredit: Bool = true) {
        self.includeCredit = includeCredit
        self.includeRallyNumber = includeRallyNumber; self.includeServeNumber = includeServeNumber
        self.includeScore = includeScore; self.includeServingTeam = includeServingTeam; self.includeSideSwitches = includeSideSwitches
    }
    private enum CodingKeys: String, CodingKey {
        case includeRallyNumber, includeServeNumber, includeScore, includeServingTeam, includeSideSwitches, includeCredit
    }
    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        includeRallyNumber = try values.decode(Bool.self, forKey: .includeRallyNumber)
        includeServeNumber = try values.decode(Bool.self, forKey: .includeServeNumber)
        includeScore = try values.decode(Bool.self, forKey: .includeScore)
        includeServingTeam = try values.decode(Bool.self, forKey: .includeServingTeam)
        includeSideSwitches = try values.decode(Bool.self, forKey: .includeSideSwitches)
        includeCredit = try values.decodeIfPresent(Bool.self, forKey: .includeCredit) ?? true
    }
}
public struct YouTubeChapter: Codable, Equatable, Sendable {
    public enum Kind: String, Codable, Sendable { case rally, sideSwitch = "side-switch" }
    public var kind: Kind
    public var outputMs: Int64
    public var sourceTimestampMs: Int64
    public var title: String
}
public enum YouTubeChapters {
    public static func defaultOptions(hasScoreTracking: Bool, hasSideSwitches: Bool) -> YouTubeChapterOptions {
        .init(includeRallyNumber: !hasScoreTracking, includeServeNumber: false, includeScore: hasScoreTracking,
              includeServingTeam: hasScoreTracking, includeSideSwitches: hasSideSwitches)
    }
    private struct OutputInterval { var interval: FinalCutInterval; var outputStartMs: Int64 }
    private struct VisibleRally { var rally: EditableRallyGroup; var outputMs: Int64; var sourceTimestampMs: Int64 }
    public static func build(intervals: [FinalCutInterval], cuts: [EditableCut], scoreTracking: ScoreTracking?,
                             options: YouTubeChapterOptions) -> [YouTubeChapter] {
        var outputStartMs: Int64 = 0
        let outputIntervals = intervals.map { interval in
            defer { outputStartMs += max(0, interval.endMs - interval.startMs) }
            return OutputInterval(interval: interval, outputStartMs: outputStartMs)
        }
        let markers = (scoreTracking?.serveMarkers ?? []).sorted(by: ScoreReducer.serveOrder)
        let visible = EditorMath.editableRallyGroups(cuts: cuts, intervals: intervals, serveMarkers: markers).compactMap { rally -> VisibleRally? in
            guard let output = outputIntervals.first(where: { !rally.cutIds.isDisjoint(with: $0.interval.cutIds) }) else { return nil }
            let source = min(max(rally.keepStartMs, output.interval.startMs), output.interval.endMs)
            return VisibleRally(rally: rally, outputMs: output.outputStartMs + source - output.interval.startMs, sourceTimestampMs: source)
        }.sorted { ($0.outputMs, $0.rally.coreStartMs, $0.rally.id) < ($1.outputMs, $1.rally.coreStartMs, $1.rally.id) }
        var serveNumbers: [String: Int] = [:], redo: Set<String> = [], claimed: Set<String> = []
        for (index, marker) in markers.enumerated() {
            serveNumbers[marker.id] = index + 1
            if index + 1 < markers.count && markers[index + 1].ignorePreviousPoint { redo.insert(marker.id) }
        }
        var chapters = visible.enumerated().compactMap { index, visible -> YouTubeChapter? in
            let marker = markerForRally(visible.rally, markers: markers, claimed: claimed)
            if scoreTracking?.enabled == true && marker == nil { return nil }
            if let marker { claimed.insert(marker.id) }
            return YouTubeChapter(kind: .rally, outputMs: visible.outputMs, sourceTimestampMs: visible.sourceTimestampMs,
                title: rallyTitle(number: index + 1, marker: marker, isRedo: marker.map { redo.contains($0.id) } ?? false,
                                  serveNumbers: serveNumbers, tracking: scoreTracking, options: options))
        }
        if options.includeSideSwitches, let scoreTracking {
            let points = scoreTracking.sideSwitchMarkers.compactMap { marker -> (Int64, Int64)? in
                if let containing = outputIntervals.first(where: { marker.timestampMs >= $0.interval.startMs && marker.timestampMs < $0.interval.endMs }) {
                    return (containing.outputStartMs + marker.timestampMs - containing.interval.startMs, marker.timestampMs)
                }
                guard let next = outputIntervals.first(where: { $0.interval.startMs > marker.timestampMs }) else { return nil }
                return (next.outputStartMs, next.interval.startMs)
            }
            chapters += points.enumerated().map { index, point in
                YouTubeChapter(kind: .sideSwitch, outputMs: point.0, sourceTimestampMs: point.1, title: "Side switch \(index + 1)")
            }
        }
        chapters.sort { ($0.outputMs, $0.kind == .rally ? 0 : 1) < ($1.outputMs, $1.kind == .rally ? 0 : 1) }
        var merged: [YouTubeChapter] = []
        for var chapter in chapters {
            let wholeSecondMs = max(0, chapter.outputMs / 1000) * 1000
            if var previous = merged.last, previous.outputMs == wholeSecondMs {
                if !previous.title.components(separatedBy: " / ").contains(chapter.title) {
                    previous.title += " / " + chapter.title
                    merged[merged.count - 1] = previous
                }
            } else { chapter.outputMs = wholeSecondMs; merged.append(chapter) }
        }
        return merged
    }
    private static func markerForRally(_ rally: EditableRallyGroup, markers: [ServeMarker], claimed: Set<String>) -> ServeMarker? {
        let available = markers.filter { !claimed.contains($0.id) }
        func order(_ a: ServeMarker, _ b: ServeMarker) -> Bool {
            (abs(a.timestampMs - rally.coreStartMs), a.timestampMs, a.id) < (abs(b.timestampMs - rally.coreStartMs), b.timestampMs, b.id)
        }
        if let marker = available.filter({ $0.rallyId.map { rally.cutIds.contains($0) } ?? false }).min(by: order) { return marker }
        let core = available.filter { $0.timestampMs >= rally.coreStartMs && $0.timestampMs < rally.coreEndMs }
        let kept = available.filter { $0.timestampMs >= rally.keepStartMs && $0.timestampMs < rally.keepEndMs }
        return (core.isEmpty ? kept : core).min(by: order)
    }
    private static func rallyTitle(number: Int, marker: ServeMarker?, isRedo: Bool, serveNumbers: [String: Int],
                                   tracking: ScoreTracking?, options: YouTubeChapterOptions) -> String {
        var parts: [String] = []
        if options.includeRallyNumber { parts.append("Rally \(number)") }
        if options.includeServeNumber, let marker, let serveNumber = serveNumbers[marker.id] { parts.append("Serve \(serveNumber)") }
        if let marker, let tracking, options.includeScore || options.includeServingTeam {
            let score = ScoreReducer.deriveAt(tracking, sourceTimestampMs: marker.timestampMs)
            if options.includeScore { parts.append("\(score.team1Score)–\(score.team2Score)") }
            if options.includeServingTeam, let team = score.servingTeamId {
                parts.append("\(team == .team1 ? tracking.team1Name : tracking.team2Name) serving")
            }
        }
        if isRedo { parts.append("Re-do") }
        return (parts.isEmpty ? ["Rally \(number)"] : parts).joined(separator: " - ")
    }
    public static func text(_ chapters: [YouTubeChapter], includeCredit: Bool = true) -> String {
        let text = chapters.map { "\(formatTimestamp($0.outputMs)) \($0.title)" }.joined(separator: "\n")
        return !text.isEmpty && includeCredit ? "Edited with https://volleysplice.com\n\n" + text : text
    }
    public static func filename(_ sourceFilename: String) -> String {
        let base = sourceFilename.lastIndex(of: ".").map { String(sourceFilename[..<$0]) } ?? sourceFilename
        let sanitized = base.replacingOccurrences(of: "[^A-Za-z0-9._-]+", with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return "\(sanitized.isEmpty ? "volleysplice" : sanitized)-youtube-chapters.txt"
    }
    public static func formatTimestamp(_ milliseconds: Int64) -> String {
        let seconds = max(0, milliseconds) / 1000, hours = seconds / 3600, minutes = seconds % 3600 / 60
        if hours > 0 { return String(format: "%lld:%02lld:%02lld", hours, minutes, seconds % 60) }
        return String(format: "%lld:%02lld", minutes, seconds % 60)
    }
}
