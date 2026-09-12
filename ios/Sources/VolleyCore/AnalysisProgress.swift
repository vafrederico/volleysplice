import Foundation

public enum AnalysisStep: String, Codable, CaseIterable, Sendable {
    case video, audio, rally, servingSide, sideSwitch
    public var label: String {
        switch self {
        case .video: return "Scanning video"
        case .audio: return "Listening for play"
        case .rally: return "Finding rallies"
        case .servingSide: return "Finding serve markers"
        case .sideSwitch: return "Finding team switches"
        }
    }
}

public struct AnalysisProgressEvent: Sendable {
    public var step: AnalysisStep
    public var fraction: Double
    public var detail: String
    public var completedFrames: Double?
    public var processedVideoSeconds: Double?
    public var uptime: Double
    public var resetsMeasurement: Bool
    public var failed: Bool
    public init(_ step: AnalysisStep, fraction: Double, detail: String, completedFrames: Double? = nil,
                processedVideoSeconds: Double? = nil, resetsMeasurement: Bool = false, failed: Bool = false,
                uptime: Double = ProcessInfo.processInfo.systemUptime) {
        self.step = step; self.fraction = fraction; self.detail = detail
        self.completedFrames = completedFrames; self.processedVideoSeconds = processedVideoSeconds; self.uptime = uptime
        self.resetsMeasurement = resetsMeasurement; self.failed = failed
    }
}

public struct AnalysisStepMeasurement: Codable, Equatable, Identifiable, Sendable {
    public enum Status: String, Codable, Sendable { case pending, running, complete, stopped }
    public var id: AnalysisStep
    public var status: Status = .pending
    public var fraction: Double = 0
    public var elapsedSeconds: Double = 0
    public var percentPerSecond: Double?
    public var framesPerSecond: Double?
    public var realtimeRatio: Double?
    public var etaSeconds: Double?
    public var detail = "Waiting for the previous step"
    public init(id: AnalysisStep) { self.id = id }
    private var rateText: String {
        var rate = "Measuring…"
        if let framesPerSecond {
            rate = String(format: "%.1f frames/s", locale: Locale(identifier: "en_US_POSIX"), framesPerSecond)
            if let realtimeRatio { rate += String(format: " · %.2fx realtime", locale: Locale(identifier: "en_US_POSIX"), realtimeRatio) }
        } else if let percentPerSecond { rate = String(format: "%.1f%%/s", locale: Locale(identifier: "en_US_POSIX"), percentPerSecond) }
        return rate
    }
    private var etaText: String { etaSeconds.map { "ETA \(Self.duration($0))" } ?? "estimating ETA" }
    /// Keep speed and ETA together in the system's compact notification subtitle.
    public var notificationMetrics: String { "\(rateText) · \(etaText)" }
    public var metrics: String {
        if status == .pending { return "Waiting" }
        if status == .stopped { return "Stopped after \(Self.duration(elapsedSeconds))" }
        let rate = rateText
        if status == .complete { return "\(Self.duration(elapsedSeconds)) elapsed" + (percentPerSecond != nil || framesPerSecond != nil ? " · \(rate)" : "") }
        return "\(Int((fraction * 100).rounded()))% · \(rate) · \(Self.duration(elapsedSeconds)) elapsed · " +
            etaText
    }
    private static func duration(_ seconds: Double) -> String {
        let rounded = Int(min(Double(Int.max / 2), max(0, seconds.isFinite ? seconds : 0)).rounded())
        return rounded >= 60 ? "\(rounded / 60)m \(rounded % 60)s" : "\(rounded)s"
    }
}

/// One monotonic clock per stage and per attempt. Cached/resumed work is not credited as new throughput.
public struct AnalysisProgressTracker: Sendable {
    private struct Timing: Sendable {
        var started: Double, updated: Double
        var baselineFraction: Double
        var baselineFrames: Double?, frames: Double?
        var baselineVideoSeconds: Double?, videoSeconds: Double?
    }
    private var rows: [AnalysisStepMeasurement]
    private var timing: [AnalysisStep: Timing] = [:]
    public init(steps: [AnalysisStep]) { rows = steps.map(AnalysisStepMeasurement.init) }
    public mutating func update(_ event: AnalysisProgressEvent) {
        guard event.uptime.isFinite, event.fraction.isFinite else { return }
        if !rows.contains(where: { $0.id == event.step }) { rows.append(.init(id: event.step)) }
        guard let index = rows.firstIndex(where: { $0.id == event.step }), rows[index].status != .complete,
              rows[index].status != .stopped else { return }
        if let previous = timing[event.step], previous.updated > event.uptime { return }
        // Advancing a stage closes earlier work, including cache hits with no decoder callbacks.
        for earlier in 0..<index where rows[earlier].status == .pending || rows[earlier].status == .running {
            finish(index: earlier, now: event.uptime, status: .complete)
        }
        let fraction = min(1, max(0, event.fraction))
        if timing[event.step] == nil || event.resetsMeasurement {
            timing[event.step] = Timing(started: event.uptime, updated: event.uptime, baselineFraction: fraction,
                baselineFrames: event.completedFrames, frames: event.completedFrames,
                baselineVideoSeconds: event.processedVideoSeconds, videoSeconds: event.processedVideoSeconds)
        }
        timing[event.step]?.updated = event.uptime
        if let value = event.completedFrames, value.isFinite { timing[event.step]?.frames = value }
        if let value = event.processedVideoSeconds, value.isFinite { timing[event.step]?.videoSeconds = value }
        rows[index].status = .running; rows[index].fraction = max(rows[index].fraction, fraction); rows[index].detail = event.detail
        if event.failed { finish(index: index, now: event.uptime, status: .stopped) }
        else if fraction >= 1 { finish(index: index, now: event.uptime, status: .complete) }
    }
    public mutating func stop(now: Double) {
        for index in rows.indices where rows[index].status == .running { finish(index: index, now: now, status: .stopped) }
    }
    public func snapshot(now: Double) -> [AnalysisStepMeasurement] {
        rows.map { row in row.status == .running ? measured(row, now: now) : row }
    }
    private mutating func finish(index: Int, now: Double, status: AnalysisStepMeasurement.Status) {
        rows[index] = measured(rows[index], now: now)
        rows[index].status = status
        if status == .complete { rows[index].fraction = 1; rows[index].etaSeconds = nil }
    }
    private func measured(_ row: AnalysisStepMeasurement, now: Double) -> AnalysisStepMeasurement {
        guard let clock = timing[row.id] else { return row }
        var result = row
        result.elapsedSeconds = max(0, (now.isFinite ? now : clock.updated) - clock.started)
        let progressed = max(0, row.fraction - clock.baselineFraction)
        if result.elapsedSeconds >= 0.5, progressed > 0 {
            result.percentPerSecond = progressed * 100 / result.elapsedSeconds
            // Android waits for three percent of measured progress before estimating.
            if progressed >= 0.03 {
                let eta = result.elapsedSeconds * (1 - row.fraction) / progressed
                if eta.isFinite { result.etaSeconds = eta }
            }
        }
        if result.elapsedSeconds >= 0.5, let end = clock.frames, let start = clock.baselineFrames, end > start {
            let rate = (end - start) / result.elapsedSeconds
            if rate.isFinite { result.framesPerSecond = rate }
        }
        if result.elapsedSeconds >= 0.5, let end = clock.videoSeconds, let start = clock.baselineVideoSeconds, end > start {
            let rate = (end - start) / result.elapsedSeconds
            if rate.isFinite { result.realtimeRatio = rate }
        }
        return result
    }
}
