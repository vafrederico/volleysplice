import Foundation

public enum ProcessingJobKind: String, Codable, Sendable {
    case analysis, scores, videoExport
    public var supportsCPUContinuation: Bool { self != .videoExport }
}
public enum ProcessingJobState: String, Codable, Sendable {
    case queued, running, cancelling, interrupted, cancelled, failed, completed
    public var isActive: Bool { self == .running || self == .cancelling }
    public var canResume: Bool { self == .interrupted || self == .failed || self == .cancelled }
}
public enum ProcessingStopReason: String, Codable, Sendable { case userCancelled, backgrounded, systemExpired }
public struct VideoExportDestination: Codable, Equatable, Sendable {
    public enum Kind: String, Codable, Sendable { case photos, files }
    public var kind: Kind
    public var folderBookmark: Data?
    public init(kind: Kind, folderBookmark: Data? = nil) {
        self.kind = kind; self.folderBookmark = folderBookmark
    }
}
public struct ProcessingAnalysisSettings: Codable, Equatable, Sendable {
    public var startMs: Int64, endMs: Int64
    public var roi: [Double]
    public var prepareScores: Bool
    public var generateSideSwitchMarkers: Bool?
    public init(startMs: Int64, endMs: Int64, roi: [Double], prepareScores: Bool, generateSideSwitchMarkers: Bool = false) {
        self.startMs = startMs; self.endMs = endMs; self.roi = roi; self.prepareScores = prepareScores; self.generateSideSwitchMarkers = generateSideSwitchMarkers
    }
}
/// Job metadata stays small. Immutable project snapshots are stored separately by job ID.
public struct ProcessingJob: Codable, Equatable, Identifiable, Sendable {
    public var id: String
    public var kind: ProcessingJobKind
    public var projectId: String
    public var sourceName: String
    public var sourceFingerprint: String
    public var analysis: ProcessingAnalysisSettings?
    public var videoDestination: VideoExportDestination?
    public var stageSeconds: [String: Double]?
    public var analysisSteps: [AnalysisStepMeasurement]?
    public var state: ProcessingJobState = .queued
    public var attempt = 0
    public var runToken: String?
    public var stopReason: ProcessingStopReason?
    public var progress: Double = 0
    public var detail = "Waiting to start"
    public var startedAtMs: Int64?
    public var createdAtMs: Int64
    public var updatedAtMs: Int64
    public var outputNames: [String] = []
    /// Persist intent until an active worker drains; old ledgers decode without this key.
    public var removalRequested: Bool?
    public init(id: String = UUID().uuidString, kind: ProcessingJobKind, projectId: String,
                sourceName: String, sourceFingerprint: String, analysis: ProcessingAnalysisSettings? = nil,
                nowMs: Int64 = Int64(Date().timeIntervalSince1970 * 1000)) {
        self.id = id; self.kind = kind; self.projectId = projectId; self.sourceName = sourceName
        self.sourceFingerprint = sourceFingerprint; self.analysis = analysis
        self.createdAtMs = nowMs; self.updatedAtMs = nowMs
    }
    public static func isFilename(_ name: String) -> Bool {
        !name.isEmpty && name != "." && name != ".." && !name.contains("/") && !name.contains("\\") && !name.contains("\0")
    }
    public func validate() throws {
        guard UUID(uuidString: id) != nil, !projectId.isEmpty, Self.isFilename(sourceName),
              sourceFingerprint.hasPrefix("sampled-sha256-v1:"), progress.isFinite, (0...1).contains(progress),
              attempt >= 0, outputNames.allSatisfy(Self.isFilename),
              !state.isActive || runToken != nil else { throw ProjectError.invalid("Invalid processing job") }
        if kind == .analysis {
            guard let settings = analysis, settings.startMs >= 0, settings.endMs > settings.startMs,
                  settings.roi.count == 4, settings.roi.allSatisfy(\.isFinite),
                  settings.roi[0] >= 0, settings.roi[1] >= 0, settings.roi[2] > 0, settings.roi[3] > 0,
                  settings.roi[0] + settings.roi[2] <= 1.000001,
                  settings.roi[1] + settings.roi[3] <= 1.000001 else { throw ProjectError.invalid("Invalid queued analysis settings") }
        }
    }
}
public struct ProcessingJobLedger: Codable, Equatable, Sendable {
    public var schemaVersion = 1
    public var jobs: [ProcessingJob] = []
    /// Completed receipts remain durable, but only work left to do belongs in the queue UI.
    public var unfinishedJobs: [ProcessingJob] { jobs.filter { $0.state != .completed } }
    public init() {}
    public func validate() throws {
        guard schemaVersion == 1, Set(jobs.map(\.id)).count == jobs.count,
              jobs.filter({ $0.state.isActive }).count <= 1 else { throw ProjectError.invalid("Invalid serial processing queue") }
        for job in jobs { try job.validate() }
    }
    public mutating func enqueue(_ job: ProcessingJob) throws {
        try job.validate()
        guard !jobs.contains(where: { $0.id == job.id || ($0.projectId == job.projectId && $0.kind == job.kind &&
              ($0.state == .queued || $0.state.isActive)) }) else { throw ProjectError.invalid("This project already has that job queued") }
        jobs.append(job)
    }
    public mutating func start(id: String, token: String, nowMs: Int64) throws -> ProcessingJob {
        guard !jobs.contains(where: { $0.state.isActive }), let index = jobs.firstIndex(where: { $0.id == id && $0.state == .queued }) else {
            throw ProjectError.invalid("Processing queue already running or job unavailable")
        }
        jobs[index].startedAtMs = nowMs
        jobs[index].state = .running; jobs[index].attempt += 1; jobs[index].runToken = token
        jobs[index].stopReason = nil; jobs[index].progress = 0; jobs[index].detail = "Starting"
        jobs[index].analysisSteps = nil
        jobs[index].updatedAtMs = nowMs
        return jobs[index]
    }
    public mutating func stop(id: String, reason: ProcessingStopReason, nowMs: Int64) {
        guard let index = jobs.firstIndex(where: { $0.id == id }) else { return }
        if jobs[index].state == .queued {
            jobs[index].state = reason == .userCancelled ? .cancelled : .interrupted
        } else if jobs[index].state == .running { jobs[index].state = .cancelling }
        else { return }
        jobs[index].stopReason = reason; jobs[index].updatedAtMs = nowMs
        jobs[index].detail = reason == .userCancelled ? "Cancelling" : "Stopping; saved features will be reused"
    }
    public mutating func remove(id: String, nowMs: Int64) {
        guard let index = jobs.firstIndex(where: { $0.id == id }) else { return }
        if jobs[index].state.isActive {
            stop(id: id, reason: .userCancelled, nowMs: nowMs)
            jobs[index].removalRequested = true
            jobs[index].detail = "Removing; stopping processing"
        } else { jobs.remove(at: index) }
    }
    /// Attempt tokens prevent late progress/completions from an older attempt affecting a resumed job.
    @discardableResult public mutating func finish(id: String, token: String, state: ProcessingJobState,
                                                  detail: String, outputNames: [String] = [], stageSeconds: [String: Double]? = nil, nowMs: Int64) -> Bool {
        guard let index = jobs.firstIndex(where: { $0.id == id && $0.runToken == token && $0.state.isActive }) else { return false }
        guard [.completed, .cancelled, .interrupted, .failed].contains(state) else { return false }
        if jobs[index].removalRequested == true { jobs.remove(at: index); return true }
        let stopped = jobs[index].state == .cancelling
        jobs[index].state = stopped ? (jobs[index].stopReason == .userCancelled ? .cancelled : .interrupted) : state
        jobs[index].detail = stopped ? "Stopped; resume when ready" : detail
        jobs[index].updatedAtMs = nowMs; jobs[index].runToken = nil
        if jobs[index].state == .completed { jobs[index].progress = 1; jobs[index].outputNames = outputNames; jobs[index].stageSeconds = stageSeconds }
        return true
    }
    public mutating func resume(id: String, nowMs: Int64) throws {
        guard let index = jobs.firstIndex(where: { $0.id == id && $0.state.canResume }) else { throw ProjectError.invalid("Job cannot be resumed") }
        guard !jobs.contains(where: { $0.id != id && $0.projectId == jobs[index].projectId && $0.kind == jobs[index].kind &&
              ($0.state == .queued || $0.state.isActive) }) else { throw ProjectError.invalid("This project already has that job queued") }
        jobs[index].state = .queued; jobs[index].runToken = nil; jobs[index].stopReason = nil
        jobs[index].detail = "Waiting to resume"; jobs[index].updatedAtMs = nowMs
    }
    public mutating func recoverAfterLaunch(nowMs: Int64) {
        jobs.removeAll { $0.removalRequested == true }
        for index in jobs.indices where jobs[index].state.isActive || jobs[index].state == .queued {
            jobs[index].state = jobs[index].stopReason == .userCancelled ? .cancelled : .interrupted
            jobs[index].runToken = nil; jobs[index].updatedAtMs = nowMs
            jobs[index].detail = "Processing was interrupted; resume to continue"
            jobs[index].analysisSteps = jobs[index].analysisSteps?.map { measurement in
                var stopped = measurement
                if stopped.status == .running { stopped.status = .stopped; stopped.etaSeconds = nil }
                return stopped
            }
        }
    }
}
