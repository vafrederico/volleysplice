import Foundation
import SwiftUI
import UIKit

struct ProcessingJobResult: Sendable {
    var detail: String
    var outputNames: [String] = []
    var stageSeconds: [String: Double]?
}

@MainActor final class ProcessingQueue: ObservableObject {
    @Published private(set) var ledger = ProcessingJobLedger()
    @Published private(set) var error: String?
    @Published private(set) var waitingForSystem = false
    var onChange: (() -> Void)?
    var execute: ((ProcessingJob) async throws -> ProcessingJobResult)?
    private let store: ProcessingJobStore?
    private let continuation = ContinuedProcessingCoordinator()
    private var worker: Task<Void, Never>?
    private var foreground = true
    private var batch: [String] = []
    private var completedInBatch = 0
    private var batchFailed = false
    private var lastProgressWrite = Date.distantPast
    private var lastProgressReport = -Double.infinity
    private var lastStepReport = -Double.infinity
    private var analysisTracker: AnalysisProgressTracker?
    private var progressTicker: Task<Void, Never>?
    private var cleanupLease: UIBackgroundTaskIdentifier = .invalid
    private var cleanupLeaseGeneration: UUID?
    private struct Submission {
        let job: ProcessingJob
        let token: UUID
        let task: Task<Void, Error>
    }
    private var submissions: [String: Submission] = [:]
    private var submissionTail: Task<Void, Error>?
    private var submissionTailToken: UUID?
    private let snapshotWriter: @Sendable (ProjectDocument, URL) async throws -> Void
    @Published private(set) var preparingInputCount = 0
    var jobs: [ProcessingJob] { ledger.jobs }
    var unfinishedJobs: [ProcessingJob] { ledger.unfinishedJobs }
    var preparingJobs: [ProcessingJob] { submissions.values.map(\.job).sorted { $0.createdAtMs < $1.createdAtMs } }
    var active: ProcessingJob? { jobs.first { $0.state.isActive } }
    var hasPendingWork: Bool { preparingInputCount > 0 || jobs.contains { $0.state.isActive || $0.state == .queued } }
    var canRun: Bool { store != nil && foreground }
    init(folder: URL, snapshotWriter: @escaping @Sendable (ProjectDocument, URL) async throws -> Void = ProcessingQueue.writeSnapshot) {
        self.snapshotWriter = snapshotWriter
        do {
            let opened = try ProcessingJobStore(folder: folder)
            var saved = try opened.load(); saved.recoverAfterLaunch(nowMs: Self.now)
            try opened.save(saved); store = opened; ledger = saved
            try? opened.discardUnreferencedSnapshots(in: saved)
            // Only abandoned staging files from this writer are disposable.
            for url in (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? [] {
                let name = url.lastPathComponent
                if name.hasPrefix(".snapshot-"), name.hasSuffix(".stage"),
                   UUID(uuidString: String(name.dropFirst(10).dropLast(6))) != nil {
                    try? FileManager.default.removeItem(at: url)
                }
            }
        } catch { store = nil; self.error = "Unable to load processing queue: \(error.localizedDescription)" }
    }
    private static var now: Int64 { Int64(Date().timeIntervalSince1970 * 1000) }
    private func change(_ body: (inout ProcessingJobLedger) throws -> Void) throws {
        guard let store else { throw ProjectError.invalid("Processing queue storage is unavailable") }
        var updated = ledger; try body(&updated); try store.save(updated); ledger = updated; onChange?()
    }
    nonisolated private static func writeSnapshot(_ project: ProjectDocument, to url: URL) async throws {
        let writer = Task.detached(priority: .userInitiated) {
            try Task.checkCancellation()
            try project.save(to: url)
            let handle = try FileHandle(forWritingTo: url)
            do { try handle.synchronize(); try handle.close() } catch { try? handle.close(); throw error }
            try Task.checkCancellation()
        }
        try await withTaskCancellationHandler(operation: { try await writer.value }, onCancel: { writer.cancel() })
    }
    func enqueue(_ job: ProcessingJob, snapshot: ProjectDocument? = nil) async throws {
        try Task.checkCancellation()
        guard foreground, let store else { throw ProjectError.invalid("Open the app to queue processing") }
        var checked = ledger; try checked.enqueue(job)
        guard !submissions.values.contains(where: { $0.job.id == job.id || ($0.job.projectId == job.projectId && $0.job.kind == job.kind) }) else {
            throw ProjectError.invalid("This project already has that job queued")
        }
        if let snapshot, snapshot.id != job.projectId { throw ProjectError.invalid("Snapshot does not belong to the queued project") }
        let token = UUID(), predecessor = submissionTail
        let stage = store.folder.appendingPathComponent(".snapshot-\(token.uuidString).stage")
        let destination = store.folder.appendingPathComponent(job.id + ".project.json")
        let operation = Task { @MainActor in
            // A cancelled predecessor still drains its writer before the next
            // submission starts. Publication order follows submission order.
            if let predecessor { _ = try? await predecessor.value }
            var snapshotMoved = false, committed = false
            defer {
                try? FileManager.default.removeItem(at: stage)
                if snapshotMoved && !committed { try? FileManager.default.removeItem(at: destination) }
                if self.submissions[job.id]?.token == token { self.submissions.removeValue(forKey: job.id) }
                if self.submissionTailToken == token { self.submissionTail = nil; self.submissionTailToken = nil }
                self.preparingInputCount = self.submissions.count; self.onChange?()
            }
            try Task.checkCancellation()
            if let snapshot { try await self.snapshotWriter(snapshot, stage) }
            try Task.checkCancellation()
            guard self.foreground, self.submissions[job.id]?.token == token else { throw CancellationError() }
            var latest = self.ledger; try latest.enqueue(job)
            if snapshot != nil {
                // No await between installing the immutable input and the
                // small durable ledger. Removal cannot interleave publication.
                try FileManager.default.moveItem(at: stage, to: destination)
                snapshotMoved = true
            }
            try store.save(latest)
            committed = true; self.ledger = latest; self.onChange?()
            self.runQueuedFromUser()
        }
        submissions[job.id] = Submission(job: job, token: token, task: operation)
        submissionTail = operation; submissionTailToken = token
        preparingInputCount = submissions.count; onChange?()
        try await withTaskCancellationHandler(operation: { try await operation.value }, onCancel: { operation.cancel() })
    }
    func inputSnapshot(_ job: ProcessingJob) async throws -> ProjectDocument {
        guard let store else { throw ProjectError.invalid("Processing queue storage unavailable") }
        try job.validate(); try Task.checkCancellation()
        guard isCurrent(job) else { throw CancellationError() }
        let url = store.folder.appendingPathComponent(job.id + ".project.json")
        let reader = Task.detached(priority: .userInitiated) {
            try Task.checkCancellation()
            let snapshot = try ProjectDocument.load(from: url)
            try Task.checkCancellation()
            return snapshot
        }
        let snapshot = try await withTaskCancellationHandler(operation: { try await reader.value }, onCancel: { reader.cancel() })
        try Task.checkCancellation()
        guard isCurrent(job) else { throw CancellationError() }
        guard snapshot.id == job.projectId else { throw ProjectError.invalid("Snapshot does not belong to the queued project") }
        return snapshot
    }
    func isCurrent(_ job: ProcessingJob) -> Bool {
        jobs.contains { $0.id == job.id && $0.runToken == job.runToken && $0.state == .running }
    }
    func resume(_ id: String) {
        do { try change { try $0.resume(id: id, nowMs: Self.now) }; runQueuedFromUser() }
        catch { self.error = error.localizedDescription }
    }
    func resumeAll() {
        do {
            try change { ledger in
                for id in ledger.jobs.filter({ $0.state == .interrupted }).map(\.id) { try ledger.resume(id: id, nowMs: Self.now) }
            }
            runQueuedFromUser()
        } catch { self.error = error.localizedDescription }
    }
    func cancel(_ id: String) { stop(id, reason: .userCancelled) }
    func remove(_ id: String) {
        submissions[id]?.task.cancel()
        let wasActive = active?.id == id
        // Invalidate publication immediately, including when the disk is full.
        ledger.remove(id: id, nowMs: Self.now)
        do {
            try store?.save(ledger)
            try store?.discardUnreferencedSnapshots(in: ledger)
        } catch { self.error = "Removal requested; cannot save queue: \(error.localizedDescription)" }
        onChange?()
        if wasActive { worker?.cancel() }
        if worker == nil {
            batch.removeAll { $0 == id }
            if batch.isEmpty { finishBatch() }
        }
    }
    private func stop(_ id: String, reason: ProcessingStopReason) {
        submissions[id]?.task.cancel()
        let wasActive = active?.id == id
        // Cancellation must invalidate publication even if storage is full.
        // The persisted running attempt is recovered as interrupted on launch.
        ledger.stop(id: id, reason: reason, nowMs: Self.now)
        do { try store?.save(ledger) }
        catch { self.error = "Cancellation requested; cannot save queue: \(error.localizedDescription)" }
        onChange?()
        if wasActive { worker?.cancel() }
        if worker == nil && !batch.contains(where: { id in jobs.contains { $0.id == id && $0.state == .queued } }) {
            continuation.finish(success: false); batch = []; waitingForSystem = false
        }
    }
    func sceneChanged(_ phase: ScenePhase) {
        foreground = phase == .active
        // Stop compositor work before the app loses foreground execution.
        // Waiting for .background can strand an in-flight CA frame while
        // cancelExport drains. CPU analysis keeps its continued-processing lease.
        if phase == .inactive, let active, active.kind == .videoExport {
            beginCleanupLease(for: active)
            stop(active.id, reason: .backgrounded)
        }
        if phase == .background {
            for submission in submissions.values { submission.task.cancel() }
            try? store?.save(ledger)
            guard let active else { return }
            if !continuation.isRunning || !active.kind.supportsCPUContinuation {
                beginCleanupLease(for: active)
                stop(active.id, reason: .backgrounded)
            }
        }
        // Returning to the app does not override a system/user cancellation. Resume is explicit.
    }
    func runQueuedFromUser() {
        guard foreground, execute != nil, worker == nil, !continuation.isPending, !continuation.isRunning,
              let first = jobs.first(where: { $0.state == .queued }) else { return }
        error = nil
        if first.kind.supportsCPUContinuation {
            batch = Array(jobs.filter { $0.state == .queued }.prefix(while: { $0.kind.supportsCPUContinuation }).map(\.id))
            completedInBatch = 0; batchFailed = false
            do {
                waitingForSystem = true
                let requested = try continuation.request(jobCount: batch.count, granted: { [weak self] in
                    self?.waitingForSystem = false; self?.startNextInBatch()
                }, expired: { [weak self] in self?.expireBatch() })
                if requested { onChange?(); return }
            } catch { self.error = "Continuing in foreground: \(error.localizedDescription)" }
            waitingForSystem = false
        } else { batch = [first.id]; completedInBatch = 0; batchFailed = false }
        startNextInBatch()
    }
    private func expireBatch() {
        batchFailed = true
        if let active { stop(active.id, reason: .systemExpired) }
        for id in batch where jobs.contains(where: { $0.id == id && $0.state == .queued }) { stop(id, reason: .systemExpired) }
        if worker == nil { continuation.finish(success: false); batch = []; waitingForSystem = false }
    }
    private func startNextInBatch() {
        guard worker == nil else { return }
        guard foreground || continuation.isRunning else { finishBatch(); return }
        guard let id = batch.first(where: { id in jobs.contains { $0.id == id && $0.state == .queued } }), let execute else {
            finishBatch(); return
        }
        do {
            var job: ProcessingJob?
            try change { job = try $0.start(id: id, token: UUID().uuidString, nowMs: Self.now) }
            guard let job, let token = job.runToken else { return }
            startProgressTracking(job)
            lastProgressReport = -Double.infinity
            lastProgressWrite = .distantPast
            worker = Task { [weak self] in
                guard let self else { return }
                do {
                    let output = try await execute(job)
                    try Task.checkCancellation()
                    self.endProgressTracking(job)
                    try self.change { $0.finish(id: id, token: token, state: .completed, detail: output.detail, outputNames: output.outputNames, stageSeconds: output.stageSeconds, nowMs: Self.now) }
                } catch {
                    self.endProgressTracking(job)
                    self.batchFailed = true
                    let state: ProcessingJobState = error is CancellationError ? .interrupted : .failed
                    do { try self.change { $0.finish(id: id, token: token, state: state, detail: error.localizedDescription, nowMs: Self.now) } }
                    catch {
                        // The worker has drained even when the durable ledger
                        // cannot be written. Release its in-memory slot safely;
                        // later work still must persist its start before running.
                        self.ledger.finish(id: id, token: token, state: state,
                                           detail: "Stopped; queue storage needs attention", nowMs: Self.now)
                        self.error = "Cannot save processing status: \(error.localizedDescription)"
                        self.onChange?()
                    }
                }
                self.worker = nil; self.completedInBatch += 1; self.endCleanupLease()
                try? self.store?.discardUnreferencedSnapshots(in: self.ledger)
                self.continuation.progress(completedJobs: self.completedInBatch, fraction: 0, detail: "Checking queued recordings")
                self.startNextInBatch()
            }
        } catch { self.error = error.localizedDescription; batchFailed = true; finishBatch(continueQueued: false) }
    }
    private func finishBatch(continueQueued: Bool = true) {
        continuation.finish(success: !batchFailed); batch = []; waitingForSystem = false; endCleanupLease(); onChange?()
        if continueQueued && foreground && worker == nil && jobs.contains(where: { $0.state == .queued }) {
            // All remaining work was explicitly queued in this foreground session.
            runQueuedFromUser()
        }
    }
    private func endCleanupLease() {
        cleanupLeaseGeneration = nil
        if cleanupLease != .invalid { UIApplication.shared.endBackgroundTask(cleanupLease); cleanupLease = .invalid }
    }
    private func beginCleanupLease(for job: ProcessingJob) {
        // A second background transition while the same worker drains must not
        // leak its first lease or let an old expiration cancel a later attempt.
        guard cleanupLease == .invalid else { return }
        let generation = UUID()
        cleanupLeaseGeneration = generation
        cleanupLease = UIApplication.shared.beginBackgroundTask(withName: "Save processing checkpoint") { [weak self] in
            Task { @MainActor in
                guard let self, self.cleanupLeaseGeneration == generation else { return }
                if self.active?.id == job.id, self.active?.runToken == job.runToken { self.worker?.cancel() }
                self.endCleanupLease()
            }
        }
    }
    func updateProgress(_ job: ProcessingJob, fraction: Double, detail: String) {
        guard isCurrent(job), let index = ledger.jobs.firstIndex(where: { $0.id == job.id }) else { return }
        let now = ProcessInfo.processInfo.systemUptime
        let next = min(1, max(ledger.jobs[index].progress, fraction.isFinite ? fraction : 0))
        guard now - lastProgressReport >= 0.25 || (next == 1 && ledger.jobs[index].progress < 1) else { return }
        lastProgressReport = now
        // One publication, at most four times per second. Only queue-specific
        // views observe these updates; workspace notifications are lifecycle-only.
        var updated = ledger
        updated.jobs[index].progress = next
        let displayedDetail = updated.jobs[index].analysisSteps?.first(where: { $0.status == .running }).map { "\($0.id.label) · \($0.metrics)" } ?? detail
        updated.jobs[index].detail = displayedDetail; updated.jobs[index].updatedAtMs = Self.now
        ledger = updated
        let measurement = updated.jobs[index].analysisSteps?.first(where: { $0.status == .running })
        continuation.progress(completedJobs: completedInBatch, fraction: next, title: measurement?.id.label ?? "Analyze recordings",
                              detail: measurement?.notificationMetrics ?? displayedDetail)
        if Date().timeIntervalSince(lastProgressWrite) >= 1 { try? store?.save(ledger); lastProgressWrite = Date() }
    }
    private func startProgressTracking(_ job: ProcessingJob) {
        progressTicker?.cancel(); analysisTracker = nil
        guard job.kind.supportsCPUContinuation else { return }
        var steps: [AnalysisStep] = job.kind == .analysis ? [.video, .audio, .rally] : []
        if job.kind == .analysis && job.analysis?.prepareScores == true {
            steps.append(.servingSide)
            if job.analysis?.generateSideSwitchMarkers ?? true { steps.append(.sideSwitch) }
        }
        // Score-only jobs discover the enabled steps from their retained project settings.
        analysisTracker = AnalysisProgressTracker(steps: steps); lastStepReport = -.infinity
        progressTicker = Task { [weak self] in
            while !Task.isCancelled {
                do { try await Task.sleep(for: .seconds(1)) } catch { return }
                guard let self, self.isCurrent(job) else { return }
                self.publishAnalysisProgress(job, force: true)
            }
        }
    }
    func updateAnalysisProgress(_ job: ProcessingJob, event: AnalysisProgressEvent) {
        guard isCurrent(job), analysisTracker != nil else { return }
        let previous = ledger.jobs.first(where: { $0.id == job.id })?.analysisSteps?.last(where: { $0.status != .pending })
        analysisTracker?.update(event)
        publishAnalysisProgress(job, force: previous?.id != event.step || event.fraction >= 1 || event.failed)
    }
    private func publishAnalysisProgress(_ job: ProcessingJob, force: Bool) {
        guard isCurrent(job), let index = ledger.jobs.firstIndex(where: { $0.id == job.id }), let tracker = analysisTracker else { return }
        let now = ProcessInfo.processInfo.systemUptime
        guard force || now - lastStepReport >= 0.25 else { return }
        lastStepReport = now
        let rows = tracker.snapshot(now: now)
        var updated = ledger; updated.jobs[index].analysisSteps = rows
        if let current = rows.first(where: { $0.status == .running }) {
            updated.jobs[index].detail = "\(current.id.label) · \(current.metrics)"
        }
        updated.jobs[index].updatedAtMs = Self.now
        ledger = updated
        let measurement = rows.first(where: { $0.status == .running })
        continuation.progress(completedJobs: completedInBatch, fraction: updated.jobs[index].progress,
                              title: measurement?.id.label ?? "Analyze recordings", detail: measurement?.notificationMetrics ?? updated.jobs[index].detail)
        if Date().timeIntervalSince(lastProgressWrite) >= 1 { try? store?.save(ledger); lastProgressWrite = Date() }
    }
    private func endProgressTracking(_ job: ProcessingJob) {
        progressTicker?.cancel(); progressTicker = nil
        guard let index = ledger.jobs.firstIndex(where: { $0.id == job.id }), analysisTracker != nil else { return }
        let now = ProcessInfo.processInfo.systemUptime
        analysisTracker?.stop(now: now)
        ledger.jobs[index].analysisSteps = analysisTracker?.snapshot(now: now)
        analysisTracker = nil
    }
}

struct ProcessingQueueBar: View {
    @ObservedObject var queue: ProcessingQueue
    let onOpen: () -> Void
    var body: some View {
        if !queue.unfinishedJobs.isEmpty || queue.preparingInputCount > 0 {
            Button(action: onOpen) {
                HStack {
                    Text("Queue \(queue.unfinishedJobs.count + queue.preparingInputCount)")
                    Text(queue.preparingInputCount > 0 ? "Preparing job input" :
                        queue.active?.detail ?? (queue.waitingForSystem ? "Waiting for iOS" : "Unfinished analysis and exports"))
                        .font(.caption).lineLimit(3)
                    Spacer(minLength: 0)
                }.padding(.horizontal).frame(minHeight: 44).contentShape(Rectangle())
            }.buttonStyle(.plain).accessibilityIdentifier("processingQueueBar")
        }
    }
}

struct ProcessingQueueSheet: View {
    @ObservedObject var queue: ProcessingQueue
    var onRelink: ((ProcessingJob) -> Void)? = nil
    @Environment(\.dismiss) private var dismiss
    var body: some View {
        NavigationStack {
            List {
                if let error = queue.error { Text(error).foregroundStyle(.red) }
                if queue.waitingForSystem { Text("Waiting for iOS to start background-capable analysis") }
                if queue.unfinishedJobs.isEmpty && queue.preparingInputCount == 0 { Text("No unfinished work.") }
                ForEach(queue.preparingJobs) { job in
                    VStack(alignment: .leading, spacing: 6) {
                        Text(job.sourceName).font(.headline)
                        Text("Preparing job input").font(.caption)
                        Button("Cancel") { queue.cancel(job.id) }.buttonStyle(.bordered)
                            .accessibilityIdentifier("cancelPreparingJob-" + job.id)
                    }
                }
                ForEach(queue.unfinishedJobs.reversed()) { job in
                    VStack(alignment: .leading, spacing: 6) {
                        Text(job.sourceName).font(.headline)
                        Text("\(job.kind == .videoExport ? "Video export" : job.kind == .scores ? "Score preparation" : "Analysis") · \(job.state.rawValue.capitalized)")
                        Text(job.detail).font(.caption)
                        if job.state.isActive { ProgressView(value: job.progress) }
                        HStack {
                            if (job.state == .queued || job.state.isActive) && job.removalRequested != true {
                                Button("Cancel") { queue.cancel(job.id) }.accessibilityIdentifier("cancelJob-" + job.id)
                            }
                            if job.state.canResume {
                                if let onRelink {
                                    Button("Re-link video") { onRelink(job) }.disabled(!queue.canRun)
                                        .accessibilityIdentifier("relinkJob-" + job.id)
                                }
                                Button("Resume") { queue.resume(job.id) }.disabled(!queue.canRun).accessibilityIdentifier("resumeJob-" + job.id)
                            }
                            Button(job.removalRequested == true ? "Removing…" : "Remove", role: .destructive) {
                                queue.remove(job.id)
                            }.disabled(job.removalRequested == true).accessibilityIdentifier("removeJob-" + job.id)
                        }.buttonStyle(.bordered)
                    }.accessibilityElement(children: .contain)
                }
            }.navigationTitle("Processing queue")
                .toolbar {
                    ToolbarItem(placement: .cancellationAction) { Button("Done") { dismiss() }.accessibilityIdentifier("queueDone") }
                    ToolbarItem(placement: .primaryAction) { Button("Resume queued") { queue.resumeAll() }.disabled(!queue.canRun) }
                }
        }
    }
}
