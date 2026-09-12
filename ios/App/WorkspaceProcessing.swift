import AVFoundation
import Foundation

/// One FIFO stream preserves decoder callback order and drains the last measurement before completion.
@MainActor final class AnalysisProgressRelay {
    enum Event: Sendable {
        case overall(Double, String)
        case step(AnalysisProgressEvent)
    }
    nonisolated private let continuation: AsyncStream<Event>.Continuation
    private let consumer: Task<Void, Never>
    init(consume: @escaping @MainActor (Event) -> Void) {
        let pair = AsyncStream<Event>.makeStream()
        continuation = pair.continuation
        consumer = Task { for await event in pair.stream { consume(event) } }
    }
    nonisolated func overall(_ fraction: Double, _ detail: String) { continuation.yield(.overall(fraction, detail)) }
    nonisolated func step(_ event: AnalysisProgressEvent) { continuation.yield(.step(event)) }
    func finish() async { continuation.finish(); await consumer.value }
}

@MainActor extension WorkspaceModel {
    private func progressRelay(for job: ProcessingJob) -> AnalysisProgressRelay {
        AnalysisProgressRelay { event in
            switch event {
            case let .overall(fraction, detail): self.queue.updateProgress(job, fraction: fraction, detail: detail)
            case let .step(measurement): self.queue.updateAnalysisProgress(job, event: measurement)
            }
        }
    }
    func enqueueAnalysis() {
        guard let source, let media, !busy, start.isFinite, end.isFinite,
              start >= 0, end > start, end <= media.duration else {
            error = "Choose a recording and valid game bounds"; return
        }
        do {
            try roi.validate()
            let job = ProcessingJob(kind: .analysis, projectId: UUID().uuidString, sourceName: sourceName ?? source.lastPathComponent,
                sourceFingerprint: try selectedFingerprint ?? ProjectArchive.sampledFingerprint(url: source),
                analysis: .init(startMs: ProjectSession.ms(start), endMs: ProjectSession.ms(end),
                    roi: [roi.x, roi.y, roi.width, roi.height], prepareScores: prepareScore, generateSideSwitchMarkers: generateSideSwitchMarkers))
            try prepareQueueFiles(source: source)
            Task {
                do {
                    try await queue.enqueue(job)
                    status = "Analysis queued; you can open another recording or project"
                } catch is CancellationError { status = "Cancelled" }
                catch { self.error = error.localizedDescription }
            }
        } catch { self.error = error.localizedDescription }
    }
    func enqueueScores() {
        guard let project, !scoreJobPending(projectId: project.id), let source, !busy, flushProject() else { return }
        do {
            let job = ProcessingJob(kind: .scores, projectId: project.id, sourceName: sourceName ?? source.lastPathComponent,
                sourceFingerprint: try selectedFingerprint ?? ProjectArchive.sampledFingerprint(url: source))
            try prepareQueueFiles(source: source)
            Task {
                // Retoggling can schedule another MainActor task before this
                // one enters enqueue. Check both staged and durable jobs here.
                guard !scoreJobPending(projectId: project.id) else { return }
                do {
                    try await queue.enqueue(job, snapshot: project)
                    // Staging does not disable the editor. Preserve any score
                    // edits (or project navigation) made while it was running.
                    if var current = self.project, current.id == project.id,
                       current.draft.scoreTracking == project.draft.scoreTracking {
                        var tracking = try ScoreTracking.fromJSON(current.draft.scoreTracking, durationMs: current.durationMs)
                        tracking.enabled = true; current.draft.scoreTracking = try tracking.jsonValue()
                        self.project = current; saveProject()
                    }
                    status = "Score preparation queued"
                } catch is CancellationError { status = "Cancelled" }
                catch { self.error = error.localizedDescription }
            }
        } catch { self.error = error.localizedDescription }
    }
    private func scoreJobPending(projectId: String) -> Bool {
        (queue.preparingJobs + queue.jobs).contains {
            $0.projectId == projectId && $0.kind == .scores && ($0.state == .queued || $0.state.isActive)
        }
    }
    func enqueueVideo(destination: VideoExportDestination? = nil) {
        guard let project, let source, !busy, flushProject() else { return }
        do {
            var job = ProcessingJob(kind: .videoExport, projectId: project.id, sourceName: sourceName ?? source.lastPathComponent,
                sourceFingerprint: try selectedFingerprint ?? ProjectArchive.sampledFingerprint(url: source))
            job.videoDestination = destination
            Task {
                do {
                    try await queue.enqueue(job, snapshot: project)
                    status = "Video export queued; keep the app open during encoding"
                } catch is CancellationError { status = "Cancelled" }
                catch { self.error = error.localizedDescription }
            }
        } catch { self.error = error.localizedDescription }
    }
    private func prepareQueueFiles(source: URL) throws {
        try FileManager.default.createDirectory(at: cacheFolder, withIntermediateDirectories: true)
        let owned = source.resolvingSymlinksInPath().deletingLastPathComponent() == documents.resolvingSymlinksInPath()
        for url in owned ? [source, cacheFolder] : [cacheFolder] {
            try FileManager.default.setAttributes([.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication], ofItemAtPath: url.path)
        }
    }
    func processJob(_ job: ProcessingJob) async throws -> ProcessingJobResult {
        let access = try await sourceLibrary.resolve(name: job.sourceName, fingerprint: job.sourceFingerprint)
        if job.kind == .videoExport, let choice = job.videoDestination, choice.kind == .files {
            guard let bookmark = choice.folderBookmark else { throw ProjectError.invalid("Choose an export folder") }
            let snapshot = try await queue.inputSnapshot(job)
            return try await CoordinatedVideoExport.inFolder(bookmark: bookmark, reading: access) { sourceURL, folder in
                guard try ProjectArchive.sampledFingerprint(url: sourceURL) == job.sourceFingerprint else {
                    throw SourceVideoUnavailable(name: job.sourceName)
                }
                try Task.checkCancellation()
                return try await self.renderQueuedVideo(job, snapshot: snapshot, sourceURL: sourceURL, folder: folder)
            }
        }
        return try await CoordinatedSourceRead.perform(access) { sourceURL in
            guard try ProjectArchive.sampledFingerprint(url: sourceURL) == job.sourceFingerprint else {
                throw SourceVideoUnavailable(name: job.sourceName)
            }
            try Task.checkCancellation()
            switch job.kind {
            case .analysis: return try await self.processAnalysis(job, sourceURL: sourceURL)
            case .scores: return try await self.processScores(job, sourceURL: sourceURL)
            case .videoExport: return try await self.processVideo(job, sourceURL: sourceURL)
            }
        }
    }
    private func requireCurrent(_ job: ProcessingJob) throws {
        try Task.checkCancellation()
        guard queue.isCurrent(job) else { throw CancellationError() }
    }
    private func processAnalysis(_ job: ProcessingJob, sourceURL: URL) async throws -> ProcessingJobResult {
        guard let settings = job.analysis else { throw ProjectError.invalid("Missing queued analysis settings") }
        let cached = cacheFolder
        let region = AnalysisRegion(x: settings.roi[0], y: settings.roi[1], width: settings.roi[2], height: settings.roi[3])
        let progress = progressRelay(for: job)
        let child = Task.detached(priority: .userInitiated) {
            try await AnalysisPipeline.analyze(url: sourceURL, roi: region, start: Double(settings.startMs) / 1000,
                end: Double(settings.endMs) / 1000, cacheFolder: cached, prepareScore: settings.prepareScores, generateSideSwitchMarkers: settings.generateSideSwitchMarkers ?? true,
                stepProgress: { progress.step($0) }) { value, detail in
                    progress.overall(value, detail)
                }
        }
        let output: AnalysisResult
        do {
            output = try await withTaskCancellationHandler(operation: { try await child.value }, onCancel: { child.cancel() })
            await progress.finish()
        } catch { await progress.finish(); throw error }
        try requireCurrent(job)
        let projectTask = Task.detached { try ProjectSession.create(sourceURL: sourceURL, result: output) }
        var created = try await withTaskCancellationHandler(operation: { try await projectTask.value }, onCancel: { projectTask.cancel() })
        try requireCurrent(job)
        created.id = job.projectId
        created.sourceName = job.sourceName
        created.feedback?["source"]?["file"]?["name"] = .string(job.sourceName)
        created.feedback?["source"]?["projectId"] = .string(job.projectId)
        created.feedback?["source"]?["analysisId"] = .string(job.projectId + "-" + ProjectArchive.modelId + "-native-source")
        created.feedback?["source"]?["generateSideSwitchMarkers"] = .bool(settings.generateSideSwitchMarkers ?? true)
        let destination = projectURL(created)
        if FileManager.default.fileExists(atPath: destination.path) {
            // A prior attempt may have committed the project immediately before a process exit.
            // Never replace a completed project's subsequent user edits during retry.
            let existing = try await loadLatestProject(destination)
            guard existing.feedback?["source"]?["file"]?["sampledFingerprint"]?.string == job.sourceFingerprint else {
                throw ProjectError.invalid("Queued project identity conflicts with an existing project")
            }
            created = existing
        } else {
            let staged = try await ProjectPersistence.prepare(created, to: destination)
            try requireCurrent(job)
            created = try ProjectPersistence.publish(staged)
        }
        let analysisURL = documents.appendingPathComponent("analysis-\(output.cacheIdentity.prefix(12)).json")
        try await Task.detached {
            let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
            try encoder.encode(output).write(to: analysisURL, options: .atomic)
        }.value
        try requireCurrent(job)
        refreshFiles()
        if selectedFingerprint == job.sourceFingerprint,
           ProjectSession.ms(start) == settings.startMs, ProjectSession.ms(end) == settings.endMs,
           project == nil && !showEditor && !busy {
            result = output; exportURL = analysisURL
            project = created; restoreSetup(created)
            showEditor = true
        }
        status = "Analysis complete: \(output.intervals.count) rallies"
        return .init(detail: output.scoreError.map { "Rallies ready; score preparation can be retried: \($0)" } ?? "Project ready",
                     outputNames: [destination.lastPathComponent, analysisURL.lastPathComponent], stageSeconds: output.stageSeconds)
    }
    private func processScores(_ job: ProcessingJob, sourceURL: URL) async throws -> ProcessingJobResult {
        let snapshot = try await queue.inputSnapshot(job), cache = cacheFolder
        let progress = progressRelay(for: job)
        let child = Task.detached(priority: .userInitiated) {
            try await ScoreAnalysis.prepare(project: snapshot, source: sourceURL, cacheFolder: cache,
                stepProgress: { progress.step($0) }) { value, detail in
                progress.overall(value, detail)
            }
        }
        let scores: ScoreAnalysis.Result
        do {
            scores = try await withTaskCancellationHandler(operation: { try await child.value }, onCancel: { child.cancel() })
            await progress.finish()
        } catch { await progress.finish(); throw error }
        try requireCurrent(job)
        let servingFeedback = try ProjectSession.servingFeedback(scores.servingSide)
        let switchFeedback = try scores.sideSwitch.map { try ProjectSession.switchFeedback($0) }
        let scoreRevision = UUID().uuidString
        func validateAnalysis(_ project: ProjectDocument) throws {
            guard project.id == snapshot.id, project.draft.sourceRevision == snapshot.draft.sourceRevision,
                  project.durationMs == snapshot.durationMs, project.gameWindow == snapshot.gameWindow,
                  project.feedback?["features"] == snapshot.feedback?["features"],
                  project.feedback?["source"]?["featureRoi"] == snapshot.feedback?["source"]?["featureRoi"],
                  project.feedback?["initialInference"]?["ranges"] == snapshot.feedback?["initialInference"]?["ranges"] else {
                throw ProjectError.invalid("Project analysis changed while scores were being prepared")
            }
        }
        // Autosave compaction may publish a fresh base while full score feedback
        // is encoding. Reuse the completed inference and re-stage against that
        // newer base; never retry an analysis replacement or a storage failure.
        for _ in 0..<4 {
            var latest: ProjectDocument
            if let project, project.id == job.projectId { latest = project }
            else {
                let loaded = try await loadLatestProject(projectURL(snapshot))
                latest = project?.id == job.projectId ? project! : loaded
            }
            try requireCurrent(job); try validateAnalysis(latest)
            if project?.id == job.projectId {
                // Preserve the current draft immediately, then let this score
                // publication replace a redundant pending full autosave.
                try ProjectPersistence.checkpoint(latest, to: projectURL(latest))
                saveTask?.cancel()
            }
            let expectedRevision = latest.persistenceRevision
            latest.feedback?["initialInference"]?["servingSide"] = servingFeedback
            if let switchFeedback { latest.feedback?["initialInference"]?["sideSwitch"] = switchFeedback }
            latest.feedback?["scorePreparationError"] = scores.error.map(JSONValue.string)
            latest.feedback?["scorePreparationRevision"] = .string(scoreRevision)
            let staged = try await ProjectPersistence.prepare(latest, to: projectURL(latest))
            try requireCurrent(job)
            // Re-read after encoding. Merge late corrections/tombstones only
            // when the staged feedback still targets the current base revision.
            let current: ProjectDocument
            if let project, project.id == job.projectId { current = project }
            else {
                let loaded = try await loadLatestProject(projectURL(snapshot))
                current = project?.id == job.projectId ? project! : loaded
            }
            try requireCurrent(job); try validateAnalysis(current)
            guard current.persistenceRevision == expectedRevision else { continue }
            if project?.id == job.projectId { saveTask?.cancel() }
            latest = try ProjectPersistence.publish(staged, current: current) { draft in
                var tracking = try ScoreTracking.fromJSON(draft.scoreTracking, durationMs: current.durationMs)
                tracking = try ScoreReducer.seedModelMarkers(tracking, output: scores.servingSide, sideSwitchOutput: scores.sideSwitch,
                    sideSwitchEnabled: current.feedback?["source"]?["generateSideSwitchMarkers"]?.bool ?? true)
                draft.scoreTracking = try tracking.jsonValue()
                draft = EditorMath.alignRallyServeMarkers(draft)
            }
            if project?.id == job.projectId { project = latest }
            refreshFiles()
            return .init(detail: scores.error ?? "Scores ready", outputNames: [projectURL(latest).lastPathComponent])
        }
        throw ProjectError.invalid("Project kept changing while scores were being saved; retry score preparation")
    }
    private func processVideo(_ job: ProcessingJob, sourceURL: URL) async throws -> ProcessingJobResult {
        let snapshot = try await queue.inputSnapshot(job)
        return try await renderQueuedVideo(job, snapshot: snapshot, sourceURL: sourceURL, folder: documents)
    }
    private func renderQueuedVideo(_ job: ProcessingJob, snapshot: ProjectDocument, sourceURL: URL, folder: URL) async throws -> ProcessingJobResult {
        let destination = folder.appendingPathComponent("VolleySplice-\(job.id).mp4")
        let chapters = VideoExporter.chaptersURL(for: destination)
        let receipt = documents.appendingPathComponent(".photo-export-\(job.id).json")
        let photoMove = job.videoDestination?.kind == .photos
        if photoMove, let data = try? Data(contentsOf: receipt),
           let saved = try? JSONDecoder().decode(PhotoDeliveryReceipt.self, from: data) {
            if saved.completed { return .init(detail: "Saved to Camera Roll", outputNames: [chapters.lastPathComponent]) }
            if !FileManager.default.fileExists(atPath: destination.path) {
                throw ProjectError.invalid("Photos may already contain this export. Check Camera Roll before starting another export.")
            }
        }
        if FileManager.default.fileExists(atPath: destination.path) {
            let existing = try await MediaDecoder.describe(destination)
            let intervals = EditorMath.finalIntervals(snapshot.draft, suppression: try snapshot.feedback.map(ProjectArchive.suppressionRegions) ?? [], bounds: snapshot.gameWindow)
            guard abs(existing.duration * 1000 - Double(EditorMath.totalFinalMs(intervals))) <= 100,
                  FileManager.default.fileExists(atPath: chapters.path) else { throw ProjectError.invalid("An incomplete export already uses this job's output name") }
        } else {
            // Only this job owns these unique output names. Stage and publish inside the chosen folder.
            if FileManager.default.fileExists(atPath: chapters.path) { try FileManager.default.removeItem(at: chapters) }
            _ = try await VideoExporter.export(source: sourceURL, project: snapshot, destination: destination,
                managesBackgroundLease: false) { fraction, detail in
                    Task { @MainActor in self.queue.updateProgress(job, fraction: fraction, detail: detail) }
                }
        }
        try requireCurrent(job)
        if photoMove {
            try JSONEncoder().encode(PhotoDeliveryReceipt(completed: false)).write(to: receipt, options: .atomic)
            let identifier = try await PhotoExportDelivery.move(destination)
            // Persist the successful external delivery even if cancellation arrives during Photos' transaction.
            try JSONEncoder().encode(PhotoDeliveryReceipt(completed: true, assetIdentifier: identifier)).write(to: receipt, options: .atomic)
            refreshFiles()
            return .init(detail: "Saved to Camera Roll", outputNames: [chapters.lastPathComponent])
        }
        refreshFiles()
        return .init(detail: job.videoDestination?.kind == .files ? "Saved to Files" : "Video export ready",
                     outputNames: [destination.lastPathComponent, chapters.lastPathComponent])
    }
    private struct PhotoDeliveryReceipt: Codable { var completed: Bool; var assetIdentifier: String? }
    func openJobProject(_ job: ProcessingJob) {
        guard let filename = job.outputNames.first(where: { $0.hasSuffix(".volleyproject.json") }) else { return }
        openProject(documents.appendingPathComponent(filename))
    }
    func shareJobOutput(_ job: ProcessingJob) {
        shareItems = job.outputNames.map { documents.appendingPathComponent($0) }.filter { FileManager.default.fileExists(atPath: $0.path) }
        guard !shareItems.isEmpty else { error = "Export files are no longer available"; return }
        showShare = true
    }
    func saveJobOutput(_ job: ProcessingJob) {
        shareItems = job.outputNames.map { documents.appendingPathComponent($0) }.filter { FileManager.default.fileExists(atPath: $0.path) }
        guard !shareItems.isEmpty else { error = "Export files are no longer available"; return }
        showFileExport = true
    }
}
