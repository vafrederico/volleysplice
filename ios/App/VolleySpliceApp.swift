import SwiftUI
import AVKit
import UniformTypeIdentifiers

@main
struct VolleySpliceApp: App {
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var model = WorkspaceModel()
    var body: some Scene {
        WindowGroup {
            WorkspaceView(model: model)
                .interfaceScaled(isWindow: true)
                .preferredColorScheme(.light)
                .onAppear { setIdle(scenePhase); model.queue.sceneChanged(scenePhase) }
                .onChange(of: scenePhase) { _, phase in
                    setIdle(phase)
                    model.queue.sceneChanged(phase)
                    if phase != .active { model.flushProject() }
                }
                .onChange(of: model.busy) { _, _ in setIdle(scenePhase) }
                .onChange(of: model.queue.hasPendingWork) { _, _ in setIdle(scenePhase) }
        }
    }
    private func setIdle(_ phase: ScenePhase) {
        #if DEBUG
        let debugKeepAwake = true
        #else
        let debugKeepAwake = false
        #endif
        UIApplication.shared.isIdleTimerDisabled = phase == .active && (model.busy || model.queue.hasPendingWork || debugKeepAwake)
    }
}

@MainActor final class WorkspaceModel: ObservableObject {
    @Published var source: URL?
    @Published var media: MediaDescription?
    @Published var player = AVPlayer()
    @Published var start = 0.0
    @Published var end = 0.0
    @Published var roi = AnalysisRegion.fullFrame
    @Published var prepareScore = true
    @Published var generateSideSwitchMarkers = false
    @Published var progress = 0.0
    @Published var status = "Choose a recording"
    @Published var busy = false
    @Published var result: AnalysisResult?
    @Published var error: String?
    @Published var files: [URL] = []
    @Published var exportURL: URL?
    @Published var project: ProjectDocument?
    @Published var projects: [URL] = []
    @Published private(set) var projectNames: [URL: String] = [:]
    @Published private(set) var projectDates: [URL: Date] = [:]
    private var projectLabelRevisions: [URL: Date] = [:]
    private var fileRefreshTask: Task<Void, Never>?
    private var fileRefreshGeneration = UUID()
    @Published var showEditor = false
    @Published var showShare = false
    @Published var shareItems: [URL] = []
    @Published var reconnecting = false
    private var reconnectProject: ProjectDocument?
    private var reconnectReplacesProject = false
    private var reconnectJob: ProcessingJob?
    var sourceAccess: SourceVideoAccess?
    var sourceName: String? { sourceAccess?.filename ?? source?.lastPathComponent }
    var selectedFingerprint: String?
    let sourceLibrary: SourceVideoLibrary
    var saveTask: Task<Void, Never>?
    private var work: Task<Void, Never>?
    let documents: URL
    var cacheFolder: URL { FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("features", isDirectory: true) }
    let queue: ProcessingQueue
    @Published var showStorage = false
    @Published var showQueue = false
    @Published var showFileExport = false
    var afterQueueDismiss: (() -> Void)?
    init(documents: URL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0],
         support: URL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]) {
        self.documents = documents
        sourceLibrary = SourceVideoLibrary(folder: support.appendingPathComponent("SourceLinks", isDirectory: true),
            documents: documents)
        queue = ProcessingQueue(folder: support.appendingPathComponent("ProcessingJobs", isDirectory: true))
        queue.execute = { [weak self] job in
            guard let self else { throw CancellationError() }
            return try await self.processJob(job)
        }
        queue.onChange = { [weak self] in self?.objectWillChange.send() }
        refreshFiles()
    }
    func clearFeatureCache() {
        guard !busy, !queue.hasPendingWork else { error = "Finish or cancel queued processing before clearing features"; return }
        do {
            let folder = cacheFolder
            if FileManager.default.fileExists(atPath: folder.path) {
                try FileManager.default.removeItem(at: folder)
            }
            error = nil; status = "Feature cache cleared"
        } catch { fail(error) }
    }
    func refreshFiles() {
        fileRefreshTask?.cancel()
        let generation = UUID(); fileRefreshGeneration = generation
        let folder = documents, previousNames = projectNames, previousDates = projectDates, previousRevisions = projectLabelRevisions
        fileRefreshTask = Task {
            let result = await Task.detached(priority: .utility) {
                let entries = (try? FileManager.default.contentsOfDirectory(at: folder,
                    includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
                let media = entries.filter { ["mp4", "mov", "m4v"].contains($0.pathExtension.lowercased()) }.sorted { $0.lastPathComponent < $1.lastPathComponent }
                let saved = entries.filter { $0.lastPathComponent.hasSuffix(".volleyproject.json") }
                let present = Set(saved)
                var names = previousNames.filter { present.contains($0.key) }, dates = previousDates.filter { present.contains($0.key) }
                var revisions = previousRevisions.filter { present.contains($0.key) }
                for url in saved {
                    let base = (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast
                    let journal = (try? ProjectPersistence.checkpointURL(for: url).resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast
                    let modified = max(base, journal)
                    guard revisions[url] != modified else { continue }
                    let metadata = try? ProjectPersistence.listMetadata(from: url)
                    names[url] = metadata?.sourceName ?? "Saved project"
                    dates[url] = metadata.flatMap { $0.updatedAtMs > 0 ? Date(timeIntervalSince1970: Double($0.updatedAtMs) / 1000) : nil } ?? modified
                    revisions[url] = modified
                }
                return (media, saved, names, dates, revisions)
            }.value
            guard !Task.isCancelled, generation == fileRefreshGeneration else { return }
            files = result.0; projectNames = result.2; projectDates = result.3; projectLabelRevisions = result.4
            if let project, result.1.contains(projectURL(project)), project.draft.updatedAtMs > 0 {
                projectNames[projectURL(project)] = project.sourceName
                projectDates[projectURL(project)] = Date(timeIntervalSince1970: Double(project.draft.updatedAtMs) / 1000)
            }
            projects = result.1.sorted {
                let first = projectDates[$0] ?? .distantPast, second = projectDates[$1] ?? .distantPast
                return first == second ? $0.lastPathComponent < $1.lastPathComponent : first > second
            }
        }
    }
    func select(_ url: URL) { select(SourceVideoAccess(url)) }
    func select(_ access: SourceVideoAccess) {
        guard !busy, flushProject() else { return }
        busy = true; error = nil; status = "Opening video"
        work = Task {
            do {
                let destination = access.url
                let fingerprint = try await SourceVideoLibrary.inspect(access)
                let info = try await MediaDecoder.describe(destination)
                try Task.checkCancellation()
                await sourceLibrary.remember(access, fingerprint: fingerprint)
                player.pause(); player.replaceCurrentItem(with: nil)
                sourceAccess = access; selectedFingerprint = fingerprint
                source = destination; media = info; start = 0; end = info.duration; result = nil; project = nil; showEditor = false
                roi = .fullFrame
                player.replaceCurrentItem(with: access.playerItem())
                status = "Set game bounds, then analyze"; busy = false; refreshFiles()
            } catch {
                if error is CancellationError { busy = false; self.error = nil; status = "Opening video cancelled" }
                else { fail(error) }
            }
        }
    }
    func beginNewProject() {
        guard !busy, flushProject() else { return }
        player.pause(); player.replaceCurrentItem(with: nil)
        sourceAccess = nil; selectedFingerprint = nil
        source = nil; media = nil; project = nil; result = nil; showEditor = false
        start = 0; end = 0; roi = .fullFrame; prepareScore = true; generateSideSwitchMarkers = false
        error = nil; progress = 0; status = "Choose a game video"
    }
    func analyze() { enqueueAnalysis() }
    func deleteCurrentProject() {
        guard !busy, let current = project else { return }
        guard !(queue.unfinishedJobs + queue.preparingJobs).contains(where: { $0.projectId == current.id }) else {
            error = "Finish or remove this project's unfinished work in Processing queue before deleting it."
            return
        }
        saveTask?.cancel()
        do {
            // No suspension between deletion and clearing the current project:
            // a delayed autosave or outgoing editor cannot publish it again.
            try ProjectPersistence.delete(current, at: projectURL(current))
            project = nil
            beginNewProject()
            refreshFiles()
            status = "Project deleted. Choose a game video"
        } catch { self.error = String(describing: error) }
    }
    func cancel() { work?.cancel(); status = "Cancelling…" }
    func prepareScores() { enqueueScores() }
    func projectURL(_ project: ProjectDocument) -> URL {
        let name = project.id.filter { $0.isLetter || $0.isNumber || $0 == "-" }
        return documents.appendingPathComponent("project-\(name).volleyproject.json")
    }
    func saveProject() {
        guard var snapshot = project else { return }
        snapshot.draft.updatedAtMs = Int64(Date().timeIntervalSince1970 * 1000)
        project?.draft.updatedAtMs = snapshot.draft.updatedAtMs
        saveTask?.cancel()
        let destination = projectURL(snapshot)
        saveTask = Task {
            do {
                try await Task.sleep(for: .milliseconds(300))
                try Task.checkCancellation()
                // Only the small draft/metadata journal is encoded on this actor.
                // Full model feedback is staged off-main by ProjectPersistence.
                try ProjectPersistence.checkpoint(snapshot, to: destination)
                // Compact off-main after the durable checkpoint so the ordinary
                // project file also catches up without stalling editing.
                let prepared = try await ProjectPersistence.prepare(snapshot, to: destination)
                try Task.checkCancellation()
                guard let current = self.project, current.id == snapshot.id,
                      current.persistenceRevision == snapshot.persistenceRevision else { return }
                self.project = try ProjectPersistence.publish(prepared, current: current)
            } catch is CancellationError { } catch { self.error = String(describing: error) }
        }
    }
    @discardableResult func flushProject() -> Bool {
        saveTask?.cancel()
        guard let project else { return true }
        do { try ProjectPersistence.checkpoint(project, to: projectURL(project)); return true }
        catch { self.error = String(describing: error); return false }
    }
    /// Decode once off-main, then reject a read superseded while decoding by a
    /// queued score completion or a newer draft checkpoint.
    func loadLatestProject(_ url: URL) async throws -> ProjectDocument {
        try await readLatestProject(url).project
    }
    private func readLatestProject(_ url: URL) async throws -> ProjectPersistence.ReadResult {
        while true {
            let read = try await Task.detached(priority: .userInitiated) { try ProjectPersistence.read(from: url) }.value
            try Task.checkCancellation()
            if try ProjectPersistence.isCurrent(read) { return read }
        }
    }
    func openProject(_ url: URL) {
        guard !busy, flushProject() else { return }
        busy = true; error = nil; status = "Opening project"
        work = Task {
            do {
                let access = url.startAccessingSecurityScopedResource()
                defer { if access { url.stopAccessingSecurityScopedResource() } }
                var restored: ProjectDocument
                var initialRead: ProjectPersistence.ReadResult?
                if url.lastPathComponent.hasSuffix(".volleyproject.json") {
                    let read = try await readLatestProject(url); initialRead = read; restored = read.project
                }
                else { restored = try await Task.detached { try ProjectArchive.importData(Data(contentsOf: url), expectedFeatureNames: FeatureSchema.base) }.value }
                let fingerprint = restored.feedback?["source"]?["file"]?["sampledFingerprint"]?.string
                let resolvedAccess: SourceVideoAccess?
                do { resolvedAccess = try await sourceLibrary.resolve(name: restored.sourceName, fingerprint: fingerprint) }
                catch is CancellationError { throw CancellationError() }
                catch { resolvedAccess = nil }
                guard let videoAccess = resolvedAccess else {
                    reconnectJob = nil
                    reconnectReplacesProject = url.standardizedFileURL != projectURL(restored).standardizedFileURL
                    reconnectProject = restored; reconnecting = true
                    status = "Choose the source recording for \(restored.sourceName)"
                    busy = false
                    return
                }
                let recording = videoAccess.url
                if let original = restored.feedback?["source"] {
                    try await Task.detached { try ProjectArchive.verifySource(url: recording, source: original) }.value
                }
                let info = try await MediaDecoder.describe(recording)
                try Task.checkCancellation()
                let local = url.standardizedFileURL == projectURL(restored).standardizedFileURL
                if local, let initialRead, try !ProjectPersistence.isCurrent(initialRead) { restored = try await loadLatestProject(url) }
                guard abs(info.duration * 1000 - Double(restored.durationMs)) <= 2 else { throw AnalysisError.invalid("Recording duration differs from project") }
                let actualFingerprint = try await SourceVideoLibrary.inspect(videoAccess)
                if !local {
                    let staged = try await ProjectPersistence.prepare(restored, to: projectURL(restored))
                    restored = try ProjectPersistence.publish(staged)
                }
                player.pause(); player.replaceCurrentItem(with: nil)
                sourceAccess = videoAccess; selectedFingerprint = actualFingerprint
                project = restored; source = recording; media = info
                restoreSetup(restored)
                player.replaceCurrentItem(with: videoAccess.playerItem()); showEditor = true; refreshFiles(); busy = false; status = "Project ready"
            } catch { fail(error) }
        }
    }
    func beginReconnect() {
        guard !busy, let project else { return }
        reconnectJob = nil; reconnectReplacesProject = false; reconnectProject = project; reconnecting = true
    }
    func restoreSetup(_ restored: ProjectDocument) {
        start = Double(restored.gameWindow.startMs) / 1000; end = Double(restored.gameWindow.endMs) / 1000
        if let region = restored.feedback?["source"]?["featureRoi"],
           let x = region["x"]?.double, let y = region["y"]?.double,
           let width = region["width"]?.double, let height = region["height"]?.double {
            roi = AnalysisRegion(x: x, y: y, width: width, height: height)
        } else { roi = AnalysisRegion() }
    }
    func reconnectSource(_ url: URL) { reconnectSource(SourceVideoAccess(url)) }
    func reconnectSource(_ access: SourceVideoAccess) {
        if let job = reconnectJob { reconnectQueuedSource(access, job: job); return }
        guard !busy, var restored = reconnectProject else { return }
        let replacing = reconnectReplacesProject
        let url = access.url
        busy = true; status = "Checking source recording"
        work = Task {
            do {
                if let original = restored.feedback?["source"] {
                    try await Task.detached { try ProjectArchive.verifySource(url: url, source: original) }.value
                }
                let info = try await MediaDecoder.describe(url)
                guard abs(info.duration * 1000 - Double(restored.durationMs)) <= 2 else { throw ProjectError.invalid("Recording duration differs from this project") }
                let destination = url
                let fingerprint = try await SourceVideoLibrary.inspect(access)
                await sourceLibrary.remember(access, fingerprint: fingerprint)
                try Task.checkCancellation()
                // Score preparation may have committed while source metadata or
                // access checks were awaiting. Reconnect changes only source location.
                if !replacing, let current = project, current.id == restored.id {
                    restored = current
                } else if !replacing, FileManager.default.fileExists(atPath: projectURL(restored).path) {
                    restored = try await loadLatestProject(projectURL(restored))
                    if let current = project, current.id == restored.id { restored = current }
                }
                try Task.checkCancellation()
                restored.sourceName = access.filename
                saveTask?.cancel()
                if !replacing, FileManager.default.fileExists(atPath: projectURL(restored).path) {
                    try ProjectPersistence.checkpoint(restored, to: projectURL(restored))
                } else {
                    let staged = try await ProjectPersistence.prepare(restored, to: projectURL(restored))
                    restored = try ProjectPersistence.publish(staged)
                }
                player.pause(); player.replaceCurrentItem(with: nil)
                sourceAccess = access; selectedFingerprint = fingerprint
                project = restored; source = destination; media = info
                restoreSetup(restored)
                player.replaceCurrentItem(with: access.playerItem())
                refreshFiles(); showEditor = true; busy = false; reconnectProject = nil; reconnectReplacesProject = false
                status = "Source recording connected"
            } catch { fail(error) }
        }
    }
    func beginJobReconnect(_ job: ProcessingJob) {
        guard !busy, job.state.canResume else { return }
        reconnectProject = nil; reconnectJob = job; reconnecting = true
    }
    private func reconnectQueuedSource(_ access: SourceVideoAccess, job: ProcessingJob) {
        guard !busy else { return }
        busy = true; error = nil; status = "Checking source video"
        work = Task {
            do {
                let fingerprint = try await SourceVideoLibrary.inspect(access)
                guard fingerprint == job.sourceFingerprint else { throw ProjectError.invalid("This is a different video. Choose the original recording for this job.") }
                await sourceLibrary.remember(access, fingerprint: fingerprint)
                try Task.checkCancellation()
                reconnectJob = nil; busy = false; status = "Video re-linked; resuming processing"
                queue.resume(job.id)
            } catch { fail(error) }
        }
    }
    func exportProject() {
        guard let snapshot = project, !busy, flushProject() else { return }
        busy = true; status = "Preparing project export"
        work = Task {
            do {
                let worker = Task.detached(priority: .userInitiated) { () throws -> (ProjectDocument, Data) in
                    guard var feedback = snapshot.feedback else { throw AnalysisError.invalid("Project analysis is unavailable") }
                    try Task.checkCancellation()
                    try ProjectArchive.updateCorrections(&feedback, draft: snapshot.draft)
                    var updated = snapshot; updated.feedback = feedback
                    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
                    return (updated, try encoder.encode(feedback))
                }
                let (updated, bytes) = try await withTaskCancellationHandler(operation: { try await worker.value }, onCancel: { worker.cancel() })
                let staged = try await ProjectPersistence.prepare(updated, to: projectURL(updated))
                guard let current = project, current.id == snapshot.id else { throw ProjectError.invalid("Project changed during export") }
                saveTask?.cancel()
                project = try ProjectPersistence.publish(staged, current: current)
                let url = documents.appendingPathComponent("\(snapshot.id).model-feedback.json")
                try await Task.detached { try bytes.write(to: url, options: .atomic) }.value
                exportURL = url; shareItems = [url]; showShare = true; busy = false; status = "Project export ready"
            } catch { fail(error) }
        }
    }
    @Published var showExportDestination = false
    @Published var showExportFolder = false
    func exportVideo() { showExportDestination = true }
    #if DEBUG
    func golden() {
        guard !busy else { return }; busy = true; status = "Checking canonical features"
        work = Task {
            do {
                status = try await Task.detached { try AnalysisPipeline.golden() }.value
                busy = false
            } catch { fail(error) }
        }
    }
    func mediaCheck() {
        guard !busy else { return }; busy = true; status = "Checking Android rotation fixtures"
        work = Task {
            status = await DeviceMediaCheck.run()
            busy = false
        }
    }
    func exportCheck() {
        guard let source, !busy else { return }
        busy = true; progress = 0; status = "Checking native video export"
        let destination = documents.appendingPathComponent("device-export-check-\(UUID().uuidString.prefix(8)).mp4")
        work = Task {
            do {
                let output = try await DeviceExportCheck.run(source: source, destination: destination) { fraction, text in
                    Task { @MainActor in if self.busy { self.progress = fraction; self.status = text } }
                }
                exportURL = output.videoURL; status = "PASS: native export 13.500 s, score overlay and chapters"; busy = false
            } catch { fail(error) }
        }
    }
    #endif
    private func fail(_ error: Error) { busy = false; self.error = String(describing: error); status = error is CancellationError ? "Cancelled; features checkpointed" : "Operation failed" }
}

struct WorkspaceView: View {
    @ObservedObject var model: WorkspaceModel
    @State private var importing = false
    @State private var choosingVideoSource = false
    @State private var showingPhotos = false
    @State private var importingPhoto = false
    @State private var photoProgress = 0.0
    @State private var choosingReconnectSource = false
    @State private var photoForReconnect = false
    private enum PickerKind { case video, project, reconnect, folder }
    @State private var pickerKind = PickerKind.video
    private let paper = Color(red: 0.973, green: 0.969, blue: 0.933)
    private let ink = Color(red: 0.09, green: 0.22, blue: 0.18)
    static func editorBinding(_ current: Binding<ProjectDocument?>, snapshot: ProjectDocument) -> Binding<ProjectDocument> {
        // SwiftUI can read or write an outgoing editor's binding while dismissing
        // its menu and tearing down the view. Keep those reads valid after reset,
        // and never let a late edit restore the old project or overwrite a new one.
        Binding(get: {
            guard let project = current.wrappedValue, project.id == snapshot.id else { return snapshot }
            return project
        }, set: { updated in
            guard current.wrappedValue?.id == snapshot.id, updated.id == snapshot.id else { return }
            current.wrappedValue = updated
        })
    }
    var body: some View {
        Group {
        if model.showEditor, let project = model.project {
            EditorWorkspace(project: Self.editorBinding($model.project, snapshot: project), player: model.player,
                            onSave: { if model.project?.id == project.id { model.saveProject() } }, onExportProject: { model.exportProject() },
                            onExportVideo: { model.exportVideo() }, onBack: { model.flushProject(); model.showEditor = false; model.player.pause() },
                            savedProjects: model.projects, projectNames: model.projectNames, projectDates: model.projectDates,
                            onSelectProject: model.openProject, onNewProject: model.beginNewProject,
                            onManageStorage: { if model.flushProject() { model.showStorage = true } },
                            onDeleteProject: model.deleteCurrentProject, onShowQueue: { model.showQueue = true },
                            onPrepareScore: { model.prepareScores() }, onReconnectSource: { model.beginReconnect() })
                .id(project.id)
                .disabled(model.busy)
        } else {
            NewProjectWorkspace(model: model, chooseVideo: { choosingReconnectSource = false; choosingVideoSource = true }, openSaved: { pickerKind = .project; importing = true })
        }
        }
        .guidedTour(stage: model.showEditor ? .editor : .setup, sourceReady: model.media != nil,
                    scoreTrackingEnabled: model.project?.draft.scoreTracking["enabled"]?.bool ?? false)
        .modifier(ExportDestinationPresentation(model: model))
        .task { await SimulatorParityFixture.openIfRequested(model) }
        .safeAreaInset(edge: .bottom) {
            ProcessingQueueBar(queue: model.queue) { model.showQueue = true }
                .background(paper).foregroundStyle(ink).tint(Color(red: 47 / 255, green: 104 / 255, blue: 85 / 255))
        }
        .sheet(isPresented: $model.showQueue, onDismiss: {
            let action = model.afterQueueDismiss; model.afterQueueDismiss = nil; action?()
        }) {
            ProcessingQueueSheet(queue: model.queue, onRelink: { job in
                model.afterQueueDismiss = { model.beginJobReconnect(job) }; model.showQueue = false
            }).interfaceScaled()
        }
        .sheet(isPresented: $model.showStorage) { AppStorageView(model: model).interfaceScaled() }
        .sheet(isPresented: $model.showFileExport) {
            DocumentExportPicker(urls: model.shareItems) { model.showFileExport = false }
        }
        .overlay(alignment: .bottom) {
            if model.showEditor && model.busy {
                VStack { Text(model.status); ProgressView(value: model.progress); Button("Cancel") { model.cancel() } }
                    .padding().background(paper).overlay(Rectangle().stroke(ink)).padding()
            }
        }
        .onChange(of: model.reconnecting) { _, show in
            if show { choosingReconnectSource = true; choosingVideoSource = true }
        }
        .onChange(of: model.showExportFolder) { _, show in
            if show { pickerKind = .folder; importing = true }
        }
        .confirmationDialog(choosingReconnectSource ? "Re-link source video" : "Choose game video", isPresented: $choosingVideoSource, titleVisibility: .visible) {
            Button("Camera Roll") { photoForReconnect = choosingReconnectSource; photoProgress = 0; showingPhotos = true }
            Button("Files") { pickerKind = choosingReconnectSource ? .reconnect : .video; importing = true }
            Button("Cancel", role: .cancel) { model.reconnecting = false }
        }
        .sheet(isPresented: $showingPhotos) {
            PhotoVideoPicker(onLoadingState: { importingPhoto = $0 }, onProgress: { photoProgress = $0 }) { result in
                showingPhotos = false
                model.reconnecting = false
                switch result {
                case .success(let selection):
                    if let selection {
                        let access = SourceVideoAccess(photo: selection)
                        if photoForReconnect { model.reconnectSource(access) }
                        else { model.select(access) }
                    }
                case .failure(let error): model.error = error.localizedDescription
                }
            }
            .interactiveDismissDisabled(importingPhoto)
            .overlay {
                if importingPhoto {
                    VStack(spacing: 16) {
                        if photoProgress > 0 && photoProgress < 1 { ProgressView(value: photoProgress) }
                        else { ProgressView() }
                        Text(photoProgress > 0 && photoProgress < 1 ? "Downloading original from iCloud · \(Int(photoProgress * 100))%" : "Opening video from Camera Roll").font(.headline)
                        Text("Uses the original recording. Photos trims and filters are not applied.").font(.footnote)
                        Button("Cancel") { showingPhotos = false }.accessibilityIdentifier("cancelPhotoVideo")
                    }.padding(24).frame(maxWidth: .infinity, maxHeight: .infinity)
                        .background(paper.opacity(0.97)).accessibilityIdentifier("photoVideoProgress")
                        .interfaceScaled()
                }
            }
        }
        .fileImporter(isPresented: $importing,
                      allowedContentTypes: pickerKind == .project ? [.json] : pickerKind == .folder ? [.folder] : [.movie],
                      allowsMultipleSelection: false) { response in
            model.reconnecting = false; model.showExportFolder = false
            switch response {
            case .success(let urls):
                guard let url = urls.first else { return }
                switch pickerKind {
                case .video: model.select(url)
                case .project: model.openProject(url)
                case .reconnect: model.reconnectSource(url)
                case .folder: model.exportToFolder(url)
                }
            case .failure(let error):
                if (error as NSError).code != NSUserCancelledError { model.error = error.localizedDescription }
            }
        }
        .onChange(of: importing) { _, show in
            if !show { model.reconnecting = false; model.showExportFolder = false }
        }
        .sheet(isPresented: $model.showShare) {
            ActivityShare(items: model.shareItems)
        }
        .alert("Unable to complete", isPresented: Binding(get: { model.error != nil }, set: { if !$0 { model.error = nil } })) {
            Button("OK") { model.error = nil }
        } message: { Text(model.error ?? "") }
    }
}

struct ActivityShare: UIViewControllerRepresentable {
    var items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController { UIActivityViewController(activityItems: items, applicationActivities: nil) }
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) { }
}
