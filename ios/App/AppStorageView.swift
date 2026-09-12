import SwiftUI

struct AppStorageView: View {
    @ObservedObject var model: WorkspaceModel
    @Environment(\.dismiss) private var dismiss
    @State private var items: [AppStorageInventory.Item] = []
    @State private var loading = false
    @State private var pending: [AppStorageInventory.Item] = []
    @State private var confirming = false
    @State private var failure: String?
    @State private var removedMessage: String?

    private var roots: AppStorageInventory.Roots {
        let fm = FileManager.default
        let support = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return .init(documents: model.documents, features: model.cacheFolder,
                     queue: support.appendingPathComponent("ProcessingJobs"), temporary: fm.temporaryDirectory,
                     support: support, caches: fm.urls(for: .cachesDirectory, in: .userDomainMask)[0])
    }
    private var protection: AppStorageInventory.Protection {
        var urls = [model.source].compactMap { $0 }
        if let project = model.project {
            urls.append(model.projectURL(project))
            urls.append(ProjectPersistence.checkpointURL(for: model.projectURL(project)))
        }
        let jobs = model.queue.unfinishedJobs + model.queue.preparingJobs
        for job in jobs {
            urls.append(model.documents.appendingPathComponent(job.sourceName))
            let name = job.projectId.filter { $0.isLetter || $0.isNumber || $0 == "-" }
            let project = model.documents.appendingPathComponent("project-\(name).volleyproject.json")
            urls += [project, ProjectPersistence.checkpointURL(for: project)]
            urls += job.outputNames.map { model.documents.appendingPathComponent($0) }
            // Output names may not have reached the ledger before a cancellation.
            urls += ["VolleySplice-\(job.id).mp4", YouTubeChapters.filename("VolleySplice-\(job.id).mp4"), ".photo-export-\(job.id).json"].map { model.documents.appendingPathComponent($0) }
        }
        return .init(urls: urls, queueJobIDs: Set(jobs.map(\.id)), blocksRemoval: model.busy || model.queue.hasPendingWork)
    }
    var body: some View {
        NavigationStack {
            List {
                Section {
                    LabeledContent("App-managed files", value: size(items.reduce(0) { $0 + $1.bytes }))
                        .font(.headline).accessibilityIdentifier("appStorageTotal")
                    Text("Manage files stored inside VolleySplice. Videos linked from locations outside the app and originals in Camera Roll are not included. File sizes may differ from iOS storage totals, which also include the installed app and system overhead.")
                        .font(.footnote).foregroundStyle(SetupPalette.muted)
                    Text("The open video, open project, and unfinished queue inputs are protected. Start a new video or remove an unfinished job from the queue before deleting its files.")
                        .font(.footnote).foregroundStyle(SetupPalette.muted)
                    let deletedBytes = items.filter { $0.category == .recentlyDeleted }.reduce(0) { $0 + $1.bytes }
                    if deletedBytes > 0 {
                        Label("Recently deleted files are still using \(size(deletedBytes)). Review them below to permanently remove them.", systemImage: "trash")
                            .font(.footnote).accessibilityIdentifier("recentlyDeletedStorageSummary")
                    }
                    if protection.blocksRemoval {
                        Label("Wait for processing or importing to finish before removing app data.", systemImage: "hourglass")
                            .font(.footnote)
                    }
                    if let removedMessage { Text(removedMessage).font(.footnote).accessibilityIdentifier("appStorageRemovalResult") }
                    if let failure { Text(failure).font(.footnote).foregroundStyle(.red).accessibilityIdentifier("appStorageError") }
                }
                ForEach(AppStorageInventory.Category.allCases, id: \.self) { category in
                    let categoryItems = items.filter { $0.category == category }
                    Section {
                        DisclosureGroup {
                            if categoryItems.isEmpty { Text("No files").foregroundStyle(SetupPalette.muted) }
                            ForEach(categoryItems) { item in fileRow(item) }
                        } label: {
                            LabeledContent(category.rawValue, value: size(categoryItems.reduce(0) { $0 + $1.bytes }))
                                .accessibilityIdentifier("appStorageCategory-\(category.rawValue)")
                        }
                        let removable = categoryItems.filter(\.removable)
                        if !removable.isEmpty {
                            Button("\(category == .recentlyDeleted ? "Permanently remove" : "Remove") \(removable.count) available \(removable.count == 1 ? "file" : "files") (\(size(removable.reduce(0) { $0 + $1.bytes })))", role: .destructive) {
                                pending = removable; confirming = true
                            }
                            .disabled(loading || protection.blocksRemoval)
                            .accessibilityIdentifier("clearAppStorage-\(category.rawValue)")
                        }
                    } footer: { Text(category.explanation) }
                }
            }
            .scrollContentBackground(.hidden).background(SetupPalette.paper)
            .foregroundStyle(SetupPalette.ink).tint(SetupPalette.green)
            .navigationTitle("App storage").navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Done") { dismiss() }.disabled(loading).accessibilityIdentifier("appStorageDone") }
                ToolbarItem(placement: .primaryAction) {
                    if loading { ProgressView() }
                    else { Button("Refresh", systemImage: "arrow.clockwise") { refresh() }.accessibilityIdentifier("refreshAppStorage") }
                }
            }
            .alert("\(pending.first?.category == .recentlyDeleted ? "Permanently remove" : "Remove") \(pending.count == 1 ? pending.first?.name ?? "file" : "\(pending.count) files")?", isPresented: $confirming) {
                Button("Cancel", role: .cancel) { pending = [] }
                Button("Remove", role: .destructive) { removePending() }
            } message: {
                Text("Remove \(size(pending.reduce(0) { $0 + $1.bytes })) from VolleySplice? This cannot be undone.\n\n\(pending.count == 1 ? "Location: \(pending.first?.location ?? "App storage")\n\n" : "")\(pending.first?.category.explanation ?? "")")
            }
            .task { refresh() }
        }
        .interactiveDismissDisabled(loading)
    }
    private func fileRow(_ item: AppStorageInventory.Item) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 3) {
                Text(item.name).font(.subheadline).lineLimit(2)
                Text(item.location).font(.caption).foregroundStyle(SetupPalette.muted)
                    .accessibilityIdentifier("appStorageLocation-\(item.location)/\(item.name)")
                Text(size(item.bytes)).font(.caption).foregroundStyle(SetupPalette.muted)
                if let reason = item.protectedReason { Label(reason, systemImage: "lock").font(.caption).foregroundStyle(SetupPalette.muted) }
            }
            Spacer(minLength: 12)
            if item.removable {
                Button(role: .destructive) { pending = [item]; confirming = true } label: { Image(systemName: "trash") }
                    .buttonStyle(.borderless).disabled(loading || protection.blocksRemoval)
                    .accessibilityLabel("Remove \(item.name) from \(item.location)").accessibilityIdentifier("removeAppStorage-\(item.name)")
            }
        }
    }
    private func size(_ bytes: Int64) -> String { ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file) }
    private func refresh() {
        guard !loading else { return }
        loading = true
        let roots = roots, protection = protection
        Task { @MainActor in
            do { items = try await Task.detached(priority: .utility) { try AppStorageInventory.scan(roots: roots, protection: protection) }.value; failure = nil }
            catch { failure = error.localizedDescription }
            loading = false
        }
    }
    private func removePending() {
        guard !loading else { return }
        let selected = pending, roots = roots, protection = protection
        pending = []; loading = true; removedMessage = nil
        Task { @MainActor in
            do {
                try await Task.detached(priority: .utility) { try AppStorageInventory.remove(selected, roots: roots, protection: protection) }.value
                removedMessage = "Removed \(selected.count) \(selected.count == 1 ? "file" : "files") (\(size(selected.reduce(0) { $0 + $1.bytes })))"
                failure = nil
            } catch { failure = error.localizedDescription }
            model.refreshFiles()
            // Preserve any deletion error while refreshing the visible sizes.
            do { items = try await Task.detached(priority: .utility) { try AppStorageInventory.scan(roots: roots, protection: protection) }.value }
            catch { failure = error.localizedDescription }
            loading = false
        }
    }
}
