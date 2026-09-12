import Foundation

/// Only app-owned roots are scanned. Bookmarked Files/Photos locations are never
/// resolved here, and symbolic links are neither followed nor offered for removal.
public enum AppStorageInventory {
    public enum Category: String, CaseIterable, Sendable {
        case recentlyDeleted = "Recently deleted", recordings = "Recordings", features = "Feature cache", exports = "Exports"
        case projects = "Projects", temporary = "Temporary files", queue = "Queue data", other = "Other app data"
        public var explanation: String {
            switch self {
            case .recentlyDeleted: "Files in VolleySplice's Recently Deleted folder still take up space. Permanently removing them frees this space and prevents restoring them from Recently Deleted."
            case .recordings: "Copies stored by VolleySplice. Removing a copy keeps project edits; reopen the original video when requested."
            case .features: "Reusable analysis features. Removing these frees space; a future analysis may need to calculate them again."
            case .exports: "Exports and analysis reports stored inside VolleySplice. Copies already saved to Files or Camera Roll stay there."
            case .projects: "Saved projects, analysis results, and your edits. Export a project first if you want to keep it."
            case .temporary: "Temporary files left by imports, project saves, or exports."
            case .queue: "Inputs for unfinished processing are protected. Unused snapshots can be removed."
            case .other: "Saved file links, app settings, and system-managed caches. These files are shown for storage accounting and cannot be removed here."
            }
        }
    }
    public struct Roots: Sendable {
        public let documents: URL, features: URL, queue: URL, temporary: URL
        public let support: URL?, caches: URL?
        public init(documents: URL, features: URL, queue: URL, temporary: URL, support: URL? = nil, caches: URL? = nil) {
            self.documents = documents.resolvingSymlinksInPath().standardizedFileURL; self.features = features.resolvingSymlinksInPath().standardizedFileURL
            self.queue = queue.resolvingSymlinksInPath().standardizedFileURL; self.temporary = temporary.resolvingSymlinksInPath().standardizedFileURL
            self.support = support?.resolvingSymlinksInPath().standardizedFileURL; self.caches = caches?.resolvingSymlinksInPath().standardizedFileURL
        }
        // Specific managed directories take precedence over their broader
        // accounting-only parents. Traversal deduplicates overlapping roots.
        var all: [URL] { [documents, features, queue, temporary] + [support, caches].compactMap { $0 } }
    }
    public struct Protection: Sendable {
        public var paths: Set<String>
        public var queueJobIDs: Set<String>
        public var blocksRemoval: Bool
        public init(urls: [URL] = [], queueJobIDs: Set<String> = [], blocksRemoval: Bool = false) {
            paths = Set(urls.map { $0.resolvingSymlinksInPath().standardizedFileURL.path }); self.queueJobIDs = queueJobIDs
            self.blocksRemoval = blocksRemoval
        }
    }
    struct FileStamp: Equatable, Sendable {
        let url: URL, inode: UInt64, size: Int64, modified: Date
    }
    public struct Item: Identifiable, Sendable {
        public let id: String, name: String, location: String, category: Category, bytes: Int64
        public let protectedReason: String?
        let files: [FileStamp]
        public var removable: Bool { protectedReason == nil }
    }
    public static func scan(roots: Roots, protection: Protection = .init()) throws -> [Item] {
        var collected: [(FileStamp, Category)] = []
        var visitedFolders: Set<String> = []
        for root in roots.all where FileManager.default.fileExists(atPath: root.path) {
            // A directory URL created before the directory exists can carry a
            // different trailing-slash hint. Compare filesystem paths, while
            // still rejecting paths redirected through a new symbolic link.
            guard root.resolvingSymlinksInPath().standardizedFileURL.path == root.path else { continue }
            try collect(root, root: root, roots: roots, visited: &visitedFolders, into: &collected)
        }
        let projects = Set(collected.filter { $0.0.url.lastPathComponent.hasSuffix(".volleyproject.json") }.map { $0.0.url.path })
        var groups: [String: [(FileStamp, Category)]] = [:]
        for entry in collected {
            let url = entry.0.url, name = url.lastPathComponent
            var key = url.path
            if name.hasPrefix("."), name.hasSuffix(".volleyproject.json.draft-checkpoint.json") {
                let base = url.deletingLastPathComponent().appendingPathComponent(String(name.dropFirst().dropLast(".draft-checkpoint.json".count)))
                if projects.contains(base.path) { key = base.path }
            }
            groups[key, default: []].append(entry)
        }
        return groups.map { key, entries in
            let category = entries.first(where: { $0.0.url.path == key })?.1 ?? entries[0].1
            let files = entries.map(\.0)
            let reason = protectionReason(files: files, category: category, roots: roots, protection: protection)
            return Item(id: key, name: URL(fileURLWithPath: key).lastPathComponent,
                        location: location(URL(fileURLWithPath: key).deletingLastPathComponent(), roots: roots), category: category,
                        bytes: files.reduce(0) { $0 + $1.size }, protectedReason: reason, files: files)
        }.sorted { $0.bytes == $1.bytes ? $0.name < $1.name : $0.bytes > $1.bytes }
    }
    /// Revalidate ownership, protection and file identity immediately before each
    /// removal. A replaced file or newly protected input is never deleted.
    public static func remove(_ items: [Item], roots: Roots, protection: Protection) throws {
        guard !protection.blocksRemoval else { throw ProjectError.invalid("Wait for processing or importing to finish before removing app data") }
        for item in items {
            guard item.removable, protectionReason(files: item.files, category: item.category, roots: roots, protection: protection) == nil else {
                throw ProjectError.invalid("This file is in use: \(item.name)")
            }
            for file in item.files {
                guard isOwnedRegularFile(file.url, roots: roots), try stamp(file.url) == file else {
                    throw ProjectError.invalid("Storage changed. Refresh the list before removing \(item.name)")
                }
            }
        }
        for item in items {
            for file in item.files {
                guard isOwnedRegularFile(file.url, roots: roots), try stamp(file.url) == file else {
                    throw ProjectError.invalid("Storage changed. Refresh the list before removing \(item.name)")
                }
                try FileManager.default.removeItem(at: file.url)
            }
        }
    }
    private static func protectionReason(files: [FileStamp], category: Category, roots: Roots, protection: Protection) -> String? {
        if category == .other { return "App metadata or system-managed cache" }
        if files.contains(where: { protection.paths.contains($0.url.path) }) { return "Used by the open video, project, or unfinished queue" }
        if category == .queue {
            for file in files {
                let name = file.url.lastPathComponent
                guard name.hasSuffix(".project.json") else { return "Processing history and delivery receipts" }
                let id = String(name.dropLast(".project.json".count))
                guard UUID(uuidString: id) != nil else { return "Processing history and delivery receipts" }
                if protection.queueJobIDs.contains(id) { return "Input for unfinished processing" }
            }
        }
        return nil
    }
    private static func collect(_ folder: URL, root: URL, roots: Roots, visited: inout Set<String>, into collected: inout [(FileStamp, Category)]) throws {
        guard visited.insert(folder.standardizedFileURL.path).inserted else { return }
        for url in try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: [.isSymbolicLinkKey, .isDirectoryKey, .isRegularFileKey]) {
            let values = try url.resourceValues(forKeys: [.isSymbolicLinkKey, .isDirectoryKey, .isRegularFileKey])
            guard values.isSymbolicLink != true, url.resolvingSymlinksInPath().standardizedFileURL.path == url.standardizedFileURL.path else { continue }
            if values.isDirectory == true { try collect(url, root: root, roots: roots, visited: &visited, into: &collected) }
            else if values.isRegularFile == true { collected.append((try stamp(url), category(url, root: root, roots: roots))) }
        }
    }
    private static func category(_ url: URL, root: URL, roots: Roots) -> Category {
        // Foundation enumeration may return /private/var even when its input
        // root is standardized as /var. Stamp/location already standardize;
        // classification must use that same representation before prefix tests.
        let path = url.standardizedFileURL.path
        // Check the exact app Documents trash subtree before file extensions;
        // otherwise deleted videos look like current recordings and remain
        // indistinguishable when their names match.
        if root.path == roots.documents.path, path.hasPrefix(roots.documents.path + "/.Trash/") { return .recentlyDeleted }
        if root.path == roots.features.path { return .features }
        if root.path == roots.queue.path { return url.lastPathComponent.hasPrefix(".snapshot-") ? .temporary : .queue }
        if root.path == roots.temporary.path {
            let relative = String(path.dropFirst(root.path.count + 1))
            return ["photo-video-", ".volleysplice-export-", "recording-import-"].contains(where: { relative.hasPrefix($0) }) ? .temporary : .other
        }
        if root.path == roots.support?.path || root.path == roots.caches?.path { return .other }
        let name = url.lastPathComponent
        if name.hasPrefix(".photo-export-") { return .queue }
        if name.hasSuffix(".volleyproject.json") || name.hasSuffix(".volleyproject.json.draft-checkpoint.json") { return .projects }
        if name.hasPrefix(".project-save-") || path.contains("/.recording-import-") || path.contains("/.volleysplice-export-") { return .temporary }
        if ["mp4", "mov", "m4v"].contains(url.pathExtension.lowercased()) { return name.hasPrefix("VolleySplice-") ? .exports : .recordings }
        if name.hasSuffix(".model-feedback.json") || name.hasSuffix(".chapters.txt") || name.hasSuffix("-youtube-chapters.txt") || name.hasPrefix("analysis-") { return .exports }
        return .other
    }
    private static func location(_ folder: URL, roots: Roots) -> String {
        let labeled: [(URL?, String)] = [
            (roots.documents, "Documents"), (roots.features, "Library/Caches/features"),
            (roots.queue, "Library/Application Support/ProcessingJobs"), (roots.temporary, "Temporary files"),
            (roots.support, "Library/Application Support"), (roots.caches, "Library/Caches")
        ]
        for (root, label) in labeled {
            guard let root else { continue }
            if folder.path == root.path { return label }
            if folder.path.hasPrefix(root.path + "/") { return label + String(folder.path.dropFirst(root.path.count)) }
        }
        return "App storage"
    }
    private static func isOwnedRegularFile(_ url: URL, roots: Roots) -> Bool {
        guard roots.all.contains(where: { url.path.hasPrefix($0.path + "/") && $0.resolvingSymlinksInPath().standardizedFileURL.path == $0.path }),
              url.standardizedFileURL.path == url.path, url.resolvingSymlinksInPath().standardizedFileURL.path == url.path,
              let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey]) else { return false }
        return values.isRegularFile == true && values.isSymbolicLink != true
    }
    private static func stamp(_ url: URL) throws -> FileStamp {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        return FileStamp(url: url.standardizedFileURL,
                         inode: (attributes[.systemFileNumber] as? NSNumber)?.uint64Value ?? 0,
                         size: (attributes[.size] as? NSNumber)?.int64Value ?? 0,
                         modified: attributes[.modificationDate] as? Date ?? .distantPast)
    }
}
