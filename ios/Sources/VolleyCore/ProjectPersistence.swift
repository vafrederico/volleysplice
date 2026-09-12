import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

/// Full project JSON stays self-contained. Draft checkpoints carry no feedback
/// arrays. Full encoding/staging occurs off-main; publication is a same-volume
/// rename guarded by the base revision. A two-entry journal makes the rename
/// crash-safe without losing edits made while encoding was in progress.
public enum ProjectPersistence {
    fileprivate struct FileVersion: Codable, Equatable, Sendable {
        let inode: UInt64, size: UInt64, modified: Date
    }
    private struct Identity: Equatable {
        let id: String, sourceRevision: String
        let durationMs: Int64
        let gameWindow: TimeRange
        init(_ project: ProjectDocument) {
            id = project.id; sourceRevision = project.draft.sourceRevision
            durationMs = project.durationMs; gameWindow = project.gameWindow
        }
    }
    private struct Checkpoint: Codable, Sendable {
        let projectId: String, sourceRevision: String
        let durationMs: Int64
        let gameWindow: TimeRange
        let baseRevision: String?
        let legacyFileVersion: FileVersion?
        let sourceName: String
        let sourceBookmark: Data?
        let draft: EditorDraft
        init(_ project: ProjectDocument, legacyVersion: FileVersion? = nil) {
            projectId = project.id; sourceRevision = project.draft.sourceRevision
            durationMs = project.durationMs; gameWindow = project.gameWindow
            baseRevision = project.persistenceRevision; sourceName = project.sourceName
            legacyFileVersion = project.persistenceRevision == nil ? legacyVersion : nil
            sourceBookmark = project.sourceBookmark; draft = project.draft
        }
        func matches(_ project: ProjectDocument, version: FileVersion?) -> Bool {
            projectId == project.id && sourceRevision == project.draft.sourceRevision && durationMs == project.durationMs &&
                gameWindow == project.gameWindow && baseRevision == project.persistenceRevision &&
                (baseRevision != nil || (legacyFileVersion != nil && legacyFileVersion == version))
        }
        func apply(to project: inout ProjectDocument) {
            project.draft = draft; project.sourceName = sourceName; project.sourceBookmark = sourceBookmark
        }
    }
    private struct Journal: Codable {
        let schema: String
        let entries: [Checkpoint]
        init(_ entries: [Checkpoint]) { schema = "volleysplice-draft-checkpoint-v1"; self.entries = entries }
    }
    private final class State: @unchecked Sendable {
        let lock = NSLock()
        var knownBases: [String: (FileVersion, String?, Identity)] = [:]
    }
    private static let state = State()
    public struct ReadResult: Sendable {
        public let project: ProjectDocument
        fileprivate let url: URL
        fileprivate let baseVersion: FileVersion?
        fileprivate let journalVersion: FileVersion?
    }
    public struct ListMetadata: Sendable {
        public let sourceName: String
        public let updatedAtMs: Int64
    }
    public final class PreparedWrite: @unchecked Sendable {
        public let project: ProjectDocument
        fileprivate let destination: URL, staging: URL
        fileprivate let expectedFile: FileVersion?
        fileprivate let expectedRevision: String?
        fileprivate init(project: ProjectDocument, destination: URL, staging: URL, expectedFile: FileVersion?, expectedRevision: String?) {
            self.project = project; self.destination = destination; self.staging = staging
            self.expectedFile = expectedFile; self.expectedRevision = expectedRevision
        }
        deinit { try? FileManager.default.removeItem(at: staging) }
    }
    public static func checkpointURL(for url: URL) -> URL {
        url.deletingLastPathComponent().appendingPathComponent("." + url.lastPathComponent + ".draft-checkpoint.json")
    }
    private static func version(_ url: URL) throws -> FileVersion? {
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        return FileVersion(inode: (attributes[.systemFileNumber] as? NSNumber)?.uint64Value ?? 0,
                           size: (attributes[.size] as? NSNumber)?.uint64Value ?? 0,
                           modified: attributes[.modificationDate] as? Date ?? .distantPast)
    }
    private static func journalData(_ url: URL) throws -> Data? {
        let journal = checkpointURL(for: url)
        guard let attributes = try version(journal) else { return nil }
        guard attributes.size <= 16 * 1024 * 1024 else { throw ProjectError.invalid("Draft checkpoint is too large") }
        return try Data(contentsOf: journal)
    }
    private static func decodeJournal(_ data: Data?) throws -> Journal? {
        guard let data else { return nil }
        let journal = try JSONDecoder().decode(Journal.self, from: data)
        guard journal.schema == "volleysplice-draft-checkpoint-v1", journal.entries.count <= 2 else {
            throw ProjectError.invalid("Invalid draft checkpoint")
        }
        return journal
    }
    private static func writeJournal(_ journal: Journal, to url: URL) throws {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(journal).write(to: checkpointURL(for: url), options: .atomic)
    }
    public static func read(from url: URL) throws -> ReadResult {
        state.lock.lock()
        let data: Data, checkpointData: Data?, baseVersion: FileVersion?, journalVersion: FileVersion?
        do {
            data = try Data(contentsOf: url, options: .mappedIfSafe)
            checkpointData = try journalData(url)
            baseVersion = try version(url); journalVersion = try version(checkpointURL(for: url))
            state.lock.unlock()
        } catch { state.lock.unlock(); throw error }
        // Parse large feature payloads outside the lock, on the caller's worker.
        var project = try JSONDecoder().decode(ProjectDocument.self, from: data)
        if let entry = try decodeJournal(checkpointData)?.entries.last(where: { $0.matches(project, version: baseVersion) }) { entry.apply(to: &project) }
        try project.validate()
        state.lock.lock(); defer { state.lock.unlock() }
        if let baseVersion, try version(url) == baseVersion { state.knownBases[url.standardizedFileURL.path] = (baseVersion, project.persistenceRevision, Identity(project)) }
        return ReadResult(project: project, url: url, baseVersion: baseVersion, journalVersion: journalVersion)
    }
    public static func isCurrent(_ read: ReadResult) throws -> Bool {
        state.lock.lock(); defer { state.lock.unlock() }
        return try version(read.url) == read.baseVersion && version(checkpointURL(for: read.url)) == read.journalVersion
    }
    /// Call on a worker for legacy files. Managed files normally need only their
    /// small journal; cold discovery falls back to a validated full read.
    public static func listMetadata(from url: URL) throws -> ListMetadata {
        state.lock.lock()
        do {
            if let known = state.knownBases[url.standardizedFileURL.path], try version(url) == known.0,
               let entry = try decodeJournal(journalData(url))?.entries.last(where: {
                   $0.baseRevision == known.1 && $0.projectId == known.2.id && $0.sourceRevision == known.2.sourceRevision &&
                       $0.durationMs == known.2.durationMs && $0.gameWindow == known.2.gameWindow &&
                       ($0.baseRevision != nil || $0.legacyFileVersion == known.0)
               }) {
                state.lock.unlock()
                return ListMetadata(sourceName: entry.sourceName, updatedAtMs: entry.draft.updatedAtMs)
            }
            state.lock.unlock()
        } catch { state.lock.unlock(); throw error }
        let project = try read(from: url).project
        return ListMetadata(sourceName: project.sourceName, updatedAtMs: project.draft.updatedAtMs)
    }
    /// Must follow a read/publication of this base. Stale project instances cannot
    /// checkpoint over newer model feedback or over an externally replaced file.
    public static func checkpoint(_ project: ProjectDocument, to url: URL) throws {
        try project.validate()
        state.lock.lock(); defer { state.lock.unlock() }
        guard let known = state.knownBases[url.standardizedFileURL.path],
              known.1 == project.persistenceRevision, known.2 == Identity(project), try version(url) == known.0 else {
            throw ProjectError.invalid("Project changed while saving; reload its latest revision")
        }
        try writeJournal(Journal([Checkpoint(project, legacyVersion: known.0)]), to: url)
    }
    public static func prepare(_ project: ProjectDocument, to destination: URL) async throws -> PreparedWrite {
        let worker = Task.detached(priority: .userInitiated) { try stage(project, to: destination) }
        return try await withTaskCancellationHandler(operation: { try await worker.value }, onCancel: { worker.cancel() })
    }
    /// Deletes only the managed project and its draft checkpoint. Staged saves
    /// and stale editor instances must not recreate or delete another revision.
    public static func delete(_ project: ProjectDocument, at url: URL) throws {
        state.lock.lock(); defer { state.lock.unlock() }
        let key = url.standardizedFileURL.path
        guard let known = state.knownBases[key], known.1 == project.persistenceRevision,
              known.2 == Identity(project), try version(url) == known.0 else {
            throw ProjectError.invalid("Project changed before deletion; reload its latest revision")
        }
        let journal = checkpointURL(for: url), backup = try journalData(url)
        if backup != nil { try FileManager.default.removeItem(at: journal) }
        do { try FileManager.default.removeItem(at: url) }
        catch {
            if let backup { try backup.write(to: journal, options: .atomic) }
            throw error
        }
        state.knownBases.removeValue(forKey: key)
    }
    /// Synchronous staging is also exposed for deterministic storage tests. App
    /// code uses prepare, which performs validation/encoding/writing off-main.
    public static func stage(_ project: ProjectDocument, to destination: URL) throws -> PreparedWrite {
        try Task.checkCancellation(); try project.validate()
        state.lock.lock()
        let expected: FileVersion?
        do { expected = try version(destination); state.lock.unlock() }
        catch { state.lock.unlock(); throw error }
        var staged = project; staged.persistenceRevision = UUID().uuidString
        let staging = destination.deletingLastPathComponent().appendingPathComponent(".project-save-\(UUID().uuidString).json")
        var completed = false
        defer { if !completed { try? FileManager.default.removeItem(at: staging) } }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let bytes = try encoder.encode(staged)
        try Task.checkCancellation()
        try bytes.write(to: staging, options: .withoutOverwriting)
        let handle = try FileHandle(forWritingTo: staging)
        do { try handle.synchronize(); try handle.close() } catch { try? handle.close(); throw error }
        try Task.checkCancellation()
        completed = true
        return PreparedWrite(project: staged, destination: destination, staging: staging,
                             expectedFile: expected, expectedRevision: project.persistenceRevision)
    }
    /// Publish only after re-reading the latest draft on the app actor. The
    /// optional mutation merges completed model results into that latest draft.
    /// Passing nil explicitly replaces/imports a project without reusing its old journal.
    @discardableResult public static func publish(_ prepared: PreparedWrite, current: ProjectDocument? = nil,
        updateDraft: (inout EditorDraft) throws -> Void = { _ in }) throws -> ProjectDocument {
        try Task.checkCancellation()
        state.lock.lock(); defer { state.lock.unlock() }
        guard try version(prepared.destination) == prepared.expectedFile else { throw ProjectError.invalid("A newer project save finished first; retry this operation") }
        var published = prepared.project
        if let current {
            guard current.id == published.id, current.draft.sourceRevision == published.draft.sourceRevision,
                  current.durationMs == published.durationMs, current.gameWindow == published.gameWindow,
                  current.persistenceRevision == prepared.expectedRevision else {
                throw ProjectError.invalid("Project identity or model feedback changed while saving")
            }
            published.draft = current.draft; published.sourceName = current.sourceName; published.sourceBookmark = current.sourceBookmark
        }
        try updateDraft(&published.draft); try published.validate()
        let previous = try journalData(prepared.destination)
        // Before rename the old base uses its current checkpoint. After rename
        // the new base uses pending, even if the process dies before compaction.
        let priorEntries = try decodeJournal(previous)?.entries ?? []
        let previousCheckpoint = current.map { Checkpoint($0, legacyVersion: prepared.expectedFile) } ?? priorEntries.last
        let entries = (previousCheckpoint.map { [$0] } ?? []) + [Checkpoint(published)]
        try writeJournal(Journal(entries), to: prepared.destination)
        do {
            #if canImport(Darwin) || canImport(Glibc)
            // rename(2) is atomic on one filesystem; failure leaves both files
            // intact. Staging always lives next to the destination.
            let result = prepared.staging.withUnsafeFileSystemRepresentation { source in
                prepared.destination.withUnsafeFileSystemRepresentation { destination in rename(source!, destination!) }
            }
            guard result == 0 else { throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno)) }
            #else
            if prepared.expectedFile == nil { try FileManager.default.moveItem(at: prepared.staging, to: prepared.destination) }
            else { _ = try FileManager.default.replaceItemAt(prepared.destination, withItemAt: prepared.staging) }
            #endif
        } catch {
            if let previous { try? previous.write(to: checkpointURL(for: prepared.destination), options: .atomic) }
            else { try? FileManager.default.removeItem(at: checkpointURL(for: prepared.destination)) }
            throw error
        }
        if let stamp = try? version(prepared.destination) { state.knownBases[prepared.destination.standardizedFileURL.path] = (stamp, published.persistenceRevision, Identity(published)) }
        // The two-entry journal is already sufficient for recovery. A failed
        // optional compaction must not turn a durable publication into failure.
        try? writeJournal(Journal([Checkpoint(published)]), to: prepared.destination)
        return published
    }
}
