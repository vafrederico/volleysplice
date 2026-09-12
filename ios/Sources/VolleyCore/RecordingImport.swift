import Foundation

/// Import picker media without replacing an existing source with the same name.
/// All file IO, including cancellation cleanup, runs outside the main actor.
public enum RecordingImport {
    public struct Result: Sendable {
        public let url: URL
        public var createdCopy: Bool { identity != nil }
        fileprivate let identity: FileIdentity?
    }
    fileprivate struct FileIdentity: Equatable, Sendable {
        let volume: UInt64, file: UInt64
        init(_ url: URL) throws {
            let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
            guard let volume = attributes[.systemNumber] as? NSNumber,
                  let file = attributes[.systemFileNumber] as? NSNumber else {
                throw CocoaError(.fileReadUnknown)
            }
            self.volume = volume.uint64Value; self.file = file.uint64Value
        }
    }

    public static func prepare(source: URL, documents: URL) async throws -> Result {
        try await prepare(source: source, documents: documents, beforeCopy: {})
    }

    /// Test seam gates the actual detached writer; no app-specific test mode.
    static func prepare(source: URL, documents: URL, beforeCopy: @escaping @Sendable () -> Void) async throws -> Result {
        let worker = Task.detached(priority: .userInitiated) {
            try copy(source: source, documents: documents, beforeCopy: beforeCopy)
        }
        return try await withTaskCancellationHandler {
            let result = try await worker.value
            do { try Task.checkCancellation(); return result }
            catch { await discard(result); throw error }
        } onCancel: { worker.cancel() }
    }

    /// Call if metadata validation fails or cancellation wins before selection.
    /// A reused local recording is never removed. Identity prevents a later
    /// replacement at the same path from being mistaken for this import's file.
    public static func discard(_ result: Result) async {
        await Task.detached(priority: .utility) { discardCopy(result) }.value
    }
    private static func discardCopy(_ result: Result) {
        guard let identity = result.identity, (try? FileIdentity(result.url)) == identity else { return }
        try? FileManager.default.removeItem(at: result.url)
    }

    private static func copy(source: URL, documents: URL, beforeCopy: () -> Void) throws -> Result {
        try Task.checkCancellation()
        let files = FileManager.default
        let directory = documents.resolvingSymlinksInPath().standardizedFileURL
        let resolvedSource = source.resolvingSymlinksInPath().standardizedFileURL
        if resolvedSource.deletingLastPathComponent() == directory {
            // Includes /private/var aliases and a picker URL resolving to an
            // already managed local movie. No digest or byte comparison needed.
            return Result(url: resolvedSource, identity: nil)
        }
        let staging = directory.appendingPathComponent(".recording-import-" + UUID().uuidString, isDirectory: true)
        try files.createDirectory(at: staging, withIntermediateDirectories: false)
        defer { try? files.removeItem(at: staging) }
        let temporary = staging.appendingPathComponent("recording.partial")
        guard files.createFile(atPath: temporary.path, contents: nil) else { throw CocoaError(.fileWriteUnknown) }
        let input = try FileHandle(forReadingFrom: source)
        defer { try? input.close() }
        let output = try FileHandle(forWritingTo: temporary)
        defer { try? output.close() }
        beforeCopy()
        while true {
            try Task.checkCancellation()
            guard let data = try input.read(upToCount: 4 * 1024 * 1024), !data.isEmpty else { break }
            try Task.checkCancellation()
            try output.write(contentsOf: data)
        }
        try output.synchronize(); try output.close(); try input.close()
        if let date = try? source.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate {
            try files.setAttributes([.modificationDate: date], ofItemAtPath: temporary.path)
        }
        let identity = try FileIdentity(temporary)
        let originalName = source.lastPathComponent
        let stem = (originalName as NSString).deletingPathExtension
        let ext = source.pathExtension.isEmpty ? "" : "." + source.pathExtension
        var suffix = 1
        while true {
            try Task.checkCancellation()
            let name = suffix == 1 ? originalName : "\(stem) (\(suffix))\(ext)"
            let destination = directory.appendingPathComponent(name)
            if files.fileExists(atPath: destination.path) { suffix += 1; continue }
            do {
                // moveItem refuses to replace. If another writer wins this
                // basename after the probe, retry a suffix with our same stage.
                try files.moveItem(at: temporary, to: destination)
            } catch {
                if files.fileExists(atPath: destination.path), files.fileExists(atPath: temporary.path) {
                    suffix += 1; continue
                }
                throw error
            }
            let result = Result(url: destination, identity: identity)
            do { try Task.checkCancellation(); return result }
            catch { discardCopy(result); throw error }
        }
    }
}
