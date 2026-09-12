import Foundation
import CryptoKit
@preconcurrency import AVFoundation

/// A read-only security scope stays alive for playback or the complete queued job.
final class SourceVideoAccess: @unchecked Sendable {
    let url: URL
    let photo: PhotoVideoSelection?
    var filename: String { photo?.filename ?? url.lastPathComponent }
    private let scoped: Bool
    init(_ url: URL) { self.url = url; photo = nil; scoped = url.startAccessingSecurityScopedResource() }
    init(photo: PhotoVideoSelection) {
        self.photo = photo; url = photo.asset.url; scoped = false
    }
    func playerItem() -> AVPlayerItem { AVPlayerItem(asset: photo?.asset ?? AVURLAsset(url: url)) }
    deinit { if scoped { url.stopAccessingSecurityScopedResource() } }
}

struct SourceVideoUnavailable: LocalizedError {
    let name: String
    var errorDescription: String? { "Re-link \(name) by choosing the original video in Camera Roll or Files. Its location or permission is no longer available." }
}

/// Local access metadata only. Bookmarks never enter portable project exports.
/// Fingerprints, rather than basenames, distinguish different videos with the same name.
actor SourceVideoLibrary {
    private struct Link: Codable {
        let fingerprint: String
        var bookmark: Data?
        var photoIdentifier: String?
    }
    let folder: URL
    let documents: URL
    private var sessionURLs: [String: URL] = [:]
    private var sessionPhotos: [String: String] = [:]
    private let resolvePhoto: @Sendable (String) async throws -> PhotoVideoSelection
    init(folder: URL, documents: URL,
         resolvePhoto: @escaping @Sendable (String) async throws -> PhotoVideoSelection = { try await PhotoKitVideoSource.resolve(identifier: $0) }) {
        self.folder = folder; self.documents = documents; self.resolvePhoto = resolvePhoto
    }

    func remember(_ access: SourceVideoAccess, fingerprint: String) {
        if let photo = access.photo {
            sessionPhotos[fingerprint] = photo.assetIdentifier
            sessionURLs[fingerprint] = nil
        } else {
            sessionURLs[fingerprint] = access.url
            sessionPhotos[fingerprint] = nil
        }
        // A provider may decline persistent access. The current selection remains
        // usable; a later session will ask the user to re-link, never copy the movie.
        do {
            let link: Link
            if let photo = access.photo {
                // Photos owns this URL. Reacquire through PhotoKit each time;
                // never persist its private path or create a copy/bookmark of it.
                link = Link(fingerprint: fingerprint, photoIdentifier: photo.assetIdentifier)
            } else {
                let bookmark = try access.url.bookmarkData(options: .minimalBookmark, includingResourceValuesForKeys: nil, relativeTo: nil)
                link = Link(fingerprint: fingerprint, bookmark: bookmark)
            }
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try JSONEncoder().encode(link).write(to: linkURL(fingerprint), options: .atomic)
        } catch { }
    }

    func resolve(name: String, fingerprint: String?) async throws -> SourceVideoAccess {
        try Task.checkCancellation()
        var candidates: [URL] = []
        if let fingerprint {
            let link = (try? Data(contentsOf: linkURL(fingerprint))).flatMap { try? JSONDecoder().decode(Link.self, from: $0) }
            if let identifier = sessionPhotos[fingerprint] ?? (link?.fingerprint == fingerprint ? link?.photoIdentifier : nil) {
                let selection = try await resolvePhoto(identifier)
                let access = SourceVideoAccess(photo: selection)
                guard try await Self.inspect(access) == fingerprint else { throw SourceVideoUnavailable(name: name) }
                try Task.checkCancellation()
                remember(access, fingerprint: fingerprint)
                return access
            }
            if let session = sessionURLs[fingerprint] { candidates.append(session) }
            if let link, link.fingerprint == fingerprint, let bookmark = link.bookmark {
                var stale = false
                if let url = try? URL(resolvingBookmarkData: bookmark, options: .withoutUI, relativeTo: nil, bookmarkDataIsStale: &stale) {
                    candidates.append(url)
                }
            }
        }
        // Backward compatibility for existing app-owned imports. No recursive
        // search and no filename-only match when the project has a fingerprint.
        if ProcessingJob.isFilename(name) { candidates.append(documents.appendingPathComponent(name)) }
        for url in candidates {
            try Task.checkCancellation()
            let access = SourceVideoAccess(url)
            do {
                let actual = try await Self.inspect(access)
                try Task.checkCancellation()
                guard fingerprint == nil || actual == fingerprint else { continue }
                remember(access, fingerprint: actual)
                return access
            } catch is CancellationError { throw CancellationError() }
            catch { try Task.checkCancellation(); continue }
        }
        try Task.checkCancellation()
        throw SourceVideoUnavailable(name: name)
    }

    nonisolated static func inspect(_ access: SourceVideoAccess) async throws -> String {
        try await CoordinatedSourceRead.perform(access) { url in
            try Task.checkCancellation()
            return try ProjectArchive.sampledFingerprint(url: url)
        }
    }

    nonisolated static func fingerprint(_ access: SourceVideoAccess) throws -> String {
        var error: NSError?
        var result: Result<String, Error>?
        NSFileCoordinator().coordinate(readingItemAt: access.url, options: .withoutChanges, error: &error) { url in
            result = Result { try ProjectArchive.sampledFingerprint(url: url) }
        }
        if let error { throw error }
        guard let result else { throw SourceVideoUnavailable(name: access.url.lastPathComponent) }
        return try result.get()
    }

    private func linkURL(_ fingerprint: String) -> URL {
        let key = SHA256.hash(data: Data(fingerprint.utf8)).map { String(format: "%02x", $0) }.joined()
        return folder.appendingPathComponent(key + ".json")
    }
}

/// Providers may download an iCloud original as part of coordination. The app
/// never creates its own video copy. Hold read coordination until work drains.
enum CoordinatedSourceRead {
    static func perform<T: Sendable>(_ access: SourceVideoAccess,
        operation: @escaping @Sendable (URL) async throws -> T) async throws -> T {
        let lifetime = FileCoordinationLifetime<T>()
        return try await withTaskCancellationHandler {
            try Task.checkCancellation()
            return try await withCheckedThrowingContinuation { continuation in
                DispatchQueue.global(qos: .userInitiated).async {
                    let coordinator = NSFileCoordinator()
                    guard lifetime.register(coordinator) else { continuation.resume(throwing: CancellationError()); return }
                    var coordinationError: NSError?
                    coordinator.coordinate(readingItemAt: access.url, options: .withoutChanges, error: &coordinationError) { url in
                        let done = DispatchSemaphore(value: 0)
                        let child = Task {
                            defer { done.signal() }
                            do { try lifetime.checkCancellation(); try Task.checkCancellation(); lifetime.store(.success(try await operation(url))) }
                            catch { lifetime.store(.failure(error)) }
                        }
                        lifetime.start(child); done.wait()
                    }
                    withExtendedLifetime(access) {
                        do {
                            try lifetime.checkCancellation()
                            if let coordinationError { throw coordinationError }
                            guard let result = lifetime.result() else { throw SourceVideoUnavailable(name: access.url.lastPathComponent) }
                            continuation.resume(with: result)
                        } catch { continuation.resume(throwing: error) }
                    }
                }
            }
        } onCancel: { lifetime.cancel() }
    }
}

/// Cancellation can arrive before coordination starts, while a provider downloads,
/// or during the accessor's work. Cancel both waits, but return only after draining.
final class FileCoordinationLifetime<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    private var coordinator: NSFileCoordinator?
    private var child: Task<Void, Never>?
    private var output: Result<T, Error>?
    func register(_ coordinator: NSFileCoordinator) -> Bool {
        lock.lock(); self.coordinator = coordinator; let stop = cancelled; lock.unlock()
        if stop { coordinator.cancel() }
        return !stop
    }
    func start(_ child: Task<Void, Never>) { lock.lock(); self.child = child; let stop = cancelled; lock.unlock(); if stop { child.cancel() } }
    func cancel() {
        lock.lock(); cancelled = true; let task = child; let coordinator = self.coordinator; lock.unlock()
        task?.cancel(); coordinator?.cancel()
    }
    func checkCancellation() throws {
        lock.lock(); let stop = cancelled; lock.unlock()
        if stop { throw CancellationError() }
    }
    func store(_ value: Result<T, Error>) { lock.lock(); output = value; lock.unlock() }
    func result() -> Result<T, Error>? { lock.lock(); defer { lock.unlock() }; return output }
}
