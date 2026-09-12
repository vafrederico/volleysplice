import SwiftUI
import Photos
import UniformTypeIdentifiers

struct ExportDestinationPresentation: ViewModifier {
    @ObservedObject var model: WorkspaceModel
    func body(content: Content) -> some View {
        content.confirmationDialog("Save video to", isPresented: $model.showExportDestination, titleVisibility: .visible) {
            Button("Camera Roll") { model.exportToPhotos() }
            Button("Files") { model.showExportFolder = true }
            Button("Cancel", role: .cancel) {}
        }
    }
}

@MainActor extension WorkspaceModel {
    func exportToPhotos() {
        Task {
            let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
            guard status == .authorized || status == .limited else {
                error = "Allow VolleySplice to add videos in Settings, or choose Files."; return
            }
            enqueueVideo(destination: .init(kind: .photos))
        }
    }
    func exportToFolder(_ folder: URL) {
        let access = folder.startAccessingSecurityScopedResource()
        defer { if access { folder.stopAccessingSecurityScopedResource() } }
        do {
            let bookmark = try folder.bookmarkData(options: .minimalBookmark, includingResourceValuesForKeys: nil, relativeTo: nil)
            enqueueVideo(destination: .init(kind: .files, folderBookmark: bookmark))
        } catch { self.error = error.localizedDescription }
    }
}

/// A completed, closed movie is moved into Photos. No direct DCIM write is permitted by iOS.
enum PhotoExportDelivery {
    static func move(_ video: URL) async throws -> String {
        guard PHPhotoLibrary.authorizationStatus(for: .addOnly) == .authorized ||
              PHPhotoLibrary.authorizationStatus(for: .addOnly) == .limited else {
            throw ProjectError.invalid("Camera Roll access was removed. Your exported video is retained; allow access in Settings and resume.")
        }
        let identifier = PhotoIdentifier()
        try await PHPhotoLibrary.shared().performChanges {
            let request = PHAssetCreationRequest.forAsset()
            let options = PHAssetResourceCreationOptions()
            options.shouldMoveFile = true
            request.addResource(with: .video, fileURL: video, options: options)
            identifier.set(request.placeholderForCreatedAsset?.localIdentifier ?? "")
        }
        return identifier.get()
    }
    private final class PhotoIdentifier: @unchecked Sendable {
        private let lock = NSLock()
        private var value = ""
        func set(_ value: String) { lock.lock(); self.value = value; lock.unlock() }
        func get() -> String { lock.lock(); defer { lock.unlock() }; return value }
    }
}

/// Keep provider permissions for the encode, but coordinate directory mutations
/// only while doing synchronous file work. AVFoundation can coordinate its own
/// reads and must never be awaited under our destination-directory write lock.
enum CoordinatedVideoExport {
    static func inFolder<T: Sendable>(bookmark: Data,
        operation: @escaping @Sendable (URL) async throws -> T) async throws -> T {
        try await withFolderAccess(bookmark: bookmark, operation: operation)
    }
    static func inFolder<T: Sendable>(bookmark: Data, reading source: SourceVideoAccess,
        operation: @escaping @Sendable (URL, URL) async throws -> T) async throws -> T {
        try await withFolderAccess(bookmark: bookmark) { folder in
            try await CoordinatedSourceRead.perform(source) { recording in
                try await operation(recording, folder)
            }
        }
    }
    private static func withFolderAccess<T: Sendable>(bookmark: Data,
        operation: @escaping @Sendable (URL) async throws -> T) async throws -> T {
        try Task.checkCancellation()
        var stale = false
        let folder = try URL(resolvingBookmarkData: bookmark, options: .withoutUI,
                             relativeTo: nil, bookmarkDataIsStale: &stale)
        guard !stale else { throw ProjectError.invalid("The export folder permission expired. Choose the folder again.") }
        let access = SourceVideoAccess(folder)
        defer { withExtendedLifetime(access) {} }
        try Task.checkCancellation()
        return try await operation(folder)
    }

    /// Only short, synchronous staging/publication operations belong here.
    /// Cleanup is allowed after task cancellation so owned partial files drain.
    static func updateFolder<T: Sendable>(_ folder: URL, reading source: URL? = nil, cleanup: Bool = false,
        operation: @escaping @Sendable (URL) throws -> T) async throws -> T {
        let lifetime = FileCoordinationLifetime<T>()
        return try await withTaskCancellationHandler {
            if !cleanup { try Task.checkCancellation() }
            return try await withCheckedThrowingContinuation { continuation in
                DispatchQueue.global(qos: .userInitiated).async {
                    let access = SourceVideoAccess(folder)
                    defer { withExtendedLifetime(access) {} }
                    let coordinator = NSFileCoordinator()
                    guard lifetime.register(coordinator) else { continuation.resume(throwing: CancellationError()); return }
                    var coordinationError: NSError?
                    var result: Result<T, Error>?
                    let update: (URL) -> Void = { destination in
                        result = Result {
                            try lifetime.checkCancellation()
                            return try operation(destination)
                        }
                    }
                    if let source {
                        coordinator.coordinate(readingItemAt: source, options: .withoutChanges,
                            writingItemAt: folder, options: .forMerging, error: &coordinationError) { _, destination in update(destination) }
                    } else {
                        coordinator.coordinate(writingItemAt: folder, options: .forMerging, error: &coordinationError, byAccessor: update)
                    }
                    // Do not lose the result of a completed mutation if cancellation
                    // races publication: the caller must clean or record that output.
                    if let result { continuation.resume(with: result) }
                    else {
                        do {
                            try lifetime.checkCancellation()
                            if let coordinationError { throw coordinationError }
                            throw ProjectError.invalid("Files did not grant access to the export folder")
                        } catch { continuation.resume(throwing: error) }
                    }
                }
            }
        } onCancel: { if !cleanup { lifetime.cancel() } }
    }
}
