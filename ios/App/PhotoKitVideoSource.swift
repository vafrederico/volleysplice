import AVFoundation
import Photos

/// Retain this lease while reading its URL. The identifier belongs only in the
/// device's source library; it is not a portable project/export identifier.
struct PhotoVideoSelection: @unchecked Sendable {
    let assetIdentifier: String
    let asset: AVURLAsset
    let filename: String
}

enum PhotoKitVideoSource {
    enum Failure: LocalizedError {
        case permission, unavailable, notVideo, unsupportedRepresentation
        var errorDescription: String? {
            switch self {
            case .permission:
                return "Allow Photos access for VolleySplice in Settings, then select the game video again. Limited access to selected videos is enough."
            case .unavailable:
                return "This video is not available to VolleySplice. Add it to the app's selected Photos in Settings, or choose another game video."
            case .notVideo:
                return "Choose a video from Photos."
            case .unsupportedRepresentation:
                return "Photos could not provide this original recording for direct access. Choose another video, or export it from Photos and select it using Files."
            }
        }
    }

    static func ensureAuthorization() async throws {
        try Task.checkCancellation()
        var status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        if status == .notDetermined {
            status = await withCheckedContinuation { continuation in
                PHPhotoLibrary.requestAuthorization(for: .readWrite) { continuation.resume(returning: $0) }
            }
        }
        try Task.checkCancellation()
        guard status == .authorized || status == .limited else { throw Failure.permission }
    }

    static func resolve(identifier: String,
                        progress: @escaping @Sendable (Double) -> Void = { _ in }) async throws -> PhotoVideoSelection {
        try await ensureAuthorization()
        guard let photo = PHAsset.fetchAssets(withLocalIdentifiers: [identifier], options: nil).firstObject else {
            throw Failure.unavailable
        }
        guard photo.mediaType == .video else { throw Failure.notVideo }
        let resources = PHAssetResource.assetResources(for: photo)
        let name = (resources.first(where: { $0.type == .video }) ?? resources.first)?.originalFilename ?? "Game video.mov"
        let options = PHVideoRequestOptions()
        options.version = .original
        options.deliveryMode = .highQualityFormat
        options.isNetworkAccessAllowed = true
        options.progressHandler = { value, _, _, _ in progress(min(1, max(0, value))) }
        let manager = PHImageManager.default()
        let selection = try await request(identifier: identifier, filename: name, start: { completion in
            manager.requestAVAsset(forVideo: photo, options: options) { asset, _, info in completion(asset, info) }
        }, cancel: { manager.cancelImageRequest($0) })
        progress(1)
        return selection
    }

    /// Wrap the actual PhotoKit request, allowing deterministic tests of callbacks
    /// arriving before request IDs and of cancellation while an ID is in flight.
    static func request(identifier: String, filename: String,
                        start: (@escaping (AVAsset?, [AnyHashable: Any]?) -> Void) -> PHImageRequestID,
                        cancel: @escaping @Sendable (PHImageRequestID) -> Void) async throws -> PhotoVideoSelection {
        let state = PhotoVideoRequestState(cancelRequest: cancel)
        let result = try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                guard state.begin(continuation) else { return }
                let requestID = start { asset, info in
                    if (info?[PHImageCancelledKey] as? NSNumber)?.boolValue == true {
                        state.finish(.failure(CancellationError()))
                    } else if let error = info?[PHImageErrorKey] as? Error {
                        state.finish(.failure(error))
                    } else if let asset = asset as? AVURLAsset, asset.url.isFileURL {
                        state.finish(.success(.init(assetIdentifier: identifier, asset: asset, filename: filename)))
                    } else {
                        state.finish(.failure(Failure.unsupportedRepresentation))
                    }
                }
                state.setRequestID(requestID)
            }
        } onCancel: { state.cancel() }
        try Task.checkCancellation()
        return result
    }
}

private final class PhotoVideoRequestState: @unchecked Sendable {
    private let lock = NSLock()
    private let cancelRequest: @Sendable (PHImageRequestID) -> Void
    private var continuation: CheckedContinuation<PhotoVideoSelection, Error>?
    private var requestID: PHImageRequestID?
    private var finished = false
    private var cancelled = false
    init(cancelRequest: @escaping @Sendable (PHImageRequestID) -> Void) { self.cancelRequest = cancelRequest }

    func begin(_ continuation: CheckedContinuation<PhotoVideoSelection, Error>) -> Bool {
        lock.lock()
        if cancelled {
            lock.unlock(); continuation.resume(throwing: CancellationError()); return false
        }
        self.continuation = continuation
        lock.unlock(); return true
    }
    func setRequestID(_ id: PHImageRequestID) {
        lock.lock()
        let shouldCancel = cancelled
        if !finished { requestID = id }
        lock.unlock()
        if shouldCancel { cancelRequest(id) }
    }
    func finish(_ result: Result<PhotoVideoSelection, Error>) {
        lock.lock()
        guard !finished else { lock.unlock(); return }
        finished = true
        let callback = continuation
        continuation = nil; requestID = nil
        lock.unlock()
        callback?.resume(with: result)
    }
    func cancel() {
        lock.lock()
        guard !finished else { lock.unlock(); return }
        cancelled = true; finished = true
        let callback = continuation, id = requestID
        continuation = nil; requestID = nil
        lock.unlock()
        if let id { cancelRequest(id) }
        callback?.resume(throwing: CancellationError())
    }
}
