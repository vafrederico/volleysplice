import PhotosUI
import SwiftUI

/// Selects a Photos identifier, then resolves its original recording directly.
/// Keep the sheet visible while loading; dismissal cancels the Photos request.
struct PhotoVideoPicker: UIViewControllerRepresentable {
    let onLoadingState: (Bool) -> Void
    let onProgress: (Double) -> Void
    let onComplete: (Result<PhotoVideoSelection?, Error>) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(self) }
    func makeUIViewController(context: Context) -> PHPickerViewController {
        var configuration = PHPickerConfiguration(photoLibrary: .shared())
        configuration.filter = .videos
        configuration.selectionLimit = 1
        configuration.preferredAssetRepresentationMode = .current
        let picker = PHPickerViewController(configuration: configuration)
        picker.delegate = context.coordinator
        return picker
    }
    func updateUIViewController(_ controller: PHPickerViewController, context: Context) {}
    static func dismantleUIViewController(_ controller: PHPickerViewController, coordinator: Coordinator) {
        coordinator.cancel()
    }

    @MainActor final class Coordinator: NSObject, PHPickerViewControllerDelegate {
        private let parent: PhotoVideoPicker
        private var work: Task<Void, Never>?
        private var started = false
        private var finished = false
        init(_ parent: PhotoVideoPicker) { self.parent = parent }

        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            guard !started, !finished else { return }
            guard let result = results.first else { finish(.success(nil)); return }
            guard let identifier = result.assetIdentifier else {
                finish(.failure(PhotoKitVideoSource.Failure.unavailable)); return
            }
            started = true
            parent.onLoadingState(true)
            work = Task {
                do {
                    try await PhotoKitVideoSource.ensureAuthorization()
                    // A PHPicker selection is not itself a grant to PhotoKit's
                    // limited library. Offer Apple's access picker when needed.
                    if PHPhotoLibrary.authorizationStatus(for: .readWrite) == .limited,
                       PHAsset.fetchAssets(withLocalIdentifiers: [identifier], options: nil).firstObject == nil {
                        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                            PHPhotoLibrary.shared().presentLimitedLibraryPicker(from: picker) { _ in
                                continuation.resume()
                            }
                        }
                    }
                    try Task.checkCancellation()
                    let selection = try await PhotoKitVideoSource.resolve(identifier: identifier) { [weak self] progress in
                        Task { @MainActor in
                            guard let self, !self.finished else { return }
                            self.parent.onProgress(progress)
                        }
                    }
                    try Task.checkCancellation()
                    if !finished { finish(.success(selection)) }
                } catch {
                    if !finished {
                        if error is CancellationError { finish(.success(nil)) }
                        else { finish(.failure(error)) }
                    }
                }
            }
        }
        func cancel() {
            guard !finished else { return }
            work?.cancel()
            finish(.success(nil))
        }
        private func finish(_ result: Result<PhotoVideoSelection?, Error>) {
            guard !finished else { return }
            finished = true
            parent.onLoadingState(false)
            parent.onComplete(result)
        }
    }
}
