import SwiftUI
import UIKit

/// Copies only completed generated outputs into the destination chosen in Files.
struct DocumentExportPicker: UIViewControllerRepresentable {
    let urls: [URL]
    let onFinish: () -> Void
    func makeCoordinator() -> Coordinator { Coordinator(onFinish: onFinish) }
    func makeUIViewController(context: Context) -> Presenter {
        let picker = UIDocumentPickerViewController(forExporting: urls, asCopy: true)
        picker.delegate = context.coordinator
        picker.modalPresentationStyle = .fullScreen
        return Presenter(picker: picker, coordinator: context.coordinator)
    }
    func updateUIViewController(_ controller: Presenter, context: Context) {
        context.coordinator.onFinish = onFinish
    }
    static func dismantleUIViewController(_ controller: Presenter, coordinator: Coordinator) {
        // The enclosing SwiftUI sheet can also be dismissed interactively.
        // Remove its UIKit presentation without firing a second completion.
        coordinator.finished = true
        if controller.presentedViewController != nil { controller.dismiss(animated: false) }
    }
    final class Presenter: UIViewController {
        private let picker: UIDocumentPickerViewController
        private let coordinator: Coordinator
        private var presentedPicker = false
        init(picker: UIDocumentPickerViewController, coordinator: Coordinator) {
            self.picker = picker; self.coordinator = coordinator
            super.init(nibName: nil, bundle: nil)
        }
        required init?(coder: NSCoder) { fatalError("Storyboard initialization is unsupported") }
        override func viewDidLoad() {
            super.viewDidLoad()
            view.backgroundColor = .systemBackground
        }
        override func viewDidAppear(_ animated: Bool) {
            super.viewDidAppear(animated)
            guard !presentedPicker, !coordinator.finished else { return }
            presentedPicker = true
            // Give native Cancel/Export actions an explicit UIKit presentation
            // owner; the enclosing SwiftUI sheet closes after that dismissal.
            present(picker, animated: false)
            picker.presentationController?.delegate = coordinator
        }
    }
    @MainActor final class Coordinator: NSObject, UIDocumentPickerDelegate, UIAdaptivePresentationControllerDelegate {
        var onFinish: () -> Void
        var finished = false
        init(onFinish: @escaping () -> Void) { self.onFinish = onFinish }
        private func finish(_ controller: UIDocumentPickerViewController) {
            guard !finished else { return }
            finished = true
            if controller.presentingViewController != nil {
                controller.dismiss(animated: true) { [self] in onFinish() }
            } else { onFinish() }
        }
        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) { finish(controller) }
        func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) { finish(controller) }
        func presentationControllerDidDismiss(_ presentationController: UIPresentationController) {
            guard !finished else { return }
            finished = true; onFinish()
        }
    }
}
