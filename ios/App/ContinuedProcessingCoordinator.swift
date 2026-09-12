@preconcurrency import BackgroundTasks
import Foundation

/// A lease covers a finite, explicitly requested batch of CPU analysis jobs.
/// Exports stay foreground-only until their GPU/compositor requirements are validated.
@MainActor final class ContinuedProcessingCoordinator {
    private var pendingIdentifier: String?
    private var heldTask: AnyObject?
    var isPending: Bool { pendingIdentifier != nil && heldTask == nil }
    var isRunning: Bool { heldTask != nil }
    func request(jobCount: Int, granted: @escaping @MainActor () -> Void,
                 expired: @escaping @MainActor () -> Void) throws -> Bool {
        guard #available(iOS 26.0, *) else { return false }
        let identifier = "com.vafrederico.VolleySplice.processing." + UUID().uuidString
        pendingIdentifier = identifier
        let scheduler = BGTaskScheduler.shared
        let registered = scheduler.register(forTaskWithIdentifier: identifier, using: .main) { [weak self] task in
            MainActor.assumeIsolated {
                guard let self, self.pendingIdentifier == identifier, let continued = task as? BGContinuedProcessingTask else {
                    task.setTaskCompleted(success: false); return
                }
                self.heldTask = continued
                continued.progress.totalUnitCount = Int64(max(1, jobCount)) * 1_000
                continued.progress.completedUnitCount = 0
                continued.expirationHandler = { Task { @MainActor in expired() } }
                granted()
            }
        }
        guard registered else { pendingIdentifier = nil; throw ProjectError.invalid("Background analysis registration unavailable") }
        let request = BGContinuedProcessingTaskRequest(identifier: identifier, title: "Analyze recordings",
            subtitle: jobCount == 1 ? "Preparing recording" : "Preparing \(jobCount) queued recordings")
        request.strategy = .queue
        request.requiredResources = []
        do { try scheduler.submit(request); return true }
        catch { pendingIdentifier = nil; throw error }
    }
    func progress(completedJobs: Int, fraction: Double, title: String = "Analyze recordings", detail: String) {
        guard #available(iOS 26.0, *), let task = heldTask as? BGContinuedProcessingTask else { return }
        task.progress.completedUnitCount = min(task.progress.totalUnitCount,
            Int64(completedJobs) * 1_000 + Int64(min(1, max(0, fraction)) * 1_000))
        task.updateTitle(title, subtitle: detail)
    }
    func finish(success: Bool) {
        if #available(iOS 26.0, *), let task = heldTask as? BGContinuedProcessingTask {
            heldTask = nil; pendingIdentifier = nil
            task.expirationHandler = nil; task.setTaskCompleted(success: success)
        } else if let identifier = pendingIdentifier {
            pendingIdentifier = nil; BGTaskScheduler.shared.cancel(taskRequestWithIdentifier: identifier)
        }
    }
}
