import Foundation
import XCTest
@testable import VolleySplice

private actor SnapshotWriteGate {
    private var started = 0
    private var observers: [(Int, CheckedContinuation<Void, Never>)] = []
    private var releases: [CheckedContinuation<Void, Never>] = []
    func write(_ project: ProjectDocument, to url: URL) async throws {
        started += 1
        await withCheckedContinuation { continuation in
            releases.append(continuation)
            let ready = observers.filter { $0.0 <= started }
            observers.removeAll { $0.0 <= started }
            for observer in ready { observer.1.resume() }
        }
        // Deliberately finish even if the caller cancelled: the queue must drain
        // a real writer and reject its late result, not assume it stopped.
        try await Task.detached { try project.save(to: url) }.value
    }
    func waitForStarts(_ count: Int) async {
        guard started < count else { return }
        await withCheckedContinuation { observers.append((count, $0)) }
    }
    func releaseNext() { releases.removeFirst().resume() }
}

@MainActor final class QueueSnapshotTests: XCTestCase {
    private func folder() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("queue-snapshot-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
    private func fixture(_ id: String) -> ProjectDocument {
        .init(id: id, sourceName: "match.mp4", durationMs: 2000,
            draft: EditorDraft(sourceRevision: "fixture", cuts: [.init(id: "R1", coreStartMs: 100, coreEndMs: 1000)]),
            feedback: .object(["rawFeatures": .string(String(repeating: "0123456789", count: 10000))]))
    }
    private func job(_ project: ProjectDocument) -> ProcessingJob {
        .init(kind: .videoExport, projectId: project.id, sourceName: project.sourceName,
              sourceFingerprint: "sampled-sha256-v1:fixture")
    }
    func testSnapshotsBecomeDurableBeforeLedgerAndKeepSubmissionOrder() async throws {
        let directory = try folder(); defer { try? FileManager.default.removeItem(at: directory) }
        let gate = SnapshotWriteGate()
        let queue = ProcessingQueue(folder: directory, snapshotWriter: { try await gate.write($0, to: $1) })
        let first = fixture("first"), second = fixture("second"), firstJob = job(first), secondJob = job(second)
        let firstTask = Task { try await queue.enqueue(firstJob, snapshot: first) }
        await gate.waitForStarts(1)
        XCTAssertTrue(queue.jobs.isEmpty)
        XCTAssertTrue(try ProcessingJobStore(folder: directory).load().jobs.isEmpty)
        let reserved = expectation(description: "Both requests reserved without blocking the app actor")
        var notified = false
        queue.onChange = {
            if queue.preparingInputCount == 2 && !notified { notified = true; reserved.fulfill() }
        }
        let secondTask = Task { try await queue.enqueue(secondJob, snapshot: second) }
        await fulfillment(of: [reserved], timeout: 5)
        await gate.releaseNext(); try await firstTask.value
        await gate.waitForStarts(2)
        XCTAssertEqual(queue.jobs.map(\.id), [firstJob.id])
        await gate.releaseNext(); try await secondTask.value
        XCTAssertEqual(queue.jobs.map(\.id), [firstJob.id, secondJob.id])
        XCTAssertEqual(try ProcessingJobStore(folder: directory).load().jobs.map(\.id), [firstJob.id, secondJob.id])
        let restored = try await Task.detached {
            try ProjectDocument.load(from: directory.appendingPathComponent(secondJob.id + ".project.json"))
        }.value
        XCTAssertEqual(restored, second)
        XCTAssertEqual(queue.preparingInputCount, 0)
        queue.onChange = nil
    }
    func testRemovalCancellationAndBackgroundCannotPublishLateSnapshot() async throws {
        for action in ["remove", "cancel", "caller-cancel", "background"] {
            let directory = try folder(); defer { try? FileManager.default.removeItem(at: directory) }
            let gate = SnapshotWriteGate()
            let queue = ProcessingQueue(folder: directory, snapshotWriter: { try await gate.write($0, to: $1) })
            let project = fixture(action), pending = job(project)
            let task = Task { try await queue.enqueue(pending, snapshot: project) }
            await gate.waitForStarts(1)
            switch action {
            case "remove": queue.remove(pending.id)
            case "cancel": queue.cancel(pending.id)
            case "caller-cancel": task.cancel()
            default: queue.sceneChanged(.background)
            }
            XCTAssertTrue(queue.jobs.isEmpty)
            XCTAssertEqual(queue.preparingInputCount, 1, "The writer must drain before its submission slot is released")
            await gate.releaseNext()
            do { try await task.value; XCTFail("A stopped submission must not become queued: \(action)") }
            catch is CancellationError { }
            XCTAssertTrue(queue.jobs.isEmpty)
            XCTAssertTrue(try ProcessingJobStore(folder: directory).load().jobs.isEmpty)
            XCTAssertEqual(queue.preparingInputCount, 0)
            XCTAssertFalse(FileManager.default.fileExists(atPath: directory.appendingPathComponent(pending.id + ".project.json").path))
            XCTAssertFalse(try FileManager.default.contentsOfDirectory(atPath: directory.path).contains { $0.hasSuffix(".stage") })
            // A fresh submission with the same ID must not be affected by the
            // cancelled token or by an old stage left behind on disk.
            queue.sceneChanged(.active)
            var replacement = project; replacement.draft.cuts[0].included = false
            let retry = Task { try await queue.enqueue(pending, snapshot: replacement) }
            await gate.waitForStarts(2); await gate.releaseNext(); try await retry.value
            XCTAssertEqual(queue.jobs.map(\.id), [pending.id])
        }
    }
    func testLedgerWriteFailureDoesNotLeavePublishedInputOrQueueEntry() async throws {
        let directory = try folder(); defer { try? FileManager.default.removeItem(at: directory) }
        let gate = SnapshotWriteGate()
        let queue = ProcessingQueue(folder: directory, snapshotWriter: { try await gate.write($0, to: $1) })
        let project = fixture("failure"), pending = job(project)
        let task = Task { try await queue.enqueue(pending, snapshot: project) }
        await gate.waitForStarts(1)
        let ledger = directory.appendingPathComponent("queue.json")
        try FileManager.default.removeItem(at: ledger)
        try FileManager.default.createDirectory(at: ledger, withIntermediateDirectories: false)
        await gate.releaseNext()
        do { try await task.value; XCTFail("Failed ledger publication must reject enqueue") } catch { }
        XCTAssertTrue(queue.jobs.isEmpty)
        XCTAssertEqual(queue.preparingInputCount, 0)
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.appendingPathComponent(pending.id + ".project.json").path))
        XCTAssertFalse(try FileManager.default.contentsOfDirectory(atPath: directory.path).contains { $0.hasSuffix(".stage") })
    }
    func testActiveWorkerLoadsExactSnapshotAndRetiredAttemptCannotReadIt() async throws {
        let directory = try folder(); defer { try? FileManager.default.removeItem(at: directory) }
        let queue = ProcessingQueue(folder: directory)
        let project = fixture("reader"), pending = job(project)
        let completed = expectation(description: "Immutable snapshot loaded and worker drained")
        var activeAttempt: ProcessingJob?
        queue.execute = { attempt in
            activeAttempt = attempt
            let restored = try await queue.inputSnapshot(attempt)
            XCTAssertEqual(restored, project)
            return .init(detail: "Read complete")
        }
        var notified = false
        queue.onChange = {
            if queue.jobs.first?.state == .completed && !notified { notified = true; completed.fulfill() }
        }
        try await queue.enqueue(pending, snapshot: project)
        await fulfillment(of: [completed], timeout: 5)
        XCTAssertTrue(queue.unfinishedJobs.isEmpty, "Successful work must immediately leave the visible queue")
        XCTAssertEqual(try ProcessingJobStore(folder: directory).load().jobs.first?.state, .completed)
        let attempt = try XCTUnwrap(activeAttempt)
        do { _ = try await queue.inputSnapshot(attempt); XCTFail("A retired worker must not consume snapshot input") }
        catch is CancellationError { }
        queue.execute = nil; queue.onChange = nil
    }

    func testRestoredQueueHidesCompletedWorkWithoutRemovingSavedFiles() throws {
        let directory = try folder(); defer { try? FileManager.default.removeItem(at: directory) }
        let project = fixture("saved-project"), completed = job(project)
        let output = directory.appendingPathComponent("saved-export.mp4")
        let outputBytes = Data("existing export contents".utf8)
        try outputBytes.write(to: output)
        let projectURL = directory.appendingPathComponent("saved-project.volleyproject.json")
        try project.save(to: projectURL)
        var ledger = ProcessingJobLedger(); try ledger.enqueue(completed)
        _ = try ledger.start(id: completed.id, token: "done", nowMs: 2)
        ledger.finish(id: completed.id, token: "done", state: .completed, detail: "Saved",
                      outputNames: [output.lastPathComponent], nowMs: 3)
        let store = try ProcessingJobStore(folder: directory); try store.save(ledger)
        let queue = ProcessingQueue(folder: directory)
        XCTAssertTrue(queue.unfinishedJobs.isEmpty)
        XCTAssertEqual(queue.jobs, ledger.jobs)
        XCTAssertEqual(try store.load().jobs, ledger.jobs)
        XCTAssertEqual(try Data(contentsOf: output), outputBytes)
        XCTAssertEqual(try ProjectDocument.load(from: projectURL), project)
    }

    func testProgressRelayPreservesMixedCallbackOrderAndDrainsFinalStep() async {
        var received: [String] = []
        var tracker = AnalysisProgressTracker(steps: [])
        let relay = AnalysisProgressRelay { event in
            switch event {
            case let .overall(_, detail): received.append(detail)
            case let .step(measurement):
                received.append(measurement.detail); tracker.update(measurement)
            }
        }
        await Task.detached {
            relay.step(.init(.servingSide, fraction: 0, detail: "serve-start", uptime: 10))
            for index in 0..<100 {
                relay.overall(Double(index) / 100, "overall-\(index)")
                relay.step(.init(.servingSide, fraction: Double(index) / 100, detail: "step-\(index)", uptime: 10 + Double(index)))
            }
            relay.step(.init(.servingSide, fraction: 1, detail: "serve-done", uptime: 110))
            relay.step(.init(.sideSwitch, fraction: 0, detail: "switch-start", uptime: 111))
            relay.step(.init(.sideSwitch, fraction: 1, detail: "switch-done", uptime: 113))
        }.value
        await relay.finish()
        let expected = ["serve-start"] + (0..<100).flatMap { ["overall-\($0)", "step-\($0)"] } + ["serve-done", "switch-start", "switch-done"]
        XCTAssertEqual(received, expected)
        XCTAssertEqual(tracker.snapshot(now: 200).map(\.id), [.servingSide, .sideSwitch])
        XCTAssertEqual(tracker.snapshot(now: 200).map(\.status), [.complete, .complete])
        XCTAssertEqual(tracker.snapshot(now: 200).map(\.elapsedSeconds), [100, 2])
        relay.overall(0, "late callback after drain")
        await relay.finish()
        XCTAssertEqual(received, expected)
    }
}
