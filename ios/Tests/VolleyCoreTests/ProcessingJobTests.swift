import Foundation
import XCTest
@testable import VolleyCore

final class ProcessingJobTests: XCTestCase {
    private func job(projectId: String = "project") -> ProcessingJob {
        .init(kind: .analysis, projectId: projectId, sourceName: "match.mp4", sourceFingerprint: "sampled-sha256-v1:test",
              analysis: .init(startMs: 0, endMs: 10_000, roi: [0, 0, 1, 1], prepareScores: true), nowMs: 1)
    }
    func testCancellationKeepsSerialSlotUntilWorkerAcknowledgesAndRejectsOldCompletion() throws {
        var ledger = ProcessingJobLedger(); let a = job(), b = job(projectId: "other")
        try ledger.enqueue(a); try ledger.enqueue(b)
        _ = try ledger.start(id: a.id, token: "old", nowMs: 2)
        ledger.stop(id: a.id, reason: .userCancelled, nowMs: 3)
        XCTAssertThrowsError(try ledger.start(id: b.id, token: "b", nowMs: 4))
        XCTAssertTrue(ledger.finish(id: a.id, token: "old", state: .completed, detail: "Late success", nowMs: 5))
        XCTAssertEqual(ledger.jobs[0].state, .cancelled)
        try ledger.resume(id: a.id, nowMs: 6)
        _ = try ledger.start(id: a.id, token: "new", nowMs: 7)
        XCTAssertFalse(ledger.finish(id: a.id, token: "old", state: .failed, detail: "Late failure", nowMs: 8))
        XCTAssertEqual(ledger.jobs[0].state, .running)
        XCTAssertEqual(ledger.jobs[0].attempt, 2)
    }
    func testDiskRoundTripRecoversInterruptedJobsWithoutResurrectingExplicitCancellation() throws {
        var ledger = ProcessingJobLedger(); let a = job(), b = job(projectId: "queued")
        try ledger.enqueue(a); try ledger.enqueue(b)
        _ = try ledger.start(id: a.id, token: "run", nowMs: 2)
        ledger.stop(id: a.id, reason: .userCancelled, nowMs: 3)
        var restored = try JSONDecoder().decode(ProcessingJobLedger.self, from: JSONEncoder().encode(ledger))
        restored.recoverAfterLaunch(nowMs: 4)
        XCTAssertEqual(restored.jobs.map(\.state), [.cancelled, .interrupted])
        XCTAssertEqual(restored.jobs[1].analysis, b.analysis)
        XCTAssertTrue(restored.jobs.allSatisfy { $0.runToken == nil })
        try restored.validate()
    }
    func testQueueRejectsDuplicateWorkAndUnsafeSourcePaths() throws {
        var ledger = ProcessingJobLedger(); try ledger.enqueue(job())
        XCTAssertThrowsError(try ledger.enqueue(job()))
        var unsafe = job(projectId: "other"); unsafe.sourceName = "../other.mp4"
        XCTAssertThrowsError(try ledger.enqueue(unsafe))
    }
    func testRemoveQueuedAndCompletedJobsKeepsOtherJobsAndRejectsLateCallbacks() throws {
        var ledger = ProcessingJobLedger(); let a = job(), b = job(projectId: "other")
        try ledger.enqueue(a); try ledger.enqueue(b)
        ledger.remove(id: a.id, nowMs: 2)
        XCTAssertEqual(ledger.jobs.map(\.id), [b.id])
        XCTAssertFalse(ledger.finish(id: a.id, token: "old", state: .completed, detail: "late", nowMs: 3))
        _ = try ledger.start(id: b.id, token: "b", nowMs: 4)
        XCTAssertTrue(ledger.finish(id: b.id, token: "b", state: .completed, detail: "done", outputNames: ["output.mp4"], nowMs: 5))
        ledger.remove(id: b.id, nowMs: 6)
        XCTAssertTrue(ledger.jobs.isEmpty)
    }
    func testRemovingActiveWorkHoldsSlotUntilDrainAndCannotPublishOutput() throws {
        var ledger = ProcessingJobLedger(); let a = job(), b = job(projectId: "other")
        try ledger.enqueue(a); try ledger.enqueue(b)
        _ = try ledger.start(id: a.id, token: "a", nowMs: 2)
        ledger.remove(id: a.id, nowMs: 3)
        XCTAssertEqual(ledger.jobs[0].state, .cancelling)
        XCTAssertEqual(ledger.jobs[0].removalRequested, true)
        XCTAssertThrowsError(try ledger.resume(id: a.id, nowMs: 4))
        XCTAssertThrowsError(try ledger.start(id: b.id, token: "b", nowMs: 4))
        XCTAssertTrue(ledger.finish(id: a.id, token: "a", state: .completed, detail: "late success", outputNames: ["output.mp4"], nowMs: 5))
        XCTAssertEqual(ledger.jobs.map(\.id), [b.id])
        _ = try ledger.start(id: b.id, token: "b", nowMs: 6)
        try ledger.validate()
    }
    func testRemovalIntentSurvivesProcessExitAndOldLedgersRemainReadable() throws {
        var ledger = ProcessingJobLedger(); let a = job(), b = job(projectId: "other")
        try ledger.enqueue(a); try ledger.enqueue(b)
        _ = try ledger.start(id: a.id, token: "a", nowMs: 2)
        ledger.remove(id: a.id, nowMs: 3)
        var restored = try JSONDecoder().decode(ProcessingJobLedger.self, from: JSONEncoder().encode(ledger))
        restored.recoverAfterLaunch(nowMs: 4)
        XCTAssertEqual(restored.jobs.map(\.id), [b.id])
        XCTAssertEqual(restored.jobs[0].state, .interrupted)
        XCTAssertNil(restored.jobs[0].removalRequested)
        try restored.validate()
    }

    func testQueuePresentationRetainsRetryableWorkAndDropsOnlySuccessfulCompletion() throws {
        var ledger = ProcessingJobLedger(); let a = job(), b = job(projectId: "other")
        try ledger.enqueue(a); try ledger.enqueue(b)
        XCTAssertEqual(ledger.unfinishedJobs.map(\.id), [a.id, b.id])
        _ = try ledger.start(id: a.id, token: "first", nowMs: 2)
        ledger.stop(id: a.id, reason: .userCancelled, nowMs: 3)
        XCTAssertEqual(ledger.unfinishedJobs.first?.state, .cancelling)
        ledger.finish(id: a.id, token: "first", state: .completed, detail: "Stopped late", nowMs: 4)
        XCTAssertEqual(ledger.unfinishedJobs.first?.state, .cancelled)
        try ledger.resume(id: a.id, nowMs: 5)
        _ = try ledger.start(id: a.id, token: "retry", nowMs: 6)
        ledger.finish(id: a.id, token: "retry", state: .failed, detail: "Out of space", nowMs: 7)
        XCTAssertEqual(ledger.unfinishedJobs.first?.state, .failed)
        try ledger.resume(id: a.id, nowMs: 8)
        _ = try ledger.start(id: a.id, token: "success", nowMs: 9)
        ledger.finish(id: a.id, token: "success", state: .completed, detail: "Saved", outputNames: ["result.mp4"], nowMs: 10)
        XCTAssertEqual(ledger.unfinishedJobs.map(\.id), [b.id])
        var restored = try JSONDecoder().decode(ProcessingJobLedger.self, from: JSONEncoder().encode(ledger))
        restored.recoverAfterLaunch(nowMs: 11)
        XCTAssertEqual(restored.unfinishedJobs.map(\.id), [b.id])
        XCTAssertEqual(restored.unfinishedJobs.first?.state, .interrupted)
        XCTAssertEqual(restored.jobs.first?.outputNames, ["result.mp4"], "Hiding completed work must retain its delivery receipt")
        XCTAssertEqual(restored.jobs.first?.state, .completed)
        try restored.validate()
    }

    func testLaunchRecoveryFreezesRunningStepMeasurementsAndRetryClearsThem() throws {
        var ledger = ProcessingJobLedger(); let pending = job()
        try ledger.enqueue(pending); _ = try ledger.start(id: pending.id, token: "interrupted", nowMs: 2)
        var tracker = AnalysisProgressTracker(steps: [.video, .audio, .rally])
        tracker.update(.init(.video, fraction: 0, detail: "Start", uptime: 0))
        tracker.update(.init(.video, fraction: 1, detail: "Ready", uptime: 10))
        tracker.update(.init(.audio, fraction: 0, detail: "Start", uptime: 10))
        tracker.update(.init(.audio, fraction: 0.5, detail: "Audio", uptime: 12))
        ledger.jobs[0].analysisSteps = tracker.snapshot(now: 12)
        var restored = try JSONDecoder().decode(ProcessingJobLedger.self, from: JSONEncoder().encode(ledger))
        restored.recoverAfterLaunch(nowMs: 50_000)
        let rows = try XCTUnwrap(restored.jobs[0].analysisSteps)
        XCTAssertEqual(rows.map(\.status), [.complete, .stopped, .pending])
        XCTAssertEqual(rows[1].elapsedSeconds, 2, "Relaunch must not count downtime as work")
        XCTAssertNil(rows[1].etaSeconds)
        XCTAssertEqual(rows[1].metrics, "Stopped after 2s")
        try restored.resume(id: pending.id, nowMs: 50_001)
        _ = try restored.start(id: pending.id, token: "new", nowMs: 50_002)
        XCTAssertNil(restored.jobs[0].analysisSteps, "A new attempt starts with fresh clocks and rates")
    }

}
