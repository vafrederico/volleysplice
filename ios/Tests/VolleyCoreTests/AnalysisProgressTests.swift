import Foundation
import XCTest
@testable import VolleyCore

final class AnalysisProgressTests: XCTestCase {
    func testVideoRatesAndAndroidETAThresholds() throws {
        var tracker = AnalysisProgressTracker(steps: [.video, .audio])
        tracker.update(.init(.video, fraction: 0, detail: "Start", completedFrames: 0, processedVideoSeconds: 0, uptime: 10))
        var row = try XCTUnwrap(tracker.snapshot(now: 10).first)
        XCTAssertTrue(row.metrics.contains("Measuring…")); XCTAssertTrue(row.metrics.contains("estimating ETA"))
        tracker.update(.init(.video, fraction: 0.02, detail: "Frames", completedFrames: 8, processedVideoSeconds: 2, uptime: 11))
        row = try XCTUnwrap(tracker.snapshot(now: 11).first)
        XCTAssertEqual(row.framesPerSecond, 8); XCTAssertEqual(row.realtimeRatio, 2)
        XCTAssertNil(row.etaSeconds, "Wait for three percent before extrapolating")
        tracker.update(.init(.video, fraction: 0.25, detail: "Frames", completedFrames: 100, processedVideoSeconds: 25, uptime: 15))
        row = try XCTUnwrap(tracker.snapshot(now: 15).first)
        XCTAssertEqual(row.framesPerSecond, 20); XCTAssertEqual(row.realtimeRatio, 5)
        XCTAssertEqual(row.etaSeconds, 15)
        XCTAssertTrue(row.metrics.contains("20.0 frames/s")); XCTAssertTrue(row.metrics.contains("ETA 15s"))
    }
    func testEachStageHasIndependentClockAndContinuesEstimatingDuringAStall() throws {
        var tracker = AnalysisProgressTracker(steps: [.video, .audio, .rally])
        tracker.update(.init(.video, fraction: 0, detail: "Start", uptime: 0))
        tracker.update(.init(.video, fraction: 1, detail: "Done", uptime: 100))
        tracker.update(.init(.audio, fraction: 0, detail: "Start audio", uptime: 100))
        XCTAssertTrue(tracker.snapshot(now: 100)[1].metrics.contains("estimating ETA"))
        tracker.update(.init(.audio, fraction: 0.5, detail: "Audio", uptime: 102))
        let rows = tracker.snapshot(now: 104)
        XCTAssertEqual(rows[0].elapsedSeconds, 100)
        XCTAssertEqual(rows[1].elapsedSeconds, 4); XCTAssertEqual(rows[1].percentPerSecond, 12.5)
        XCTAssertEqual(rows[1].etaSeconds, 4)
        tracker.update(.init(.rally, fraction: 0, detail: "Models", uptime: 105))
        XCTAssertEqual(tracker.snapshot(now: 106)[1].elapsedSeconds, 5)
        XCTAssertTrue(tracker.snapshot(now: 106)[2].metrics.contains("estimating ETA"))
    }
    func testResumeExcludesCachedWorkAndRejectsInvalidOrLateEvents() throws {
        var tracker = AnalysisProgressTracker(steps: [.video, .audio])
        tracker.update(.init(.video, fraction: 0.5, detail: "Cached half", completedFrames: 200,
                             processedVideoSeconds: 50, uptime: 10))
        tracker.update(.init(.video, fraction: 0.6, detail: "Fresh frames", completedFrames: 240,
                             processedVideoSeconds: 60, uptime: 15))
        let row = tracker.snapshot(now: 15)[0]
        XCTAssertEqual(try XCTUnwrap(row.framesPerSecond), 8, accuracy: 0.001)
        XCTAssertEqual(try XCTUnwrap(row.etaSeconds), 20, accuracy: 0.001)
        tracker.update(.init(.video, fraction: .nan, detail: "Invalid", uptime: 16))
        tracker.update(.init(.video, fraction: 0.8, detail: "Late callback", uptime: 12))
        XCTAssertEqual(tracker.snapshot(now: 15)[0], row)
        tracker.stop(now: 16)
        XCTAssertEqual(tracker.snapshot(now: 100)[0].elapsedSeconds, 6)
        XCTAssertEqual(tracker.snapshot(now: 100)[0].status, .stopped)
        XCTAssertEqual(tracker.snapshot(now: 100)[1].status, .pending)
        let freshAttempt = AnalysisProgressTracker(steps: [.video, .audio])
        XCTAssertTrue(freshAttempt.snapshot(now: 100).allSatisfy { $0.status == .pending && $0.etaSeconds == nil })
    }
    func testAllStepLabelsAndEarlyMetricsMatchTheAndroidProgressPanel() {
        XCTAssertEqual(AnalysisStep.allCases.map(\.label), ["Scanning video", "Listening for play", "Finding rallies", "Finding serve markers", "Finding team switches"])
        for step in AnalysisStep.allCases {
            var tracker = AnalysisProgressTracker(steps: [step])
            tracker.update(.init(step, fraction: 0, detail: "Starting", uptime: 0))
            let row = tracker.snapshot(now: 0.1)[0]
            XCTAssertTrue(row.metrics.contains("Measuring…")); XCTAssertTrue(row.metrics.contains("estimating ETA"))
        }
    }
}
