import Foundation

#if DEBUG
/// Physical-device acceptance fixture: short-gap join, exact-threshold cut,
/// ignored barrier, score change and team switch, all on the chosen source.
/// Exports an overlay-off control followed by the overlay-on result using the
/// same source ranges, transform, source color properties and encoder settings.
@MainActor enum DeviceExportCheck {
    static func run(source: URL, destination: URL, progress: @escaping @Sendable (Double, String) -> Void) async throws -> VideoExportResult {
        let info = try await MediaDecoder.describe(source)
        guard info.duration >= 20 else { throw ProjectError.invalid("Export check needs a recording at least 20 seconds long") }
        var draft = EditorDraft(sourceRevision: "device-export-check-v1", cuts: [
            EditableCut(id: "R001", coreStartMs: 2000, coreEndMs: 4000, keepStartMs: 1000, keepEndMs: 5000),
            EditableCut(id: "R002", coreStartMs: 8000, coreEndMs: 10000, keepStartMs: 7000, keepEndMs: 11000),
            EditableCut(id: "R003", coreStartMs: 15000, coreEndMs: 17000, keepStartMs: 14000, keepEndMs: 18000)])
        draft.beforePaddingMs = 1000; draft.afterPaddingMs = 1000
        draft.renderScoreOverlay = true; draft.renderScoreTimeline = true
        draft.selectedSuppressionPolicy = "none"
        draft.ignoredIntervals = [.init(id: "ignore-fixture", startMs: 6000, endMs: 6500)]
        draft.scoreTracking = try ScoreTracking(serveMarkers: [
            .init(id: "S1", timestampMs: 2000, side: .near, rallyId: "R001"),
            .init(id: "S2", timestampMs: 8000, side: .far, rallyId: "R002"),
            .init(id: "S3", timestampMs: 15000, side: .near, rallyId: "R003")],
            sideSwitchMarkers: [.init(id: "W1", timestampMs: 12000)]).jsonValue()
        let project = ProjectDocument(id: "device-export-check", sourceName: source.lastPathComponent,
                                      durationMs: ProjectSession.ms(info.duration), draft: draft)
        let intervals = EditorMath.finalIntervals(draft)
        guard intervals.map({ TimeRange(startMs: $0.startMs, endMs: $0.endMs) }) == [
            TimeRange(startMs: 1000, endMs: 6000), TimeRange(startMs: 6500, endMs: 11000), TimeRange(startMs: 14000, endMs: 18000)] else {
            throw ProjectError.invalid("Export acceptance fixture interval mismatch")
        }
        let controlDestination = destination.deletingPathExtension().appendingPathExtension("overlay-off.mp4")
        var controlProject = project
        controlProject.draft.renderScoreOverlay = false
        // Keep score tracking/timeline settings and chapters identical: this flag
        // alone selects native composition without score/point pixel blending.
        try controlProject.save(to: controlDestination.deletingPathExtension().appendingPathExtension("volleyproject.json"))
        let control = try await VideoExporter.export(source: source, project: controlProject, destination: controlDestination) { fraction, detail in
            progress(fraction * 0.5, "Overlay off (1/2): \(detail)")
        }
        try Task.checkCancellation()
        try project.save(to: destination.deletingPathExtension().appendingPathExtension("volleyproject.json"))
        let output = try await VideoExporter.export(source: source, project: project, destination: destination) { fraction, detail in
            progress(0.5 + fraction * 0.5, "Overlay on (2/2): \(detail)")
        }
        guard control.durationMs == 13500 else { throw ProjectError.invalid("Overlay-off acceptance fixture duration mismatch") }
        guard output.durationMs == 13500 else { throw ProjectError.invalid("Export acceptance fixture duration mismatch") }
        guard try Data(contentsOf: control.chaptersURL) == Data(contentsOf: output.chaptersURL) else {
            throw ProjectError.invalid("Paired acceptance fixture chapters differ")
        }
        return VideoExportResult(videoURL: output.videoURL, chaptersURL: output.chaptersURL, durationMs: output.durationMs,
            notes: output.notes + ["Overlay-off control: \(control.videoURL.lastPathComponent)",
                                   "Both exports use identical source ranges [1,6), [6.5,11), [14,18) seconds."])
    }
}
#endif
