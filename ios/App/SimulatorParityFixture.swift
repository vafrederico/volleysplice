import AVFoundation
import Foundation

/// Synthetic editor interaction fixture, NOT model predictions or inference parity
/// evidence. Media is the Android overlay test's 1-second MP4 repeated to 20s by
/// stream copy. Only the explicit UI-test launch argument enables this in Simulator.
@MainActor enum SimulatorParityFixture {
    private static var opened = false
    static func openIfRequested(_ model: WorkspaceModel) async {
        #if DEBUG && targetEnvironment(simulator)
        guard !opened, ProcessInfo.processInfo.arguments.contains("--parity-editor") else { return }
        opened = true
        do {
            guard let bundledSource = Bundle.main.url(forResource: "ios-editor-fixture", withExtension: "mp4") else {
                throw ProjectError.invalid("Synthetic UI fixture media ios-editor-fixture.mp4 is missing")
            }
            let deferred = ProcessInfo.processInfo.arguments.contains("--parity-deferred-score")
            var source = bundledSource
            if deferred {
                // Reserved synthetic file only; the real queue requires a writable
                // Documents source. Never modify bundled media or user recordings.
                model.queue.execute = nil
                source = model.documents.appendingPathComponent("ios-parity-deferred-score.mp4")
                if !FileManager.default.fileExists(atPath: source.path) {
                    try FileManager.default.copyItem(at: bundledSource, to: source)
                }
                guard try ProjectArchive.sampledFingerprint(url: source) == ProjectArchive.sampledFingerprint(url: bundledSource) else {
                    throw ProjectError.invalid("Reserved deferred-score fixture source has unexpected contents")
                }
            }
            let media = try await MediaDecoder.describe(source)
            guard abs(media.duration - 20) < 0.01 else { throw ProjectError.invalid("Synthetic UI fixture must be 20 seconds") }
            var fixture = try document()
            if deferred {
                fixture.id = "00000000-0000-4000-8000-000000000024"
                fixture.sourceName = source.lastPathComponent
                fixture.draft.scoreTracking = try ScoreTracking(enabled: false).jsonValue()
                fixture.feedback?["initialInference"]?["servingSide"] = nil
                fixture.feedback?["initialInference"]?["sideSwitch"] = nil
                fixture.feedback?["source"]?["generateSideSwitchMarkers"] = .bool(false)
                for job in model.queue.preparingJobs + model.queue.jobs where job.projectId == fixture.id {
                    model.queue.remove(job.id)
                }
            }
            let prepared = try await ProjectPersistence.prepare(fixture, to: model.projectURL(fixture))
            let project = try ProjectPersistence.publish(prepared)
            model.player.pause()
            model.project = project; model.source = source; model.media = media
            model.start = 0; model.end = 20; model.prepareScore = false
            model.player.replaceCurrentItem(with: AVPlayerItem(url: source))
            model.status = "SYNTHETIC UI FIXTURE - no inference was run"
            model.showEditor = true
            if ProcessInfo.processInfo.arguments.contains("--parity-queue") {
                // A paused synthetic entry exercises the actual Remove button without video work.
                model.queue.execute = nil
                let id = "00000000-0000-4000-8000-000000000023"
                model.queue.remove(id)
                try await model.queue.enqueue(ProcessingJob(id: id, kind: .videoExport, projectId: project.id,
                    sourceName: project.sourceName, sourceFingerprint: "sampled-sha256-v1:synthetic-queue"), snapshot: project)
            }
        } catch { model.error = "Synthetic editor fixture failed: \(error.localizedDescription)" }
        #endif
    }

    #if DEBUG && targetEnvironment(simulator)
    private static func document() throws -> ProjectDocument {
        let cuts: [EditableCut] = [
            .init(id: "R001", coreStartMs: 1_000, coreEndMs: 2_500, keepStartMs: 500, keepEndMs: 3_000),
            .init(id: "R002", coreStartMs: 4_000, coreEndMs: 5_500, keepStartMs: 3_500, keepEndMs: 6_000, confidence: 0.35),
            .init(id: "R003", coreStartMs: 7_500, coreEndMs: 9_000, keepStartMs: 7_000, keepEndMs: 9_500),
            .init(id: "R004", coreStartMs: 11_000, coreEndMs: 12_500, keepStartMs: 10_500, keepEndMs: 13_000),
            .init(id: "R005", coreStartMs: 15_000, coreEndMs: 17_000, keepStartMs: 14_500, keepEndMs: 17_500),
            .init(id: "M001", coreStartMs: 18_500, coreEndMs: 19_500, origin: .manual)
        ]
        var draft = EditorDraft(sourceRevision: "synthetic-ui-fixture-v1", cuts: cuts)
        draft.beforePaddingMs = 500; draft.afterPaddingMs = 500; draft.joinGapMs = 1_000
        draft.ignoredIntervals = [.init(id: "I001", startMs: 6_000, endMs: 6_500)]
        let times: [Int64] = [800, 2_200, 3_700, 5_200, 7_200, 8_800, 10_700, 12_200, 15_200, 16_400, 18_800]
        let markers = times.enumerated().map { index, timestamp in
            ServeMarker(id: String(format: "S%02d", index + 1), timestampMs: timestamp,
                side: index == 8 ? .review : index.isMultiple(of: 2) ? .near : .far,
                origin: .manual)
        }
        draft.scoreTracking = try ScoreTracking(enabled: true, team1Name: "Synthetic A", team2Name: "Synthetic B",
            serveMarkers: markers, sideSwitchMarkers: [.init(id: "W01", timestampMs: 13_800, origin: .manual)]).jsonValue()
        draft.renderScoreOverlay = true; draft.renderScoreTimeline = true
        func suggestion(_ id: String, _ start: Double, _ end: Double) -> JSONValue {
            .object(["id": .string(id), "logicalId": .string(id), "start": .number(start), "end": .number(end),
                     "eligiblePolicyIds": .array([.string("aggressive")])])
        }
        let feedback: JSONValue = .object([
            "syntheticUIFixture": .bool(true), "fixturePurpose": .string("Editor interactions only; no inference or feature payload"),
            "source": .object(["media": .object(["duration": .number(20)]),
                               "gameWindow": .object(["start": .number(0), "end": .number(20)])]),
            "initialInference": .object([
                "ranges": .array(cuts.filter { $0.origin == .inferred }.map {
                    .object(["start": .number(Double($0.coreStartMs) / 1_000), "end": .number(Double($0.coreEndMs) / 1_000), "confidence": .number(Double($0.confidence))]) }),
                "suppression": .object(["suggestions": .array([suggestion("C01", 8, 8.5), suggestion("C02", 11.5, 12), suggestion("C-ignored", 6, 6.5)])]),
                "servingSide": .object(["synthetic": .bool(true)]), "sideSwitch": .object(["synthetic": .bool(true)])
            ])
        ])
        let project = ProjectDocument(id: "00000000-0000-4000-8000-000000000011", sourceName: "ios-editor-fixture.mp4",
            durationMs: 20_000, gameWindow: .init(startMs: 0, endMs: 20_000), draft: draft, feedback: feedback)
        try project.validate()
        return project
    }
    #endif
}
