@preconcurrency import AVFoundation
import Foundation
import UIKit

struct VideoExportResult: Sendable {
    let videoURL: URL
    let chaptersURL: URL
    let durationMs: Int64
    let notes: [String]
}

/// Native on-device source-color H.264/HEVC and AAC export. The Mac lab only compiles this path.
/// Orientation is baked into upright pixels, including portrait source transforms.
@MainActor enum VideoExporter {
    @discardableResult
    static func export(source: URL, project: ProjectDocument, destination: URL,
                       managesBackgroundLease: Bool = true,
                       progress: @escaping @Sendable (Double, String) -> Void) async throws -> VideoExportResult {
        try project.validate()
        let suppression = try project.feedback.map { try ProjectArchive.suppressionRegions($0) } ?? []
        let intervals = EditorMath.finalIntervals(project.draft, suppression: suppression, bounds: project.gameWindow)
        return try await perform(source: source, project: project, intervals: intervals, allIntervals: intervals,
                                 destination: destination, managesBackgroundLease: managesBackgroundLease, progress: progress)
    }

    /// Serial exports preserve the global match score in each clip. A failure leaves
    /// already completed clips intact and removes only the current partial output.
    static func exportRallies(source: URL, project: ProjectDocument, destinationDirectory: URL,
                              progress: @escaping @Sendable (Double, String) -> Void) async throws -> [VideoExportResult] {
        try project.validate()
        let directoryAccess = destinationDirectory.startAccessingSecurityScopedResource()
        defer { if directoryAccess { destinationDirectory.stopAccessingSecurityScopedResource() } }
        let suppression = try project.feedback.map { try ProjectArchive.suppressionRegions($0) } ?? []
        let intervals = EditorMath.finalIntervals(project.draft, suppression: suppression, bounds: project.gameWindow)
        let tracking = try visibleTracking(project, intervals: intervals)
        let groups = EditorMath.editableRallyGroups(cuts: project.draft.cuts, intervals: intervals, serveMarkers: tracking.serveMarkers)
            .filter { group in intervals.contains { !group.cutIds.isDisjoint(with: $0.cutIds) } }
        guard !groups.isEmpty else { throw ProjectError.invalid("There are no included rallies to export") }
        try FileManager.default.createDirectory(at: destinationDirectory, withIntermediateDirectories: true)
        var results: [VideoExportResult] = []
        for (index, group) in groups.enumerated() {
            try Task.checkCancellation()
            let selected = intervals.compactMap { interval -> FinalCutInterval? in
                guard !group.cutIds.isDisjoint(with: interval.cutIds) else { return nil }
                let start = max(interval.startMs, group.keepStartMs), end = min(interval.endMs, group.keepEndMs)
                guard end > start else { return nil }
                return FinalCutInterval(startMs: start, endMs: end, cutIds: interval.cutIds.filter { group.cutIds.contains($0) })
            }
            let filename = String(format: "rally-%03d-%@.mp4", index + 1, UUID().uuidString.prefix(8).description)
            let result = try await perform(source: source, project: project, intervals: selected, allIntervals: intervals,
                destination: destinationDirectory.appendingPathComponent(filename)) { fraction, detail in
                    progress((Double(index) + fraction) / Double(groups.count), "Rally \(index + 1)/\(groups.count): \(detail)")
                }
            results.append(result)
        }
        return results
    }

    static func chaptersURL(for destination: URL) -> URL {
        destination.deletingLastPathComponent().appendingPathComponent(YouTubeChapters.filename(destination.lastPathComponent))
    }

    private static func visibleTracking(_ project: ProjectDocument, intervals: [FinalCutInterval]) throws -> ScoreTracking {
        let tracking = try ScoreTracking.fromJSON(project.draft.scoreTracking, durationMs: project.durationMs)
        let kept = Set(intervals.flatMap(\.cutIds))
        let excluded = Set(project.draft.cuts.filter { $0.origin == .inferred && (!$0.included || !kept.contains($0.id)) }.map(\.id))
        return ScoreReducer.visibleTracking(tracking, ignoredIntervals: project.draft.ignoredIntervals, excludedRallyIds: excluded)
    }

    private static func perform(source: URL, project: ProjectDocument, intervals: [FinalCutInterval], allIntervals: [FinalCutInterval],
                                destination: URL, managesBackgroundLease: Bool = true,
                                progress: @escaping @Sendable (Double, String) -> Void) async throws -> VideoExportResult {
        try Task.checkCancellation()
        let timeline = try ExportTimeline(intervals: intervals)
        guard timeline.durationMs > 0 else { throw ProjectError.invalid("There are no included video ranges to export") }
        let manager = FileManager.default
        let sourcePath = source.standardizedFileURL.resolvingSymlinksInPath()
        let destinationPath = destination.standardizedFileURL.resolvingSymlinksInPath()
        let chapterDestination = chaptersURL(for: destination)
        guard source.isFileURL, destination.isFileURL, destination.pathExtension.lowercased() == "mp4",
              sourcePath != destinationPath, !manager.fileExists(atPath: destination.path),
              !manager.fileExists(atPath: chapterDestination.path) else {
            throw ProjectError.invalid("Choose a new MP4 filename; recordings and existing exports cannot be overwritten")
        }
        let sourceAccess = source.startAccessingSecurityScopedResource()
        let destinationAccess = destination.deletingLastPathComponent().startAccessingSecurityScopedResource()
        let wasIdleDisabled = UIApplication.shared.isIdleTimerDisabled
        UIApplication.shared.isIdleTimerDisabled = true
        defer {
            UIApplication.shared.isIdleTimerDisabled = wasIdleDisabled
            if sourceAccess { source.stopAccessingSecurityScopedResource() }
            if destinationAccess { destination.deletingLastPathComponent().stopAccessingSecurityScopedResource() }
        }
        let parent = destination.deletingLastPathComponent()
        let asset = AVURLAsset(url: source)
        return try await withTaskCancellationHandler {
            try Task.checkCancellation()
            return try await withStaging(in: parent, source: source) { staging in
                let temporaryVideo = staging.appendingPathComponent("video.mp4"), temporaryChapters = staging.appendingPathComponent("chapters.txt")
                progress(0.01, "Preparing exact video and audio cuts")
                guard let sourceVideo = try await asset.loadTracks(withMediaType: .video).first else {
                    throw ProjectError.invalid("The source has no video track")
                }
                let duration = try await asset.load(.duration)
                guard duration.seconds.isFinite, intervals.allSatisfy({ $0.startMs >= 0 && Double($0.endMs) / 1000 <= duration.seconds + 0.002 }) else {
                    throw ProjectError.invalid("Project export ranges exceed the selected recording")
                }
                let naturalSize = try await sourceVideo.load(.naturalSize)
                let preferred = try await sourceVideo.load(.preferredTransform)
                let displayedRect = CGRect(origin: .zero, size: naturalSize).applying(preferred)
                guard [displayedRect.minX, displayedRect.minY, displayedRect.width, displayedRect.height].allSatisfy(\.isFinite),
                      displayedRect.width > 0, displayedRect.height > 0 else { throw ProjectError.invalid("Invalid source video geometry") }
                // H.264 dimensions are even. Padding by at most one pixel avoids stretching.
                let renderSize = CGSize(width: ceil(displayedRect.width / 2) * 2, height: ceil(displayedRect.height / 2) * 2)
                let upright = preferred.concatenating(CGAffineTransform(translationX: -displayedRect.minX, y: -displayedRect.minY))
                let nominalRate = try await sourceVideo.load(.nominalFrameRate)
                let sourceFrameDuration = try await sourceVideo.load(.minFrameDuration)
                let frameDuration = sourceFrameDuration.isNumeric && sourceFrameDuration.seconds > 0
                    ? sourceFrameDuration : CMTime(seconds: 1 / Double(nominalRate > 0 ? nominalRate : 30), preferredTimescale: 60_000)
                let composition = AVMutableComposition()
                guard let video = composition.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid) else {
                    throw ProjectError.invalid("Cannot create the export video track")
                }
                let sourceAudio = try await asset.loadTracks(withMediaType: .audio).first
                let audio: AVMutableCompositionTrack?
                let audioRange: CMTimeRange?
                if let sourceAudio {
                    guard let track = composition.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid) else {
                        throw ProjectError.invalid("Cannot create the export audio track")
                    }
                    audio = track; audioRange = try await sourceAudio.load(.timeRange)
                } else { audio = nil; audioRange = nil }
                for span in timeline.spans {
                    try Task.checkCancellation()
                    let range = CMTimeRange(start: time(span.source.startMs), end: time(span.source.endMs))
                    try video.insertTimeRange(range, of: sourceVideo, at: time(span.outputStartMs))
                    if let sourceAudio, let audio, let audioRange {
                        let overlap = CMTimeRangeGetIntersection(range, otherRange: audioRange)
                        if overlap.isValid && !overlap.isEmpty {
                            let at = CMTimeAdd(time(span.outputStartMs), CMTimeSubtract(overlap.start, range.start))
                            try audio.insertTimeRange(overlap, of: sourceAudio, at: at)
                        }
                    }
                }
                let videoComposition = AVMutableVideoComposition()
                videoComposition.renderSize = renderSize
                videoComposition.frameDuration = frameDuration
                let sourceFormat = try await sourceVideo.load(.formatDescriptions).first
                if let format = sourceFormat {
                    // Preserve the source color space through the compositor and overlay pass.
                    videoComposition.colorPrimaries = CMFormatDescriptionGetExtension(format, extensionKey: kCMFormatDescriptionExtension_ColorPrimaries) as? String
                    videoComposition.colorYCbCrMatrix = CMFormatDescriptionGetExtension(format, extensionKey: kCMFormatDescriptionExtension_YCbCrMatrix) as? String
                    videoComposition.colorTransferFunction = CMFormatDescriptionGetExtension(format, extensionKey: kCMFormatDescriptionExtension_TransferFunction) as? String
                }
                let tracking = try visibleTracking(project, intervals: allIntervals)
                let sourceIsHDR = [AVVideoTransferFunction_ITU_R_2100_HLG, AVVideoTransferFunction_SMPTE_ST_2084_PQ]
                    .contains(videoComposition.colorTransferFunction ?? "")
                var exportColor: VideoExportColor?
                if sourceIsHDR || (project.draft.renderScoreOverlay && tracking.enabled) ||
                    (videoComposition.colorPrimaries != nil && videoComposition.colorTransferFunction != nil && videoComposition.colorYCbCrMatrix != nil) {
                    exportColor = try VideoExportColor(primaries: videoComposition.colorPrimaries,
                        transfer: videoComposition.colorTransferFunction, matrix: videoComposition.colorYCbCrMatrix,
                        fullRange: sourceFormat.flatMap { CMFormatDescriptionGetExtension($0, extensionKey: kCMFormatDescriptionExtension_FullRangeVideo) as? Bool } ?? false)
                }
                let instruction = AVMutableVideoCompositionInstruction()
                instruction.timeRange = CMTimeRange(start: .zero, duration: time(timeline.durationMs))
                instruction.backgroundColor = UIColor.black.cgColor
                let layerInstruction = AVMutableVideoCompositionLayerInstruction(assetTrack: video)
                layerInstruction.setTransform(upright, at: .zero)
                instruction.layerInstructions = [layerInstruction]
                videoComposition.instructions = [instruction]
                let retainedIds = Set(allIntervals.flatMap(\.cutIds))
                let ranges = project.draft.cuts.filter { retainedIds.contains($0.id) }.map {
                    ScoreRallyRange(coreStartMs: $0.coreStartMs, coreEndMs: $0.coreEndMs, keepStartMs: $0.keepStartMs, keepEndMs: $0.keepEndMs)
                }
                if project.draft.renderScoreOverlay && tracking.enabled {
                    progress(0.04, "Preparing score and point timeline overlays")
                    guard let exportColor, YUVScoreCompositor.accepts(transform: upright) else {
                        throw ProjectError.invalid("Score overlays require a right-angle source orientation")
                    }
                    // Freeze CPU-rendered strips and evaluate their fade in source time
                    // for every output frame. No CA animation track or full-frame RGB
                    // conversion participates in video export.
                    #if DEBUG
                    try YUVScoreCompositor.checkKernels()
                    try ScoreOverlayRenderer.checkPixels()
                    #endif
                    videoComposition.customVideoCompositorClass = exportColor.isHDR ? HDRYUVScoreCompositor.self : YUVScoreCompositor.self
                    let prepared = PreparedScoreOverlay(tracking: tracking, rallyRanges: ranges)
                    struct RasterKey: Hashable { let pixels: Data; let width: Int; let height: Int; let scoreWidth: Int }
                    var convertedRasters: [RasterKey: ScoreBlendRaster] = [:]
                    var overlayInstructions: [AVVideoCompositionInstructionProtocol] = []
                    for span in timeline.spans {
                        let boundaries = [span.source.startMs] + prepared.changeTimestamps.filter {
                            $0 > span.source.startMs && $0 < span.source.endMs
                        } + [span.source.endMs]
                        for (start, end) in zip(boundaries, boundaries.dropFirst()) {
                            try Task.checkCancellation()
                            let snapshot = prepared.snapshot(at: start)
                            let raster = try ScoreOverlayRenderer.raster(snapshot: snapshot, videoWidth: Int(renderSize.width),
                                videoHeight: Int(renderSize.height), renderTimeline: project.draft.renderScoreTimeline)
                            let key = RasterKey(pixels: raster.pixels, width: raster.width, height: raster.height, scoreWidth: raster.scoreWidth)
                            let converted = convertedRasters[key] ?? ScoreBlendRaster(bgra: raster.pixels, width: raster.width,
                                height: raster.height, scoreWidth: raster.scoreWidth, color: exportColor.space)
                            convertedRasters[key] = converted
                            let outputStart = span.outputStartMs + start - span.source.startMs
                            overlayInstructions.append(YUVScoreInstruction(timeRange: CMTimeRange(start: time(outputStart), duration: time(end - start)),
                                videoTrackID: video.trackID, transform: upright, raster: converted, sourceStartMs: start,
                                pointRevealTimestampMs: snapshot.revealTimestampMs,
                                fadeScoreOverlay: project.draft.fadeScoreOverlay == true))
                            // Preparing many score states must not starve scene changes
                            // or cancellation while UIKit rasterizes their small strips.
                            await Task.yield()
                        }
                    }
                    videoComposition.instructions = overlayInstructions
                }
                let chapterOptions = project.draft.chapterOptions ?? YouTubeChapters.defaultOptions(
                    hasScoreTracking: tracking.enabled, hasSideSwitches: !tracking.sideSwitchMarkers.isEmpty)
                let chapters = YouTubeChapters.build(intervals: intervals, cuts: project.draft.cuts, scoreTracking: tracking,
                    options: chapterOptions)
                try Data(YouTubeChapters.text(chapters, includeCredit: chapterOptions.includeCredit).utf8).write(to: temporaryChapters, options: .withoutOverwriting)
                guard let session = AVAssetExportSession(asset: composition, presetName: sourceIsHDR ? AVAssetExportPresetHEVCHighestQuality : AVAssetExportPresetHighestQuality),
                      session.supportedFileTypes.contains(.mp4) else { throw ProjectError.invalid("Source-color MP4 export is unavailable for this recording") }
                session.outputURL = temporaryVideo
                session.outputFileType = .mp4
                session.shouldOptimizeForNetworkUse = true
                session.videoComposition = videoComposition
                session.timeRange = CMTimeRange(start: .zero, duration: time(timeline.durationMs))
                let operation = ExportSessionOperation(session)
                let backgroundTask: UIBackgroundTaskIdentifier = managesBackgroundLease
                    ? UIApplication.shared.beginBackgroundTask(withName: "VolleySplice video export") { operation.cancel() } : .invalid
                defer { if backgroundTask != .invalid { UIApplication.shared.endBackgroundTask(backgroundTask) } }
                let monitor = Task { @MainActor in
                    while !Task.isCancelled {
                        progress(0.05 + Double(session.progress) * 0.90, sourceIsHDR ? "Encoding HDR HEVC video and AAC audio" : "Encoding H.264 video and AAC audio")
                        do { try await Task.sleep(for: .milliseconds(250)) } catch { break }
                    }
                }
                defer { monitor.cancel() }
                try await withTaskCancellationHandler {
                    try Task.checkCancellation()
                    try await withCheckedThrowingContinuation { continuation in operation.start(continuation) }
                } onCancel: { operation.cancel() }
                monitor.cancel()
                try Task.checkCancellation()
                progress(0.96, "Embedding chapter titles and checking chapter timing")
                let embeddedChapterCount = try await MP4ChapterWriter.embed(in: temporaryVideo, chapters: chapters, durationMs: timeline.durationMs)
                progress(0.98, "Checking exported duration and tracks")
                let exported = AVURLAsset(url: temporaryVideo)
                let (actualDuration, exportedVideo, exportedAudio) = try await withTaskCancellationHandler {
                    try Task.checkCancellation()
                    let duration = try await exported.load(.duration).seconds
                    let video = try await exported.loadTracks(withMediaType: .video)
                    let audio = try await exported.loadTracks(withMediaType: .audio)
                    return (duration, video, audio)
                } onCancel: { exported.cancelLoading() }
                if let exportColor {
                    guard let track = exportedVideo.first, let format = try await track.load(.formatDescriptions).first else {
                        throw ProjectError.invalid("Export verification failed: missing output color description")
                    }
                    try exportColor.verify(format)
                }
                let expectedAudio = audio?.segments.contains { !$0.isEmpty } ?? false
                guard !exportedVideo.isEmpty, !expectedAudio || !exportedAudio.isEmpty,
                      abs(actualDuration - Double(timeline.durationMs) / 1000) <= max(0.1, frameDuration.seconds * 2) else {
                    throw ProjectError.invalid("Export verification failed: duration or media tracks do not match the planned cuts")
                }
                try Task.checkCancellation()
                // Move never replaces existing files; stage on the destination volume.
                try await CoordinatedVideoExport.updateFolder(parent, reading: source) { folder in
                    let chapters = folder.appendingPathComponent(chapterDestination.lastPathComponent)
                    let video = folder.appendingPathComponent(destination.lastPathComponent)
                    try manager.moveItem(at: temporaryChapters, to: chapters)
                    do { try manager.moveItem(at: temporaryVideo, to: video) }
                    catch { try? manager.removeItem(at: chapters); throw error }
                }
                progress(1, "Video and YouTube chapters exported")
                return VideoExportResult(videoURL: destination, chaptersURL: chapterDestination, durationMs: timeline.durationMs,
                    notes: ["\(embeddedChapterCount) chapters embedded in the MP4 and saved in a UTF-8 YouTube sidecar.",
                            "Video is reencoded as upright \(sourceIsHDR ? "10-bit HDR HEVC" : "H.264") with a constant source frame cadence; audio uses AAC.",
                            sourceIsHDR ? "Source HDR primaries and transfer are retained; original Dolby Vision dynamic metadata is not copied after editing." : (exportColor == nil ? "Source has no complete color tags; AVFoundation selected the output color description." : "Source color space is retained.")])
            }
        } onCancel: { asset.cancelLoading() }
    }

    private static func withStaging<T: Sendable>(in parent: URL, source: URL,
        operation: @MainActor (URL) async throws -> T) async throws -> T {
        let staging = try await CoordinatedVideoExport.updateFolder(parent, reading: source) { folder in
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let staging = folder.appendingPathComponent(".volleysplice-export-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
            return staging
        }
        do {
            let result = try await operation(staging)
            try? await CoordinatedVideoExport.updateFolder(parent, cleanup: true) { _ in
                try FileManager.default.removeItem(at: staging)
            }
            return result
        } catch {
            try? await CoordinatedVideoExport.updateFolder(parent, cleanup: true) { _ in
                try FileManager.default.removeItem(at: staging)
            }
            throw error
        }
    }

    private static func time(_ milliseconds: Int64) -> CMTime { CMTime(value: milliseconds, timescale: 1000) }

}

/// Serializes session lifecycle calls away from the main thread. Cancelling a
/// session may synchronously drain its compositor; UIKit must remain responsive
/// while that happens. The caller waits for AVFoundation's terminal callback,
/// keeping staging files, source access and the queue's worker alive until then.
private final class ExportSessionOperation: @unchecked Sendable {
    private let session: AVAssetExportSession
    private let lock = NSLock()
    private let lifecycle = DispatchQueue(label: "com.volleysplice.export.lifecycle", qos: .userInitiated)
    private var cancelled = false
    // Accessed only on lifecycle.
    private var started = false
    init(_ session: AVAssetExportSession) { self.session = session }
    private var cancellationRequested: Bool {
        lock.lock(); defer { lock.unlock() }; return cancelled
    }
    func start(_ continuation: CheckedContinuation<Void, any Error>) {
        lifecycle.async { [self] in
            if cancellationRequested { continuation.resume(throwing: CancellationError()); return }
            started = true
            session.exportAsynchronously { [self] in
                // Do not block an AVFoundation callback on our lifecycle queue:
                // cancelExport may be waiting for this callback to return.
                lifecycle.async { [self] in
                    started = false
                    if cancellationRequested { continuation.resume(throwing: CancellationError()); return }
                    switch session.status {
                    case .completed: continuation.resume()
                    case .cancelled: continuation.resume(throwing: CancellationError())
                    default: continuation.resume(throwing: session.error ?? ProjectError.invalid("Video encoding failed"))
                    }
                }
            }
        }
    }
    func cancel() {
        lock.lock()
        let firstCancellation = !cancelled
        cancelled = true
        lock.unlock()
        guard firstCancellation else { return }
        lifecycle.async { [self] in
            if started { session.cancelExport() }
        }
    }
}
