import AVFoundation
import Foundation

/// Shared AVFoundation loop, also exercised by the small CPU-only decode probe.
enum SpecialistFrameReader {
    struct Incomplete: Error, LocalizedError, CustomStringConvertible {
        let description: String
        var errorDescription: String? { description }
    }

    static func read(asset: AVAsset, track: AVAssetTrack, targets: [Double], duration: Double,
                     outputSettings: [String: Any],
                     consume: (CVPixelBuffer, Double, Range<Int>) throws -> Void) throws {
        var selection = try SpecialistFrameSelection(targets: targets, duration: duration)
        guard !selection.isComplete else { return }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw Incomplete(description: "Specialist decoder unavailable") }
        reader.add(output)
        // A target can lie after the last frame's PTS but within its duration.
        // Keep pre-roll so even an isolated terminal request sees that image.
        let start = selection.readStart
        reader.timeRange = CMTimeRange(start: CMTime(seconds: start, preferredTimescale: 60000),
                                      end: CMTime(seconds: duration, preferredTimescale: 60000))
        guard reader.startReading() else { throw reader.error ?? Incomplete(description: "Specialist decode did not start") }
        defer { reader.cancelReading() }
        var terminal: (buffer: CVPixelBuffer, pts: Double)?
        while !selection.isComplete, let sample = output.copyNextSampleBuffer() {
            try Task.checkCancellation()
            let pts = CMSampleBufferGetPresentationTimeStamp(sample).seconds
            guard pts.isFinite, let buffer = CMSampleBufferGetImageBuffer(sample) else {
                throw Incomplete(description: "Invalid specialist frame")
            }
            // Retain the actual image even if it also satisfies a previous request.
            // EOS may arrive separately, without another pixel buffer.
            if pts >= duration - 1 { terminal = (buffer, pts) }
            if let indices = selection.accept(presentationTime: pts) {
                try autoreleasepool { try consume(buffer, pts, indices) }
            }
        }
        try Task.checkCancellation()
        if reader.status == .failed { throw reader.error ?? Incomplete(description: "Specialist frame decode failed") }
        // Only a completed reader may reuse a terminal image. A truncated middle
        // segment, cancellation or decode failure must still fail explicitly.
        if let terminal, let indices = selection.finish(completed: reader.status == .completed) {
            try autoreleasepool { try consume(terminal.buffer, terminal.pts, indices) }
        }
        guard selection.isComplete else {
            let pts = selection.lastPTS.map { String(format: "%.3f", $0) } ?? "none"
            throw Incomplete(description: String(format:
                "Specialist frame segment ended early (target %.3fs, last frame %@s, source %.3fs, reader %ld)",
                targets[selection.index], pts, duration, reader.status.rawValue))
        }
    }
}
