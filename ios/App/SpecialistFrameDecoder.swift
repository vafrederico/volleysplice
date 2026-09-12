import AVFoundation
import Foundation

/// One shared native decode schedule, restarting after gaps over five seconds.
/// Samples use Android's first PTS + 1 ms at/after target rule and terminal fallback.
enum SpecialistFrameDecoder {
    struct Samples {
        var servingGray: [Double: [UInt8]] = [:]
        var sideSwitchBGR: [Double: [UInt8]] = [:]
        var decodedTimes: [Double: Double] = [:]
        var segments = 0
    }
    static func decode(url: URL, media: MediaDescription, roi: AnalysisRegion,
                       servingTimes: [Double], sideSwitchTimes: [Double] = [],
                       progress: @Sendable (Double, String) -> Void) async throws -> Samples {
        try roi.validate()
        let requested = try ServingSideInference.sharedRequestedTimestamps(serving: servingTimes, sideSwitch: sideSwitchTimes)
        guard requested.allSatisfy({ $0 < media.duration }) else { throw AnalysisError.invalid("Specialist timestamp exceeds source") }
        let serving = Set(servingTimes), switches = Set(sideSwitchTimes)
        var groups: [[Double]] = []
        for time in requested {
            if let last = groups.last?.last, time - last <= 5 { groups[groups.count - 1].append(time) }
            else { groups.append([time]) }
        }
        var result = Samples(); result.segments = groups.count
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw AnalysisError.invalid("Missing specialist video track") }
        for group in groups {
            try Task.checkCancellation()
            let reader = try AVAssetReader(asset: asset)
            let output = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange])
            output.alwaysCopiesSampleData = false
            guard reader.canAdd(output) else { throw AnalysisError.invalid("Specialist decoder unavailable") }
            reader.add(output)
            reader.timeRange = CMTimeRange(start: CMTime(seconds: max(0, group[0] - 0.001), preferredTimescale: 60000),
                                          end: CMTime(seconds: media.duration, preferredTimescale: 60000))
            guard reader.startReading() else { throw reader.error ?? AnalysisError.invalid("Specialist decode did not start") }
            defer { reader.cancelReading() }
            var index = 0
            var fallback: (buffer: CVPixelBuffer, pts: Double)?
            func store(_ buffer: CVPixelBuffer, pts: Double, through upper: Int) throws {
                let targets = group[index..<upper]
                let gray = try targets.contains(where: { serving.contains($0) }) ? ServingSideFeatureExtractor.grayscale(MediaDecoder.sampleRGBA(buffer, roi: roi, rotation: media.rotation)) : nil
                var bgr: [UInt8]?
                if targets.contains(where: { switches.contains($0) }) {
                    let rgba = try MediaDecoder.sampleRGBA(buffer, roi: roi, rotation: media.rotation, outputWidth: 256, outputHeight: 144)
                    var pixels = [UInt8](repeating: 0, count: 256 * 144 * 3)
                    for p in 0..<(256 * 144) { pixels[p * 3] = rgba[p * 4 + 2]; pixels[p * 3 + 1] = rgba[p * 4 + 1]; pixels[p * 3 + 2] = rgba[p * 4] }
                    bgr = pixels
                }
                for time in targets {
                    if serving.contains(time) { result.servingGray[time] = gray }
                    if switches.contains(time) { result.sideSwitchBGR[time] = bgr }
                    result.decodedTimes[time] = pts
                }
                index = upper
                progress(Double(result.decodedTimes.count) / Double(max(1, requested.count)), "Score frames \(result.decodedTimes.count)/\(requested.count)")
            }
            while index < group.count, let sample = output.copyNextSampleBuffer() {
                try Task.checkCancellation()
                let pts = CMSampleBufferGetPresentationTimeStamp(sample).seconds
                guard pts.isFinite, let buffer = CMSampleBufferGetImageBuffer(sample) else { throw AnalysisError.invalid("Invalid specialist frame") }
                if pts + 0.001 < group[index] {
                    if pts >= media.duration - 1 { fallback = (buffer, pts) }
                    continue
                }
                var upper = index + 1
                while upper < group.count && pts + 0.001 >= group[upper] { upper += 1 }
                try autoreleasepool { try store(buffer, pts: pts, through: upper) }
                fallback = nil
            }
            if reader.status == .failed { throw reader.error ?? AnalysisError.invalid("Specialist frame decode failed") }
            if index < group.count, let fallback { try store(fallback.buffer, pts: fallback.pts, through: group.count) }
            guard index == group.count else { throw AnalysisError.invalid("Specialist frame segment ended early") }
        }
        return result
    }
}
