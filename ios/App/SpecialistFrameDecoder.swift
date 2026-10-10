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
        let outputSettings = try await AnalysisVideoColor.outputSettings(for: track)
        for group in groups {
            try Task.checkCancellation()
            try SpecialistFrameReader.read(asset: asset, track: track, targets: group, duration: media.duration,
                                           outputSettings: outputSettings) { buffer, pts, indices in
                let targets = group[indices]
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
                progress(Double(result.decodedTimes.count) / Double(max(1, requested.count)), "Score frames \(result.decodedTimes.count)/\(requested.count)")
            }
        }
        return result
    }
}
