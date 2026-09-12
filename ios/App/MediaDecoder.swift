import AVFoundation
import CoreVideo
import Foundation

struct AnalysisRegion: Codable, Equatable, Sendable {
    var x = 0.03, y = 0.12, width = 0.94, height = 0.86
    /// New analyses use every pixel; retain the legacy default for old projects.
    static let fullFrame = AnalysisRegion(x: 0, y: 0, width: 1, height: 1)
    /// Android AnalysisEngine.inferRoi: ordered, case-sensitive filename matches.
    static func inferred(filename: String) -> AnalysisRegion {
        let profiles: [(String, AnalysisRegion)] = [
            ("beach-source-02", AnalysisRegion(x: 0.02, y: 0.12, width: 0.96, height: 0.86)),
            ("beach-source-01", AnalysisRegion(x: 0.02, y: 0.12, width: 0.96, height: 0.86)),
            ("Dm", AnalysisRegion(x: 0.02, y: 0.22, width: 0.96, height: 0.76)),
            ("GYU", AnalysisRegion(x: 0.02, y: 0.18, width: 0.96, height: 0.80)),
            ("qpd", AnalysisRegion(x: 0.02, y: 0.18, width: 0.96, height: 0.80)),
            ("rSs", AnalysisRegion(x: 0.02, y: 0.22, width: 0.96, height: 0.76)),
            ("9lc", AnalysisRegion(x: 0.04, y: 0.14, width: 0.92, height: 0.84)),
            ("indoor-source-07", AnalysisRegion(x: 0.04, y: 0.14, width: 0.92, height: 0.84)),
            ("tds", AnalysisRegion(x: 0.03, y: 0.12, width: 0.94, height: 0.86))
        ]
        return profiles.first { filename.contains($0.0) }?.1 ?? AnalysisRegion()
    }
    func validate() throws {
        guard [x,y,width,height].allSatisfy(\.isFinite), x >= 0, y >= 0, width > 0, height > 0,
              x + width <= 1, y + height <= 1 else { throw AnalysisError.invalid("Invalid court region") }
    }
}

struct MediaDescription: Codable, Sendable {
    var duration: Double
    var width: Int
    var height: Int
    var rotation: Int
    var audio: Bool
    var videoCodec: String?
    var audioCodec: String?
}

/// Used on the device. The lab Mac has no GPU acceleration and runs math tests only.
enum MediaDecoder {
    static func describe(_ url: URL) async throws -> MediaDescription {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw AnalysisError.invalid("No video track")
        }
        let duration = try await asset.load(.duration).seconds
        let size = try await track.load(.naturalSize), transform = try await track.load(.preferredTransform)
        let rotation = (Int((atan2(transform.b, transform.a) * 180 / .pi).rounded()) + 360) % 360
        guard duration.isFinite, duration > 0, [0,90,180,270].contains(rotation), transform.determinant > 0 else {
            throw AnalysisError.invalid("Unsupported video duration or display transform")
        }
        let audioTrack = try await asset.loadTracks(withMediaType: .audio).first
        let videoFormat = try await track.load(.formatDescriptions).first
        let audioFormat = try await audioTrack?.load(.formatDescriptions).first
        func codec(_ format: CMFormatDescription?) -> String? {
            guard let format else { return nil }
            switch CMFormatDescriptionGetMediaSubType(format) {
            case kCMVideoCodecType_H264: return "video/avc"
            case kCMVideoCodecType_HEVC: return "video/hevc"
            case kAudioFormatMPEG4AAC: return "audio/mp4a-latm"
            default: return "fourcc:\(CMFormatDescriptionGetMediaSubType(format))"
            }
        }
        return MediaDescription(duration: duration, width: Int(size.width), height: Int(size.height), rotation: rotation,
                                audio: audioTrack != nil, videoCodec: codec(videoFormat), audioCodec: codec(audioFormat))
    }

    static func video(url: URL, media: MediaDescription, roi: AnalysisRegion, start: Double, end: Double,
                      resumeTimes: [Double] = [], resumeVisual: [Float] = [],
                      measurement: @Sendable (Double, Int, Double) -> Void = { _, _, _ in },
                      progress: @Sendable (Double, String) -> Void,
                      checkpoint: ([Double], [Double], [Float]) throws -> Void) async throws -> (times: [Double], decodedTimes: [Double], visual: [Float]) {
        try roi.validate()
        guard start.isFinite, end.isFinite, start >= 0, end > start, end <= media.duration + 0.001,
              resumeVisual.count == resumeTimes.count * 73 else { throw AnalysisError.invalid("Invalid analysis window/cache") }
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw AnalysisError.invalid("No video track") }
        let reader = try AVAssetReader(asset: asset)
        // Request video-range NV12 and reproduce Android's integer YUV sampler.
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw AnalysisError.invalid("Video decoder unavailable") }
        reader.add(output)
        var times = resumeTimes, decodedTimes: [Double] = [], visual = resumeVisual
        let firstTick = Int(ceil(start * 4 - 1e-9)), resumeTick = firstTick + resumeTimes.count
        // Decode one prior analysis frame to restore the temporal OpenCV state.
        var targetTick = max(firstTick, resumeTick - 1)
        let warmStart = Double(targetTick) / 4
        // The final requested target may need the first frame just after the game end.
        reader.timeRange = CMTimeRange(start: CMTime(seconds: max(0, warmStart - 0.001), preferredTimescale: 60000), end: CMTime(seconds: media.duration, preferredTimescale: 60000))
        guard reader.startReading() else { throw reader.error ?? AnalysisError.invalid("Video decode did not start") }
        defer { reader.cancelReading() }
        let extractor = VisualFeatureExtractor()
        while let sample = output.copyNextSampleBuffer() {
            try Task.checkCancellation()
            let pts = CMSampleBufferGetPresentationTimeStamp(sample).seconds
            guard pts.isFinite else { throw AnalysisError.invalid("Invalid frame timestamp") }
            let target = Double(targetTick) / 4
            if target >= end || target >= media.duration { break }
            if pts + 0.001 < target { continue }
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else { throw AnalysisError.invalid("Missing decoded pixels") }
            let features = try autoreleasepool { try extractor.extract(sampleRGBA(pixelBuffer, roi: roi, rotation: media.rotation)) }
            if targetTick >= resumeTick {
                times.append(target); decodedTimes.append(pts); visual += features
                if times.count % 16 == 0 { try checkpoint(times, decodedTimes, visual) }
            }
            targetTick += 1
            // Match Android: at most one analysis sample per decoded source frame.
            progress(min(1, (pts - start) / (end - start)), "Decoded \(times.count) samples")
            measurement(min(1, (pts - start) / (end - start)), times.count, Double(times.count) / Double(FeatureSchema.analysisFPS))
        }
        if reader.status == .failed { throw reader.error ?? AnalysisError.invalid("Video decode failed") }
        try Task.checkCancellation()
        guard !times.isEmpty else { throw AnalysisError.invalid("No video samples in game window") }
        try checkpoint(times, decodedTimes, visual)
        return (times, decodedTimes, visual)
    }

    static func audio(url: URL, times: [Double], start: Double, end: Double,
                      progress: @Sendable (Double, String) -> Void) async throws -> [Float] {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .audio).first else {
            return [Float](repeating: 0, count: times.count * 27)
        }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            AVFormatIDKey: kAudioFormatLinearPCM, AVLinearPCMIsFloatKey: true,
            AVLinearPCMBitDepthKey: 32, AVLinearPCMIsNonInterleaved: false, AVLinearPCMIsBigEndianKey: false
        ])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw AnalysisError.invalid("Audio decoder unavailable") }
        reader.add(output)
        reader.timeRange = CMTimeRange(start: CMTime(seconds: start, preferredTimescale: 60000), end: CMTime(seconds: end, preferredTimescale: 60000))
        guard reader.startReading() else { throw reader.error ?? AnalysisError.invalid("Audio decode did not start") }
        defer { reader.cancelReading() }
        let extractor = AudioFeatureExtractor()
        while let sample = output.copyNextSampleBuffer() {
            try Task.checkCancellation()
            guard let format = CMSampleBufferGetFormatDescription(sample),
                  let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(format)?.pointee,
                  let buffer = CMSampleBufferGetDataBuffer(sample) else { throw AnalysisError.invalid("Invalid PCM buffer") }
            let channels = Int(asbd.mChannelsPerFrame), count = CMSampleBufferGetNumSamples(sample)
            guard channels > 0, asbd.mBitsPerChannel == 32 else { throw AnalysisError.invalid("Unsupported PCM layout") }
            var pcm = [Float](repeating: 0, count: count * channels)
            let status = pcm.withUnsafeMutableBytes { CMBlockBufferCopyDataBytes(buffer, atOffset: 0, dataLength: $0.count, destination: $0.baseAddress!) }
            guard status == kCMBlockBufferNoErr else { throw AnalysisError.invalid("PCM copy failed") }
            var mono = [Float](repeating: 0, count: count)
            for frame in 0..<count {
                var sum = 0.0
                for channel in 0..<channels { sum += Double(pcm[frame * channels + channel]) }
                mono[frame] = Float(sum / Double(channels))
            }
            let pts = CMSampleBufferGetPresentationTimeStamp(sample).seconds
            let kept = min(mono.count, max(0, Int(ceil((end - pts) * asbd.mSampleRate))))
            try extractor.push(Array(mono.prefix(kept)), timestampSeconds: pts - start, sampleRate: Int(asbd.mSampleRate))
            progress(min(1, (pts - start) / (end - start)), "Generating audio features")
        }
        if reader.status == .failed { throw reader.error ?? AnalysisError.invalid("Audio decode failed") }
        try Task.checkCancellation()
        let features = try extractor.finishAndPool(times.map { $0 - start })
        try Task.checkCancellation()
        return features
    }

    static func sampleRGBA(_ buffer: CVPixelBuffer, roi: AnalysisRegion, rotation: Int, outputWidth: Int = 192, outputHeight: Int = 108) throws -> [UInt8] {
        guard CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
              CVPixelBufferGetPlaneCount(buffer) == 2 else { throw AnalysisError.invalid("Decoder did not return video-range NV12") }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let yPlane = CVPixelBufferGetBaseAddressOfPlane(buffer, 0)?.assumingMemoryBound(to: UInt8.self),
              let uvPlane = CVPixelBufferGetBaseAddressOfPlane(buffer, 1)?.assumingMemoryBound(to: UInt8.self) else { throw AnalysisError.invalid("Missing YUV planes") }
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        let clean = CVImageBufferGetCleanRect(buffer).intersection(CGRect(x: 0, y: 0, width: width, height: height))
        guard !clean.isNull, clean.width > 0, clean.height > 0 else { throw AnalysisError.invalid("Invalid video clean aperture") }
        let yStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0), uvStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1)
        var rgba = [UInt8](repeating: 255, count: outputWidth * outputHeight * 4)
        func byte(_ value: Int) -> UInt8 { UInt8(max(0, min(255, value))) }
        for y in 0..<outputHeight { for x in 0..<outputWidth {
            let u = roi.x + (Double(x) + 0.5) / Double(outputWidth) * roi.width, v = roi.y + (Double(y) + 0.5) / Double(outputHeight) * roi.height
            let su: Double, sv: Double
            switch rotation {
            case 90: su = v; sv = 1 - u
            case 180: su = 1 - u; sv = 1 - v
            case 270: su = 1 - v; sv = u
            default: su = u; sv = v
            }
            let sx = max(0, min(width - 1, Int(floor(clean.minX + su * clean.width))))
            let sy = max(0, min(height - 1, Int(floor(clean.minY + sv * clean.height))))
            let c = max(0, Int(yPlane[sy * yStride + sx]) - 16)
            let uv = sy / 2 * uvStride + sx / 2 * 2
            let d = Int(uvPlane[uv]) - 128, e = Int(uvPlane[uv + 1]) - 128, index = (y * outputWidth + x) * 4
            rgba[index] = byte((298 * c + 409 * e + 128) >> 8)
            rgba[index + 1] = byte((298 * c - 100 * d - 208 * e + 128) >> 8)
            rgba[index + 2] = byte((298 * c + 516 * d + 128) >> 8)
        } }
        return rgba
    }
}

private extension CGAffineTransform { var determinant: CGFloat { a * d - b * c } }
