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

    /// The optional neural consumer runs in the same sequential decoder pass.
    /// RGB values are fractional 0...255 HWC pixels, with a black 224-square
    /// letterbox. Its synchronous callback bounds memory and provides backpressure.
    static func video(url: URL, media: MediaDescription, roi: AnalysisRegion, start: Double, end: Double,
                      resumeTimes: [Double] = [], resumeVisual: [Float] = [],
                      embeddingFrame: ((Double, Double, [Float], NeuralImageGeometry) throws -> Void)? = nil,
                      measurement: @Sendable (Double, Int, Double) -> Void = { _, _, _ in },
                      progress: @Sendable (Double, String) -> Void,
                      checkpoint: ([Double], [Double], [Float]) throws -> Void) async throws -> (times: [Double], decodedTimes: [Double], visual: [Float]) {
        try roi.validate()
        let targets = try VideoFrameSelection.targets(start: start, end: end, fps: 4)
        guard end <= media.duration + 0.001, resumeVisual.count == resumeTimes.count * 73,
              resumeTimes == Array(targets.prefix(resumeTimes.count)), resumeTimes.count <= targets.count,
              embeddingFrame == nil || resumeTimes.isEmpty else {
            throw AnalysisError.invalid("Invalid analysis window/cache")
        }
        var times = resumeTimes, decodedTimes: [Double] = [], visual = resumeVisual
        let firstRow = max(0, resumeTimes.count - 1)
        let extractor = VisualFeatureExtractor()
        let neuralTargets = embeddingFrame == nil ? [] : try VideoFrameSelection.targets(start: floor(start * 2) / 2, end: end, fps: 2)
        var neuralRow = 0, visualRow = firstRow
        try await frames(url: url, end: end, targets: Array(Set(Array(targets.dropFirst(firstRow)) + neuralTargets)).sorted()) { target, pts, buffer in
            try Task.checkCancellation()
            if neuralRow < neuralTargets.count && target == neuralTargets[neuralRow], let embeddingFrame {
                let frame = try sampleEmbeddingFrame(buffer, roi: roi, rotation: media.rotation)
                try embeddingFrame(target, pts, frame.pixels, frame.geometry)
                neuralRow += 1
            }
            // The neural plan can include a half-second tick before a clipped game.
            if visualRow < targets.count && targets[visualRow] == target {
                let features = try extractor.extract(sampleRGBA(buffer, roi: roi, rotation: media.rotation))
                if visualRow >= resumeTimes.count {
                    times.append(target); decodedTimes.append(pts); visual += features
                    if times.count % 16 == 0 { try checkpoint(times, decodedTimes, visual) }
                }
                visualRow += 1
            }
            let fraction = min(1, max(0, (target - start + 0.25) / (end - start)))
            progress(fraction, embeddingFrame == nil ? "Decoded \(times.count) video samples" : "Video features + selected model images: \(times.count) samples")
            measurement(fraction, times.count, Double(times.count) / Double(FeatureSchema.analysisFPS))
        }
        guard times.count == targets.count, !times.isEmpty, neuralRow == neuralTargets.count else {
            throw AnalysisError.invalid("Incomplete video sampling")
        }
        try checkpoint(times, decodedTimes, visual)
        return (times, decodedTimes, visual)
    }

    /// Cached AV features can reuse the same sampler without recomputing OpenCV.
    static func embeddings(url: URL, media: MediaDescription, roi: AnalysisRegion, start: Double, end: Double,
                           progress: @Sendable (Double, String) -> Void,
                           embeddingFrame: (Double, Double, [Float], NeuralImageGeometry) throws -> Void) async throws {
        try roi.validate()
        guard end <= media.duration + 0.001 else { throw AnalysisError.invalid("Invalid embedding window") }
        let targets = try VideoFrameSelection.targets(start: floor(start * 2) / 2, end: end, fps: 2)
        var completed = 0
        try await frames(url: url, end: end, targets: targets) { target, pts, buffer in
            let frame = try sampleEmbeddingFrame(buffer, roi: roi, rotation: media.rotation)
            try embeddingFrame(target, pts, frame.pixels, frame.geometry)
            completed += 1
            progress(Double(completed) / Double(targets.count), "Selected model images: \(completed) / \(targets.count)")
        }
        guard completed == targets.count else { throw AnalysisError.invalid("Incomplete embedding sampling") }
    }

    /// Keep only two decoded NV12 buffers, and process each target as soon as its
    /// bracketing source frame arrives. Repeated target selections intentionally
    /// reuse the same image, including at low or variable source frame rates.
    private static func frames(url: URL, end: Double, targets: [Double],
                               consume: (Double, Double, CVPixelBuffer) throws -> Void) async throws {
        guard !targets.isEmpty else { return }
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw AnalysisError.invalid("No video track") }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw AnalysisError.invalid("Video decoder unavailable") }
        reader.add(output)
        // Read source PTS, never infer them from average FPS or seek per target.
        // Starting at zero also preserves the prior frame for clipped windows.
        guard reader.startReading() else { throw reader.error ?? AnalysisError.invalid("Video decode did not start") }
        defer { reader.cancelReading() }
        var previous: (Double, CVPixelBuffer)?, row = 0
        while let sample = output.copyNextSampleBuffer() {
            try Task.checkCancellation()
            let pts = CMSampleBufferGetPresentationTimeStamp(sample).seconds
            guard pts.isFinite else { throw AnalysisError.invalid("Invalid frame timestamp") }
            if pts >= end { break } // Same exclusive source-frame limit as Android.
            if let previous {
                guard pts >= previous.0 else { throw AnalysisError.invalid("Unsorted decoded frame timestamps") }
                if pts == previous.0 { continue } // First image wins duplicate PTS.
            }
            guard let buffer = CMSampleBufferGetImageBuffer(sample) else { throw AnalysisError.invalid("Missing decoded pixels") }
            while row < targets.count && targets[row] <= pts {
                let chosen: (Double, CVPixelBuffer)
                if let previous, VideoFrameSelection.prefersEarlier(target: targets[row], earlier: previous.0, later: pts) {
                    chosen = previous
                } else { chosen = (pts, buffer) }
                try autoreleasepool { try consume(targets[row], chosen.0, chosen.1) }
                row += 1
            }
            previous = (pts, buffer)
            if row == targets.count { break }
        }
        if reader.status == .failed { throw reader.error ?? AnalysisError.invalid("Video decode failed") }
        if let previous {
            while row < targets.count {
                try Task.checkCancellation()
                try autoreleasepool { try consume(targets[row], previous.0, previous.1) }
                row += 1
            }
        }
        guard row == targets.count else { throw AnalysisError.invalid("No video samples in game window") }
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
            progress(min(0.8, max(0, 0.8 * (pts - start) / (end - start))), "Decoding audio + generating feature frames")
        }
        if reader.status == .failed { throw reader.error ?? AnalysisError.invalid("Audio decode failed") }
        try Task.checkCancellation()
        let features = try extractor.finishAndPool(times.map { $0 - start }, progress: progress)
        try Task.checkCancellation()
        return features
    }

    private static func withNV12<T>(_ buffer: CVPixelBuffer,
                                    _ body: (UnsafePointer<UInt8>, Int32, UnsafePointer<UInt8>, Int32, CGRect, Int32, Int32) throws -> T) throws -> T {
        let format = CVPixelBufferGetPixelFormatType(buffer)
        guard [kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange].contains(format),
              CVPixelBufferGetPlaneCount(buffer) == 2 else { throw AnalysisError.invalid("Decoder did not return 8-bit NV12") }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let y = CVPixelBufferGetBaseAddressOfPlane(buffer, 0)?.assumingMemoryBound(to: UInt8.self),
              let uv = CVPixelBufferGetBaseAddressOfPlane(buffer, 1)?.assumingMemoryBound(to: UInt8.self) else { throw AnalysisError.invalid("Missing YUV planes") }
        let bounds = CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(buffer), height: CVPixelBufferGetHeight(buffer))
        let clean = CVImageBufferGetCleanRect(buffer).intersection(bounds)
        guard !clean.isNull, clean.width >= 1, clean.height >= 1 else { throw AnalysisError.invalid("Invalid video clean aperture") }
        let matrix: Int32
        if let value = CVBufferCopyAttachment(buffer, kCVImageBufferYCbCrMatrixKey, nil) {
            if CFEqual(value, kCVImageBufferYCbCrMatrix_ITU_R_709_2) { matrix = 1 }
            else if CFEqual(value, kCVImageBufferYCbCrMatrix_ITU_R_601_4) { matrix = 0 }
            else if CFEqual(value, kCVImageBufferYCbCrMatrix_ITU_R_2020) { matrix = 2 }
            else { throw AnalysisError.invalid("Unsupported decoded YUV matrix") }
        } else { matrix = 0 } // Explicit BT.601 fallback, matching Android.
        return try body(y, Int32(CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)), uv,
                        Int32(CVPixelBufferGetBytesPerRowOfPlane(buffer, 1)), clean, matrix,
                        format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange ? 1 : 0)
    }

    static func sampleRGBA(_ buffer: CVPixelBuffer, roi: AnalysisRegion, rotation: Int, outputWidth: Int = 192, outputHeight: Int = 108) throws -> [UInt8] {
        try roi.validate()
        guard outputWidth > 0, outputHeight > 0 else { throw AnalysisError.invalid("Invalid sample dimensions") }
        var rgba = [UInt8](repeating: 0, count: outputWidth * outputHeight * 4)
        let status = try withNV12(buffer) { y, ys, uv, uvs, clean, matrix, full in
            vc_nv12_area(y, ys, uv, uvs, Int32(clean.minX), Int32(clean.minY), Int32(clean.width), Int32(clean.height),
                         Int32(rotation), matrix, full, [roi.x, roi.y, roi.width, roi.height], Int32(outputWidth), Int32(outputHeight), &rgba)
        }
        guard status == 0 else { throw AnalysisError.invalid("NV12 area sampling failed") }
        return rgba
    }

    static func sampleEmbeddingRGB(_ buffer: CVPixelBuffer, roi: AnalysisRegion, rotation: Int) throws -> [Float] {
        try sampleEmbeddingFrame(buffer, roi: roi, rotation: rotation).pixels
    }

    /// Return the exact clean-aperture geometry used to prepare these pixels.
    /// Encoded track dimensions can include padding and must not set pool masks.
    static func sampleEmbeddingFrame(_ buffer: CVPixelBuffer, roi: AnalysisRegion, rotation: Int) throws -> (pixels: [Float], geometry: NeuralImageGeometry) {
        try roi.validate()
        var rgb = [Float](repeating: 0, count: 224 * 224 * 3)
        return try withNV12(buffer) { y, ys, uv, uvs, clean, matrix, full in
            let geometry = try NeuralRallyContract.geometry(width: Int(clean.width), height: Int(clean.height), rotation: rotation,
                                                           roi: [roi.x, roi.y, roi.width, roi.height])
            let values = [geometry.x, geometry.y, geometry.cropWidth, geometry.cropHeight, geometry.resizedWidth, geometry.resizedHeight, geometry.left, geometry.top].map(Int32.init)
            let status = vc_nv12_letterbox(y, ys, uv, uvs, Int32(clean.minX), Int32(clean.minY), Int32(clean.width), Int32(clean.height),
                                          Int32(rotation), matrix, full, values, 224, &rgb)
            guard status == 0 else { throw AnalysisError.invalid("NV12 embedding sampling failed") }
            return (rgb, geometry)
        }
    }
}

private extension CGAffineTransform { var determinant: CGFloat { a * d - b * c } }
