@preconcurrency import AVFoundation
import Accelerate
import Foundation

/// Source 420v samples never take a full-frame RGB round trip. Immutable CPU
/// score graphics touch only the top strip, with point opacity in source time.
final class YUVScoreInstruction: NSObject, AVVideoCompositionInstructionProtocol, @unchecked Sendable {
    let timeRange: CMTimeRange
    let enablePostProcessing = false
    let containsTweening = true
    let passthroughTrackID = kCMPersistentTrackID_Invalid
    let requiredSourceTrackIDs: [NSValue]?
    let videoTrackID: CMPersistentTrackID
    let transform: CGAffineTransform
    let raster: ScoreOverlayRaster
    let sourceStartMs: Int64
    let pointRevealTimestampMs: Int64?
    let uses601Matrix: Bool

    init(timeRange: CMTimeRange, videoTrackID: CMPersistentTrackID, transform: CGAffineTransform,
         raster: ScoreOverlayRaster, sourceStartMs: Int64, pointRevealTimestampMs: Int64?, uses601Matrix: Bool) {
        self.timeRange = timeRange; self.videoTrackID = videoTrackID
        self.transform = transform; self.raster = raster
        self.sourceStartMs = sourceStartMs; self.pointRevealTimestampMs = pointRevealTimestampMs
        self.uses601Matrix = uses601Matrix
        requiredSourceTrackIDs = [NSNumber(value: videoTrackID)]
        super.init()
    }
}

final class YUVScoreCompositor: NSObject, AVVideoCompositing, @unchecked Sendable {
    // No CA input: AVFoundation supplies only source 420v, preserving its samples.
    var sourcePixelBufferAttributes: [String: Any]? {
        [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange]
    }
    var requiredPixelBufferAttributesForRenderContext: [String: Any] {
        [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange]
    }
    var canConformColorOfSourceFrames: Bool { true }
    var supportsWideColorSourceFrames: Bool { false }
    var supportsHDRSourceFrames: Bool { false }
    private let cancellationLock = NSCondition()
    private var cancellationGeneration: UInt64 = 0
    private var inFlightRequests = 0
    private var cancellationDrainers = 0
    func renderContextChanged(_ newRenderContext: AVVideoCompositionRenderContext) {}
    func cancelAllPendingVideoCompositionRequests() {
        cancellationLock.lock()
        cancellationGeneration &+= 1
        cancellationDrainers += 1
        // AVVideoCompositing requires this method to return only after every
        // pending request has called a finish method. A generation alone marks
        // frames as cancelled but does not provide that required drain barrier.
        while inFlightRequests > 0 { cancellationLock.wait() }
        cancellationDrainers -= 1
        cancellationLock.unlock()
    }
    private func generation() -> UInt64 {
        cancellationLock.lock(); defer { cancellationLock.unlock() }; return cancellationGeneration
    }
    func startRequest(_ request: AVAsynchronousVideoCompositionRequest) {
        cancellationLock.lock()
        let current = cancellationGeneration
        let alreadyCancelling = cancellationDrainers > 0
        inFlightRequests += 1
        cancellationLock.unlock()
        defer {
            cancellationLock.lock()
            inFlightRequests -= 1
            if inFlightRequests == 0 { cancellationLock.broadcast() }
            cancellationLock.unlock()
        }
        if alreadyCancelling { request.finishCancelledRequest(); return }
        // Finish synchronously; there is no unbounded frame queue or shared buffer.
        autoreleasepool {
            do {
                guard let instruction = request.videoCompositionInstruction as? YUVScoreInstruction,
                      let source = request.sourceFrame(byTrackID: instruction.videoTrackID),
                      let output = request.renderContext.newPixelBuffer() else {
                    throw ProjectError.invalid("The score compositor is missing a video frame")
                }
                let elapsed = CMTimeConvertScale(CMTimeSubtract(request.compositionTime, instruction.timeRange.start), timescale: 1000, method: .roundTowardZero).value
                let sourceTimestamp = instruction.sourceStartMs + max(0, elapsed)
                let opacity = instruction.pointRevealTimestampMs.map {
                    ExportTimeline.pointTimelineOpacity(sourceTimestampMs: sourceTimestamp, revealTimestampMs: $0)
                } ?? 0
                try Self.compose(source: source, raster: instruction.raster, output: output,
                                 transform: instruction.transform, pointOpacity: opacity, uses601Matrix: instruction.uses601Matrix)
                if generation() == current { request.finish(withComposedVideoFrame: output) }
                else { request.finishCancelledRequest() }
            } catch {
                if generation() == current { request.finish(with: error) }
                else { request.finishCancelledRequest() }
            }
        }
    }

    static func accepts(transform: CGAffineTransform) -> Bool {
        let values = [transform.a, transform.b, transform.c, transform.d]
        return values.allSatisfy { $0.isFinite && abs($0 - $0.rounded()) < 0.00001 && abs($0) <= 1.00001 }
            && abs(abs(transform.a) + abs(transform.c) - 1) < 0.00001
            && abs(abs(transform.b) + abs(transform.d) - 1) < 0.00001
            && abs(abs(transform.a * transform.d - transform.b * transform.c) - 1) < 0.00001
    }

    static func compose(source: CVPixelBuffer, raster: ScoreOverlayRaster, output: CVPixelBuffer,
                        transform: CGAffineTransform, pointOpacity: Float, uses601Matrix: Bool) throws {
        guard CVPixelBufferGetPixelFormatType(source) == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
              CVPixelBufferGetPixelFormatType(output) == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
              CVPixelBufferGetPlaneCount(source) == 2, CVPixelBufferGetPlaneCount(output) == 2,
              accepts(transform: transform) else {
            throw ProjectError.invalid("Score overlays require SDR video-range YUV and a right-angle source orientation")
        }
        guard CVPixelBufferLockBaseAddress(source, .readOnly) == kCVReturnSuccess else { throw ProjectError.invalid("Cannot read source pixels") }
        defer { CVPixelBufferUnlockBaseAddress(source, .readOnly) }
        guard CVPixelBufferLockBaseAddress(output, []) == kCVReturnSuccess else { throw ProjectError.invalid("Cannot write export pixels") }
        defer { CVPixelBufferUnlockBaseAddress(output, []) }
        for plane in 0..<2 {
            guard let input = CVPixelBufferGetBaseAddressOfPlane(source, plane),
                  let destination = CVPixelBufferGetBaseAddressOfPlane(output, plane) else { throw ProjectError.invalid("Missing YUV plane") }
            try copyPlane(input: input, output: destination,
                width: CVPixelBufferGetWidthOfPlane(source, plane), height: CVPixelBufferGetHeightOfPlane(source, plane),
                outputWidth: CVPixelBufferGetWidthOfPlane(output, plane), outputHeight: CVPixelBufferGetHeightOfPlane(output, plane),
                inputStride: CVPixelBufferGetBytesPerRowOfPlane(source, plane), outputStride: CVPixelBufferGetBytesPerRowOfPlane(output, plane),
                bytesPerPixel: plane == 0 ? 1 : 2, transform: transform)
        }
        // Preserve source primaries, transfer function, matrix and chroma tags.
        CVBufferPropagateAttachments(source, output)
        guard let y = CVPixelBufferGetBaseAddressOfPlane(output, 0)?.assumingMemoryBound(to: UInt8.self),
              let uv = CVPixelBufferGetBaseAddressOfPlane(output, 1)?.assumingMemoryBound(to: UInt8.self) else {
            throw ProjectError.invalid("Missing score blend pixels")
        }
        let width = min(CVPixelBufferGetWidth(output), raster.width)
        let height = min(CVPixelBufferGetHeight(output), raster.height)
        raster.pixels.withUnsafeBytes { bytes in
            blend(bgra: bytes.bindMemory(to: UInt8.self).baseAddress!, bgraStride: raster.rowBytes,
                  y: y, yStride: CVPixelBufferGetBytesPerRowOfPlane(output, 0),
                  uv: uv, uvStride: CVPixelBufferGetBytesPerRowOfPlane(output, 1), width: width, height: height,
                  uses601Matrix: uses601Matrix, pointStartX: raster.scoreWidth, pointOpacity: Double(pointOpacity))
        }
    }

    private static func copyPlane(input: UnsafeMutableRawPointer, output: UnsafeMutableRawPointer,
                                  width: Int, height: Int, outputWidth: Int, outputHeight: Int,
                                  inputStride: Int, outputStride: Int, bytesPerPixel: Int, transform: CGAffineTransform) throws {
        let a = Int(transform.a.rounded()), b = Int(transform.b.rounded())
        let c = Int(transform.c.rounded()), d = Int(transform.d.rounded())
        let rotatedWidth = a == 0 ? height : width, rotatedHeight = a == 0 ? width : height
        guard rotatedWidth <= outputWidth, rotatedHeight <= outputHeight else { throw ProjectError.invalid("Score compositor output geometry mismatch") }
        // Only optional one-pixel H.264 padding lies outside the rotated source.
        memset(output, bytesPerPixel == 1 ? 16 : 128, outputStride * outputHeight)
        if a == 1 && d == 1 {
            for row in 0..<height { memcpy(output.advanced(by: row * outputStride), input.advanced(by: row * inputStride), width * bytesPerPixel) }
            return
        }
        if a * d - b * c == 1 && rotatedWidth == outputWidth && rotatedHeight == outputHeight {
            var source = vImage_Buffer(data: input, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: inputStride)
            var target = vImage_Buffer(data: output, height: vImagePixelCount(outputHeight), width: vImagePixelCount(outputWidth), rowBytes: outputStride)
            let rotation = UInt8(a == -1 ? kRotate180DegreesClockwise : b == 1 ? kRotate90DegreesClockwise : kRotate270DegreesClockwise)
            // Treat each interleaved CbCr pair as an opaque 16-bit sample: the
            // rotation copies bytes without resampling or interpreting colors.
            let error = bytesPerPixel == 1
                ? vImageRotate90_Planar8(&source, &target, rotation, 16, vImage_Flags(kvImageNoFlags))
                : vImageRotate90_Planar16U(&source, &target, rotation, 0x8080, vImage_Flags(kvImageNoFlags))
            guard error == kvImageNoError else { throw ProjectError.invalid("YUV rotation failed (\(error))") }
            return
        }
        // Signed permutation maps also cover mirrored orientations and odd
        // dimensions, with padding anchored at the right/bottom rather than center.
        let tx = (a < 0 ? width - 1 : 0) + (c < 0 ? height - 1 : 0)
        let ty = (b < 0 ? width - 1 : 0) + (d < 0 ? height - 1 : 0)
        let src = input.assumingMemoryBound(to: UInt8.self), dst = output.assumingMemoryBound(to: UInt8.self)
        for sy in 0..<height {
            for sx in 0..<width {
                let from = sy * inputStride + sx * bytesPerPixel
                let to = (b * sx + d * sy + ty) * outputStride + (a * sx + c * sy + tx) * bytesPerPixel
                dst[to] = src[from]
                if bytesPerPixel == 2 { dst[to + 1] = src[from + 1] }
            }
        }
    }

    private static func byte(_ value: Double) -> UInt8 { UInt8(max(0, min(255, value.rounded()))) }
    private static func blend(bgra: UnsafePointer<UInt8>, bgraStride: Int,
                              y: UnsafeMutablePointer<UInt8>, yStride: Int,
                              uv: UnsafeMutablePointer<UInt8>, uvStride: Int, width: Int, height: Int, uses601Matrix: Bool = false,
                              pointStartX: Int = .max, pointOpacity: Double = 1) {
        // BT.709/601 non-linear RGB -> studio-range Y'CbCr. This is the standard
        // coding matrix for the graphics, not a transfer/gamma adjustment of video.
        let kr = uses601Matrix ? 0.299 : 0.2126, kb = uses601Matrix ? 0.114 : 0.0722
        for row in stride(from: 0, to: height, by: 2) {
            for column in stride(from: 0, to: width, by: 2) {
                var alphaSum = 0.0, cbSum = 0.0, crSum = 0.0
                for dy in 0..<2 where row + dy < height {
                    for dx in 0..<2 where column + dx < width {
                        let offset = (row + dy) * bgraStride + (column + dx) * 4
                        // BGRA is premultiplied: fade color and alpha together.
                        // Evaluate per pixel so an odd-width scoreboard does not
                        // fade the score side of its shared 2x2 chroma block.
                        let opacity = column + dx >= pointStartX ? max(0, min(1, pointOpacity)) : 1
                        let alpha = Double(bgra[offset + 3]) / 255 * opacity
                        if alpha == 0 { continue }
                        let blue = Double(bgra[offset]) * opacity, green = Double(bgra[offset + 1]) * opacity, red = Double(bgra[offset + 2]) * opacity
                        let luma = kr * red + (1 - kr - kb) * green + kb * blue
                        let yi = (row + dy) * yStride + column + dx
                        y[yi] = byte(Double(y[yi]) * (1 - alpha) + 16 * alpha + (219.0 / 255) * luma)
                        alphaSum += alpha
                        cbSum += (224.0 / 255) * (blue - luma) / (2 * (1 - kb))
                        crSum += (224.0 / 255) * (red - luma) / (2 * (1 - kr))
                    }
                }
                if alphaSum == 0 { continue }
                let index = (row / 2) * uvStride + (column / 2) * 2
                let alpha = alphaSum / 4
                uv[index] = byte(Double(uv[index]) * (1 - alpha) + 128 * alpha + cbSum / 4)
                uv[index + 1] = byte(Double(uv[index + 1]) * (1 - alpha) + 128 * alpha + crSum / 4)
            }
        }
    }

    #if DEBUG
    /// Runs on the iPad before the first real overlay frame. The expected pixel
    /// arrangements are explicit so this checks vImage direction and UV pairing.
    static func checkKernels() throws {
        let cases: [(CGAffineTransform, [Int])] = [
            (.identity, [0,1,2,3,4,5,6,7]),
            (.init(a: 0, b: 1, c: -1, d: 0, tx: 2, ty: 0), [4,0,5,1,6,2,7,3]),
            (.init(a: -1, b: 0, c: 0, d: -1, tx: 4, ty: 2), [7,6,5,4,3,2,1,0]),
            (.init(a: 0, b: -1, c: 1, d: 0, tx: 0, ty: 4), [3,7,2,6,1,5,0,4]),
            (.init(a: -1, b: 0, c: 0, d: 1, tx: 4, ty: 0), [3,2,1,0,7,6,5,4]),
            (.init(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: 2), [4,5,6,7,0,1,2,3]),
            (.init(a: 0, b: 1, c: 1, d: 0, tx: 0, ty: 0), [0,4,1,5,2,6,3,7]),
            (.init(a: 0, b: -1, c: -1, d: 0, tx: 2, ty: 4), [7,3,6,2,5,1,4,0])]
        for bytes in [1, 2] {
            let inputStride = 4 * bytes + 4
            var input = [UInt8](repeating: 99, count: inputStride * 2)
            for index in 0..<8 {
                let offset = (index / 4) * inputStride + (index % 4) * bytes
                input[offset] = UInt8(index)
                if bytes == 2 { input[offset + 1] = UInt8(255 - index) }
            }
            for (transform, expected) in cases {
                let width = transform.a == 0 ? 2 : 4, height = transform.a == 0 ? 4 : 2
                let outputStride = width * bytes + 4
                var output = [UInt8](repeating: 77, count: outputStride * height)
                try input.withUnsafeMutableBytes { source in
                    try output.withUnsafeMutableBytes { target in
                        try copyPlane(input: source.baseAddress!, output: target.baseAddress!, width: 4, height: 2,
                            outputWidth: width, outputHeight: height, inputStride: inputStride, outputStride: outputStride,
                            bytesPerPixel: bytes, transform: transform)
                    }
                }
                for index in 0..<8 {
                    let offset = (index / width) * outputStride + (index % width) * bytes
                    guard output[offset] == UInt8(expected[index]), bytes == 1 || output[offset + 1] == UInt8(255 - expected[index]) else {
                        throw ProjectError.invalid("YUV rotation kernel self-check failed")
                    }
                }
            }
        }
        var graphics = [UInt8](repeating: 0, count: 4 * 4 * 4)
        var luma = [UInt8](repeating: 100, count: 16), chroma = [UInt8](repeating: 128, count: 8)
        func runBlend(pointStartX: Int = .max, pointOpacity: Double = 1) {
            graphics.withUnsafeBufferPointer { image in
                luma.withUnsafeMutableBufferPointer { y in
                    chroma.withUnsafeMutableBufferPointer { uv in
                        blend(bgra: image.baseAddress!, bgraStride: 16, y: y.baseAddress!, yStride: 4,
                              uv: uv.baseAddress!, uvStride: 4, width: 4, height: 2,
                              pointStartX: pointStartX, pointOpacity: pointOpacity)
                    }
                }
            }
        }
        runBlend()
        guard luma.allSatisfy({ $0 == 100 }), chroma.allSatisfy({ $0 == 128 }) else { throw ProjectError.invalid("Transparent overlay changed source samples") }
        for row in 0..<2 {
            for col in 0..<2 {
                let offset = row * 16 + col * 4
                for channel in 0..<4 { graphics[offset + channel] = 255 }
            }
        }
        runBlend()
        guard luma == [235,235,100,100,235,235,100,100,100,100,100,100,100,100,100,100], chroma.allSatisfy({ $0 == 128 }) else {
            throw ProjectError.invalid("Score blend changed pixels outside its region")
        }
        luma = [UInt8](repeating: 100, count: 16)
        for index in graphics.indices { if graphics[index] == 255 { graphics[index] = 128 } }
        runBlend()
        guard luma[0] == 168, luma[2] == 100, chroma.allSatisfy({ $0 == 128 }) else { throw ProjectError.invalid("Score blend premultiplied-alpha self-check failed") }
        // A point timeline can begin on an odd column in the same chroma block
        // as the scoreboard. Its fade must never fade the score pixels.
        graphics = [UInt8](repeating: 255, count: graphics.count)
        luma = [UInt8](repeating: 100, count: 16)
        runBlend(pointStartX: 1, pointOpacity: 0.5)
        guard Array(luma.prefix(4)) == [235, 168, 168, 168], Array(luma.suffix(8)) == Array(repeating: 100, count: 8) else {
            throw ProjectError.invalid("Point timeline fade changed the scoreboard or pixels below the overlay")
        }
        luma = [UInt8](repeating: 100, count: 16)
        runBlend(pointStartX: 1, pointOpacity: 0)
        guard Array(luma.prefix(4)) == [235, 100, 100, 100] else { throw ProjectError.invalid("Hidden point timeline changed source pixels") }
    }
    #endif
}
