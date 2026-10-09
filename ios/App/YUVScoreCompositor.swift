@preconcurrency import AVFoundation
import Accelerate
import Foundation

/// Source NV12/P010 samples never take a full-frame RGB round trip. Immutable CPU
/// score graphics touch only the top strip, with point opacity in source time.
final class YUVScoreInstruction: NSObject, AVVideoCompositionInstructionProtocol, @unchecked Sendable {
    let timeRange: CMTimeRange
    let enablePostProcessing = false
    let containsTweening = true
    let passthroughTrackID = kCMPersistentTrackID_Invalid
    let requiredSourceTrackIDs: [NSValue]?
    let videoTrackID: CMPersistentTrackID
    let transform: CGAffineTransform
    let raster: ScoreBlendRaster
    let sourceStartMs: Int64
    let pointRevealTimestampMs: Int64?
    let fadeScoreOverlay: Bool

    init(timeRange: CMTimeRange, videoTrackID: CMPersistentTrackID, transform: CGAffineTransform,
         raster: ScoreBlendRaster, sourceStartMs: Int64, pointRevealTimestampMs: Int64?,
         fadeScoreOverlay: Bool = false) {
        self.timeRange = timeRange; self.videoTrackID = videoTrackID
        self.transform = transform; self.raster = raster
        self.sourceStartMs = sourceStartMs; self.pointRevealTimestampMs = pointRevealTimestampMs
        self.fadeScoreOverlay = fadeScoreOverlay
        requiredSourceTrackIDs = [NSNumber(value: videoTrackID)]
        super.init()
    }
}

class YUVScoreCompositor: NSObject, AVVideoCompositing, @unchecked Sendable {
    // Separate compositor classes make source and destination bit depth agree.
    var pixelFormat: OSType { kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange }
    var sourcePixelBufferAttributes: [String: Any]? {
        [kCVPixelBufferPixelFormatTypeKey as String: pixelFormat]
    }
    var requiredPixelBufferAttributesForRenderContext: [String: Any] {
        [kCVPixelBufferPixelFormatTypeKey as String: pixelFormat]
    }
    var canConformColorOfSourceFrames: Bool { false }
    var supportsWideColorSourceFrames: Bool { true }
    var supportsHDRSourceFrames: Bool { true }
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
                                 transform: instruction.transform, pointOpacity: opacity,
                                 scoreOpacity: instruction.fadeScoreOverlay ? opacity : 1)
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

    static func compose(source: CVPixelBuffer, raster: ScoreBlendRaster, output: CVPixelBuffer,
                        transform: CGAffineTransform, pointOpacity: Float, scoreOpacity: Float = 1) throws {
        let format = CVPixelBufferGetPixelFormatType(source)
        let tenBit = format == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
        guard (tenBit || format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange),
              CVPixelBufferGetPixelFormatType(output) == format,
              CVPixelBufferGetPlaneCount(source) == 2, CVPixelBufferGetPlaneCount(output) == 2,
              accepts(transform: transform) else {
            throw ProjectError.invalid("Score overlays require 8/10-bit video-range YUV and a right-angle source orientation")
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
                bytesPerPixel: (plane == 0 ? 1 : 2) * (tenBit ? 2 : 1), transform: transform,
                fillSample: plane == 0 ? (tenBit ? 64 << 6 : 16) : (tenBit ? 512 << 6 : 128), tenBit: tenBit)
        }
        // Dolby Vision per-frame metadata is invalid after editing. Propagate
        // color/chroma interpretation only; AVFoundation may generate new metadata.
        for key in [kCVImageBufferColorPrimariesKey, kCVImageBufferTransferFunctionKey,
                    kCVImageBufferYCbCrMatrixKey, kCVImageBufferChromaLocationTopFieldKey,
                    kCVImageBufferChromaLocationBottomFieldKey] {
            if let value = CVBufferCopyAttachment(source, key, nil) {
                CVBufferSetAttachment(output, key, value, .shouldPropagate)
            }
        }
        guard let y = CVPixelBufferGetBaseAddressOfPlane(output, 0),
              let uv = CVPixelBufferGetBaseAddressOfPlane(output, 1) else {
            throw ProjectError.invalid("Missing score blend pixels")
        }
        raster.blend(y: y, yStride: CVPixelBufferGetBytesPerRowOfPlane(output, 0),
                     uv: uv, uvStride: CVPixelBufferGetBytesPerRowOfPlane(output, 1),
                     frameWidth: CVPixelBufferGetWidth(output), frameHeight: CVPixelBufferGetHeight(output),
                     tenBit: tenBit, pointOpacity: Double(pointOpacity), scoreOpacity: Double(scoreOpacity))
    }

    private static func copyPlane(input: UnsafeMutableRawPointer, output: UnsafeMutableRawPointer,
                                  width: Int, height: Int, outputWidth: Int, outputHeight: Int,
                                  inputStride: Int, outputStride: Int, bytesPerPixel: Int, transform: CGAffineTransform, fillSample: Int = 16, tenBit: Bool = false) throws {
        let a = Int(transform.a.rounded()), b = Int(transform.b.rounded())
        let c = Int(transform.c.rounded()), d = Int(transform.d.rounded())
        let rotatedWidth = a == 0 ? height : width, rotatedHeight = a == 0 ? width : height
        guard rotatedWidth <= outputWidth, rotatedHeight <= outputHeight else { throw ProjectError.invalid("Score compositor output geometry mismatch") }
        // Only optional one-pixel codec padding lies outside the rotated source.
        if tenBit {
            output.assumingMemoryBound(to: UInt16.self).initialize(repeating: UInt16(fillSample), count: outputStride * outputHeight / 2)
        } else { memset(output, Int32(fillSample), outputStride * outputHeight) }
        if a == 1 && d == 1 {
            for row in 0..<height { memcpy(output.advanced(by: row * outputStride), input.advanced(by: row * inputStride), width * bytesPerPixel) }
            return
        }
        if bytesPerPixel <= 2 && a * d - b * c == 1 && rotatedWidth == outputWidth && rotatedHeight == outputHeight {
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
                for byte in 0..<bytesPerPixel { dst[to + byte] = src[from + byte] }
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
        for bytes in [1, 2, 4] {
            let inputStride = 4 * bytes + 4
            var input = [UInt8](repeating: 99, count: inputStride * 2)
            for index in 0..<8 {
                let offset = (index / 4) * inputStride + (index % 4) * bytes
                input[offset] = UInt8(index)
                for byte in 1..<bytes { input[offset + byte] = UInt8(255 - index - byte) }
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
                    guard output[offset] == UInt8(expected[index]), (1..<bytes).allSatisfy({ output[offset + $0] == UInt8(255 - expected[index] - $0) }) else {
                        throw ProjectError.invalid("YUV rotation kernel self-check failed")
                    }
                }
            }
        }
    }
    #endif
}

final class HDRYUVScoreCompositor: YUVScoreCompositor, @unchecked Sendable {
    override var pixelFormat: OSType { kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange }
}
