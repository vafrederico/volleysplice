// Standalone CPU/AVFoundation probe. Compile with Sources/VolleyCore/*.swift,
// App/VideoExportColor.swift and App/YUVScoreCompositor.swift using -D DEBUG.
// Usage: hdr-score-probe <empty-output-directory>
// Designed for a CPU-only lab; codec/compositor availability is checked explicitly.
// No UIKit, simulator or private recording is used. Each video is 3 frames.
@preconcurrency import AVFoundation
import CoreVideo
import Foundation
import VideoToolbox
import Darwin

@main struct HDRScoreProbe {
    static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw ProjectError.invalid(message) }
    }
    static func buffer(hdr: Bool) throws -> CVPixelBuffer {
        var result: CVPixelBuffer?
        try require(CVPixelBufferCreate(kCFAllocatorDefault, 64, 48,
            hdr ? kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            nil, &result) == kCVReturnSuccess, "Allocate YUV buffer")
        let frame = result!
        CVPixelBufferLockBaseAddress(frame, [])
        for plane in 0..<2 {
            let base = CVPixelBufferGetBaseAddressOfPlane(frame, plane)!
            let count = CVPixelBufferGetBytesPerRowOfPlane(frame, plane) * CVPixelBufferGetHeightOfPlane(frame, plane)
            if hdr { base.assumingMemoryBound(to: UInt16.self).initialize(repeating: UInt16(plane == 0 ? 256 : 512) << 6, count: count / 2) }
            else { memset(base, plane == 0 ? 64 : 128, count) }
        }
        CVPixelBufferUnlockBaseAddress(frame, [])
        return frame
    }

    static func createSource(_ url: URL, color: VideoExportColor) async throws {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        var settings: [String: Any] = [AVVideoCodecKey: color.isHDR ? AVVideoCodecType.hevc : AVVideoCodecType.h264,
            AVVideoWidthKey: 64, AVVideoHeightKey: 48,
            AVVideoColorPropertiesKey: [AVVideoColorPrimariesKey: color.primaries,
                AVVideoTransferFunctionKey: color.transfer, AVVideoYCbCrMatrixKey: color.matrix]]
        if color.isHDR {
            // These synthetic inputs are plain HLG/PQ. Avoid asking the fixture
            // writer to synthesize Dolby Vision metadata on a CPU-only host.
            settings[AVVideoCompressionPropertiesKey] = [AVVideoProfileLevelKey: kVTProfileLevel_HEVC_Main10_AutoLevel,
                kVTCompressionPropertyKey_HDRMetadataInsertionMode as String: kVTHDRMetadataInsertionMode_None]
        }
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: nil)
        writer.add(input)
        try require(writer.startWriting(), "Start sample writer: \(String(describing: writer.error))")
        writer.startSession(atSourceTime: .zero)
        for index in 0..<3 {
            let deadline = Date().addingTimeInterval(30)
            while !input.isReadyForMoreMediaData && writer.status == .writing && Date() < deadline {
                try await Task.sleep(for: .milliseconds(10))
            }
            try require(input.isReadyForMoreMediaData, "Sample writer did not become ready")
            let frame = try buffer(hdr: color.isHDR)
            for (key, value) in [(kCVImageBufferColorPrimariesKey, color.primaries),
                                 (kCVImageBufferTransferFunctionKey, color.transfer), (kCVImageBufferYCbCrMatrixKey, color.matrix)] {
                CVBufferSetAttachment(frame, key, value as CFString, .shouldPropagate)
            }
            try require(adaptor.append(frame, withPresentationTime: CMTime(value: Int64(index), timescale: 3)), "Append sample")
        }
        writer.endSession(atSourceTime: CMTime(value: 1, timescale: 1)); input.markAsFinished()
        let writerTimeout = Task { try await Task.sleep(for: .seconds(30)); writer.cancelWriting() }
        defer { writerTimeout.cancel() }
        await writer.finishWriting()
        try require(writer.status == .completed, "Finish source: \(String(describing: writer.error))")
    }

    static func checkDirectComposition(color: VideoExportColor) throws {
        let source = try buffer(hdr: color.isHDR), destination = try buffer(hdr: color.isHDR)
        CVBufferSetAttachment(source, kCVImageBufferTransferFunctionKey, color.transfer as CFString, .shouldPropagate)
        CVBufferSetAttachment(source, "ProbeInvalidatedMetadata" as CFString, "do not copy" as CFString, .shouldPropagate)
        let raster = ScoreBlendRaster(bgra: Data(repeating: 255, count: 2 * 2 * 4), width: 2, height: 2, scoreWidth: 2, color: color.space)
        try YUVScoreCompositor.compose(source: source, raster: raster, output: destination, transform: .identity, pointOpacity: 1)
        try require(CVBufferCopyAttachment(destination, kCVImageBufferTransferFunctionKey, nil) as? String == color.transfer, "Color tag not propagated")
        try require(CVBufferCopyAttachment(destination, "ProbeInvalidatedMetadata" as CFString, nil) == nil, "Non-color metadata propagated")
        CVPixelBufferLockBaseAddress(source, .readOnly); defer { CVPixelBufferUnlockBaseAddress(source, .readOnly) }
        CVPixelBufferLockBaseAddress(destination, .readOnly); defer { CVPixelBufferUnlockBaseAddress(destination, .readOnly) }
        let size = color.isHDR ? 2 : 1
        for plane in 0..<2 {
            let input = CVPixelBufferGetBaseAddressOfPlane(source, plane)!.assumingMemoryBound(to: UInt8.self)
            let output = CVPixelBufferGetBaseAddressOfPlane(destination, plane)!.assumingMemoryBound(to: UInt8.self)
            let inputStride = CVPixelBufferGetBytesPerRowOfPlane(source, plane), outputStride = CVPixelBufferGetBytesPerRowOfPlane(destination, plane)
            for row in 0..<CVPixelBufferGetHeightOfPlane(source, plane) {
                for column in 0..<(64 * size) where plane == 1 || row >= 2 || column >= 2 * size {
                    try require(input[row * inputStride + column] == output[row * outputStride + column], "Direct composition changed uncovered samples")
                }
            }
        }
        let y = CVPixelBufferGetBaseAddressOfPlane(destination, 0)!
        let sample = color.isHDR ? Double(y.load(as: UInt16.self) >> 6) / 4 : Double(y.load(as: UInt8.self))
        let expected = 16 + 219 * color.space.encode(SIMD3(repeating: color.isHDR ? 203 : 1)).x
        try require(abs(sample - expected) < 0.51, "Direct score white incorrect")
        print("PASS \(color.space.transfer.rawValue) direct CVPixelBuffer compositor/pixels/attachments")
    }

    static func export(_ source: URL, to output: URL, color: VideoExportColor, overlay: Bool) async throws {
        let asset = AVURLAsset(url: source)
        let track = try await asset.loadTracks(withMediaType: .video).first!
        let composition = AVMutableVideoComposition()
        composition.renderSize = CGSize(width: 64, height: 48)
        composition.frameDuration = CMTime(value: 1, timescale: 3)
        composition.colorPrimaries = color.primaries; composition.colorTransferFunction = color.transfer; composition.colorYCbCrMatrix = color.matrix
        let range = CMTimeRange(start: .zero, duration: CMTime(value: 1, timescale: 1))
        if overlay {
            composition.customVideoCompositorClass = color.isHDR ? HDRYUVScoreCompositor.self : YUVScoreCompositor.self
            let raster = ScoreBlendRaster(bgra: Data(repeating: 255, count: 16 * 16 * 4), width: 16, height: 16, scoreWidth: 16, color: color.space)
            composition.instructions = [YUVScoreInstruction(timeRange: range,
                videoTrackID: track.trackID, transform: .identity, raster: raster, sourceStartMs: 0, pointRevealTimestampMs: nil)]
        } else {
            // Exercise the production score-off built-in compositor separately;
            // its runtime may require graphics unavailable in the CPU-only lab.
            let instruction = AVMutableVideoCompositionInstruction()
            instruction.timeRange = range
            instruction.layerInstructions = [AVMutableVideoCompositionLayerInstruction(assetTrack: track)]
            composition.instructions = [instruction]
        }
        let session = AVAssetExportSession(asset: asset, presetName: color.preset)!
        session.outputURL = output; session.outputFileType = .mp4; session.videoComposition = composition
        let exportTimeout = Task { try await Task.sleep(for: .seconds(30)); session.cancelExport() }
        defer { exportTimeout.cancel() }
        await session.export()
        try require(session.status == .completed, "Export: \(String(describing: session.error))")
        let exported = AVURLAsset(url: output)
        let exportedTrack = try await exported.loadTracks(withMediaType: .video).first!
        let description = try await exportedTrack.load(.formatDescriptions).first!
        try color.verify(description)
        let reader = try AVAssetReader(asset: exported)
        let outputTrack = AVAssetReaderTrackOutput(track: exportedTrack, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String:
            color.isHDR ? kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange])
        reader.add(outputTrack); try require(reader.startReading(), "Read output")
        guard let sample = outputTrack.copyNextSampleBuffer(), let frame = CMSampleBufferGetImageBuffer(sample) else {
            throw ProjectError.invalid("Decode exported sample")
        }
        CVPixelBufferLockBaseAddress(frame, .readOnly); defer { CVPixelBufferUnlockBaseAddress(frame, .readOnly) }
        let base = CVPixelBufferGetBaseAddressOfPlane(frame, 0)!, stride = CVPixelBufferGetBytesPerRowOfPlane(frame, 0)
        func luma(_ x: Int, _ y: Int) -> Double {
            color.isHDR ? Double(base.load(fromByteOffset: y * stride + x * 2, as: UInt16.self) >> 6) / 4 : Double(base.load(fromByteOffset: y * stride + x, as: UInt8.self))
        }
        try require(abs(luma(40, 32) - 64) < 5, "Background luma changed")
        let expected = overlay ? 16 + 219 * color.space.encode(SIMD3(repeating: color.isHDR ? 203 : 1)).x : 64
        try require(abs(luma(8, 8) - expected) < 5, "Overlay diffuse-white luma incorrect")
        reader.cancelReading()
        print("PASS \(color.space.transfer.rawValue) overlay=\(overlay) color tags/codec/decoded pixels")
    }

    static func run() async throws {
        for invalid in [0, 1, 2] {
            do {
                _ = try VideoExportColor(primaries: invalid == 0 ? nil : invalid == 2 ? AVVideoColorPrimaries_P3_D65 : AVVideoColorPrimaries_ITU_R_709_2,
                    transfer: invalid == 2 ? AVVideoTransferFunction_ITU_R_2100_HLG : AVVideoTransferFunction_ITU_R_709_2,
                    matrix: AVVideoYCbCrMatrix_ITU_R_709_2, fullRange: invalid == 1)
            } catch { continue }
            throw ProjectError.invalid("Invalid color policy was accepted")
        }
        print("PASS color policy rejects missing tags/full range/non-BT.2020 HLG")
        try YUVScoreCompositor.checkKernels()
        let folder = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        var failures: [String] = []
        for transfer in [AVVideoTransferFunction_ITU_R_709_2, AVVideoTransferFunction_ITU_R_2100_HLG, AVVideoTransferFunction_SMPTE_ST_2084_PQ] {
            let hdr = transfer != AVVideoTransferFunction_ITU_R_709_2
            let color = try VideoExportColor(primaries: hdr ? AVVideoColorPrimaries_ITU_R_2020 : AVVideoColorPrimaries_ITU_R_709_2,
                transfer: transfer, matrix: hdr ? AVVideoYCbCrMatrix_ITU_R_2020 : AVVideoYCbCrMatrix_ITU_R_709_2)
            try require(color.preset == (hdr ? AVAssetExportPresetHEVCHighestQuality : AVAssetExportPresetHighestQuality), "Incorrect color export preset")
            try checkDirectComposition(color: color)
            let source = folder.appendingPathComponent("\(color.space.transfer.rawValue)-source.mp4")
            do { try await createSource(source, color: color) }
            catch { failures.append("\(color.space.transfer.rawValue) source: \(error)"); continue }
            for overlay in [true, false] {
                do { try await export(source, to: folder.appendingPathComponent("\(color.space.transfer.rawValue)-\(overlay).mp4"), color: color, overlay: overlay) }
                catch { failures.append("\(color.space.transfer.rawValue) overlay=\(overlay): \(error)") }
            }
        }
        for failure in failures { print("FAIL \(failure)") }
        try require(failures.isEmpty, "One or more bounded HDR sample checks failed")
    }

    static func main() async {
        do { try await run() }
        catch {
            FileHandle.standardError.write(Data("\(error)\n".utf8))
            exit(1) // Flush per-case stdout instead of Swift's fatal top-level trap.
        }
    }
}
