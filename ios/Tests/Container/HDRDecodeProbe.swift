// CPU-only native decode diagnostic: supply a local movie and optional seek time.
// Reads three frames per mode, without UI, model inference, or full-video export.
// Compile with App/AnalysisVideoColor.swift. Inputs and receipts stay outside Git.
import AVFoundation
import Foundation

@main struct HDRDecodeProbe {
    static func main() async {
        guard CommandLine.arguments.count >= 2 else { return }
        let source = AVURLAsset(url: URL(fileURLWithPath: CommandLine.arguments[1]))
        let start = CommandLine.arguments.count > 2 ? Double(CommandLine.arguments[2]) ?? 0 : 0
        do {
            guard let track = try await source.loadTracks(withMediaType: .video).first else { return }
            let settings = try await AnalysisVideoColor.outputSettings(for: track)
            print("Analysis settings: \(settings)")
            for (name, outputSettings) in [
                ("legacy-nv12", [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange] as [String: Any]),
                ("analysis-sdr709", settings),
                ("preserve-p010", [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange] as [String: Any])
            ] {
                let reader = try AVAssetReader(asset: source)
                let output = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
                output.alwaysCopiesSampleData = false
                reader.add(output)
                let duration = try await source.load(.duration)
                reader.timeRange = CMTimeRange(start: CMTime(seconds: start, preferredTimescale: 60000), end: duration)
                let started = reader.startReading()
                var frames = 0
                while started && frames < 3, let sample = output.copyNextSampleBuffer() {
                    guard let buffer = CMSampleBufferGetImageBuffer(sample) else { break }
                    if frames < 3 {
                        let transfer = CVBufferCopyAttachment(buffer, kCVImageBufferTransferFunctionKey, nil)
                        let matrix = CVBufferCopyAttachment(buffer, kCVImageBufferYCbCrMatrixKey, nil)
                        CVPixelBufferLockBaseAddress(buffer, .readOnly)
                        let base = CVPixelBufferGetBaseAddressOfPlane(buffer, 0)!
                        let stride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
                        let tenBit = CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
                        var minimum = 1023, maximum = 0
                        for y in 0..<CVPixelBufferGetHeightOfPlane(buffer, 0) {
                            for x in 0..<CVPixelBufferGetWidthOfPlane(buffer, 0) {
                                let value = tenBit ? Int(base.load(fromByteOffset: y * stride + x * 2, as: UInt16.self) >> 6) : Int(base.load(fromByteOffset: y * stride + x, as: UInt8.self))
                                minimum = min(minimum, value); maximum = max(maximum, value)
                            }
                        }
                        CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
                        print("\(name) pts=\(CMSampleBufferGetPresentationTimeStamp(sample).seconds) luma=\(minimum)...\(maximum) format=\(CVPixelBufferGetPixelFormatType(buffer)) transfer=\(String(describing: transfer)) matrix=\(String(describing: matrix))")
                    }
                    frames += 1
                }
                print("\(frames == 3 ? "PASS" : "FAIL") \(name) seek=\(start) frames=\(frames) error=\(String(describing: reader.error))")
                reader.cancelReading()
            }
        } catch { print("FAIL native decode setup: \(error)") }
    }
}
