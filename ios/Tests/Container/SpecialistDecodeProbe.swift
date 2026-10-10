// Compile with AnalysisVideoColor.swift, SpecialistFrameReader.swift and
// Sources/VolleyCore/SpecialistFrameSelection.swift. Reads short seek windows,
// never runs model inference or exports a full video. Supply a local movie.
import AVFoundation
import Foundation

@main struct SpecialistDecodeProbe {
    static func main() async throws {
        guard CommandLine.arguments.count == 2 else { return }
        let asset = AVURLAsset(url: URL(fileURLWithPath: CommandLine.arguments[1]))
        let duration = try await asset.load(.duration).seconds
        guard duration > 0.1, let track = try await asset.loadTracks(withMediaType: .video).first else { return }
        let settings = try await AnalysisVideoColor.outputSettings(for: track)
        print("Source duration=\(duration)")
        let cases: [(String, [Double])] = [
            ("start", [0]),
            ("middle", [min(10, duration / 2)]),
            ("isolated-terminal", [duration - 0.01]),
            ("shared-terminal", [duration - 0.04, duration - 0.01])
        ]
        for (name, targets) in cases {
            var decoded: [Double: Double] = [:]
            try SpecialistFrameReader.read(asset: asset, track: track, targets: targets,
                                           duration: duration, outputSettings: settings) { buffer, pts, indices in
                guard CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange else {
                    throw SpecialistFrameReader.Incomplete(description: "Unexpected specialist pixel format")
                }
                for index in indices { decoded[targets[index]] = pts }
                print("\(name) pts=\(pts) targets=\(targets[indices]) transfer=\(String(describing: CVBufferCopyAttachment(buffer, kCVImageBufferTransferFunctionKey, nil)))")
            }
            guard decoded.count == targets.count,
                  decoded.allSatisfy({ $0.value + 1 >= $0.key }) else {
                throw SpecialistFrameReader.Incomplete(description: "Missing or stale specialist sample")
            }
            print("PASS \(name) count=\(decoded.count)")
        }
    }
}
