#if DEBUG
import Foundation

/// Runs the tiny, checked-in Android instrumentation videos on the iPad.
/// Android's independent assertions cover display dimensions, not source pixels.
/// Consequently this is a decode/ROI smoke check, not a pixel-parity golden test.
enum DeviceMediaCheck {
    static func run() async -> String {
        let fixtures: [(String, Int, Int)] = [
            ("overlay-fixture", 320, 240),
            ("overlay-fixture-90", 240, 320),
            ("overlay-fixture-180", 320, 240),
            ("overlay-fixture-270", 240, 320)
        ]
        var lines = ["Android rotation fixtures — device decode check"]
        var passed = 0
        var observedRotations = Set<Int>()
        for (name, expectedWidth, expectedHeight) in fixtures {
            do {
                try Task.checkCancellation()
                guard let url = Bundle.main.url(forResource: name, withExtension: "mp4") else {
                    throw AnalysisError.invalid("Missing bundled fixture \(name).mp4")
                }
                let media = try await MediaDecoder.describe(url)
                let quarterTurn = media.rotation == 90 || media.rotation == 270
                let width = quarterTurn ? media.height : media.width
                let height = quarterTurn ? media.width : media.height
                // ScoreOverlayInstrumentedTest independently asserts these output sizes.
                guard width == expectedWidth, height == expectedHeight else {
                    throw AnalysisError.invalid("Display dimensions \(width)×\(height), expected \(expectedWidth)×\(expectedHeight)")
                }
                observedRotations.insert(media.rotation)
                for roi in [AnalysisRegion(x: 0, y: 0, width: 1, height: 1), AnalysisRegion()] {
                    let result = try await MediaDecoder.video(
                        url: url, media: media, roi: roi, start: 0, end: min(1, media.duration),
                        progress: { _, _ in }, checkpoint: { _, _, _ in })
                    guard result.times == [0, 0.25, 0.5, 0.75],
                          result.decodedTimes.count == 4,
                          result.visual.count == 4 * 73,
                          result.visual.allSatisfy(\.isFinite) else {
                        throw AnalysisError.invalid("Expected four finite 73-channel feature rows at 4 Hz")
                    }
                    guard zip(result.times, result.decodedTimes).allSatisfy({ pair in
                        pair.1.isFinite && pair.1 + 0.001 >= pair.0
                    }) else {
                        throw AnalysisError.invalid("Decoded frame precedes its Android sampling target")
                    }
                }
                passed += 1
                lines.append("PASS \(name): \(media.rotation)°, \(width)×\(height), full frame + court ROI decoded")
            } catch is CancellationError {
                lines.append("Cancelled")
                return lines.joined(separator: "\n")
            } catch {
                lines.append("FAIL \(name): \(error.localizedDescription)")
            }
        }
        let allRotations = observedRotations == Set([0, 90, 180, 270])
        lines.append("\(passed)/4 fixtures passed; all four display rotations observed: \(allRotations ? "yes" : "NO")")
        lines.append("Source pixel/ROI parity remains unverified: Android has no independent raw-pixel expectations for these clips. Its overlay-color checks apply after export.")
        return lines.joined(separator: "\n")
    }
}
#endif
