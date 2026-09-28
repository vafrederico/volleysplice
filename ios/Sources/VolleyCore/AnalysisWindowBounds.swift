import Foundation

/// Queue and project boundaries use integer milliseconds. Keep their representable
/// end at or before the actual media duration, including older rounded-up jobs.
public enum AnalysisWindowBounds {
    public static func maximumEndMilliseconds(duration: Double) throws -> Int64 {
        guard duration.isFinite, duration > 0, duration < Double(Int64.max / 1000) else {
            throw AnalysisError.invalid("Invalid source duration")
        }
        let rounded = Int64((duration * 1000).rounded())
        // Compare in seconds instead of flooring a product such as10.001*1000,
        // which can lie just below an integer due to binary representation.
        return Double(rounded) / 1000 > duration ? rounded - 1 : rounded
    }

    public static func normalize(start: Double, end: Double, duration: Double) throws -> TimeRange {
        let maximum = try maximumEndMilliseconds(duration: duration)
        guard start.isFinite, end.isFinite, start >= 0, end > start, end <= duration + 0.001,
              end < Double(Int64.max / 1000) else { throw AnalysisError.invalid("Invalid game window") }
        let a = Int64((start * 1000).rounded()), b = min(maximum, Int64((end * 1000).rounded()))
        guard b > a else { throw AnalysisError.invalid("Game window is shorter than one millisecond") }
        return TimeRange(startMs: a, endMs: b)
    }
}
