import Foundation

/// Shared analysis sampling contract: source-grid timestamps, exclusive end,
/// nearest presented source frame, with earlier timestamps winning exact ties.
public enum VideoFrameSelection {
    public static func targets(start: Double, end: Double, fps: Int) throws -> [Double] {
        guard start.isFinite, end.isFinite, start >= 0, end > start, fps > 0,
              end * Double(fps) < Double(Int.max) else {
            throw AnalysisError.invalid("Invalid video sampling window")
        }
        let first = Int(ceil(start * Double(fps) - 1e-9))
        let last = Int(ceil(end * Double(fps) - 1e-9))
        return (first..<last).map { Double($0) / Double(fps) }
    }

    public static func prefersEarlier(target: Double, earlier: Double, later: Double) -> Bool {
        // Integer microsecond PTS are the Android decoder's time base. Comparing
        // distances on that base also avoids floating-point half-tie drift.
        let earlierUS = (earlier * 1_000_000).rounded()
        let laterUS = (later * 1_000_000).rounded()
        let targetUS = target * 1_000_000
        return abs(targetUS - earlierUS) <= abs(laterUS - targetUS)
    }

    public static func select(presentationTimes: [Double], targets: [Double], end: Double) throws -> [Double] {
        guard end.isFinite, presentationTimes.allSatisfy(\.isFinite), targets.allSatisfy(\.isFinite),
              zip(presentationTimes, presentationTimes.dropFirst()).allSatisfy({ $0.0 <= $0.1 }),
              zip(targets, targets.dropFirst()).allSatisfy({ $0.0 <= $0.1 }) else {
            throw AnalysisError.invalid("Video timestamps must be finite and sorted")
        }
        let source = presentationTimes.filter { $0 < end }
        guard !source.isEmpty else { return [] }
        var right = 0
        return targets.prefix(while: { $0 < end }).map { target in
            while right < source.count && source[right] < target { right += 1 }
            let before = source[max(0, right - 1)], after = source[min(right, source.count - 1)]
            return prefersEarlier(target: target, earlier: before, later: after) ? before : after
        }
    }
}
