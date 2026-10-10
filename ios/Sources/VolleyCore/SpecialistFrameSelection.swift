import Foundation

/// Specialist sampling uses the first PTS + 1 ms at/after each target.
/// At completed EOS, only terminal requests may reuse the last actual image.
public struct SpecialistFrameSelection {
    public enum Failure: Error { case invalidTargets }
    public let targets: [Double]
    public let duration: Double
    public private(set) var index = 0
    public private(set) var lastPTS: Double?
    public private(set) var terminalPTS: Double?
    public var isComplete: Bool { index == targets.count }
    public var readStart: Double { max(0, (targets.first ?? 0) - 1) }

    public init(targets: [Double], duration: Double) throws {
        guard duration.isFinite, duration > 0,
              targets.allSatisfy({ $0.isFinite && $0 >= 0 && $0 < duration }),
              zip(targets, targets.dropFirst()).allSatisfy({ $0.0 < $0.1 }) else {
            throw Failure.invalidTargets
        }
        self.targets = targets; self.duration = duration
    }

    public mutating func accept(presentationTime pts: Double) -> Range<Int>? {
        guard pts.isFinite, !isComplete else { return nil }
        lastPTS = pts
        if pts >= duration - 1 { terminalPTS = pts }
        guard pts + 0.001 >= targets[index] else { return nil }
        let lower = index
        while index < targets.count && pts + 0.001 >= targets[index] { index += 1 }
        return lower..<index
    }

    public mutating func finish(completed: Bool) -> Range<Int>? {
        guard !isComplete, completed, let terminalPTS, let last = targets.last,
              targets[index] >= duration - 1, last - terminalPTS <= 1 else { return nil }
        let lower = index
        index = targets.count
        return lower..<index
    }
}
