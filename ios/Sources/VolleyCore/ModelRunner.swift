import Foundation

public struct Interval: Codable, Sendable, Equatable {
    public var start: Double
    public var end: Double
    public var confidence: Float
    public var agreement: String?
    public init(start: Double, end: Double, confidence: Float, agreement: String? = nil) {
        self.start = start; self.end = end; self.confidence = confidence; self.agreement = agreement
    }
}

public struct Serve: Codable, Sendable, Equatable {
    public var time: Double
    public var confidence: Float
    public init(time: Double, confidence: Float) { self.time = time; self.confidence = confidence }
}

/// Native port of Android ModelRunner. Keep scalar Float operations in their original order:
/// changing accumulation precision or using a fused dot product can move threshold crossings.
public struct ModelRunner: Sendable {
    public struct RunResult: Codable, Sendable {
        public let intervals: [Interval]
        public let rallyProbabilities: [Float]
        public let serveProbabilities: [Float]
        public let deadStateProbabilities: [Float]
        public let serveDetections: [Serve]
        public let profileMilliseconds: [String: Double]
    }

    private struct Head: Decodable, Sendable {
        let mean: [Float], scale: [Float], weights: [Float], bias: Float
    }
    private struct ProbabilityConfig: Decodable, Sendable {
        let smoothingSeconds: Double, enterThreshold: Double, exitThreshold: Double
        let minLiveSeconds: Double, bridgeGapSeconds: Double
        let shortEventMinSeconds: Double, shortEventThreshold: Double
        enum CodingKeys: String, CodingKey {
            case smoothingSeconds = "smoothing_seconds", enterThreshold = "enter_threshold"
            case exitThreshold = "exit_threshold", minLiveSeconds = "min_live_seconds"
            case bridgeGapSeconds = "bridge_gap_seconds", shortEventMinSeconds = "short_event_min_seconds"
            case shortEventThreshold = "short_event_threshold"
        }
        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            smoothingSeconds = try c.decode(Double.self, forKey: .smoothingSeconds)
            enterThreshold = try c.decode(Double.self, forKey: .enterThreshold)
            exitThreshold = try c.decode(Double.self, forKey: .exitThreshold)
            minLiveSeconds = try c.decode(Double.self, forKey: .minLiveSeconds)
            bridgeGapSeconds = try c.decode(Double.self, forKey: .bridgeGapSeconds)
            shortEventMinSeconds = try c.decodeIfPresent(Double.self, forKey: .shortEventMinSeconds) ?? minLiveSeconds
            shortEventThreshold = try c.decodeIfPresent(Double.self, forKey: .shortEventThreshold) ?? 1
        }
    }
    private struct ServeConfig: Decodable, Sendable {
        let threshold: Double, minSeparationSeconds: Double
        let timeOffsetSeconds: Double?
    }
    private struct CompositionConfig: Decodable, Sendable {
        let associationSeconds: Double, fallbackSeconds: Double, maxRescueSeconds: Double
        let permissiveDecoder: ProbabilityConfig
    }
    private struct DeadConfig: Decodable, Sendable {
        let deadThreshold: Double, liveResetThreshold: Double
        let minimumLiveSamples: Int?, minimumDeadSamples: Int?
        let minAfterServeSeconds: Double?, maxAfterServeSeconds: Double?, timeOffsetSeconds: Double?
    }
    private struct Refinement: Decodable, Sendable { let endWindowSeconds: Double? }
    private struct RallyBundle: Decodable, Sendable {
        let decoder: ProbabilityConfig
    }
    private struct ServeBundle: Decodable, Sendable {
        let decoder: ServeConfig, composition: CompositionConfig
    }
    private struct DeadBundle: Decodable, Sendable {
        let decoder: DeadConfig, refinement: Refinement
    }
    private struct Bundle: Decodable, Sendable {
        let schemaVersion: Int, analysisFps: Float, featureNames: [String]
        let rally: RallyBundle, serve: ServeBundle, deadState: DeadBundle
    }
    private struct Heads: Decodable { let rally: Head, serve: Head, deadState: Head }
    private let bundle: Bundle
    private let rally: Head, serve: Head, dead: Head
    public var featureNames: [String] { bundle.featureNames }

    public init(data: Data, expectedFeatureNames: [String] = FeatureSchema.contextualNames) throws {
        let decoder = JSONDecoder()
        let parsed = try decoder.decode(Bundle.self, from: data)
        let heads = try decoder.decode(Heads.self, from: data)
        guard parsed.schemaVersion == 1, !expectedFeatureNames.isEmpty,
              parsed.featureNames == expectedFeatureNames,
              parsed.analysisFps.isFinite, parsed.analysisFps > 0 else {
            throw AnalysisError.invalid("Unsupported model schema or feature signature")
        }
        for head in [heads.rally, heads.serve, heads.deadState] {
            guard head.mean.count == expectedFeatureNames.count, head.scale.count == expectedFeatureNames.count,
                  head.weights.count == expectedFeatureNames.count, head.mean.allSatisfy(\.isFinite),
                  head.weights.allSatisfy(\.isFinite), head.scale.allSatisfy({ $0.isFinite && $0 > 0 }),
                  head.bias.isFinite else { throw AnalysisError.invalid("Invalid model vector") }
        }
        for config in [parsed.rally.decoder, parsed.serve.composition.permissiveDecoder] {
            guard [config.smoothingSeconds, config.minLiveSeconds, config.bridgeGapSeconds,
                   config.shortEventMinSeconds].allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 86400 }),
                  [config.enterThreshold, config.exitThreshold, config.shortEventThreshold].allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 }) else {
                throw AnalysisError.invalid("Invalid probability decoder")
            }
        }
        let serving = parsed.serve.decoder, composition = parsed.serve.composition, deadConfig = parsed.deadState.decoder
        guard [serving.threshold, deadConfig.deadThreshold, deadConfig.liveResetThreshold].allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 }),
              [serving.minSeparationSeconds, composition.associationSeconds, composition.fallbackSeconds,
               composition.maxRescueSeconds, parsed.deadState.refinement.endWindowSeconds ?? 1,
               deadConfig.minAfterServeSeconds ?? 0.25, deadConfig.maxAfterServeSeconds ?? 4]
                .allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 86400 }),
              [serving.timeOffsetSeconds ?? 0, deadConfig.timeOffsetSeconds ?? 0].allSatisfy({ $0.isFinite && abs($0) <= 86400 }),
              (deadConfig.minimumLiveSamples ?? 1) > 0, (deadConfig.minimumDeadSamples ?? 2) > 0 else {
            throw AnalysisError.invalid("Invalid serve or dead-state decoder")
        }
        bundle = parsed; rally = heads.rally; serve = heads.serve; dead = heads.deadState
    }

    public func run(times: [Double], contextual: [Float], duration: Double) throws -> RunResult {
        guard duration.isFinite, duration >= 0 else { throw AnalysisError.invalid("Invalid video duration") }
        if times.isEmpty {
            guard contextual.isEmpty else { throw AnalysisError.invalid("Model matrix shape mismatch") }
            return RunResult(intervals: [], rallyProbabilities: [], serveProbabilities: [], deadStateProbabilities: [], serveDetections: [], profileMilliseconds: [:])
        }
        try FeatureMath.validate(times: times, values: contextual, columns: featureNames.count)
        guard times.last! <= duration else { throw AnalysisError.invalid("Sample exceeds video duration") }
        var profile: [String: Double] = [:]
        func measure<T>(_ name: String, _ operation: () -> T) -> T {
            let start = DispatchTime.now().uptimeNanoseconds
            let result = operation()
            profile[name] = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
            return result
        }
        let rallyP = measure("rally_head") { predict(rally, contextual, rows: times.count) }
        let serveP = measure("serve_head") { predict(serve, contextual, rows: times.count) }
        let deadP = measure("dead_state_head") { predict(dead, contextual, rows: times.count) }
        let fps = effectiveFps(times)
        let primary = measure("primary_rally_decode") { decode(times, rallyP, duration, bundle.rally.decoder, fps) }
        let permissive = measure("permissive_rally_decode") { decode(times, rallyP, duration, bundle.serve.composition.permissiveDecoder, fps) }
        let serves = measure("serve_decode") { decodeServes(times, serveP, duration) }
        let composed = measure("serve_composition") { compose(primary, permissive, serves, duration, 1 / fps) }
        let refined = measure("dead_state_refinement") { refineEnds(times, deadP, composed, duration, 1 / fps) }
        return RunResult(intervals: refined, rallyProbabilities: rallyP, serveProbabilities: serveP,
                         deadStateProbabilities: deadP, serveDetections: serves, profileMilliseconds: profile)
    }

    private func predict(_ head: Head, _ values: [Float], rows: Int) -> [Float] {
        let width = head.weights.count
        return (0..<rows).map { row in
            var logit = head.bias
            for column in 0..<width {
                let normalized = (values[row * width + column] - head.mean[column]) / head.scale[column]
                logit += normalized * head.weights[column]
            }
            return Float(1 / (1 + exp(-Double(max(-30, min(30, logit))))))
        }
    }

    private func effectiveFps(_ times: [Double]) -> Double {
        guard times.count > 1 else { return 1 }
        let differences = zip(times, times.dropFirst()).map { $1 - $0 }.sorted()
        let mid = differences.count / 2
        let median = differences.count % 2 == 1 ? differences[mid] : (differences[mid - 1] + differences[mid]) / 2
        return median.isFinite && median > 0 ? 1 / median : Double(bundle.analysisFps)
    }

    private func runs(_ mask: [Bool], _ expected: Bool) -> [Range<Int>] {
        var output: [Range<Int>] = [], start: Int?
        for i in mask.indices {
            if mask[i] == expected && start == nil { start = i }
            if mask[i] != expected, let lower = start { output.append(lower..<i); start = nil }
        }
        if let start { output.append(start..<mask.count) }
        return output
    }

    private func decode(_ times: [Double], _ probabilities: [Float], _ duration: Double,
                        _ config: ProbabilityConfig, _ fps: Double) -> [Interval] {
        // Python/Java use ties-to-even, unlike Swift's default rounding rule.
        func samples(_ seconds: Double) -> Int {
            let requested = (seconds * fps).rounded(.toNearestOrEven)
            return requested >= Double(Int.max) ? Int.max : Int(requested)
        }
        let window = min(probabilities.count, max(1, samples(config.smoothingSeconds)))
        var smoothed = probabilities
        if window > 1 {
            let scale: Float = 1 / Float(window)
            for output in probabilities.indices {
                var sum: Float = 0
                for kernel in 0..<window {
                    let source = max(0, min(probabilities.count - 1, output + kernel - window / 2))
                    sum += probabilities[source] * scale
                }
                smoothed[output] = sum
            }
        }
        var live = false
        var mask = smoothed.map { value in
            if !live && Double(value) >= config.enterThreshold { live = true }
            else if live && Double(value) < config.exitThreshold { live = false }
            return live
        }
        let bridge = max(0, samples(config.bridgeGapSeconds))
        if bridge > 0 {
            for run in runs(mask, false) where run.lowerBound > 0 && run.upperBound < mask.count && run.count <= bridge {
                for i in run { mask[i] = true }
            }
        }
        let minimum = max(1, samples(config.minLiveSeconds)), shortMinimum = max(1, samples(config.shortEventMinSeconds))
        if minimum > 1 {
            for run in runs(mask, true) {
                let keepShort = run.count >= shortMinimum && Double(smoothed[run].max()!) >= config.shortEventThreshold
                if run.count < minimum && !keepShort { for i in run { mask[i] = false } }
            }
        }
        return runs(mask, true).compactMap { run in
            let start = max(0, times[run.lowerBound] - 0.5 / fps)
            let end = min(duration, times[run.upperBound - 1] + 0.5 / fps)
            guard end > start else { return nil }
            var sum: Float = 0
            for i in run { sum += smoothed[i] }
            return Interval(start: start, end: end, confidence: sum / Float(run.count))
        }
    }

    private func decodeServes(_ times: [Double], _ probabilities: [Float], _ duration: Double) -> [Serve] {
        let config = bundle.serve.decoder
        let mask = probabilities.map { Double($0) >= config.threshold }
        var candidates = runs(mask, true).map { run in
            var peak = run.lowerBound
            for i in run where probabilities[i] > probabilities[peak] { peak = i }
            return Serve(time: min(duration, max(0, times[peak] + (config.timeOffsetSeconds ?? 0))), confidence: probabilities[peak])
        }
        candidates.sort { $0.confidence == $1.confidence ? $0.time < $1.time : $0.confidence > $1.confidence }
        var retained: [Serve] = []
        for candidate in candidates where retained.allSatisfy({ abs(candidate.time - $0.time) >= config.minSeparationSeconds }) {
            retained.append(candidate)
        }
        return retained.sorted { $0.time < $1.time }
    }

    private func compose(_ primary: [Interval], _ permissive: [Interval], _ serves: [Serve],
                         _ duration: Double, _ sampleSeconds: Double) -> [Interval] {
        let config = bundle.serve.composition
        var rows = primary, unused: [Serve] = []
        for serve in serves {
            let associated = rows.indices.filter { rows[$0].start - config.associationSeconds <= serve.time && serve.time < rows[$0].end }
            guard var selected = associated.first else { unused.append(serve); continue }
            for candidate in associated where abs(rows[candidate].start - serve.time) < abs(rows[selected].start - serve.time) { selected = candidate }
            if serve.time < rows[selected].start {
                rows[selected].start = serve.time
                rows[selected].confidence = max(rows[selected].confidence, serve.confidence)
            }
        }
        for serve in unused {
            var selected: Interval?, distance = Double.infinity
            for interval in permissive where interval.end >= serve.time - config.associationSeconds
                && interval.start <= serve.time + config.associationSeconds && interval.end - interval.start <= config.maxRescueSeconds {
                let candidateDistance = min(abs(interval.start - serve.time), abs(interval.end - serve.time))
                if candidateDistance < distance { selected = interval; distance = candidateDistance }
            }
            if let selected {
                rows.append(Interval(start: serve.time, end: max(serve.time + sampleSeconds, selected.end), confidence: max(serve.confidence, selected.confidence)))
            } else if config.fallbackSeconds > 0 {
                rows.append(Interval(start: serve.time, end: serve.time + config.fallbackSeconds, confidence: serve.confidence))
            }
        }
        rows = rows.filter { $0.end > $0.start && $0.start < duration && $0.end > 0 }.map {
            Interval(start: max(0, $0.start), end: min(duration, $0.end), confidence: $0.confidence)
        }
        rows.sort {
            if $0.start != $1.start { return $0.start < $1.start }
            if $0.end != $1.end { return $0.end < $1.end }
            return $0.confidence < $1.confidence
        }
        var merged: [Interval] = []
        for row in rows {
            if let previous = merged.last, row.start < previous.end {
                merged[merged.count - 1].end = max(previous.end, row.end)
                merged[merged.count - 1].confidence = max(previous.confidence, row.confidence)
            } else { merged.append(row) }
        }
        return merged
    }

    private func refineEnds(_ times: [Double], _ probabilities: [Float], _ intervals: [Interval],
                            _ duration: Double, _ sampleSeconds: Double) -> [Interval] {
        let window = bundle.deadState.refinement.endWindowSeconds ?? 1
        return intervals.indices.map { i in
            let interval = intervals[i], nextStart: Double? = i + 1 < intervals.count ? intervals[i + 1].start : nil
            let anchor = max(0, interval.end - window)
            guard let transition = deadTransition(times, probabilities, anchor: anchor,
                                                 upper: interval.end + window,
                                                 next: nextStart.flatMap { $0 > anchor ? $0 : nil }, duration: duration) else { return interval }
            let transitionTime = min(duration, min(interval.end + window, max(interval.end - window, transition.time)))
            let refinedEnd = min(duration, min(nextStart ?? duration, transitionTime))
            if refinedEnd < interval.start + sampleSeconds - 1e-9 { return interval }
            return Interval(start: interval.start, end: refinedEnd, confidence: max(interval.confidence, transition.confidence))
        }
    }

    private func deadTransition(_ times: [Double], _ probabilities: [Float], anchor: Double,
                                upper: Double, next: Double?, duration: Double) -> Serve? {
        let config = bundle.deadState.decoder
        var liveRun = 0, armed = false, deadStart: Int?, deadLength = 0
        for i in times.indices {
            let time = times[i], probability = probabilities[i]
            if time < anchor { continue }
            if time > upper || (next != nil && time >= next!) { break }
            if !armed {
                if Double(probability) <= config.liveResetThreshold {
                    liveRun += 1
                    if liveRun >= (config.minimumLiveSamples ?? 1) { armed = true }
                } else { liveRun = 0 }
            }
            if !armed { deadStart = nil; deadLength = 0; continue }
            if Double(probability) >= config.deadThreshold {
                if deadStart == nil { deadStart = i }
                deadLength += 1
            } else { deadStart = nil; deadLength = 0 }
            if let deadStart, deadLength >= (config.minimumDeadSamples ?? 2) {
                let confidence = probabilities[deadStart...i].min()!
                return Serve(time: min(duration, max(0, times[deadStart] + (config.timeOffsetSeconds ?? 0))), confidence: confidence)
            }
        }
        return nil
    }
}
