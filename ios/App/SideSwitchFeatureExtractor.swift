import Foundation

/// SIDE-SWITCH-UNION34-V1's 22 visual columns. Court remap/alignment/color conversion
/// are OpenCV primitives; palettes, player proposals and statistical reductions are Swift.
enum SideSwitchFeatureExtractor {
    static let width = 256, height = 144
    private static let size = width * height
    struct CourtGeometry: Codable, Equatable { var netYRatio: Double, confidence: Double }
    private struct Prepared { let gray: [[UInt8]], hsv: [[UInt8]], median: [Double], motion: [Double], maximumShift: Double, minimumResponse: Double }
    private struct Palette { let values: [Double], instability: Double }
    private struct Broad { let near: Palette, far: Palette, tightNear: Palette, tightFar: Palette, global: Palette, maximumShift: Double, minimumResponse: Double }
    private struct Player { let near: [Double], far: [Double], global: [Double], instability: Double, coverage: Double, count: Double, nearSupport: Double, farSupport: Double }
    private struct Box { let x: Int, y: Int, width: Int, height: Int, area: Int }
    private struct Weighted { let palette: [Double], weight: Double }

    static func visualFeatures(plan: SideSwitchInference.FramePlan, duration: Double, bgrByTime: [Double: [UInt8]],
                               progress: (Double, String) -> Void = { _, _ in }) throws -> [Double] {
        if plan.candidates.isEmpty { return [] }
        func frame(_ time: Double) throws -> [UInt8] {
            let clamped = min(max(time, 0), max(duration - 0.01, 0))
            guard let value = bgrByTime[clamped], value.count == size * 3 else { throw AnalysisError.invalid("Missing 256x144 side-switch BGR frame at \(clamped)") }
            return value
        }
        let geometry = try estimateCourtGeometry(plan.calibrationTimes.map(frame))
        var result: [Double] = []
        for (i, candidate) in plan.candidates.enumerated() {
            try Task.checkCancellation()
            let times = SideSwitchInference.candidateSampleTimes(candidate)
            let before = try times.prefix(7).map(frame), after = try times.suffix(7).map(frame)
            result += try extract(beforeFrames: before, afterFrames: after, geometry: geometry)
            progress(Double(i + 1) / Double(plan.candidates.count), "Comparing team sides · \(i + 1)/\(plan.candidates.count)")
        }
        return result
    }
    static func evaluate(input: SideSwitchAnalysisInput, plan: SideSwitchInference.FramePlan, duration: Double,
                          bgrByTime: [Double: [UInt8]], runner: SideSwitchInference,
                          progress: (Double, String) -> Void = { _, _ in }) throws -> SideSwitchOutput {
        let visual = try visualFeatures(plan: plan, duration: duration, bgrByTime: bgrByTime, progress: progress)
        return try runner.evaluate(input: input, plan: plan, visualFeatures: visual)
    }
    static func estimateCourtGeometry(_ frames: [[UInt8]]) throws -> CourtGeometry {
        var candidates: [(Double, Double)] = []
        for frame in frames {
            try Task.checkCancellation()
            guard frame.count == size * 3 else { throw AnalysisError.invalid("Invalid side-switch calibration frame") }
            var pair = [Double](repeating: 0, count: 2)
            let status = vc_switch_net_line(frame, &pair)
            if status < 0 { throw AnalysisError.invalid("Side-switch Hough calibration failed") }
            if status > 0 { candidates.append((pair[0], pair[1])) }
        }
        if candidates.isEmpty { return CourtGeometry(netYRatio: 0.5, confidence: 0) }
        let positions = candidates.map { $0.0 }
        var center = quantile(positions, 0.5)
        let initialCenter = center, deviations = positions.map { abs($0 - initialCenter) }
        let cutoff = max(0.035, quantile(deviations, 0.5) * 2.5)
        var weighted = 0.0, weight = 0.0
        for i in candidates.indices where deviations[i] <= cutoff { weighted += candidates[i].0 * candidates[i].1; weight += candidates[i].1 }
        if weight > 0 { center = weighted / weight }
        let median = quantile(positions, 0.5), dispersion = quantile(positions.map { abs($0 - median) }, 0.5)
        return CourtGeometry(netYRatio: min(max(center, 0.25), 0.68), confidence: min(max(Double(candidates.count) / Double(frames.count) * max(0, 1 - dispersion / 0.1), 0), 1))
    }
    static func extract(beforeFrames: [[UInt8]], afterFrames: [[UInt8]], geometry: CourtGeometry) throws -> [Double] {
        let before = try prepare(beforeFrames, geometry), after = try prepare(afterFrames, geometry)
        let bb = broad(before), ab = broad(after), bp = players(before), ap = players(after)
        let broadSame = assignment(bb.near.values, bb.far.values, ab.near.values, ab.far.values)
        let tightSame = assignment(bb.tightNear.values, bb.tightFar.values, ab.tightNear.values, ab.tightFar.values)
        let same = (hellinger(bp.near, ap.near) + hellinger(bp.far, ap.far)) / 2
        let swapped = (hellinger(bp.near, ap.far) + hellinger(bp.far, ap.near)) / 2
        let beforeSeparation = hellinger(bp.near, bp.far), afterSeparation = hellinger(ap.near, ap.far)
        let result: [Double] = [broadSame.0, tightSame.0, (broadSame.1 + tightSame.1) / 2, hellinger(bb.global.values, ab.global.values),
            max(bb.maximumShift, ab.maximumShift), min(bb.minimumResponse, ab.minimumResponse), same, swapped, same - swapped,
            -cosine(subtract(bp.near, bp.far), subtract(ap.near, ap.far)), min(beforeSeparation, afterSeparation), abs(beforeSeparation - afterSeparation),
            bp.instability, ap.instability, hellinger(bp.global, ap.global), min(bp.coverage, ap.coverage), abs(bp.coverage - ap.coverage),
            min(bp.count, ap.count), abs(bp.count - ap.count), min(bp.nearSupport, ap.nearSupport), min(bp.farSupport, ap.farSupport),
            abs(abs(bp.nearSupport - bp.farSupport) - abs(ap.nearSupport - ap.farSupport))]
        guard result.count == 22, result.allSatisfy(\.isFinite) else { throw AnalysisError.invalid("Side-switch visual feature contract failed") }
        return result
    }
    private static func prepare(_ frames: [[UInt8]], _ geometry: CourtGeometry) throws -> Prepared {
        guard frames.count == 7, frames.allSatisfy({ $0.count == size * 3 }), geometry.netYRatio.isFinite else {
            throw AnalysisError.invalid("Side-switch requires seven 256x144 BGR frames per window")
        }
        let packed = frames.flatMap { $0 }
        var gray = [UInt8](repeating: 0, count: 7 * size), hsv = [UInt8](repeating: 0, count: 7 * size * 3), quality = [0.0, 0.0]
        guard vc_switch_prepare(packed, geometry.netYRatio, &gray, &hsv, &quality) == 0 else { throw AnalysisError.invalid("Side-switch OpenCV preparation failed") }
        var median = [Double](repeating: 0, count: size), motion = median
        for pixel in 0..<size {
            let values = (0..<7).map { Double(gray[$0 * size + pixel]) }.sorted()
            median[pixel] = values[3]; motion[pixel] = (values[6] - values[0]) / 255
        }
        let grays = (0..<7).map { Array(gray[($0 * size)..<(($0 + 1) * size)]) }
        let hsvs = (0..<7).map { Array(hsv[($0 * size * 3)..<(($0 + 1) * size * 3)]) }
        return Prepared(gray: grays, hsv: hsvs, median: median, motion: motion, maximumShift: quality[0], minimumResponse: quality[1])
    }
    private static func broad(_ prepared: Prepared) -> Broad {
        Broad(near: pooledRegion(prepared, 0.44, 0.98), far: pooledRegion(prepared, 0.15, 0.6),
              tightNear: pooledRegion(prepared, 0.56, 0.91), tightFar: pooledRegion(prepared, 0.27, 0.53),
              global: pooledRegion(prepared, 0.08, 0.98), maximumShift: prepared.maximumShift, minimumResponse: prepared.minimumResponse)
    }
    private static func rounded(_ value: Float) -> Int { Int(floor(Double(value) + 0.5)) }
    private static func pooledRegion(_ prepared: Prepared, _ topRatio: Double, _ bottomRatio: Double) -> Palette {
        let top = rounded(Float(height) * Float(topRatio)), bottom = rounded(Float(height) * Float(bottomRatio))
        let left = rounded(Float(width) * 0.06), right = rounded(Float(width) * 0.94)
        var palettes: [[Double]] = []
        for frame in prepared.gray.indices {
            var weights = [Double](repeating: 0, count: size)
            for pixel in 0..<size {
                weights[pixel] = max(abs(Double(prepared.gray[frame][pixel]) - prepared.median[pixel]) / 255 - 2 / 255.0, 0)
                    + 0.2 * max(prepared.motion[pixel] - 5 / 255.0, 0)
            }
            palettes.append(weightedPalette(prepared.hsv[frame], weights, left: left, top: top, right: right, bottom: bottom, uniformWhenEmpty: false))
        }
        let palette = normalizedAverage(palettes)
        return Palette(values: palette, instability: quantile(palettes.map { hellinger($0, palette) }, 0.5))
    }
    private static func weightedPalette(_ hsv: [UInt8], _ weights: [Double], left: Int, top: Int, right: Int, bottom: Int, uniformWhenEmpty: Bool) -> [Double] {
        var bins = [Double](repeating: 0, count: 52), total = 0.0
        for y in top..<bottom { for x in left..<right {
            let pixel = y * width + x, weight = weights[pixel]
            if weight <= 0 { continue }
            let offset = pixel * 3, hue = min(11, Int(hsv[offset]) / 15), saturation = min(3, Int(hsv[offset + 1]) / 64), value = min(3, Int(hsv[offset + 2]) / 64)
            bins[hue * 4 + saturation] += weight; bins[48 + value] += weight; total += weight * 2
        } }
        if total < 1e-8 {
            if uniformWhenEmpty { return [Double](repeating: 1 / 52.0, count: 52) }
            return weightedPalette(hsv, [Double](repeating: 1, count: weights.count), left: left, top: top, right: right, bottom: bottom, uniformWhenEmpty: true)
        }
        return bins.map { $0 / total }
    }
    private static func normalizedAverage(_ palettes: [[Double]], weights: [Double]? = nil) -> [Double] {
        guard let first = palettes.first else { return [Double](repeating: 1 / 52.0, count: 52) }
        var output = [Double](repeating: 0, count: first.count), weightTotal = 0.0
        for row in palettes.indices {
            let weight = weights?[row] ?? 1; weightTotal += weight
            for i in output.indices { output[i] += palettes[row][i] * weight }
        }
        var total = 0.0
        for i in output.indices { output[i] /= max(weightTotal, 1e-12); total += output[i] }
        return output.map { $0 / max(total, 1e-12) }
    }
    private static func pooledWeighted(_ values: [Weighted]) -> Palette {
        if values.isEmpty { return Palette(values: [Double](repeating: 1 / 52.0, count: 52), instability: 0) }
        let palette = normalizedAverage(values.map(\.palette), weights: values.map(\.weight))
        return Palette(values: palette, instability: quantile(values.map { hellinger($0.palette, palette) }, 0.5))
    }
    private static func players(_ prepared: Prepared) -> Player {
        var near: [Weighted] = [], far: [Weighted] = [], global: [Weighted] = []
        var coverages: [Double] = [], counts: [Double] = [], nearSupport = 0.0, farSupport = 0.0
        for frame in prepared.gray.indices {
            let gray = prepared.gray[frame], hsv = prepared.hsv[frame]
            let difference = (0..<size).map { abs(Double(gray[$0]) - prepared.median[$0]) / 255 }
            let boxes = proposalBoxes(difference)
            var covered = [UInt8](repeating: 0, count: size)
            counts.append(Double(boxes.count))
            for box in boxes {
                var weights = [Double](repeating: 0, count: size), support = 0.0
                for y in box.y..<(box.y + box.height) { for x in box.x..<(box.x + box.width) {
                    let p = y * width + x; covered[p] = 1
                    let saturation = Double(hsv[p * 3 + 1]) / 255
                    weights[p] = difference[p] + 0.15 * prepared.motion[p] + 0.01 * saturation; support += weights[p]
                } }
                support = max(support, 1e-8)
                let palette = weightedPalette(hsv, weights, left: box.x, top: box.y, right: box.x + box.width, bottom: box.y + box.height, uniformWhenEmpty: true)
                let footY = Double(box.y + box.height) / Double(height), nearProbability = 1 / (1 + exp(-(footY - 0.63) / 0.06)), farProbability = 1 - nearProbability
                if nearProbability >= 0.08 { let weight = support * nearProbability; near.append(.init(palette: palette, weight: weight)); nearSupport += weight }
                if farProbability >= 0.08 { let weight = support * farProbability; far.append(.init(palette: palette, weight: weight)); farSupport += weight }
                global.append(.init(palette: palette, weight: support))
            }
            coverages.append(Double(covered.reduce(0) { $0 + Int($1) }) / Double(size))
        }
        let np = pooledWeighted(near), fp = pooledWeighted(far), gp = pooledWeighted(global), total = max(nearSupport + farSupport, 1e-12)
        return Player(near: np.values, far: fp.values, global: gp.values, instability: (np.instability + fp.instability) / 2,
                      coverage: mean(coverages), count: mean(counts), nearSupport: nearSupport / total, farSupport: farSupport / total)
    }
    private static func proposalBoxes(_ difference: [Double]) -> [Box] {
        let numeric = difference.map { min(max(floor($0 * 255), 0), 255) }, threshold = max(12, floor(quantile(numeric, 0.94)))
        let mask: [UInt8] = numeric.map { $0 >= threshold ? 1 : 0 }
        var candidates: [(Box, Double, Int)] = []
        for box in components(closeMask(mask)) {
            let bottom = box.y + box.height, aspect = Double(box.height) / Double(max(box.width, 1))
            if box.area < 10 || Double(box.area) > Double(size) * 0.035 || box.width < 2 || box.height < 4 ||
                Double(box.width) > Double(width) * 0.18 || Double(box.height) > Double(height) * 0.48 || aspect < 0.45 || aspect > 5.5 ||
                Double(bottom) < Double(height) * 0.35 || Double(box.y) > Double(height) * 0.95 { continue }
            let px = max(1, rounded(Float(box.width) * 0.18)), py = max(1, rounded(Float(box.height) * 0.1))
            let x = max(0, box.x - px), y = max(0, box.y - py), right = min(width, box.x + box.width + px), lower = min(height, bottom + py)
            let expanded = Box(x: x, y: y, width: right - x, height: lower - y, area: box.area)
            let score = Double(box.area) * pow(0.5 + Double(lower) / Double(height), 2) * min(aspect, 2.5)
            candidates.append((expanded, score, candidates.count))
        }
        candidates.sort { $0.1 == $1.1 ? $0.2 < $1.2 : $0.1 > $1.1 }
        var selected: [Box] = []
        for candidate in candidates {
            if selected.allSatisfy({ iou(candidate.0, $0) < 0.35 }) { selected.append(candidate.0) }
            if selected.count == 6 { break }
        }
        return selected
    }
    private static func closeMask(_ mask: [UInt8]) -> [UInt8] {
        let offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        var dilated = [UInt8](repeating: 0, count: size), eroded = dilated
        for y in 0..<height { for x in 0..<width {
            for (dx, dy) in offsets {
                let nx = x + dx, ny = y + dy
                if nx >= 0 && nx < width && ny >= 0 && ny < height && mask[ny * width + nx] > 0 { dilated[y * width + x] = 1; break }
            }
        } }
        for y in 0..<height { for x in 0..<width {
            var all = true
            for (dx, dy) in offsets {
                let nx = x + dx, ny = y + dy
                if nx < 0 || nx >= width || ny < 0 || ny >= height || dilated[ny * width + nx] == 0 { all = false; break }
            }
            eroded[y * width + x] = all ? 1 : 0
        } }
        return eroded
    }
    private static func components(_ mask: [UInt8]) -> [Box] {
        var seen = [Bool](repeating: false, count: size), queue = [Int](repeating: 0, count: size), result: [Box] = []
        for origin in 0..<size {
            if mask[origin] == 0 || seen[origin] { continue }
            var head = 0, tail = 1, minX = width, maxX = 0, minY = height, maxY = 0
            queue[0] = origin; seen[origin] = true
            while head < tail {
                let pixel = queue[head]; head += 1
                let y = pixel / width, x = pixel % width
                minX = min(minX, x); maxX = max(maxX, x); minY = min(minY, y); maxY = max(maxY, y)
                for dy in -1...1 { for dx in -1...1 {
                    if dx == 0 && dy == 0 { continue }
                    let nx = x + dx, ny = y + dy
                    if nx < 0 || nx >= width || ny < 0 || ny >= height { continue }
                    let next = ny * width + nx
                    if mask[next] != 0 && !seen[next] { seen[next] = true; queue[tail] = next; tail += 1 }
                } }
            }
            result.append(.init(x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1, area: tail))
        }
        return result
    }
    private static func iou(_ a: Box, _ b: Box) -> Double {
        let w = max(0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x)), h = max(0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y))
        let intersection = w * h, union = a.width * a.height + b.width * b.height - intersection
        return union == 0 ? 0 : Double(intersection) / Double(union)
    }
    private static func assignment(_ bn: [Double], _ bf: [Double], _ an: [Double], _ af: [Double]) -> (Double, Double) {
        let same = (hellinger(bn, an) + hellinger(bf, af)) / 2, swapped = (hellinger(bn, af) + hellinger(bf, an)) / 2
        return (same, same - swapped)
    }
    private static func subtract(_ a: [Double], _ b: [Double]) -> [Double] { a.indices.map { a[$0] - b[$0] } }
    private static func hellinger(_ a: [Double], _ b: [Double]) -> Double {
        var total = 0.0
        for i in a.indices { total += pow(sqrt(max(a[i], 0)) - sqrt(max(b[i], 0)), 2) }
        return sqrt(total) / sqrt(2)
    }
    private static func cosine(_ a: [Double], _ b: [Double]) -> Double {
        var dot = 0.0, an = 0.0, bn = 0.0
        for i in a.indices { dot += a[i] * b[i]; an += a[i] * a[i]; bn += b[i] * b[i] }
        let denominator = sqrt(an * bn); return denominator > 1e-12 ? dot / denominator : 0
    }
    private static func quantile(_ values: [Double], _ p: Double) -> Double {
        if values.isEmpty { return 0 }
        let sorted = values.sorted(), position = Double(values.count - 1) * min(max(p, 0), 1)
        let lower = Int(floor(position)), upper = Int(ceil(position))
        if lower == upper { return sorted[lower] }
        return sorted[lower] * (Double(upper) - position) + sorted[upper] * (position - Double(lower))
    }
    private static func mean(_ values: [Double]) -> Double { values.isEmpty ? 0 : values.reduce(0, +) / Double(values.count) }
    static func validateStaticFixture() throws {
        let frame = [UInt8](repeating: 73, count: size * 3), frames = [[UInt8]](repeating: frame, count: 7)
        let geometry = try estimateCourtGeometry(frames)
        guard geometry.netYRatio == 0.5, geometry.confidence == 0 else { throw AnalysisError.invalid("Static court fallback mismatch") }
        let features = try extract(beforeFrames: frames, afterFrames: frames, geometry: geometry)
        // Alignment response is an OpenCV result, not a fabricated constant. All appearance/proposal differences are zero.
        guard features.enumerated().allSatisfy({ $0.offset == 4 || $0.offset == 5 || abs($0.element) < 1e-12 }) else {
            throw AnalysisError.invalid("Static side-switch visual fixture mismatch")
        }
    }
}
