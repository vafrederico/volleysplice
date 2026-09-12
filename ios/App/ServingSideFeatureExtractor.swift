import Foundation

/// Android SERVSIDE237-FLIGHT: 82 court-flow and 155 fixed-flight raw Float64 values.
/// Only grayscale, optical flow and morphology use OpenCV. Quantization points,
/// affine fitting, flood-fill label order and all reductions mirror Android Java.
enum ServingSideFeatureExtractor {
    static let width = 192, height = 108
    private static let size = width * height
    private struct Components { var labels: [Int], areas: [Int], centroidX: [Double], centroidY: [Double] }
    private struct Motion { let energy: [Double], flowX: [Double], flowY: [Double], divergence: [Double] }
    private struct PairSummary { let grid: [Double], rows: [Double], statistics: [Double] }

    static func grayscale(_ rgba: [UInt8]) throws -> [UInt8] {
        guard rgba.count == size * 4 else { throw AnalysisError.invalid("Serving-side RGBA frame must be 192x108") }
        var gray = [UInt8](repeating: 0, count: size)
        guard vc_serving_gray(rgba, &gray) == 0 else { throw AnalysisError.invalid("Serving-side grayscale conversion failed") }
        return gray
    }
    static func extract(courtRGBA: [[UInt8]], flightRGBA: [[UInt8]]) throws -> [Double] {
        try extract(courtGray: courtRGBA.map(grayscale), flightGray: flightRGBA.map(grayscale))
    }
    static func extract(candidate: ServingSideInference.CandidateInterval, duration: Double,
                         rgbaByTime: [Double: [UInt8]]) throws -> [Double] {
        var grayByTime: [Double: [UInt8]] = [:]
        for offset in ServingSideInference.courtFlowOffsets + ServingSideInference.flightOffsets {
            let time = ServingSideInference.clampedTimestamp(anchor: candidate.start, offset: offset, duration: duration)
            if grayByTime[time] == nil {
                guard let frame = rgbaByTime[time] else { throw AnalysisError.invalid("Missing serving-side target frame at \(time)") }
                grayByTime[time] = try grayscale(frame)
            }
        }
        return try extract(candidate: candidate, duration: duration, grayByTime: grayByTime)
    }
    static func extract(candidate: ServingSideInference.CandidateInterval, duration: Double,
                         grayByTime: [Double: [UInt8]]) throws -> [Double] {
        func frames(_ offsets: [Double]) throws -> [[UInt8]] {
            try offsets.map { offset in
                let time = ServingSideInference.clampedTimestamp(anchor: candidate.start, offset: offset, duration: duration)
                guard let frame = grayByTime[time] else { throw AnalysisError.invalid("Missing serving-side target frame at \(time)") }
                return frame
            }
        }
        return try extract(courtGray: frames(ServingSideInference.courtFlowOffsets), flightGray: frames(ServingSideInference.flightOffsets))
    }
    static func extract(courtGray: [[UInt8]], flightGray: [[UInt8]]) throws -> [Double] {
        let court = try courtFlow(courtGray), flight = try flight(flightGray)
        return try finite(court + flight, expected: 237)
    }
    static func courtFlow(_ frames: [[UInt8]]) throws -> [Double] {
        guard frames.count == 8 else { throw AnalysisError.invalid("Court-flow requires eight ordered grayscale frames") }
        let phasePairs = [[(0, 1), (1, 2)], [(2, 3), (3, 4), (4, 5)], [(5, 6), (6, 7)]]
        let band = max(1, Int((Float(height) * 0.32).rounded()))
        var phases: [[[Double]]] = []
        for pairs in phasePairs {
            var near: [[Double]] = [], far: [[Double]] = []
            for (before, after) in pairs {
                try Task.checkCancellation()
                let flow = try flow(frames[before], frames[after], flight: false)
                let xs = (0..<size).map { flow[$0 * 2] }, ys = (0..<size).map { flow[$0 * 2 + 1] }
                let medianX = Double(Float(quantile(xs, 0.5))), medianY = Double(Float(quantile(ys, 0.5)))
                var residual = [Double](repeating: 0, count: size * 2), magnitude = [Double](repeating: 0, count: size)
                var active = [UInt8](repeating: 0, count: size), opened = active
                for i in 0..<size {
                    let x = Float(flow[i * 2] - medianX), y = Float(flow[i * 2 + 1] - medianY)
                    residual[i * 2] = Double(x); residual[i * 2 + 1] = Double(y)
                    magnitude[i] = Double(Float(hypot(Double(x), Double(y))))
                    active[i] = magnitude[i] >= 1 ? 1 : 0
                }
                guard vc_serving_open(active, &opened) == 0 else { throw AnalysisError.invalid("Serving-side morphology failed") }
                near.append(courtZone(residual, magnitude, opened, y0: height - band, y1: height))
                far.append(courtZone(residual, magnitude, opened, y0: 0, y1: band))
            }
            phases.append([meanVectors(near), meanVectors(far)])
        }
        var result: [Double] = []
        for phase in 0..<3 {
            result += phases[phase][0]; result += phases[phase][1]
            for i in 0..<5 { result.append(phases[phase][0][i] - phases[phase][1][i]) }
        }
        for zone in 0..<2 {
            let pre = phases[0][zone], contact = phases[1][zone], post = phases[2][zone]
            for i in [0, 2, 3] { result.append(contact[i] - pre[i]) }
            for i in [0, 2] { result.append(post[i] - contact[i]) }
        }
        for i in [0, 2, 3] { result.append((phases[1][0][i] - phases[0][0][i]) - (phases[1][1][i] - phases[0][1][i])) }
        return try finite(result, expected: 82)
    }
    static func flight(_ frames: [[UInt8]]) throws -> [Double] {
        guard frames.count == 9 else { throw AnalysisError.invalid("Flight requires nine ordered grayscale frames") }
        var pairs: [PairSummary] = []
        for i in 0..<8 {
            try Task.checkCancellation()
            pairs.append(summarizeFlight(try flightMotion(frames[i], frames[i + 1])))
        }
        var phases: [PairSummary] = [], result: [Double] = []
        for indices in [[0, 1, 2], [3, 4, 5], [6, 7]] {
            let phase = PairSummary(grid: meanVectors(indices.map { pairs[$0].grid }), rows: meanVectors(indices.map { pairs[$0].rows }),
                                    statistics: meanVectors(indices.map { pairs[$0].statistics }))
            phases.append(phase); result += phase.grid; result += phase.rows; result += phase.statistics
        }
        for (before, after) in [(0, 1), (1, 2)] {
            for i in [3, 5, 6, 9, 10, 11, 12, 13, 14] { result.append(phases[after].statistics[i] - phases[before].statistics[i]) }
            for row in 0..<4 {
                var old = 0.0, new = 0.0
                for column in 0..<6 { old += phases[before].grid[row * 6 + column]; new += phases[after].grid[row * 6 + column] }
                result.append(new - old)
            }
        }
        return try finite(result, expected: 155)
    }
    private static func flow(_ before: [UInt8], _ after: [UInt8], flight: Bool) throws -> [Double] {
        guard before.count == size, after.count == size else { throw AnalysisError.invalid("Serving-side gray frame must be 192x108") }
        var output = [Float](repeating: 0, count: size * 2)
        guard vc_serving_flow(before, after, flight ? 1 : 0, &output) == 0 else { throw AnalysisError.invalid("Serving-side Farneback failed") }
        return output.map(Double.init)
    }
    private static func courtZone(_ residual: [Double], _ magnitude: [Double], _ opened: [UInt8], y0: Int, y1: Int) -> [Double] {
        let localHeight = y1 - y0, count = width * localHeight
        var localActive = [UInt8](repeating: 0, count: count), localMagnitude = [Double](repeating: 0, count: count)
        var activePixels = 0, fx = 0.0, fy = 0.0
        for y in y0..<y1 { for x in 0..<width {
            let source = y * width + x, target = (y - y0) * width + x
            localMagnitude[target] = magnitude[source]
            if opened[source] != 0 {
                localActive[target] = 1; activePixels += 1; fx += residual[source * 2]; fy += residual[source * 2 + 1]
            }
        } }
        let components = components(localActive, width: width, height: localHeight)
        var largest = 0, cx = 0.0, cy = 0.0
        for i in components.areas.indices where components.areas[i] > largest {
            largest = components.areas[i]; cx = components.centroidX[i] / Double(max(width - 1, 1))
            cy = components.centroidY[i] / Double(max(localHeight - 1, 1))
        }
        let diagonal = hypot(Double(localHeight), Double(width)), n = Double(count)
        return [localMagnitude.reduce(0, +) / n / diagonal, quantile(localMagnitude, 0.9) / diagonal,
                Double(activePixels) / n, Double(largest) / n, Double(components.areas.count) / max(n / 1000, 1), cx, cy,
                activePixels == 0 ? 0 : fx / Double(activePixels) / Double(width),
                activePixels == 0 ? 0 : fy / Double(activePixels) / Double(localHeight)]
    }
    private static func flightMotion(_ before: [UInt8], _ after: [UInt8]) throws -> Motion {
        let flow = try flow(before, after, flight: true)
        var coefficients = affineCoefficients(flow)
        let stride = max(1, min(width, height) / 24)
        var sampleResiduals: [Double] = []
        for y in Swift.stride(from: 0, to: height, by: stride) { for x in Swift.stride(from: 0, to: width, by: stride) {
            let dx = Double(x) / Double(width - 1), dy = Double(y) / Double(height - 1), offset = (y * width + x) * 2
            let predictedX = dx * coefficients[0][0] + dy * coefficients[0][1] + coefficients[0][2]
            let predictedY = dx * coefficients[1][0] + dy * coefficients[1][1] + coefficients[1][2]
            sampleResiduals.append(hypot(flow[offset] - predictedX, flow[offset + 1] - predictedY))
        } }
        let cutoff = quantile(sampleResiduals, 0.75), retained = sampleResiduals.map { $0 <= cutoff }
        if retained.filter({ $0 }).count >= 6 { coefficients = affineCoefficients(flow, retained: retained) }
        var fx = [Double](repeating: 0, count: size), fy = fx, magnitude = fx
        let diagonal = hypot(Double(height), Double(width))
        for y in 0..<height { for x in 0..<width {
            let i = y * width + x, dx = Double(x) / Double(width - 1), dy = Double(y) / Double(height - 1)
            let px = Float(dx * coefficients[0][0] + dy * coefficients[0][1] + coefficients[0][2])
            let py = Float(dx * coefficients[1][0] + dy * coefficients[1][1] + coefficients[1][2])
            let rx = Float(flow[i * 2] - Double(px)), ry = Float(flow[i * 2 + 1] - Double(py))
            fx[i] = Double(rx / Float(width)); fy[i] = Double(ry / Float(height))
            magnitude[i] = Double(Float(hypot(Double(rx), Double(ry)) / diagonal))
        } }
        let threshold = max(7.5e-4, quantile(magnitude, 0.9))
        let energy = magnitude.map { Double(Float(max($0 - threshold, 0))) }
        var divergence = [Double](repeating: 0, count: size)
        for y in 0..<height { for x in 0..<width {
            let i = y * width + x
            let dx = x == 0 ? fx[i + 1] - fx[i] : x == width - 1 ? fx[i] - fx[i - 1] : (fx[i + 1] - fx[i - 1]) / 2
            let dy = y == 0 ? fy[i + width] - fy[i] : y == height - 1 ? fy[i] - fy[i - width] : (fy[i + width] - fy[i - width]) / 2
            divergence[i] = Double(Float(dx * Double(width) + dy * Double(height)))
        } }
        return Motion(energy: energy, flowX: fx, flowY: fy, divergence: divergence)
    }
    private static func summarizeFlight(_ motion: Motion) -> PairSummary {
        let total = motion.energy.reduce(0, +)
        var cx = 0.5, cy = 0.5, sx = 0.0, sy = 0.0, entropy = 0.0
        if total > 0 {
            cx = 0; cy = 0
            for y in 0..<height { for x in 0..<width {
                let e = motion.energy[y * width + x]
                cx += Double(x) / Double(width - 1) * e; cy += Double(y) / Double(height - 1) * e
            } }
            cx /= total; cy /= total
            var vx = 0.0, vy = 0.0
            for y in 0..<height { for x in 0..<width {
                let e = motion.energy[y * width + x]
                vx += pow(Double(x) / Double(width - 1) - cx, 2) * e
                vy += pow(Double(y) / Double(height - 1) - cy, 2) * e
                if e > 0 { let p = e / total; entropy -= p * log(p) }
            } }
            sx = sqrt(max(0, vx / total)); sy = sqrt(max(0, vy / total)); entropy /= max(log(Double(size)), 1)
        }
        var grid = [Double](repeating: 0, count: 24), rows = [Double](repeating: 0, count: 4)
        for row in 0..<4 {
            let y0 = row * height / 4, y1 = (row + 1) * height / 4
            var rowTotal = 0.0, rowFlow = 0.0
            for column in 0..<6 {
                let x0 = column * width / 6, x1 = (column + 1) * width / 6
                var cellTotal = 0.0
                for y in y0..<y1 { for x in x0..<x1 { cellTotal += motion.energy[y * width + x] } }
                grid[row * 6 + column] = total > 0 ? cellTotal / total : 0; rowTotal += cellTotal
            }
            if rowTotal > 0 {
                for y in y0..<y1 { for x in 0..<width { let i = y * width + x; rowFlow += motion.flowY[i] * motion.energy[i] } }
                rows[row] = rowFlow / rowTotal
            }
        }
        let active: [UInt8] = motion.energy.map { $0 > 0 ? 1 : 0 }
        let components = components(active, width: width, height: height)
        let largest = components.areas.max() ?? 0, smallThreshold = max(4, Double(size) * 0.0025)
        let smallLabels = Set(components.areas.indices.filter { Double(components.areas[$0]) <= smallThreshold })
        var activeCount = 0, bottom = 0.0, top = 0.0, smallTotal = 0.0, smallY = 0.0, smallFlowY = 0.0
        for y in 0..<height { for x in 0..<width {
            let i = y * width + x, e = motion.energy[i]
            if e > 0 { activeCount += 1 }
            if y >= height / 2 { bottom += e } else { top += e }
            if e > 0 && smallLabels.contains(components.labels[i]) {
                smallTotal += e; smallY += Double(y) / Double(height - 1) * e; smallFlowY += motion.flowY[i] * e
            }
        } }
        let statistics = [total / Double(size), Double(activeCount) / Double(size), cx, cy, sx, sy, entropy, Double(largest) / Double(size),
            weightedMean(motion.flowX, motion.energy, total), weightedMean(motion.flowY, motion.energy, total),
            weightedMean(motion.divergence, motion.energy, total), total > 0 ? (bottom - top) / total : 0,
            total > 0 ? smallTotal / total : 0, smallTotal > 0 ? smallY / smallTotal : 0, smallTotal > 0 ? smallFlowY / smallTotal : 0]
        return PairSummary(grid: grid, rows: rows, statistics: statistics)
    }
    private static func affineCoefficients(_ flow: [Double], retained: [Bool]? = nil) -> [[Double]] {
        let stride = max(1, min(width, height) / 24)
        var matrix = [Double](repeating: 0, count: 9), targetX = [Double](repeating: 0, count: 3), targetY = targetX
        var sampleIndex = 0
        for y in Swift.stride(from: 0, to: height, by: stride) { for x in Swift.stride(from: 0, to: width, by: stride) {
            let include = retained == nil || retained![sampleIndex]; sampleIndex += 1
            if !include { continue }
            let design = [Double(x) / Double(width - 1), Double(y) / Double(height - 1), 1], offset = (y * width + x) * 2
            for row in 0..<3 {
                targetX[row] += design[row] * flow[offset]; targetY[row] += design[row] * flow[offset + 1]
                for column in 0..<3 { matrix[row * 3 + column] += design[row] * design[column] }
            }
        } }
        return [solve3(matrix, targetX), solve3(matrix, targetY)]
    }
    private static func solve3(_ matrix: [Double], _ vector: [Double]) -> [Double] {
        var work = [[matrix[0], matrix[1], matrix[2], vector[0]], [matrix[3], matrix[4], matrix[5], vector[1]], [matrix[6], matrix[7], matrix[8], vector[2]]]
        for column in 0..<3 {
            var pivot = column
            for row in (column + 1)..<3 where abs(work[row][column]) > abs(work[pivot][column]) { pivot = row }
            work.swapAt(column, pivot)
            if abs(work[column][column]) < 1e-12 { return [0, 0, 0] }
            for row in (column + 1)..<3 {
                let factor = work[row][column] / work[column][column]
                for i in column..<4 { work[row][i] -= factor * work[column][i] }
            }
        }
        var result = [Double](repeating: 0, count: 3)
        for row in stride(from: 2, through: 0, by: -1) {
            var value = work[row][3]
            for column in (row + 1)..<3 { value -= work[row][column] * result[column] }
            result[row] = value / work[row][row]
        }
        return result
    }
    private static func components(_ active: [UInt8], width: Int, height: Int) -> Components {
        var labels = [Int](repeating: -1, count: active.count), queue = [Int](repeating: 0, count: active.count)
        var areas: [Int] = [], cx: [Double] = [], cy: [Double] = []
        for origin in active.indices {
            if active[origin] == 0 || labels[origin] != -1 { continue }
            let label = areas.count
            var head = 0, tail = 1, sumX = 0, sumY = 0
            queue[0] = origin; labels[origin] = label
            while head < tail {
                let i = queue[head]; head += 1
                let y = i / width, x = i % width
                sumX += x; sumY += y
                for dy in -1...1 {
                    let nextY = y + dy; if nextY < 0 || nextY >= height { continue }
                    for dx in -1...1 {
                        if dx == 0 && dy == 0 { continue }
                        let nextX = x + dx; if nextX < 0 || nextX >= width { continue }
                        let next = nextY * width + nextX
                        if active[next] != 0 && labels[next] == -1 { labels[next] = label; queue[tail] = next; tail += 1 }
                    }
                }
            }
            areas.append(tail); cx.append(Double(sumX) / Double(tail)); cy.append(Double(sumY) / Double(tail))
        }
        return Components(labels: labels, areas: areas, centroidX: cx, centroidY: cy)
    }
    private static func weightedMean(_ values: [Double], _ weights: [Double], _ total: Double) -> Double {
        if total <= 0 { return 0 }
        var sum = 0.0; for i in values.indices { sum += values[i] * weights[i] }; return sum / total
    }
    private static func quantile(_ values: [Double], _ probability: Double) -> Double {
        if values.isEmpty { return 0 }
        let sorted = values.sorted(), position = Double(values.count - 1) * max(0, min(1, probability))
        let lower = Int(floor(position)), upper = Int(ceil(position)), fraction = position - Double(lower)
        return sorted[lower] * (1 - fraction) + sorted[upper] * fraction
    }
    private static func meanVectors(_ values: [[Double]]) -> [Double] {
        guard let first = values.first else { return [] }
        var result = [Double](repeating: 0, count: first.count)
        for value in values { for i in result.indices { result[i] += value[i] } }
        for i in result.indices { result[i] /= Double(values.count) }
        return result
    }
    private static func finite(_ values: [Double], expected: Int) throws -> [Double] {
        guard values.count == expected, values.allSatisfy(\.isFinite) else { throw AnalysisError.invalid("Serving-side feature signature or finite-value check failed") }
        return values
    }
    /// Same frozen static-gray fixture as Android's instrumented visual test, suitable for the iPad diagnostics action.
    static func validateStaticGrayFixture() throws {
        let frame = [UInt8](repeating: 73, count: size)
        let values = try extract(courtGray: [[UInt8]](repeating: frame, count: 8), flightGray: [[UInt8]](repeating: frame, count: 9))
        var expected = [Double](repeating: 0, count: 237)
        for phase in 0..<3 { expected[82 + phase * 43 + 30] = 0.5; expected[82 + phase * 43 + 31] = 0.5 }
        guard zip(values, expected).allSatisfy({ abs($0 - $1) <= 1e-12 }) else {
            throw AnalysisError.invalid("Serving-side static-gray fixture differs from Android")
        }
    }
}
