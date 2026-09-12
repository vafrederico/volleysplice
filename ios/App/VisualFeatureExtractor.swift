import Foundation

/// Android's 73-channel schema and reductions, backed by the same OpenCV 4.12 primitives.
final class VisualFeatureExtractor {
    private var previous: [UInt8]?
    private let w = 192, h = 108
    private var diagonal: Double { hypot(Double(w), Double(h)) }

    func extract(_ rgba: [UInt8]) throws -> [Float] {
        let n = w * h
        guard rgba.count == n * 4 else { throw AnalysisError.invalid("Expected 192×108 RGBA") }
        var gray = [UInt8](repeating: 0, count: n), hsv = [UInt8](repeating: 0, count: n * 3)
        var edges = [UInt8](repeating: 0, count: n)
        var laplacian = [Float](repeating: 0, count: n), gx = laplacian, gy = laplacian
        var flow = [Float](repeating: 0, count: n * 2), shift = [Double](repeating: 0, count: 3)
        let status: Int32
        if let previous {
            status = previous.withUnsafeBufferPointer { p in
                vc_filters(rgba, p.baseAddress, Int32(w), Int32(h), &gray, &hsv, &edges, &laplacian, &gx, &gy, &flow, &shift)
            }
        } else {
            status = vc_filters(rgba, nil, Int32(w), Int32(h), &gray, &hsv, &edges, &laplacian, &gx, &gy, &flow, &shift)
        }
        guard status == 0 else { throw AnalysisError.invalid("OpenCV visual extraction failed") }
        let pixels = gray.map(Float.init), saturation = (0..<n).map { Float(hsv[$0 * 3 + 1]) }
        let lumaMean = mean(pixels), lumaStd = std(pixels), lapVariance = pow(std(laplacian), 2)
        var values: [Float] = []
        func add(_ numbers: Double...) { values += numbers.map(Float.init) }
        func scaled(_ numbers: [Float], _ scale: Double) { values += numbers.map { Float(Double($0) * scale) } }
        add(lumaMean / 255, lumaStd / 255, mean(saturation) / 255, std(saturation) / 255,
            Double(edges.filter { $0 > 0 }.count) / Double(n), min(lapVariance / 2000, 5))
        scaled(grid(pixels), 1 / 255)
        let diff = (0..<n).map { i -> Float in previous.map { Float(abs(Int(gray[i]) - Int($0[i]))) } ?? 0 }
        add(mean(diff) / 255, std(diff) / 255, Double(FeatureMath.quantile(diff, 0.9) / 255),
            Double(diff.filter { $0 >= 18 }.count) / Double(n))
        scaled(grid(diff), 1 / 255)
        let focus = lapVariance / (lapVariance + 100)
        let dark = Double(pixels.filter { $0 <= 12 }.count) / Double(n)
        let bright = Double(pixels.filter { $0 >= 243 }.count) / Double(n)
        let lowTexture = Double((0..<n).filter { hypot(Double(gx[$0]), Double(gy[$0])) < 8 }.count) / Double(n)
        var occluded = 0
        for row in 0..<3 { for column in 0..<3 {
            let cell = rect(pixels, row: row, column: column), m = mean(cell)
            if (m <= 16 || m >= 239) && std(cell) <= 8 { occluded += 1 }
        } }
        let occlusion = Double(occluded) / 9
        let visibility = sqrt(max(0, focus * max(0, 1 - min(1, dark + bright)) * min(1, lumaStd / 255 / 0.12))) * (1 - occlusion)
        let shiftMagnitude = hypot(shift[0], shift[1]) / diagonal
        add(focus, 1 - focus, dark, bright, lowTexture, occlusion, visibility,
            shift[0] / Double(w), shift[1] / Double(h), shiftMagnitude, max(0, min(1, shift[2])))
        let flowX = (0..<n).map { flow[$0 * 2] }, flowY = (0..<n).map { flow[$0 * 2 + 1] }
        let magnitude = (0..<n).map { Float(hypot(Double(flowX[$0]), Double(flowY[$0]))) }
        let medianX = FeatureMath.quantile(flowX, 0.5), medianY = FeatureMath.quantile(flowY, 0.5)
        add(mean(magnitude) / diagonal, Double(FeatureMath.quantile(magnitude, 0.9)) / diagonal,
            Double(magnitude.filter { $0 >= 1 }.count) / Double(n), Double(medianX / Float(w)), Double(medianY / Float(h)))
        scaled(grid(magnitude), 1 / diagonal)
        var residual = [Float](repeating: 0, count: n), active = 0
        var vectorX = 0.0, vectorY = 0.0, activeMagnitude = 0.0
        for i in 0..<n {
            let x = Double(flowX[i] - medianX), y = Double(flowY[i] - medianY)
            residual[i] = Float(hypot(x, y))
            if residual[i] >= 1 { active += 1; vectorX += x; vectorY += y; activeMagnitude += Double(residual[i]) }
        }
        let residualGrid = grid(residual), total = residualGrid.reduce(0.0) { $0 + Double($1) }
        var entropy = 0.0
        if total > 1e-9 {
            for value in residualGrid { let p = Double(value) / total; if p > 0 { entropy -= p * log(p) } }
            entropy /= log(9)
        }
        let geometry = geometry(residual)
        let coherence = active > 0 ? hypot(vectorX / Double(active), vectorY / Double(active)) / max(activeMagnitude / Double(active), 1e-6) : 0
        let residualMean = mean(residual) / diagonal
        add(residualMean, Double(FeatureMath.quantile(residual, 0.9)) / diagonal,
            Double(active) / Double(n), Double(residualGrid.filter { $0 >= 0.75 }.count) / 9, entropy,
            geometry[0], geometry[1], geometry[2], geometry[3], coherence,
            residualMean * visibility * max(0, 1 - min(1, shiftMagnitude / 0.03)))
        scaled(residualGrid, 1 / diagonal)
        guard values.count == 73, values.allSatisfy(\.isFinite) else { throw AnalysisError.invalid("Visual signature or values invalid") }
        previous = gray
        return values
    }

    private func mean(_ xs: [Float]) -> Double { xs.isEmpty ? 0 : xs.reduce(0.0) { $0 + Double($1) } / Double(xs.count) }
    private func std(_ xs: [Float]) -> Double {
        let m = mean(xs)
        return xs.isEmpty ? 0 : sqrt(xs.reduce(0.0) { $0 + pow(Double($1) - m, 2) } / Double(xs.count))
    }
    private func rect(_ xs: [Float], row: Int, column: Int) -> [Float] {
        let top = Int((Float(row * h) / 3).rounded()), bottom = Int((Float((row + 1) * h) / 3).rounded())
        let left = Int((Float(column * w) / 3).rounded()), right = Int((Float((column + 1) * w) / 3).rounded())
        return (top..<bottom).flatMap { Array(xs[($0 * w + left)..<($0 * w + right)]) }
    }
    private func grid(_ xs: [Float]) -> [Float] {
        (0..<9).map { Float(mean(rect(xs, row: $0 / 3, column: $0 % 3))) }
    }
    private func geometry(_ magnitude: [Float]) -> [Double] {
        var xWeights = [Double](repeating: 0, count: w), yWeights = [Double](repeating: 0, count: h), total = 0.0
        for y in 0..<h { for x in 0..<w {
            let v = Double(magnitude[y * w + x]); if v < 0.5 { continue }
            xWeights[x] += v; yWeights[y] += v; total += v
        } }
        if total <= 1e-9 { return [0.5, 0.5, 0, 0] }
        var cx = 0.0, cy = 0.0, vx = 0.0, vy = 0.0
        for x in 0..<w { cx += xWeights[x] * ((Double(x) + 0.5) / Double(w)) }
        for y in 0..<h { cy += yWeights[y] * ((Double(y) + 0.5) / Double(h)) }
        cx /= total; cy /= total
        for x in 0..<w { vx += xWeights[x] * pow((Double(x) + 0.5) / Double(w) - cx, 2) }
        for y in 0..<h { vy += yWeights[y] * pow((Double(y) + 0.5) / Double(h) - cy, 2) }
        return [cx, cy, sqrt(vx / total), sqrt(vy / total)]
    }
}
