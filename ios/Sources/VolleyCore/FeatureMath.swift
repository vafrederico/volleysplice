import Foundation

public enum AnalysisError: Error, CustomStringConvertible {
    case invalid(String)
    public var description: String { switch self { case .invalid(let message): return message } }
}

public enum FeatureMath {
    public static let absolute: Set<String> = ["audio_available", "focus_quality", "blur_probability",
        "occlusion_fraction", "visibility_quality", "camera_shift_response"]
    public static let contextOffsets = [-2, -1, 0, 1, 2]

    public static func validate(times: [Double], values: [Float], columns: Int) throws {
        guard columns > 0, !times.isEmpty, values.count / columns == times.count,
              values.count % columns == 0, values.allSatisfy(\.isFinite),
              times.allSatisfy({ $0.isFinite && $0 >= 0 }),
              zip(times, times.dropFirst()).allSatisfy({ $0 < $1 }) else {
            throw AnalysisError.invalid("Expected a finite matrix with strictly increasing source timestamps")
        }
    }

    public static func ranks(_ values: [Float], rows: Int, columns: Int) throws -> [Float] {
        guard rows >= 0, columns > 0, values.count / columns == rows,
              values.count % columns == 0, values.allSatisfy(\.isFinite) else {
            throw AnalysisError.invalid("Invalid percentile matrix")
        }
        if rows == 0 { return [] }
        if rows == 1 { return [Float](repeating: 0.5, count: columns) }
        var output = [Float](repeating: 0, count: values.count)
        for column in 0..<columns {
            let order = (0..<rows).sorted { values[$0 * columns + column] < values[$1 * columns + column] }
            var start = 0
            while start < rows {
                var end = start + 1
                while end < rows && values[order[end] * columns + column] == values[order[start] * columns + column] { end += 1 }
                let rank = Float(Double(start + end - 1) / 2 / Double(rows - 1))
                for i in start..<end { output[order[i] * columns + column] = rank }
                start = end
            }
        }
        return output
    }

    public static func nearest(_ times: [Double], _ target: Double) -> Int {
        var lower = 0, upper = times.count
        while lower < upper {
            let mid = (lower + upper) / 2
            if times[mid] < target { lower = mid + 1 } else { upper = mid }
        }
        let right = min(times.count - 1, lower), left = max(0, right - 1)
        return abs(times[left] - target) <= abs(times[right] - target) ? left : right
    }

    public static func contextualize(times: [Double], base: [Float], names: [String]) throws -> [Float] {
        try validate(times: times, values: base, columns: names.count)
        let columns = names.count, width = columns * contextOffsets.count
        var ranked = try ranks(base, rows: times.count, columns: columns)
        for column in names.indices where absolute.contains(names[column]) {
            for row in times.indices { ranked[row * columns + column] = base[row * columns + column] }
        }
        var output = [Float](repeating: 0, count: times.count * width)
        for (block, offset) in contextOffsets.enumerated() {
            for row in times.indices {
                let source = nearest(times, times[row] + Double(offset))
                for column in 0..<columns { output[row * width + block * columns + column] = ranked[source * columns + column] }
            }
        }
        return output
    }

    public static func rollingMean(_ values: [Float], window: Int, future: Bool) -> [Float] {
        var cumulative = [Double](repeating: 0, count: values.count + 1)
        for i in values.indices { cumulative[i + 1] = cumulative[i] + Double(values[i]) }
        return values.indices.map { i in
            let start = future ? i : max(0, i - max(1, window) + 1)
            let end = future ? min(values.count, i + max(1, window)) : i + 1
            return Float((cumulative[end] - cumulative[start]) / Double(end - start))
        }
    }

    public static func quantile(_ values: [Float], _ percentile: Double) -> Float {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted(), position = Double(values.count - 1) * max(0, min(1, percentile))
        let lower = Int(position.rounded(.down)), upper = Int(position.rounded(.up)), weight = position - Double(lower)
        return Float(Double(sorted[lower]) * (1 - weight) + Double(sorted[upper]) * weight)
    }
}
