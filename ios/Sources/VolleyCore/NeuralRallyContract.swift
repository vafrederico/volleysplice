import Foundation

public struct NeuralRallyAsset: Codable, Sendable, Equatable {
    public let name: String, sha256: String
    public let sizeBytes: Int
}

public struct NeuralRallyBundle: Codable, Sendable, Equatable {
    public struct Files: Codable, Sendable, Equatable {
        public let encoder: NeuralRallyAsset, temporal: NeuralRallyAsset, pipeline: NeuralRallyAsset
    }
    public let id: String, label: String, directory: String
    public let encoderWeightsSha256: String, temporalWeightsSha256: String, embeddingPrecision: String
    public let files: Files
}

public struct NeuralRallyManifest: Sendable {
    public let schemaVersion: Int, family: String, precision: String, defaultVariant: String
    public let variants: [String: NeuralRallyBundle]

    public init(data: Data) throws {
        struct Payload: Decodable {
            let schemaVersion: Int, family: String, precision: String, defaultVariant: String
            let variants: [String: NeuralRallyBundle]
        }
        let value = try JSONDecoder().decode(Payload.self, from: data)
        guard value.schemaVersion == 1, value.family == "distilled-mobilenet-v3-large-tcn",
              value.precision == "fp32", value.defaultVariant == RallyModel.default.variantKey,
              Set(value.variants.keys) == Set(["high-recall", "high-f1"]) else {
            throw AnalysisError.invalid("The rally model catalog is incompatible")
        }
        for model in [RallyModel.maximumCoverage, .balanced] {
            guard let bundle = value.variants[model.variantKey!], bundle.id == model.modelId,
                  bundle.directory == model.assetDirectory, bundle.embeddingPrecision == "fp32",
                  Self.validHash(bundle.encoderWeightsSha256), Self.validHash(bundle.temporalWeightsSha256) else {
                throw AnalysisError.invalid("The selected rally bundle does not match its model identity")
            }
            for (asset, name) in [(bundle.files.encoder, "encoder.onnx"), (bundle.files.temporal, "temporal.onnx"), (bundle.files.pipeline, "pipeline.json")] {
                guard asset.name == name, Self.validHash(asset.sha256), asset.sizeBytes > 0 else {
                    throw AnalysisError.invalid("The rally model asset manifest is incomplete")
                }
            }
        }
        schemaVersion = value.schemaVersion; family = value.family; precision = value.precision
        defaultVariant = value.defaultVariant; variants = value.variants
    }

    public func bundle(for model: RallyModel) throws -> NeuralRallyBundle {
        guard let key = model.variantKey, let bundle = variants[key] else {
            throw AnalysisError.invalid("The selected model does not use a neural bundle")
        }
        return bundle
    }

    private static func validHash(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }
}

public struct NeuralDecoderConfig: Codable, Sendable, Equatable {
    public let smoothing: Double, minimum: Double, enter: Double
    public let boundary: Bool
    public init(smoothing: Double, minimum: Double, enter: Double, boundary: Bool) throws {
        guard smoothing.isFinite, smoothing >= 0, smoothing <= 86400,
              minimum.isFinite, minimum > 0, minimum <= 86400,
              enter.isFinite, enter > 0.1, enter <= 1 else {
            throw AnalysisError.invalid("Invalid neural rally decoder")
        }
        self.smoothing = smoothing; self.minimum = minimum; self.enter = enter; self.boundary = boundary
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(smoothing: c.decode(Double.self, forKey: .smoothing), minimum: c.decode(Double.self, forKey: .minimum),
                      enter: c.decode(Double.self, forKey: .enter), boundary: c.decode(Bool.self, forKey: .boundary))
    }
}

public struct NeuralPipelineConfig: Sendable {
    public let modelIdentity: String, selectionMode: String
    public let recallTargetPercent: Int, tokenDimension: Int
    public let mean: [Double], scale: [Double]
    public let decoder: NeuralDecoderConfig
    public let weightsSha256: String, encoderWeightsSha256: String

    public init(data: Data, model: RallyModel, bundle: NeuralRallyBundle? = nil) throws {
        struct Payload: Decodable {
            let modelIdentity: String, selectionMode: String
            let recallTargetPercent: Int, tokenDimension: Int
            let mean: [Double], scale: [Double]
            let decoder: NeuralDecoderConfig
            let weightsSha256: String, encoderWeightsSha256: String
        }
        let value = try JSONDecoder().decode(Payload.self, from: data)
        guard model.isNeural, value.modelIdentity == "dino-distilled-mobilenet-v3-large-tcn",
              value.selectionMode == (model == .maximumCoverage ? "recall" : "f1"), value.recallTargetPercent == 99,
              value.tokenDimension == NeuralRallyContract.tokenDimension, value.mean.count == 112, value.scale.count == 112,
              value.mean.allSatisfy(\.isFinite), value.scale.allSatisfy({ $0.isFinite && $0 > 0 }) else {
            throw AnalysisError.invalid("The rally model preprocessing configuration is incompatible")
        }
        if let bundle {
            guard bundle.id == model.modelId, bundle.encoderWeightsSha256 == value.encoderWeightsSha256,
                  bundle.temporalWeightsSha256 == value.weightsSha256 else {
                throw AnalysisError.invalid("The rally encoder, temporal head and scalers must come from the same variant")
            }
        }
        modelIdentity = value.modelIdentity; selectionMode = value.selectionMode
        recallTargetPercent = value.recallTargetPercent; tokenDimension = value.tokenDimension
        mean = value.mean; scale = value.scale; decoder = value.decoder
        weightsSha256 = value.weightsSha256; encoderWeightsSha256 = value.encoderWeightsSha256
    }
}

public struct NeuralImageGeometry: Sendable, Equatable {
    public let x: Int, y: Int, cropWidth: Int, cropHeight: Int
    public let resizedWidth: Int, resizedHeight: Int, left: Int, top: Int
    public var box: [Double] {
        [left, top, left + resizedWidth, top + resizedHeight].map { Double($0) / Double(NeuralRallyContract.inputSize) }
    }
}

public struct NeuralTemporalChunk: Sendable, Equatable {
    public let core: Range<Int>, input: Range<Int>
}

/// Frozen native contract. Preserve scalar operation precision and order at threshold crossings.
/// Native encoders keep FP32 tokens; the browser's optional FP16 storage is not used here.
public enum NeuralRallyContract {
    public static let inputSize = 224, tokenDimension = 3840, fusedDimension = 3952
    public static let contextualDimension = 520, avDimension = 104
    public static let chunkRows = 128, contextRows = 62
    private static let imageMean: [Float] = [0.485, 0.456, 0.406]
    private static let imageScale: [Float] = [0.229, 0.224, 0.225]

    public static func embeddingTimes(duration: Double, start: Double, end: Double) throws -> [Double] {
        guard [duration, start, end].allSatisfy(\.isFinite), start >= 0, end <= duration, end > start,
              end <= Double(Int.max / 4) else { throw AnalysisError.invalid("Invalid embedding analysis window") }
        let first = Int(floor(start * 2)), last = Int(ceil(end * 2 - 1e-9))
        return (first..<last).map { Double($0) / 2 }
    }

    public static func geometry(width: Int, height: Int, rotation: Int = 0, roi: [Double] = [0, 0, 1, 1]) throws -> NeuralImageGeometry {
        guard width > 0, height > 0, width <= 1_000_000, height <= 1_000_000,
              [0, 90, 180, 270].contains(rotation), roi.count == 4, roi.allSatisfy(\.isFinite),
              roi[0] >= 0, roi[1] >= 0, roi[2] > 0, roi[3] > 0,
              roi[0] + roi[2] <= 1 + 1e-9, roi[1] + roi[3] <= 1 + 1e-9 else {
            throw AnalysisError.invalid("Invalid neural image geometry")
        }
        let w = rotation == 90 || rotation == 270 ? height : width
        let h = rotation == 90 || rotation == 270 ? width : height
        // Android uses Math.round for positive pixel geometry (ties round upward).
        let x = Int(floor(roi[0] * Double(w) + 0.5)), y = Int(floor(roi[1] * Double(h) + 0.5))
        let cropWidth = Int(floor(roi[2] * Double(w) + 0.5)), cropHeight = Int(floor(roi[3] * Double(h) + 0.5))
        guard cropWidth > 0, cropHeight > 0, x + cropWidth <= w, y + cropHeight <= h else {
            throw AnalysisError.invalid("Empty or out-of-bounds neural crop")
        }
        let scale = min(Double(inputSize) / Double(cropWidth), Double(inputSize) / Double(cropHeight))
        let resizedWidth = max(1, Int(floor(Double(cropWidth) * scale + 0.5)))
        let resizedHeight = max(1, Int(floor(Double(cropHeight) * scale + 0.5)))
        return NeuralImageGeometry(x: x, y: y, cropWidth: cropWidth, cropHeight: cropHeight,
            resizedWidth: resizedWidth, resizedHeight: resizedHeight,
            left: (inputSize - resizedWidth) / 2, top: (inputSize - resizedHeight) / 2)
    }

    public static func regionalPoolWeights(box: [Double]) throws -> [Float] {
        guard box.count == 4, box.allSatisfy(\.isFinite), box[0] >= 0, box[1] >= 0,
              box[2] <= 1, box[3] <= 1, box[2] > box[0], box[3] > box[1] else {
            throw AnalysisError.invalid("Invalid regional pool content bounds")
        }
        let left = box[0], top = box[1], right = box[2], bottom = box[3]
        var weights = [Float](repeating: 0, count: 196)
        for (region, limits) in [(0.0, 1.0), (0.5, 1.0), (0.0, 0.5), (0.4, 0.6)].enumerated() {
            let a = top + limits.0 * (bottom - top), b = top + limits.1 * (bottom - top)
            var areas = [Double](repeating: 0, count: 49), total = 0.0
            for y in 0..<7 { for x in 0..<7 {
                let area = max(0, min(Double(x + 1) / 7, right) - max(Double(x) / 7, left))
                    * max(0, min(Double(y + 1) / 7, b) - max(Double(y) / 7, a))
                areas[y * 7 + x] = area; total += area
            } }
            guard total > 0 else { throw AnalysisError.invalid("Empty regional pool") }
            for i in 0..<49 { weights[region * 49 + i] = Float(areas[i] / total) }
        }
        return weights
    }

    public static func normalizedLetterbox(rgb: [UInt8], geometry: NeuralImageGeometry) throws -> [Float] {
        try normalizedLetterbox(rgb: rgb.map(Float.init), geometry: geometry)
    }

    /// Input already contains the 224-square letterbox, in fractional HWC RGB 0...255.
    public static func normalizedCHW(rgb: [Float]) throws -> [Float] {
        let plane = inputSize * inputSize
        guard rgb.count == plane * 3, rgb.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 255 }) else {
            throw AnalysisError.invalid("Neural RGB tensor shape or values differ")
        }
        var output = [Float](repeating: 0, count: rgb.count)
        for channel in 0..<3 { for i in 0..<plane {
            output[channel * plane + i] = (rgb[i * 3 + channel] / 255 - imageMean[channel]) / imageScale[channel]
        } }
        return output
    }

    /// Input is resized content-only HWC RGB in 0...255; preserve bilinear fractional pixels.
    public static func normalizedLetterbox(rgb: [Float], geometry: NeuralImageGeometry) throws -> [Float] {
        let w = geometry.resizedWidth, h = geometry.resizedHeight, plane = inputSize * inputSize
        guard rgb.count == w * h * 3, rgb.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 255 }) else {
            throw AnalysisError.invalid("Neural RGB content shape or values differ")
        }
        var output = [Float](repeating: 0, count: plane * 3)
        for channel in 0..<3 {
            let padding = -imageMean[channel] / imageScale[channel]
            for i in 0..<plane { output[channel * plane + i] = padding }
            for y in 0..<h { for x in 0..<w {
                let value = rgb[(y * w + x) * 3 + channel] / 255
                output[channel * plane + (y + geometry.top) * inputSize + x + geometry.left] = (value - imageMean[channel]) / imageScale[channel]
            } }
        }
        return output
    }

    /// The six image-quality scalars are calculated on content only, never black padding.
    public static func quality(chw: [Float], geometry: NeuralImageGeometry) throws -> [Float] {
        let w = geometry.resizedWidth, h = geometry.resizedHeight, n = w * h, plane = inputSize * inputSize
        guard chw.count == plane * 3, chw.allSatisfy(\.isFinite) else { throw AnalysisError.invalid("Invalid normalized image tensor") }
        var gray = [Float](repeating: 0, count: n), sum = 0.0, squares = 0.0, clipped = 0.0
        func channel(_ index: Int, _ channel: Int) -> Double {
            let value = (chw[channel * plane + index] * imageScale[channel] + imageMean[channel]) * 255
            return floor(Double(value) + 0.5)
        }
        for y in 0..<h { for x in 0..<w {
            let i = (y + geometry.top) * inputSize + x + geometry.left
            let r = channel(i, 0), g = channel(i, 1), b = channel(i, 2)
            let v = Float(floor(0.299 * r + 0.587 * g + 0.114 * b + 0.5)) / 255
            gray[y * w + x] = v; sum += Double(v); squares += Double(v * v)
            if v <= 2 / Float(255) || v >= 253 / Float(255) { clipped += 1 }
        } }
        var lapSum = 0.0, lapSquares = 0.0
        for y in 0..<h { for x in 0..<w {
            let xl = x == 0 ? min(1, w - 1) : x - 1, xr = x == w - 1 ? max(0, w - 2) : x + 1
            let yu = y == 0 ? min(1, h - 1) : y - 1, yd = y == h - 1 ? max(0, h - 2) : y + 1
            let v = Double(gray[y * w + xl] + gray[y * w + xr] + gray[yu * w + x] + gray[yd * w + x] - 4 * gray[y * w + x])
            lapSum += v; lapSquares += v * v
        } }
        let count = Double(n)
        return [Float(n) / Float(plane), Float(sum / count), Float(sqrt(max(0, squares / count - sum * sum / count / count))),
                Float(max(0, lapSquares / count - lapSum * lapSum / count / count)), Float(clipped / count), 0]
    }

    public static func chunkPlan(rows: Int) throws -> [NeuralTemporalChunk] {
        guard rows >= 0, rows <= Int.max - chunkRows - contextRows else { throw AnalysisError.invalid("Invalid neural row count") }
        return stride(from: 0, to: rows, by: chunkRows).map { start in
            let end = min(rows, start + chunkRows)
            return NeuralTemporalChunk(core: start..<end, input: max(0, start - contextRows)..<min(rows, end + contextRows))
        }
    }

    /// Center block of contextualized AV is ranked before normalization, matching native Android.
    /// Only materialize the requested halo chunk; do not allocate a full-video fused matrix.
    public static func fuse(times: [Double], contextual: [Float], embeddingTimes: [Double], tokens: [Float], quality: [Float],
                            config: NeuralPipelineConfig, rows: Range<Int>? = nil) throws -> [Float] {
        let selected = rows ?? 0..<times.count
        guard selected.lowerBound >= 0, selected.upperBound <= times.count,
              contextual.count / contextualDimension == times.count, contextual.count % contextualDimension == 0,
              tokens.count / tokenDimension == embeddingTimes.count, tokens.count % tokenDimension == 0,
              quality.count / 6 == embeddingTimes.count, quality.count % 6 == 0,
              times.allSatisfy({ $0.isFinite && $0 >= 0 }), zip(times, times.dropFirst()).allSatisfy({ $0 < $1 }),
              embeddingTimes.allSatisfy({ $0.isFinite && $0 >= 0 }),
              zip(embeddingTimes, embeddingTimes.dropFirst()).allSatisfy({ abs($1 - $0 - 0.5) < 1e-9 }) else {
            throw AnalysisError.invalid("Neural fusion dimensions or timestamps differ")
        }
        guard !selected.isEmpty else { return [] }
        guard let sampleStart = embeddingTimes.first else { throw AnalysisError.invalid("Uncovered AV timestamp") }
        var output = [Float](repeating: 0, count: selected.count * fusedDimension)
        func scale(_ value: Float, _ column: Int) throws -> Float {
            guard value.isFinite else { throw AnalysisError.invalid("Non-finite neural feature") }
            return Float(max(-10, min(10, (Double(value) - config.mean[column]) / config.scale[column])))
        }
        for row in selected {
            let dst = (row - selected.lowerBound) * fusedDimension
            for c in 0..<avDimension { output[dst + c] = try scale(contextual[row * contextualDimension + 208 + c], c) }
            let sampleValue = floor((times[row] - sampleStart) * 2 + 1e-8)
            guard sampleValue >= 0, sampleValue < Double(embeddingTimes.count) else { throw AnalysisError.invalid("Uncovered AV timestamp") }
            let sample = Int(sampleValue)
            for c in 0..<tokenDimension {
                let value = tokens[sample * tokenDimension + c]
                guard value.isFinite else { throw AnalysisError.invalid("Non-finite neural token") }
                output[dst + avDimension + c] = value
            }
            for c in 0..<8 {
                let value = c < 6 ? quality[sample * 6 + c] : c == 6 ? Float(times[row] - embeddingTimes[sample]) : 1
                output[dst + avDimension + tokenDimension + c] = try scale(value, avDimension + c)
            }
        }
        return output
    }

    public static func probabilities(logits: [Float]) throws -> [Float] {
        guard logits.count % 4 == 0, logits.allSatisfy(\.isFinite) else { throw AnalysisError.invalid("Invalid neural logits") }
        return logits.map { Float(1 / (1 + exp(-Double($0)))) }
    }

    public static func decode(times: [Double], probabilities: [Float], duration: Double, decoder: NeuralDecoderConfig) throws -> [Interval] {
        guard duration.isFinite, duration >= 0, probabilities.count / 4 == times.count, probabilities.count % 4 == 0,
              probabilities.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 }),
              times.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= duration }),
              zip(times, times.dropFirst()).allSatisfy({ $0 < $1 }) else { throw AnalysisError.invalid("Invalid neural probability timeline") }
        let n = times.count
        guard n > 0 else { return [] }
        let window = min(n, max(1, Int((decoder.smoothing * 4).rounded(.toNearestOrEven))))
        let minimum = max(1, Int((decoder.minimum * 4).rounded(.toNearestOrEven)))
        var smooth = [Float](repeating: 0, count: n), mask = [Bool](repeating: false, count: n), live = false
        for i in 0..<n {
            for k in 0..<window { smooth[i] += probabilities[max(0, min(n - 1, i + k - window / 2)) * 4] / Float(window) }
            if !live && Double(smooth[i]) >= decoder.enter { live = true }
            else if live && Double(smooth[i]) < decoder.enter - 0.1 { live = false }
            mask[i] = live
        }
        var i = 0
        while i < n {
            var end = i + 1
            while end < n && mask[end] == mask[i] { end += 1 }
            if !mask[i] && i > 0 && end < n && end - i <= 2 { for k in i..<end { mask[k] = true } }
            i = end
        }
        var output: [Interval] = []; i = 0
        while i < n {
            if !mask[i] { i += 1; continue }
            var end = i + 1
            while end < n && mask[end] { end += 1 }
            var peak: Float = 0, sum: Float = 0
            for k in i..<end { peak = max(peak, smooth[k]); sum += smooth[k] }
            if end - i >= minimum || Double(peak) >= 0.9 {
                var a = max(0, times[i] - 0.125), b = min(duration, times[end - 1] + 0.125)
                if decoder.boundary {
                    let originalA = a, originalB = b
                    var bestA: Float = -1, bestB: Float = -1
                    for k in 0..<n {
                        if abs(times[k] - originalA) <= 0.75 && Double(probabilities[k * 4 + 1]) >= 0.65 && probabilities[k * 4 + 1] > bestA {
                            a = times[k]; bestA = probabilities[k * 4 + 1]
                        }
                        if abs(times[k] - originalB) <= 0.75 && Double(probabilities[k * 4 + 2]) >= 0.65 && probabilities[k * 4 + 2] > bestB {
                            b = times[k]; bestB = probabilities[k * 4 + 2]
                        }
                    }
                }
                if b > a { output.append(Interval(start: a, end: b, confidence: sum / Float(end - i), agreement: "neural")) }
            }
            i = end
        }
        return output
    }
}
