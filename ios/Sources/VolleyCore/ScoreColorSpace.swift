import Foundation

/// Export graphics are authored in sRGB. Video samples stay in their original
/// primaries/transfer/matrix; only pixels covered by graphics enter linear RGB.
/// HDR graphics use BT.2408 diffuse white (203 cd/m²), with the BT.2100 HLG
/// reference display (1000 cd/m², system gamma 1.2, zero black).
struct ScoreColorSpace: Sendable, Equatable {
    enum Primaries: String, Sendable { case bt709, bt2020, p3D65, smpteC, ebu3213 }
    enum Transfer: String, Sendable { case bt709, sRGB, hlg, pq }
    enum Matrix: String, Sendable { case bt709, bt601, bt2020 }
    let primaries: Primaries
    let transfer: Transfer
    let matrix: Matrix
    var isHDR: Bool { transfer == .hlg || transfer == .pq }
    static let rec709 = Self(primaries: .bt709, transfer: .bt709, matrix: .bt709)
    var kr: Double { matrix == .bt601 ? 0.299 : matrix == .bt2020 ? 0.2627 : 0.2126 }
    var kb: Double { matrix == .bt601 ? 0.114 : matrix == .bt2020 ? 0.0593 : 0.0722 }

    /// Rows map linear sRGB/BT.709 into the destination RGB primaries (D65).
    private var gamut: [SIMD3<Double>] {
        switch primaries {
        case .bt709: return [.init(1, 0, 0), .init(0, 1, 0), .init(0, 0, 1)]
        case .bt2020: return [.init(0.627403896, 0.329283038, 0.043313066), .init(0.069097289, 0.919540395, 0.011362316), .init(0.016391439, 0.088013308, 0.895595253)]
        case .p3D65: return [.init(0.822461969, 0.177538031, 0), .init(0.033194199, 0.966805801, 0), .init(0.017082631, 0.072397441, 0.910519928)]
        case .smpteC: return [.init(1.065379342, -0.055400873, -0.009978469), .init(-0.019632549, 1.036363091, -0.016730542), .init(0.001632052, 0.004412373, 0.993955575)]
        case .ebu3213: return [.init(0.957822048, 0.042177952, 0), .init(0, 1, 0), .init(0, -0.011936665, 1.011936665)]
        }
    }
    static func srgbToLinear(_ value: Double) -> Double {
        value <= 0.04045 ? value / 12.92 : pow((value + 0.055) / 1.055, 2.4)
    }
    func graphicLinear(_ srgb: SIMD3<Double>) -> SIMD3<Double> {
        let input = SIMD3(Self.srgbToLinear(srgb.x), Self.srgbToLinear(srgb.y), Self.srgbToLinear(srgb.z))
        let rows = gamut
        let converted = SIMD3((rows[0] * input).sum(), (rows[1] * input).sum(), (rows[2] * input).sum())
        return SIMD3(max(0, converted.x), max(0, converted.y), max(0, converted.z)) * (isHDR ? 203 : 1)
    }
    func decode(_ rgb: SIMD3<Double>) -> SIMD3<Double> {
        func channel(_ v: Double) -> Double {
            let v = max(0, v)
            switch transfer {
            case .sRGB: return Self.srgbToLinear(v)
            case .bt709: return v < 0.081 ? v / 4.5 : pow((v + 0.099) / 1.099, 1 / 0.45)
            case .pq:
                let p = pow(v, 1 / (2523.0 / 32))
                return 10000 * pow(max(0, p - 3424.0 / 4096) / max(1e-12, 2413.0 / 128 - 2392.0 / 128 * p), 1 / (2610.0 / 16384))
            case .hlg:
                return v <= 0.5 ? v * v / 3 : (exp((v - 0.55991073) / 0.17883277) + 0.28466892) / 12
            }
        }
        let linear = SIMD3(channel(rgb.x), channel(rgb.y), channel(rgb.z))
        if transfer != .hlg { return linear }
        // HLG OOTF couples channels through scene luminance, not a per-channel gamma.
        let luminance = max(0, 0.2627 * linear.x + 0.6780 * linear.y + 0.0593 * linear.z)
        return linear * (1000 * pow(luminance, 0.2))
    }
    func encode(_ linear: SIMD3<Double>) -> SIMD3<Double> {
        var scene = linear
        if transfer == .hlg {
            let displayY = max(0, (0.2627 * linear.x + 0.6780 * linear.y + 0.0593 * linear.z) / 1000)
            scene = displayY > 0 ? linear / (1000 * pow(displayY, 1 / 6.0)) : .zero
        }
        func channel(_ input: Double) -> Double {
            let v = max(0, input)
            switch transfer {
            case .sRGB: return v <= 0.0031308 ? 12.92 * v : 1.055 * pow(v, 1 / 2.4) - 0.055
            case .bt709: return v < 0.018 ? 4.5 * v : 1.099 * pow(v, 0.45) - 0.099
            case .pq:
                let p = pow(v / 10000, 2610.0 / 16384)
                return pow((3424.0 / 4096 + 2413.0 / 128 * p) / (1 + 2392.0 / 128 * p), 2523.0 / 32)
            case .hlg: return v <= 1 / 12.0 ? sqrt(3 * v) : 0.17883277 * log(12 * v - 0.28466892) + 0.55991073
            }
        }
        return SIMD3(channel(scene.x), channel(scene.y), channel(scene.z))
    }
    func yuv(_ rgb: SIMD3<Double>) -> SIMD3<Double> {
        let y = kr * rgb.x + (1 - kr - kb) * rgb.y + kb * rgb.z
        return SIMD3(y, (rgb.z - y) / (2 * (1 - kb)), (rgb.x - y) / (2 * (1 - kr)))
    }
    func rgb(_ yuv: SIMD3<Double>) -> SIMD3<Double> {
        let red = yuv.x + 2 * (1 - kr) * yuv.z, blue = yuv.x + 2 * (1 - kb) * yuv.y
        return SIMD3(red, (yuv.x - kr * red - kb * blue) / (1 - kr - kb), blue)
    }
}

struct ScoreBlendPixel: Sendable {
    let alpha: Double
    let linear: SIMD3<Double>
    let encodedYUV: SIMD3<Double>
}

struct ScoreBlendRaster: Sendable {
    let width: Int, height: Int, scoreWidth: Int
    let color: ScoreColorSpace
    let palette: [ScoreBlendPixel]
    let indices: [UInt32]
    init(bgra: Data, width: Int, height: Int, scoreWidth: Int, color: ScoreColorSpace) {
        precondition(bgra.count == width * height * 4)
        self.width = width; self.height = height; self.scoreWidth = scoreWidth; self.color = color
        var cache: [UInt32: UInt32] = [:]
        var palette: [ScoreBlendPixel] = []
        indices = stride(from: 0, to: bgra.count, by: 4).map { index in
            let key = UInt32(bgra[index]) | UInt32(bgra[index + 1]) << 8 | UInt32(bgra[index + 2]) << 16 | UInt32(bgra[index + 3]) << 24
            if let index = cache[key] { return index }
            let a = Double(bgra[index + 3])
            let linear = a > 0 ? color.graphicLinear(SIMD3(Double(bgra[index + 2]) / a, Double(bgra[index + 1]) / a, Double(bgra[index]) / a)) : .zero
            let pixel = ScoreBlendPixel(alpha: a / 255, linear: linear, encodedYUV: color.yuv(color.encode(linear)))
            let paletteIndex = UInt32(palette.count)
            palette.append(pixel); cache[key] = paletteIndex
            return paletteIndex
        }
        self.palette = palette
    }

    /// Both planes are video range: NV12 bytes or P010 (MSB-aligned UInt16).
    /// Chroma is shared by 2x2 pixels. Uncovered luma and untouched chroma blocks
    /// remain bit exact, including the unused low six bits of P010 containers.
    func blend(y: UnsafeMutableRawPointer, yStride: Int, uv: UnsafeMutableRawPointer, uvStride: Int,
               frameWidth: Int, frameHeight: Int, tenBit: Bool, pointOpacity: Double, scoreOpacity: Double) {
        let size = tenBit ? 2 : 1, scale = tenBit ? 4.0 : 1.0
        func read(_ base: UnsafeMutableRawPointer, _ offset: Int) -> Double {
            tenBit ? Double(base.load(fromByteOffset: offset, as: UInt16.self) >> 6) : Double(base.load(fromByteOffset: offset, as: UInt8.self))
        }
        func write(_ base: UnsafeMutableRawPointer, _ offset: Int, _ value: Double) {
            let value = max(0, min(tenBit ? 1023 : 255, value.rounded()))
            if tenBit { base.storeBytes(of: UInt16(value) << 6, toByteOffset: offset, as: UInt16.self) }
            else { base.storeBytes(of: UInt8(value), toByteOffset: offset, as: UInt8.self) }
        }
        for row in stride(from: 0, to: min(height, frameHeight), by: 2) {
            for column in stride(from: 0, to: min(width, frameWidth), by: 2) {
                let ci = row / 2 * uvStride + column / 2 * 2 * size
                let cb = (read(uv, ci) - 128 * scale) / (224 * scale)
                let cr = (read(uv, ci + size) - 128 * scale) / (224 * scale)
                var chromaDelta = SIMD2<Double>.zero, covered = false
                let count = min(2, frameWidth - column) * min(2, frameHeight - row)
                for dy in 0..<2 where row + dy < min(height, frameHeight) {
                    for dx in 0..<2 where column + dx < min(width, frameWidth) {
                        let pixel = palette[Int(indices[(row + dy) * width + column + dx])]
                        let opacity = max(0, min(1, column + dx >= scoreWidth ? pointOpacity : scoreOpacity))
                        let alpha = pixel.alpha * opacity
                        if alpha <= 0 { continue }
                        let yi = (row + dy) * yStride + (column + dx) * size
                        let result: SIMD3<Double>
                        if alpha >= 1 { result = pixel.encodedYUV }
                        else {
                            let sourceY = (read(y, yi) - 16 * scale) / (219 * scale)
                            let background = color.decode(color.rgb(SIMD3(sourceY, cb, cr)))
                            result = color.yuv(color.encode(background * (1 - alpha) + pixel.linear * alpha))
                        }
                        write(y, yi, scale * (16 + 219 * result.x))
                        chromaDelta += SIMD2(result.y - cb, result.z - cr)
                        covered = true
                    }
                }
                if covered {
                    write(uv, ci, scale * (128 + 224 * (cb + chromaDelta.x / Double(count))))
                    write(uv, ci + size, scale * (128 + 224 * (cr + chromaDelta.y / Double(count))))
                }
            }
        }
    }
}
