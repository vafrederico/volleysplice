import XCTest
@testable import VolleyCore

final class ScoreColorSpaceTests: XCTestCase {
    private let hdr = [ScoreColorSpace(primaries: .bt2020, transfer: .hlg, matrix: .bt2020),
                       ScoreColorSpace(primaries: .bt2020, transfer: .pq, matrix: .bt2020)]

    func testHDRDiffuseWhiteAndST2084ReferenceValues() {
        XCTAssertEqual(hdr[0].encode(SIMD3(repeating: 203)).x, 0.75, accuracy: 0.0002)
        XCTAssertEqual(hdr[1].encode(SIMD3(repeating: 100)).x, 0.50807842, accuracy: 0.0000001)
        XCTAssertEqual(hdr[1].encode(SIMD3(repeating: 1000)).x, 0.75182710, accuracy: 0.0000001)
        XCTAssertEqual(hdr[1].encode(SIMD3(repeating: 10000)).x, 1, accuracy: 0.0000001)
        for space in hdr {
            XCTAssertEqual(space.graphicLinear(SIMD3(repeating: 1)).x, 203, accuracy: 0.000001)
            let red = space.graphicLinear(SIMD3(1, 0, 0)) / 203
            XCTAssertEqual(red.x, 0.627403896, accuracy: 0.000001)
            XCTAssertEqual(red.y, 0.069097289, accuracy: 0.000001)
            XCTAssertEqual(red.z, 0.016391439, accuracy: 0.000001)
        }
    }

    func testTransferAndMatrixRoundTripIncludingColoredHLGOOTF() {
        for space in [.rec709] + hdr {
            for encoded in [SIMD3(0.2, 0.7, 0.5), SIMD3(1.0, 0.6, 0.1), .zero] {
                let roundtrip = space.encode(space.decode(encoded))
                for i in 0..<3 { XCTAssertEqual(roundtrip[i], encoded[i], accuracy: 0.00001) }
                let matrix = space.rgb(space.yuv(encoded))
                for i in 0..<3 { XCTAssertEqual(matrix[i], encoded[i], accuracy: 1e-9) }
            }
        }
    }

    func testPremultipliedGraphicsUnpremultiplyBeforeTransfer() {
        let raster = ScoreBlendRaster(bgra: Data([128, 128, 128, 128]), width: 1, height: 1, scoreWidth: 1, color: hdr[1])
        XCTAssertEqual(raster.palette[Int(raster.indices[0])].linear.x, 203, accuracy: 1e-8)
        XCTAssertEqual(raster.palette[Int(raster.indices[0])].alpha, 128.0 / 255, accuracy: 1e-8)
    }

    func testNV12AndP010BlendInLinearLightPreservingUntouchedSamplesAndStrides() {
        for space in [.rec709] + hdr {
            for tenBit in [false, true] {
                let size = tenBit ? 2 : 1, yStride = 8 * size, uvStride = 8 * size
                let raster = ScoreBlendRaster(bgra: Data([255,255,255,255, 255,255,255,255,
                                                         255,255,255,255, 255,255,255,255]),
                    width: 2, height: 2, scoreWidth: 1, color: space)
                var y = Data(repeating: 0x7b, count: yStride * 4), uv = Data(repeating: 0x7b, count: uvStride * 2)
                func store(_ data: inout Data, _ index: Int, _ value: Int) {
                    if tenBit { data[index] = UInt8((value << 6) & 255); data[index + 1] = UInt8(value >> 2) }
                    else { data[index] = UInt8(value) }
                }
                func read(_ data: Data, _ index: Int) -> Double {
                    tenBit ? Double((UInt16(data[index]) | UInt16(data[index + 1]) << 8) >> 6) : Double(data[index])
                }
                for row in 0..<4 { for col in 0..<6 { store(&y, row * yStride + col * size, tenBit ? 64 : 16) } }
                for row in 0..<2 { for col in 0..<6 { store(&uv, row * uvStride + col * size, tenBit ? 512 : 128) } }
                let beforeY = y, beforeUV = uv
                y.withUnsafeMutableBytes { yp in uv.withUnsafeMutableBytes { cp in
                    raster.blend(y: yp.baseAddress!, yStride: yStride, uv: cp.baseAddress!, uvStride: uvStride,
                                 frameWidth: 6, frameHeight: 4, tenBit: tenBit, pointOpacity: 0.5, scoreOpacity: 1)
                } }
                let scale = tenBit ? 4.0 : 1.0
                let white = space.encode(SIMD3(repeating: space.isHDR ? 203 : 1)).x
                let halfWhite = space.encode(SIMD3(repeating: space.isHDR ? 101.5 : 0.5)).x
                XCTAssertEqual(read(y, 0), scale * (16 + 219 * white), accuracy: 0.51)
                XCTAssertEqual(read(y, size), scale * (16 + 219 * halfWhite), accuracy: 0.51)
                XCTAssertGreaterThan(halfWhite, white * 0.5, "Encoded-space alpha would produce dark edges")
                for row in 0..<4 { for byte in 0..<yStride where row >= 2 || byte >= 2 * size {
                    XCTAssertEqual(y[row * yStride + byte], beforeY[row * yStride + byte])
                } }
                XCTAssertEqual(uv, beforeUV, "Neutral graphics keep chroma and row padding unchanged")
            }
        }
    }

    func testFullyTransparentP010DoesNotChangeAnyBits() {
        let raster = ScoreBlendRaster(bgra: Data(repeating: 0, count: 3 * 3 * 4), width: 3, height: 3, scoreWidth: 1, color: hdr[0])
        var y = Data(repeating: 57, count: 32), uv = Data(repeating: 211, count: 16)
        let beforeY = y, beforeUV = uv
        y.withUnsafeMutableBytes { yp in uv.withUnsafeMutableBytes { cp in
            raster.blend(y: yp.baseAddress!, yStride: 8, uv: cp.baseAddress!, uvStride: 8,
                         frameWidth: 4, frameHeight: 4, tenBit: true, pointOpacity: 1, scoreOpacity: 1)
        } }
        XCTAssertEqual(y, beforeY); XCTAssertEqual(uv, beforeUV)
    }

    func testColoredP010PatchAndFadeAgainstReferenceCodeValues() {
        // Independently evaluated BT.2100/BT.2020 reference vectors for sRGB
        // red at 203 nits, then red at 101.5 nits over video black.
        let expected = [[[392, 395, 715], [323, 418, 689]], [[392, 438, 608], [342, 446, 601]]]
        for (transferIndex, color) in hdr.enumerated() {
            for (fadeIndex, opacity) in [1.0, 0.5].enumerated() {
                let raster = ScoreBlendRaster(bgra: Data([0,0,255,255, 0,0,255,255, 0,0,255,255, 0,0,255,255]),
                    width: 2, height: 2, scoreWidth: 2, color: color)
                var y = [UInt16](repeating: 64 << 6, count: 4), uv = [UInt16](repeating: 512 << 6, count: 2)
                y.withUnsafeMutableBytes { yp in uv.withUnsafeMutableBytes { cp in
                    raster.blend(y: yp.baseAddress!, yStride: 4, uv: cp.baseAddress!, uvStride: 4,
                        frameWidth: 2, frameHeight: 2, tenBit: true, pointOpacity: opacity, scoreOpacity: opacity)
                } }
                let actual = [Int(y[0] >> 6), Int(uv[0] >> 6), Int(uv[1] >> 6)]
                XCTAssertEqual(actual, expected[transferIndex][fadeIndex])
                XCTAssertTrue(y.allSatisfy { $0 == y[0] })
            }
        }
    }
}
