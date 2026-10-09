import XCTest
import AVFoundation
@testable import VolleySplice

final class VideoExportColorTests: XCTestCase {
    func testHDRPresetsAndUnsupportedDescriptions() throws {
        for transfer in [AVVideoTransferFunction_ITU_R_2100_HLG, AVVideoTransferFunction_SMPTE_ST_2084_PQ] {
            let policy = try VideoExportColor(primaries: AVVideoColorPrimaries_ITU_R_2020,
                transfer: transfer, matrix: AVVideoYCbCrMatrix_ITU_R_2020)
            XCTAssertTrue(policy.isHDR)
            XCTAssertEqual(policy.preset, AVAssetExportPresetHEVCHighestQuality)
        }
        let sdr = try VideoExportColor(primaries: AVVideoColorPrimaries_ITU_R_709_2,
            transfer: AVVideoTransferFunction_ITU_R_709_2, matrix: AVVideoYCbCrMatrix_ITU_R_709_2)
        XCTAssertFalse(sdr.isHDR)
        XCTAssertEqual(sdr.preset, AVAssetExportPresetHighestQuality)
        XCTAssertThrowsError(try VideoExportColor(primaries: nil, transfer: nil, matrix: nil))
        XCTAssertThrowsError(try VideoExportColor(primaries: AVVideoColorPrimaries_ITU_R_709_2,
            transfer: AVVideoTransferFunction_ITU_R_709_2, matrix: AVVideoYCbCrMatrix_ITU_R_709_2, fullRange: true))
        XCTAssertThrowsError(try VideoExportColor(primaries: AVVideoColorPrimaries_P3_D65,
            transfer: AVVideoTransferFunction_ITU_R_2100_HLG, matrix: AVVideoYCbCrMatrix_ITU_R_2020))
    }
}
