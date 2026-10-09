@preconcurrency import AVFoundation
import Foundation

/// A source's color description controls both AVFoundation and CPU graphics.
/// Unknown descriptions fail explicitly rather than silently tone-map HDR.
struct VideoExportColor {
    let primaries: String
    let transfer: String
    let matrix: String
    let space: ScoreColorSpace
    var isHDR: Bool { space.isHDR }
    var preset: String { isHDR ? AVAssetExportPresetHEVCHighestQuality : AVAssetExportPresetHighestQuality }

    init(primaries: String?, transfer: String?, matrix: String?, fullRange: Bool = false) throws {
        guard !fullRange else {
            throw ProjectError.invalid("Full-range video export with preserved range is currently unsupported. Use a video-range recording.")
        }
        guard let primaries, let transfer, let matrix else {
            throw ProjectError.invalid("The recording is missing color tags required to preserve its color space. Export without scores or use a tagged recording.")
        }
        let p: ScoreColorSpace.Primaries
        let ebuPrimaries = kCVImageBufferColorPrimaries_EBU_3213 as String
        switch primaries {
        case AVVideoColorPrimaries_ITU_R_709_2: p = .bt709
        case AVVideoColorPrimaries_ITU_R_2020: p = .bt2020
        case AVVideoColorPrimaries_P3_D65: p = .p3D65
        case AVVideoColorPrimaries_SMPTE_C: p = .smpteC
        case ebuPrimaries: p = .ebu3213
        default: throw ProjectError.invalid("The recording's color primaries are unsupported for score export")
        }
        let t: ScoreColorSpace.Transfer
        let srgbTransfer = kCVImageBufferTransferFunction_sRGB as String
        switch transfer {
        case AVVideoTransferFunction_ITU_R_709_2: t = .bt709
        case srgbTransfer: t = .sRGB
        case AVVideoTransferFunction_ITU_R_2100_HLG: t = .hlg
        case AVVideoTransferFunction_SMPTE_ST_2084_PQ: t = .pq
        default: throw ProjectError.invalid("The recording's transfer function is unsupported for score export")
        }
        let m: ScoreColorSpace.Matrix
        switch matrix {
        case AVVideoYCbCrMatrix_ITU_R_709_2: m = .bt709
        case AVVideoYCbCrMatrix_ITU_R_601_4: m = .bt601
        case AVVideoYCbCrMatrix_ITU_R_2020: m = .bt2020
        default: throw ProjectError.invalid("The recording's color matrix is unsupported for score export")
        }
        // The HLG OOTF uses BT.2100's BT.2020 luminance coefficients.
        guard t != .hlg || p == .bt2020 else { throw ProjectError.invalid("HLG score export requires BT.2020 primaries") }
        self.primaries = primaries; self.transfer = transfer; self.matrix = matrix
        space = ScoreColorSpace(primaries: p, transfer: t, matrix: m)
    }

    func verify(_ format: CMFormatDescription) throws {
        let actual = [CMFormatDescriptionGetExtension(format, extensionKey: kCMFormatDescriptionExtension_ColorPrimaries) as? String,
                      CMFormatDescriptionGetExtension(format, extensionKey: kCMFormatDescriptionExtension_TransferFunction) as? String,
                      CMFormatDescriptionGetExtension(format, extensionKey: kCMFormatDescriptionExtension_YCbCrMatrix) as? String]
        let fullRange = CMFormatDescriptionGetExtension(format, extensionKey: kCMFormatDescriptionExtension_FullRangeVideo) as? Bool ?? false
        let atoms = CMFormatDescriptionGetExtension(format, extensionKey: kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms) as? [String: Any]
        let hvcc = atoms?["hvcC"] as? Data
        let main10 = hvcc.map { $0.count >= 23 && $0[0] == 1 && ($0[17] & 7) == 2 && ($0[18] & 7) == 2 } ?? false
        guard actual == [primaries, transfer, matrix], !fullRange,
              !isHDR || (CMFormatDescriptionGetMediaSubType(format) == kCMVideoCodecType_HEVC && main10) else {
            throw ProjectError.invalid("Export verification failed: the encoder changed the recording's color space")
        }
    }
}
