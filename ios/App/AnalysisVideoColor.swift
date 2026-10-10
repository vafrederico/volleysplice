import AVFoundation

/// Model inputs are SDR. Request color conversion together with 8-bit decoding;
/// requesting NV12 alone does not define HDR-to-SDR conversion.
enum AnalysisVideoColor {
    static func outputSettings(isHDR: Bool) -> [String: Any] {
        var settings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        ]
        if isHDR {
            settings[AVVideoColorPropertiesKey] = [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2
            ]
        }
        return settings
    }

    static func outputSettings(for track: AVAssetTrack) async throws -> [String: Any] {
        let formats = try await track.load(.formatDescriptions)
        let hdr = formats.contains { format in
            let transfer = CMFormatDescriptionGetExtension(format, extensionKey: kCMFormatDescriptionExtension_TransferFunction) as? String
            return transfer == AVVideoTransferFunction_ITU_R_2100_HLG || transfer == AVVideoTransferFunction_SMPTE_ST_2084_PQ
        }
        return outputSettings(isHDR: hdr)
    }
}
