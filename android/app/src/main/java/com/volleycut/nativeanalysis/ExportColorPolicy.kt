package com.volleycut.nativeanalysis

import android.content.Context
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import androidx.annotation.OptIn
import androidx.media3.common.C
import androidx.media3.common.ColorInfo
import androidx.media3.common.Format
import androidx.media3.common.MimeTypes
import androidx.media3.common.util.MediaFormatUtil
import androidx.media3.common.util.UnstableApi
import androidx.media3.muxer.Muxer
import androidx.media3.muxer.MuxerException
import androidx.media3.transformer.EncoderUtil

/** Preserve source primaries, transfer and range, including HDR during score compositing. */
@OptIn(UnstableApi::class)
internal data class ExportColorPolicy(val color: ColorInfo) {
    val isHdr: Boolean get() = ColorInfo.isTransferHdr(color)
    val outputMimeType: String get() = if (isHdr) MimeTypes.VIDEO_H265 else MimeTypes.VIDEO_H264

    fun requireHdrEncoder() {
        check(!isHdr || EncoderUtil.getSupportedEncodersForHdrEditing(outputMimeType, color).isNotEmpty()) {
            "This device cannot export this video in its original HDR color space. " +
                "Use a device with HDR video encoding support."
        }
    }

    // MediaCodec often omits bit depth; its HDR encoder/profile capability check
    // still requires Main10. Reject an explicitly reported 8-bit output.
    fun matches(output: ColorInfo?): Boolean = output != null &&
        color.colorSpace == output.colorSpace && color.colorTransfer == output.colorTransfer &&
        color.colorRange == output.colorRange && (!isHdr || output.lumaBitdepth != 8)

    /** Runs on the actual encoder/muxer format, so a framework fallback cannot publish SDR. */
    fun preservingMuxer(factory: Muxer.Factory): Muxer.Factory = object : Muxer.Factory by factory {
        override fun create(path: String): Muxer {
            val muxer = factory.create(path)
            return object : Muxer by muxer {
                override fun addTrack(format: Format): Int {
                    if (MimeTypes.isVideo(format.sampleMimeType) && !matches(format.colorInfo)) {
                        throw MuxerException(
                            "The encoder could not preserve the video's original color space.",
                            IllegalStateException("Export color characteristics changed"),
                        )
                    }
                    return muxer.addTrack(format)
                }
            }
        }
    }

    companion object {
        fun read(context: Context, source: Uri): ExportColorPolicy {
            val extractor = MediaExtractor()
            try {
                extractor.setDataSource(context, source, null)
                for (index in 0 until extractor.trackCount) {
                    val format = extractor.getTrackFormat(index)
                    if (format.getString(MediaFormat.KEY_MIME)?.startsWith("video/") != true) continue
                    val color = MediaFormatUtil.getColorInfo(format)
                    if (ColorInfo.isTransferHdr(color)) {
                        check(color != null && color.isDataSpaceValid && color.colorSpace == C.COLOR_SPACE_BT2020) {
                            "This HDR video's color metadata is unsupported."
                        }
                        // Media3 requests an HLG base layer from a Dolby Vision decoder.
                        // That preserves iPhone profile 8.4, but would change a PQ source.
                        check(format.getString(MediaFormat.KEY_MIME) != MimeTypes.VIDEO_DOLBY_VISION ||
                            color.colorTransfer == C.COLOR_TRANSFER_HLG) {
                            "This Dolby Vision format cannot be exported in its original color space."
                        }
                        return ExportColorPolicy(color)
                    }
                    // This is also Media3's default for recordings without complete SDR tags.
                    return ExportColorPolicy(
                        color?.takeIf { it.isDataSpaceValid } ?: ColorInfo.SDR_BT709_LIMITED,
                    )
                }
                error("The recording has no video track.")
            } finally {
                extractor.release()
            }
        }
    }
}
