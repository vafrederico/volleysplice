package com.volleycut.video;

import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaFormat;

import java.io.IOException;

/** Decoder policy shared by feature extraction and independent neural frame sampling. */
public final class AnalysisVideoDecoder {
    private AnalysisVideoDecoder() {}

    public static boolean usesHlgBaseLayer(String mime, int profile, int standard, int transfer) {
        // Profile 8 with an HLG base layer is iPhone Dolby Vision 8.4. Do not
        // reinterpret other Dolby Vision profiles (especially profile 5) as HEVC.
        return MediaFormat.MIMETYPE_VIDEO_DOLBY_VISION.equals(mime)
                && profile == MediaCodecInfo.CodecProfileLevel.DolbyVisionProfileDvheSt
                && standard == MediaFormat.COLOR_STANDARD_BT2020
                && transfer == MediaFormat.COLOR_TRANSFER_HLG;
    }

    public static boolean isHdr(int transfer) {
        return transfer == MediaFormat.COLOR_TRANSFER_HLG
                || transfer == MediaFormat.COLOR_TRANSFER_ST2084;
    }

    public static MediaCodec create(MediaFormat format) throws IOException {
        String mime = format.getString(MediaFormat.KEY_MIME);
        if (mime == null) throw new IOException("Video track has no MIME type");
        if (usesHlgBaseLayer(mime, value(format, MediaFormat.KEY_PROFILE),
                value(format, MediaFormat.KEY_COLOR_STANDARD),
                value(format, MediaFormat.KEY_COLOR_TRANSFER))) {
            mime = MediaFormat.MIMETYPE_VIDEO_HEVC;
            format.setString(MediaFormat.KEY_MIME, mime);
            // Dolby Vision profile/level values belong to a different enum.
            // Preserve HEVC CSD and source color tags, but configure Main10.
            format.setInteger(MediaFormat.KEY_PROFILE,
                    MediaCodecInfo.CodecProfileLevel.HEVCProfileMain10);
            format.removeKey(MediaFormat.KEY_LEVEL);
        }
        if (isHdr(value(format, MediaFormat.KEY_COLOR_TRANSFER))) {
            // The models consume 8-bit SDR images; the exporter separately reads
            // the original HDR source. Request real decoder tone mapping rather
            // than interpreting truncated HLG/PQ samples as SDR pixels.
            format.setInteger(MediaFormat.KEY_COLOR_TRANSFER_REQUEST,
                    MediaFormat.COLOR_TRANSFER_SDR_VIDEO);
        }
        try {
            return MediaCodec.createDecoderByType(mime);
        } catch (IllegalArgumentException error) {
            throw new IOException("This device has no usable " + mime + " analysis decoder.", error);
        }
    }

    public static void requireConfiguration(MediaCodec codec, MediaFormat format) throws IOException {
        if (value(format, MediaFormat.KEY_COLOR_TRANSFER_REQUEST) == 0) return;
        if (value(codec.getInputFormat(), MediaFormat.KEY_COLOR_TRANSFER_REQUEST)
                != MediaFormat.COLOR_TRANSFER_SDR_VIDEO) {
            throw new IOException("This device cannot tone-map HDR video for analysis. Decoder: "
                    + codec.getName());
        }
    }

    public static void requireOutput(MediaFormat output, MediaFormat format) throws IOException {
        if (value(format, MediaFormat.KEY_COLOR_TRANSFER_REQUEST) == 0) return;
        if (value(output, MediaFormat.KEY_COLOR_TRANSFER) != MediaFormat.COLOR_TRANSFER_SDR_VIDEO) {
            throw new IOException("The HDR analysis decoder did not produce SDR frames.");
        }
    }

    private static int value(MediaFormat format, String key) {
        return format.getInteger(key, 0);
    }
}
