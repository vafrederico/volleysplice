package com.volleycut.nativeanalysis;
import com.volleycut.video.AnalysisVideoDecoder;

import android.media.MediaCodecInfo;
import android.media.MediaFormat;
import org.junit.Test;
import static org.junit.Assert.*;

public final class AnalysisVideoDecoderTest {
    @Test public void onlyVerifiedHlgProfileEightUsesHevcBaseLayer() {
        int profile = MediaCodecInfo.CodecProfileLevel.DolbyVisionProfileDvheSt;
        int standard = MediaFormat.COLOR_STANDARD_BT2020;
        int transfer = MediaFormat.COLOR_TRANSFER_HLG;
        String mime = MediaFormat.MIMETYPE_VIDEO_DOLBY_VISION;
        assertTrue(AnalysisVideoDecoder.usesHlgBaseLayer(mime, profile, standard, transfer));
        assertFalse(AnalysisVideoDecoder.usesHlgBaseLayer(mime, 0, standard, transfer));
        assertFalse(AnalysisVideoDecoder.usesHlgBaseLayer(mime,
                MediaCodecInfo.CodecProfileLevel.DolbyVisionProfileDvheStn, standard, transfer));
        assertFalse(AnalysisVideoDecoder.usesHlgBaseLayer(mime,
                MediaCodecInfo.CodecProfileLevel.DolbyVisionProfileDvheDtb, standard, transfer));
        assertFalse(AnalysisVideoDecoder.usesHlgBaseLayer(mime, profile, standard,
                MediaFormat.COLOR_TRANSFER_ST2084));
        assertFalse(AnalysisVideoDecoder.usesHlgBaseLayer(mime, profile,
                MediaFormat.COLOR_STANDARD_BT709, transfer));
        assertFalse(AnalysisVideoDecoder.usesHlgBaseLayer(
                MediaFormat.MIMETYPE_VIDEO_HEVC, profile, standard, transfer));
    }

    @Test public void hdrAnalysisRequiresToneMappingButSdrDoesNot() {
        assertTrue(AnalysisVideoDecoder.isHdr(MediaFormat.COLOR_TRANSFER_HLG));
        assertTrue(AnalysisVideoDecoder.isHdr(MediaFormat.COLOR_TRANSFER_ST2084));
        assertFalse(AnalysisVideoDecoder.isHdr(MediaFormat.COLOR_TRANSFER_SDR_VIDEO));
        assertFalse(AnalysisVideoDecoder.isHdr(0));
    }
}
