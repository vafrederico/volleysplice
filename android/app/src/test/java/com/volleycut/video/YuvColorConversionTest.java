package com.volleycut.video;

import static com.volleycut.video.YuvColorConversion.*;
import static org.junit.Assert.*;

import org.junit.Test;

public final class YuvColorConversionTest {
    @Test public void fullRangeRetainsNeutralBlackShadowsMidGrayAndWhite() {
        for (int matrix : new int[]{BT601_PAL, BT601_NTSC, BT709, BT2020}) {
            YuvColorConversion color = of(matrix, FULL);
            for (int gray : new int[]{0, 1, 16, 64, 128, 235, 254, 255}) {
                assertEquals(gray * 0x010101, color.rgb(gray, 128, 128));
            }
        }
    }

    @Test public void limitedRangeExpandsNominalBlackAndWhiteAndClipsFootroom() {
        for (int matrix : new int[]{BT601_NTSC, BT709, BT2020}) {
            YuvColorConversion color = of(matrix, LIMITED);
            assertEquals(0, color.rgb(16, 128, 128));
            assertEquals(0, color.rgb(0, 128, 128));
            assertEquals(0xffffff, color.rgb(235, 128, 128));
            assertEquals(0xffffff, color.rgb(255, 128, 128));
            assertEquals(0x828282, color.rgb(128, 128, 128));
        }
    }

    @Test public void bt709NominalColorBarPrimariesPreserveRgbOrder() {
        // Quantized Y'CbCr values for nominal red, green, blue bars. Quantization
        // can move reconstructed channels by one or two levels; RGB ordering may not.
        assertRgbClose(0xff0000, of(BT709, FULL).rgb(54, 99, 255), 2);
        assertRgbClose(0x00ff00, of(BT709, FULL).rgb(182, 30, 12), 2);
        assertRgbClose(0x0000ff, of(BT709, FULL).rgb(18, 255, 116), 2);
        assertRgbClose(0xff0000, of(BT709, LIMITED).rgb(63, 102, 240), 2);
        assertRgbClose(0x00ff00, of(BT709, LIMITED).rgb(173, 42, 26), 2);
        assertRgbClose(0x0000ff, of(BT709, LIMITED).rgb(32, 240, 118), 2);
    }

    @Test public void bt601NominalColorBarPrimariesUseTheirOwnMatrix() {
        assertRgbClose(0xff0000, of(BT601_NTSC, FULL).rgb(76, 85, 255), 2);
        assertRgbClose(0x00ff00, of(BT601_NTSC, FULL).rgb(150, 44, 21), 2);
        assertRgbClose(0x0000ff, of(BT601_NTSC, FULL).rgb(29, 255, 107), 2);
        assertRgbClose(0xff0000, of(BT601_NTSC, LIMITED).rgb(81, 90, 240), 2);
        assertRgbClose(0x00ff00, of(BT601_NTSC, LIMITED).rgb(145, 54, 34), 2);
        assertRgbClose(0x0000ff, of(BT601_NTSC, LIMITED).rgb(41, 240, 110), 2);
        assertNotEquals(of(BT709, FULL).rgb(54, 99, 255), of(BT601_NTSC, FULL).rgb(54, 99, 255));
    }

    @Test public void rangeIsAppliedBeforeFinalRgbClipping() {
        // Below-black luma is not pre-clamped before chroma is added.
        assertRgbClose(0xb80000, of(BT601_NTSC, LIMITED).rgb(0, 128, 255), 1);
    }

    @Test public void decoderRangeAndMatrixOverrideTrackEvenIfDecoderConvertedThem() {
        Profile profile = resolve(BT601_PAL, LIMITED, BT709, FULL);
        assertEquals(BT601_PAL, profile.standard());
        assertEquals(LIMITED, profile.range());
        assertEquals("decoder-output", profile.standardSource());
        assertEquals("decoder-output", profile.rangeSource());
    }

    @Test public void missingOutputFieldsFallBackIndividuallyToExtractor() {
        Profile profile = resolve(BT709, 0, BT601_NTSC, FULL);
        assertEquals(BT709, profile.standard());
        assertEquals(FULL, profile.range());
        assertEquals("decoder-output", profile.standardSource());
        assertEquals("extractor-track", profile.rangeSource());
        Profile missingMatrix = resolve(0, LIMITED, BT709, FULL);
        assertEquals(BT709, missingMatrix.standard());
        assertEquals(LIMITED, missingMatrix.range());
        assertEquals("extractor-track", missingMatrix.standardSource());
    }

    @Test public void unspecifiedMetadataRetainsAnExplicitLegacyFallback() {
        Profile profile = resolve(0, 0, 0, 0);
        assertEquals(BT601_NTSC, profile.standard());
        assertEquals(LIMITED, profile.range());
        assertEquals("assumed-bt601", profile.standardSource());
        assertEquals("assumed-limited", profile.rangeSource());
    }

    @Test public void unsupportedExplicitMetadataIsNotSilentlyReplacedByTrackMetadata() {
        assertThrows(IllegalArgumentException.class, () -> resolve(99, LIMITED, BT709, FULL));
        assertThrows(IllegalArgumentException.class, () -> resolve(BT709, 99, BT709, FULL));
    }

    private static void assertRgbClose(int expected, int actual, int tolerance) {
        for (int shift : new int[]{16, 8, 0}) {
            int e = (expected >> shift) & 255, a = (actual >> shift) & 255;
            assertTrue("RGB channel shift " + shift + ": expected " + e + ", got " + a,
                    Math.abs(e - a) <= tolerance);
        }
    }
}
