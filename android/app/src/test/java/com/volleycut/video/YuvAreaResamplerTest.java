package com.volleycut.video;

import static org.junit.Assert.*;

import java.nio.ByteBuffer;
import java.util.Arrays;
import org.junit.Test;

public final class YuvAreaResamplerTest {
    private static final YuvColorConversion GRAY = YuvColorConversion.of(
            YuvColorConversion.BT709, YuvColorConversion.FULL);

    @Test public void areaAveragingRemovesCheckerboardAliasing() {
        byte[] luma = new byte[64 * 32];
        for (int y = 0; y < 32; y++) for (int x = 0; x < 64; x++) {
            luma[y * 64 + x] = (byte) (((x + y) % 2 == 0) ? 0 : 255);
        }
        byte[] result = gray(luma, 64, 32, 0, 16, 8);
        for (int pixel = 0; pixel < 16 * 8; pixel++) assertPixel(result, pixel, 128, 128, 128);
    }

    @Test public void fractionalAreaWeightsIncludeBothBorderPixels() {
        byte[] result = gray(new byte[]{0, 50, 100, (byte) 150, (byte) 200}, 5, 1, 0, 2, 1);
        assertPixel(result, 0, 40, 40, 40);
        assertPixel(result, 1, 160, 160, 160);
    }

    @Test public void averagesClippedRgbRatherThanConvertingAverageYuv() {
        ByteBuffer y = ByteBuffer.wrap(new byte[]{0, (byte) 255});
        ByteBuffer u = ByteBuffer.wrap(new byte[]{0});
        ByteBuffer v = ByteBuffer.wrap(new byte[]{(byte) 255});
        byte[] result = plan(2, 1, 0, 1, 1).convert(y, 2, 1, u, 1, 1, v, 1, 1, GRAY);
        int black = GRAY.rgb(0, 0, 255), white = GRAY.rgb(255, 0, 255);
        int[] expected = new int[3];
        for (int channel = 0; channel < 3; channel++) {
            int shift = (2 - channel) * 8;
            expected[channel] = (int) Math.rint((((black >> shift) & 255)
                    + ((white >> shift) & 255)) / 2.0);
        }
        assertPixel(result, 0, expected[0], expected[1], expected[2]);
        int averagedYuv = GRAY.rgb(128, 0, 255);
        assertNotEquals(averagedYuv, (expected[0] << 16) | (expected[1] << 8) | expected[2]);
    }

    @Test public void allDisplayRotationsPreservePixelPlacement() {
        byte[] luma = {10, 20, 30, 40, 50, 60};
        int[][] expected = {{10,20,30,40,50,60}, {40,10,50,20,60,30},
                {60,50,40,30,20,10}, {30,60,20,50,10,40}};
        for (int quarter = 0; quarter < 4; quarter++) {
            boolean transpose = quarter % 2 != 0;
            byte[] result = gray(luma, 3, 2, quarter * 90, transpose ? 2 : 3, transpose ? 3 : 2);
            for (int pixel = 0; pixel < 6; pixel++) {
                int value = expected[quarter][pixel];
                assertPixel(result, pixel, value, value, value);
            }
        }
    }

    @Test public void cropUsesDisplayCoordinatesAndDesktopRoundToEven() {
        byte[] luma = {10, 20, 30, 40, 50, 60, 70, 80};
        // Displayed after 90 degrees: [50,10; 60,20; 70,30; 80,40].
        // x=0.25 rounds 0.5 to zero, right=0.75 rounds 1.5 to two.
        YuvAreaResampler sampler = new YuvAreaResampler(0, 0, 4, 2, 90,
                .25, .25, .5, .5, 2, 2);
        byte[] chroma = {(byte) 128, (byte) 128};
        byte[] result = sampler.convert(ByteBuffer.wrap(luma), 4, 1,
                ByteBuffer.wrap(chroma), 2, 1, ByteBuffer.wrap(chroma), 2, 1, GRAY);
        int[] expected = {60,20,70,30};
        for (int pixel = 0; pixel < 4; pixel++) {
            assertPixel(result, pixel, expected[pixel], expected[pixel], expected[pixel]);
        }
    }

    @Test public void nonzeroCropAndPaddedPixelStridesUseAbsolutePlaneCoordinates() {
        int width = 8, height = 6, yStride = 21, chromaStride = 11;
        byte[] luma = new byte[yStride * height], chroma = new byte[chromaStride * 3];
        Arrays.fill(luma, (byte) 250);
        Arrays.fill(chroma, (byte) 17);
        for (int y = 0; y < height; y++) for (int x = 0; x < width; x++) luma[y * yStride + x * 2] = (byte) (10 * y + x);
        for (int y = 0; y < 3; y++) for (int x = 0; x < 4; x++) chroma[y * chromaStride + x * 2] = (byte) 128;
        YuvAreaResampler sampler = new YuvAreaResampler(1, 1, 6, 4, 0, 0, 0, 1, 1, 3, 2);
        byte[] result = sampler.convert(ByteBuffer.wrap(luma), yStride, 2,
                ByteBuffer.wrap(chroma), chromaStride, 2, ByteBuffer.wrap(chroma), chromaStride, 2, GRAY);
        int[] expected = {16,18,20,36,38,40};
        for (int pixel = 0; pixel < 6; pixel++) {
            assertPixel(result, pixel, expected[pixel], expected[pixel], expected[pixel]);
        }
    }

    @Test public void repeatedFramesReuseThePlanWithoutRetainingPreviousPixels() {
        YuvAreaResampler sampler = plan(4, 2, 0, 2, 1);
        ByteBuffer y = ByteBuffer.allocate(8), chroma = ByteBuffer.wrap(new byte[]{(byte) 128, (byte) 128});
        byte[] first = sampler.convert(y, 4, 1, chroma, 2, 1, chroma, 2, 1, GRAY);
        assertPixel(first, 0, 0, 0, 0);
        for (int index = 0; index < 8; index++) y.put(index, (byte) 200);
        byte[] second = sampler.convert(y, 4, 1, chroma, 2, 1, chroma, 2, 1, GRAY);
        assertSame(first, second);
        assertPixel(second, 0, 200, 200, 200);
        assertPixel(second, 1, 200, 200, 200);
    }

    @Test public void rejectsInvalidGeometry() {
        assertThrows(IllegalArgumentException.class, () -> plan(4, 2, 45, 2, 1));
        assertThrows(IllegalArgumentException.class, () -> plan(4, 2, 0, 0, 1));
        assertThrows(IllegalArgumentException.class, () -> new YuvAreaResampler(
                0, 0, 4, 2, 0, Double.NaN, 0, 1, 1, 2, 1));
    }

    private static YuvAreaResampler plan(int width, int height, int rotation, int outWidth, int outHeight) {
        return new YuvAreaResampler(0, 0, width, height, rotation, 0, 0, 1, 1, outWidth, outHeight);
    }

    private static byte[] gray(byte[] luma, int width, int height, int rotation, int outWidth, int outHeight) {
        int chromaWidth = (width + 1) / 2;
        byte[] chroma = new byte[chromaWidth * ((height + 1) / 2)];
        Arrays.fill(chroma, (byte) 128);
        return plan(width, height, rotation, outWidth, outHeight).convert(ByteBuffer.wrap(luma), width, 1,
                ByteBuffer.wrap(chroma), chromaWidth, 1, ByteBuffer.wrap(chroma), chromaWidth, 1, GRAY);
    }

    private static void assertPixel(byte[] rgba, int pixel, int red, int green, int blue) {
        assertEquals(red, rgba[pixel * 4] & 255);
        assertEquals(green, rgba[pixel * 4 + 1] & 255);
        assertEquals(blue, rgba[pixel * 4 + 2] & 255);
        assertEquals(255, rgba[pixel * 4 + 3] & 255);
    }
}
