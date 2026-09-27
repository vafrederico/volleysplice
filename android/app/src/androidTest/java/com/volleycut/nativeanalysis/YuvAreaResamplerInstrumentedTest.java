package com.volleycut.nativeanalysis;

import static org.junit.Assert.*;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.volleycut.video.YuvAreaResampler;
import com.volleycut.video.YuvColorConversion;
import java.nio.ByteBuffer;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/** Compares the deployed sampler to native OpenCV, not a second area implementation. */
@RunWith(AndroidJUnit4.class)
public final class YuvAreaResamplerInstrumentedTest {
    @BeforeClass public static void loadOpenCv() {
        assertTrue("OpenCV initializes", OpenCVLoader.initLocal());
    }

    @Test public void checkerboardDownsamplingMatchesOpenCvAreaAndRemovesAliasing() {
        Fixture fixture = new Fixture(768, 432, true);
        byte[] actual = compare(fixture, 0, 0, 768, 432, 0, 0, 0, 1, 1, 192, 108,
                YuvColorConversion.of(YuvColorConversion.BT709, YuvColorConversion.FULL));
        for (int index = 0; index < actual.length; index += 4) {
            assertEquals(128, actual[index] & 255);
            assertEquals(128, actual[index + 1] & 255);
            assertEquals(128, actual[index + 2] & 255);
        }
    }

    @Test public void fractionalAreaCropRotationAndClippedColorsMatchOpenCv() {
        Fixture fixture = new Fixture(386, 218, false);
        for (int rotation : new int[]{0, 90, 180, 270}) {
            for (int standard : new int[]{YuvColorConversion.BT601_NTSC,
                    YuvColorConversion.BT709, YuvColorConversion.BT2020}) {
                for (int range : new int[]{YuvColorConversion.FULL, YuvColorConversion.LIMITED}) {
                    compare(fixture, 3, 5, 378, 208, rotation,
                            .071, .113, .819, .731, 97, 53,
                            YuvColorConversion.of(standard, range));
                }
            }
        }
    }

    @Test public void integerTwentyToOneShrinkMatchesOpenCvWithoutFullRgbAllocationInSampler() {
        // Representative high-resolution shrink. The test reference deliberately
        // allocates full RGB; the app sampler keeps only its 192x108 result.
        compare(new Fixture(3840, 2160, false), 0, 0, 3840, 2160, 0,
                0, 0, 1, 1, 192, 108,
                YuvColorConversion.of(YuvColorConversion.BT709, YuvColorConversion.LIMITED));
    }

    private static byte[] compare(Fixture fixture, int cropLeft, int cropTop,
            int cropWidth, int cropHeight, int rotation,
            double roiX, double roiY, double roiWidth, double roiHeight,
            int outWidth, int outHeight, YuvColorConversion color) {
        YuvAreaResampler sampler = new YuvAreaResampler(cropLeft, cropTop, cropWidth, cropHeight,
                rotation, roiX, roiY, roiWidth, roiHeight, outWidth, outHeight);
        byte[] actual = sampler.convert(fixture.y, fixture.yStride, 1,
                fixture.u, fixture.chromaStride, 2, fixture.v, fixture.chromaStride, 2, color);

        byte[] sourceRgba = new byte[cropWidth * cropHeight * 4];
        for (int row = 0; row < cropHeight; row++) {
            for (int column = 0; column < cropWidth; column++) {
                int x = cropLeft + column, y = cropTop + row;
                int rgb = color.rgb(fixture.y.get(y * fixture.yStride + x) & 255,
                        fixture.u.get((y / 2) * fixture.chromaStride + (x / 2) * 2) & 255,
                        fixture.v.get((y / 2) * fixture.chromaStride + (x / 2) * 2) & 255);
                int index = (row * cropWidth + column) * 4;
                sourceRgba[index] = (byte) (rgb >>> 16);
                sourceRgba[index + 1] = (byte) (rgb >>> 8);
                sourceRgba[index + 2] = (byte) rgb;
                sourceRgba[index + 3] = (byte) 255;
            }
        }
        Mat source = new Mat(cropHeight, cropWidth, CvType.CV_8UC4);
        Mat displayed = new Mat(), resized = new Mat(), cropped = null;
        try {
            source.put(0, 0, sourceRgba);
            switch (rotation) {
                case 90 -> Core.rotate(source, displayed, Core.ROTATE_90_CLOCKWISE);
                case 180 -> Core.rotate(source, displayed, Core.ROTATE_180);
                case 270 -> Core.rotate(source, displayed, Core.ROTATE_90_COUNTERCLOCKWISE);
                default -> source.copyTo(displayed);
            }
            int left = Math.max(0, Math.min(displayed.cols() - 1, (int) Math.rint(roiX * displayed.cols())));
            int top = Math.max(0, Math.min(displayed.rows() - 1, (int) Math.rint(roiY * displayed.rows())));
            int right = Math.min(displayed.cols(), Math.max(left + 1, (int) Math.rint((roiX + roiWidth) * displayed.cols())));
            int bottom = Math.min(displayed.rows(), Math.max(top + 1, (int) Math.rint((roiY + roiHeight) * displayed.rows())));
            cropped = displayed.submat(new Rect(left, top, right - left, bottom - top));
            Imgproc.resize(cropped, resized, new Size(outWidth, outHeight), 0, 0, Imgproc.INTER_AREA);
            byte[] expected = new byte[actual.length];
            resized.get(0, 0, expected);
            int maxError = 0;
            for (int index = 0; index < actual.length; index++) {
                maxError = Math.max(maxError, Math.abs((actual[index] & 255) - (expected[index] & 255)));
                if (index % 4 == 3) assertEquals(255, actual[index] & 255);
            }
            // OpenCV accumulates fractional areas in float (and has specialized
            // integer kernels); its final uint8 tie rounding may differ by 1 LSB.
            assertTrue("INTER_AREA max channel error " + maxError + " at rotation " + rotation, maxError <= 1);
            return actual;
        } finally {
            if (cropped != null) cropped.release();
            resized.release();
            displayed.release();
            source.release();
        }
    }

    private static final class Fixture {
        final ByteBuffer y, u, v;
        final int yStride, chromaStride;

        Fixture(int width, int height, boolean checkerboard) {
            yStride = width + 13;
            chromaStride = ((width + 1) / 2) * 2 + 7;
            y = ByteBuffer.allocateDirect(yStride * height);
            u = ByteBuffer.allocateDirect(chromaStride * ((height + 1) / 2));
            v = ByteBuffer.allocateDirect(chromaStride * ((height + 1) / 2));
            for (int row = 0; row < height; row++) for (int col = 0; col < width; col++) {
                int value = checkerboard ? ((row + col) % 2) * 255 : (col * 37 + row * 73) & 255;
                y.put(row * yStride + col, (byte) value);
            }
            for (int row = 0; row < (height + 1) / 2; row++) for (int col = 0; col < (width + 1) / 2; col++) {
                int index = row * chromaStride + col * 2;
                u.put(index, (byte) (checkerboard ? 128 : (col * 47 + row * 13) & 255));
                v.put(index, (byte) (checkerboard ? 128 : (col * 11 + row * 61) & 255));
            }
        }
    }
}
