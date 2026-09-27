package com.volleycut.video;

import static org.junit.Assert.*;

import android.content.Context;
import android.os.Debug;
import android.os.SystemClock;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Exact Java/native equivalence plus an isolated, non-gating timing receipt. */
@RunWith(AndroidJUnit4.class)
public final class NativeYuvAreaInstrumentedTest {
    private static volatile byte[] sink;

    @BeforeClass public static void requireNativeBackend() {
        assertTrue("Packaged JNI kernel loads on this device", YuvAreaResampler.nativeAvailable());
    }

    @Test public void nativeMatchesJavaAcrossMatricesRotationsStridesAndFractionalCrops() {
        for (int pixelStride : new int[]{1, 2}) {
            Fixture fixture = new Fixture(386, 218, pixelStride);
            for (int rotation : new int[]{0, 90, 180, 270}) {
                for (int standard : new int[]{YuvColorConversion.BT601_NTSC,
                        YuvColorConversion.BT709, YuvColorConversion.BT2020}) {
                    for (int range : new int[]{YuvColorConversion.FULL, YuvColorConversion.LIMITED}) {
                        YuvColorConversion color = YuvColorConversion.of(standard, range);
                        compare(new YuvAreaResampler(3, 5, 378, 208, rotation,
                                .071, .113, .819, .731, 97, 53), fixture, color);
                        compare(new YuvAreaResampler(2, 2, 380, 212, rotation,
                                0, 0, 1, 1, rotation % 180 == 0 ? 190 : 106,
                                rotation % 180 == 0 ? 106 : 190), fixture, color);
                    }
                }
            }
        }
    }

    @Test public void highResolutionTenAndTwentyToOneMatchJavaExactly() {
        for (int scale : new int[]{10, 20}) {
            Fixture fixture = new Fixture(192 * scale, 108 * scale, 2);
            compare(new YuvAreaResampler(0, 0, fixture.width, fixture.height, 0,
                    0, 0, 1, 1, 192, 108), fixture,
                    YuvColorConversion.of(YuvColorConversion.BT709, YuvColorConversion.LIMITED));
        }
    }

    @Test public void directBufferLimitsAreCheckedBeforeNativeReads() {
        Fixture fixture = new Fixture(32, 16, 2);
        YuvAreaResampler plan = new YuvAreaResampler(0, 0, 32, 16, 0, 0, 0, 1, 1, 8, 4);
        fixture.y.limit(fixture.y.limit() - 1);
        assertThrows(IllegalArgumentException.class, () -> fixture.convert(plan, true,
                YuvColorConversion.of(YuvColorConversion.BT709, YuvColorConversion.FULL)));
    }

    @Test public void measureJavaAndNativeAreaCost() throws Exception {
        Fixture fixture = new Fixture(1920, 1080, 2);
        YuvColorConversion color = YuvColorConversion.of(YuvColorConversion.BT709, YuvColorConversion.LIMITED);
        YuvAreaResampler plan = new YuvAreaResampler(0, 0, 1920, 1080, 0, 0, 0, 1, 1, 192, 108);
        byte[] expected = fixture.convert(plan, false, color).clone();
        for (int i = 0; i < 3; i++) {
            sink = fixture.convert(plan, false, color);
            sink = fixture.convert(plan, true, color);
        }
        JSONArray samples = new JSONArray();
        double[] javaWall = new double[3], nativeWall = new double[3];
        for (int repetition = 0; repetition < 3; repetition++) {
            for (int order = 0; order < 2; order++) {
                boolean useNative = ((repetition + order) & 1) != 0;
                long cpu = Debug.threadCpuTimeNanos(), start = SystemClock.elapsedRealtimeNanos(), end;
                int frames = 0;
                do {
                    sink = fixture.convert(plan, useNative, color);
                    frames++;
                    end = SystemClock.elapsedRealtimeNanos();
                } while (end - start < 750_000_000L || frames < 2);
                long cpuElapsed = Debug.threadCpuTimeNanos() - cpu;
                assertArrayEquals("Timed output retains exact repaired pixels", expected, sink);
                double wall = (end - start) / (frames * 1e6);
                (useNative ? nativeWall : javaWall)[repetition] = wall;
                samples.put(new JSONObject().put("backend", useNative ? "native" : "java")
                        .put("repetition", repetition).put("frames", frames)
                        .put("wallMsPerFrame", wall).put("cpuMsPerFrame", cpuElapsed / (frames * 1e6)));
            }
        }
        Arrays.sort(javaWall); Arrays.sort(nativeWall);
        JSONObject report = new JSONObject().put("schema", "native-area-kernel-timing-v1")
                .put("sourceWidth", 1920).put("sourceHeight", 1080).put("outputWidth", 192).put("outputHeight", 108)
                .put("fixture", "deterministic-strided-sliced-yuv420")
                .put("exactPixels", true).put("samples", samples)
                .put("javaMedianMs", javaWall[1]).put("nativeMedianMs", nativeWall[1])
                .put("speedup", javaWall[1] / nativeWall[1]);
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        File directory = new File(context.getFilesDir(), "benchmark");
        assertTrue(directory.isDirectory() || directory.mkdirs());
        try (FileOutputStream output = new FileOutputStream(new File(directory, "native-area-timing.json"))) {
            output.write((report.toString(2) + "\n").getBytes(StandardCharsets.UTF_8));
        }
    }

    private static void compare(YuvAreaResampler plan, Fixture fixture, YuvColorConversion color) {
        int yp = fixture.y.position(), up = fixture.u.position(), vp = fixture.v.position();
        byte[] expected = fixture.convert(plan, false, color).clone();
        assertArrayEquals(expected, fixture.convert(plan, true, color));
        assertEquals(yp, fixture.y.position()); assertEquals(up, fixture.u.position()); assertEquals(vp, fixture.v.position());
    }

    private static final class Fixture {
        final int width, height, yStride, uStride, vStride, chromaPixel;
        final ByteBuffer y, u, v;
        Fixture(int width, int height, int chromaPixel) {
            this.width = width; this.height = height; this.chromaPixel = chromaPixel;
            yStride = width + 13; uStride = ((width + 1) / 2) * chromaPixel + 9;
            vStride = uStride + 5;
            y = buffer(yStride * height); u = buffer(uStride * ((height + 1) / 2));
            v = buffer(vStride * ((height + 1) / 2));
            for (int row = 0; row < height; row++) for (int col = 0; col < width; col++) {
                y.put(row * yStride + col, (byte) ((col * 37 + row * 73) & 255));
            }
            for (int row = 0; row < (height + 1) / 2; row++) for (int col = 0; col < (width + 1) / 2; col++) {
                u.put(row * uStride + col * chromaPixel, (byte) ((col * 47 + row * 13) & 255));
                v.put(row * vStride + col * chromaPixel, (byte) ((col * 11 + row * 61) & 255));
            }
            y.limit((height - 1) * yStride + width);
            u.limit(((height - 1) / 2) * uStride + ((width - 1) / 2) * chromaPixel + 1);
            v.limit(((height - 1) / 2) * vStride + ((width - 1) / 2) * chromaPixel + 1);
            y.position(3); u.position(2); v.position(1);
        }
        byte[] convert(YuvAreaResampler plan, boolean nativeBackend, YuvColorConversion color) {
            return nativeBackend ? plan.convert(y, yStride, 1, u, uStride, chromaPixel, v, vStride, chromaPixel, color)
                    : plan.convertJava(y, yStride, 1, u, uStride, chromaPixel, v, vStride, chromaPixel, color);
        }
        private static ByteBuffer buffer(int size) {
            ByteBuffer storage = ByteBuffer.allocateDirect(size + 19);
            storage.position(7); storage.limit(7 + size);
            return storage.slice();
        }
    }
}
