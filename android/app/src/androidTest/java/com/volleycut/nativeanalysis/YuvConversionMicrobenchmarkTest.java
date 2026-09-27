package com.volleycut.nativeanalysis;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import android.content.Context;
import android.os.Build;
import android.os.Debug;
import android.os.SystemClock;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import com.volleycut.video.YuvAreaResampler;
import com.volleycut.video.YuvColorConversion;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;

/**
 * Isolates the changed Java YUV conversion/resizing work from MediaCodec and inference.
 * It measures a reused synthetic frame, so its ratio is not an end-to-end speed claim.
 * Correctness against OpenCV is covered by YuvAreaResamplerInstrumentedTest.
 */
@RunWith(AndroidJUnit4.class)
public final class YuvConversionMicrobenchmarkTest {
    private static final int WIDTH = 1920, HEIGHT = 1080;
    private static final int OUTPUT_WIDTH = 192, OUTPUT_HEIGHT = 108;
    private static final int WARMUP_FRAMES = 3, REPETITIONS = 3;
    private static final long BATCH_WALL_NANOS = 750_000_000L;
    private static volatile byte[] outputSink;
    private static volatile long checksumSink;

    @Test public void comparePointAndAreaConversionCost() throws Exception {
        Fixture fixture = new Fixture();
        YuvColorConversion color = YuvColorConversion.of(
                YuvColorConversion.BT709, YuvColorConversion.LIMITED);
        YuvAreaResampler area = new YuvAreaResampler(0, 0, WIDTH, HEIGHT, 0,
                0, 0, 1, 1, OUTPUT_WIDTH, OUTPUT_HEIGHT);
        LegacyPointSampler point = new LegacyPointSampler(fixture);
        Converter pointConversion = () -> point.convert(fixture, color);
        Converter areaConversion = () -> area.convert(fixture.y, fixture.yStride, 1,
                fixture.u, fixture.chromaStride, 2,
                fixture.v, fixture.chromaStride, 2, color);

        for (int index = 0; index < WARMUP_FRAMES; index++) {
            consume(pointConversion.convert(), index);
            consume(areaConversion.convert(), index);
        }
        long pointChecksum = checksum(pointConversion.convert());
        long areaChecksum = checksum(areaConversion.convert());
        JSONArray pointSamples = new JSONArray(), areaSamples = new JSONArray();
        JSONArray executionOrder = new JSONArray();
        double[] pointWall = new double[REPETITIONS], areaWall = new double[REPETITIONS];
        double[] pointCpu = new double[REPETITIONS], areaCpu = new double[REPETITIONS];
        for (int index = 0; index < REPETITIONS; index++) {
            // Alternate first case to expose order/JIT/host scheduling effects in raw samples.
            Measurement pointResult, areaResult;
            if ((index & 1) == 0) {
                executionOrder.put("point,area");
                pointResult = measure(pointConversion, pointChecksum);
                areaResult = measure(areaConversion, areaChecksum);
            } else {
                executionOrder.put("area,point");
                areaResult = measure(areaConversion, areaChecksum);
                pointResult = measure(pointConversion, pointChecksum);
            }
            pointSamples.put(pointResult.json(index));
            areaSamples.put(areaResult.json(index));
            pointWall[index] = pointResult.wallMsPerFrame();
            areaWall[index] = areaResult.wallMsPerFrame();
            pointCpu[index] = pointResult.cpuMsPerFrame();
            areaCpu[index] = areaResult.cpuMsPerFrame();
        }

        JSONObject report = new JSONObject()
                .put("schemaVersion", 1)
                .put("benchmark", "yuv-point-versus-area-conversion")
                .put("fixture", "deterministic-synthetic-yuv420-strided-v1")
                .put("sourceWidth", WIDTH).put("sourceHeight", HEIGHT)
                .put("outputWidth", OUTPUT_WIDTH).put("outputHeight", OUTPUT_HEIGHT)
                .put("yRowStride", fixture.yStride)
                .put("chromaRowStride", fixture.chromaStride)
                .put("yPixelStride", 1).put("chromaPixelStride", 2)
                .put("rotationDegrees", 0).put("fullFrame", true)
                .put("colorStandard", "BT709").put("colorRange", "limited")
                .put("androidApi", Build.VERSION.SDK_INT)
                .put("supportedAbis", new JSONArray(Arrays.asList(Build.SUPPORTED_ABIS)))
                .put("warmupFramesPerCase", WARMUP_FRAMES)
                .put("checksumFramesPerCaseBeforeTiming", 1)
                .put("repetitions", REPETITIONS)
                .put("batchTargetWallMs", BATCH_WALL_NANOS / 1_000_000.0)
                .put("executionOrder", executionOrder)
                .put("point", summary(pointSamples, pointWall, pointCpu, pointChecksum))
                .put("area", summary(areaSamples, areaWall, areaCpu, areaChecksum))
                .put("areaOverPointMedianWallRatio", median(areaWall) / median(pointWall))
                .put("areaOverPointMedianThreadCpuRatio", median(areaCpu) / median(pointCpu))
                .put("measurementScope", "Repeated conversion to reused RGBA byte arrays; "
                        + "includes the same small volatile output consumption for both cases. "
                        + "Excludes fixture/plan allocation, MediaCodec, RGB Mat allocation/copy, "
                        + "OpenCV features, audio, embeddings, model inference and file output. "
                        + "The cached synthetic fixture and emulator CPU do not predict phone latency.")
                .put("legacyPointImplementation", "Preserved precomputed-offset loop from "
                        + "NativeVideoDecoder.YuvCropSampler, full-frame rotation-zero case");

        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        File directory = new File(context.getFilesDir(), "benchmark");
        assertTrue("Benchmark output directory", directory.isDirectory() || directory.mkdirs());
        File output = new File(directory, "yuv-conversion-microbenchmark.json");
        try (FileOutputStream stream = new FileOutputStream(output)) {
            stream.write((report.toString(2) + "\n").getBytes(StandardCharsets.UTF_8));
        }
        assertTrue("Benchmark result saved", output.length() > 0);
    }

    private static Measurement measure(Converter converter, long expectedChecksum) {
        long cpuStart = Debug.threadCpuTimeNanos();
        long wallStart = SystemClock.elapsedRealtimeNanos();
        long wallEnd;
        int frames = 0;
        byte[] result;
        do {
            result = converter.convert();
            consume(result, frames++);
            wallEnd = SystemClock.elapsedRealtimeNanos();
        } while (wallEnd - wallStart < BATCH_WALL_NANOS || frames < 2);
        long cpuElapsed = Debug.threadCpuTimeNanos() - cpuStart;
        long wallElapsed = wallEnd - wallStart;
        assertEquals("RGBA output size", OUTPUT_WIDTH * OUTPUT_HEIGHT * 4, result.length);
        assertEquals("Repeated conversion is deterministic", expectedChecksum, checksum(result));
        assertTrue("Valid wall-clock sample", wallElapsed > 0);
        assertTrue("Thread CPU timing supported", cpuElapsed > 0);
        Measurement measurement = new Measurement(frames, wallElapsed, cpuElapsed);
        assertTrue(Double.isFinite(measurement.wallMsPerFrame()));
        assertTrue(Double.isFinite(measurement.cpuMsPerFrame()));
        return measurement;
    }

    private static void consume(byte[] result, int iteration) {
        // Escapes the complete output and reads a changing position without a full-array
        // checksum inside the timer, which would disproportionately burden point sampling.
        outputSink = result;
        checksumSink += result[(iteration * 127) % result.length] & 255;
    }

    private static long checksum(byte[] bytes) {
        long result = 1;
        for (byte value : bytes) result = 31 * result + (value & 255);
        return result;
    }

    private static JSONObject summary(JSONArray samples, double[] wall, double[] cpu, long checksum)
            throws Exception {
        return new JSONObject().put("samples", samples)
                .put("medianWallMsPerFrame", median(wall))
                .put("medianThreadCpuMsPerFrame", median(cpu))
                .put("outputChecksum", Long.toHexString(checksum));
    }

    private static double median(double[] values) {
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        return sorted[sorted.length / 2];
    }

    private interface Converter { byte[] convert(); }

    private record Measurement(int frames, long wallNanos, long threadCpuNanos) {
        double wallMsPerFrame() { return wallNanos / (frames * 1_000_000.0); }
        double cpuMsPerFrame() { return threadCpuNanos / (frames * 1_000_000.0); }
        JSONObject json(int repetition) throws Exception {
            return new JSONObject().put("repetition", repetition).put("frames", frames)
                    .put("wallNanos", wallNanos).put("threadCpuNanos", threadCpuNanos)
                    .put("wallMsPerFrame", wallMsPerFrame())
                    .put("threadCpuMsPerFrame", cpuMsPerFrame());
        }
    }

    /** Matches the old loop and offset plan without requiring an android.media.Image. */
    private static final class LegacyPointSampler {
        final int[] yOffsets = new int[OUTPUT_WIDTH * OUTPUT_HEIGHT];
        final int[] uOffsets = new int[yOffsets.length], vOffsets = new int[yOffsets.length];
        final byte[] rgba = new byte[yOffsets.length * 4];

        LegacyPointSampler(Fixture fixture) {
            int index = 0;
            for (int row = 0; row < OUTPUT_HEIGHT; row++) {
                for (int column = 0; column < OUTPUT_WIDTH; column++) {
                    int sourceX = (int) Math.floor((column + .5) / OUTPUT_WIDTH * WIDTH);
                    int sourceY = (int) Math.floor((row + .5) / OUTPUT_HEIGHT * HEIGHT);
                    yOffsets[index] = sourceY * fixture.yStride + sourceX;
                    uOffsets[index] = (sourceY / 2) * fixture.chromaStride + (sourceX / 2) * 2;
                    vOffsets[index] = uOffsets[index];
                    index++;
                }
            }
        }

        byte[] convert(Fixture fixture, YuvColorConversion color) {
            int outputIndex = 0;
            for (int index = 0; index < yOffsets.length; index++) {
                int yValue = fixture.y.get(yOffsets[index]) & 255;
                int uValue = fixture.u.get(uOffsets[index]) & 255;
                int vValue = fixture.v.get(vOffsets[index]) & 255;
                int rgb = color.rgb(yValue, uValue, vValue);
                rgba[outputIndex++] = (byte) (rgb >> 16);
                rgba[outputIndex++] = (byte) (rgb >> 8);
                rgba[outputIndex++] = (byte) rgb;
                rgba[outputIndex++] = (byte) 255;
            }
            return rgba;
        }
    }

    private static final class Fixture {
        final int yStride = WIDTH + 16, chromaStride = WIDTH + 16;
        final ByteBuffer y = ByteBuffer.allocateDirect(yStride * HEIGHT);
        final ByteBuffer u = ByteBuffer.allocateDirect(chromaStride * (HEIGHT / 2));
        final ByteBuffer v = ByteBuffer.allocateDirect(chromaStride * (HEIGHT / 2));

        Fixture() {
            for (int row = 0; row < HEIGHT; row++) {
                for (int col = 0; col < WIDTH; col++) {
                    y.put(row * yStride + col, (byte) ((col * 37 + row * 73) & 255));
                }
            }
            for (int row = 0; row < HEIGHT / 2; row++) {
                for (int col = 0; col < WIDTH / 2; col++) {
                    int offset = row * chromaStride + col * 2;
                    u.put(offset, (byte) ((col * 47 + row * 13) & 255));
                    v.put(offset, (byte) ((col * 11 + row * 61) & 255));
                }
            }
        }
    }
}
