package com.volleycut.nativeanalysis;

import android.content.Context;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.media.Image;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.net.Uri;
import android.os.Debug;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BooleanSupplier;

final class NativeVideoDecoder {
    private static final long CODEC_TIMEOUT_US = 10_000;

    private final Context context;

    NativeVideoDecoder(Context context) {
        this.context = context.getApplicationContext();
    }

    AnalysisTypes.VideoFeatures decode(
            Uri uri,
            AnalysisTypes.MediaInfo media,
            AnalysisTypes.Roi roi,
            double[] times,
            AnalysisTypes.ProgressListener progress,
            BooleanSupplier cancelled
    ) throws IOException {
        long pipelineStartedNanos = System.nanoTime();
        long threadCpuStartedNanos = Debug.threadCpuTimeNanos();
        NanoProfiler profiler = new NanoProfiler();
        MediaExtractor extractor = new MediaExtractor();
        MediaCodec codec = null;
        FeatureWorker featureWorker = null;
        try {
            long setupStarted = System.nanoTime();
            extractor.setDataSource(context, uri, null);
            int trackIndex = findTrack(extractor, "video/");
            if (trackIndex < 0) throw new IOException("The selected file has no video track");
            extractor.selectTrack(trackIndex);
            MediaFormat format = extractor.getTrackFormat(trackIndex);
            String mime = format.getString(MediaFormat.KEY_MIME);
            if (mime == null) throw new IOException("Video track has no MIME type");
            format.setInteger(MediaFormat.KEY_COLOR_FORMAT,
                    MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible);
            codec = MediaCodec.createDecoderByType(mime);
            String decoderName = codec.getName();
            boolean hardware = !codec.getCodecInfo().isSoftwareOnly();
            codec.configure(format, null, null, 0);
            codec.start();
            profiler.add("setup", System.nanoTime() - setupStarted);

            long videoStartedNanos = System.nanoTime();
            long lastProgressNanos = 0;
            float[] output = new float[times.length * FeatureSchema.FRAME.size()];
            featureWorker = new FeatureWorker(output);
            MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
            boolean inputEnded = false;
            boolean outputEnded = false;
            int target = 0;
            int decodedSourceFrames = 0;
            long previousPresentationUs = -1;
            long lastPresentationUs = -1;
            double sampleTimestampErrorTotalMs = 0;
            double sampleTimestampErrorMaxMs = 0;
            while (!outputEnded
                    && target < times.length
                    && decodedSourceFrames < FeatureSchema.BENCHMARK_SOURCE_FRAME_LIMIT) {
                if (cancelled.getAsBoolean()) throw new InterruptedExceptionAsIo();
                featureWorker.throwIfFailed();
                if (!inputEnded) {
                    long operationStarted = System.nanoTime();
                    int inputIndex = codec.dequeueInputBuffer(CODEC_TIMEOUT_US);
                    profiler.add("codec_input_dequeue", System.nanoTime() - operationStarted);
                    if (inputIndex >= 0) {
                        ByteBuffer input = codec.getInputBuffer(inputIndex);
                        if (input == null) throw new IOException("Video decoder returned no input buffer");
                        operationStarted = System.nanoTime();
                        int sampleSize = extractor.readSampleData(input, 0);
                        long sampleTime = sampleSize < 0 ? 0 : extractor.getSampleTime();
                        int sampleFlags = sampleSize < 0 ? 0 : extractor.getSampleFlags();
                        profiler.add("demux_read", System.nanoTime() - operationStarted);
                        operationStarted = System.nanoTime();
                        if (sampleSize < 0) {
                            codec.queueInputBuffer(inputIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                            inputEnded = true;
                        } else {
                            codec.queueInputBuffer(inputIndex, 0, sampleSize, sampleTime, sampleFlags);
                            profiler.add("codec_input_queue", System.nanoTime() - operationStarted);
                            operationStarted = System.nanoTime();
                            extractor.advance();
                            profiler.add("demux_advance", System.nanoTime() - operationStarted);
                            operationStarted = 0;
                        }
                        if (operationStarted != 0) profiler.add("codec_input_queue", System.nanoTime() - operationStarted);
                    }
                }

                long operationStarted = System.nanoTime();
                int outputIndex = codec.dequeueOutputBuffer(info, CODEC_TIMEOUT_US);
                profiler.add("codec_output_dequeue_wait", System.nanoTime() - operationStarted);
                if (outputIndex == MediaCodec.INFO_TRY_AGAIN_LATER) continue;
                if (outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) continue;
                if (outputIndex < 0) continue;
                boolean eos = (info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0;
                long presentationUs = info.presentationTimeUs;
                if (info.size > 0) {
                    decodedSourceFrames++;
                    previousPresentationUs = lastPresentationUs;
                    lastPresentationUs = presentationUs;
                }
                boolean sampleThisFrame = info.size > 0
                        && target < times.length
                        && presentationUs + 1_000 >= Math.round(times[target] * 1_000_000);
                if (sampleThisFrame) {
                    long sampleStarted = System.nanoTime();
                    operationStarted = System.nanoTime();
                    Image image = codec.getOutputImage(outputIndex);
                    profiler.add("output_image_acquire", System.nanoTime() - operationStarted);
                    if (image == null || image.getFormat() != ImageFormat.YUV_420_888) {
                        if (image != null) image.close();
                        throw new IOException(
                                "The device decoder did not expose YUV_420_888 output. "
                                        + "Decoder: " + decoderName
                        );
                    }
                    try {
                        operationStarted = System.nanoTime();
                        Mat rgba = imageToAnalysisRgba(image, roi, media.rotation());
                        profiler.add("yuv_crop_scale_color", System.nanoTime() - operationStarted);
                        try {
                            operationStarted = System.nanoTime();
                            featureWorker.submit(target, rgba, cancelled);
                            profiler.add("feature_queue_backpressure", System.nanoTime() - operationStarted);
                            rgba = null;
                        } finally {
                            if (rgba != null) rgba.release();
                        }
                    } finally {
                        image.close();
                    }
                    double timestampErrorMs = Math.abs(
                            presentationUs / 1000.0 - times[target] * 1000.0
                    );
                    sampleTimestampErrorTotalMs += timestampErrorMs;
                    sampleTimestampErrorMaxMs = Math.max(sampleTimestampErrorMaxMs, timestampErrorMs);
                    target++;
                    long now = System.nanoTime();
                    if (now - lastProgressNanos >= 250_000_000L || target == times.length) {
                        lastProgressNanos = now;
                        double elapsedSeconds = (now - videoStartedNanos) / 1_000_000_000.0;
                        double framesPerSecond = target / Math.max(elapsedSeconds, 1e-9);
                        double generatedVideoSeconds = target / (double) FeatureSchema.ANALYSIS_FPS;
                        double realtimeRatio = generatedVideoSeconds / Math.max(elapsedSeconds, 1e-9);
                        double sourceFramesPerSecond = decodedSourceFrames
                                / Math.max(elapsedSeconds, 1e-9);
                        double etaSeconds = sourceFramesPerSecond > 0
                                ? (FeatureSchema.BENCHMARK_SOURCE_FRAME_LIMIT - decodedSourceFrames)
                                        / sourceFramesPerSecond
                                : Double.NaN;
                        Runtime runtime = Runtime.getRuntime();
                        progress.onPerformance(new AnalysisTypes.PerformanceStats(
                                target,
                                -1,
                                decodedSourceFrames,
                                generatedVideoSeconds,
                                elapsedSeconds,
                                framesPerSecond,
                                realtimeRatio,
                                etaSeconds,
                                runtime.totalMemory() - runtime.freeMemory(),
                                runtime.maxMemory()
                        ));
                        progress.onProgress(
                                "video",
                                decodedSourceFrames
                                        / (double) FeatureSchema.BENCHMARK_SOURCE_FRAME_LIMIT,
                                String.format(Locale.US,
                                        "Native decode + OpenCV · %.2f× real time · %.1f frames/s",
                                        realtimeRatio,
                                        framesPerSecond)
                        );
                    }
                    profiler.add("sampled_frame_total", System.nanoTime() - sampleStarted);
                }
                operationStarted = System.nanoTime();
                codec.releaseOutputBuffer(outputIndex, false);
                profiler.add("codec_output_release", System.nanoTime() - operationStarted);
                outputEnded = eos;
            }
            long operationStarted = System.nanoTime();
            featureWorker.finish(cancelled);
            profiler.add("feature_worker_finish_wait", System.nanoTime() - operationStarted);
            profiler.appendMilliseconds("", featureWorker.performanceMilliseconds());
            boolean sourceFrameLimitReached = decodedSourceFrames
                    >= FeatureSchema.BENCHMARK_SOURCE_FRAME_LIMIT;
            if (target == 0) {
                throw new IOException("Video ended after " + target + " of " + times.length + " analysis frames");
            }
            double analyzedDurationSeconds;
            if (sourceFrameLimitReached && lastPresentationUs >= 0) {
                long frameDurationUs = previousPresentationUs >= 0
                        ? Math.max(1, lastPresentationUs - previousPresentationUs)
                        : Math.round(1_000_000.0 / 30);
                analyzedDurationSeconds = Math.min(
                        media.durationSeconds(),
                        (lastPresentationUs + frameDurationUs) / 1_000_000.0
                );
            } else {
                analyzedDurationSeconds = media.durationSeconds();
            }
            profiler.add("video_pipeline_wall", System.nanoTime() - pipelineStartedNanos);
            double workerCpuMilliseconds = featureWorker.performanceMilliseconds()
                    .getOrDefault("feature_worker_thread_cpu", 0.0);
            return new AnalysisTypes.VideoFeatures(
                    Arrays.copyOf(output, target * FeatureSchema.FRAME.size()),
                    Arrays.copyOf(times, target),
                    analyzedDurationSeconds,
                    sourceFrameLimitReached,
                    decoderName,
                    hardware,
                    decodedSourceFrames,
                    (Debug.threadCpuTimeNanos() - threadCpuStartedNanos) / 1_000_000.0
                            + workerCpuMilliseconds,
                    target > 0 ? sampleTimestampErrorTotalMs / target : 0,
                    sampleTimestampErrorMaxMs,
                    profiler.milliseconds()
            );
        } finally {
            if (featureWorker != null) featureWorker.close();
            if (codec != null) {
                try { codec.stop(); } catch (RuntimeException ignored) {}
                codec.release();
            }
            extractor.release();
        }
    }

    private static Mat imageToAnalysisRgba(Image image, AnalysisTypes.Roi roi, int rotation) {
        Image.Plane[] planes = image.getPlanes();
        if (planes.length < 3) throw new IllegalArgumentException("YUV image has fewer than three planes");
        Rect crop = image.getCropRect();
        byte[] rgba = new byte[FeatureSchema.ANALYSIS_WIDTH * FeatureSchema.ANALYSIS_HEIGHT * 4];
        ByteBuffer yBuffer = planes[0].getBuffer().duplicate();
        ByteBuffer uBuffer = planes[1].getBuffer().duplicate();
        ByteBuffer vBuffer = planes[2].getBuffer().duplicate();
        int yRowStride = planes[0].getRowStride();
        int yPixelStride = planes[0].getPixelStride();
        int uRowStride = planes[1].getRowStride();
        int uPixelStride = planes[1].getPixelStride();
        int vRowStride = planes[2].getRowStride();
        int vPixelStride = planes[2].getPixelStride();

        int outputIndex = 0;
        for (int y = 0; y < FeatureSchema.ANALYSIS_HEIGHT; y++) {
            double displayV = roi.y() + (y + 0.5) / FeatureSchema.ANALYSIS_HEIGHT * roi.height();
            for (int x = 0; x < FeatureSchema.ANALYSIS_WIDTH; x++) {
                double displayU = roi.x() + (x + 0.5) / FeatureSchema.ANALYSIS_WIDTH * roi.width();
                double sourceU;
                double sourceV;
                switch (rotation) {
                    case 90 -> { sourceU = displayV; sourceV = 1 - displayU; }
                    case 180 -> { sourceU = 1 - displayU; sourceV = 1 - displayV; }
                    case 270 -> { sourceU = 1 - displayV; sourceV = displayU; }
                    default -> { sourceU = displayU; sourceV = displayV; }
                }
                int sourceX = clamp(crop.left + (int) Math.floor(sourceU * crop.width()), crop.left, crop.right - 1);
                int sourceY = clamp(crop.top + (int) Math.floor(sourceV * crop.height()), crop.top, crop.bottom - 1);
                int yValue = yBuffer.get(sourceY * yRowStride + sourceX * yPixelStride) & 0xff;
                int chromaX = sourceX / 2;
                int chromaY = sourceY / 2;
                int uValue = uBuffer.get(chromaY * uRowStride + chromaX * uPixelStride) & 0xff;
                int vValue = vBuffer.get(chromaY * vRowStride + chromaX * vPixelStride) & 0xff;

                int c = Math.max(0, yValue - 16);
                int d = uValue - 128;
                int e = vValue - 128;
                int red = clamp((298 * c + 409 * e + 128) >> 8, 0, 255);
                int green = clamp((298 * c - 100 * d - 208 * e + 128) >> 8, 0, 255);
                int blue = clamp((298 * c + 516 * d + 128) >> 8, 0, 255);
                rgba[outputIndex++] = (byte) red;
                rgba[outputIndex++] = (byte) green;
                rgba[outputIndex++] = (byte) blue;
                rgba[outputIndex++] = (byte) 255;
            }
        }
        Mat result = new Mat(FeatureSchema.ANALYSIS_HEIGHT, FeatureSchema.ANALYSIS_WIDTH, CvType.CV_8UC4);
        result.put(0, 0, rgba);
        return result;
    }

    static int findTrack(MediaExtractor extractor, String prefix) {
        for (int index = 0; index < extractor.getTrackCount(); index++) {
            String mime = extractor.getTrackFormat(index).getString(MediaFormat.KEY_MIME);
            if (mime != null && mime.startsWith(prefix)) return index;
        }
        return -1;
    }

    private static int clamp(int value, int lower, int upper) {
        return Math.max(lower, Math.min(upper, value));
    }

    private static final class FeatureWorker implements AutoCloseable {
        private static final int QUEUE_CAPACITY = 2;
        private static final SampleTask FINISH = new SampleTask(-1, null);

        private final float[] output;
        private final ArrayBlockingQueue<SampleTask> queue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
        private final AtomicReference<Throwable> failure = new AtomicReference<>();
        private final NanoProfiler profiler = new NanoProfiler();
        private final VisualFeatureExtractor extractor = new VisualFeatureExtractor(
                FeatureSchema.ANALYSIS_WIDTH,
                FeatureSchema.ANALYSIS_HEIGHT
        );
        private final Thread thread;
        private volatile boolean closeRequested;

        FeatureWorker(float[] output) {
            this.output = output;
            thread = new Thread(this::run, "VolleyCut-OpenCV");
            thread.start();
        }

        void submit(int rowIndex, Mat rgba, BooleanSupplier cancelled) throws IOException {
            SampleTask task = new SampleTask(rowIndex, rgba);
            try {
                while (true) {
                    throwIfFailed();
                    if (cancelled.getAsBoolean()) throw new InterruptedExceptionAsIo();
                    if (queue.offer(task, 10, TimeUnit.MILLISECONDS)) return;
                }
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                throw new IOException("Interrupted while waiting for OpenCV feature worker", interrupted);
            }
        }

        void finish(BooleanSupplier cancelled) throws IOException {
            try {
                while (thread.isAlive()) {
                    throwIfFailed();
                    if (cancelled.getAsBoolean()) throw new InterruptedExceptionAsIo();
                    if (queue.offer(FINISH, 10, TimeUnit.MILLISECONDS)) break;
                }
                while (thread.isAlive()) {
                    thread.join(25);
                    if (cancelled.getAsBoolean()) throw new InterruptedExceptionAsIo();
                }
                throwIfFailed();
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                throw new IOException("Interrupted while finishing OpenCV feature worker", interrupted);
            }
        }

        void throwIfFailed() throws IOException {
            Throwable error = failure.get();
            if (error == null) return;
            if (error instanceof IOException io) throw io;
            throw new IOException("OpenCV feature worker failed", error);
        }

        Map<String, Double> performanceMilliseconds() {
            return profiler.milliseconds();
        }

        private void run() {
            long wallStarted = System.nanoTime();
            long cpuStarted = Debug.threadCpuTimeNanos();
            try {
                while (true) {
                    SampleTask task = queue.take();
                    if (task == FINISH) break;
                    try {
                        long operationStarted = System.nanoTime();
                        float[] row = extractor.extract(task.rgba());
                        profiler.add("opencv_feature_call", System.nanoTime() - operationStarted);
                        operationStarted = System.nanoTime();
                        System.arraycopy(
                                row, 0, output,
                                task.rowIndex() * FeatureSchema.FRAME.size(), row.length
                        );
                        profiler.add("feature_matrix_copy", System.nanoTime() - operationStarted);
                    } finally {
                        task.release();
                    }
                }
            } catch (Throwable error) {
                if (!closeRequested || !(error instanceof InterruptedException)) {
                    failure.compareAndSet(null, error);
                }
            } finally {
                SampleTask pending;
                while ((pending = queue.poll()) != null) pending.release();
                extractor.close();
                profiler.appendMilliseconds("opencv/", extractor.performanceMilliseconds());
                profiler.add("feature_worker_thread_cpu", Debug.threadCpuTimeNanos() - cpuStarted);
                profiler.add("feature_worker_wall", System.nanoTime() - wallStarted);
            }
        }

        @Override
        public void close() {
            closeRequested = true;
            if (thread.isAlive()) thread.interrupt();
            try {
                thread.join();
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
            }
            SampleTask pending;
            while ((pending = queue.poll()) != null) pending.release();
        }
    }

    private record SampleTask(int rowIndex, Mat rgba) {
        void release() {
            if (rgba != null) rgba.release();
        }
    }

    private static final class InterruptedExceptionAsIo extends IOException {
        InterruptedExceptionAsIo() {
            super("Analysis cancelled");
        }
    }
}
