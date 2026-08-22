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
import android.os.Handler;
import android.os.HandlerThread;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BooleanSupplier;

final class NativeVideoDecoder {
    private static final long CODEC_TIMEOUT_US = 10_000;
    private static final int SAMPLE_PLAN_REORDER_LOOKAHEAD = 64;

    private final Context context;

    NativeVideoDecoder(Context context) {
        this.context = context.getApplicationContext();
    }

    AnalysisTypes.VideoFeatures decode(
            Uri uri,
            AnalysisTypes.MediaInfo media,
            AnalysisTypes.Roi roi,
            double[] times,
            int sourceFrameLimit,
            AnalysisTypes.VideoDecoderOptions decoderOptions,
            NativeFeatureCache.LoadedVisual cached,
            NativeFeatureCache.VisualWriter cacheWriter,
            AnalysisTypes.ProgressListener progress,
            BooleanSupplier cancelled
    ) throws IOException {
        long pipelineStartedNanos = System.nanoTime();
        long threadCpuStartedNanos = Debug.threadCpuTimeNanos();
        NanoProfiler profiler = new NanoProfiler();
        MediaExtractor extractor = new MediaExtractor();
        MediaCodec codec = null;
        FeatureWorker featureWorker = null;
        HandlerThread codecThread = null;
        try {
            long setupStarted = System.nanoTime();
            extractor.setDataSource(context, uri, null);
            int trackIndex = findTrack(extractor, "video/");
            if (trackIndex < 0) throw new IOException("The selected file has no video track");
            extractor.selectTrack(trackIndex);
            MediaFormat format = extractor.getTrackFormat(trackIndex);
            String mime = format.getString(MediaFormat.KEY_MIME);
            if (mime == null) throw new IOException("Video track has no MIME type");

            long operationStarted = System.nanoTime();
            SamplePlan samplePlan = buildSamplePlan(extractor, format, times, sourceFrameLimit);
            profiler.add(
                    samplePlan.cfrFastPath() ? "sample_plan_cfr_probe" : "sample_plan_scan",
                    System.nanoTime() - operationStarted
            );
            if (samplePlan.sourceFrameCount() == 0 || samplePlan.sampleCount() == 0) {
                throw new IOException("Video has no decodable frames in the benchmark window");
            }
            if (cached.rows() > samplePlan.sampleCount()) {
                throw new IOException("Cached visual feature rows exceed the sample plan");
            }
            extractor.seekTo(0, MediaExtractor.SEEK_TO_PREVIOUS_SYNC);

            format.setInteger(MediaFormat.KEY_COLOR_FORMAT,
                    MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible);
            if (decoderOptions.operatingRate() > 0) {
                format.setInteger(MediaFormat.KEY_OPERATING_RATE, decoderOptions.operatingRate());
            }
            if (decoderOptions.priority() >= 0) {
                format.setInteger(MediaFormat.KEY_PRIORITY, decoderOptions.priority());
            }
            codec = MediaCodec.createDecoderByType(mime);
            String decoderName = codec.getName();
            boolean hardware = codec.getCodecInfo().isHardwareAccelerated();
            float[] output = new float[times.length * FeatureSchema.FRAME.size()];
            if (cached.rows() > 0) {
                System.arraycopy(cached.values(), 0, output, 0, cached.values().length);
            }
            int resumeStartRow = cached.rows() == 0 ? 0 : cached.rows() - 1;
            Set<Long> materializedPresentationUs = activeSamplePresentationUs(
                    samplePlan.sortedPresentationUs(), times, resumeStartRow
            );
            featureWorker = new FeatureWorker(output, times, cacheWriter);
            codecThread = new HandlerThread("VolleyCut-MediaCodec");
            codecThread.start();
            AsyncDecodeState state = new AsyncDecodeState(
                    extractor, samplePlan, media, roi, times,
                    progress, cancelled, profiler, featureWorker, decoderName,
                    System.nanoTime(), resumeStartRow, cached.rows(), materializedPresentationUs
            );
            codec.setCallback(state, new Handler(codecThread.getLooper()));
            codec.configure(format, null, null, 0);
            codec.start();
            profiler.add("setup", System.nanoTime() - setupStarted);

            state.awaitCompletion();
            operationStarted = System.nanoTime();
            featureWorker.finish(cancelled);
            profiler.add("feature_worker_finish_wait", System.nanoTime() - operationStarted);
            profiler.appendMilliseconds("", featureWorker.performanceMilliseconds());
            int target = state.targetCount();
            if (target != samplePlan.sampleCount()) {
                throw new IOException("Video produced " + target + " of "
                        + samplePlan.sampleCount() + " planned analysis frames");
            }
            double analyzedDurationSeconds = samplePlan.sourceFrameLimitReached()
                    ? Math.min(media.durationSeconds(),
                            (samplePlan.lastPresentationUs() + samplePlan.frameDurationUs()) / 1_000_000.0)
                    : media.durationSeconds();
            profiler.add("video_pipeline_wall", System.nanoTime() - pipelineStartedNanos);
            double workerCpuMilliseconds = featureWorker.performanceMilliseconds()
                    .getOrDefault("feature_worker_thread_cpu", 0.0);
            double callbackCpuMilliseconds = profiler.milliseconds()
                    .getOrDefault("codec_callback_thread_cpu", 0.0);
            return new AnalysisTypes.VideoFeatures(
                    Arrays.copyOf(output, target * FeatureSchema.FRAME.size()),
                    Arrays.copyOf(times, target),
                    analyzedDurationSeconds,
                    samplePlan.sourceFrameLimitReached(),
                    decoderName,
                    hardware,
                    samplePlan.sourceFrameCount(),
                    state.decodeOnlySourceFrames(),
                    state.decoderOutputFrames(),
                    (Debug.threadCpuTimeNanos() - threadCpuStartedNanos) / 1_000_000.0
                            + callbackCpuMilliseconds + workerCpuMilliseconds,
                    state.meanSampleTimestampErrorMilliseconds(),
                    state.maxSampleTimestampErrorMilliseconds(),
                    profiler.milliseconds()
            );
        } finally {
            // MediaCodec callbacks can still be executing when cancellation interrupts the
            // analysis thread. Drain the callback looper before touching the codec, extractor,
            // or feature worker; releasing any of them first can race native decoder teardown.
            if (codecThread != null) {
                codecThread.quitSafely();
                joinHandlerThread(codecThread);
            }
            if (codec != null) {
                try { codec.stop(); } catch (RuntimeException ignored) {}
                codec.release();
            }
            if (featureWorker != null) featureWorker.close();
            extractor.release();
        }
    }

    // Retained as an in-tree control implementation for direct synchronous/async A/B work.
    @SuppressWarnings("unused")
    private AnalysisTypes.VideoFeatures decodeSynchronous(
            Uri uri,
            AnalysisTypes.MediaInfo media,
            AnalysisTypes.Roi roi,
            double[] times,
            int sourceFrameLimit,
            AnalysisTypes.VideoDecoderOptions decoderOptions,
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
            if (decoderOptions.operatingRate() > 0) {
                format.setInteger(MediaFormat.KEY_OPERATING_RATE, decoderOptions.operatingRate());
            }
            if (decoderOptions.priority() >= 0) {
                format.setInteger(MediaFormat.KEY_PRIORITY, decoderOptions.priority());
            }
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
                    && decodedSourceFrames < sourceFrameLimit) {
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
                            codec.queueInputBuffer(
                                    inputIndex, 0, sampleSize, sampleTime,
                                    codecInputFlags(sampleFlags)
                            );
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
                                ? (sourceFrameLimit - decodedSourceFrames)
                                        / sourceFramesPerSecond
                                : Double.NaN;
                        Runtime runtime = Runtime.getRuntime();
                        progress.onPerformance(new AnalysisTypes.PerformanceStats(
                                target,
                                sourceFrameLimit,
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
                                        / (double) sourceFrameLimit,
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
                    >= sourceFrameLimit;
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
                    0,
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

    private static SamplePlan buildSamplePlan(
            MediaExtractor extractor,
            MediaFormat format,
            double[] times,
            int sourceFrameLimit
    ) {
        SamplePlan cfrPlan = buildCfrSamplePlan(extractor, format, times, sourceFrameLimit);
        if (cfrPlan != null) return cfrPlan;
        extractor.seekTo(0, MediaExtractor.SEEK_TO_PREVIOUS_SYNC);
        int scanLimit = sourceFrameLimit > Integer.MAX_VALUE - SAMPLE_PLAN_REORDER_LOOKAHEAD
                ? Integer.MAX_VALUE : sourceFrameLimit + SAMPLE_PLAN_REORDER_LOOKAHEAD;
        // Full-file mode uses Integer.MAX_VALUE as its logical limit. Do not use
        // that sentinel as an eager ArrayList allocation for variable-frame-rate media.
        ArrayList<Long> presentationTimes = new ArrayList<>(Math.min(scanLimit, 65_536));
        while (presentationTimes.size() < scanLimit) {
            long presentationUs = extractor.getSampleTime();
            if (presentationUs < 0) break;
            presentationTimes.add(presentationUs);
            if (!extractor.advance()) break;
        }
        ArrayList<Long> sorted = new ArrayList<>(presentationTimes);
        Collections.sort(sorted);
        int sourceFrameCount = Math.min(sourceFrameLimit, sorted.size());
        long[] sortedPresentationUs = new long[sourceFrameCount];
        HashSet<Long> sourceWindowPresentationUs = new HashSet<>();
        for (int index = 0; index < sourceFrameCount; index++) {
            sortedPresentationUs[index] = sorted.get(index);
            sourceWindowPresentationUs.add(sortedPresentationUs[index]);
        }
        int inputFrameCount = 0;
        for (int index = 0; index < presentationTimes.size(); index++) {
            if (sourceWindowPresentationUs.contains(presentationTimes.get(index))) {
                inputFrameCount = index + 1;
            }
        }
        HashSet<Long> sampledPresentationUs = new HashSet<>();
        int target = 0;
        for (long presentationUs : sortedPresentationUs) {
            if (target >= times.length) break;
            if (presentationUs + 1_000 >= Math.round(times[target] * 1_000_000)) {
                sampledPresentationUs.add(presentationUs);
                target++;
            }
        }
        long lastPresentationUs = sortedPresentationUs.length == 0
                ? -1 : sortedPresentationUs[sortedPresentationUs.length - 1];
        long frameDurationUs = sortedPresentationUs.length >= 2
                ? Math.max(1, lastPresentationUs - sortedPresentationUs[sortedPresentationUs.length - 2])
                : Math.round(1_000_000.0 / 30);
        return new SamplePlan(
                Set.copyOf(sampledPresentationUs),
                sortedPresentationUs,
                inputFrameCount,
                sourceFrameCount,
                presentationTimes.size() >= sourceFrameLimit,
                target,
                lastPresentationUs,
                frameDurationUs,
                false
        );
    }

    private static Set<Long> activeSamplePresentationUs(
            long[] sortedPresentationUs,
            double[] times,
            int firstTarget
    ) {
        HashSet<Long> active = new HashSet<>();
        int target = 0;
        for (long presentationUs : sortedPresentationUs) {
            if (target >= times.length) break;
            if (presentationUs + 1_000 >= Math.round(times[target] * 1_000_000)) {
                if (target >= firstTarget) active.add(presentationUs);
                target++;
            }
        }
        return Set.copyOf(active);
    }

    private static SamplePlan buildCfrSamplePlan(
            MediaExtractor extractor,
            MediaFormat format,
            double[] times,
            int sourceFrameLimit
    ) {
        // Avoid a full presentation-timestamp scan only after the declared rate exactly
        // predicts a reordered two-second probe. VFR inputs retain the scan-based path.
        if (!format.containsKey(MediaFormat.KEY_FRAME_RATE)
                || !format.containsKey(MediaFormat.KEY_DURATION)) return null;
        double frameRate;
        int totalFrameCount;
        try {
            Number frameRateValue = format.getNumber(MediaFormat.KEY_FRAME_RATE);
            if (frameRateValue == null) return null;
            frameRate = frameRateValue.doubleValue();
            long durationUs = format.getLong(MediaFormat.KEY_DURATION);
            totalFrameCount = (int) Math.min(
                    Integer.MAX_VALUE,
                    Math.round(durationUs / 1_000_000.0 * frameRate)
            );
        } catch (ClassCastException | NullPointerException ignored) {
            return null;
        }
        if (frameRate <= 0 || totalFrameCount <= 0) return null;

        int validationCount = Math.min(SAMPLE_PLAN_REORDER_LOOKAHEAD, totalFrameCount);
        int probeCount = Math.min(
                totalFrameCount,
                validationCount + SAMPLE_PLAN_REORDER_LOOKAHEAD
        );
        long[] probePresentationUs = new long[probeCount];
        int readCount = 0;
        while (readCount < probeCount) {
            long presentationUs = extractor.getSampleTime();
            if (presentationUs < 0) break;
            probePresentationUs[readCount++] = presentationUs;
            if (!extractor.advance()) break;
        }
        if (readCount != probeCount) return null;
        long[] sortedProbePresentationUs = probePresentationUs.clone();
        Arrays.sort(sortedProbePresentationUs);
        long firstPresentationUs = sortedProbePresentationUs[0];
        if (Math.abs(firstPresentationUs) > 1_000) return null;
        double frameDurationExactUs = 1_000_000.0 / frameRate;
        for (int index = 0; index < validationCount; index++) {
            long expectedUs = firstPresentationUs + (long) Math.floor(index * frameDurationExactUs);
            if (Math.abs(sortedProbePresentationUs[index] - expectedUs) > 2) return null;
        }
        long validationLastPresentationUs = sortedProbePresentationUs[validationCount - 1];
        int validationInputExtent = 0;
        for (int index = 0; index < probePresentationUs.length; index++) {
            if (probePresentationUs[index] <= validationLastPresentationUs) {
                validationInputExtent = index + 1;
            }
        }
        int observedReorderGuard = Math.max(2, validationInputExtent - validationCount);

        int sourceFrameCount = Math.min(sourceFrameLimit, totalFrameCount);
        long[] sortedPresentationUs = new long[sourceFrameCount];
        HashSet<Long> sampledPresentationUs = new HashSet<>();
        int target = 0;
        for (int index = 0; index < sourceFrameCount; index++) {
            long presentationUs = firstPresentationUs
                    + (long) Math.floor(index * frameDurationExactUs);
            sortedPresentationUs[index] = presentationUs;
            if (target < times.length
                    && presentationUs + 1_000 >= Math.round(times[target] * 1_000_000)) {
                sampledPresentationUs.add(presentationUs);
                target++;
            }
        }
        int inputFrameCount = Math.min(
                totalFrameCount,
                sourceFrameCount + observedReorderGuard
        );
        long lastPresentationUs = sortedPresentationUs.length == 0
                ? -1 : sortedPresentationUs[sortedPresentationUs.length - 1];
        long frameDurationUs = sortedPresentationUs.length >= 2
                ? Math.max(1, lastPresentationUs
                        - sortedPresentationUs[sortedPresentationUs.length - 2])
                : Math.max(1, Math.round(frameDurationExactUs));
        return new SamplePlan(
                Set.copyOf(sampledPresentationUs),
                sortedPresentationUs,
                inputFrameCount,
                sourceFrameCount,
                totalFrameCount >= sourceFrameLimit,
                target,
                lastPresentationUs,
                frameDurationUs,
                true
        );
    }

    private record SamplePlan(
            Set<Long> sampledPresentationUs,
            long[] sortedPresentationUs,
            int inputFrameCount,
            int sourceFrameCount,
            boolean sourceFrameLimitReached,
            int sampleCount,
            long lastPresentationUs,
            long frameDurationUs,
            boolean cfrFastPath
    ) {
        int sourceFramesAtOrBefore(long presentationUs) {
            int index = Arrays.binarySearch(sortedPresentationUs, presentationUs);
            if (index < 0) return -index - 1;
            while (index + 1 < sortedPresentationUs.length
                    && sortedPresentationUs[index + 1] <= presentationUs) index++;
            return index + 1;
        }
    }

    private static final class AsyncDecodeState extends MediaCodec.Callback {
        private final MediaExtractor extractor;
        private final SamplePlan samplePlan;
        private final AnalysisTypes.MediaInfo media;
        private final AnalysisTypes.Roi roi;
        private final double[] times;
        private final AnalysisTypes.ProgressListener progress;
        private final BooleanSupplier cancelled;
        private final NanoProfiler profiler;
        private final FeatureWorker featureWorker;
        private final String decoderName;
        private final long videoStartedNanos;
        private final int resumedRows;
        private final Set<Long> materializedPresentationUs;
        private final CountDownLatch completion = new CountDownLatch(1);
        private final AtomicReference<Throwable> failure = new AtomicReference<>();

        private boolean inputEnded;
        private int queuedSourceFrames;
        private int decodeOnlySourceFrames;
        private int decoderOutputFrames;
        private int target;
        private long lastProgressNanos;
        private double sampleTimestampErrorTotalMs;
        private double sampleTimestampErrorMaxMs;
        private YuvCropSampler yuvCropSampler;

        AsyncDecodeState(
                MediaExtractor extractor,
                SamplePlan samplePlan,
                AnalysisTypes.MediaInfo media,
                AnalysisTypes.Roi roi,
                double[] times,
                AnalysisTypes.ProgressListener progress,
                BooleanSupplier cancelled,
                NanoProfiler profiler,
                FeatureWorker featureWorker,
                String decoderName,
                long videoStartedNanos,
                int firstTarget,
                int resumedRows,
                Set<Long> materializedPresentationUs
        ) {
            this.extractor = extractor;
            this.samplePlan = samplePlan;
            this.media = media;
            this.roi = roi;
            this.times = times;
            this.progress = progress;
            this.cancelled = cancelled;
            this.profiler = profiler;
            this.featureWorker = featureWorker;
            this.decoderName = decoderName;
            this.videoStartedNanos = videoStartedNanos;
            this.target = firstTarget;
            this.resumedRows = resumedRows;
            this.materializedPresentationUs = materializedPresentationUs;
        }

        @Override
        public void onInputBufferAvailable(MediaCodec codec, int inputIndex) {
            long callbackStarted = System.nanoTime();
            long cpuStarted = Debug.threadCpuTimeNanos();
            try {
                if (completion.getCount() == 0 || inputEnded) return;
                if (cancelled.getAsBoolean()) throw new InterruptedExceptionAsIo();
                if (queuedSourceFrames >= samplePlan.inputFrameCount()) {
                    long operationStarted = System.nanoTime();
                    codec.queueInputBuffer(
                            inputIndex, 0, 0,
                            Math.max(0, samplePlan.lastPresentationUs() + samplePlan.frameDurationUs()),
                            MediaCodec.BUFFER_FLAG_END_OF_STREAM
                    );
                    profiler.add("codec_input_queue", System.nanoTime() - operationStarted);
                    inputEnded = true;
                    return;
                }
                ByteBuffer input = codec.getInputBuffer(inputIndex);
                if (input == null) throw new IOException("Video decoder returned no input buffer");
                long operationStarted = System.nanoTime();
                int sampleSize = extractor.readSampleData(input, 0);
                long sampleTime = sampleSize < 0 ? 0 : extractor.getSampleTime();
                int sampleFlags = sampleSize < 0 ? 0 : extractor.getSampleFlags();
                profiler.add("demux_read", System.nanoTime() - operationStarted);
                if (sampleSize < 0) {
                    operationStarted = System.nanoTime();
                    codec.queueInputBuffer(inputIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                    profiler.add("codec_input_queue", System.nanoTime() - operationStarted);
                    inputEnded = true;
                    return;
                }
                sampleFlags = codecInputFlags(sampleFlags);
                if (!materializedPresentationUs.contains(sampleTime)) {
                    sampleFlags |= MediaCodec.BUFFER_FLAG_DECODE_ONLY;
                    decodeOnlySourceFrames++;
                }
                operationStarted = System.nanoTime();
                codec.queueInputBuffer(inputIndex, 0, sampleSize, sampleTime, sampleFlags);
                profiler.add("codec_input_queue", System.nanoTime() - operationStarted);
                queuedSourceFrames++;
                operationStarted = System.nanoTime();
                extractor.advance();
                profiler.add("demux_advance", System.nanoTime() - operationStarted);
            } catch (Throwable error) {
                fail(error);
            } finally {
                profiler.add("codec_input_callback", System.nanoTime() - callbackStarted);
                profiler.add("codec_callback_thread_cpu", Debug.threadCpuTimeNanos() - cpuStarted);
            }
        }

        @Override
        public void onOutputBufferAvailable(
                MediaCodec codec,
                int outputIndex,
                MediaCodec.BufferInfo info
        ) {
            long callbackStarted = System.nanoTime();
            long cpuStarted = Debug.threadCpuTimeNanos();
            boolean eos = (info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0;
            try {
                if (info.size > 0) decoderOutputFrames++;
                long presentationUs = info.presentationTimeUs;
                boolean sampleThisFrame = info.size > 0
                        && target < samplePlan.sampleCount()
                        && presentationUs + 1_000 >= Math.round(times[target] * 1_000_000);
                if (sampleThisFrame) processSample(codec, outputIndex, presentationUs);
            } catch (Throwable error) {
                fail(error);
            } finally {
                try {
                    long operationStarted = System.nanoTime();
                    codec.releaseOutputBuffer(outputIndex, false);
                    profiler.add("codec_output_release", System.nanoTime() - operationStarted);
                } catch (Throwable error) {
                    fail(error);
                }
                if (eos) {
                    if (target != samplePlan.sampleCount()) {
                        fail(new IOException("Decoder reached EOS after " + target + " of "
                                + samplePlan.sampleCount() + " planned analysis frames"));
                    }
                    completion.countDown();
                }
                profiler.add("codec_output_callback", System.nanoTime() - callbackStarted);
                profiler.add("codec_callback_thread_cpu", Debug.threadCpuTimeNanos() - cpuStarted);
            }
        }

        private void processSample(MediaCodec codec, int outputIndex, long presentationUs)
                throws IOException {
            long sampleStarted = System.nanoTime();
            long operationStarted = System.nanoTime();
            Image image = codec.getOutputImage(outputIndex);
            profiler.add("output_image_acquire", System.nanoTime() - operationStarted);
            if (image == null || image.getFormat() != ImageFormat.YUV_420_888) {
                if (image != null) image.close();
                throw new IOException(
                        "The device decoder did not expose YUV_420_888 output. Decoder: "
                                + decoderName
                );
            }
            try {
                operationStarted = System.nanoTime();
                if (yuvCropSampler == null || !yuvCropSampler.matches(image)) {
                    yuvCropSampler = new YuvCropSampler(image, roi, media.rotation());
                    profiler.add("yuv_sampler_setup", System.nanoTime() - operationStarted);
                    operationStarted = System.nanoTime();
                }
                Mat rgba = yuvCropSampler.convert(image);
                profiler.add("yuv_crop_scale_color", System.nanoTime() - operationStarted);
                try {
                    operationStarted = System.nanoTime();
                    featureWorker.submit(target, rgba, target >= resumedRows, cancelled);
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
            updateProgress(presentationUs);
            profiler.add("sampled_frame_total", System.nanoTime() - sampleStarted);
        }

        private void updateProgress(long presentationUs) {
            long now = System.nanoTime();
            if (now - lastProgressNanos < 250_000_000L && target != samplePlan.sampleCount()) return;
            lastProgressNanos = now;
            double elapsedSeconds = (now - videoStartedNanos) / 1_000_000_000.0;
            double framesPerSecond = target / Math.max(elapsedSeconds, 1e-9);
            double generatedVideoSeconds = target / (double) FeatureSchema.ANALYSIS_FPS;
            double realtimeRatio = generatedVideoSeconds / Math.max(elapsedSeconds, 1e-9);
            int decodedEquivalent = samplePlan.sourceFramesAtOrBefore(presentationUs);
            int totalSourceFrames = samplePlan.sourceFrameCount();
            double sourceFramesPerSecond = decodedEquivalent / Math.max(elapsedSeconds, 1e-9);
            double etaSeconds = sourceFramesPerSecond > 0
                    ? Math.max(0, totalSourceFrames - decodedEquivalent) / sourceFramesPerSecond
                    : Double.NaN;
            Runtime runtime = Runtime.getRuntime();
            progress.onPerformance(new AnalysisTypes.PerformanceStats(
                    target,
                    totalSourceFrames,
                    decodedEquivalent,
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
                    totalSourceFrames > 0 ? decodedEquivalent / (double) totalSourceFrames : 1.0,
                    String.format(Locale.US,
                            "Async decode + OpenCV | %.2fx real time | %.1f frames/s",
                            realtimeRatio,
                            framesPerSecond)
            );
        }

        @Override
        public void onOutputFormatChanged(MediaCodec codec, MediaFormat format) {}

        @Override
        public void onError(MediaCodec codec, MediaCodec.CodecException error) {
            fail(error);
        }

        void awaitCompletion() throws IOException {
            try {
                while (!completion.await(25, TimeUnit.MILLISECONDS)) {
                    if (cancelled.getAsBoolean()) throw new InterruptedExceptionAsIo();
                    featureWorker.throwIfFailed();
                }
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                throw new IOException("Interrupted while waiting for video decoder", interrupted);
            }
            throwIfFailed();
            featureWorker.throwIfFailed();
        }

        private void fail(Throwable error) {
            failure.compareAndSet(null, error);
            completion.countDown();
        }

        private void throwIfFailed() throws IOException {
            Throwable error = failure.get();
            if (error == null) return;
            if (error instanceof IOException io) throw io;
            throw new IOException("Asynchronous video decoder failed", error);
        }

        int targetCount() {
            return target;
        }

        int decodeOnlySourceFrames() {
            return decodeOnlySourceFrames;
        }

        int decoderOutputFrames() {
            return decoderOutputFrames;
        }

        double meanSampleTimestampErrorMilliseconds() {
            return target > 0 ? sampleTimestampErrorTotalMs / target : 0;
        }

        double maxSampleTimestampErrorMilliseconds() {
            return sampleTimestampErrorMaxMs;
        }
    }

    static Mat imageToAnalysisRgba(Image image, AnalysisTypes.Roi roi, int rotation) {
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

    private static final class YuvCropSampler {
        private static final int PIXEL_COUNT = FeatureSchema.ANALYSIS_WIDTH
                * FeatureSchema.ANALYSIS_HEIGHT;
        private static final int[] Y_COMPONENT = new int[256];
        private static final int[] RED_FROM_V = new int[256];
        private static final int[] GREEN_FROM_U = new int[256];
        private static final int[] GREEN_FROM_V = new int[256];
        private static final int[] BLUE_FROM_U = new int[256];

        static {
            for (int value = 0; value < 256; value++) {
                Y_COMPONENT[value] = 298 * Math.max(0, value - 16);
                int chroma = value - 128;
                RED_FROM_V[value] = 409 * chroma;
                GREEN_FROM_U[value] = -100 * chroma;
                GREEN_FROM_V[value] = -208 * chroma;
                BLUE_FROM_U[value] = 516 * chroma;
            }
        }

        private final Rect crop;
        private final int yRowStride;
        private final int yPixelStride;
        private final int uRowStride;
        private final int uPixelStride;
        private final int vRowStride;
        private final int vPixelStride;
        private final int[] yOffsets = new int[PIXEL_COUNT];
        private final int[] uOffsets = new int[PIXEL_COUNT];
        private final int[] vOffsets = new int[PIXEL_COUNT];
        private final byte[] rgba = new byte[PIXEL_COUNT * 4];

        YuvCropSampler(Image image, AnalysisTypes.Roi roi, int rotation) {
            Image.Plane[] planes = image.getPlanes();
            if (planes.length < 3) {
                throw new IllegalArgumentException("YUV image has fewer than three planes");
            }
            crop = new Rect(image.getCropRect());
            yRowStride = planes[0].getRowStride();
            yPixelStride = planes[0].getPixelStride();
            uRowStride = planes[1].getRowStride();
            uPixelStride = planes[1].getPixelStride();
            vRowStride = planes[2].getRowStride();
            vPixelStride = planes[2].getPixelStride();

            int index = 0;
            for (int y = 0; y < FeatureSchema.ANALYSIS_HEIGHT; y++) {
                double displayV = roi.y()
                        + (y + 0.5) / FeatureSchema.ANALYSIS_HEIGHT * roi.height();
                for (int x = 0; x < FeatureSchema.ANALYSIS_WIDTH; x++) {
                    double displayU = roi.x()
                            + (x + 0.5) / FeatureSchema.ANALYSIS_WIDTH * roi.width();
                    double sourceU;
                    double sourceV;
                    switch (rotation) {
                        case 90 -> { sourceU = displayV; sourceV = 1 - displayU; }
                        case 180 -> { sourceU = 1 - displayU; sourceV = 1 - displayV; }
                        case 270 -> { sourceU = 1 - displayV; sourceV = displayU; }
                        default -> { sourceU = displayU; sourceV = displayV; }
                    }
                    int sourceX = clamp(
                            crop.left + (int) Math.floor(sourceU * crop.width()),
                            crop.left, crop.right - 1
                    );
                    int sourceY = clamp(
                            crop.top + (int) Math.floor(sourceV * crop.height()),
                            crop.top, crop.bottom - 1
                    );
                    yOffsets[index] = sourceY * yRowStride + sourceX * yPixelStride;
                    int chromaX = sourceX / 2;
                    int chromaY = sourceY / 2;
                    uOffsets[index] = chromaY * uRowStride + chromaX * uPixelStride;
                    vOffsets[index] = chromaY * vRowStride + chromaX * vPixelStride;
                    index++;
                }
            }
        }

        boolean matches(Image image) {
            Image.Plane[] planes = image.getPlanes();
            if (planes.length < 3 || !crop.equals(image.getCropRect())) return false;
            return yRowStride == planes[0].getRowStride()
                    && yPixelStride == planes[0].getPixelStride()
                    && uRowStride == planes[1].getRowStride()
                    && uPixelStride == planes[1].getPixelStride()
                    && vRowStride == planes[2].getRowStride()
                    && vPixelStride == planes[2].getPixelStride();
        }

        Mat convert(Image image) {
            Image.Plane[] planes = image.getPlanes();
            ByteBuffer yBuffer = planes[0].getBuffer().duplicate();
            ByteBuffer uBuffer = planes[1].getBuffer().duplicate();
            ByteBuffer vBuffer = planes[2].getBuffer().duplicate();
            int outputIndex = 0;
            for (int index = 0; index < PIXEL_COUNT; index++) {
                int yValue = yBuffer.get(yOffsets[index]) & 0xff;
                int uValue = uBuffer.get(uOffsets[index]) & 0xff;
                int vValue = vBuffer.get(vOffsets[index]) & 0xff;
                int yComponent = Y_COMPONENT[yValue];
                int red = clamp((yComponent + RED_FROM_V[vValue] + 128) >> 8, 0, 255);
                int green = clamp((yComponent + GREEN_FROM_U[uValue]
                        + GREEN_FROM_V[vValue] + 128) >> 8, 0, 255);
                int blue = clamp((yComponent + BLUE_FROM_U[uValue] + 128) >> 8, 0, 255);
                rgba[outputIndex++] = (byte) red;
                rgba[outputIndex++] = (byte) green;
                rgba[outputIndex++] = (byte) blue;
                rgba[outputIndex++] = (byte) 255;
            }
            Mat result = new Mat(
                    FeatureSchema.ANALYSIS_HEIGHT,
                    FeatureSchema.ANALYSIS_WIDTH,
                    CvType.CV_8UC4
            );
            result.put(0, 0, rgba);
            return result;
        }
    }

    static int findTrack(MediaExtractor extractor, String prefix) {
        for (int index = 0; index < extractor.getTrackCount(); index++) {
            String mime = extractor.getTrackFormat(index).getString(MediaFormat.KEY_MIME);
            if (mime != null && mime.startsWith(prefix)) return index;
        }
        return -1;
    }

    /** Join a codec callback thread without allowing executor cancellation to skip the join. */
    private static void joinHandlerThread(HandlerThread thread) {
        boolean interrupted = false;
        try {
            while (thread.isAlive()) {
                try {
                    thread.join(5_000);
                } catch (InterruptedException error) {
                    // shutdownNow() interrupts the analysis thread while callback teardown still
                    // needs to finish. Preserve the flag after the callback thread is quiescent.
                    interrupted = true;
                }
            }
        } finally {
            if (interrupted) Thread.currentThread().interrupt();
        }
    }

    private static int clamp(int value, int lower, int upper) {
        return Math.max(lower, Math.min(upper, value));
    }

    private static final class FeatureWorker implements AutoCloseable {
        private static final int QUEUE_CAPACITY = 2;
        private static final SampleTask FINISH = new SampleTask(-1, null, false);

        private final float[] output;
        private final double[] times;
        private final NativeFeatureCache.VisualWriter cacheWriter;
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
            this(output, null, null);
        }

        FeatureWorker(
                float[] output,
                double[] times,
                NativeFeatureCache.VisualWriter cacheWriter
        ) {
            this.output = output;
            this.times = times;
            this.cacheWriter = cacheWriter;
            thread = new Thread(this::run, "VolleyCut-OpenCV");
            thread.start();
        }

        void submit(int rowIndex, Mat rgba, BooleanSupplier cancelled) throws IOException {
            submit(rowIndex, rgba, true, cancelled);
        }

        void submit(
                int rowIndex,
                Mat rgba,
                boolean storeResult,
                BooleanSupplier cancelled
        ) throws IOException {
            SampleTask task = new SampleTask(rowIndex, rgba, storeResult);
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
                        if (task.storeResult()) {
                            operationStarted = System.nanoTime();
                            System.arraycopy(
                                    row, 0, output,
                                    task.rowIndex() * FeatureSchema.FRAME.size(), row.length
                            );
                            profiler.add("feature_matrix_copy", System.nanoTime() - operationStarted);
                            if (cacheWriter != null) {
                                cacheWriter.append(task.rowIndex(), times[task.rowIndex()], row);
                            }
                        }
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
            boolean interrupted = false;
            while (thread.isAlive()) {
                try {
                    thread.join(5_000);
                } catch (InterruptedException error) {
                    // The analysis executor is interrupted during timeout cancellation, but the
                    // OpenCV worker must still finish releasing native Mats before teardown.
                    interrupted = true;
                }
            }
            SampleTask pending;
            while ((pending = queue.poll()) != null) pending.release();
            if (interrupted) Thread.currentThread().interrupt();
        }
    }

    private record SampleTask(int rowIndex, Mat rgba, boolean storeResult) {
        void release() {
            if (rgba != null) rgba.release();
        }
    }

    /** MediaExtractor flags are not the same bit field as MediaCodec flags. */
    private static int codecInputFlags(int extractorFlags) {
        return (extractorFlags & MediaExtractor.SAMPLE_FLAG_PARTIAL_FRAME) != 0
                ? MediaCodec.BUFFER_FLAG_PARTIAL_FRAME
                : 0;
    }

    private static final class InterruptedExceptionAsIo extends IOException {
        InterruptedExceptionAsIo() {
            super("Analysis cancelled");
        }
    }
}
