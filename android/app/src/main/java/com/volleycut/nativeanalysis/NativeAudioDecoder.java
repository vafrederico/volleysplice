package com.volleycut.nativeanalysis;

import android.content.Context;
import android.media.AudioFormat;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.net.Uri;
import android.os.Build;
import android.os.Debug;
import android.os.Handler;
import android.os.HandlerThread;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayDeque;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BooleanSupplier;

final class NativeAudioDecoder {
    record Result(
            float[] features,
            String decoderName,
            long decodedPcmFrames,
            long resampledOutputSamples,
            int audioFeatureFrames,
            int codecOperatingRate,
            int codecPriority,
            boolean multipleFramesSupported,
            String decoderMode,
            long inputAccessUnits,
            long inputBatches,
            long outputAccessUnits,
            long outputBatches,
            String featureSha256,
            double threadCpuMilliseconds,
            Map<String, Double> profileMilliseconds
    ) {}

    private static final long CODEC_TIMEOUT_US = 10_000;
    private static final float CODEC_OPERATING_RATE_MULTIPLIER = 4f;
    private static final int CODEC_PRIORITY = 0;
    private static final int MAX_BATCH_ACCESS_UNITS = 32;
    private final Context context;

    NativeAudioDecoder(Context context) {
        this.context = context.getApplicationContext();
    }

    Result decode(
            Uri uri,
            AnalysisTypes.AnalysisWindow analysisWindow,
            double[] analysisTimes,
            AnalysisTypes.AudioDecoderMode decoderMode,
            AnalysisTypes.ProgressListener progress,
            BooleanSupplier cancelled
    ) throws IOException {
        long pipelineStartedNanos = System.nanoTime();
        long threadCpuStartedNanos = Debug.threadCpuTimeNanos();
        NanoProfiler profiler = new NanoProfiler();
        MediaExtractor extractor = new MediaExtractor();
        MediaCodec codec = null;
        double analysisDuration = analysisWindow.end() - analysisWindow.start();
        long startUs = Math.max(0, Math.round(analysisWindow.start() * 1_000_000));
        long endUs = Math.max(startUs + 1, Math.round(analysisWindow.end() * 1_000_000));
        try {
            long setupStarted = System.nanoTime();
            extractor.setDataSource(context, uri, null);
            int track = NativeVideoDecoder.findTrack(extractor, "audio/");
            if (track < 0) {
                progress.onProgress("audio", 1, "No audio track; using zero audio features");
                return new Result(
                        new float[analysisTimes.length * FeatureSchema.AUDIO.size()],
                        "none", 0, 0, 0, 0, -1, false,
                        decoderMode.wireName(), 0, 0, 0, 0, "",
                        (Debug.threadCpuTimeNanos() - threadCpuStartedNanos) / 1_000_000.0,
                        Map.of()
                );
            }
            extractor.selectTrack(track);
            extractor.seekTo(startUs, MediaExtractor.SEEK_TO_PREVIOUS_SYNC);
            MediaFormat inputFormat = extractor.getTrackFormat(track);
            String mime = inputFormat.getString(MediaFormat.KEY_MIME);
            if (mime == null) throw new IOException("Audio track has no MIME type");
            int sampleRate = inputFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE);
            int codecOperatingRate = Math.round(sampleRate * CODEC_OPERATING_RATE_MULTIPLIER);
            inputFormat.setFloat(MediaFormat.KEY_OPERATING_RATE, codecOperatingRate);
            inputFormat.setInteger(MediaFormat.KEY_PRIORITY, CODEC_PRIORITY);
            codec = MediaCodec.createDecoderByType(mime);
            String decoderName = codec.getName();
            boolean multipleFramesSupported = codec.getCodecInfo()
                    .getCapabilitiesForType(mime)
                    .isFeatureSupported(MediaCodecInfo.CodecCapabilities.FEATURE_MultipleFrames);
            boolean batchingAvailable = Build.VERSION.SDK_INT >= 35 && multipleFramesSupported;
            if (decoderMode == AnalysisTypes.AudioDecoderMode.BATCHED_ACCESS_UNITS
                    && !batchingAvailable) {
                throw new IOException(Build.VERSION.SDK_INT < 35
                        ? "Batched audio decoding requires Android 15 or newer"
                        : "The selected audio decoder does not support multiple frames");
            }
            if (decoderMode == AnalysisTypes.AudioDecoderMode.BATCHED_ACCESS_UNITS
                    || decoderMode == AnalysisTypes.AudioDecoderMode.AUTO && batchingAvailable) {
                return decodeBatched(
                        extractor, codec, inputFormat, analysisWindow, analysisTimes,
                        startUs, endUs, analysisDuration, sampleRate, codecOperatingRate,
                        decoderName, multipleFramesSupported, setupStarted, pipelineStartedNanos,
                        threadCpuStartedNanos, profiler, progress, cancelled
                );
            }
            codec.configure(inputFormat, null, null, 0);
            codec.start();
            profiler.add("setup", System.nanoTime() - setupStarted);
            progress.onProgress("audio", 0, "Decoding PCM + resampling + per-frame FFT");

            int channels = inputFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT);
            int encoding = AudioFormat.ENCODING_PCM_16BIT;
            AudioFeatureExtractor accumulator = new AudioFeatureExtractor();
            MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
            boolean inputEnded = false;
            boolean outputEnded = false;
            long lastReportedSecond = -1;
            long decodedPcmFrames = 0;
            long inputAccessUnits = 0;
            long outputAccessUnits = 0;
            while (!outputEnded) {
                if (cancelled.getAsBoolean()) throw new IOException("Analysis cancelled");
                if (!inputEnded) {
                    long operationStarted = System.nanoTime();
                    int inputIndex = codec.dequeueInputBuffer(CODEC_TIMEOUT_US);
                    profiler.add("codec_input_dequeue", System.nanoTime() - operationStarted);
                    if (inputIndex >= 0) {
                        ByteBuffer input = codec.getInputBuffer(inputIndex);
                        if (input == null) throw new IOException("Audio decoder returned no input buffer");
                        operationStarted = System.nanoTime();
                        long nextSampleTime = extractor.getSampleTime();
                        int sampleSize = nextSampleTime == -1 || nextSampleTime >= endUs
                                ? -1
                                : extractor.readSampleData(input, 0);
                        long sampleTime = sampleSize < 0 ? 0 : extractor.getSampleTime();
                        int sampleFlags = sampleSize < 0 ? 0 : extractor.getSampleFlags();
                        profiler.add("demux_read", System.nanoTime() - operationStarted);
                        operationStarted = System.nanoTime();
                        if (sampleSize < 0) {
                            codec.queueInputBuffer(inputIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                            inputEnded = true;
                        } else {
                            codec.queueInputBuffer(inputIndex, 0, sampleSize, sampleTime, sampleFlags);
                            inputAccessUnits++;
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
                if (outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    MediaFormat outputFormat = codec.getOutputFormat();
                    sampleRate = outputFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE);
                    channels = outputFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT);
                    if (outputFormat.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
                        encoding = outputFormat.getInteger(MediaFormat.KEY_PCM_ENCODING);
                    }
                    continue;
                }
                if (outputIndex < 0) continue;
                if (info.size > 0 && info.presentationTimeUs < endUs) {
                    outputAccessUnits++;
                    operationStarted = System.nanoTime();
                    ByteBuffer buffer = codec.getOutputBuffer(outputIndex);
                    if (buffer == null) throw new IOException("Audio decoder returned no output buffer");
                    float[] mono = pcmToMono(buffer, info.offset, info.size, channels, encoding);
                    long remainingUs = endUs - info.presentationTimeUs;
                    int framesToKeep = (int) Math.min(
                            mono.length,
                            Math.max(0, (remainingUs * sampleRate + 999_999) / 1_000_000)
                    );
                    if (framesToKeep != mono.length) {
                        mono = java.util.Arrays.copyOf(mono, framesToKeep);
                    }
                    profiler.add("pcm_copy_and_downmix", System.nanoTime() - operationStarted);
                    decodedPcmFrames += mono.length;
                    operationStarted = System.nanoTime();
                    accumulator.push(
                            mono,
                            info.presentationTimeUs / 1_000_000.0 - analysisWindow.start(),
                            sampleRate
                    );
                    profiler.add("accumulator_push", System.nanoTime() - operationStarted);
                    long second = Math.max(
                            0,
                            (info.presentationTimeUs - startUs) / 1_000_000
                    );
                    if (second != lastReportedSecond) {
                        lastReportedSecond = second;
                        progress.onProgress(
                                "audio",
                                Math.min(0.80, 0.80 * second / analysisDuration),
                                String.format(
                                        java.util.Locale.US,
                                        "Decoding PCM + resampling + per-frame FFT · %d / %.0f s",
                                        second,
                                        analysisDuration
                                )
                        );
                    }
                }
                outputEnded = (info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0;
                operationStarted = System.nanoTime();
                codec.releaseOutputBuffer(outputIndex, false);
                profiler.add("codec_output_release", System.nanoTime() - operationStarted);
            }
            long finishStarted = System.nanoTime();
            double[] relativeTimes = new double[analysisTimes.length];
            for (int index = 0; index < analysisTimes.length; index++) {
                relativeTimes[index] = analysisTimes[index] - analysisWindow.start();
            }
            float[] features = accumulator.finishAndPool(
                    relativeTimes,
                    (fraction, detail) -> progress.onProgress("audio", fraction, detail)
            );
            profiler.add("dsp_finish_and_pool_call", System.nanoTime() - finishStarted);
            profiler.appendMilliseconds("dsp/", accumulator.performanceMilliseconds());
            profiler.add("audio_pipeline_wall", System.nanoTime() - pipelineStartedNanos);
            return new Result(
                    features,
                    decoderName,
                    decodedPcmFrames,
                    accumulator.resampledOutputSamples(),
                    accumulator.audioFeatureFrameCount(),
                    codecOperatingRate,
                    CODEC_PRIORITY,
                    multipleFramesSupported,
                    AnalysisTypes.AudioDecoderMode.SINGLE_ACCESS_UNIT.wireName(),
                    inputAccessUnits,
                    inputAccessUnits,
                    outputAccessUnits,
                    outputAccessUnits,
                    featureSha256(features),
                    (Debug.threadCpuTimeNanos() - threadCpuStartedNanos) / 1_000_000.0,
                    profiler.milliseconds()
            );
        } finally {
            if (codec != null) {
                try { codec.stop(); } catch (RuntimeException ignored) {}
                codec.release();
            }
            extractor.release();
        }
    }

    private Result decodeBatched(
            MediaExtractor extractor,
            MediaCodec codec,
            MediaFormat inputFormat,
            AnalysisTypes.AnalysisWindow analysisWindow,
            double[] analysisTimes,
            long startUs,
            long endUs,
            double analysisDuration,
            int initialSampleRate,
            int codecOperatingRate,
            String decoderName,
            boolean multipleFramesSupported,
            long setupStarted,
            long pipelineStartedNanos,
            long threadCpuStartedNanos,
            NanoProfiler profiler,
            AnalysisTypes.ProgressListener progress,
            BooleanSupplier cancelled
    ) throws IOException {
        int initialChannels = inputFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT);
        int oneSecondPcmBytes = Math.min(
                1_048_576,
                Math.max(16_384, initialSampleRate * initialChannels * 2)
        );
        inputFormat.setInteger(MediaFormat.KEY_BUFFER_BATCH_MAX_OUTPUT_SIZE, oneSecondPcmBytes);
        inputFormat.setInteger(
                MediaFormat.KEY_BUFFER_BATCH_THRESHOLD_OUTPUT_SIZE,
                oneSecondPcmBytes / 2
        );

        HandlerThread codecThread = new HandlerThread("volleycut-audio-codec");
        codecThread.start();
        Handler codecHandler = new Handler(codecThread.getLooper());
        AudioFeatureExtractor accumulator = new AudioFeatureExtractor();
        BatchedDecodeState state = new BatchedDecodeState(
                extractor, analysisWindow, startUs, endUs, analysisDuration,
                initialSampleRate, initialChannels, accumulator, profiler, progress, cancelled
        );
        try {
            codec.setCallback(state, codecHandler);
            codec.configure(inputFormat, null, null, 0);
            codec.start();
            profiler.add("setup", System.nanoTime() - setupStarted);
            progress.onProgress(
                    "audio", 0,
                    "Decoding batched PCM + resampling + per-frame FFT"
            );
            state.awaitCompletion();

            CountDownLatch callbacksDrained = new CountDownLatch(1);
            codecHandler.post(callbacksDrained::countDown);
            try {
                if (!callbacksDrained.await(5, TimeUnit.SECONDS)) {
                    throw new IOException("Timed out draining audio codec callbacks");
                }
            } catch (InterruptedException error) {
                Thread.currentThread().interrupt();
                throw new IOException("Interrupted while draining audio codec callbacks", error);
            }

            long finishStarted = System.nanoTime();
            double[] relativeTimes = new double[analysisTimes.length];
            for (int index = 0; index < analysisTimes.length; index++) {
                relativeTimes[index] = analysisTimes[index] - analysisWindow.start();
            }
            float[] features = accumulator.finishAndPool(
                    relativeTimes,
                    (fraction, detail) -> progress.onProgress("audio", fraction, detail)
            );
            profiler.add("dsp_finish_and_pool_call", System.nanoTime() - finishStarted);
            profiler.appendMilliseconds("dsp/", accumulator.performanceMilliseconds());
            profiler.add("audio_pipeline_wall", System.nanoTime() - pipelineStartedNanos);
            return new Result(
                    features,
                    decoderName,
                    state.decodedPcmFrames,
                    accumulator.resampledOutputSamples(),
                    accumulator.audioFeatureFrameCount(),
                    codecOperatingRate,
                    CODEC_PRIORITY,
                    multipleFramesSupported,
                    AnalysisTypes.AudioDecoderMode.BATCHED_ACCESS_UNITS.wireName(),
                    state.inputAccessUnits,
                    state.inputBatches,
                    state.outputAccessUnits,
                    state.outputBatches,
                    featureSha256(features),
                    (Debug.threadCpuTimeNanos() - threadCpuStartedNanos
                            + state.callbackCpuNanos) / 1_000_000.0,
                    profiler.milliseconds()
            );
        } finally {
            codecThread.quitSafely();
            try {
                codecThread.join(5_000);
            } catch (InterruptedException error) {
                Thread.currentThread().interrupt();
            }
        }
    }

    static final class DecodedAccessUnitAssembler {
        @FunctionalInterface
        interface Consumer {
            void accept(float[] frames, long presentationTimeUs);
        }

        private final ArrayDeque<Long> inputTimes = new ArrayDeque<>();
        private int framesPerUnit;
        private int encoderDelayFrames;
        private int encoderPaddingFrames;
        private float[] pendingFrames = new float[0];
        private int pendingLength;
        private int pendingTargetLength;
        private long pendingTimeUs;
        private boolean gaplessStartApplied;

        void addInputTimes(long[] timestamps) {
            for (long timestamp : timestamps) inputTimes.add(timestamp);
        }

        void setFramesPerUnit(int framesPerUnit) {
            if (framesPerUnit <= 0 || this.framesPerUnit != 0) return;
            this.framesPerUnit = framesPerUnit;
            pendingFrames = new float[framesPerUnit];
        }

        void setGaplessTrimming(int encoderDelayFrames, int encoderPaddingFrames) {
            this.encoderDelayFrames = Math.max(0, encoderDelayFrames);
            this.encoderPaddingFrames = Math.max(0, encoderPaddingFrames);
        }

        int accept(
                float[] frames,
                long fallbackTimeUs,
                int sampleRate,
            Consumer consumer
        ) {
            if (frames.length == 0) return 0;
            if (framesPerUnit == 0) {
                consumer.accept(frames, takeTime(fallbackTimeUs));
                return 1;
            }
            if (!gaplessStartApplied) {
                int delayedUnits = (encoderDelayFrames + framesPerUnit - 1) / framesPerUnit;
                for (int index = 0; index < delayedUnits && !inputTimes.isEmpty(); index++) {
                    inputTimes.poll();
                }
                int partialFrames = encoderPaddingFrames % framesPerUnit;
                pendingTargetLength = partialFrames == 0
                        ? framesPerUnit
                        : framesPerUnit - partialFrames;
                gaplessStartApplied = true;
            }
            int emitted = 0;
            int offset = 0;
            while (offset < frames.length) {
                if (pendingLength == 0) {
                    long offsetUs = Math.round(offset * 1_000_000.0 / sampleRate);
                    pendingTimeUs = takeTime(fallbackTimeUs + offsetUs);
                }
                int copied = Math.min(
                        pendingTargetLength - pendingLength,
                        frames.length - offset
                );
                System.arraycopy(frames, offset, pendingFrames, pendingLength, copied);
                pendingLength += copied;
                offset += copied;
                if (pendingLength == pendingTargetLength) {
                    consumer.accept(
                            java.util.Arrays.copyOf(pendingFrames, pendingLength),
                            pendingTimeUs
                    );
                    pendingLength = 0;
                    pendingTargetLength = framesPerUnit;
                    emitted++;
                }
            }
            return emitted;
        }

        int flush(Consumer consumer) {
            if (pendingLength == 0) return 0;
            consumer.accept(java.util.Arrays.copyOf(pendingFrames, pendingLength), pendingTimeUs);
            pendingLength = 0;
            return 1;
        }

        int pendingInputTimeCount() {
            return inputTimes.size();
        }

        private long takeTime(long fallbackTimeUs) {
            Long inputTimeUs = inputTimes.poll();
            return inputTimeUs == null ? fallbackTimeUs : inputTimeUs;
        }
    }

    private static final class BatchedDecodeState extends MediaCodec.Callback {
        private final MediaExtractor extractor;
        private final AnalysisTypes.AnalysisWindow analysisWindow;
        private final long startUs;
        private final long endUs;
        private final double analysisDuration;
        private final AudioFeatureExtractor accumulator;
        private final NanoProfiler profiler;
        private final AnalysisTypes.ProgressListener progress;
        private final BooleanSupplier cancelled;
        private final CountDownLatch completion = new CountDownLatch(1);
        private final AtomicReference<Throwable> failure = new AtomicReference<>();
        private final AtomicBoolean terminal = new AtomicBoolean();
        private final DecodedAccessUnitAssembler pcmAssembler =
                new DecodedAccessUnitAssembler();
        private int sampleRate;
        private int channels;
        private int encoding = AudioFormat.ENCODING_PCM_16BIT;
        private boolean inputEnded;
        private long lastReportedSecond = -1;
        private long decodedPcmFrames;
        private long inputAccessUnits;
        private long inputBatches;
        private long outputAccessUnits;
        private long outputBatches;
        private long callbackCpuNanos;

        BatchedDecodeState(
                MediaExtractor extractor,
                AnalysisTypes.AnalysisWindow analysisWindow,
                long startUs,
                long endUs,
                double analysisDuration,
                int sampleRate,
                int channels,
                AudioFeatureExtractor accumulator,
                NanoProfiler profiler,
                AnalysisTypes.ProgressListener progress,
                BooleanSupplier cancelled
        ) {
            this.extractor = extractor;
            this.analysisWindow = analysisWindow;
            this.startUs = startUs;
            this.endUs = endUs;
            this.analysisDuration = analysisDuration;
            this.sampleRate = sampleRate;
            this.channels = channels;
            this.accumulator = accumulator;
            this.profiler = profiler;
            this.progress = progress;
            this.cancelled = cancelled;
        }

        @Override
        public void onInputBufferAvailable(MediaCodec codec, int index) {
            long callbackCpuStarted = Debug.threadCpuTimeNanos();
            try {
                if (terminal.get()) return;
                if (cancelled.getAsBoolean()) {
                    fail(new IOException("Analysis cancelled"));
                    return;
                }
                if (inputEnded) return;
                ByteBuffer input = codec.getInputBuffer(index);
                if (input == null) throw new IOException("Audio decoder returned no input buffer");
                input.clear();
                ArrayDeque<MediaCodec.BufferInfo> infos = new ArrayDeque<>();
                int offset = 0;
                boolean reachedEnd = false;
                while (infos.size() < MAX_BATCH_ACCESS_UNITS) {
                    long sampleTime = extractor.getSampleTime();
                    if (sampleTime == -1 || sampleTime >= endUs) {
                        reachedEnd = true;
                        break;
                    }
                    long declaredSize = extractor.getSampleSize();
                    if (declaredSize > input.capacity() - offset) {
                        if (infos.isEmpty()) {
                            throw new IOException(String.format(
                                    java.util.Locale.US,
                                    "Audio access unit %,d bytes exceeds codec input capacity %,d",
                                    declaredSize, input.capacity()
                            ));
                        }
                        break;
                    }
                    if (declaredSize < 0 && !infos.isEmpty()) break;
                    long operationStarted = System.nanoTime();
                    int sampleSize = extractor.readSampleData(input, offset);
                    profiler.add("demux_read", System.nanoTime() - operationStarted);
                    if (sampleSize < 0) {
                        reachedEnd = true;
                        break;
                    }
                    MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
                    info.set(offset, sampleSize, sampleTime, extractor.getSampleFlags());
                    infos.add(info);
                    offset += sampleSize;
                    operationStarted = System.nanoTime();
                    extractor.advance();
                    profiler.add("demux_advance", System.nanoTime() - operationStarted);
                }
                if (infos.isEmpty()) {
                    if (!reachedEnd) {
                        throw new IOException("Could not fill an audio codec input batch");
                    }
                    long operationStarted = System.nanoTime();
                    codec.queueInputBuffer(
                            index, 0, 0, endUs, MediaCodec.BUFFER_FLAG_END_OF_STREAM
                    );
                    profiler.add("codec_input_queue", System.nanoTime() - operationStarted);
                    inputEnded = true;
                    return;
                }
                long operationStarted = System.nanoTime();
                long[] batchTimes = new long[infos.size()];
                int timestampIndex = 0;
                for (MediaCodec.BufferInfo info : infos) {
                    batchTimes[timestampIndex++] = info.presentationTimeUs;
                }
                codec.queueInputBuffers(index, infos);
                profiler.add("codec_input_queue", System.nanoTime() - operationStarted);
                pcmAssembler.addInputTimes(batchTimes);
                if (batchTimes.length >= 2) {
                    pcmAssembler.setFramesPerUnit((int) Math.max(1, Math.round(
                            (batchTimes[1] - batchTimes[0]) * sampleRate / 1_000_000.0
                    )));
                }
                inputAccessUnits += infos.size();
                inputBatches++;
            } catch (Throwable error) {
                fail(error);
            } finally {
                callbackCpuNanos += Debug.threadCpuTimeNanos() - callbackCpuStarted;
            }
        }

        @Override
        public void onOutputBufferAvailable(
                MediaCodec codec,
                int index,
                MediaCodec.BufferInfo info
        ) {
            ArrayDeque<MediaCodec.BufferInfo> infos = new ArrayDeque<>();
            infos.add(info);
            handleOutputBuffers(codec, index, infos);
        }

        @Override
        public void onOutputBuffersAvailable(
                MediaCodec codec,
                int index,
                ArrayDeque<MediaCodec.BufferInfo> infos
        ) {
            handleOutputBuffers(codec, index, infos);
        }

        private void handleOutputBuffers(
                MediaCodec codec,
                int index,
                ArrayDeque<MediaCodec.BufferInfo> infos
        ) {
            long callbackCpuStarted = Debug.threadCpuTimeNanos();
            boolean outputEnded = false;
            try {
                if (terminal.get()) return;
                ByteBuffer buffer = codec.getOutputBuffer(index);
                boolean containsPcm = false;
                for (MediaCodec.BufferInfo info : infos) {
                    outputEnded |= (info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0;
                    if (info.size <= 0 || info.presentationTimeUs >= endUs) continue;
                    if (buffer == null) {
                        throw new IOException("Audio decoder returned no output buffer");
                    }
                    containsPcm = true;
                    long operationStarted = System.nanoTime();
                    float[] mono = pcmToMono(buffer, info.offset, info.size, channels, encoding);
                    profiler.add("pcm_copy_and_downmix", System.nanoTime() - operationStarted);
                    operationStarted = System.nanoTime();
                    pcmAssembler.accept(
                            mono,
                            info.presentationTimeUs,
                            sampleRate,
                            this::pushOutputAccessUnit
                    );
                    profiler.add("accumulator_push", System.nanoTime() - operationStarted);
                    reportProgress(info.presentationTimeUs);
                }
                if (containsPcm) outputBatches++;
                if (outputEnded) {
                    long operationStarted = System.nanoTime();
                    pcmAssembler.flush(this::pushOutputAccessUnit);
                    profiler.add("accumulator_push", System.nanoTime() - operationStarted);
                }
            } catch (Throwable error) {
                fail(error);
            } finally {
                try {
                    long operationStarted = System.nanoTime();
                    codec.releaseOutputBuffer(index, false);
                    profiler.add("codec_output_release", System.nanoTime() - operationStarted);
                } catch (Throwable error) {
                    fail(error);
                }
                callbackCpuNanos += Debug.threadCpuTimeNanos() - callbackCpuStarted;
            }
            if (outputEnded) complete();
        }

        private void pushOutputAccessUnit(float[] mono, long presentationTimeUs) {
            int framesToKeep = outputFramesToKeep(
                    mono.length, presentationTimeUs, endUs, sampleRate
            );
            if (framesToKeep != mono.length) {
                mono = java.util.Arrays.copyOf(mono, framesToKeep);
            }
            if (mono.length > 0) {
                accumulator.push(
                        mono,
                        presentationTimeUs / 1_000_000.0 - analysisWindow.start(),
                        sampleRate
                );
            }
            decodedPcmFrames += mono.length;
            outputAccessUnits++;
        }

        private void reportProgress(long presentationTimeUs) {
            long second = Math.max(0, (presentationTimeUs - startUs) / 1_000_000);
            if (second == lastReportedSecond) return;
            lastReportedSecond = second;
            progress.onProgress(
                    "audio",
                    Math.min(0.80, 0.80 * second / analysisDuration),
                    String.format(
                            java.util.Locale.US,
                            "Decoding batched PCM + resampling + per-frame FFT · %d / %.0f s",
                            second,
                            analysisDuration
                    )
            );
        }

        @Override
        public void onOutputFormatChanged(MediaCodec codec, MediaFormat format) {
            long callbackCpuStarted = Debug.threadCpuTimeNanos();
            try {
                sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE);
                channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT);
                pcmAssembler.setGaplessTrimming(
                        format.containsKey(MediaFormat.KEY_ENCODER_DELAY)
                                ? format.getInteger(MediaFormat.KEY_ENCODER_DELAY)
                                : 0,
                        format.containsKey(MediaFormat.KEY_ENCODER_PADDING)
                                ? format.getInteger(MediaFormat.KEY_ENCODER_PADDING)
                                : 0
                );
                if (format.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
                    encoding = format.getInteger(MediaFormat.KEY_PCM_ENCODING);
                }
            } catch (Throwable error) {
                fail(error);
            } finally {
                callbackCpuNanos += Debug.threadCpuTimeNanos() - callbackCpuStarted;
            }
        }

        @Override
        public void onError(MediaCodec codec, MediaCodec.CodecException error) {
            fail(error);
        }

        private void complete() {
            if (terminal.compareAndSet(false, true)) completion.countDown();
        }

        private void fail(Throwable error) {
            failure.compareAndSet(null, error);
            if (terminal.compareAndSet(false, true)) completion.countDown();
        }

        void awaitCompletion() throws IOException {
            try {
                while (!completion.await(25, TimeUnit.MILLISECONDS)) {
                    if (cancelled.getAsBoolean()) fail(new IOException("Analysis cancelled"));
                }
            } catch (InterruptedException error) {
                Thread.currentThread().interrupt();
                throw new IOException("Interrupted while decoding audio", error);
            }
            Throwable error = failure.get();
            if (error == null) return;
            if (error instanceof IOException ioError) throw ioError;
            if (error instanceof RuntimeException runtimeError) throw runtimeError;
            throw new IOException("Batched audio decode failed", error);
        }
    }

    static int outputFramesToKeep(
            int decodedFrames,
            long presentationTimeUs,
            long endUs,
            int sampleRate
    ) {
        long remainingUs = endUs - presentationTimeUs;
        return (int) Math.min(
                decodedFrames,
                Math.max(0, (remainingUs * sampleRate + 999_999) / 1_000_000)
        );
    }

    static String featureSha256(float[] features) {
        MessageDigest digest = newSha256();
        updateFloatDigest(digest, features);
        return hexDigest(digest);
    }

    private static MessageDigest newSha256() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 is unavailable", impossible);
        }
    }

    private static void updateFloatDigest(MessageDigest digest, float[] values) {
        ByteBuffer words = ByteBuffer.allocate(values.length * Integer.BYTES).order(ByteOrder.BIG_ENDIAN);
        for (float value : values) {
            words.putInt(Float.floatToIntBits(value));
        }
        digest.update(words.array());
    }

    private static String hexDigest(MessageDigest digest) {
        StringBuilder result = new StringBuilder(64);
        for (byte value : digest.digest()) result.append(String.format("%02x", value));
        return result.toString();
    }

    private static float[] pcmToMono(ByteBuffer source, int offset, int size, int channels, int encoding)
            throws IOException {
        ByteBuffer buffer = source.duplicate().order(ByteOrder.nativeOrder());
        buffer.position(offset);
        buffer.limit(offset + size);
        int bytesPerSample;
        if (encoding == AudioFormat.ENCODING_PCM_FLOAT) bytesPerSample = 4;
        else if (encoding == AudioFormat.ENCODING_PCM_16BIT) bytesPerSample = 2;
        else throw new IOException("Unsupported decoded PCM encoding: " + encoding);
        int frames = size / bytesPerSample / channels;
        float[] mono = new float[frames];
        for (int frame = 0; frame < frames; frame++) {
            double total = 0;
            for (int channel = 0; channel < channels; channel++) {
                total += encoding == AudioFormat.ENCODING_PCM_FLOAT
                        ? buffer.getFloat()
                        : buffer.getShort() / 32768f;
            }
            mono[frame] = (float) (total / channels);
        }
        return mono;
    }
}
