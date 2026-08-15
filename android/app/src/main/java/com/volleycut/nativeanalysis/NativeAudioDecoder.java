package com.volleycut.nativeanalysis;

import android.content.Context;
import android.media.AudioFormat;
import android.media.MediaCodec;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.net.Uri;
import android.os.Debug;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Map;
import java.util.function.BooleanSupplier;

final class NativeAudioDecoder {
    record Result(
            float[] features,
            String decoderName,
            long decodedPcmFrames,
            long resampledOutputSamples,
            int audioFeatureFrames,
            double threadCpuMilliseconds,
            Map<String, Double> profileMilliseconds
    ) {}

    private static final long CODEC_TIMEOUT_US = 10_000;
    private final Context context;

    NativeAudioDecoder(Context context) {
        this.context = context.getApplicationContext();
    }

    Result decode(
            Uri uri,
            double duration,
            double[] analysisTimes,
            AnalysisTypes.ProgressListener progress,
            BooleanSupplier cancelled
    ) throws IOException {
        long pipelineStartedNanos = System.nanoTime();
        long threadCpuStartedNanos = Debug.threadCpuTimeNanos();
        NanoProfiler profiler = new NanoProfiler();
        MediaExtractor extractor = new MediaExtractor();
        MediaCodec codec = null;
        long durationUs = Math.max(1, Math.round(duration * 1_000_000));
        try {
            long setupStarted = System.nanoTime();
            extractor.setDataSource(context, uri, null);
            int track = NativeVideoDecoder.findTrack(extractor, "audio/");
            if (track < 0) return new Result(
                    new float[analysisTimes.length * FeatureSchema.AUDIO.size()],
                    "none", 0, 0, 0,
                    (Debug.threadCpuTimeNanos() - threadCpuStartedNanos) / 1_000_000.0,
                    Map.of()
            );
            extractor.selectTrack(track);
            MediaFormat inputFormat = extractor.getTrackFormat(track);
            String mime = inputFormat.getString(MediaFormat.KEY_MIME);
            if (mime == null) throw new IOException("Audio track has no MIME type");
            codec = MediaCodec.createDecoderByType(mime);
            String decoderName = codec.getName();
            codec.configure(inputFormat, null, null, 0);
            codec.start();
            profiler.add("setup", System.nanoTime() - setupStarted);

            int sampleRate = inputFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE);
            int channels = inputFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT);
            int encoding = AudioFormat.ENCODING_PCM_16BIT;
            AudioFeatureExtractor accumulator = new AudioFeatureExtractor();
            MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
            boolean inputEnded = false;
            boolean outputEnded = false;
            long lastReportedSecond = -1;
            long decodedPcmFrames = 0;
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
                        int sampleSize = nextSampleTime == -1 || nextSampleTime >= durationUs
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
                if (info.size > 0 && info.presentationTimeUs < durationUs) {
                    operationStarted = System.nanoTime();
                    ByteBuffer buffer = codec.getOutputBuffer(outputIndex);
                    if (buffer == null) throw new IOException("Audio decoder returned no output buffer");
                    float[] mono = pcmToMono(buffer, info.offset, info.size, channels, encoding);
                    long remainingUs = durationUs - info.presentationTimeUs;
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
                    accumulator.push(mono, info.presentationTimeUs / 1_000_000.0, sampleRate);
                    profiler.add("accumulator_push", System.nanoTime() - operationStarted);
                    long second = Math.max(0, info.presentationTimeUs / 1_000_000);
                    if (second != lastReportedSecond) {
                        lastReportedSecond = second;
                        progress.onProgress("audio", Math.min(1, second / duration),
                                "Native audio decode + 16 kHz DSP · " + second + " s");
                    }
                }
                outputEnded = (info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0;
                operationStarted = System.nanoTime();
                codec.releaseOutputBuffer(outputIndex, false);
                profiler.add("codec_output_release", System.nanoTime() - operationStarted);
            }
            long finishStarted = System.nanoTime();
            float[] features = accumulator.finishAndPool(analysisTimes);
            profiler.add("dsp_finish_and_pool_call", System.nanoTime() - finishStarted);
            profiler.appendMilliseconds("dsp/", accumulator.performanceMilliseconds());
            profiler.add("audio_pipeline_wall", System.nanoTime() - pipelineStartedNanos);
            return new Result(
                    features,
                    decoderName,
                    decodedPcmFrames,
                    accumulator.resampledOutputSamples(),
                    accumulator.audioFeatureFrameCount(),
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
