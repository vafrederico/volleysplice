package com.volleycut.nativeanalysis;

import android.net.Uri;

import java.util.List;
import java.util.Map;

final class AnalysisTypes {
    record Roi(double x, double y, double width, double height, String label) {}

    record VideoDecoderOptions(int operatingRate, int priority) {
        static VideoDecoderOptions defaults() {
            return new VideoDecoderOptions(240, 1);
        }
    }

    record Interval(double start, double end, float confidence) {}

    record Serve(double time, float confidence) {}

    record MediaInfo(
            double durationSeconds,
            int width,
            int height,
            int rotation,
            String videoMime,
            String audioMime
    ) {}

    record VideoFeatures(
            float[] values,
            double[] analysisTimes,
            double analyzedDurationSeconds,
            boolean sourceFrameLimitReached,
            String decoderName,
            boolean hardwareDecoder,
            int decodedSourceFrames,
            int decodeOnlySourceFrames,
            int decoderOutputFrames,
            double threadCpuMilliseconds,
            double meanSampleTimestampErrorMilliseconds,
            double maxSampleTimestampErrorMilliseconds,
            Map<String, Double> profileMilliseconds
    ) {}

    record PerformanceStats(
            int generatedFrames,
            int totalFrames,
            int decodedSourceFrames,
            double generatedVideoSeconds,
            double elapsedSeconds,
            double framesPerSecond,
            double realtimeRatio,
            double etaSeconds,
            long usedHeapBytes,
            long maxHeapBytes
    ) {}

    record AnalysisResult(
            Uri source,
            String displayName,
            MediaInfo media,
            Roi roi,
            double analyzedDurationSeconds,
            int sourceFrameLimit,
            boolean sourceFrameLimitReached,
            int sampleRows,
            int decodedSourceFrames,
            int decodeOnlySourceFrames,
            int decoderOutputFrames,
            long decodedAudioFrames,
            long resampledAudioSamples,
            int audioFeatureFrames,
            String videoDecoder,
            boolean hardwareVideoDecoder,
            int codecOperatingRate,
            int codecPriority,
            String audioDecoder,
            List<Interval> ranges,
            Map<String, Long> stageMilliseconds,
            Map<String, Double> profileMilliseconds,
            double threadCpuMilliseconds,
            int thermalStatusStart,
            int thermalStatusEnd,
            long gcCountDelta,
            double gcMillisecondsDelta,
            long javaHeapUsedBytes,
            long nativeHeapAllocatedBytes,
            long pssKilobytes,
            int availableProcessors,
            long totalMilliseconds
    ) {}

    interface ProgressListener {
        void onProgress(String stage, double fraction, String detail);

        default void onPerformance(PerformanceStats stats) {}
    }

    private AnalysisTypes() {}
}
