package com.volleycut.nativeanalysis;

import android.net.Uri;

import java.util.List;
import java.util.Map;

public final class AnalysisTypes {
    public static final double MIN_ANALYSIS_WINDOW_SECONDS = 1.0;

    public record AnalysisWindow(double start, double end) {
        static AnalysisWindow full(double duration) {
            return new AnalysisWindow(0, Math.max(0, duration));
        }

        static AnalysisWindow normalize(AnalysisWindow requested, double duration) {
            double safeDuration = Double.isFinite(duration) ? Math.max(0, duration) : 0;
            if (requested == null || !Double.isFinite(requested.start())
                    || !Double.isFinite(requested.end())) {
                return full(safeDuration);
            }
            double start = Math.max(0, Math.min(safeDuration, requested.start()));
            double end = Math.max(start, Math.min(safeDuration, requested.end()));
            return new AnalysisWindow(start, end);
        }

        boolean isFull(double duration) {
            AnalysisWindow normalized = normalize(this, duration);
            return normalized.start() <= 1e-9
                    && Math.abs(normalized.end() - duration) <= 1e-9;
        }
    }

    public record Roi(double x, double y, double width, double height, String label) {}

    public record VideoDecoderOptions(int operatingRate, int priority) {
        static VideoDecoderOptions defaults() {
            return new VideoDecoderOptions(240, 1);
        }
    }

    public record Interval(double start, double end, float confidence, String agreement) {
        public Interval(double start, double end, float confidence) {
            this(start, end, confidence, null);
        }
    }

    public record Serve(double time, float confidence) {}

    public record MediaInfo(
            double durationSeconds,
            int width,
            int height,
            int rotation,
            String videoMime,
            String audioMime
    ) {}

    public record VideoFeatures(
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

    public record PerformanceStats(
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

    public record AnalysisResult(
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
            long totalMilliseconds,
            NativeFeatureCache.CacheStats featureCache
    ) {}

    public interface ProgressListener {
        void onProgress(String stage, double fraction, String detail);

        default void onPerformance(PerformanceStats stats) {}
    }

    private AnalysisTypes() {}
}
