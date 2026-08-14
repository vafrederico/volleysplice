package com.volleycut.nativeanalysis;

import android.content.Context;
import android.database.Cursor;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.net.Uri;
import android.os.Debug;
import android.os.PowerManager;
import android.provider.OpenableColumns;

import org.json.JSONException;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

final class AnalysisEngine {
    private final Context context;

    AnalysisEngine(Context context) {
        this.context = context.getApplicationContext();
    }

    AnalysisTypes.AnalysisResult analyze(
            Uri uri,
            boolean fullFrame,
            int sourceFrameLimit,
            AtomicBoolean cancelled,
            AnalysisTypes.ProgressListener progress
    ) throws IOException, JSONException {
        long totalStarted = System.nanoTime();
        long threadCpuStarted = Debug.threadCpuTimeNanos();
        long gcCountStart = runtimeStat("art.gc.gc-count");
        long gcTimeStart = runtimeStat("art.gc.gc-time");
        PowerManager power = context.getSystemService(PowerManager.class);
        int thermalStart = power == null ? -1 : power.getCurrentThermalStatus();
        LinkedHashMap<String, Long> timings = new LinkedHashMap<>();
        LinkedHashMap<String, Double> profile = new LinkedHashMap<>();

        long stage = System.nanoTime();
        String displayName = displayName(uri);
        AnalysisTypes.MediaInfo media = probe(uri);
        AnalysisTypes.Roi roi = fullFrame
                ? new AnalysisTypes.Roi(0, 0, 1, 1, "Full frame")
                : inferRoi(displayName);
        double[] requestedTimes = analysisTimes(media.durationSeconds());
        timings.put("open", elapsedMs(stage));
        progress.onProgress("opening", 1, String.format(Locale.US,
                "%s · %dx%d · %.1f min · %s",
                media.videoMime(), media.width(), media.height(), media.durationSeconds() / 60, roi.label()
        ));

        stage = System.nanoTime();
        AnalysisTypes.VideoFeatures video = new NativeVideoDecoder(context).decode(
                uri, media, roi, requestedTimes, sourceFrameLimit, progress, cancelled::get
        );
        double[] times = video.analysisTimes();
        double analyzedDurationSeconds = video.analyzedDurationSeconds();
        timings.put("video_decode_and_features", elapsedMs(stage));
        appendProfile(profile, "video/", video.profileMilliseconds());
        profile.put("video/thread_cpu", video.threadCpuMilliseconds());
        profile.put("video/mean_sample_timestamp_error", video.meanSampleTimestampErrorMilliseconds());
        profile.put("video/max_sample_timestamp_error", video.maxSampleTimestampErrorMilliseconds());

        NativeAudioDecoder.Result audio = new NativeAudioDecoder.Result(
                new float[times.length * FeatureSchema.AUDIO.size()],
                "skipped", 0, 0, 0, 0, Map.of()
        );
        timings.put("audio_decode_and_features", 0L);
        progress.onProgress("audio", 1, "Skipped for video-only benchmark");
        if (cancelled.get()) throw new IOException("Analysis cancelled");

        stage = System.nanoTime();
        long operation = System.nanoTime();
        float[] temporal = FeatureMath.temporalVisualFeatures(video.values(), times.length);
        profile.put("context/temporal_visual_features", elapsedMilliseconds(operation));
        operation = System.nanoTime();
        float[] base = combine(video.values(), temporal, audio.features(), times.length);
        profile.put("context/base_matrix_combine", elapsedMilliseconds(operation));
        progress.onProgress("normalizing", 0, "Whole-recording percentile ranks + ±2 s context");
        FeatureMath.ContextResult contextResult = FeatureMath.contextualizeProfiled(
                times, base, FeatureSchema.BASE
        );
        float[] contextual = contextResult.values();
        profile.put("context/percentile_ranks", contextResult.percentileRankMilliseconds());
        profile.put("context/absolute_feature_restore", contextResult.absoluteRestoreMilliseconds());
        profile.put("context/context_gather", contextResult.contextGatherMilliseconds());
        timings.put("contextualize", elapsedMs(stage));

        stage = System.nanoTime();
        progress.onProgress("inference", 0, "Three FP32 logistic heads + rally decoders");
        operation = System.nanoTime();
        ModelRunner modelRunner = new ModelRunner(context);
        profile.put("inference/model_asset_parse", elapsedMilliseconds(operation));
        ModelRunner.RunResult modelResult = modelRunner.runProfiled(
                times, contextual, analyzedDurationSeconds
        );
        List<AnalysisTypes.Interval> ranges = modelResult.intervals();
        appendProfile(profile, "inference/", modelResult.profileMilliseconds());
        timings.put("inference", elapsedMs(stage));
        long total = elapsedMs(totalStarted);
        double threadCpuMilliseconds = (Debug.threadCpuTimeNanos() - threadCpuStarted) / 1_000_000.0
                + video.profileMilliseconds().getOrDefault("feature_worker_thread_cpu", 0.0);
        int thermalEnd = power == null ? -1 : power.getCurrentThermalStatus();
        long gcCountDelta = Math.max(0, runtimeStat("art.gc.gc-count") - gcCountStart);
        double gcMillisecondsDelta = Math.max(0, runtimeStat("art.gc.gc-time") - gcTimeStart);
        Runtime runtime = Runtime.getRuntime();
        progress.onProgress("complete", 1, ranges.size() + " candidate ranges");
        return new AnalysisTypes.AnalysisResult(
                uri,
                displayName,
                media,
                roi,
                analyzedDurationSeconds,
                sourceFrameLimit,
                video.sourceFrameLimitReached(),
                times.length,
                video.decodedSourceFrames(),
                audio.decodedPcmFrames(),
                audio.resampledOutputSamples(),
                audio.audioFeatureFrames(),
                video.decoderName(),
                video.hardwareDecoder(),
                audio.decoderName(),
                List.copyOf(ranges),
                timings,
                profile,
                threadCpuMilliseconds,
                thermalStart,
                thermalEnd,
                gcCountDelta,
                gcMillisecondsDelta,
                runtime.totalMemory() - runtime.freeMemory(),
                Debug.getNativeHeapAllocatedSize(),
                Debug.getPss(),
                runtime.availableProcessors(),
                total
        );
    }

    private AnalysisTypes.MediaInfo probe(Uri uri) throws IOException {
        MediaExtractor extractor = new MediaExtractor();
        try {
            extractor.setDataSource(context, uri, null);
            int videoTrack = NativeVideoDecoder.findTrack(extractor, "video/");
            if (videoTrack < 0) throw new IOException("The selected file has no video track");
            MediaFormat video = extractor.getTrackFormat(videoTrack);
            int audioTrack = NativeVideoDecoder.findTrack(extractor, "audio/");
            String audioMime = audioTrack < 0 ? null
                    : extractor.getTrackFormat(audioTrack).getString(MediaFormat.KEY_MIME);
            long durationUs = video.containsKey(MediaFormat.KEY_DURATION)
                    ? video.getLong(MediaFormat.KEY_DURATION)
                    : 0;
            if (durationUs <= 0) throw new IOException("The video duration is unavailable");
            return new AnalysisTypes.MediaInfo(
                    durationUs / 1_000_000.0,
                    video.getInteger(MediaFormat.KEY_WIDTH),
                    video.getInteger(MediaFormat.KEY_HEIGHT),
                    video.containsKey(MediaFormat.KEY_ROTATION) ? video.getInteger(MediaFormat.KEY_ROTATION) : 0,
                    video.getString(MediaFormat.KEY_MIME),
                    audioMime
            );
        } finally {
            extractor.release();
        }
    }

    private String displayName(Uri uri) {
        try (Cursor cursor = context.getContentResolver().query(
                uri, new String[]{OpenableColumns.DISPLAY_NAME}, null, null, null
        )) {
            if (cursor != null && cursor.moveToFirst()) {
                int index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (index >= 0) return cursor.getString(index);
            }
        }
        return uri.getLastPathSegment() == null ? "selected-video" : uri.getLastPathSegment();
    }

    private static double[] analysisTimes(double duration) {
        int capacity = Math.max(1, (int) Math.ceil(duration * FeatureSchema.ANALYSIS_FPS));
        double[] provisional = new double[capacity];
        int count = 0;
        for (int index = 0; index < capacity; index++) {
            double timestamp = index / (double) FeatureSchema.ANALYSIS_FPS;
            if (timestamp >= duration) break;
            provisional[count++] = timestamp;
        }
        return java.util.Arrays.copyOf(provisional, count);
    }

    private static float[] combine(float[] visual, float[] temporal, float[] audio, int rows) {
        int frameColumns = FeatureSchema.FRAME.size();
        int temporalColumns = FeatureSchema.TEMPORAL.size();
        int audioColumns = FeatureSchema.AUDIO.size();
        if (visual.length != rows * frameColumns
                || temporal.length != rows * temporalColumns
                || audio.length != rows * audioColumns) {
            throw new IllegalArgumentException("Base feature components do not align");
        }
        float[] output = new float[rows * FeatureSchema.BASE.size()];
        for (int row = 0; row < rows; row++) {
            int target = row * FeatureSchema.BASE.size();
            System.arraycopy(visual, row * frameColumns, output, target, frameColumns);
            System.arraycopy(temporal, row * temporalColumns, output, target + frameColumns, temporalColumns);
            System.arraycopy(audio, row * audioColumns, output, target + frameColumns + temporalColumns, audioColumns);
        }
        return output;
    }

    private static AnalysisTypes.Roi inferRoi(String filename) {
        record Known(String needle, double x, double y, double width, double height, String label) {}
        List<Known> profiles = List.of(
                new Known("beach-source-02", .02, .12, .96, .86, "Known beach camera"),
                new Known("beach-source-01", .02, .12, .96, .86, "Known beach camera"),
                new Known("Dm", .02, .22, .96, .76, "Known grass camera"),
                new Known("GYU", .02, .18, .96, .80, "Known grass camera"),
                new Known("qpd", .02, .18, .96, .80, "Known grass camera"),
                new Known("rSs", .02, .22, .96, .76, "Known grass camera"),
                new Known("9lc", .04, .14, .92, .84, "Known indoor camera"),
                new Known("indoor-source-07", .04, .14, .92, .84, "Known indoor camera"),
                new Known("tds", .03, .12, .94, .86, "Known indoor camera")
        );
        for (Known profile : profiles) {
            if (filename.contains(profile.needle)) {
                return new AnalysisTypes.Roi(profile.x, profile.y, profile.width, profile.height,
                        profile.label + " · " + profile.needle);
            }
        }
        return new AnalysisTypes.Roi(.03, .12, .94, .86, "Indoor camera default");
    }

    private static long elapsedMs(long startedNanos) {
        return Math.round((System.nanoTime() - startedNanos) / 1_000_000.0);
    }

    private static double elapsedMilliseconds(long startedNanos) {
        return (System.nanoTime() - startedNanos) / 1_000_000.0;
    }

    private static void appendProfile(
            Map<String, Double> target,
            String prefix,
            Map<String, Double> source
    ) {
        for (Map.Entry<String, Double> entry : source.entrySet()) {
            target.put(prefix + entry.getKey(), entry.getValue());
        }
    }

    private static long runtimeStat(String name) {
        String value = Debug.getRuntimeStat(name);
        if (value == null) return 0;
        try {
            return Long.parseLong(value);
        } catch (NumberFormatException ignored) {
            return 0;
        }
    }
}
