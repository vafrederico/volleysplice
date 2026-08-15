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
            AnalysisTypes.VideoDecoderOptions decoderOptions,
            NativeFeatureCache.Mode cacheMode,
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
        NativeFeatureCache cache = NativeFeatureCache.open(
                context, uri, displayName, media, roi, sourceFrameLimit,
                requestedTimes.length, cacheMode
        );
        timings.put("open", elapsedMs(stage));
        progress.onProgress("opening", 1, String.format(Locale.US,
                "%s · %dx%d · %.1f min · %s",
                media.videoMime(), media.width(), media.height(), media.durationSeconds() / 60, roi.label()
        ));

        stage = System.nanoTime();
        long cacheReadStarted = System.nanoTime();
        NativeFeatureCache.LoadedVisual cachedVisual = cache.loadVisual();
        if (!cache.timesMatch(cachedVisual, requestedTimes)) {
            cachedVisual = NativeFeatureCache.LoadedVisual.empty();
        }
        profile.put("cache/visual_read", elapsedMilliseconds(cacheReadStarted));
        AnalysisTypes.VideoFeatures video;
        NativeFeatureCache.VisualWriter visualWriter = cache.newVisualWriter(cachedVisual);
        if (cachedVisual.complete() && cachedVisual.rows() > 0) {
            progress.onProgress("video", 1, String.format(
                    Locale.US, "Loaded %,d cached visual feature rows", cachedVisual.rows()
            ));
            video = cachedVideoFeatures(cachedVisual);
        } else {
            if (cachedVisual.rows() > 0) {
                progress.onProgress("video", 0, String.format(
                        Locale.US, "Resuming after %,d cached visual feature rows",
                        cachedVisual.rows()
                ));
            }
            try {
                video = new NativeVideoDecoder(context).decode(
                        uri, media, roi, requestedTimes, sourceFrameLimit, decoderOptions,
                        cachedVisual, visualWriter, progress, cancelled::get
                );
                visualWriter.finish(video);
            } catch (IOException | RuntimeException error) {
                visualWriter.checkpoint();
                throw error;
            }
        }
        profile.put("cache/visual_write", visualWriter.writeMilliseconds());
        double[] times = video.analysisTimes();
        double analyzedDurationSeconds = video.analyzedDurationSeconds();
        timings.put("video_decode_and_features", elapsedMs(stage));
        appendProfile(profile, "video/", video.profileMilliseconds());
        profile.put("video/thread_cpu", video.threadCpuMilliseconds());
        profile.put("video/mean_sample_timestamp_error", video.meanSampleTimestampErrorMilliseconds());
        profile.put("video/max_sample_timestamp_error", video.maxSampleTimestampErrorMilliseconds());

        stage = System.nanoTime();
        cacheReadStarted = System.nanoTime();
        float[] cachedAudio = cache.loadAudio(times.length);
        profile.put("cache/audio_read", elapsedMilliseconds(cacheReadStarted));
        NativeAudioDecoder.Result audio;
        if (cachedAudio != null) {
            progress.onProgress("audio", 1, "Loaded cached audio features");
            audio = new NativeAudioDecoder.Result(
                    cachedAudio, "feature-cache", 0, 0, 0, 0, Map.of()
            );
        } else {
            audio = new NativeAudioDecoder(context).decode(
                    uri,
                    analyzedDurationSeconds,
                    times,
                    progress,
                    cancelled::get
            );
            long cacheWriteStarted = System.nanoTime();
            cache.storeAudio(audio.features(), times.length);
            profile.put("cache/audio_write", elapsedMilliseconds(cacheWriteStarted));
        }
        timings.put("audio_decode_and_features", elapsedMs(stage));
        appendProfile(profile, "audio/", audio.profileMilliseconds());
        profile.put("audio/thread_cpu", audio.threadCpuMilliseconds());
        if (cancelled.get()) throw new IOException("Analysis cancelled");

        stage = System.nanoTime();
        cacheReadStarted = System.nanoTime();
        float[] contextual = cache.loadContext(times.length);
        profile.put("cache/context_read", elapsedMilliseconds(cacheReadStarted));
        if (contextual == null) {
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
            contextual = contextResult.values();
            profile.put("context/percentile_ranks", contextResult.percentileRankMilliseconds());
            profile.put("context/absolute_feature_restore", contextResult.absoluteRestoreMilliseconds());
            profile.put("context/context_gather", contextResult.contextGatherMilliseconds());
            long cacheWriteStarted = System.nanoTime();
            cache.storeContext(contextual, times.length);
            profile.put("cache/context_write", elapsedMilliseconds(cacheWriteStarted));
        } else {
            progress.onProgress("normalizing", 1, "Loaded cached contextual features");
        }
        timings.put("contextualize", elapsedMs(stage));

        stage = System.nanoTime();
        progress.onProgress("inference", 0, "Running all-labels v2 model stack on CPU");
        long operation = System.nanoTime();
        ModelRunner allLabelsRunner = new ModelRunner(context, FeatureSchema.ALL_LABELS_V2_MODEL_ID);
        profile.put("inference/all_labels_v2_model_asset_parse", elapsedMilliseconds(operation));
        ModelRunner.RunResult allLabelsResult = allLabelsRunner.runProfiled(
                times, contextual, analyzedDurationSeconds
        );
        appendProfile(profile, "inference/all_labels_v2/", allLabelsResult.profileMilliseconds());
        progress.onProgress("inference", .5, "Running previous production model stack on CPU");
        operation = System.nanoTime();
        ModelRunner previousRunner = new ModelRunner(
                context, FeatureSchema.PREVIOUS_PRODUCTION_MODEL_ID
        );
        profile.put("inference/previous_production_model_asset_parse", elapsedMilliseconds(operation));
        ModelRunner.RunResult previousResult = previousRunner.runProfiled(
                times, contextual, analyzedDurationSeconds
        );
        appendProfile(profile, "inference/previous_production/", previousResult.profileMilliseconds());
        operation = System.nanoTime();
        List<AnalysisTypes.Interval> ranges = ProductionEnsemble.merge(
                allLabelsResult.intervals(), previousResult.intervals()
        );
        profile.put("inference/ensemble_merge", elapsedMilliseconds(operation));
        timings.put("inference", elapsedMs(stage));
        long total = elapsedMs(totalStarted);
        double threadCpuMilliseconds = (Debug.threadCpuTimeNanos() - threadCpuStarted) / 1_000_000.0
                + video.profileMilliseconds().getOrDefault("feature_worker_thread_cpu", 0.0)
                + audio.threadCpuMilliseconds();
        int thermalEnd = power == null ? -1 : power.getCurrentThermalStatus();
        long gcCountDelta = Math.max(0, runtimeStat("art.gc.gc-count") - gcCountStart);
        double gcMillisecondsDelta = Math.max(0, runtimeStat("art.gc.gc-time") - gcTimeStart);
        Runtime runtime = Runtime.getRuntime();
        progress.onProgress("complete", 1, ranges.size() + " merged candidate ranges");
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
                video.decodeOnlySourceFrames(),
                video.decoderOutputFrames(),
                audio.decodedPcmFrames(),
                audio.resampledOutputSamples(),
                audio.audioFeatureFrames(),
                video.decoderName(),
                video.hardwareDecoder(),
                decoderOptions.operatingRate(),
                decoderOptions.priority(),
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
                total,
                cache.stats()
        );
    }

    private static AnalysisTypes.VideoFeatures cachedVideoFeatures(
            NativeFeatureCache.LoadedVisual cached
    ) {
        return new AnalysisTypes.VideoFeatures(
                cached.values(),
                cached.times(),
                cached.analyzedDurationSeconds(),
                cached.sourceFrameLimitReached(),
                cached.decoderName(),
                cached.hardwareDecoder(),
                cached.decodedSourceFrames(),
                cached.decodeOnlySourceFrames(),
                cached.decoderOutputFrames(),
                0,
                0,
                0,
                Map.of("feature_cache_hit", 1.0)
        );
    }

    AnalysisTypes.MediaInfo probe(Uri uri) throws IOException {
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

    String displayName(Uri uri) {
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

    static AnalysisTypes.Roi inferRoi(String filename) {
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
