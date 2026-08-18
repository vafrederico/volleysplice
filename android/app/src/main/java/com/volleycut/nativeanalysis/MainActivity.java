package com.volleycut.nativeanalysis;

import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Intent;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowInsets;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONObject;
import org.opencv.android.OpenCVLoader;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

public final class MainActivity extends Activity {
    private static final String TAG = "VolleyCut";
    private static final String BENCHMARK_TAG = "VolleyCutBenchmark";
    private static final String EXTRA_AUTO_RUN = "benchmark_auto_run";
    private static final String EXTRA_RUN_ID = "benchmark_run_id";
    private static final String EXTRA_SOURCE_FRAME_LIMIT = "benchmark_source_frame_limit";
    private static final String EXTRA_CODEC_OPERATING_RATE = "benchmark_codec_operating_rate";
    private static final String EXTRA_CODEC_PRIORITY = "benchmark_codec_priority";
    private static final String EXTRA_FEATURE_CACHE_MODE = "benchmark_feature_cache_mode";
    private static final String EXTRA_STAGES = "benchmark_stages";
    private static final String EXTRA_DURATION_MILLISECONDS = "benchmark_duration_milliseconds";
    private static final String BENCHMARK_RESULT_FILE = "benchmark-result.json";
    private static final int PICK_VIDEO = 10;
    private static final int ORANGE = Color.rgb(239, 91, 53);
    private static final int INK = Color.rgb(32, 32, 30);
    private static final int PAPER = Color.rgb(247, 244, 238);

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final AtomicBoolean cancelled = new AtomicBoolean();
    private final AtomicBoolean analysisRunning = new AtomicBoolean();
    private Uri selectedUri;
    private boolean automatedRun;
    private String automatedRunId;
    private int sourceFrameLimit = FeatureSchema.BENCHMARK_SOURCE_FRAME_LIMIT;
    private AnalysisTypes.VideoDecoderOptions decoderOptions = AnalysisTypes.VideoDecoderOptions.defaults();
    private NativeFeatureCache.Mode cacheMode = NativeFeatureCache.Mode.USE;
    private AnalysisTypes.AnalysisStages benchmarkStages = AnalysisTypes.AnalysisStages.all();
    private int benchmarkDurationMilliseconds;
    private AnalysisTypes.AnalysisResult lastResult;
    private TextView fileLabel;
    private TextView stageLabel;
    private TextView performanceText;
    private TextView resultText;
    private ProgressBar progressBar;
    private Button analyzeButton;
    private Button cancelButton;
    private Button copyButton;
    private Button editorButton;
    private Button clearCacheButton;
    private CheckBox fullFrame;
    private CheckBox limitSourceFrames;
    private CheckBox useFeatureCache;
    private TextView cacheLabel;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (!OpenCVLoader.initLocal()) {
            Toast.makeText(this, "Could not initialize the native OpenCV runtime", Toast.LENGTH_LONG).show();
        }
        getWindow().setNavigationBarColor(PAPER);
        getWindow().setStatusBarColor(PAPER);
        setContentView(buildUi());
        handleLaunchIntent(getIntent());
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        handleLaunchIntent(intent);
    }

    private View buildUi() {
        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        scroll.setBackgroundColor(PAPER);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(20), dp(20), dp(20), dp(32));
        scroll.addView(root, new ScrollView.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT
        ));
        scroll.setOnApplyWindowInsetsListener((view, insets) -> {
            int top;
            int bottom;
            if (Build.VERSION.SDK_INT >= 30) {
                top = insets.getInsets(WindowInsets.Type.statusBars()).top;
                bottom = insets.getInsets(WindowInsets.Type.navigationBars()).bottom;
            } else {
                top = insets.getSystemWindowInsetTop();
                bottom = insets.getSystemWindowInsetBottom();
            }
            root.setPadding(dp(20), dp(20) + top, dp(20), dp(32) + bottom);
            return insets;
        });

        TextView brand = text("VOLLEYCUT", 14, ORANGE);
        brand.setLetterSpacing(.18f);
        root.addView(brand);
        TextView title = text("Analysis benchmark", 30, INK);
        title.setPadding(0, dp(8), 0, dp(5));
        root.addView(title);
        root.addView(text(
                "Pixel 10 Pro on Android 17 · target API 36\nNative MediaCodec + native OpenCV · no WebCodecs, JavaScript, or WASM",
                15, Color.DKGRAY
        ));

        Button choose = button("Choose video");
        choose.setOnClickListener(view -> chooseVideo());
        root.addView(choose, margins(dp(0), dp(22), 0, 0));
        fileLabel = text("No video selected", 14, Color.DKGRAY);
        root.addView(fileLabel, margins(0, dp(8), 0, 0));

        fullFrame = new CheckBox(this);
        fullFrame.setText("Use full frame (distribution-shift diagnostic)");
        fullFrame.setTextColor(INK);
        fullFrame.setTextSize(15);
        root.addView(fullFrame, margins(0, dp(12), 0, 0));

        limitSourceFrames = new CheckBox(this);
        limitSourceFrames.setText("Stop after 1,000 source frames (benchmark)");
        limitSourceFrames.setTextColor(INK);
        limitSourceFrames.setTextSize(15);
        limitSourceFrames.setChecked(true);
        root.addView(limitSourceFrames, margins(0, dp(4), 0, 0));

        useFeatureCache = new CheckBox(this);
        useFeatureCache.setText("Reuse saved video + audio features");
        useFeatureCache.setTextColor(INK);
        useFeatureCache.setTextSize(15);
        useFeatureCache.setChecked(true);
        root.addView(useFeatureCache, margins(0, dp(4), 0, 0));

        LinearLayout cacheActions = new LinearLayout(this);
        cacheActions.setOrientation(LinearLayout.HORIZONTAL);
        cacheLabel = text(cacheSizeLabel(), 13, Color.DKGRAY);
        cacheActions.addView(cacheLabel, new LinearLayout.LayoutParams(0, dp(48), 1));
        clearCacheButton = button("Clear cache");
        clearCacheButton.setOnClickListener(view -> {
            NativeFeatureCache.clearAll(this);
            cacheLabel.setText(cacheSizeLabel());
            Toast.makeText(this, "Saved features cleared", Toast.LENGTH_SHORT).show();
        });
        cacheActions.addView(clearCacheButton, new LinearLayout.LayoutParams(dp(130), dp(48)));
        root.addView(cacheActions, margins(0, dp(2), 0, 0));

        LinearLayout actions = new LinearLayout(this);
        actions.setOrientation(LinearLayout.HORIZONTAL);
        analyzeButton = button("Run analysis");
        analyzeButton.setEnabled(false);
        analyzeButton.setOnClickListener(view -> runAnalysis());
        cancelButton = button("Cancel");
        cancelButton.setEnabled(false);
        cancelButton.setOnClickListener(view -> cancelled.set(true));
        actions.addView(analyzeButton, new LinearLayout.LayoutParams(0, dp(52), 1));
        LinearLayout.LayoutParams cancelParams = new LinearLayout.LayoutParams(dp(110), dp(52));
        cancelParams.setMarginStart(dp(8));
        actions.addView(cancelButton, cancelParams);
        root.addView(actions, margins(0, dp(15), 0, 0));

        progressBar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progressBar.setMax(1000);
        progressBar.setProgressTintList(android.content.res.ColorStateList.valueOf(ORANGE));
        root.addView(progressBar, margins(0, dp(18), 0, 0));
        stageLabel = text("Ready", 14, Color.DKGRAY);
        root.addView(stageLabel, margins(0, dp(6), 0, 0));

        performanceText = text("Feature speed will appear after the first sampled frame.", 14, INK);
        performanceText.setTypeface(android.graphics.Typeface.MONOSPACE);
        performanceText.setPadding(dp(14), dp(12), dp(14), dp(12));
        performanceText.setBackgroundColor(Color.rgb(237, 234, 227));
        root.addView(performanceText, margins(0, dp(10), 0, 0));

        resultText = text(
                "The result will show native decoder names, per-stage wall time, real-time throughput, and unpadded candidate ranges.",
                14, INK
        );
        resultText.setTextIsSelectable(true);
        resultText.setMovementMethod(new ScrollingMovementMethod());
        resultText.setPadding(dp(14), dp(14), dp(14), dp(14));
        resultText.setBackgroundColor(Color.WHITE);
        root.addView(resultText, margins(0, dp(18), 0, 0));

        copyButton = button("Copy result JSON");
        copyButton.setEnabled(false);
        copyButton.setOnClickListener(view -> copyResult());
        root.addView(copyButton, margins(0, dp(10), 0, 0));

        editorButton = button("Open cut editor");
        boolean canResumeEditor = EditorActivity.createResumeIntent(this) != null;
        editorButton.setEnabled(canResumeEditor);
        if (canResumeEditor) editorButton.setText("Resume cut editor");
        editorButton.setOnClickListener(view -> openEditor());
        root.addView(editorButton, margins(0, dp(8), 0, 0));

        TextView caveat = text(
                "Benchmark note: MediaCodec decodes the source in one asynchronous pass and only exposes the 192×108 YUV frames needed at 4 Hz. "
                        + "The model and decoder match the web bundle; Android YUV conversion and native OpenCV may shift feature values, "
                        + "so compare both wall time and range parity.",
                13, Color.DKGRAY
        );
        caveat.setPadding(0, dp(18), 0, 0);
        root.addView(caveat);
        return scroll;
    }

    private void chooseVideo() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("video/*");
        startActivityForResult(intent, PICK_VIDEO);
    }

    private void handleLaunchIntent(Intent intent) {
        if (intent == null || intent.getData() == null) return;
        selectedUri = intent.getData();
        fileLabel.setText(selectedUri.toString());
        analyzeButton.setEnabled(true);
        copyButton.setEnabled(false);
        editorButton.setEnabled(false);
        editorButton.setText("Open cut editor");
        lastResult = null;

        if (!intent.getBooleanExtra(EXTRA_AUTO_RUN, false)) return;
        automatedRun = true;
        automatedRunId = intent.getStringExtra(EXTRA_RUN_ID);
        if (automatedRunId == null || automatedRunId.isBlank()) {
            automatedRunId = Long.toUnsignedString(System.nanoTime());
        }
        sourceFrameLimit = Math.max(1, Math.min(
                1_000_000,
                intent.getIntExtra(EXTRA_SOURCE_FRAME_LIMIT, FeatureSchema.BENCHMARK_SOURCE_FRAME_LIMIT)
        ));
        limitSourceFrames.setChecked(sourceFrameLimit == FeatureSchema.BENCHMARK_SOURCE_FRAME_LIMIT);
        AnalysisTypes.VideoDecoderOptions defaults = AnalysisTypes.VideoDecoderOptions.defaults();
        int operatingRate = intent.getIntExtra(EXTRA_CODEC_OPERATING_RATE, defaults.operatingRate());
        int priority = intent.getIntExtra(EXTRA_CODEC_PRIORITY, defaults.priority());
        decoderOptions = new AnalysisTypes.VideoDecoderOptions(
                operatingRate > 0 ? operatingRate : -1,
                priority >= 0 && priority <= 1 ? priority : -1
        );
        cacheMode = NativeFeatureCache.Mode.fromWireName(
                intent.getStringExtra(EXTRA_FEATURE_CACHE_MODE)
        );
        benchmarkStages = AnalysisTypes.AnalysisStages.fromWireName(
                intent.getStringExtra(EXTRA_STAGES)
        );
        benchmarkDurationMilliseconds = Math.max(
                0,
                intent.getIntExtra(EXTRA_DURATION_MILLISECONDS, 0)
        );
        useFeatureCache.setChecked(cacheMode != NativeFeatureCache.Mode.BYPASS);
        JSONObject running = new JSONObject();
        try {
            running.put("benchmarkStatus", "running");
            running.put("benchmarkRunId", automatedRunId);
            running.put("sourceUri", selectedUri.toString());
            running.put("sourceFrameLimit", sourceFrameLimit);
            running.put("codecOperatingRate", decoderOptions.operatingRate());
            running.put("codecPriority", decoderOptions.priority());
            running.put("featureCacheMode", cacheMode.wireName());
            running.put("benchmarkStages", benchmarkStages.wireName());
            running.put("benchmarkDurationSeconds", benchmarkDurationMilliseconds / 1000.0);
        } catch (Exception impossible) {
            throw new IllegalStateException(impossible);
        }
        writeAutomationResult(running);
        stageLabel.setText("automation · " + benchmarkStages.wireName());
        runAnalysis();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != PICK_VIDEO || resultCode != RESULT_OK || data == null || data.getData() == null) return;
        selectedUri = data.getData();
        try {
            getContentResolver().takePersistableUriPermission(selectedUri, Intent.FLAG_GRANT_READ_URI_PERMISSION);
        } catch (SecurityException ignored) {}
        automatedRun = false;
        automatedRunId = null;
        sourceFrameLimit = FeatureSchema.BENCHMARK_SOURCE_FRAME_LIMIT;
        limitSourceFrames.setChecked(true);
        decoderOptions = AnalysisTypes.VideoDecoderOptions.defaults();
        cacheMode = NativeFeatureCache.Mode.USE;
        benchmarkStages = AnalysisTypes.AnalysisStages.all();
        benchmarkDurationMilliseconds = 0;
        useFeatureCache.setChecked(true);
        fileLabel.setText(selectedUri.toString());
        analyzeButton.setEnabled(true);
        copyButton.setEnabled(false);
        editorButton.setEnabled(false);
        editorButton.setText("Open cut editor");
        lastResult = null;
    }

    private void runAnalysis() {
        if (selectedUri == null || !analysisRunning.compareAndSet(false, true)) return;
        cancelled.set(false);
        analyzeButton.setEnabled(false);
        cancelButton.setEnabled(true);
        clearCacheButton.setEnabled(false);
        copyButton.setEnabled(false);
        progressBar.setProgress(0);
        performanceText.setText("Starting native decoder…");
        resultText.setText("Analysis running…");
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        try { getWindow().setSustainedPerformanceMode(true); } catch (RuntimeException ignored) {}
        Uri uri = selectedUri;
        boolean useFullFrame = fullFrame.isChecked();
        boolean writeAutomationOutput = automatedRun;
        String runId = automatedRunId;
        int requestedSourceFrameLimit = automatedRun
                ? sourceFrameLimit
                : (limitSourceFrames.isChecked()
                        ? FeatureSchema.BENCHMARK_SOURCE_FRAME_LIMIT
                        : FeatureSchema.FULL_SOURCE_FRAME_LIMIT);
        AnalysisTypes.VideoDecoderOptions requestedDecoderOptions = decoderOptions;
        NativeFeatureCache.Mode requestedCacheMode = automatedRun
                ? cacheMode
                : (useFeatureCache.isChecked()
                        ? NativeFeatureCache.Mode.USE
                        : NativeFeatureCache.Mode.BYPASS);
        AnalysisTypes.AnalysisStages requestedStages = writeAutomationOutput
                ? benchmarkStages
                : AnalysisTypes.AnalysisStages.all();
        AnalysisTypes.AnalysisWindow requestedWindow = writeAutomationOutput
                && benchmarkDurationMilliseconds > 0
                ? new AnalysisTypes.AnalysisWindow(0, benchmarkDurationMilliseconds / 1000.0)
                : null;
        executor.submit(() -> {
            try {
                AnalysisTypes.AnalysisResult result = new AnalysisEngine(this).analyze(
                        uri,
                        useFullFrame,
                        requestedSourceFrameLimit,
                        requestedDecoderOptions,
                        requestedCacheMode,
                        requestedWindow,
                        requestedStages,
                        cancelled,
                        new AnalysisTypes.ProgressListener() {
                            @Override
                            public void onProgress(String stage, double fraction, String detail) {
                                runOnUiThread(() -> {
                                    progressBar.setProgress((int) Math.round(
                                            Math.max(0, Math.min(1, fraction)) * 1000
                                    ));
                                    stageLabel.setText(stage + " · " + detail);
                                });
                            }

                            @Override
                            public void onPerformance(AnalysisTypes.PerformanceStats stats) {
                                runOnUiThread(() -> showLivePerformance(stats));
                            }
                        }
                );
                if (writeAutomationOutput) {
                    JSONObject json = resultJson(result);
                    json.put("benchmarkStatus", "complete");
                    json.put("benchmarkRunId", runId);
                    json.put("codecOperatingRate", requestedDecoderOptions.operatingRate());
                    json.put("codecPriority", requestedDecoderOptions.priority());
                    json.put("featureCacheMode", requestedCacheMode.wireName());
                    json.put("benchmarkStages", requestedStages.wireName());
                    json.put("benchmarkDurationSeconds",
                            requestedWindow == null ? 0 : requestedWindow.end());
                    writeAutomationResult(json);
                    Log.i(BENCHMARK_TAG, String.format(Locale.US,
                            "RESULT runId=%s source=%s frames=%d videoMs=%d totalMs=%d",
                            runId, result.displayName(), result.decodedSourceFrames(),
                            result.stageMilliseconds().getOrDefault("video_decode_and_features", 0L),
                            result.totalMilliseconds()));
                }
                runOnUiThread(() -> showResult(result));
            } catch (Exception error) {
                Log.e(TAG, "Native analysis failed", error);
                if (writeAutomationOutput) writeAutomationError(runId, error);
                runOnUiThread(() -> showError(error));
            } finally {
                analysisRunning.set(false);
                runOnUiThread(() -> {
                    if (writeAutomationOutput && runId != null && runId.equals(automatedRunId)) {
                        automatedRun = false;
                        automatedRunId = null;
                    }
                    finishWorkState();
                });
            }
        });
    }

    private void showResult(AnalysisTypes.AnalysisResult result) {
        lastResult = result;
        copyButton.setEnabled(true);
        editorButton.setEnabled(true);
        editorButton.setText("Open cut editor");
        progressBar.setProgress(1000);
        double totalSeconds = result.totalMilliseconds() / 1000.0;
        double overallRealtime = totalSeconds > 0 ? result.analyzedDurationSeconds() / totalSeconds : 0;
        double videoSeconds = result.stageMilliseconds().getOrDefault("video_decode_and_features", 0L) / 1000.0;
        double generatedVideoSeconds = result.sampleRows() / (double) FeatureSchema.ANALYSIS_FPS;
        double featureFramesPerSecond = videoSeconds > 0 ? result.sampleRows() / videoSeconds : 0;
        double featureRealtime = videoSeconds > 0 ? generatedVideoSeconds / videoSeconds : 0;
        double audioSeconds = result.stageMilliseconds().getOrDefault("audio_decode_and_features", 0L) / 1000.0;
        double audioRealtime = audioSeconds > 0 ? result.analyzedDurationSeconds() / audioSeconds : 0;
        StringBuilder output = new StringBuilder();
        output.append(result.displayName()).append('\n');
        output.append(String.format(Locale.US, "%.2f min · %,d samples · %,d ranges\n\n",
                result.media().durationSeconds() / 60, result.sampleRows(), result.ranges().size()));
        if (result.sourceFrameLimit() == FeatureSchema.FULL_SOURCE_FRAME_LIMIT) {
            output.append(String.format(Locale.US,
                    "Analyzed %.2f s; full-file mode; %,d source frames\n\n",
                    result.analyzedDurationSeconds(),
                    result.decodedSourceFrames()));
        } else {
            output.append(String.format(Locale.US,
                    "Analyzed %.2f s; benchmark cap %,d source frames%s\n\n",
                    result.analyzedDurationSeconds(),
                    result.sourceFrameLimit(),
                    result.sourceFrameLimitReached() ? " (reached)" : " (short clip)"));
        }
        output.append("Video decoder: ").append(result.videoDecoder())
                .append(result.hardwareVideoDecoder() ? " (hardware)" : " (software)").append('\n');
        output.append(String.format(Locale.US,
                "Codec request: operating rate %d fps · priority %d\n",
                result.codecOperatingRate(), result.codecPriority()));
        output.append("Audio decoder: ").append(result.audioDecoder()).append('\n');
        NativeFeatureCache.CacheStats cache = result.featureCache();
        output.append(String.format(Locale.US,
                "Feature cache: %s · video %s · audio %s · context %s · resumed %,d rows · %.1f MiB\n",
                cache.mode(), cache.visualHit() ? "hit" : "miss",
                cache.audioHit() ? "hit" : "miss", cache.contextHit() ? "hit" : "miss",
                cache.resumedVisualRows(),
                cache.bytes() / 1_048_576.0));
        if (!cache.failure().isBlank()) {
            output.append("Cache warning: ").append(cache.failure()).append('\n');
        }
        output.append("ROI: ").append(result.roi().label()).append('\n');
        output.append(String.format(Locale.US,
                "Feature speed: %.2f× real time · %.1f frames/s\n",
                featureRealtime, featureFramesPerSecond));
        output.append(String.format(Locale.US,
                "Overall speed: %.2f× real time · %.2f s total\n",
                overallRealtime, totalSeconds));
        output.append(String.format(Locale.US,
                "Audio speed: %.2f× real time\n", audioRealtime));
        output.append(String.format(Locale.US,
                "Source frames: %,d decoded · %,d discarded\n",
                result.decodedSourceFrames(),
                Math.max(0, result.decodedSourceFrames() - result.sampleRows())));
        output.append(String.format(Locale.US,
                "Decoder suppression: %,d decode-only inputs · %,d output frames\n",
                result.decodeOnlySourceFrames(), result.decoderOutputFrames()));
        output.append(String.format(Locale.US,
                "Source decode: %.1f frames/s · sample every %.1f frames\n",
                videoSeconds > 0 ? result.decodedSourceFrames() / videoSeconds : 0,
                result.sampleRows() > 0 ? result.decodedSourceFrames() / (double) result.sampleRows() : 0));
        output.append(String.format(Locale.US,
                "Audio PCM: %,d source frames · %,d resampled samples · %,d FFT frames\n",
                result.decodedAudioFrames(), result.resampledAudioSamples(), result.audioFeatureFrames()));
        output.append(String.format(Locale.US,
                "Analysis-thread CPU: %.2f s · %.0f%% of wall\n",
                result.threadCpuMilliseconds() / 1000,
                result.totalMilliseconds() > 0
                        ? result.threadCpuMilliseconds() / result.totalMilliseconds() * 100
                        : 0));
        output.append(String.format(Locale.US,
                "Memory: Java %.1f MiB · native %.1f MiB · PSS %.1f MiB · %,d CPUs\n",
                result.javaHeapUsedBytes() / 1_048_576.0,
                result.nativeHeapAllocatedBytes() / 1_048_576.0,
                result.pssKilobytes() / 1024.0,
                result.availableProcessors()));
        output.append(String.format(Locale.US,
                "GC: %,d runs · %.0f ms · thermal %s → %s\n",
                result.gcCountDelta(), result.gcMillisecondsDelta(),
                thermalLabel(result.thermalStatusStart()), thermalLabel(result.thermalStatusEnd())));
        output.append(String.format(Locale.US,
                "Sample timestamp error: mean %.2f ms · max %.2f ms\n",
                result.profileMilliseconds().getOrDefault("video/mean_sample_timestamp_error", 0.0),
                result.profileMilliseconds().getOrDefault("video/max_sample_timestamp_error", 0.0)));
        for (Map.Entry<String, Long> stage : result.stageMilliseconds().entrySet()) {
            output.append(String.format(Locale.US, "  %-27s %8.2f s\n", stage.getKey(), stage.getValue() / 1000.0));
        }
        appendProfileSection(output, result, "video/", "VIDEO PIPELINE", videoSeconds * 1000);
        appendProfileSection(output, result, "audio/", "AUDIO PIPELINE", audioSeconds * 1000);
        appendProfileSection(
                output, result, "context/", "CONTEXT PIPELINE",
                result.stageMilliseconds().getOrDefault("contextualize", 0L)
        );
        appendProfileSection(
                output, result, "inference/", "INFERENCE PIPELINE",
                result.stageMilliseconds().getOrDefault("inference", 0L)
        );
        appendOptimizationSignals(output, result, videoSeconds * 1000);
        output.append("\nUnpadded ranges\n");
        for (int i = 0; i < result.ranges().size(); i++) {
            AnalysisTypes.Interval range = result.ranges().get(i);
            output.append(String.format(Locale.US, "R%03d  %s – %s  %.3f  %s\n",
                    i + 1, clock(range.start()), clock(range.end()), range.confidence(),
                    range.agreement() == null ? "single-model" : range.agreement()));
        }
        resultText.setText(output.toString());
        performanceText.setText(String.format(Locale.US,
                "FEATURE  %6.2f× real time   %7.1f frames/s\nOVERALL  %6.2f× real time   %7.2f s",
                featureRealtime, featureFramesPerSecond, overallRealtime, totalSeconds));
        stageLabel.setText("complete · " + result.ranges().size() + " candidate ranges");
    }

    private void showLivePerformance(AnalysisTypes.PerformanceStats stats) {
        String eta = Double.isFinite(stats.etaSeconds())
                ? clock(stats.etaSeconds())
                : "estimating";
        performanceText.setText(String.format(Locale.US,
                "%6.2f× real time   %7.1f frames/s\n"
                        + "%,d sampled   %,d / %,d source decoded\n"
                        + "%s elapsed   %s ETA   heap %.0f / %.0f MiB",
                stats.realtimeRatio(),
                stats.framesPerSecond(),
                stats.generatedFrames(),
                stats.decodedSourceFrames(),
                stats.totalFrames(),
                clock(stats.elapsedSeconds()),
                eta,
                stats.usedHeapBytes() / 1_048_576.0,
                stats.maxHeapBytes() / 1_048_576.0));
    }

    private void showError(Exception error) {
        String message = error.getMessage() == null ? error.getClass().getSimpleName() : error.getMessage();
        stageLabel.setText(cancelled.get() ? "cancelled" : "error");
        resultText.setText((cancelled.get() ? "Analysis cancelled." : "Analysis failed.") + "\n\n" + message);
    }

    private void finishWorkState() {
        analyzeButton.setEnabled(selectedUri != null);
        cancelButton.setEnabled(false);
        clearCacheButton.setEnabled(true);
        cacheLabel.setText(cacheSizeLabel());
        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        try { getWindow().setSustainedPerformanceMode(false); } catch (RuntimeException ignored) {}
    }

    private void copyResult() {
        if (lastResult == null) return;
        ClipboardManager clipboard = getSystemService(ClipboardManager.class);
        clipboard.setPrimaryClip(ClipData.newPlainText("VolleyCut analysis", resultJson(lastResult).toString()));
        Toast.makeText(this, "Result JSON copied", Toast.LENGTH_SHORT).show();
    }

    private static JSONObject resultJson(AnalysisTypes.AnalysisResult result) {
        JSONObject json = new JSONObject();
        try {
            json.put("schemaVersion", 1);
            json.put("method", "android-mediacodec-av-cache-opencv-ensemble-v6");
            json.put("modelId", FeatureSchema.MODEL_ID);
            json.put("sourceName", result.displayName());
            json.put("duration", result.media().durationSeconds());
            json.put("analyzedDuration", result.analyzedDurationSeconds());
            json.put("sourceFrameLimit", result.sourceFrameLimit());
            json.put("sourceFrameLimitReached", result.sourceFrameLimitReached());
            json.put("analysisFps", FeatureSchema.ANALYSIS_FPS);
            json.put("sampleRows", result.sampleRows());
            json.put("decodedSourceFrames", result.decodedSourceFrames());
            json.put("decodeOnlySourceFrames", result.decodeOnlySourceFrames());
            json.put("decoderOutputFrames", result.decoderOutputFrames());
            json.put("decodedAudioFrames", result.decodedAudioFrames());
            json.put("resampledAudioSamples", result.resampledAudioSamples());
            json.put("audioFeatureFrames", result.audioFeatureFrames());
            json.put("device", Build.MANUFACTURER + " " + Build.MODEL);
            json.put("androidRelease", Build.VERSION.RELEASE);
            json.put("sdkInt", Build.VERSION.SDK_INT);
            json.put("videoDecoder", result.videoDecoder());
            json.put("hardwareVideoDecoder", result.hardwareVideoDecoder());
            json.put("codecOperatingRate", result.codecOperatingRate());
            json.put("codecPriority", result.codecPriority());
            json.put("audioDecoder", result.audioDecoder());
            NativeFeatureCache.CacheStats cache = result.featureCache();
            JSONObject cacheJson = new JSONObject();
            cacheJson.put("mode", cache.mode());
            cacheJson.put("key", cache.key());
            cacheJson.put("visualHit", cache.visualHit());
            cacheJson.put("audioHit", cache.audioHit());
            cacheJson.put("contextHit", cache.contextHit());
            cacheJson.put("resumedVisualRows", cache.resumedVisualRows());
            cacheJson.put("savedVisualRows", cache.savedVisualRows());
            cacheJson.put("complete", cache.complete());
            cacheJson.put("bytes", cache.bytes());
            cacheJson.put("failure", cache.failure());
            json.put("featureCache", cacheJson);
            json.put("totalMilliseconds", result.totalMilliseconds());
            double videoMilliseconds = result.stageMilliseconds().getOrDefault(
                    "video_decode_and_features", 0L
            );
            double generatedVideoSeconds = result.sampleRows() / (double) FeatureSchema.ANALYSIS_FPS;
            json.put("featureFramesPerSecond",
                    videoMilliseconds > 0 ? result.sampleRows() / (videoMilliseconds / 1000) : 0);
            json.put("featureRealtimeRatio",
                    videoMilliseconds > 0 ? generatedVideoSeconds / (videoMilliseconds / 1000) : 0);
            double audioMilliseconds = result.stageMilliseconds().getOrDefault(
                    "audio_decode_and_features", 0L
            );
            json.put("audioRealtimeRatio",
                    audioMilliseconds > 0
                            ? result.analyzedDurationSeconds() / (audioMilliseconds / 1000)
                            : 0);
            json.put("overallRealtimeRatio",
                    result.totalMilliseconds() > 0
                            ? result.analyzedDurationSeconds() / (result.totalMilliseconds() / 1000.0)
                            : 0);
            json.put("analysisThreadCpuMilliseconds", result.threadCpuMilliseconds());
            json.put("thermalStatusStart", result.thermalStatusStart());
            json.put("thermalStatusEnd", result.thermalStatusEnd());
            json.put("gcCountDelta", result.gcCountDelta());
            json.put("gcMillisecondsDelta", result.gcMillisecondsDelta());
            json.put("javaHeapUsedBytes", result.javaHeapUsedBytes());
            json.put("nativeHeapAllocatedBytes", result.nativeHeapAllocatedBytes());
            json.put("pssKilobytes", result.pssKilobytes());
            json.put("availableProcessors", result.availableProcessors());
            JSONObject stage = new JSONObject();
            for (Map.Entry<String, Long> value : result.stageMilliseconds().entrySet()) stage.put(value.getKey(), value.getValue());
            json.put("stageMilliseconds", stage);
            JSONObject profile = new JSONObject();
            for (Map.Entry<String, Double> value : result.profileMilliseconds().entrySet()) {
                profile.put(value.getKey(), value.getValue());
            }
            json.put("profileMilliseconds", profile);
            JSONObject roi = new JSONObject();
            roi.put("x", result.roi().x());
            roi.put("y", result.roi().y());
            roi.put("width", result.roi().width());
            roi.put("height", result.roi().height());
            roi.put("label", result.roi().label());
            json.put("roi", roi);
            JSONArray ranges = new JSONArray();
            for (AnalysisTypes.Interval range : result.ranges()) {
                JSONObject item = new JSONObject();
                item.put("start", range.start());
                item.put("end", range.end());
                item.put("confidence", range.confidence());
                item.put("agreement", range.agreement());
                ranges.put(item);
            }
            json.put("ranges", ranges);
        } catch (Exception impossible) {
            throw new IllegalStateException(impossible);
        }
        return json;
    }

    private void writeAutomationError(String runId, Exception error) {
        JSONObject json = new JSONObject();
        try {
            json.put("benchmarkStatus", cancelled.get() ? "cancelled" : "failed");
            json.put("benchmarkRunId", runId);
            json.put("errorType", error.getClass().getName());
            json.put("errorMessage", error.getMessage() == null ? "" : error.getMessage());
        } catch (Exception impossible) {
            throw new IllegalStateException(impossible);
        }
        writeAutomationResult(json);
        Log.e(BENCHMARK_TAG, "ERROR runId=" + runId + " message=" + error.getMessage());
    }

    private void writeAutomationResult(JSONObject json) {
        File target = getFileStreamPath(BENCHMARK_RESULT_FILE);
        File temporary = getFileStreamPath(BENCHMARK_RESULT_FILE + ".tmp");
        try {
            Files.write(temporary.toPath(), json.toString().getBytes(StandardCharsets.UTF_8));
            try {
                Files.move(
                        temporary.toPath(), target.toPath(),
                        StandardCopyOption.REPLACE_EXISTING,
                        StandardCopyOption.ATOMIC_MOVE
                );
            } catch (IOException atomicMoveUnsupported) {
                Files.move(
                        temporary.toPath(), target.toPath(),
                        StandardCopyOption.REPLACE_EXISTING
                );
            }
        } catch (IOException error) {
            Log.e(BENCHMARK_TAG, "Could not write automation result", error);
        }
    }

    private static void appendProfileSection(
            StringBuilder output,
            AnalysisTypes.AnalysisResult result,
            String prefix,
            String title,
            double wallMilliseconds
    ) {
        output.append("\n").append(title).append("\n");
        for (Map.Entry<String, Double> entry : result.profileMilliseconds().entrySet()) {
            if (!entry.getKey().startsWith(prefix)) continue;
            String name = entry.getKey().substring(prefix.length());
            if (name.equals("mean_sample_timestamp_error")
                    || name.equals("max_sample_timestamp_error")) continue;
            int depth = Math.max(0, name.split("/").length - 1);
            String label = name.substring(name.lastIndexOf('/') + 1).replace('_', ' ');
            double milliseconds = entry.getValue();
            double percent = wallMilliseconds > 0 ? milliseconds / wallMilliseconds * 100 : 0;
            int units = prefix.equals("video/")
                    ? result.sampleRows()
                    : prefix.equals("audio/")
                            ? result.audioFeatureFrames()
                            : result.sampleRows();
            double perUnit = units > 0 ? milliseconds / units : 0;
            output.append("  ").append("  ".repeat(depth));
            output.append(String.format(Locale.US,
                    "%-31s %9.2f ms  %5.1f%%  %6.3f ms/unit\n",
                    label, milliseconds, percent, perUnit));
        }
    }

    private static void appendOptimizationSignals(
            StringBuilder output,
            AnalysisTypes.AnalysisResult result,
            double videoWallMilliseconds
    ) {
        Map<String, Double> profile = result.profileMilliseconds();
        double codecWait = profile.getOrDefault("video/codec_input_dequeue", 0.0)
                + profile.getOrDefault("video/codec_output_dequeue_wait", 0.0);
        double yuv = profile.getOrDefault("video/yuv_crop_scale_color", 0.0);
        double openCv = profile.getOrDefault("video/opencv_feature_call", 0.0);
        double workerQueueWait = profile.getOrDefault("video/feature_queue_backpressure", 0.0);
        double workerFinishWait = profile.getOrDefault("video/feature_worker_finish_wait", 0.0);
        double demux = profile.getOrDefault("video/demux_read", 0.0)
                + profile.getOrDefault("video/demux_advance", 0.0);
        output.append("\nOPTIMIZATION SIGNALS\n");
        if (result.decodeOnlySourceFrames() > 0) {
            double asyncInput = profile.getOrDefault("video/codec_input_callback", 0.0);
            output.append(String.format(Locale.US,
                    "  Async decode-only: %,d inputs suppressed; %,d decoder outputs produced.\n",
                    result.decodeOnlySourceFrames(), result.decoderOutputFrames()));
            output.append(String.format(Locale.US,
                    "  Serial codec input callbacks: %.1f%%; OpenCV critical waits: queue %.2f ms, drain %.2f ms.\n",
                    percent(asyncInput, videoWallMilliseconds), workerQueueWait, workerFinishWait));
            output.append("  Next lead: reduce per-access-unit codec queue overhead; OpenCV remains off the critical path.\n");
            return;
        }
        double maximum = Math.max(Math.max(codecWait, yuv), Math.max(openCv, demux));
        if (maximum == codecWait) {
            output.append("  Largest bucket: codec waits. Test asynchronous codec callbacks or a Surface/GPU sampling path.\n");
        } else if (maximum == yuv) {
            output.append("  Largest bucket: scalar YUV crop/scale/color. Precomputed coordinates plus JNI/libyuv is the clearest next spike.\n");
        } else if (maximum == openCv) {
            output.append("  Largest bucket: OpenCV feature compute. Use its largest nested substage to choose the next kernel optimization.\n");
        } else {
            output.append("  Largest bucket: demux. Storage access and extractor batching deserve the next measurement.\n");
        }
        output.append(String.format(Locale.US,
                "  Buckets: codec %.1f%% · YUV %.1f%% · OpenCV %.1f%% · demux %.1f%%\n",
                percent(codecWait, videoWallMilliseconds), percent(yuv, videoWallMilliseconds),
                percent(openCv, videoWallMilliseconds), percent(demux, videoWallMilliseconds)));
        output.append(String.format(Locale.US,
                "  Parallel OpenCV critical-path waits: queue %.2f ms; final drain %.2f ms\n",
                workerQueueWait, workerFinishWait));
        if (result.gcCountDelta() > 0) {
            output.append(String.format(Locale.US,
                    "  Allocation pressure: %,d GC runs / %.0f ms; buffer reuse may reduce jitter.\n",
                    result.gcCountDelta(), result.gcMillisecondsDelta()));
        }
        if (result.thermalStatusEnd() > result.thermalStatusStart()) {
            output.append("  Thermal status increased during the run; compare repeated runs before treating a delta as algorithmic.\n");
        }
    }

    private static double percent(double value, double denominator) {
        return denominator > 0 ? value / denominator * 100 : 0;
    }

    private static String thermalLabel(int status) {
        return switch (status) {
            case 0 -> "none";
            case 1 -> "light";
            case 2 -> "moderate";
            case 3 -> "severe";
            case 4 -> "critical";
            case 5 -> "emergency";
            case 6 -> "shutdown";
            default -> "unknown";
        };
    }

    private TextView text(String value, int sp, int color) {
        TextView view = new TextView(this);
        view.setText(value);
        view.setTextSize(sp);
        view.setTextColor(color);
        view.setLineSpacing(0, 1.12f);
        return view;
    }

    private Button button(String value) {
        Button button = new Button(this);
        button.setText(value);
        button.setTextSize(14);
        button.setAllCaps(false);
        button.setGravity(Gravity.CENTER);
        return button;
    }

    private void openEditor() {
        Intent intent = lastResult != null
                ? EditorActivity.createIntent(this, lastResult)
                : EditorActivity.createResumeIntent(this);
        if (intent != null) startActivity(intent);
    }

    private String cacheSizeLabel() {
        return String.format(
                Locale.US,
                "Saved features: %.1f MiB",
                NativeFeatureCache.totalBytes(this) / 1_048_576.0
        );
    }

    private LinearLayout.LayoutParams margins(int left, int top, int right, int bottom) {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT
        );
        params.setMargins(left, top, right, bottom);
        return params;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private static String clock(double seconds) {
        int minutes = (int) Math.floor(Math.max(0, seconds) / 60);
        double remainder = Math.max(0, seconds) - minutes * 60;
        return String.format(Locale.US, "%d:%05.2f", minutes, remainder);
    }

    @Override
    protected void onDestroy() {
        cancelled.set(true);
        executor.shutdownNow();
        super.onDestroy();
    }
}
