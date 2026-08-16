package com.volleycut.nativeanalysis;

import android.content.Context;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.io.InputStream;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

final class ModelRunner {
    record RunResult(
            List<AnalysisTypes.Interval> intervals,
            float[] rallyProbabilities,
            float[] serveProbabilities,
            float[] deadStateProbabilities,
            java.util.Map<String, Double> profileMilliseconds
    ) {}
    private record Head(float[] mean, float[] scale, float[] weights, float bias) {}
    private record ProbabilityConfig(
            double smoothingSeconds,
            double enterThreshold,
            double exitThreshold,
            double minLiveSeconds,
            double bridgeGapSeconds,
            double shortEventMinSeconds,
            double shortEventThreshold
    ) {}
    private record ServeConfig(double threshold, double minSeparationSeconds, double timeOffsetSeconds) {}
    private record CompositionConfig(
            double associationSeconds,
            double fallbackSeconds,
            double maxRescueSeconds,
            ProbabilityConfig permissive
    ) {}
    private record DeadConfig(
            double deadThreshold,
            double liveResetThreshold,
            int minimumLiveSamples,
            int minimumDeadSamples,
            double minAfterServeSeconds,
            double maxAfterServeSeconds,
            double timeOffsetSeconds
    ) {}
    private record DeadRefinement(double endWindowSeconds) {}
    private record DecodeResult(List<AnalysisTypes.Interval> intervals, float[] smoothed) {}
    private record Bundle(
            float analysisFps,
            Head rally,
            ProbabilityConfig rallyDecoder,
            Head serve,
            ServeConfig serveDecoder,
            CompositionConfig composition,
            Head dead,
            DeadConfig deadDecoder,
            DeadRefinement refinement
    ) {}

    private final Bundle bundle;

    ModelRunner(Context context, String modelId) throws IOException, JSONException {
        try (InputStream input = context.getAssets().open(FeatureSchema.modelAsset(modelId))) {
            ByteArrayOutputStream output = new ByteArrayOutputStream();
            byte[] chunk = new byte[8192];
            int count;
            while ((count = input.read(chunk)) >= 0) output.write(chunk, 0, count);
            JSONObject json = new JSONObject(new String(output.toByteArray(), StandardCharsets.UTF_8));
            this.bundle = parseBundle(json);
        }
    }

    ModelRunner(JSONObject json) throws JSONException {
        this.bundle = parseBundle(json);
    }

    List<AnalysisTypes.Interval> run(double[] times, float[] contextual, double duration) {
        return runProfiled(times, contextual, duration).intervals();
    }

    RunResult runProfiled(double[] times, float[] contextual, double duration) {
        NanoProfiler profiler = new NanoProfiler();
        long operationStarted = System.nanoTime();
        int rows = times.length;
        float[] rallyProbabilities = predict(bundle.rally, contextual, rows);
        profiler.add("rally_head", System.nanoTime() - operationStarted);
        operationStarted = System.nanoTime();
        float[] serveProbabilities = predict(bundle.serve, contextual, rows);
        profiler.add("serve_head", System.nanoTime() - operationStarted);
        operationStarted = System.nanoTime();
        float[] deadProbabilities = predict(bundle.dead, contextual, rows);
        profiler.add("dead_state_head", System.nanoTime() - operationStarted);
        double fps = effectiveFps(times, bundle.analysisFps);
        operationStarted = System.nanoTime();
        DecodeResult primary = decodeProbabilities(times, rallyProbabilities, duration, bundle.rallyDecoder, fps);
        profiler.add("primary_rally_decode", System.nanoTime() - operationStarted);
        operationStarted = System.nanoTime();
        DecodeResult permissive = decodeProbabilities(times, rallyProbabilities, duration, bundle.composition.permissive, fps);
        profiler.add("permissive_rally_decode", System.nanoTime() - operationStarted);
        operationStarted = System.nanoTime();
        List<AnalysisTypes.Serve> serves = decodeServes(times, serveProbabilities, bundle.serveDecoder, duration);
        profiler.add("serve_decode", System.nanoTime() - operationStarted);
        operationStarted = System.nanoTime();
        List<AnalysisTypes.Interval> composed = compose(
                primary.intervals, permissive.intervals, serves, duration, bundle.composition, 1 / fps
        );
        profiler.add("serve_composition", System.nanoTime() - operationStarted);
        operationStarted = System.nanoTime();
        List<AnalysisTypes.Interval> refined = refineEnds(
                times, deadProbabilities, composed, duration, 1 / fps,
                bundle.deadDecoder, bundle.refinement
        );
        profiler.add("dead_state_refinement", System.nanoTime() - operationStarted);
        return new RunResult(
                refined,
                rallyProbabilities,
                serveProbabilities,
                deadProbabilities,
                profiler.milliseconds()
        );
    }

    private static Bundle parseBundle(JSONObject json) throws JSONException {
        if (json.getInt("schemaVersion") != 1) throw new JSONException("Unsupported model schema");
        JSONArray featureNames = json.getJSONArray("featureNames");
        List<String> expected = FeatureSchema.contextualNames();
        if (featureNames.length() != expected.size()) throw new JSONException("Model feature count mismatch");
        for (int i = 0; i < expected.size(); i++) {
            if (!expected.get(i).equals(featureNames.getString(i))) {
                throw new JSONException("Model feature signature differs at column " + i);
            }
        }
        int dimensions = expected.size();
        JSONObject rally = json.getJSONObject("rally");
        JSONObject serve = json.getJSONObject("serve");
        JSONObject dead = json.getJSONObject("deadState");
        JSONObject serveDecoder = serve.getJSONObject("decoder");
        JSONObject composition = serve.getJSONObject("composition");
        JSONObject deadDecoder = dead.getJSONObject("decoder");
        JSONObject refinement = dead.getJSONObject("refinement");
        return new Bundle(
                (float) json.getDouble("analysisFps"),
                parseHead(rally, dimensions),
                parseProbability(rally.getJSONObject("decoder")),
                parseHead(serve, dimensions),
                new ServeConfig(
                        serveDecoder.getDouble("threshold"),
                        serveDecoder.getDouble("minSeparationSeconds"),
                        serveDecoder.optDouble("timeOffsetSeconds", 0)
                ),
                new CompositionConfig(
                        composition.getDouble("associationSeconds"),
                        composition.getDouble("fallbackSeconds"),
                        composition.getDouble("maxRescueSeconds"),
                        parseProbability(composition.getJSONObject("permissiveDecoder"))
                ),
                parseHead(dead, dimensions),
                new DeadConfig(
                        deadDecoder.getDouble("deadThreshold"),
                        deadDecoder.getDouble("liveResetThreshold"),
                        deadDecoder.optInt("minimumLiveSamples", 1),
                        deadDecoder.optInt("minimumDeadSamples", 2),
                        deadDecoder.optDouble("minAfterServeSeconds", 0.25),
                        deadDecoder.optDouble("maxAfterServeSeconds", 4),
                        deadDecoder.optDouble("timeOffsetSeconds", 0)
                ),
                new DeadRefinement(refinement.optDouble("endWindowSeconds", 1))
        );
    }

    private static Head parseHead(JSONObject json, int dimensions) throws JSONException {
        return new Head(
                floatArray(json.getJSONArray("mean"), dimensions),
                floatArray(json.getJSONArray("scale"), dimensions),
                floatArray(json.getJSONArray("weights"), dimensions),
                (float) json.getDouble("bias")
        );
    }

    private static ProbabilityConfig parseProbability(JSONObject json) throws JSONException {
        double minimum = json.getDouble("min_live_seconds");
        return new ProbabilityConfig(
                json.getDouble("smoothing_seconds"),
                json.getDouble("enter_threshold"),
                json.getDouble("exit_threshold"),
                minimum,
                json.getDouble("bridge_gap_seconds"),
                json.optDouble("short_event_min_seconds", minimum),
                json.optDouble("short_event_threshold", 1)
        );
    }

    private static float[] floatArray(JSONArray array, int expected) throws JSONException {
        if (array.length() != expected) throw new JSONException("Model vector length mismatch");
        float[] values = new float[expected];
        for (int i = 0; i < expected; i++) values[i] = (float) array.getDouble(i);
        return values;
    }

    private static float[] predict(Head head, float[] values, int rows) {
        int dimensions = head.weights.length;
        if (values.length != rows * dimensions) throw new IllegalArgumentException("Model matrix shape mismatch");
        float[] probabilities = new float[rows];
        for (int row = 0; row < rows; row++) {
            int offset = row * dimensions;
            float logit = head.bias;
            for (int column = 0; column < dimensions; column++) {
                float normalized = (values[offset + column] - head.mean[column]) / head.scale[column];
                logit += normalized * head.weights[column];
            }
            double clipped = Math.max(-30, Math.min(30, logit));
            probabilities[row] = (float) (1 / (1 + Math.exp(-clipped)));
        }
        return probabilities;
    }

    private static DecodeResult decodeProbabilities(
            double[] times,
            float[] probabilities,
            double duration,
            ProbabilityConfig config,
            double fps
    ) {
        int smoothingSamples = Math.max(1, pythonRound(config.smoothingSeconds * fps));
        float[] smoothed = smooth(probabilities, smoothingSamples);
        boolean[] mask = new boolean[smoothed.length];
        boolean live = false;
        for (int i = 0; i < smoothed.length; i++) {
            if (!live && smoothed[i] >= config.enterThreshold) live = true;
            else if (live && smoothed[i] < config.exitThreshold) live = false;
            mask[i] = live;
        }
        int bridgeSamples = Math.max(0, pythonRound(config.bridgeGapSeconds * fps));
        if (bridgeSamples > 0) {
            for (int[] run : runs(mask, false)) {
                if (run[0] > 0 && run[1] < mask.length && run[1] - run[0] <= bridgeSamples) {
                    for (int i = run[0]; i < run[1]; i++) mask[i] = true;
                }
            }
        }
        int minimumLive = Math.max(1, pythonRound(config.minLiveSeconds * fps));
        int shortMinimum = Math.max(1, pythonRound(config.shortEventMinSeconds * fps));
        if (minimumLive > 1) {
            for (int[] run : runs(mask, true)) {
                float peak = -Float.MAX_VALUE;
                for (int i = run[0]; i < run[1]; i++) peak = Math.max(peak, smoothed[i]);
                boolean keepShort = run[1] - run[0] >= shortMinimum && peak >= config.shortEventThreshold;
                if (run[1] - run[0] < minimumLive && !keepShort) {
                    for (int i = run[0]; i < run[1]; i++) mask[i] = false;
                }
            }
        }
        double sampleWidth = 1 / fps;
        ArrayList<AnalysisTypes.Interval> intervals = new ArrayList<>();
        for (int[] run : runs(mask, true)) {
            double start = Math.max(0, times[run[0]] - sampleWidth / 2);
            double end = Math.min(duration, times[run[1] - 1] + sampleWidth / 2);
            if (end > start) intervals.add(new AnalysisTypes.Interval(
                    start, end, meanFloat(smoothed, run[0], run[1])
            ));
        }
        return new DecodeResult(intervals, smoothed);
    }

    private static List<AnalysisTypes.Serve> decodeServes(
            double[] times,
            float[] probabilities,
            ServeConfig config,
            double duration
    ) {
        ArrayList<AnalysisTypes.Serve> candidates = new ArrayList<>();
        Integer activeStart = null;
        for (int i = 0; i < probabilities.length; i++) {
            boolean active = probabilities[i] >= config.threshold;
            if (active && activeStart == null) activeStart = i;
            boolean closes = activeStart != null && (!active || i == probabilities.length - 1);
            if (!closes) continue;
            int end = active ? i + 1 : i;
            int peak = activeStart;
            for (int candidate = activeStart + 1; candidate < end; candidate++) {
                if (probabilities[candidate] > probabilities[peak]) peak = candidate;
            }
            double time = Math.min(duration, Math.max(0, times[peak] + config.timeOffsetSeconds));
            candidates.add(new AnalysisTypes.Serve(time, probabilities[peak]));
            activeStart = null;
        }
        candidates.sort(Comparator.comparingDouble((AnalysisTypes.Serve value) -> -value.confidence())
                .thenComparingDouble(AnalysisTypes.Serve::time));
        ArrayList<AnalysisTypes.Serve> retained = new ArrayList<>();
        for (AnalysisTypes.Serve candidate : candidates) {
            boolean separated = true;
            for (AnalysisTypes.Serve previous : retained) {
                if (Math.abs(candidate.time() - previous.time()) < config.minSeparationSeconds) {
                    separated = false;
                    break;
                }
            }
            if (separated) retained.add(candidate);
        }
        retained.sort(Comparator.comparingDouble(AnalysisTypes.Serve::time));
        return retained;
    }

    private static List<AnalysisTypes.Interval> compose(
            List<AnalysisTypes.Interval> primary,
            List<AnalysisTypes.Interval> permissive,
            List<AnalysisTypes.Serve> serves,
            double duration,
            CompositionConfig config,
            double sampleSeconds
    ) {
        ArrayList<double[]> rows = new ArrayList<>();
        for (AnalysisTypes.Interval interval : primary) {
            rows.add(new double[]{interval.start(), interval.end(), interval.confidence()});
        }
        ArrayList<AnalysisTypes.Serve> unused = new ArrayList<>();
        for (AnalysisTypes.Serve serve : serves) {
            ArrayList<Integer> associated = new ArrayList<>();
            for (int i = 0; i < rows.size(); i++) {
                if (rows.get(i)[0] - config.associationSeconds <= serve.time() && serve.time() < rows.get(i)[1]) {
                    associated.add(i);
                }
            }
            if (associated.isEmpty()) {
                unused.add(serve);
                continue;
            }
            int selected = associated.get(0);
            for (int candidate : associated) {
                if (Math.abs(rows.get(candidate)[0] - serve.time()) < Math.abs(rows.get(selected)[0] - serve.time())) {
                    selected = candidate;
                }
            }
            if (serve.time() < rows.get(selected)[0]) {
                rows.get(selected)[0] = serve.time();
                rows.get(selected)[2] = Math.max(rows.get(selected)[2], serve.confidence());
            }
        }
        for (AnalysisTypes.Serve serve : unused) {
            AnalysisTypes.Interval selected = null;
            double selectedDistance = Double.POSITIVE_INFINITY;
            for (AnalysisTypes.Interval interval : permissive) {
                if (interval.end() >= serve.time() - config.associationSeconds
                        && interval.start() <= serve.time() + config.associationSeconds
                        && interval.end() - interval.start() <= config.maxRescueSeconds) {
                    double distance = Math.min(Math.abs(interval.start() - serve.time()), Math.abs(interval.end() - serve.time()));
                    if (distance < selectedDistance) {
                        selected = interval;
                        selectedDistance = distance;
                    }
                }
            }
            if (selected != null) {
                rows.add(new double[]{serve.time(), Math.max(serve.time() + sampleSeconds, selected.end()),
                        Math.max(serve.confidence(), selected.confidence())});
            } else if (config.fallbackSeconds > 0) {
                rows.add(new double[]{serve.time(), serve.time() + config.fallbackSeconds, serve.confidence()});
            }
        }
        rows.removeIf(row -> row[1] <= row[0] || row[0] >= duration || row[1] <= 0);
        for (double[] row : rows) {
            row[0] = Math.max(0, row[0]);
            row[1] = Math.min(duration, row[1]);
        }
        rows.sort(Comparator.<double[]>comparingDouble(row -> row[0])
                .thenComparingDouble(row -> row[1]).thenComparingDouble(row -> row[2]));
        ArrayList<double[]> merged = new ArrayList<>();
        for (double[] row : rows) {
            double[] previous = merged.isEmpty() ? null : merged.get(merged.size() - 1);
            if (previous != null && row[0] < previous[1]) {
                previous[1] = Math.max(previous[1], row[1]);
                previous[2] = Math.max(previous[2], row[2]);
            } else merged.add(row.clone());
        }
        ArrayList<AnalysisTypes.Interval> result = new ArrayList<>();
        for (double[] row : merged) result.add(new AnalysisTypes.Interval(row[0], row[1], (float) row[2]));
        return result;
    }

    private static List<AnalysisTypes.Interval> refineEnds(
            double[] times,
            float[] probabilities,
            List<AnalysisTypes.Interval> intervals,
            double duration,
            double sampleSeconds,
            DeadConfig decoder,
            DeadRefinement refinement
    ) {
        ArrayList<AnalysisTypes.Interval> result = new ArrayList<>();
        for (int i = 0; i < intervals.size(); i++) {
            AnalysisTypes.Interval interval = intervals.get(i);
            Double nextStart = i + 1 < intervals.size() ? intervals.get(i + 1).start() : null;
            double anchor = Math.max(0, interval.end() - refinement.endWindowSeconds);
            DeadConfig local = new DeadConfig(
                    decoder.deadThreshold, decoder.liveResetThreshold,
                    decoder.minimumLiveSamples, decoder.minimumDeadSamples,
                    0, interval.end() + refinement.endWindowSeconds - anchor,
                    decoder.timeOffsetSeconds
            );
            AnalysisTypes.Serve transition = decodeDeadAfterServe(
                    times, probabilities, anchor, local,
                    nextStart != null && nextStart > anchor ? nextStart : null,
                    duration
            );
            if (transition == null) {
                result.add(interval);
                continue;
            }
            double transitionTime = Math.min(duration,
                    Math.min(interval.end() + refinement.endWindowSeconds,
                            Math.max(interval.end() - refinement.endWindowSeconds, transition.time())));
            double refinedEnd = Math.min(duration, Math.min(nextStart == null ? duration : nextStart, transitionTime));
            if (refinedEnd < interval.start() + sampleSeconds - 1e-9) result.add(interval);
            else result.add(new AnalysisTypes.Interval(interval.start(), refinedEnd,
                    Math.max(interval.confidence(), transition.confidence())));
        }
        return result;
    }

    private static AnalysisTypes.Serve decodeDeadAfterServe(
            double[] times,
            float[] probabilities,
            double serveTime,
            DeadConfig config,
            Double nextServeTime,
            double duration
    ) {
        double lower = serveTime + config.minAfterServeSeconds;
        double upper = serveTime + config.maxAfterServeSeconds;
        int liveRun = 0;
        boolean armed = false;
        Integer deadRunStart = null;
        int deadRunLength = 0;
        for (int i = 0; i < times.length; i++) {
            double time = times[i];
            float probability = probabilities[i];
            if (time < serveTime) continue;
            if (time > upper || (nextServeTime != null && time >= nextServeTime)) break;
            if (!armed) {
                if (probability <= config.liveResetThreshold) {
                    liveRun++;
                    if (liveRun >= config.minimumLiveSamples) armed = true;
                } else liveRun = 0;
            }
            if (!armed || time < lower) {
                deadRunStart = null;
                deadRunLength = 0;
                continue;
            }
            if (probability >= config.deadThreshold) {
                if (deadRunStart == null) deadRunStart = i;
                deadRunLength++;
            } else {
                deadRunStart = null;
                deadRunLength = 0;
            }
            if (deadRunStart != null && deadRunLength >= config.minimumDeadSamples) {
                float confidence = Float.POSITIVE_INFINITY;
                for (int run = deadRunStart; run <= i; run++) confidence = Math.min(confidence, probabilities[run]);
                double detection = Math.min(duration, Math.max(0, times[deadRunStart] + config.timeOffsetSeconds));
                return new AnalysisTypes.Serve(detection, confidence);
            }
        }
        return null;
    }

    private static float[] smooth(float[] probabilities, int requestedWindow) {
        if (probabilities.length == 0 || requestedWindow <= 1) return probabilities.clone();
        int window = Math.min(requestedWindow, probabilities.length);
        int leftPadding = window / 2;
        float scale = 1f / window;
        float[] result = new float[probabilities.length];
        for (int output = 0; output < probabilities.length; output++) {
            float sum = 0;
            for (int kernel = 0; kernel < window; kernel++) {
                int source = Math.max(0, Math.min(probabilities.length - 1, output + kernel - leftPadding));
                sum += probabilities[source] * scale;
            }
            result[output] = sum;
        }
        return result;
    }

    private static List<int[]> runs(boolean[] mask, boolean expected) {
        ArrayList<int[]> result = new ArrayList<>();
        Integer start = null;
        for (int i = 0; i < mask.length; i++) {
            if (mask[i] == expected && start == null) start = i;
            if (mask[i] != expected && start != null) {
                result.add(new int[]{start, i});
                start = null;
            }
        }
        if (start != null) result.add(new int[]{start, mask.length});
        return result;
    }

    private static float meanFloat(float[] values, int start, int end) {
        float sum = 0;
        for (int i = start; i < end; i++) sum += values[i];
        return sum / (end - start);
    }

    private static int pythonRound(double value) {
        int floor = (int) Math.floor(value);
        double fraction = value - floor;
        if (fraction < 0.5) return floor;
        if (fraction > 0.5) return floor + 1;
        return floor % 2 == 0 ? floor : floor + 1;
    }

    private static double effectiveFps(double[] times, double fallback) {
        if (times.length <= 1) return 1;
        double[] differences = new double[times.length - 1];
        for (int i = 1; i < times.length; i++) differences[i - 1] = times[i] - times[i - 1];
        java.util.Arrays.sort(differences);
        int middle = differences.length / 2;
        double median = differences.length % 2 == 1
                ? differences[middle]
                : (differences[middle - 1] + differences[middle]) / 2;
        return Double.isFinite(median) && median > 0 ? 1 / median : fallback;
    }
}
