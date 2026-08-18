package com.volleycut.nativeanalysis;

import android.content.Context;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/** Exact single-head runner for the frozen overlap-safe suppression specialist. */
final class SuppressionModelRunner {
    record Result(float[] probabilities, List<AnalysisTypes.Interval> decodedIntervals) {}
    private record Head(float[] mean, float[] scale, float[] weights, float bias) {}
    private record Decoder(
            double smoothingSeconds,
            double enterThreshold,
            double exitThreshold,
            double minLiveSeconds,
            double bridgeGapSeconds,
            double shortEventMinSeconds,
            double shortEventThreshold
    ) {}

    private final Head head;
    private final Decoder decoder;
    private final double analysisFps;

    SuppressionModelRunner(Context context) throws IOException, JSONException {
        this(readAsset(context));
    }

    SuppressionModelRunner(JSONObject json) throws JSONException {
        validateIdentity(json);
        validateFeatures(json.getJSONArray("featureNames"));
        analysisFps = json.getDouble("analysisFps");
        JSONObject values = json.getJSONObject("head");
        int dimensions = FeatureSchema.contextualNames().size();
        head = new Head(
                floats(values.getJSONArray("mean"), dimensions),
                floats(values.getJSONArray("scale"), dimensions),
                floats(values.getJSONArray("weights"), dimensions),
                (float) values.getDouble("bias")
        );
        JSONObject held = json.getJSONObject("decoder");
        decoder = new Decoder(
                held.getDouble("smoothing_seconds"),
                held.getDouble("enter_threshold"),
                held.getDouble("exit_threshold"),
                held.getDouble("min_live_seconds"),
                held.getDouble("bridge_gap_seconds"),
                held.getDouble("short_event_min_seconds"),
                held.getDouble("short_event_threshold")
        );
        validateHeldDecoder(decoder);
    }

    private static JSONObject readAsset(Context context) throws IOException, JSONException {
        try (InputStream input = context.getAssets().open(
                FeatureSchema.SUPPRESSION_MODEL_ID + ".json")) {
            ByteArrayOutputStream output = new ByteArrayOutputStream();
            byte[] buffer = new byte[8192];
            int count;
            while ((count = input.read(buffer)) >= 0) output.write(buffer, 0, count);
            return new JSONObject(output.toString(StandardCharsets.UTF_8));
        }
    }

    Result run(double[] times, float[] contextual, double durationSeconds) {
        int rows = times.length;
        int dimensions = head.weights.length;
        if (contextual.length != rows * dimensions) {
            throw new IllegalArgumentException("Suppression matrix shape mismatch");
        }
        float[] probabilities = new float[rows];
        for (int row = 0; row < rows; row++) {
            int offset = row * dimensions;
            float logit = head.bias;
            for (int column = 0; column < dimensions; column++) {
                logit += ((contextual[offset + column] - head.mean[column]) / head.scale[column])
                        * head.weights[column];
            }
            probabilities[row] = (float) (1 / (1 + Math.exp(-Math.max(-30, Math.min(30, logit)))));
        }
        return new Result(probabilities, decode(times, probabilities, durationSeconds));
    }

    private List<AnalysisTypes.Interval> decode(
            double[] times, float[] probabilities, double durationSeconds
    ) {
        double fps = effectiveFps(times, analysisFps);
        int smoothingSamples = Math.max(1, pythonRound(decoder.smoothingSeconds * fps));
        float[] smoothed = smooth(probabilities, smoothingSamples);
        boolean[] live = new boolean[smoothed.length];
        boolean active = false;
        for (int index = 0; index < smoothed.length; index++) {
            if (!active && smoothed[index] >= decoder.enterThreshold) active = true;
            else if (active && smoothed[index] < decoder.exitThreshold) active = false;
            live[index] = active;
        }
        int bridgeSamples = Math.max(0, pythonRound(decoder.bridgeGapSeconds * fps));
        for (int[] run : runs(live, false)) {
            if (run[0] > 0 && run[1] < live.length && run[1] - run[0] <= bridgeSamples) {
                for (int index = run[0]; index < run[1]; index++) live[index] = true;
            }
        }
        int minimumLive = Math.max(1, pythonRound(decoder.minLiveSeconds * fps));
        int shortMinimum = Math.max(1, pythonRound(decoder.shortEventMinSeconds * fps));
        for (int[] run : runs(live, true)) {
            float peak = -Float.MAX_VALUE;
            for (int index = run[0]; index < run[1]; index++) peak = Math.max(peak, smoothed[index]);
            if (run[1] - run[0] < minimumLive
                    && !(run[1] - run[0] >= shortMinimum && peak >= decoder.shortEventThreshold)) {
                for (int index = run[0]; index < run[1]; index++) live[index] = false;
            }
        }
        double sampleWidth = 1 / fps;
        ArrayList<AnalysisTypes.Interval> result = new ArrayList<>();
        for (int[] run : runs(live, true)) {
            double start = Math.max(0, times[run[0]] - sampleWidth / 2);
            double end = Math.min(durationSeconds, times[run[1] - 1] + sampleWidth / 2);
            if (end <= start) continue;
            float score = 0;
            for (int index = run[0]; index < run[1]; index++) score += smoothed[index];
            result.add(new AnalysisTypes.Interval(start, end, score / (run[1] - run[0])));
        }
        return List.copyOf(result);
    }

    private static void validateIdentity(JSONObject json) throws JSONException {
        if (json.getInt("schemaVersion") != 1
                || !FeatureSchema.SUPPRESSION_MODEL_ID.equals(json.getString("modelId"))
                || !FeatureSchema.SUPPRESSION_ARTIFACT_SHA256.equals(json.getString("artifactSha256"))
                || !FeatureSchema.SUPPRESSION_WEIGHTS_SHA256.equals(json.getString("weightsSha256"))
                || !FeatureSchema.SUPPRESSION_DECODER_VERSION.equals(json.getString("decoderVersion"))) {
            throw new JSONException("Suppression model identity does not match the product contract");
        }
    }

    private static void validateFeatures(JSONArray names) throws JSONException {
        List<String> expected = FeatureSchema.contextualNames();
        if (names.length() != expected.size()) throw new JSONException("Suppression feature count mismatch");
        for (int index = 0; index < names.length(); index++) {
            if (!expected.get(index).equals(names.getString(index))) {
                throw new JSONException("Suppression feature signature differs at column " + index);
            }
        }
    }

    private static void validateHeldDecoder(Decoder value) throws JSONException {
        if (value.smoothingSeconds != 1.0 || value.enterThreshold != .75
                || value.exitThreshold != .65 || value.minLiveSeconds != .5
                || value.bridgeGapSeconds != .5 || value.shortEventMinSeconds != .25
                || value.shortEventThreshold != .9) {
            throw new JSONException("Suppression decoder is not the held production decoder");
        }
    }

    private static float[] floats(JSONArray array, int expected) throws JSONException {
        if (array.length() != expected) throw new JSONException("Suppression vector length mismatch");
        float[] result = new float[expected];
        for (int index = 0; index < expected; index++) result[index] = (float) array.getDouble(index);
        return result;
    }

    private static float[] smooth(float[] values, int requestedWindow) {
        if (values.length == 0 || requestedWindow <= 1) return values.clone();
        int window = Math.min(requestedWindow, values.length);
        int leftPadding = window / 2;
        float[] result = new float[values.length];
        for (int output = 0; output < values.length; output++) {
            float sum = 0;
            for (int kernel = 0; kernel < window; kernel++) {
                int source = Math.max(0, Math.min(values.length - 1, output + kernel - leftPadding));
                sum += values[source] / window;
            }
            result[output] = sum;
        }
        return result;
    }

    private static List<int[]> runs(boolean[] values, boolean expected) {
        ArrayList<int[]> result = new ArrayList<>();
        Integer start = null;
        for (int index = 0; index < values.length; index++) {
            if (values[index] == expected && start == null) start = index;
            if (values[index] != expected && start != null) {
                result.add(new int[]{start, index});
                start = null;
            }
        }
        if (start != null) result.add(new int[]{start, values.length});
        return result;
    }

    private static int pythonRound(double value) {
        int floor = (int) Math.floor(value);
        double fraction = value - floor;
        if (fraction < .5) return floor;
        if (fraction > .5) return floor + 1;
        return floor % 2 == 0 ? floor : floor + 1;
    }

    private static double effectiveFps(double[] times, double fallback) {
        if (times.length <= 1) return fallback;
        double[] differences = new double[times.length - 1];
        for (int index = 1; index < times.length; index++) {
            differences[index - 1] = times[index] - times[index - 1];
        }
        java.util.Arrays.sort(differences);
        double median = differences[differences.length / 2];
        return Double.isFinite(median) && median > 0 ? 1 / median : fallback;
    }
}
