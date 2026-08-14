package com.volleycut.nativeanalysis;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

final class FeatureMath {
    record ContextResult(
            float[] values,
            double percentileRankMilliseconds,
            double absoluteRestoreMilliseconds,
            double contextGatherMilliseconds
    ) {}

    private FeatureMath() {}

    static double mean(float[] values) {
        if (values.length == 0) return 0;
        double total = 0;
        for (float value : values) total += value;
        return total / values.length;
    }

    static double mean(float[] values, int start, int end) {
        if (end <= start) return 0;
        double total = 0;
        for (int index = start; index < end; index++) total += values[index];
        return total / (end - start);
    }

    static double standardDeviation(float[] values) {
        return standardDeviation(values, mean(values));
    }

    static double standardDeviation(float[] values, double average) {
        if (values.length == 0) return 0;
        double sumSquares = 0;
        for (float value : values) {
            double delta = value - average;
            sumSquares += delta * delta;
        }
        return Math.sqrt(sumSquares / values.length);
    }

    static float quantile(float[] values, double percentile) {
        if (values.length == 0) return 0;
        float[] sorted = values.clone();
        Arrays.sort(sorted);
        double position = (sorted.length - 1) * Math.max(0, Math.min(1, percentile));
        int lower = (int) Math.floor(position);
        int upper = (int) Math.ceil(position);
        double weight = position - lower;
        return (float) (sorted[lower] * (1 - weight) + sorted[upper] * weight);
    }

    static float[] rollingMean(float[] values, int windowSamples, boolean future) {
        float[] result = new float[values.length];
        if (values.length == 0) return result;
        int window = Math.max(1, windowSamples);
        double[] cumulative = new double[values.length + 1];
        for (int i = 0; i < values.length; i++) cumulative[i + 1] = cumulative[i] + values[i];
        for (int i = 0; i < values.length; i++) {
            int start = future ? i : Math.max(0, i - window + 1);
            int end = future ? Math.min(values.length, i + window) : i + 1;
            result[i] = (float) ((cumulative[end] - cumulative[start]) / (end - start));
        }
        return result;
    }

    static float[] percentileRanks(float[] values, int rows, int columns) {
        if (values.length != rows * columns) throw new IllegalArgumentException("Feature dimensions do not align");
        if (rows == 0) return values.clone();
        if (rows == 1) {
            float[] single = new float[columns];
            Arrays.fill(single, 0.5f);
            return single;
        }
        float[] result = new float[values.length];
        Integer[] order = new Integer[rows];
        for (int column = 0; column < columns; column++) {
            for (int row = 0; row < rows; row++) order[row] = row;
            int selectedColumn = column;
            Arrays.sort(order, Comparator.comparingDouble(row -> values[row * columns + selectedColumn]));
            int start = 0;
            while (start < rows) {
                float value = values[order[start] * columns + column];
                int end = start + 1;
                while (end < rows && values[order[end] * columns + column] == value) end++;
                float midrank = (float) ((start + end - 1) / 2.0 / (rows - 1));
                for (int i = start; i < end; i++) result[order[i] * columns + column] = midrank;
                start = end;
            }
        }
        return result;
    }

    static float[] contextualize(double[] times, float[] base, List<String> featureNames) {
        return contextualizeProfiled(times, base, featureNames).values();
    }

    static ContextResult contextualizeProfiled(
            double[] times,
            float[] base,
            List<String> featureNames
    ) {
        int rows = times.length;
        int columns = featureNames.size();
        if (rows == 0 || base.length != rows * columns) {
            throw new IllegalArgumentException("Context input must be a non-empty aligned matrix");
        }
        long operationStarted = System.nanoTime();
        float[] ranked = percentileRanks(base, rows, columns);
        double percentileRankMilliseconds = (System.nanoTime() - operationStarted) / 1_000_000.0;
        operationStarted = System.nanoTime();
        for (int column = 0; column < columns; column++) {
            if (!FeatureSchema.ABSOLUTE.contains(featureNames.get(column))) continue;
            for (int row = 0; row < rows; row++) ranked[row * columns + column] = base[row * columns + column];
        }
        double absoluteRestoreMilliseconds = (System.nanoTime() - operationStarted) / 1_000_000.0;
        int blockColumns = columns * FeatureSchema.CONTEXT_OFFSETS_SECONDS.length;
        float[] output = new float[rows * blockColumns];
        operationStarted = System.nanoTime();
        for (int block = 0; block < FeatureSchema.CONTEXT_OFFSETS_SECONDS.length; block++) {
            int offset = FeatureSchema.CONTEXT_OFFSETS_SECONDS[block];
            for (int row = 0; row < rows; row++) {
                int source = nearestIndex(times, times[row] + offset);
                System.arraycopy(ranked, source * columns, output, row * blockColumns + block * columns, columns);
            }
        }
        double contextGatherMilliseconds = (System.nanoTime() - operationStarted) / 1_000_000.0;
        return new ContextResult(
                output,
                percentileRankMilliseconds,
                absoluteRestoreMilliseconds,
                contextGatherMilliseconds
        );
    }

    static int lowerBound(double[] values, double target) {
        int lower = 0;
        int upper = values.length;
        while (lower < upper) {
            int middle = (lower + upper) >>> 1;
            if (values[middle] < target) lower = middle + 1;
            else upper = middle;
        }
        return lower;
    }

    static int upperBound(double[] values, double target) {
        int lower = 0;
        int upper = values.length;
        while (lower < upper) {
            int middle = (lower + upper) >>> 1;
            if (values[middle] <= target) lower = middle + 1;
            else upper = middle;
        }
        return lower;
    }

    private static int nearestIndex(double[] times, double target) {
        if (target <= times[0]) return 0;
        if (target >= times[times.length - 1]) return times.length - 1;
        int right = Math.min(times.length - 1, lowerBound(times, target));
        int left = Math.max(0, right - 1);
        return Math.abs(times[left] - target) <= Math.abs(times[right] - target) ? left : right;
    }

    static float[] column(float[] matrix, int rows, int columns, int selected) {
        float[] output = new float[rows];
        for (int row = 0; row < rows; row++) output[row] = matrix[row * columns + selected];
        return output;
    }

    static float[] temporalVisualFeatures(float[] visual, int rows) {
        int columns = FeatureSchema.FRAME.size();
        int motionIndex = FeatureSchema.FRAME.indexOf("player_motion_mean");
        int zonesIndex = FeatureSchema.FRAME.indexOf("player_motion_active_zone_fraction");
        float[] motion = column(visual, rows, columns, motionIndex);
        float[] zones = column(visual, rows, columns, zonesIndex);
        int window = FeatureSchema.ANALYSIS_FPS;
        int shortWindow = Math.max(1, Math.round(0.5f * FeatureSchema.ANALYSIS_FPS));
        float[] pastMotion = rollingMean(motion, window, false);
        float[] futureMotion = rollingMean(motion, shortWindow, true);
        float[] pastZones = rollingMean(zones, window, false);
        float[] futureZones = rollingMean(zones, shortWindow, true);
        String[] geometryNames = {
                "player_motion_centroid_x", "player_motion_centroid_y",
                "player_motion_spread_x", "player_motion_spread_y"
        };
        List<float[]> geometryChanges = new ArrayList<>();
        for (String name : geometryNames) {
            float[] values = column(visual, rows, columns, FeatureSchema.FRAME.indexOf(name));
            float[] past = rollingMean(values, window, false);
            float[] future = rollingMean(values, window, true);
            float[] changes = new float[rows];
            for (int row = 0; row < rows; row++) changes[row] = future[row] - past[row];
            geometryChanges.add(changes);
        }
        float[] output = new float[rows * FeatureSchema.TEMPORAL.size()];
        for (int row = 0; row < rows; row++) {
            float onset = Math.max(futureMotion[row] - pastMotion[row], 0);
            float collapse = Math.max(pastMotion[row] - futureMotion[row], 0);
            float synchronizedStandDown = collapse * Math.max(pastZones[row] - futureZones[row], 0);
            double geometrySquares = 0;
            for (float[] changes : geometryChanges) geometrySquares += changes[row] * changes[row];
            float activityGate = Math.min(1, (pastMotion[row] + futureMotion[row]) / 0.015f);
            int offset = row * FeatureSchema.TEMPORAL.size();
            output[offset] = onset;
            output[offset + 1] = collapse;
            output[offset + 2] = synchronizedStandDown;
            output[offset + 3] = (float) Math.sqrt(geometrySquares) * activityGate;
        }
        return output;
    }
}
