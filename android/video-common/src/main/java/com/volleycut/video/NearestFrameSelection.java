package com.volleycut.video;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/** Select actual display timestamps, never inferred frame-rate timestamps. */
public final class NearestFrameSelection {
    private NearestFrameSelection() {}

    /**
     * One source PTS per requested grid row before the exclusive analysis end.
     * Equidistant frames choose the earlier PTS. A source frame may serve several
     * grid rows (low frame rates, duplicate requests, or the final partial tick).
     * Callers must constrain source PTS to the permitted source-frame/window bound.
     */
    public static long[] select(long[] sortedPresentationUs, double[] targetSeconds,
            double exclusiveEndSeconds) {
        if (!Double.isFinite(exclusiveEndSeconds)) {
            throw new IllegalArgumentException("Analysis end must be finite");
        }
        for (int i = 1; i < sortedPresentationUs.length; i++) {
            if (sortedPresentationUs[i] < sortedPresentationUs[i - 1]) {
                throw new IllegalArgumentException("Source timestamps must be sorted");
            }
        }
        for (int i = 0; i < targetSeconds.length; i++) {
            if (!Double.isFinite(targetSeconds[i])
                    || (i > 0 && targetSeconds[i] < targetSeconds[i - 1])) {
                throw new IllegalArgumentException("Target timestamps must be finite and sorted");
            }
        }
        if (sortedPresentationUs.length == 0) return new long[0];
        long[] selected = new long[targetSeconds.length];
        int count = 0;
        int right = 0;
        for (double seconds : targetSeconds) {
            if (seconds >= exclusiveEndSeconds) break;
            double targetUs = seconds * 1_000_000.0;
            while (right < sortedPresentationUs.length
                    && sortedPresentationUs[right] < targetUs) right++;
            int after = Math.min(right, sortedPresentationUs.length - 1);
            int before = Math.max(0, right - 1);
            selected[count++] = Math.abs(targetUs - sortedPresentationUs[before])
                    <= Math.abs(sortedPresentationUs[after] - targetUs)
                    ? sortedPresentationUs[before] : sortedPresentationUs[after];
        }
        return Arrays.copyOf(selected, count);
    }

    /** Materialize the previous row too when resuming a temporal feature stream. */
    public static Set<Long> materializedTimestamps(long[] selectedPresentationUs, int firstRow) {
        if (firstRow < 0 || firstRow > selectedPresentationUs.length) {
            throw new IllegalArgumentException("Invalid first requested row");
        }
        HashSet<Long> result = new HashSet<>();
        for (int i = firstRow; i < selectedPresentationUs.length; i++) {
            result.add(selectedPresentationUs[i]);
        }
        return Set.copyOf(result);
    }
}
