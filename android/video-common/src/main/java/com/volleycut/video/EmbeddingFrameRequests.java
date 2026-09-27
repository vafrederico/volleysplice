package com.volleycut.video;

import java.util.Arrays;

/** Neural sampling from a shared real-PTS inventory, with its own exclusive end. */
public final class EmbeddingFrameRequests {
    private EmbeddingFrameRequests() {}
    public record Plan(double[] targetSeconds, long[] selectedPresentationUs) {}
    public static Plan create(long[] sortedPresentationUs, double start, double end, double fps) {
        if (!Double.isFinite(start) || start < 0 || !Double.isFinite(end) || end <= start
                || !Double.isFinite(fps) || fps <= 0) throw new IllegalArgumentException("Invalid sampling window");
        int permitted = 0;
        for (int i=0; i<sortedPresentationUs.length; i++) {
            if (i>0 && sortedPresentationUs[i]<sortedPresentationUs[i-1]) throw new IllegalArgumentException("Unsorted PTS");
            if (sortedPresentationUs[i]<end*1e6) permitted++;
        }
        double count = Math.ceil((end-start)*fps-1e-9);
        if (count > Integer.MAX_VALUE) throw new IllegalArgumentException("Sampling window too large");
        double[] targets = new double[(int)count];
        for(int i=0;i<targets.length;i++) targets[i]=start+i/fps;
        long[] selected = NearestFrameSelection.select(Arrays.copyOf(sortedPresentationUs, permitted), targets, end);
        return new Plan(Arrays.copyOf(targets, selected.length), selected);
    }
}
