package com.volleycut.nativeanalysis;

import java.io.IOException;
import java.util.Locale;
import java.util.function.BooleanSupplier;
import java.util.function.LongSupplier;

/** Reports timestamp inventory, not decoded frames or inference completion. */
final class VideoPreparationProgress {
    private final double durationSeconds;
    private final int scanLimit;
    private final AnalysisTypes.ProgressListener progress;
    private final BooleanSupplier cancelled;
    private final LongSupplier clock;
    private long lastEmission;
    private long furthestUs;
    private int samples;

    VideoPreparationProgress(double durationSeconds, int scanLimit,
            AnalysisTypes.ProgressListener progress, BooleanSupplier cancelled) {
        this(durationSeconds, scanLimit, progress, cancelled, System::nanoTime);
    }

    VideoPreparationProgress(double durationSeconds, int scanLimit,
            AnalysisTypes.ProgressListener progress, BooleanSupplier cancelled, LongSupplier clock) {
        this.durationSeconds = durationSeconds;
        this.scanLimit = scanLimit;
        this.progress = progress;
        this.cancelled = cancelled;
        this.clock = clock;
        lastEmission = clock.getAsLong();
        progress.onProgress("video-indexing", 0, "Reading video timestamps before scanning frames");
    }

    void sample(long presentationUs) throws IOException {
        if (cancelled.getAsBoolean()) throw new IOException("Analysis cancelled");
        samples++;
        furthestUs = Math.max(furthestUs, presentationUs);
        long now = clock.getAsLong();
        if (now - lastEmission < 250_000_000L) return;
        lastEmission = now;
        double timeFraction = durationSeconds > 0 ? furthestUs / 1_000_000.0 / durationSeconds : 0;
        double countFraction = scanLimit < Integer.MAX_VALUE ? samples / (double) scanLimit : 0;
        progress.onProgress("video-indexing", Math.min(.99, Math.max(timeFraction, countFraction)),
                String.format(Locale.US, "Reading video timestamps: %.1f of %.1f seconds; %,d frame entries",
                        Math.min(durationSeconds, furthestUs / 1_000_000.0), durationSeconds, samples));
    }

    void complete() throws IOException {
        if (cancelled.getAsBoolean()) throw new IOException("Analysis cancelled");
        progress.onProgress("video-indexing", 1,
                String.format(Locale.US, "Video timestamps ready: %,d frame entries", samples));
    }
}
