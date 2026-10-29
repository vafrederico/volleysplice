package com.volleycut.nativeanalysis;

import android.content.Context;
import android.net.Uri;

import java.io.IOException;
import java.util.Map;
import java.util.function.BooleanSupplier;

/** Backward-compatible serving-side view over the shared gap-aware specialist decoder. */
final class ServingSideFrameDecoder {
    private final SpecialistFrameDecoder decoder;

    ServingSideFrameDecoder(Context context) {
        decoder = new SpecialistFrameDecoder(context);
    }

    ServingSideFrameDecoder(Context context, double maximumSequentialGapSeconds) {
        decoder = new SpecialistFrameDecoder(context, maximumSequentialGapSeconds);
    }

    Map<Double, byte[]> decode(
            Uri uri,
            AnalysisTypes.MediaInfo media,
            AnalysisTypes.Roi roi,
            double[] requestedTimes,
            AnalysisTypes.ProgressListener progress,
            BooleanSupplier cancelled
    ) throws IOException {
        return decoder.decode(
                uri, media, roi, requestedTimes, new double[0], progress, cancelled
        ).servingGray();
    }
}
