package com.volleycut.video;

import android.media.MediaFormat;

/** Reads the format of the actual decoded output buffer, not the compressed input alone. */
public final class DecodedVideoColor {
    private DecodedVideoColor() {}

    public static YuvColorConversion.Profile resolve(MediaFormat output, MediaFormat track) {
        return YuvColorConversion.resolve(
                value(output, MediaFormat.KEY_COLOR_STANDARD), value(output, MediaFormat.KEY_COLOR_RANGE),
                value(track, MediaFormat.KEY_COLOR_STANDARD), value(track, MediaFormat.KEY_COLOR_RANGE));
    }

    private static int value(MediaFormat format, String key) {
        return format == null || !format.containsKey(key) ? 0 : format.getInteger(key, 0);
    }
}
