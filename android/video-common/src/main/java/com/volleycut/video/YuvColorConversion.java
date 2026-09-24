package com.volleycut.video;

/** Converts eight-bit nonlinear Y'CbCr to packed nonlinear RGB (0xRRGGBB).
 * This is a matrix/range conversion, not HDR tone mapping or gamut conversion. */
public final class YuvColorConversion {
    // Values are the public MediaFormat COLOR_STANDARD_* / COLOR_RANGE_* constants.
    public static final int BT709 = 1, BT601_PAL = 2, BT601_NTSC = 4, BT2020 = 6;
    public static final int FULL = 1, LIMITED = 2;
    private static final int SHIFT = 16, HALF = 1 << (SHIFT - 1);
    private static final YuvColorConversion[][] CONVERSIONS = {
            {new YuvColorConversion(.299, .114, true), new YuvColorConversion(.299, .114, false)},
            {new YuvColorConversion(.2126, .0722, true), new YuvColorConversion(.2126, .0722, false)},
            {new YuvColorConversion(.2627, .0593, true), new YuvColorConversion(.2627, .0593, false)}
    };
    private final int[] luma = new int[256], redV = new int[256], greenU = new int[256],
            greenV = new int[256], blueU = new int[256];

    private YuvColorConversion(double kr, double kb, boolean full) {
        double kg = 1 - kr - kb;
        double yScale = full ? 1 : 255.0 / 219;
        double cScale = full ? 1 : 255.0 / 224;
        for (int i = 0; i < 256; i++) {
            // Preserve foot/headroom until final RGB clipping, including chromatic pixels.
            luma[i] = fixed((i - (full ? 0 : 16)) * yScale);
            double chroma = (i - 128) * cScale;
            redV[i] = fixed(2 * (1 - kr) * chroma);
            blueU[i] = fixed(2 * (1 - kb) * chroma);
            greenU[i] = fixed(-2 * kb * (1 - kb) / kg * chroma);
            greenV[i] = fixed(-2 * kr * (1 - kr) / kg * chroma);
        }
    }

    public static YuvColorConversion of(int standard, int range) {
        int matrix = switch (standard) {
            case BT601_PAL, BT601_NTSC -> 0;
            case BT709 -> 1;
            case BT2020 -> 2; // Non-constant-luminance matrix; no HDR tone mapping.
            default -> throw new IllegalArgumentException("Unsupported YUV color standard: " + standard);
        };
        if (range != FULL && range != LIMITED) {
            throw new IllegalArgumentException("Unsupported YUV color range: " + range);
        }
        return CONVERSIONS[matrix][range == FULL ? 0 : 1];
    }

    public int rgb(int y, int u, int v) {
        int l = luma[y];
        int r = channel(l + redV[v]);
        int g = channel(l + greenU[u] + greenV[v]);
        int b = channel(l + blueU[u]);
        return (r << 16) | (g << 8) | b;
    }

    private static int fixed(double value) { return (int) Math.round(value * (1 << SHIFT)); }
    private static int channel(int value) {
        return Math.max(0, Math.min(255, (value + HALF) >> SHIFT));
    }

    /** Metadata precedence is per field: decoded buffer, track, then explicit legacy fallback.
     * A decoder can convert range/matrix, so track metadata never overrides its output. */
    public record Profile(int standard, int range, String standardSource, String rangeSource) {
        public YuvColorConversion conversion() { return of(standard, range); }
    }

    public static Profile resolve(int outputStandard, int outputRange, int trackStandard, int trackRange) {
        int standard = firstSpecified(outputStandard, trackStandard, BT601_NTSC);
        int range = firstSpecified(outputRange, trackRange, LIMITED);
        of(standard, range); // Reject explicitly unsupported metadata rather than silently reinterpreting it.
        return new Profile(standard, range, source(outputStandard, trackStandard, "assumed-bt601"),
                source(outputRange, trackRange, "assumed-limited"));
    }

    private static int firstSpecified(int output, int track, int fallback) {
        return output > 0 ? output : track > 0 ? track : fallback;
    }
    private static String source(int output, int track, String fallback) {
        return output > 0 ? "decoder-output" : track > 0 ? "extractor-track" : fallback;
    }
}
