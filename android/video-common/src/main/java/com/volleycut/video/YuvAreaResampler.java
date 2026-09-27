package com.volleycut.video;

import java.nio.ByteBuffer;

/**
 * Area-downsamples a displayed YUV420 crop after converting each source pixel to
 * clipped RGB. AV training resizes decoded RGB with INTER_AREA; averaging YUV
 * first, bilinear interpolation, or sampling one source pixel changes that input
 * contract, especially the Laplacian focus/blur features.
 *
 * <p>The plan and output are reusable. No full-resolution RGB image is allocated.
 * A caller must consume/copy the returned RGBA bytes before the next conversion.
 */
public final class YuvAreaResampler {
    private final int outputWidth;
    private final int outputHeight;
    private final int rotation;
    private final int sourceLeft;
    private final int sourceTop;
    private final int sourceWidth;
    private final int sourceHeight;
    private final int reducedWidth;
    private final int reducedHeight;
    private final int integerScaleX;
    private final int integerScaleY;
    private final Span[] columns;
    private final Span[] rows;
    private final byte[] rgba;
    private final int[] nativeGeometry;

    public YuvAreaResampler(int cropLeft, int cropTop, int cropWidth, int cropHeight,
            int rotation, double roiX, double roiY, double roiWidth, double roiHeight,
            int outputWidth, int outputHeight) {
        if (cropLeft < 0 || cropTop < 0 || cropWidth <= 0 || cropHeight <= 0
                || outputWidth <= 0 || outputHeight <= 0) {
            throw new IllegalArgumentException("Image dimensions must be positive");
        }
        if (!Double.isFinite(roiX) || !Double.isFinite(roiY)
                || !Double.isFinite(roiWidth) || !Double.isFinite(roiHeight)
                || roiWidth <= 0 || roiHeight <= 0) {
            throw new IllegalArgumentException("ROI must have finite positive dimensions");
        }
        this.rotation = Math.floorMod(rotation, 360);
        if (this.rotation % 90 != 0) {
            throw new IllegalArgumentException("Rotation must be a multiple of 90 degrees");
        }
        this.outputWidth = outputWidth;
        this.outputHeight = outputHeight;
        boolean transpose = this.rotation == 90 || this.rotation == 270;
        int displayWidth = transpose ? cropHeight : cropWidth;
        int displayHeight = transpose ? cropWidth : cropHeight;
        // Match Python round() in analysis.features._crop_roi, including ties.
        int left = clamp((int) Math.rint(roiX * displayWidth), 0, displayWidth - 1);
        int top = clamp((int) Math.rint(roiY * displayHeight), 0, displayHeight - 1);
        int right = clamp((int) Math.rint((roiX + roiWidth) * displayWidth), left + 1, displayWidth);
        int bottom = clamp((int) Math.rint((roiY + roiHeight) * displayHeight), top + 1, displayHeight);
        int displayedWidth = right - left;
        int displayedHeight = bottom - top;
        sourceWidth = transpose ? displayedHeight : displayedWidth;
        sourceHeight = transpose ? displayedWidth : displayedHeight;
        sourceLeft = cropLeft + switch (this.rotation) {
            case 90 -> top;
            case 180 -> cropWidth - right;
            case 270 -> cropWidth - bottom;
            default -> left;
        };
        sourceTop = cropTop + switch (this.rotation) {
            case 90 -> cropHeight - right;
            case 180 -> cropHeight - bottom;
            case 270 -> left;
            default -> top;
        };
        reducedWidth = transpose ? outputHeight : outputWidth;
        reducedHeight = transpose ? outputWidth : outputHeight;
        boolean integer = sourceWidth % reducedWidth == 0 && sourceHeight % reducedHeight == 0;
        integerScaleX = integer ? sourceWidth / reducedWidth : 0;
        integerScaleY = integer ? sourceHeight / reducedHeight : 0;
        columns = integer ? null : spans(sourceWidth, reducedWidth);
        rows = integer ? null : spans(sourceHeight, reducedHeight);
        rgba = new byte[Math.multiplyExact(Math.multiplyExact(outputWidth, outputHeight), 4)];
        nativeGeometry = new int[]{sourceLeft, sourceTop, sourceWidth, sourceHeight,
                reducedWidth, reducedHeight, this.rotation, outputWidth, outputHeight};
    }

    public byte[] convert(ByteBuffer y, int yRowStride, int yPixelStride,
            ByteBuffer u, int uRowStride, int uPixelStride,
            ByteBuffer v, int vRowStride, int vPixelStride, YuvColorConversion color) {
        if (NativeYuvArea.AVAILABLE && y.isDirect() && u.isDirect() && v.isDirect()) {
            NativeYuvArea.resample(y, yRowStride, yPixelStride, y.limit(),
                    u, uRowStride, uPixelStride, u.limit(),
                    v, vRowStride, vPixelStride, v.limit(),
                    nativeGeometry, color.nativeTables(), rgba);
            return rgba;
        }
        return convertJava(y, yRowStride, yPixelStride, u, uRowStride, uPixelStride,
                v, vRowStride, vPixelStride, color);
    }

    /** Exposes backend availability for instrumentation and benchmark receipts. */
    public static boolean nativeAvailable() { return NativeYuvArea.AVAILABLE; }

    // Retained as the portable fallback and exact instrumentation reference.
    byte[] convertJava(ByteBuffer y, int yRowStride, int yPixelStride,
            ByteBuffer u, int uRowStride, int uPixelStride,
            ByteBuffer v, int vRowStride, int vPixelStride, YuvColorConversion color) {
        if (integerScaleX > 0 && integerScaleY > 0) {
            convertInteger(y, yRowStride, yPixelStride, u, uRowStride, uPixelStride,
                    v, vRowStride, vPixelStride, color);
        } else {
            convertWeighted(y, yRowStride, yPixelStride, u, uRowStride, uPixelStride,
                    v, vRowStride, vPixelStride, color);
        }
        return rgba;
    }

    private void convertInteger(ByteBuffer y, int yRowStride, int yPixelStride,
            ByteBuffer u, int uRowStride, int uPixelStride,
            ByteBuffer v, int vRowStride, int vPixelStride, YuvColorConversion color) {
        double denominator = (double) integerScaleX * integerScaleY;
        for (int dy = 0; dy < reducedHeight; dy++) {
            int top = sourceTop + dy * integerScaleY;
            for (int dx = 0; dx < reducedWidth; dx++) {
                int left = sourceLeft + dx * integerScaleX;
                long red = 0, green = 0, blue = 0;
                for (int sy = top; sy < top + integerScaleY; sy++) {
                    int yOffset = sy * yRowStride + left * yPixelStride;
                    int uRow = (sy / 2) * uRowStride;
                    int vRow = (sy / 2) * vRowStride;
                    for (int sx = left; sx < left + integerScaleX; sx++) {
                        int rgb = color.rgb(y.get(yOffset) & 255,
                                u.get(uRow + (sx / 2) * uPixelStride) & 255,
                                v.get(vRow + (sx / 2) * vPixelStride) & 255);
                        red += (rgb >>> 16) & 255;
                        green += (rgb >>> 8) & 255;
                        blue += rgb & 255;
                        yOffset += yPixelStride;
                    }
                }
                put(dx, dy, red / denominator, green / denominator, blue / denominator);
            }
        }
    }

    private void convertWeighted(ByteBuffer y, int yRowStride, int yPixelStride,
            ByteBuffer u, int uRowStride, int uPixelStride,
            ByteBuffer v, int vRowStride, int vPixelStride, YuvColorConversion color) {
        double denominator = ((double) sourceWidth / reducedWidth) * ((double) sourceHeight / reducedHeight);
        for (int dy = 0; dy < reducedHeight; dy++) {
            Span row = rows[dy];
            for (int dx = 0; dx < reducedWidth; dx++) {
                Span column = columns[dx];
                double red = 0, green = 0, blue = 0;
                for (int iy = 0; iy < row.weights.length; iy++) {
                    int sy = sourceTop + row.first + iy;
                    int yOffset = sy * yRowStride + (sourceLeft + column.first) * yPixelStride;
                    int uRow = (sy / 2) * uRowStride;
                    int vRow = (sy / 2) * vRowStride;
                    for (int ix = 0; ix < column.weights.length; ix++) {
                        int sx = sourceLeft + column.first + ix;
                        int rgb = color.rgb(y.get(yOffset) & 255,
                                u.get(uRow + (sx / 2) * uPixelStride) & 255,
                                v.get(vRow + (sx / 2) * vPixelStride) & 255);
                        double weight = row.weights[iy] * column.weights[ix];
                        red += ((rgb >>> 16) & 255) * weight;
                        green += ((rgb >>> 8) & 255) * weight;
                        blue += (rgb & 255) * weight;
                        yOffset += yPixelStride;
                    }
                }
                put(dx, dy, red / denominator, green / denominator, blue / denominator);
            }
        }
    }

    private void put(int x, int y, double red, double green, double blue) {
        int displayX = switch (rotation) {
            case 90 -> outputWidth - 1 - y;
            case 180 -> outputWidth - 1 - x;
            case 270 -> y;
            default -> x;
        };
        int displayY = switch (rotation) {
            case 90 -> x;
            case 180 -> outputHeight - 1 - y;
            case 270 -> outputHeight - 1 - x;
            default -> y;
        };
        int index = (displayY * outputWidth + displayX) * 4;
        rgba[index] = (byte) Math.rint(red);
        rgba[index + 1] = (byte) Math.rint(green);
        rgba[index + 2] = (byte) Math.rint(blue);
        rgba[index + 3] = (byte) 255;
    }

    private static Span[] spans(int source, int target) {
        Span[] result = new Span[target];
        double scale = (double) source / target;
        for (int index = 0; index < target; index++) {
            double start = index * scale, end = Math.min(source, (index + 1) * scale);
            int first = (int) Math.floor(start), last = (int) Math.ceil(end);
            double[] weights = new double[last - first];
            for (int offset = 0; offset < weights.length; offset++) {
                int pixel = first + offset;
                weights[offset] = Math.max(0, Math.min(end, pixel + 1) - Math.max(start, pixel));
            }
            result[index] = new Span(first, weights);
        }
        return result;
    }

    private static int clamp(int value, int lower, int upper) {
        return Math.max(lower, Math.min(upper, value));
    }

    private record Span(int first, double[] weights) {}
}
