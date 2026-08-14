package com.volleycut.nativeanalysis;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;

import java.util.ArrayList;
import java.util.List;

final class VisualFeatureExtractor implements AutoCloseable {
    private final int width;
    private final int height;
    private final double diagonal;
    private final NanoProfiler profiler = new NanoProfiler();
    private Mat previousGray;

    VisualFeatureExtractor(int width, int height) {
        this.width = width;
        this.height = height;
        this.diagonal = Math.hypot(width, height);
    }

    float[] extract(Mat rgba) {
        long extractStarted = System.nanoTime();
        Mat gray = new Mat();
        Mat rgb = new Mat();
        Mat hsv = new Mat();
        Mat edges = new Mat();
        Mat laplacian = new Mat();
        Mat gradientX = new Mat();
        Mat gradientY = new Mat();
        Mat flow = new Mat();
        try {
            long segmentStarted = System.nanoTime();
            Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY);
            Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB);
            Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV);
            Imgproc.Canny(gray, edges, 60, 140);
            Imgproc.Laplacian(gray, laplacian, CvType.CV_32F);
            Imgproc.Sobel(gray, gradientX, CvType.CV_32F, 1, 0, 3);
            Imgproc.Sobel(gray, gradientY, CvType.CV_32F, 0, 1, 3);
            profiler.add("opencv_filters", System.nanoTime() - segmentStarted);

            segmentStarted = System.nanoTime();
            int pixelsCount = width * height;
            byte[] grayBytes = new byte[pixelsCount];
            byte[] hsvBytes = new byte[pixelsCount * 3];
            byte[] edgeBytes = new byte[pixelsCount];
            float[] laplacianValues = new float[pixelsCount];
            float[] gradientXValues = new float[pixelsCount];
            float[] gradientYValues = new float[pixelsCount];
            gray.get(0, 0, grayBytes);
            hsv.get(0, 0, hsvBytes);
            edges.get(0, 0, edgeBytes);
            laplacian.get(0, 0, laplacianValues);
            gradientX.get(0, 0, gradientXValues);
            gradientY.get(0, 0, gradientYValues);
            profiler.add("mat_readback", System.nanoTime() - segmentStarted);

            segmentStarted = System.nanoTime();
            float[] pixels = unsigned(grayBytes);
            float[] saturation = new float[pixelsCount];
            for (int i = 0; i < pixelsCount; i++) saturation[i] = hsvBytes[i * 3 + 1] & 0xff;
            double lumaMean = FeatureMath.mean(pixels);
            double lumaStd = FeatureMath.standardDeviation(pixels, lumaMean);
            double laplacianVariance = Math.pow(FeatureMath.standardDeviation(laplacianValues), 2);
            int edgeCount = 0;
            for (byte value : edgeBytes) if ((value & 0xff) > 0) edgeCount++;

            ArrayList<Float> values = new ArrayList<>(FeatureSchema.FRAME.size());
            add(values, lumaMean / 255, lumaStd / 255, FeatureMath.mean(saturation) / 255,
                    FeatureMath.standardDeviation(saturation) / 255,
                    (double) edgeCount / pixelsCount, Math.min(laplacianVariance / 2000, 5));
            addScaled(values, gridMeans(pixels), 1 / 255.0);

            float[] difference = new float[pixelsCount];
            if (previousGray != null) {
                byte[] previousBytes = new byte[pixelsCount];
                previousGray.get(0, 0, previousBytes);
                for (int i = 0; i < pixelsCount; i++) {
                    difference[i] = Math.abs((grayBytes[i] & 0xff) - (previousBytes[i] & 0xff));
                }
            }
            int activeDifference = 0;
            for (float value : difference) if (value >= 18) activeDifference++;
            double differenceMean = FeatureMath.mean(difference);
            add(values, differenceMean / 255,
                    FeatureMath.standardDeviation(difference, differenceMean) / 255,
                    FeatureMath.quantile(difference, 0.9) / 255,
                    (double) activeDifference / difference.length);
            addScaled(values, gridMeans(difference), 1 / 255.0);

            double focusQuality = laplacianVariance / (laplacianVariance + 100);
            double blurProbability = 1 - focusQuality;
            int dark = 0;
            int bright = 0;
            int lowTexture = 0;
            for (int i = 0; i < pixelsCount; i++) {
                if (pixels[i] <= 12) dark++;
                if (pixels[i] >= 243) bright++;
                if (Math.hypot(gradientXValues[i], gradientYValues[i]) < 8) lowTexture++;
            }
            double darkFraction = (double) dark / pixelsCount;
            double brightFraction = (double) bright / pixelsCount;
            int occludedCells = 0;
            for (int row = 0; row < 3; row++) {
                int top = Math.round(row * height / 3f);
                int bottom = Math.round((row + 1) * height / 3f);
                for (int column = 0; column < 3; column++) {
                    int left = Math.round(column * width / 3f);
                    int right = Math.round((column + 1) * width / 3f);
                    float[] cell = rectValues(pixels, left, top, right, bottom);
                    double cellMean = FeatureMath.mean(cell);
                    if ((cellMean <= 16 || cellMean >= 239)
                            && FeatureMath.standardDeviation(cell, cellMean) <= 8) occludedCells++;
                }
            }
            double occlusionFraction = occludedCells / 9.0;
            double exposureQuality = Math.max(0, 1 - Math.min(1, darkFraction + brightFraction));
            double contrastQuality = Math.min(1, lumaStd / 255 / 0.12);
            double visibilityQuality = Math.sqrt(Math.max(0, focusQuality * exposureQuality * contrastQuality))
                    * (1 - occlusionFraction);
            profiler.add("static_diff_quality_reductions", System.nanoTime() - segmentStarted);

            double shiftX = 0;
            double shiftY = 0;
            double shiftResponse = 0;
            segmentStarted = System.nanoTime();
            if (previousGray != null) {
                Mat first = new Mat();
                Mat second = new Mat();
                Mat window = new Mat();
                try {
                    previousGray.convertTo(first, CvType.CV_32F);
                    gray.convertTo(second, CvType.CV_32F);
                    double[] response = new double[1];
                    Point shift = Imgproc.phaseCorrelate(first, second, window, response);
                    shiftX = shift.x;
                    shiftY = shift.y;
                    shiftResponse = Math.max(0, Math.min(1, response[0]));
                } catch (RuntimeException ignored) {
                    // Optional camera-motion features use the same zero fallback as the web path.
                } finally {
                    first.release();
                    second.release();
                    window.release();
                }
            }
            profiler.add("phase_correlation", System.nanoTime() - segmentStarted);
            double cameraShiftMagnitude = Math.hypot(shiftX, shiftY) / diagonal;
            add(values, focusQuality, blurProbability, darkFraction, brightFraction,
                    (double) lowTexture / pixelsCount, occlusionFraction, visibilityQuality,
                    shiftX / width, shiftY / height, cameraShiftMagnitude, shiftResponse);

            float[] flowValues = new float[pixelsCount * 2];
            segmentStarted = System.nanoTime();
            if (previousGray != null) {
                Video.calcOpticalFlowFarneback(previousGray, gray, flow, 0.5, 2, 13, 2, 5, 1.1, 0);
                flow.get(0, 0, flowValues);
            }
            profiler.add("farneback_optical_flow", System.nanoTime() - segmentStarted);
            segmentStarted = System.nanoTime();
            float[] magnitude = new float[pixelsCount];
            float[] flowX = new float[pixelsCount];
            float[] flowY = new float[pixelsCount];
            int activeFlow = 0;
            for (int i = 0; i < pixelsCount; i++) {
                flowX[i] = flowValues[i * 2];
                flowY[i] = flowValues[i * 2 + 1];
                magnitude[i] = (float) Math.hypot(flowX[i], flowY[i]);
                if (magnitude[i] >= 1) activeFlow++;
            }
            float medianX = FeatureMath.quantile(flowX, 0.5);
            float medianY = FeatureMath.quantile(flowY, 0.5);
            add(values, FeatureMath.mean(magnitude) / diagonal,
                    FeatureMath.quantile(magnitude, 0.9) / diagonal,
                    (double) activeFlow / pixelsCount, medianX / width, medianY / height);
            addScaled(values, gridMeans(magnitude), 1 / diagonal);

            float[] residual = new float[pixelsCount];
            int activeResidual = 0;
            double activeVectorX = 0;
            double activeVectorY = 0;
            double activeMagnitude = 0;
            for (int i = 0; i < pixelsCount; i++) {
                double x = flowX[i] - medianX;
                double y = flowY[i] - medianY;
                residual[i] = (float) Math.hypot(x, y);
                if (residual[i] >= 1) {
                    activeResidual++;
                    activeVectorX += x;
                    activeVectorY += y;
                    activeMagnitude += residual[i];
                }
            }
            float[] residualGrid = gridMeans(residual);
            double gridTotal = 0;
            for (float value : residualGrid) gridTotal += value;
            double entropy = 0;
            if (gridTotal > 1e-9) {
                for (float value : residualGrid) {
                    double probability = value / gridTotal;
                    if (probability > 0) entropy -= probability * Math.log(probability);
                }
                entropy /= Math.log(residualGrid.length);
            }
            double[] geometry = weightedMotionGeometry(residual);
            double coherence = activeResidual > 0
                    ? Math.hypot(activeVectorX / activeResidual, activeVectorY / activeResidual)
                    / Math.max(activeMagnitude / activeResidual, 1e-6)
                    : 0;
            double cameraGate = Math.max(0, 1 - Math.min(1, cameraShiftMagnitude / 0.03));
            double residualMean = FeatureMath.mean(residual) / diagonal;
            int activeZones = 0;
            for (float value : residualGrid) if (value >= 0.75) activeZones++;
            add(values, residualMean, FeatureMath.quantile(residual, 0.9) / diagonal,
                    (double) activeResidual / pixelsCount, activeZones / 9.0, entropy,
                    geometry[0], geometry[1], geometry[2], geometry[3], coherence,
                    residualMean * visibilityQuality * cameraGate);
            addScaled(values, residualGrid, 1 / diagonal);
            profiler.add("flow_and_residual_reductions", System.nanoTime() - segmentStarted);

            if (values.size() != FeatureSchema.FRAME.size()) {
                throw new IllegalStateException("Visual feature signature mismatch: " + values.size());
            }
            segmentStarted = System.nanoTime();
            float[] result = new float[values.size()];
            for (int i = 0; i < values.size(); i++) result[i] = values.get(i);
            if (previousGray != null) previousGray.release();
            previousGray = gray.clone();
            profiler.add("frame_state_and_result_copy", System.nanoTime() - segmentStarted);
            return result;
        } finally {
            long cleanupStarted = System.nanoTime();
            gray.release();
            rgb.release();
            hsv.release();
            edges.release();
            laplacian.release();
            gradientX.release();
            gradientY.release();
            flow.release();
            profiler.add("mat_cleanup", System.nanoTime() - cleanupStarted);
            profiler.add("opencv_feature_total", System.nanoTime() - extractStarted);
        }
    }

    java.util.Map<String, Double> performanceMilliseconds() {
        return profiler.milliseconds();
    }

    private float[] gridMeans(float[] source) {
        float[] output = new float[9];
        int index = 0;
        for (int row = 0; row < 3; row++) {
            int top = Math.round(row * height / 3f);
            int bottom = Math.round((row + 1) * height / 3f);
            for (int column = 0; column < 3; column++) {
                int left = Math.round(column * width / 3f);
                int right = Math.round((column + 1) * width / 3f);
                double total = 0;
                int count = 0;
                for (int y = top; y < bottom; y++) {
                    for (int x = left; x < right; x++) {
                        total += source[y * width + x];
                        count++;
                    }
                }
                output[index++] = count == 0 ? 0 : (float) (total / count);
            }
        }
        return output;
    }

    private double[] weightedMotionGeometry(float[] magnitude) {
        double[] xWeights = new double[width];
        double[] yWeights = new double[height];
        double total = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float value = magnitude[y * width + x];
                if (value < 0.5) continue;
                xWeights[x] += value;
                yWeights[y] += value;
                total += value;
            }
        }
        if (total <= 1e-9) return new double[]{0.5, 0.5, 0, 0};
        double centroidX = 0;
        double centroidY = 0;
        for (int x = 0; x < width; x++) centroidX += xWeights[x] * ((x + 0.5) / width);
        for (int y = 0; y < height; y++) centroidY += yWeights[y] * ((y + 0.5) / height);
        centroidX /= total;
        centroidY /= total;
        double varianceX = 0;
        double varianceY = 0;
        for (int x = 0; x < width; x++) varianceX += xWeights[x] * Math.pow((x + 0.5) / width - centroidX, 2);
        for (int y = 0; y < height; y++) varianceY += yWeights[y] * Math.pow((y + 0.5) / height - centroidY, 2);
        return new double[]{centroidX, centroidY, Math.sqrt(varianceX / total), Math.sqrt(varianceY / total)};
    }

    private static float[] rectValues(float[] source, int left, int top, int right, int bottom) {
        float[] result = new float[Math.max(0, right - left) * Math.max(0, bottom - top)];
        int index = 0;
        for (int y = top; y < bottom; y++) {
            for (int x = left; x < right; x++) result[index++] = source[y * FeatureSchema.ANALYSIS_WIDTH + x];
        }
        return result;
    }

    private static float[] unsigned(byte[] values) {
        float[] output = new float[values.length];
        for (int i = 0; i < values.length; i++) output[i] = values[i] & 0xff;
        return output;
    }

    private static void add(List<Float> target, double... values) {
        for (double value : values) target.add((float) value);
    }

    private static void addScaled(List<Float> target, float[] values, double scale) {
        for (float value : values) target.add((float) (value * scale));
    }

    @Override
    public void close() {
        if (previousGray != null) {
            previousGray.release();
            previousGray = null;
        }
    }
}
