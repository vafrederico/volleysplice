package com.volleycut.nativeanalysis;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/** Exact native port of the frozen SIDE-SWITCH-UNION34-V1 visual feature contract. */
final class SideSwitchFeatureExtractor {
    private static final int WIDTH = SideSwitchModelsKt.SIDE_SWITCH_WIDTH;
    private static final int HEIGHT = SideSwitchModelsKt.SIDE_SWITCH_HEIGHT;
    private static final int CHANNELS = 3;
    private static final int SIZE = WIDTH * HEIGHT;
    private static final double CANONICAL_NET_Y = .5;

    record CourtGeometry(double netYRatio, double confidence) {}
    private record Converted(byte[] gray, byte[] hsv) {}
    private record Prepared(
            Converted[] converted,
            double[] median,
            double[] motion,
            double maximumShift,
            double minimumResponse
    ) {}
    private record Palette(double[] values, double instability) {}
    private record PaletteSummary(double[] near, double[] far, double instability) {}
    private record BroadSummary(
            PaletteSummary broad,
            PaletteSummary tight,
            double[] global,
            double foregroundCoverage,
            double maximumCameraShift,
            double minimumAlignmentResponse
    ) {}
    private record WeightedPalette(double[] palette, double weight) {}
    private record PlayerSummary(
            double[] near,
            double[] far,
            double[] global,
            double instability,
            double proposalCoverage,
            double proposalCount,
            double nearSupport,
            double farSupport
    ) {}
    private record Box(int x, int y, int width, int height, int area) {}
    private record ScoredBox(Box box, double score) {}

    private SideSwitchFeatureExtractor() {}

    static CourtGeometry estimateCourtGeometry(byte[][] frames) {
        ArrayList<double[]> candidates = new ArrayList<>();
        for (byte[] frame : frames) {
            double[] candidate = horizontalLineCandidate(frame);
            if (candidate != null) candidates.add(candidate);
        }
        if (candidates.isEmpty()) return new CourtGeometry(.5, 0);
        double[] positions = candidates.stream().mapToDouble(value -> value[0]).toArray();
        double center = quantile(positions, .5);
        double initialCenter = center;
        double[] deviations = Arrays.stream(positions)
                .map(value -> Math.abs(value - initialCenter)).toArray();
        double cutoff = Math.max(.035, quantile(deviations, .5) * 2.5);
        double weighted = 0;
        double weight = 0;
        for (int index = 0; index < candidates.size(); index++) {
            if (deviations[index] <= cutoff) {
                weighted += candidates.get(index)[0] * candidates.get(index)[1];
                weight += candidates.get(index)[1];
            }
        }
        if (weight > 0) center = weighted / weight;
        double median = quantile(positions, .5);
        double dispersion = quantile(
                Arrays.stream(positions).map(value -> Math.abs(value - median)).toArray(), .5
        );
        return new CourtGeometry(
                clamp(center, .25, .68),
                clamp((double) candidates.size() / frames.length * Math.max(0, 1 - dispersion / .1), 0, 1)
        );
    }

    static double[] extract(byte[][] beforeFrames, byte[][] afterFrames, CourtGeometry geometry) {
        if (beforeFrames.length != 7 || afterFrames.length != 7) {
            throw new IllegalArgumentException("Side-switch comparisons require seven frames per side");
        }
        Prepared beforePrepared = prepare(beforeFrames, geometry);
        Prepared afterPrepared = prepare(afterFrames, geometry);
        BroadSummary beforeBroad = summarizeBroad(beforePrepared);
        BroadSummary afterBroad = summarizeBroad(afterPrepared);
        PlayerSummary beforePlayers = summarizePlayers(beforePrepared);
        PlayerSummary afterPlayers = summarizePlayers(afterPrepared);
        double[] broad = broadFeatures(beforeBroad, afterBroad);
        double[] result = playerFeatures(beforePlayers, afterPlayers, broad);
        if (result.length != 22 || Arrays.stream(result).anyMatch(value -> !Double.isFinite(value))) {
            throw new IllegalStateException("Side-switch visual feature contract changed");
        }
        return result;
    }

    private static BroadSummary summarizeBroad(Prepared prepared) {
        Palette broadFar = pooledRegion(prepared, .15, .6);
        Palette broadNear = pooledRegion(prepared, .44, .98);
        Palette tightFar = pooledRegion(prepared, .27, .53);
        Palette tightNear = pooledRegion(prepared, .56, .91);
        Palette global = pooledRegion(prepared, .08, .98);
        int active = 0;
        for (double value : prepared.motion) if (value > 10 / 255.0) active++;
        return new BroadSummary(
                new PaletteSummary(
                        broadNear.values, broadFar.values,
                        (broadNear.instability + broadFar.instability) / 2
                ),
                new PaletteSummary(
                        tightNear.values, tightFar.values,
                        (tightNear.instability + tightFar.instability) / 2
                ),
                global.values,
                (double) active / prepared.motion.length,
                prepared.maximumShift,
                prepared.minimumResponse
        );
    }

    private static PlayerSummary summarizePlayers(Prepared prepared) {
        ArrayList<WeightedPalette> near = new ArrayList<>();
        ArrayList<WeightedPalette> far = new ArrayList<>();
        ArrayList<WeightedPalette> global = new ArrayList<>();
        double[] coverages = new double[prepared.converted.length];
        double[] counts = new double[prepared.converted.length];
        double nearSupport = 0;
        double farSupport = 0;
        for (int frameIndex = 0; frameIndex < prepared.converted.length; frameIndex++) {
            Converted frame = prepared.converted[frameIndex];
            double[] difference = new double[SIZE];
            for (int pixel = 0; pixel < SIZE; pixel++) {
                difference[pixel] = Math.abs((frame.gray[pixel] & 0xff) - prepared.median[pixel]) / 255;
            }
            List<Box> boxes = proposalBoxes(difference);
            byte[] covered = new byte[SIZE];
            counts[frameIndex] = boxes.size();
            for (Box box : boxes) {
                double[] weights = new double[SIZE];
                double support = 0;
                for (int y = box.y; y < box.y + box.height; y++) {
                    for (int x = box.x; x < box.x + box.width; x++) {
                        int pixel = y * WIDTH + x;
                        covered[pixel] = 1;
                        double saturation = (frame.hsv[pixel * 3 + 1] & 0xff) / 255.0;
                        weights[pixel] = difference[pixel] + .15 * prepared.motion[pixel] + .01 * saturation;
                        support += weights[pixel];
                    }
                }
                support = Math.max(support, 1e-8);
                double[] palette = weightedPalette(
                        frame.hsv, weights, box.x, box.y,
                        box.x + box.width, box.y + box.height, true
                );
                double footY = (double) (box.y + box.height) / HEIGHT;
                double nearProbability = 1 / (1 + Math.exp(-(footY - .63) / .06));
                double farProbability = 1 - nearProbability;
                if (nearProbability >= .08) {
                    double itemWeight = support * nearProbability;
                    near.add(new WeightedPalette(palette, itemWeight));
                    nearSupport += itemWeight;
                }
                if (farProbability >= .08) {
                    double itemWeight = support * farProbability;
                    far.add(new WeightedPalette(palette, itemWeight));
                    farSupport += itemWeight;
                }
                global.add(new WeightedPalette(palette, support));
            }
            int coveredPixels = 0;
            for (byte value : covered) coveredPixels += value;
            coverages[frameIndex] = (double) coveredPixels / covered.length;
        }
        Palette nearPalette = pooledWeightedPalettes(near);
        Palette farPalette = pooledWeightedPalettes(far);
        Palette globalPalette = pooledWeightedPalettes(global);
        double totalSupport = Math.max(nearSupport + farSupport, 1e-12);
        return new PlayerSummary(
                nearPalette.values, farPalette.values, globalPalette.values,
                (nearPalette.instability + farPalette.instability) / 2,
                mean(coverages), mean(counts),
                nearSupport / totalSupport, farSupport / totalSupport
        );
    }

    private static Prepared prepare(byte[][] frames, CourtGeometry geometry) {
        byte[][] normalized = new byte[frames.length][];
        for (int index = 0; index < frames.length; index++) {
            normalized[index] = normalizeCourtFrame(frames[index], geometry);
        }
        Object[] aligned = alignFrames(normalized);
        byte[][] alignedFrames = (byte[][]) aligned[0];
        Converted[] converted = new Converted[alignedFrames.length];
        for (int index = 0; index < alignedFrames.length; index++) {
            converted[index] = grayAndHsv(alignedFrames[index]);
        }
        double[] median = new double[SIZE];
        double[] motion = new double[SIZE];
        double[] values = new double[converted.length];
        for (int pixel = 0; pixel < SIZE; pixel++) {
            for (int frame = 0; frame < converted.length; frame++) {
                values[frame] = converted[frame].gray[pixel] & 0xff;
            }
            Arrays.sort(values);
            median[pixel] = quantile(values, .5);
            motion[pixel] = (values[values.length - 1] - values[0]) / 255;
        }
        double[] quality = (double[]) aligned[1];
        return new Prepared(converted, median, motion, quality[0], quality[1]);
    }

    /** Returns aligned frames and [maximum normalized shift, minimum response]. */
    private static Object[] alignFrames(byte[][] frames) {
        int calibrationHeight = Math.round(HEIGHT * .42f);
        Mat reference = bgr(frames[frames.length / 2]);
        Mat referenceGray = new Mat();
        Mat referenceBlur = new Mat();
        Mat referenceRoi = null;
        Mat referenceWindowed = null;
        byte[][] aligned = new byte[frames.length][];
        double maximumShift = 0;
        double minimumResponse = Double.POSITIVE_INFINITY;
        try {
            Imgproc.cvtColor(reference, referenceGray, Imgproc.COLOR_BGR2GRAY);
            referenceRoi = referenceGray.submat(new Rect(0, 0, WIDTH, calibrationHeight));
            Imgproc.GaussianBlur(referenceRoi, referenceBlur, new Size(7, 7), 0);
            referenceWindowed = hanningWindowed(referenceBlur);
            for (int index = 0; index < frames.length; index++) {
                Mat source = bgr(frames[index]);
                Mat gray = new Mat();
                Mat blur = new Mat();
                Mat roi = null;
                Mat windowed = null;
                Mat phaseWindow = new Mat();
                Mat transform = null;
                Mat output = new Mat();
                try {
                    Imgproc.cvtColor(source, gray, Imgproc.COLOR_BGR2GRAY);
                    roi = gray.submat(new Rect(0, 0, WIDTH, calibrationHeight));
                    Imgproc.GaussianBlur(roi, blur, new Size(7, 7), 0);
                    windowed = hanningWindowed(blur);
                    double[] response = new double[1];
                    Point shift = Imgproc.phaseCorrelate(referenceWindowed, windowed, phaseWindow, response);
                    double normalizedShift = Math.hypot(shift.x / WIDTH, shift.y / HEIGHT);
                    double finiteResponse = Double.isFinite(response[0]) ? response[0] : 0;
                    minimumResponse = Math.min(minimumResponse, finiteResponse);
                    if (Double.isFinite(normalizedShift) && finiteResponse >= .02 && normalizedShift <= .12) {
                        transform = new Mat(2, 3, CvType.CV_64FC1);
                        transform.put(0, 0, 1, 0, -shift.x, 0, 1, -shift.y);
                        Imgproc.warpAffine(
                                source, output, transform, new Size(WIDTH, HEIGHT),
                                Imgproc.INTER_LINEAR, Core.BORDER_REFLECT, Scalar.all(0)
                        );
                        maximumShift = Math.max(maximumShift, normalizedShift);
                        aligned[index] = bytes(output, SIZE * CHANNELS);
                    } else {
                        aligned[index] = frames[index].clone();
                    }
                } catch (RuntimeException ignored) {
                    minimumResponse = Math.min(minimumResponse, 0);
                    aligned[index] = frames[index].clone();
                } finally {
                    source.release(); gray.release(); blur.release(); output.release();
                    phaseWindow.release();
                    if (roi != null) roi.release();
                    if (windowed != null) windowed.release();
                    if (transform != null) transform.release();
                }
            }
        } finally {
            reference.release(); referenceGray.release(); referenceBlur.release();
            if (referenceRoi != null) referenceRoi.release();
            if (referenceWindowed != null) referenceWindowed.release();
        }
        if (!Double.isFinite(minimumResponse)) minimumResponse = 0;
        return new Object[]{aligned, new double[]{maximumShift, minimumResponse}};
    }

    private static Mat hanningWindowed(Mat source) {
        byte[] sourceBytes = bytes(source, source.rows() * source.cols());
        float[] values = new float[sourceBytes.length];
        for (int y = 0; y < source.rows(); y++) {
            double vertical = source.rows() <= 1 ? 1 :
                    .5 * (1 - Math.cos(2 * Math.PI * y / (source.rows() - 1)));
            for (int x = 0; x < source.cols(); x++) {
                double horizontal = source.cols() <= 1 ? 1 :
                        .5 * (1 - Math.cos(2 * Math.PI * x / (source.cols() - 1)));
                int index = y * source.cols() + x;
                values[index] = (float) ((sourceBytes[index] & 0xff)
                        * Math.sqrt(vertical * horizontal));
            }
        }
        Mat result = new Mat(source.rows(), source.cols(), CvType.CV_32FC1);
        result.put(0, 0, values);
        return result;
    }

    private static byte[] normalizeCourtFrame(byte[] frame, CourtGeometry geometry) {
        Mat source = bgr(frame);
        Mat output = new Mat();
        Mat mapX = new Mat(HEIGHT, WIDTH, CvType.CV_32FC1);
        Mat mapY = new Mat(HEIGHT, WIDTH, CvType.CV_32FC1);
        float[] xValues = new float[SIZE];
        float[] yValues = new float[SIZE];
        double net = clamp(geometry.netYRatio, .2, .8);
        for (int y = 0; y < HEIGHT; y++) {
            double outputY = (double) y / Math.max(HEIGHT - 1, 1);
            double sourceY = outputY <= CANONICAL_NET_Y
                    ? outputY * net / CANONICAL_NET_Y
                    : net + (outputY - CANONICAL_NET_Y) * (1 - net) / (1 - CANONICAL_NET_Y);
            for (int x = 0; x < WIDTH; x++) {
                int index = y * WIDTH + x;
                xValues[index] = x;
                yValues[index] = (float) (sourceY * (HEIGHT - 1));
            }
        }
        mapX.put(0, 0, xValues);
        mapY.put(0, 0, yValues);
        try {
            Imgproc.remap(
                    source, output, mapX, mapY, Imgproc.INTER_LINEAR,
                    Core.BORDER_REPLICATE, Scalar.all(0)
            );
            return bytes(output, SIZE * CHANNELS);
        } finally {
            source.release(); output.release(); mapX.release(); mapY.release();
        }
    }

    private static Converted grayAndHsv(byte[] frame) {
        Mat source = bgr(frame);
        Mat gray = new Mat();
        Mat hsv = new Mat();
        try {
            Imgproc.cvtColor(source, gray, Imgproc.COLOR_BGR2GRAY);
            Imgproc.cvtColor(source, hsv, Imgproc.COLOR_BGR2HSV);
            return new Converted(bytes(gray, SIZE), bytes(hsv, SIZE * CHANNELS));
        } finally {
            source.release(); gray.release(); hsv.release();
        }
    }

    private static Palette pooledRegion(Prepared prepared, double topRatio, double bottomRatio) {
        int top = Math.round(HEIGHT * (float) topRatio);
        int bottom = Math.round(HEIGHT * (float) bottomRatio);
        int left = Math.round(WIDTH * .06f);
        int right = Math.round(WIDTH * .94f);
        double[][] palettes = new double[prepared.converted.length][];
        for (int frameIndex = 0; frameIndex < prepared.converted.length; frameIndex++) {
            Converted frame = prepared.converted[frameIndex];
            double[] weights = new double[SIZE];
            for (int pixel = 0; pixel < SIZE; pixel++) {
                weights[pixel] = Math.max(
                        Math.abs((frame.gray[pixel] & 0xff) - prepared.median[pixel]) / 255 - 2 / 255.0,
                        0
                ) + .2 * Math.max(prepared.motion[pixel] - 5 / 255.0, 0);
            }
            palettes[frameIndex] = weightedPalette(frame.hsv, weights, left, top, right, bottom, false);
        }
        double[] palette = normalizedAverage(Arrays.asList(palettes), null);
        double[] differences = new double[palettes.length];
        for (int index = 0; index < palettes.length; index++) {
            differences[index] = hellinger(palettes[index], palette);
        }
        return new Palette(palette, quantile(differences, .5));
    }

    private static double[] weightedPalette(
            byte[] hsv,
            double[] weights,
            int left,
            int top,
            int right,
            int bottom,
            boolean uniformWhenEmpty
    ) {
        double[] bins = new double[52];
        double total = 0;
        for (int y = top; y < bottom; y++) {
            for (int x = left; x < right; x++) {
                int pixel = y * WIDTH + x;
                double weight = weights[pixel];
                if (weight <= 0) continue;
                int offset = pixel * CHANNELS;
                int hue = Math.min(11, (hsv[offset] & 0xff) / 15);
                int saturation = Math.min(3, (hsv[offset + 1] & 0xff) / 64);
                int value = Math.min(3, (hsv[offset + 2] & 0xff) / 64);
                bins[hue * 4 + saturation] += weight;
                bins[48 + value] += weight;
                total += weight * 2;
            }
        }
        if (total < 1e-8) {
            if (uniformWhenEmpty) {
                Arrays.fill(bins, 1.0 / bins.length);
                return bins;
            }
            double[] uniform = new double[weights.length];
            Arrays.fill(uniform, 1);
            return weightedPalette(hsv, uniform, left, top, right, bottom, true);
        }
        for (int index = 0; index < bins.length; index++) bins[index] /= total;
        return bins;
    }

    private static double[] normalizedAverage(List<double[]> values, List<Double> weights) {
        if (values.isEmpty()) {
            double[] uniform = new double[52];
            Arrays.fill(uniform, 1.0 / uniform.length);
            return uniform;
        }
        double[] output = new double[values.get(0).length];
        double weightTotal = 0;
        for (int row = 0; row < values.size(); row++) {
            double weight = weights == null ? 1 : weights.get(row);
            weightTotal += weight;
            for (int index = 0; index < output.length; index++) {
                output[index] += values.get(row)[index] * weight;
            }
        }
        double total = 0;
        for (int index = 0; index < output.length; index++) {
            output[index] /= Math.max(weightTotal, 1e-12);
            total += output[index];
        }
        for (int index = 0; index < output.length; index++) output[index] /= Math.max(total, 1e-12);
        return output;
    }

    private static double[] broadFeatures(BroadSummary before, BroadSummary after) {
        double[] broad = assignment(before.broad, after.broad);
        double[] tight = assignment(before.tight, after.tight);
        return new double[]{
                broad[0],
                tight[0],
                (broad[2] + tight[2]) / 2,
                hellinger(before.global, after.global),
                Math.max(before.maximumCameraShift, after.maximumCameraShift),
                Math.min(before.minimumAlignmentResponse, after.minimumAlignmentResponse),
        };
    }

    private static double[] playerFeatures(PlayerSummary before, PlayerSummary after, double[] v4) {
        double same = (hellinger(before.near, after.near) + hellinger(before.far, after.far)) / 2;
        double swapped = (hellinger(before.near, after.far) + hellinger(before.far, after.near)) / 2;
        double[] beforeOrientation = subtract(before.near, before.far);
        double[] afterOrientation = subtract(after.near, after.far);
        double beforeSeparation = hellinger(before.near, before.far);
        double afterSeparation = hellinger(after.near, after.far);
        return new double[]{
                v4[0], v4[1], v4[2], v4[3], v4[4], v4[5],
                same, swapped, same - swapped, -cosine(beforeOrientation, afterOrientation),
                Math.min(beforeSeparation, afterSeparation),
                Math.abs(beforeSeparation - afterSeparation),
                before.instability, after.instability,
                hellinger(before.global, after.global),
                Math.min(before.proposalCoverage, after.proposalCoverage),
                Math.abs(before.proposalCoverage - after.proposalCoverage),
                Math.min(before.proposalCount, after.proposalCount),
                Math.abs(before.proposalCount - after.proposalCount),
                Math.min(before.nearSupport, after.nearSupport),
                Math.min(before.farSupport, after.farSupport),
                Math.abs(Math.abs(before.nearSupport - before.farSupport)
                        - Math.abs(after.nearSupport - after.farSupport)),
        };
    }

    private static double[] assignment(PaletteSummary before, PaletteSummary after) {
        double same = (hellinger(before.near, after.near) + hellinger(before.far, after.far)) / 2;
        double swapped = (hellinger(before.near, after.far) + hellinger(before.far, after.near)) / 2;
        double[] beforeOrientation = subtract(before.near, before.far);
        double[] afterOrientation = subtract(after.near, after.far);
        return new double[]{same, swapped, same - swapped, -cosine(beforeOrientation, afterOrientation)};
    }

    private static List<Box> proposalBoxes(double[] difference) {
        byte[] numeric = new byte[SIZE];
        double[] unsigned = new double[SIZE];
        for (int index = 0; index < SIZE; index++) {
            int value = (int) clamp(Math.floor(difference[index] * 255), 0, 255);
            numeric[index] = (byte) value;
            unsigned[index] = value;
        }
        int threshold = Math.max(12, (int) Math.floor(quantile(unsigned, .94)));
        byte[] mask = new byte[SIZE];
        for (int index = 0; index < SIZE; index++) mask[index] = (byte) ((numeric[index] & 0xff) >= threshold ? 1 : 0);
        ArrayList<ScoredBox> candidates = new ArrayList<>();
        for (Box box : components(closeMask(mask))) {
            int bottom = box.y + box.height;
            double aspect = (double) box.height / Math.max(box.width, 1);
            if (box.area < 10 || box.area > SIZE * .035 || box.width < 2 || box.height < 4
                    || box.width > WIDTH * .18 || box.height > HEIGHT * .48
                    || aspect < .45 || aspect > 5.5 || bottom < HEIGHT * .35 || box.y > HEIGHT * .95) {
                continue;
            }
            int padX = Math.max(1, Math.round(box.width * .18f));
            int padY = Math.max(1, Math.round(box.height * .1f));
            int x = Math.max(0, box.x - padX);
            int y = Math.max(0, box.y - padY);
            int right = Math.min(WIDTH, box.x + box.width + padX);
            int lower = Math.min(HEIGHT, bottom + padY);
            Box expanded = new Box(x, y, right - x, lower - y, box.area);
            double score = box.area * Math.pow(.5 + (double) lower / HEIGHT, 2) * Math.min(aspect, 2.5);
            candidates.add(new ScoredBox(expanded, score));
        }
        candidates.sort(Comparator.comparingDouble(ScoredBox::score).reversed());
        ArrayList<Box> selected = new ArrayList<>();
        for (ScoredBox candidate : candidates) {
            if (selected.stream().allMatch(box -> iou(candidate.box, box) < .35)) selected.add(candidate.box);
            if (selected.size() == 6) break;
        }
        return selected;
    }

    private static byte[] closeMask(byte[] mask) {
        int[][] offsets = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        byte[] dilated = new byte[SIZE];
        for (int y = 0; y < HEIGHT; y++) for (int x = 0; x < WIDTH; x++) {
            for (int[] offset : offsets) {
                int nextX = x + offset[0];
                int nextY = y + offset[1];
                if (nextX >= 0 && nextX < WIDTH && nextY >= 0 && nextY < HEIGHT
                        && mask[nextY * WIDTH + nextX] > 0) {
                    dilated[y * WIDTH + x] = 1;
                    break;
                }
            }
        }
        byte[] eroded = new byte[SIZE];
        for (int y = 0; y < HEIGHT; y++) for (int x = 0; x < WIDTH; x++) {
            boolean all = true;
            for (int[] offset : offsets) {
                int nextX = x + offset[0];
                int nextY = y + offset[1];
                if (nextX < 0 || nextX >= WIDTH || nextY < 0 || nextY >= HEIGHT
                        || dilated[nextY * WIDTH + nextX] == 0) {
                    all = false;
                    break;
                }
            }
            eroded[y * WIDTH + x] = (byte) (all ? 1 : 0);
        }
        return eroded;
    }

    private static List<Box> components(byte[] mask) {
        byte[] seen = new byte[SIZE];
        int[] queue = new int[SIZE];
        ArrayList<Box> result = new ArrayList<>();
        for (int origin = 0; origin < SIZE; origin++) {
            if (mask[origin] == 0 || seen[origin] != 0) continue;
            int head = 0;
            int tail = 0;
            int minX = WIDTH;
            int maxX = 0;
            int minY = HEIGHT;
            int maxY = 0;
            queue[tail++] = origin;
            seen[origin] = 1;
            while (head < tail) {
                int pixel = queue[head++];
                int y = pixel / WIDTH;
                int x = pixel - y * WIDTH;
                minX = Math.min(minX, x); maxX = Math.max(maxX, x);
                minY = Math.min(minY, y); maxY = Math.max(maxY, y);
                for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    int nextX = x + dx;
                    int nextY = y + dy;
                    if (nextX < 0 || nextX >= WIDTH || nextY < 0 || nextY >= HEIGHT) continue;
                    int next = nextY * WIDTH + nextX;
                    if (mask[next] != 0 && seen[next] == 0) {
                        seen[next] = 1;
                        queue[tail++] = next;
                    }
                }
            }
            result.add(new Box(minX, minY, maxX - minX + 1, maxY - minY + 1, tail));
        }
        return result;
    }

    private static Palette pooledWeightedPalettes(List<WeightedPalette> values) {
        if (values.isEmpty()) {
            double[] uniform = new double[52];
            Arrays.fill(uniform, 1.0 / uniform.length);
            return new Palette(uniform, 0);
        }
        List<double[]> palettes = values.stream().map(WeightedPalette::palette).toList();
        List<Double> weights = values.stream().map(WeightedPalette::weight).toList();
        double[] palette = normalizedAverage(palettes, weights);
        double[] differences = values.stream().mapToDouble(value -> hellinger(value.palette, palette)).toArray();
        return new Palette(palette, quantile(differences, .5));
    }

    private static double[] horizontalLineCandidate(byte[] frame) {
        Mat source = bgr(frame);
        Mat gray = new Mat();
        Mat blurred = new Mat();
        Mat edges = new Mat();
        Mat lines = new Mat();
        try {
            Imgproc.cvtColor(source, gray, Imgproc.COLOR_BGR2GRAY);
            Imgproc.GaussianBlur(gray, blurred, new Size(5, 5), 0);
            Imgproc.Canny(blurred, edges, 45, 110);
            Imgproc.HoughLinesP(
                    edges, lines, 1, Math.PI / 180,
                    Math.max(24, WIDTH / 10), WIDTH * .32, WIDTH * .12
            );
            int[] data = new int[(int) (lines.total() * lines.channels())];
            lines.get(0, 0, data);
            double[] best = null;
            for (int index = 0; index + 3 < data.length; index += 4) {
                int x1 = data[index]; int y1 = data[index + 1];
                int x2 = data[index + 2]; int y2 = data[index + 3];
                double length = Math.hypot(x2 - x1, y2 - y1);
                double coverage = (double) Math.abs(x2 - x1) / WIDTH;
                double position = (double) (y1 + y2) / (2 * HEIGHT);
                double slope = (double) Math.abs(y2 - y1) / Math.max(Math.abs(x2 - x1), 1);
                if (position < .25 || position > .68 || slope > .08 || coverage < .3) continue;
                double score = Math.min(1, coverage * (1 + .35 * position) + .1 * length / WIDTH);
                if (best == null || score > best[1]) best = new double[]{position, score};
            }
            return best;
        } finally {
            source.release(); gray.release(); blurred.release(); edges.release(); lines.release();
        }
    }

    private static double iou(Box first, Box second) {
        int width = Math.max(0, Math.min(first.x + first.width, second.x + second.width)
                - Math.max(first.x, second.x));
        int height = Math.max(0, Math.min(first.y + first.height, second.y + second.height)
                - Math.max(first.y, second.y));
        int intersection = width * height;
        int union = first.width * first.height + second.width * second.height - intersection;
        return union == 0 ? 0 : (double) intersection / union;
    }

    private static double[] subtract(double[] left, double[] right) {
        double[] result = new double[left.length];
        for (int index = 0; index < result.length; index++) result[index] = left[index] - right[index];
        return result;
    }

    private static double hellinger(double[] left, double[] right) {
        double total = 0;
        for (int index = 0; index < left.length; index++) {
            total += Math.pow(Math.sqrt(Math.max(left[index], 0)) - Math.sqrt(Math.max(right[index], 0)), 2);
        }
        return Math.sqrt(total) / Math.sqrt(2);
    }

    private static double cosine(double[] left, double[] right) {
        double dot = 0;
        double leftNorm = 0;
        double rightNorm = 0;
        for (int index = 0; index < left.length; index++) {
            dot += left[index] * right[index];
            leftNorm += left[index] * left[index];
            rightNorm += right[index] * right[index];
        }
        double denominator = Math.sqrt(leftNorm * rightNorm);
        return denominator > 1e-12 ? dot / denominator : 0;
    }

    private static double quantile(double[] values, double probability) {
        if (values.length == 0) return 0;
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        double position = (sorted.length - 1) * clamp(probability, 0, 1);
        int lower = (int) Math.floor(position);
        int upper = (int) Math.ceil(position);
        if (lower == upper) return sorted[lower];
        return sorted[lower] * (upper - position) + sorted[upper] * (position - lower);
    }

    private static double mean(double[] values) {
        return values.length == 0 ? 0 : Arrays.stream(values).average().orElse(0);
    }

    private static Mat bgr(byte[] values) {
        if (values.length != SIZE * CHANNELS) throw new IllegalArgumentException("Invalid side-switch frame");
        Mat result = new Mat(HEIGHT, WIDTH, CvType.CV_8UC3);
        result.put(0, 0, values);
        return result;
    }

    private static byte[] bytes(Mat source, int length) {
        byte[] values = new byte[length];
        source.get(0, 0, values);
        return values;
    }

    private static double clamp(double value, double minimum, double maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }
}
