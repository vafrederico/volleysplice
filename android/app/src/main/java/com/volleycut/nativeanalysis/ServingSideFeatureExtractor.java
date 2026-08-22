package com.volleycut.nativeanalysis;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/** Exact native port of the frozen SERVSIDE237-FLIGHT raw feature contract. */
final class ServingSideFeatureExtractor {
    private static final int WIDTH = ServingSideModelsKt.SERVING_SIDE_WIDTH;
    private static final int HEIGHT = ServingSideModelsKt.SERVING_SIDE_HEIGHT;
    private static final int SIZE = WIDTH * HEIGHT;
    private static final int GRID_ROWS = 4;
    private static final int GRID_COLUMNS = 6;

    private record Components(int[] labels, int[] areas, double[] centroidX, double[] centroidY) {}
    private record Motion(double[] energy, double[] flowX, double[] flowY, double[] divergence) {}
    private record PairSummary(double[] grid, double[] rows, double[] statistics) {}

    private ServingSideFeatureExtractor() {}

    static double[] extract(byte[][] courtFrames, byte[][] flightFrames) {
        double[] court = extractCourtFlowRawFeatures(courtFrames);
        double[] flight = extractFlightRawFeatures(flightFrames);
        double[] result = Arrays.copyOf(court, court.length + flight.length);
        System.arraycopy(flight, 0, result, court.length, flight.length);
        if (result.length != ServingSideModelsKt.SERVING_SIDE_FEATURE_COLUMNS) {
            throw new IllegalStateException("Serving-side feature contract changed");
        }
        return result;
    }

    static double[] extractCourtFlowRawFeatures(byte[][] frames) {
        if (frames.length != 8) throw new IllegalArgumentException(
                "Court-flow extraction requires eight ordered frames"
        );
        int[][][] phasePairs = {
                {{0, 1}, {1, 2}},
                {{2, 3}, {3, 4}, {4, 5}},
                {{5, 6}, {6, 7}},
        };
        int band = Math.max(1, Math.round(HEIGHT * .32f));
        double[][][] phases = new double[3][2][];
        for (int phase = 0; phase < phasePairs.length; phase++) {
            List<double[]> nearValues = new ArrayList<>();
            List<double[]> farValues = new ArrayList<>();
            for (int[] pair : phasePairs[phase]) {
                double[] flow = flowFromFrames(frames[pair[0]], frames[pair[1]],
                        .5, 2, 13, 2, 5, 1.1, 0);
                double[] xValues = new double[SIZE];
                double[] yValues = new double[SIZE];
                for (int i = 0; i < SIZE; i++) {
                    xValues[i] = flow[i * 2];
                    yValues[i] = flow[i * 2 + 1];
                }
                double medianX = (float) quantile(xValues, .5);
                double medianY = (float) quantile(yValues, .5);
                double[] residual = new double[SIZE * 2];
                double[] magnitude = new double[SIZE];
                byte[] active = new byte[SIZE];
                for (int i = 0; i < SIZE; i++) {
                    float x = (float) (flow[i * 2] - medianX);
                    float y = (float) (flow[i * 2 + 1] - medianY);
                    residual[i * 2] = x;
                    residual[i * 2 + 1] = y;
                    magnitude[i] = (float) Math.hypot(x, y);
                    active[i] = magnitude[i] >= 1 ? (byte) 1 : 0;
                }
                byte[] opened = opened3x3(active);
                nearValues.add(courtZoneStatistics(
                        residual, magnitude, opened, HEIGHT - band, HEIGHT
                ));
                farValues.add(courtZoneStatistics(residual, magnitude, opened, 0, band));
            }
            phases[phase][0] = meanVectors(nearValues);
            phases[phase][1] = meanVectors(farValues);
        }
        ArrayList<Double> values = new ArrayList<>(82);
        for (int phase = 0; phase < 3; phase++) {
            append(values, phases[phase][0]);
            append(values, phases[phase][1]);
            for (int index = 0; index < 5; index++) {
                values.add(phases[phase][0][index] - phases[phase][1][index]);
            }
        }
        for (int zone = 0; zone < 2; zone++) {
            double[] pre = phases[0][zone];
            double[] contact = phases[1][zone];
            double[] post = phases[2][zone];
            values.add(contact[0] - pre[0]);
            values.add(contact[2] - pre[2]);
            values.add(contact[3] - pre[3]);
            values.add(post[0] - contact[0]);
            values.add(post[2] - contact[2]);
        }
        for (int index : new int[]{0, 2, 3}) {
            values.add((phases[1][0][index] - phases[0][0][index])
                    - (phases[1][1][index] - phases[0][1][index]));
        }
        return finiteArray(values, 82, "Court-flow");
    }

    static double[] extractFlightRawFeatures(byte[][] frames) {
        if (frames.length != 9) throw new IllegalArgumentException(
                "Flight extraction requires nine ordered frames"
        );
        PairSummary[] pairs = new PairSummary[8];
        for (int i = 0; i < pairs.length; i++) {
            pairs[i] = summarizeFlightPair(extractFlightMotion(frames[i], frames[i + 1]));
        }
        int[][] phaseIndices = {{0, 1, 2}, {3, 4, 5}, {6, 7}};
        PairSummary[] phases = new PairSummary[3];
        ArrayList<Double> output = new ArrayList<>(155);
        for (int phase = 0; phase < phaseIndices.length; phase++) {
            List<double[]> grids = new ArrayList<>();
            List<double[]> rows = new ArrayList<>();
            List<double[]> statistics = new ArrayList<>();
            for (int index : phaseIndices[phase]) {
                grids.add(pairs[index].grid);
                rows.add(pairs[index].rows);
                statistics.add(pairs[index].statistics);
            }
            phases[phase] = new PairSummary(
                    meanVectors(grids), meanVectors(rows), meanVectors(statistics)
            );
            append(output, phases[phase].grid);
            append(output, phases[phase].rows);
            append(output, phases[phase].statistics);
        }
        List<String> globals = List.of(
                "energyMean", "activeFraction", "centroidX", "centroidY", "spreadX",
                "spreadY", "entropy", "largestComponentFraction", "flowX", "flowY",
                "divergence", "bottomMinusTop", "smallComponentEnergyFraction",
                "smallComponentCentroidY", "smallComponentFlowY"
        );
        List<String> trajectories = List.of(
                "centroidY", "spreadY", "entropy", "flowY", "divergence", "bottomMinusTop",
                "smallComponentEnergyFraction", "smallComponentCentroidY", "smallComponentFlowY"
        );
        for (int[] transition : new int[][]{{0, 1}, {1, 2}}) {
            for (String name : trajectories) {
                int index = globals.indexOf(name);
                output.add(phases[transition[1]].statistics[index]
                        - phases[transition[0]].statistics[index]);
            }
            for (int row = 0; row < GRID_ROWS; row++) {
                double before = 0;
                double after = 0;
                for (int column = 0; column < GRID_COLUMNS; column++) {
                    int index = row * GRID_COLUMNS + column;
                    before += phases[transition[0]].grid[index];
                    after += phases[transition[1]].grid[index];
                }
                output.add(after - before);
            }
        }
        return finiteArray(output, 155, "Flight");
    }

    private static double[] flowFromFrames(
            byte[] before, byte[] after, double pyrScale, int levels, int winsize,
            int iterations, int polyN, double polySigma, int flags
    ) {
        if (before.length != SIZE || after.length != SIZE) {
            throw new IllegalArgumentException("Serving-side grayscale frame has the wrong size");
        }
        Mat beforeMat = new Mat(HEIGHT, WIDTH, CvType.CV_8UC1);
        Mat afterMat = new Mat(HEIGHT, WIDTH, CvType.CV_8UC1);
        Mat flow = new Mat();
        try {
            beforeMat.put(0, 0, before);
            afterMat.put(0, 0, after);
            Video.calcOpticalFlowFarneback(
                    beforeMat, afterMat, flow, pyrScale, levels, winsize,
                    iterations, polyN, polySigma, flags
            );
            float[] values = new float[SIZE * 2];
            flow.get(0, 0, values);
            double[] result = new double[values.length];
            for (int i = 0; i < values.length; i++) result[i] = values[i];
            return result;
        } finally {
            beforeMat.release();
            afterMat.release();
            flow.release();
        }
    }

    private static byte[] opened3x3(byte[] active) {
        Mat source = new Mat(HEIGHT, WIDTH, CvType.CV_8UC1);
        Mat opened = new Mat();
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        try {
            byte[] binary = new byte[active.length];
            for (int i = 0; i < active.length; i++) binary[i] = active[i] == 0 ? 0 : (byte) 255;
            source.put(0, 0, binary);
            Imgproc.morphologyEx(source, opened, Imgproc.MORPH_OPEN, kernel);
            opened.get(0, 0, binary);
            for (int i = 0; i < binary.length; i++) binary[i] = binary[i] == 0 ? 0 : (byte) 1;
            return binary;
        } finally {
            source.release();
            opened.release();
            kernel.release();
        }
    }

    private static double[] courtZoneStatistics(
            double[] residual, double[] magnitude, byte[] opened, int y0, int y1
    ) {
        int height = y1 - y0;
        int size = WIDTH * height;
        byte[] localActive = new byte[size];
        double[] localMagnitudes = new double[size];
        int activePixels = 0;
        double flowX = 0;
        double flowY = 0;
        for (int y = y0; y < y1; y++) {
            for (int x = 0; x < WIDTH; x++) {
                int source = y * WIDTH + x;
                int target = (y - y0) * WIDTH + x;
                localMagnitudes[target] = magnitude[source];
                if (opened[source] != 0) {
                    localActive[target] = 1;
                    activePixels++;
                    flowX += residual[source * 2];
                    flowY += residual[source * 2 + 1];
                }
            }
        }
        Components components = connectedComponents(localActive, WIDTH, height);
        int largest = 0;
        double centroidX = 0;
        double centroidY = 0;
        for (int i = 0; i < components.areas.length; i++) {
            if (components.areas[i] > largest) {
                largest = components.areas[i];
                centroidX = components.centroidX[i] / Math.max(WIDTH - 1, 1);
                centroidY = components.centroidY[i] / Math.max(height - 1, 1);
            }
        }
        double total = 0;
        for (double value : localMagnitudes) total += value;
        double diagonal = Math.hypot(height, WIDTH);
        return new double[]{
                total / size / diagonal,
                quantile(localMagnitudes, .9) / diagonal,
                (double) activePixels / size,
                (double) largest / size,
                (double) components.areas.length / Math.max(size / 1000.0, 1),
                centroidX,
                centroidY,
                activePixels == 0 ? 0 : flowX / activePixels / WIDTH,
                activePixels == 0 ? 0 : flowY / activePixels / height,
        };
    }

    private static Motion extractFlightMotion(byte[] before, byte[] after) {
        double[] flow = flowFromFrames(before, after, .5, 3, 15, 3, 5, 1.1, 0);
        double[][] coefficients = affineCoefficients(flow, null);
        int stride = Math.max(1, Math.min(WIDTH, HEIGHT) / 24);
        ArrayList<Double> sampleResiduals = new ArrayList<>();
        for (int y = 0; y < HEIGHT; y += stride) {
            for (int x = 0; x < WIDTH; x += stride) {
                double designX = x / (double) Math.max(WIDTH - 1, 1);
                double designY = y / (double) Math.max(HEIGHT - 1, 1);
                int offset = (y * WIDTH + x) * 2;
                double predictedX = designX * coefficients[0][0]
                        + designY * coefficients[0][1] + coefficients[0][2];
                double predictedY = designX * coefficients[1][0]
                        + designY * coefficients[1][1] + coefficients[1][2];
                sampleResiduals.add(Math.hypot(
                        flow[offset] - predictedX, flow[offset + 1] - predictedY
                ));
            }
        }
        double cutoff = quantile(sampleResiduals.stream().mapToDouble(Double::doubleValue).toArray(), .75);
        byte[] retained = new byte[sampleResiduals.size()];
        int retainedCount = 0;
        for (int i = 0; i < retained.length; i++) {
            if (sampleResiduals.get(i) <= cutoff) {
                retained[i] = 1;
                retainedCount++;
            }
        }
        if (retainedCount >= 6) coefficients = affineCoefficients(flow, retained);
        double[] flowX = new double[SIZE];
        double[] flowY = new double[SIZE];
        double[] magnitude = new double[SIZE];
        double diagonal = Math.hypot(HEIGHT, WIDTH);
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                int index = y * WIDTH + x;
                double designX = x / (double) Math.max(WIDTH - 1, 1);
                double designY = y / (double) Math.max(HEIGHT - 1, 1);
                float predictedX = (float) (designX * coefficients[0][0]
                        + designY * coefficients[0][1] + coefficients[0][2]);
                float predictedY = (float) (designX * coefficients[1][0]
                        + designY * coefficients[1][1] + coefficients[1][2]);
                float residualX = (float) (flow[index * 2] - predictedX);
                float residualY = (float) (flow[index * 2 + 1] - predictedY);
                flowX[index] = (float) (residualX / WIDTH);
                flowY[index] = (float) (residualY / HEIGHT);
                magnitude[index] = (float) (Math.hypot(residualX, residualY) / diagonal);
            }
        }
        double threshold = Math.max(7.5e-4, quantile(magnitude, .9));
        double[] energy = new double[SIZE];
        for (int i = 0; i < SIZE; i++) energy[i] = (float) Math.max(magnitude[i] - threshold, 0);
        double[] divergence = new double[SIZE];
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                int index = y * WIDTH + x;
                double dx = x == 0 ? flowX[index + 1] - flowX[index]
                        : x == WIDTH - 1 ? flowX[index] - flowX[index - 1]
                        : (flowX[index + 1] - flowX[index - 1]) / 2;
                double dy = y == 0 ? flowY[index + WIDTH] - flowY[index]
                        : y == HEIGHT - 1 ? flowY[index] - flowY[index - WIDTH]
                        : (flowY[index + WIDTH] - flowY[index - WIDTH]) / 2;
                divergence[index] = (float) (dx * WIDTH + dy * HEIGHT);
            }
        }
        return new Motion(energy, flowX, flowY, divergence);
    }

    private static PairSummary summarizeFlightPair(Motion motion) {
        double total = 0;
        for (double value : motion.energy) total += value;
        double centroidX = .5;
        double centroidY = .5;
        double spreadX = 0;
        double spreadY = 0;
        double entropy = 0;
        if (total > 0) {
            centroidX = 0;
            centroidY = 0;
            for (int y = 0; y < HEIGHT; y++) {
                for (int x = 0; x < WIDTH; x++) {
                    double energy = motion.energy[y * WIDTH + x];
                    centroidX += x / (double) Math.max(WIDTH - 1, 1) * energy;
                    centroidY += y / (double) Math.max(HEIGHT - 1, 1) * energy;
                }
            }
            centroidX /= total;
            centroidY /= total;
            double varianceX = 0;
            double varianceY = 0;
            for (int y = 0; y < HEIGHT; y++) {
                for (int x = 0; x < WIDTH; x++) {
                    double energy = motion.energy[y * WIDTH + x];
                    varianceX += Math.pow(x / (double) Math.max(WIDTH - 1, 1) - centroidX, 2) * energy;
                    varianceY += Math.pow(y / (double) Math.max(HEIGHT - 1, 1) - centroidY, 2) * energy;
                    if (energy > 0) {
                        double probability = energy / total;
                        entropy -= probability * Math.log(probability);
                    }
                }
            }
            spreadX = Math.sqrt(Math.max(0, varianceX / total));
            spreadY = Math.sqrt(Math.max(0, varianceY / total));
            entropy /= Math.max(Math.log(SIZE), 1);
        }
        double[] grid = new double[GRID_ROWS * GRID_COLUMNS];
        double[] rows = new double[GRID_ROWS];
        for (int row = 0; row < GRID_ROWS; row++) {
            int y0 = row * HEIGHT / GRID_ROWS;
            int y1 = (row + 1) * HEIGHT / GRID_ROWS;
            double rowTotal = 0;
            double rowFlow = 0;
            for (int column = 0; column < GRID_COLUMNS; column++) {
                int x0 = column * WIDTH / GRID_COLUMNS;
                int x1 = (column + 1) * WIDTH / GRID_COLUMNS;
                double cellTotal = 0;
                for (int y = y0; y < y1; y++) {
                    for (int x = x0; x < x1; x++) cellTotal += motion.energy[y * WIDTH + x];
                }
                grid[row * GRID_COLUMNS + column] = total > 0 ? cellTotal / total : 0;
                rowTotal += cellTotal;
            }
            if (rowTotal > 0) {
                for (int y = y0; y < y1; y++) {
                    for (int x = 0; x < WIDTH; x++) {
                        int index = y * WIDTH + x;
                        rowFlow += motion.flowY[index] * motion.energy[index];
                    }
                }
                rows[row] = rowFlow / rowTotal;
            }
        }
        byte[] active = new byte[SIZE];
        for (int i = 0; i < SIZE; i++) active[i] = motion.energy[i] > 0 ? (byte) 1 : 0;
        Components components = connectedComponents(active, WIDTH, HEIGHT);
        int largestArea = 0;
        for (int area : components.areas) largestArea = Math.max(largestArea, area);
        double largestComponent = largestArea / (double) SIZE;
        double smallThreshold = Math.max(4, SIZE * .0025);
        Set<Integer> smallLabels = new HashSet<>();
        for (int label = 0; label < components.areas.length; label++) {
            if (components.areas[label] <= smallThreshold) smallLabels.add(label);
        }
        int activeCount = 0;
        double bottom = 0;
        double top = 0;
        double smallTotal = 0;
        double smallY = 0;
        double smallFlowY = 0;
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                int index = y * WIDTH + x;
                double energy = motion.energy[index];
                if (energy > 0) activeCount++;
                if (y >= HEIGHT / 2) bottom += energy; else top += energy;
                if (energy > 0 && smallLabels.contains(components.labels[index])) {
                    smallTotal += energy;
                    smallY += y / (double) Math.max(HEIGHT - 1, 1) * energy;
                    smallFlowY += motion.flowY[index] * energy;
                }
            }
        }
        double[] statistics = {
                total / SIZE,
                activeCount / (double) SIZE,
                centroidX, centroidY, spreadX, spreadY, entropy, largestComponent,
                weightedMean(motion.flowX, motion.energy, total),
                weightedMean(motion.flowY, motion.energy, total),
                weightedMean(motion.divergence, motion.energy, total),
                total > 0 ? (bottom - top) / total : 0,
                total > 0 ? smallTotal / total : 0,
                smallTotal > 0 ? smallY / smallTotal : 0,
                smallTotal > 0 ? smallFlowY / smallTotal : 0,
        };
        return new PairSummary(grid, rows, statistics);
    }

    private static double[][] affineCoefficients(double[] flow, byte[] retained) {
        int stride = Math.max(1, Math.min(WIDTH, HEIGHT) / 24);
        double[] matrix = new double[9];
        double[] targetX = new double[3];
        double[] targetY = new double[3];
        int sampleIndex = 0;
        for (int y = 0; y < HEIGHT; y += stride) {
            for (int x = 0; x < WIDTH; x += stride) {
                boolean include = retained == null || retained[sampleIndex] != 0;
                sampleIndex++;
                if (!include) continue;
                double[] design = {
                        x / (double) Math.max(WIDTH - 1, 1),
                        y / (double) Math.max(HEIGHT - 1, 1), 1,
                };
                int offset = (y * WIDTH + x) * 2;
                for (int row = 0; row < 3; row++) {
                    targetX[row] += design[row] * flow[offset];
                    targetY[row] += design[row] * flow[offset + 1];
                    for (int column = 0; column < 3; column++) {
                        matrix[row * 3 + column] += design[row] * design[column];
                    }
                }
            }
        }
        return new double[][]{solveThreeByThree(matrix, targetX), solveThreeByThree(matrix, targetY)};
    }

    private static double[] solveThreeByThree(double[] matrix, double[] vector) {
        double[][] work = {
                {matrix[0], matrix[1], matrix[2], vector[0]},
                {matrix[3], matrix[4], matrix[5], vector[1]},
                {matrix[6], matrix[7], matrix[8], vector[2]},
        };
        for (int column = 0; column < 3; column++) {
            int pivot = column;
            for (int row = column + 1; row < 3; row++) {
                if (Math.abs(work[row][column]) > Math.abs(work[pivot][column])) pivot = row;
            }
            double[] swap = work[column];
            work[column] = work[pivot];
            work[pivot] = swap;
            if (Math.abs(work[column][column]) < 1e-12) return new double[3];
            for (int row = column + 1; row < 3; row++) {
                double factor = work[row][column] / work[column][column];
                for (int index = column; index < 4; index++) {
                    work[row][index] -= factor * work[column][index];
                }
            }
        }
        double[] result = new double[3];
        for (int row = 2; row >= 0; row--) {
            double value = work[row][3];
            for (int column = row + 1; column < 3; column++) value -= work[row][column] * result[column];
            result[row] = value / work[row][row];
        }
        return result;
    }

    private static Components connectedComponents(byte[] active, int width, int height) {
        int[] labels = new int[active.length];
        Arrays.fill(labels, -1);
        int[] queue = new int[active.length];
        ArrayList<Integer> areas = new ArrayList<>();
        ArrayList<Double> centroidX = new ArrayList<>();
        ArrayList<Double> centroidY = new ArrayList<>();
        for (int origin = 0; origin < active.length; origin++) {
            if (active[origin] == 0 || labels[origin] != -1) continue;
            int label = areas.size();
            int head = 0;
            int tail = 0;
            long sumX = 0;
            long sumY = 0;
            queue[tail++] = origin;
            labels[origin] = label;
            while (head < tail) {
                int index = queue[head++];
                int y = index / width;
                int x = index - y * width;
                sumX += x;
                sumY += y;
                for (int dy = -1; dy <= 1; dy++) {
                    int nextY = y + dy;
                    if (nextY < 0 || nextY >= height) continue;
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dx == 0 && dy == 0) continue;
                        int nextX = x + dx;
                        if (nextX < 0 || nextX >= width) continue;
                        int next = nextY * width + nextX;
                        if (active[next] != 0 && labels[next] == -1) {
                            labels[next] = label;
                            queue[tail++] = next;
                        }
                    }
                }
            }
            areas.add(tail);
            centroidX.add(sumX / (double) tail);
            centroidY.add(sumY / (double) tail);
        }
        return new Components(
                labels,
                areas.stream().mapToInt(Integer::intValue).toArray(),
                centroidX.stream().mapToDouble(Double::doubleValue).toArray(),
                centroidY.stream().mapToDouble(Double::doubleValue).toArray()
        );
    }

    private static double weightedMean(double[] values, double[] weights, double total) {
        if (total <= 0) return 0;
        double sum = 0;
        for (int i = 0; i < values.length; i++) sum += values[i] * weights[i];
        return sum / total;
    }

    private static double quantile(double[] values, double probability) {
        if (values.length == 0) return 0;
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        double position = (sorted.length - 1) * Math.max(0, Math.min(1, probability));
        int lower = (int) Math.floor(position);
        int upper = (int) Math.ceil(position);
        double fraction = position - lower;
        return sorted[lower] * (1 - fraction) + sorted[upper] * fraction;
    }

    private static double[] meanVectors(List<double[]> values) {
        if (values.isEmpty()) return new double[0];
        double[] result = new double[values.get(0).length];
        for (double[] value : values) {
            for (int i = 0; i < result.length; i++) result[i] += value[i];
        }
        for (int i = 0; i < result.length; i++) result[i] /= values.size();
        return result;
    }

    private static void append(List<Double> target, double[] values) {
        for (double value : values) target.add(value);
    }

    private static double[] finiteArray(List<Double> values, int expected, String label) {
        if (values.size() != expected) throw new IllegalStateException(
                label + " feature count changed: " + values.size()
        );
        double[] result = new double[values.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = values.get(i);
            if (!Double.isFinite(result[i])) throw new IllegalStateException(
                    label + " feature became non-finite at " + i
            );
        }
        return result;
    }
}
