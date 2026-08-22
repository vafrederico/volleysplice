import type { Mat } from "@techstark/opencv-js";
import { CanvasSink } from "mediabunny";

import { runtimeAssetUrl } from "../runtime-assets.ts";
import type { OpenedMedia } from "./media.ts";
import {
  composeServingSideVerdict,
  COURT_FLOW_OFFSETS_SECONDS,
  COURT_FLOW_STATISTICS,
  courtFlowFeatureNames,
  FLIGHT_GLOBAL_STATISTICS,
  FLIGHT_GRID_COLUMNS,
  FLIGHT_GRID_ROWS,
  FLIGHT_OFFSETS_SECONDS,
  FLIGHT_TRAJECTORY_STATISTICS,
  flightFeatureNames,
  parseServingSideRuntime,
  SERVING_SIDE_ANCHOR_CONTRACT,
  SERVING_SIDE_ASSET,
  SERVING_SIDE_FEATURE_VERSION,
  SERVING_SIDE_HEIGHT,
  SERVING_SIDE_MODEL_FINGERPRINT,
  SERVING_SIDE_MODEL_ID,
  SERVING_SIDE_WIDTH,
  servingSideFeatureNames,
  tiedPercentileRanks,
  type ServingSideRuntimeModel,
} from "./serving-side-model.ts";
import type {
  NormalizedRoi,
  OnDeviceInterval,
  OnDeviceServingSideOutput,
  ProductionServeOutputs,
} from "./types.ts";
import { loadOpenCv } from "./visual-features.ts";

type CvRuntime = typeof import("@techstark/opencv-js");

export type ServingSideAnalysisInput = {
  intervals?: readonly OnDeviceInterval[];
  rallies?: readonly OnDeviceInterval[];
  times?: ArrayLike<number>;
  inferenceTimes?: ArrayLike<number>;
  productionServeOutputs?: ProductionServeOutputs;
};

export type ServingSideProgress = {
  stage: "loading" | "frames" | "features" | "complete";
  completed: number;
  total: number;
  detail: string;
};

type GrayFrame = Uint8Array;

type Motion = {
  energy: Float64Array;
  flowX: Float64Array;
  flowY: Float64Array;
  divergence: Float64Array;
};

let runtimePromise: Promise<ServingSideRuntimeModel> | null = null;

export function loadServingSideRuntime(): Promise<ServingSideRuntimeModel> {
  runtimePromise ??= fetch(runtimeAssetUrl(SERVING_SIDE_ASSET))
    .then(async (response) => {
      if (!response.ok) throw new Error(`Could not load serving-side model (${response.status}).`);
      return parseServingSideRuntime(await response.json());
    });
  return runtimePromise;
}

function quantile(values: ArrayLike<number>, probability: number): number {
  if (values.length === 0) return 0;
  const sorted = Array.from(values).sort((left, right) => left - right);
  const position = (sorted.length - 1) * Math.max(0, Math.min(1, probability));
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const fraction = position - lower;
  return sorted[lower] * (1 - fraction) + sorted[upper] * fraction;
}

function meanVectors(values: readonly Float64Array[]): Float64Array {
  if (values.length === 0) return new Float64Array(0);
  const result = new Float64Array(values[0].length);
  for (const value of values) {
    for (let index = 0; index < result.length; index += 1) result[index] += value[index];
  }
  for (let index = 0; index < result.length; index += 1) result[index] /= values.length;
  return result;
}

function flowFromFrames(
  cv: CvRuntime,
  before: GrayFrame,
  after: GrayFrame,
  parameters: readonly [number, number, number, number, number, number, number],
): Float64Array {
  const beforeMat = cv.matFromArray(
    SERVING_SIDE_HEIGHT, SERVING_SIDE_WIDTH, cv.CV_8UC1, before,
  );
  const afterMat = cv.matFromArray(
    SERVING_SIDE_HEIGHT, SERVING_SIDE_WIDTH, cv.CV_8UC1, after,
  );
  const flow = new cv.Mat();
  try {
    cv.calcOpticalFlowFarneback(beforeMat, afterMat, flow, ...parameters);
    return Float64Array.from(flow.data32F);
  } finally {
    beforeMat.delete();
    afterMat.delete();
    flow.delete();
  }
}

function opened3x3WithOpenCv(
  cv: CvRuntime,
  active: Uint8Array,
  width: number,
  height: number,
): Uint8Array {
  const source = cv.matFromArray(height, width, cv.CV_8UC1, active);
  const opened = new cv.Mat();
  const kernel = cv.Mat.ones(3, 3, cv.CV_8U);
  try {
    cv.morphologyEx(source, opened, cv.MORPH_OPEN, kernel);
    return Uint8Array.from(opened.data, (value) => value ? 1 : 0);
  } finally {
    source.delete();
    opened.delete();
    kernel.delete();
  }
}

type Components = {
  labels: Int32Array;
  areas: number[];
  centroidX: number[];
  centroidY: number[];
};

function connectedComponents(active: Uint8Array, width: number, height: number): Components {
  const labels = new Int32Array(active.length);
  labels.fill(-1);
  const queue = new Int32Array(active.length);
  const areas: number[] = [];
  const centroidX: number[] = [];
  const centroidY: number[] = [];
  for (let origin = 0; origin < active.length; origin += 1) {
    if (!active[origin] || labels[origin] !== -1) continue;
    const label = areas.length;
    let head = 0;
    let tail = 0;
    let sumX = 0;
    let sumY = 0;
    queue[tail++] = origin;
    labels[origin] = label;
    while (head < tail) {
      const index = queue[head++];
      const y = Math.floor(index / width);
      const x = index - y * width;
      sumX += x;
      sumY += y;
      for (let dy = -1; dy <= 1; dy += 1) {
        const nextY = y + dy;
        if (nextY < 0 || nextY >= height) continue;
        for (let dx = -1; dx <= 1; dx += 1) {
          if (dx === 0 && dy === 0) continue;
          const nextX = x + dx;
          if (nextX < 0 || nextX >= width) continue;
          const next = nextY * width + nextX;
          if (active[next] && labels[next] === -1) {
            labels[next] = label;
            queue[tail++] = next;
          }
        }
      }
    }
    areas.push(tail);
    centroidX.push(sumX / tail);
    centroidY.push(sumY / tail);
  }
  return { labels, areas, centroidX, centroidY };
}

function courtZoneStatistics(
  residual: Float64Array,
  magnitude: Float64Array,
  opened: Uint8Array,
  y0: number,
  y1: number,
): Float64Array {
  const width = SERVING_SIDE_WIDTH;
  const height = y1 - y0;
  const size = width * height;
  const localActive = new Uint8Array(size);
  const localMagnitudes = new Float64Array(size);
  let activePixels = 0;
  let flowX = 0;
  let flowY = 0;
  for (let y = y0; y < y1; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const source = y * width + x;
      const target = (y - y0) * width + x;
      localMagnitudes[target] = magnitude[source];
      if (opened[source]) {
        localActive[target] = 1;
        activePixels += 1;
        flowX += residual[source * 2];
        flowY += residual[source * 2 + 1];
      }
    }
  }
  const components = connectedComponents(localActive, width, height);
  let largest = 0;
  let centroidX = 0;
  let centroidY = 0;
  for (let index = 0; index < components.areas.length; index += 1) {
    if (components.areas[index] > largest) {
      largest = components.areas[index];
      centroidX = components.centroidX[index] / Math.max(width - 1, 1);
      centroidY = components.centroidY[index] / Math.max(height - 1, 1);
    }
  }
  let magnitudeTotal = 0;
  for (const value of localMagnitudes) magnitudeTotal += value;
  const diagonal = Math.hypot(height, width);
  return Float64Array.of(
    magnitudeTotal / size / diagonal,
    quantile(localMagnitudes, 0.9) / diagonal,
    activePixels / size,
    largest / size,
    components.areas.length / Math.max(size / 1000, 1),
    centroidX,
    centroidY,
    activePixels ? flowX / activePixels / width : 0,
    activePixels ? flowY / activePixels / height : 0,
  );
}

export function extractCourtFlowRawFeatures(
  cv: CvRuntime,
  frames: readonly GrayFrame[],
): Float64Array {
  if (frames.length !== COURT_FLOW_OFFSETS_SECONDS.length) {
    throw new Error("Court-flow extraction requires eight ordered frames.");
  }
  const pairs = {
    pre: [[0, 1], [1, 2]],
    contact: [[2, 3], [3, 4], [4, 5]],
    post: [[5, 6], [6, 7]],
  } as const;
  const band = Math.max(1, Math.round(SERVING_SIDE_HEIGHT * 0.32));
  const phases = new Map<string, { near: Float64Array; far: Float64Array }>();
  for (const [phase, indices] of Object.entries(pairs)) {
    const nearValues: Float64Array[] = [];
    const farValues: Float64Array[] = [];
    for (const [before, after] of indices) {
      const flow = flowFromFrames(cv, frames[before], frames[after], [0.5, 2, 13, 2, 5, 1.1, 0]);
      const xValues = new Float64Array(flow.length / 2);
      const yValues = new Float64Array(flow.length / 2);
      for (let index = 0; index < xValues.length; index += 1) {
        xValues[index] = flow[index * 2];
        yValues[index] = flow[index * 2 + 1];
      }
      const medianX = Math.fround(quantile(xValues, 0.5));
      const medianY = Math.fround(quantile(yValues, 0.5));
      const residual = new Float64Array(flow.length);
      const magnitude = new Float64Array(flow.length / 2);
      const active = new Uint8Array(magnitude.length);
      for (let index = 0; index < magnitude.length; index += 1) {
        const x = Math.fround(flow[index * 2] - medianX);
        const y = Math.fround(flow[index * 2 + 1] - medianY);
        residual[index * 2] = x;
        residual[index * 2 + 1] = y;
        magnitude[index] = Math.fround(Math.hypot(x, y));
        active[index] = magnitude[index] >= 1 ? 1 : 0;
      }
      const opened = opened3x3WithOpenCv(
        cv, active, SERVING_SIDE_WIDTH, SERVING_SIDE_HEIGHT,
      );
      nearValues.push(courtZoneStatistics(
        residual, magnitude, opened, SERVING_SIDE_HEIGHT - band, SERVING_SIDE_HEIGHT,
      ));
      farValues.push(courtZoneStatistics(residual, magnitude, opened, 0, band));
    }
    phases.set(phase, { near: meanVectors(nearValues), far: meanVectors(farValues) });
  }
  const values: number[] = [];
  for (const phase of ["pre", "contact", "post"] as const) {
    const current = phases.get(phase)!;
    values.push(...current.near, ...current.far);
    for (let index = 0; index < 5; index += 1) values.push(current.near[index] - current.far[index]);
  }
  for (const zone of ["near", "far"] as const) {
    const pre = phases.get("pre")![zone];
    const contact = phases.get("contact")![zone];
    const post = phases.get("post")![zone];
    values.push(
      contact[0] - pre[0],
      contact[2] - pre[2],
      contact[3] - pre[3],
      post[0] - contact[0],
      post[2] - contact[2],
    );
  }
  for (const index of [0, 2, 3]) {
    values.push(
      (phases.get("contact")!.near[index] - phases.get("pre")!.near[index]) -
      (phases.get("contact")!.far[index] - phases.get("pre")!.far[index]),
    );
  }
  if (values.length !== courtFlowFeatureNames().length || values.some((value) => !Number.isFinite(value))) {
    throw new Error("Court-flow feature contract changed or became non-finite.");
  }
  return Float64Array.from(values);
}

function solveThreeByThree(matrix: Float64Array, vector: Float64Array): Float64Array {
  const work = [
    [matrix[0], matrix[1], matrix[2], vector[0]],
    [matrix[3], matrix[4], matrix[5], vector[1]],
    [matrix[6], matrix[7], matrix[8], vector[2]],
  ];
  for (let column = 0; column < 3; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < 3; row += 1) {
      if (Math.abs(work[row][column]) > Math.abs(work[pivot][column])) pivot = row;
    }
    [work[column], work[pivot]] = [work[pivot], work[column]];
    if (Math.abs(work[column][column]) < 1e-12) return new Float64Array(3);
    for (let row = column + 1; row < 3; row += 1) {
      const factor = work[row][column] / work[column][column];
      for (let index = column; index < 4; index += 1) {
        work[row][index] -= factor * work[column][index];
      }
    }
  }
  const result = new Float64Array(3);
  for (let row = 2; row >= 0; row -= 1) {
    let value = work[row][3];
    for (let column = row + 1; column < 3; column += 1) value -= work[row][column] * result[column];
    result[row] = value / work[row][row];
  }
  return result;
}

function affineCoefficients(flow: Float64Array, retained?: Uint8Array): [Float64Array, Float64Array] {
  const width = SERVING_SIDE_WIDTH;
  const height = SERVING_SIDE_HEIGHT;
  const stride = Math.max(1, Math.floor(Math.min(width, height) / 24));
  const matrix = new Float64Array(9);
  const targetX = new Float64Array(3);
  const targetY = new Float64Array(3);
  let sampleIndex = 0;
  for (let y = 0; y < height; y += stride) {
    for (let x = 0; x < width; x += stride) {
      const include = !retained || retained[sampleIndex];
      sampleIndex += 1;
      if (!include) continue;
      const design = [x / Math.max(width - 1, 1), y / Math.max(height - 1, 1), 1];
      const offset = (y * width + x) * 2;
      for (let row = 0; row < 3; row += 1) {
        targetX[row] += design[row] * flow[offset];
        targetY[row] += design[row] * flow[offset + 1];
        for (let column = 0; column < 3; column += 1) {
          matrix[row * 3 + column] += design[row] * design[column];
        }
      }
    }
  }
  return [solveThreeByThree(matrix, targetX), solveThreeByThree(matrix, targetY)];
}

function extractFlightMotion(cv: CvRuntime, before: GrayFrame, after: GrayFrame): Motion {
  const width = SERVING_SIDE_WIDTH;
  const height = SERVING_SIDE_HEIGHT;
  const flow = flowFromFrames(cv, before, after, [0.5, 3, 15, 3, 5, 1.1, 0]);
  let coefficients = affineCoefficients(flow);
  const stride = Math.max(1, Math.floor(Math.min(width, height) / 24));
  const sampleResiduals: number[] = [];
  for (let y = 0; y < height; y += stride) {
    for (let x = 0; x < width; x += stride) {
      const designX = x / Math.max(width - 1, 1);
      const designY = y / Math.max(height - 1, 1);
      const offset = (y * width + x) * 2;
      const predictedX = designX * coefficients[0][0] + designY * coefficients[0][1] + coefficients[0][2];
      const predictedY = designX * coefficients[1][0] + designY * coefficients[1][1] + coefficients[1][2];
      sampleResiduals.push(Math.hypot(flow[offset] - predictedX, flow[offset + 1] - predictedY));
    }
  }
  const cutoff = quantile(sampleResiduals, 0.75);
  const retained = Uint8Array.from(sampleResiduals, (value) => value <= cutoff ? 1 : 0);
  if (retained.reduce((sum, value) => sum + value, 0) >= 6) coefficients = affineCoefficients(flow, retained);
  const size = width * height;
  const flowX = new Float64Array(size);
  const flowY = new Float64Array(size);
  const magnitude = new Float64Array(size);
  const diagonal = Math.hypot(height, width);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      const designX = x / Math.max(width - 1, 1);
      const designY = y / Math.max(height - 1, 1);
      const predictedX = Math.fround(
        designX * coefficients[0][0] + designY * coefficients[0][1] + coefficients[0][2],
      );
      const predictedY = Math.fround(
        designX * coefficients[1][0] + designY * coefficients[1][1] + coefficients[1][2],
      );
      const residualX = Math.fround(flow[index * 2] - predictedX);
      const residualY = Math.fround(flow[index * 2 + 1] - predictedY);
      flowX[index] = Math.fround(residualX / width);
      flowY[index] = Math.fround(residualY / height);
      magnitude[index] = Math.fround(Math.hypot(residualX, residualY) / diagonal);
    }
  }
  const threshold = Math.max(7.5e-4, quantile(magnitude, 0.9));
  const energy = Float64Array.from(
    magnitude,
    (value) => Math.fround(Math.max(value - threshold, 0)),
  );
  const divergence = new Float64Array(size);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      const dx = x === 0
        ? flowX[index + 1] - flowX[index]
        : x === width - 1
          ? flowX[index] - flowX[index - 1]
          : (flowX[index + 1] - flowX[index - 1]) / 2;
      const dy = y === 0
        ? flowY[index + width] - flowY[index]
        : y === height - 1
          ? flowY[index] - flowY[index - width]
          : (flowY[index + width] - flowY[index - width]) / 2;
      divergence[index] = Math.fround(dx * width + dy * height);
    }
  }
  return { energy, flowX, flowY, divergence };
}

function weightedMean(values: ArrayLike<number>, weights: ArrayLike<number>, total: number): number {
  if (total <= 0) return 0;
  let sum = 0;
  for (let index = 0; index < values.length; index += 1) sum += values[index] * weights[index];
  return sum / total;
}

function summarizeFlightPair(motion: Motion): [Float64Array, Float64Array, Float64Array] {
  const width = SERVING_SIDE_WIDTH;
  const height = SERVING_SIDE_HEIGHT;
  const size = width * height;
  let total = 0;
  for (const value of motion.energy) total += value;
  let centroidX = 0.5;
  let centroidY = 0.5;
  let spreadX = 0;
  let spreadY = 0;
  let entropy = 0;
  if (total > 0) {
    centroidX = 0;
    centroidY = 0;
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const energy = motion.energy[y * width + x];
        centroidX += x / Math.max(width - 1, 1) * energy;
        centroidY += y / Math.max(height - 1, 1) * energy;
      }
    }
    centroidX /= total;
    centroidY /= total;
    let varianceX = 0;
    let varianceY = 0;
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const energy = motion.energy[y * width + x];
        varianceX += (x / Math.max(width - 1, 1) - centroidX) ** 2 * energy;
        varianceY += (y / Math.max(height - 1, 1) - centroidY) ** 2 * energy;
        if (energy > 0) {
          const probability = energy / total;
          entropy -= probability * Math.log(probability);
        }
      }
    }
    spreadX = Math.sqrt(Math.max(0, varianceX / total));
    spreadY = Math.sqrt(Math.max(0, varianceY / total));
    entropy /= Math.max(Math.log(size), 1);
  }
  const grid = new Float64Array(FLIGHT_GRID_ROWS * FLIGHT_GRID_COLUMNS);
  const rows = new Float64Array(FLIGHT_GRID_ROWS);
  for (let row = 0; row < FLIGHT_GRID_ROWS; row += 1) {
    const y0 = Math.floor(row * height / FLIGHT_GRID_ROWS);
    const y1 = Math.floor((row + 1) * height / FLIGHT_GRID_ROWS);
    let rowTotal = 0;
    let rowFlow = 0;
    for (let column = 0; column < FLIGHT_GRID_COLUMNS; column += 1) {
      const x0 = Math.floor(column * width / FLIGHT_GRID_COLUMNS);
      const x1 = Math.floor((column + 1) * width / FLIGHT_GRID_COLUMNS);
      let cellTotal = 0;
      for (let y = y0; y < y1; y += 1) {
        for (let x = x0; x < x1; x += 1) cellTotal += motion.energy[y * width + x];
      }
      grid[row * FLIGHT_GRID_COLUMNS + column] = total > 0 ? cellTotal / total : 0;
      rowTotal += cellTotal;
    }
    if (rowTotal > 0) {
      for (let y = y0; y < y1; y += 1) {
        for (let x = 0; x < width; x += 1) {
          const index = y * width + x;
          rowFlow += motion.flowY[index] * motion.energy[index];
        }
      }
      rows[row] = rowFlow / rowTotal;
    }
  }
  const active = Uint8Array.from(motion.energy, (value) => value > 0 ? 1 : 0);
  const components = connectedComponents(active, width, height);
  const largestComponent = components.areas.length ? Math.max(...components.areas) / size : 0;
  const smallThreshold = Math.max(4, size * 0.0025);
  const smallLabels = new Set<number>();
  components.areas.forEach((area, label) => {
    if (area <= smallThreshold) smallLabels.add(label);
  });
  let activeCount = 0;
  let bottom = 0;
  let top = 0;
  let smallTotal = 0;
  let smallY = 0;
  let smallFlowY = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      const energy = motion.energy[index];
      if (energy > 0) activeCount += 1;
      if (y >= Math.floor(height / 2)) bottom += energy;
      else top += energy;
      if (energy > 0 && smallLabels.has(components.labels[index])) {
        smallTotal += energy;
        smallY += y / Math.max(height - 1, 1) * energy;
        smallFlowY += motion.flowY[index] * energy;
      }
    }
  }
  const statistics = Float64Array.of(
    total / size,
    activeCount / size,
    centroidX,
    centroidY,
    spreadX,
    spreadY,
    entropy,
    largestComponent,
    weightedMean(motion.flowX, motion.energy, total),
    weightedMean(motion.flowY, motion.energy, total),
    weightedMean(motion.divergence, motion.energy, total),
    total > 0 ? (bottom - top) / total : 0,
    total > 0 ? smallTotal / total : 0,
    smallTotal > 0 ? smallY / smallTotal : 0,
    smallTotal > 0 ? smallFlowY / smallTotal : 0,
  );
  return [grid, rows, statistics];
}

export function extractFlightRawFeatures(
  cv: CvRuntime,
  frames: readonly GrayFrame[],
): Float64Array {
  if (frames.length !== FLIGHT_OFFSETS_SECONDS.length) {
    throw new Error("Flight extraction requires nine ordered frames.");
  }
  const motions = frames.slice(0, -1).map((frame, index) =>
    extractFlightMotion(cv, frame, frames[index + 1]));
  const pairValues = motions.map(summarizeFlightPair);
  const phaseIndices = {
    launch: [0, 1, 2],
    early: [3, 4, 5],
    late: [6, 7],
  } as const;
  const phases = new Map<string, [Float64Array, Float64Array, Float64Array]>();
  const output: number[] = [];
  for (const [phase, indices] of Object.entries(phaseIndices)) {
    const grid = meanVectors(indices.map((index) => pairValues[index][0]));
    const rows = meanVectors(indices.map((index) => pairValues[index][1]));
    const statistics = meanVectors(indices.map((index) => pairValues[index][2]));
    phases.set(phase, [grid, rows, statistics]);
    output.push(...grid, ...rows, ...statistics);
  }
  const statisticIndices = new Map(
    FLIGHT_GLOBAL_STATISTICS.map((name, index) => [name, index]),
  );
  for (const [before, after] of [["launch", "early"], ["early", "late"]] as const) {
    for (const name of FLIGHT_TRAJECTORY_STATISTICS) {
      const index = statisticIndices.get(name)!;
      output.push(phases.get(after)![2][index] - phases.get(before)![2][index]);
    }
    for (let row = 0; row < FLIGHT_GRID_ROWS; row += 1) {
      let beforeEnergy = 0;
      let afterEnergy = 0;
      for (let column = 0; column < FLIGHT_GRID_COLUMNS; column += 1) {
        const index = row * FLIGHT_GRID_COLUMNS + column;
        beforeEnergy += phases.get(before)![0][index];
        afterEnergy += phases.get(after)![0][index];
      }
      output.push(afterEnergy - beforeEnergy);
    }
  }
  if (output.length !== flightFeatureNames().length || output.some((value) => !Number.isFinite(value))) {
    throw new Error("Flight feature contract changed or became non-finite.");
  }
  return Float64Array.from(output);
}

function candidateIntervals(input: ServingSideAnalysisInput): OnDeviceInterval[] {
  const source = input.intervals ?? input.rallies ?? [];
  const ids = new Set<string>();
  const candidates = source
    .filter((interval) => interval.included && Number.isFinite(interval.start) &&
      Number.isFinite(interval.end) && interval.start >= 0 && interval.end > interval.start)
    .map((interval) => ({ ...interval }))
    .sort((left, right) => left.start - right.start || left.end - right.end || left.id.localeCompare(right.id));
  for (const candidate of candidates) {
    if (!candidate.id || ids.has(candidate.id)) throw new Error("Serving-side candidates need stable unique IDs.");
    ids.add(candidate.id);
  }
  return candidates;
}

function clampedTimestamp(anchor: number, offset: number, duration: number): number {
  return Math.min(Math.max(0, anchor + offset), Math.max(0, duration - 0.01));
}

async function sampleGrayFrames(
  cv: CvRuntime,
  media: OpenedMedia,
  roi: NormalizedRoi,
  candidates: readonly OnDeviceInterval[],
  onProgress?: (progress: ServingSideProgress) => void,
): Promise<Map<number, GrayFrame>> {
  const requested = [...new Set(candidates.flatMap((candidate) =>
    [...COURT_FLOW_OFFSETS_SECONDS, ...FLIGHT_OFFSETS_SECONDS].map((offset) =>
      clampedTimestamp(candidate.start, offset, media.info.duration))))]
    .sort((left, right) => left - right);
  const left = Math.round(roi.x * media.info.width);
  const top = Math.round(roi.y * media.info.height);
  const right = Math.round((roi.x + roi.width) * media.info.width);
  const bottom = Math.round((roi.y + roi.height) * media.info.height);
  const sink = new CanvasSink(media.videoTrack, {
    crop: {
      left,
      top,
      width: Math.max(1, right - left),
      height: Math.max(1, bottom - top),
    },
    width: SERVING_SIDE_WIDTH,
    height: SERVING_SIDE_HEIGHT,
    fit: "fill",
    poolSize: 1,
    decoderOptions: { hardwareAcceleration: "prefer-hardware" },
  });
  const result = new Map<number, GrayFrame>();
  const iterator = sink.canvasesAtTimestamps(requested);
  let index = 0;
  try {
    for await (const wrapped of iterator) {
      if (!wrapped) throw new Error(`No video frame was available at ${requested[index]} seconds.`);
      const rgba = cv.imread(wrapped.canvas as unknown as HTMLCanvasElement);
      const gray = new cv.Mat();
      try {
        cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
        result.set(requested[index], Uint8Array.from(gray.data));
      } finally {
        rgba.delete();
        gray.delete();
      }
      index += 1;
      if (index % 8 === 0 || index === requested.length) {
        onProgress?.({
          stage: "frames",
          completed: index,
          total: requested.length,
          detail: `Sampling serving-side windows · ${index}/${requested.length} frames`,
        });
        await new Promise<void>((resolve) => setTimeout(resolve, 0));
      }
    }
  } finally {
    await iterator.return(undefined);
  }
  if (index !== requested.length) throw new Error("Serving-side frame sampling ended early.");
  return result;
}

function framesForOffsets(
  sampled: ReadonlyMap<number, GrayFrame>,
  anchor: number,
  offsets: readonly number[],
  duration: number,
): GrayFrame[] {
  return offsets.map((offset) => {
    const timestamp = clampedTimestamp(anchor, offset, duration);
    const frame = sampled.get(timestamp);
    if (!frame) throw new Error(`Missing sampled serving-side frame at ${timestamp} seconds.`);
    return frame;
  });
}

/**
 * Runs the frozen SERVSIDE237-FLIGHT model at each merged production interval start.
 * The complete candidate set is extracted before tied within-recording ranks are made.
 */
export async function inferServingSides(
  media: OpenedMedia,
  roi: NormalizedRoi,
  analysis: ServingSideAnalysisInput,
  onProgress?: (progress: ServingSideProgress) => void,
): Promise<OnDeviceServingSideOutput> {
  const candidates = candidateIntervals(analysis);
  const times = analysis.times ?? analysis.inferenceTimes;
  const serveOutputs = analysis.productionServeOutputs;
  if (!times || !serveOutputs) {
    throw new Error("Serving-side inference requires both production serve-head outputs.");
  }
  if (candidates.length === 0) {
    return {
      modelId: SERVING_SIDE_MODEL_ID,
      modelFingerprint: SERVING_SIDE_MODEL_FINGERPRINT,
      featureVersion: SERVING_SIDE_FEATURE_VERSION,
      anchorContract: SERVING_SIDE_ANCHOR_CONTRACT,
      features: {
        rows: 0,
        columns: servingSideFeatureNames().length,
        values: new Float64Array(0),
      },
      candidates: [],
    };
  }
  onProgress?.({ stage: "loading", completed: 0, total: candidates.length, detail: "Loading serving-side model" });
  const [cv, runtime] = await Promise.all([loadOpenCv(), loadServingSideRuntime()]);
  const sampled = await sampleGrayFrames(cv, media, roi, candidates, onProgress);
  const columns = runtime.featureNames.length;
  const raw = new Float64Array(candidates.length * columns);
  for (let row = 0; row < candidates.length; row += 1) {
    const candidate = candidates[row];
    const court = extractCourtFlowRawFeatures(
      cv,
      framesForOffsets(sampled, candidate.start, COURT_FLOW_OFFSETS_SECONDS, media.info.duration),
    );
    const flight = extractFlightRawFeatures(
      cv,
      framesForOffsets(sampled, candidate.start, FLIGHT_OFFSETS_SECONDS, media.info.duration),
    );
    raw.set(court, row * columns);
    raw.set(flight, row * columns + court.length);
    onProgress?.({
      stage: "features",
      completed: row + 1,
      total: candidates.length,
      detail: `Measuring serving side · ${row + 1}/${candidates.length} rallies`,
    });
    if ((row + 1) % 2 === 0) await new Promise<void>((resolve) => setTimeout(resolve, 0));
  }
  const ranked = tiedPercentileRanks(raw, candidates.length, columns);
  const verdicts = candidates.map((candidate, row) => composeServingSideVerdict(
    candidate,
    ranked.subarray(row * columns, (row + 1) * columns),
    times,
    serveOutputs.allLabelsV2,
    serveOutputs.previousProduction,
    runtime,
  ));
  const output: OnDeviceServingSideOutput = {
    modelId: runtime.modelId,
    modelFingerprint: runtime.fingerprint,
    featureVersion: SERVING_SIDE_FEATURE_VERSION,
    anchorContract: SERVING_SIDE_ANCHOR_CONTRACT,
    features: {
      rows: candidates.length,
      columns,
      values: raw,
    },
    candidates: verdicts,
  };
  onProgress?.({
    stage: "complete",
    completed: candidates.length,
    total: candidates.length,
    detail: `Serving-side verdicts ready · ${candidates.length} rallies`,
  });
  return output;
}
