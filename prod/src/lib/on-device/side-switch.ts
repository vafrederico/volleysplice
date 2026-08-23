import type { Mat } from "@techstark/opencv-js";
import { CanvasSink } from "mediabunny";

import type { OpenedMedia } from "./media.ts";
import {
  decodeSideSwitchCandidates,
  emptySideSwitchOutput,
  generateSideSwitchCandidates,
  loadSideSwitchRuntime,
  predictSideSwitchProbabilities,
  SIDE_SWITCH_CANDIDATE_CONTRACT,
  SIDE_SWITCH_FEATURE_NAMES,
  SIDE_SWITCH_FEATURE_VERSION,
  sideSwitchCalibrationTimes,
  sideSwitchCandidateSampleTimes,
  sideSwitchStateFeatures,
  type SideSwitchAnalysisInput,
  type SideSwitchCandidateProposal,
} from "./side-switch-model.ts";
import type { NormalizedRoi, OnDeviceSideSwitchOutput } from "./types.ts";
import { loadOpenCv, phaseCorrelate } from "./visual-features.ts";

type CvRuntime = typeof import("@techstark/opencv-js");
type ColorFrame = Uint8Array;

export type SideSwitchProgress = {
  stage: "loading" | "frames" | "features" | "complete";
  completed: number;
  total: number;
  detail: string;
};

const WIDTH = 256;
const HEIGHT = 144;
const CHANNELS = 3;
const CANONICAL_NET_Y = 0.5;

type CourtGeometry = { netYRatio: number; confidence: number };
type PaletteSummary = {
  near: Float64Array;
  far: Float64Array;
  instability: number;
};
type BroadSummary = {
  broad: PaletteSummary;
  tight: PaletteSummary;
  global: Float64Array;
  foregroundCoverage: number;
  maximumCameraShift: number;
  minimumAlignmentResponse: number;
};
type PlayerSummary = {
  near: Float64Array;
  far: Float64Array;
  global: Float64Array;
  instability: number;
  proposalCoverage: number;
  proposalCount: number;
  nearSupport: number;
  farSupport: number;
};

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function timestampKey(value: number): string {
  return value.toFixed(9);
}

function clampedTimestamp(value: number, duration: number): number {
  return Math.min(Math.max(0, value), Math.max(0, duration - 0.01));
}

function quantile(values: ArrayLike<number>, probability: number): number {
  if (values.length === 0) return 0;
  const sorted = Array.from(values).sort((left, right) => left - right);
  const position = (sorted.length - 1) * clamp(probability, 0, 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  return (
    sorted[lower] * (upper - position) + sorted[upper] * (position - lower || 1)
  );
}

function mean(values: ArrayLike<number>): number {
  if (values.length === 0) return 0;
  let total = 0;
  for (let index = 0; index < values.length; index += 1) total += values[index];
  return total / values.length;
}

function hellinger(left: ArrayLike<number>, right: ArrayLike<number>): number {
  let total = 0;
  for (let index = 0; index < left.length; index += 1) {
    total +=
      (Math.sqrt(Math.max(left[index], 0)) -
        Math.sqrt(Math.max(right[index], 0))) **
      2;
  }
  return Math.sqrt(total) / Math.SQRT2;
}

function cosine(left: ArrayLike<number>, right: ArrayLike<number>): number {
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let index = 0; index < left.length; index += 1) {
    dot += left[index] * right[index];
    leftNorm += left[index] ** 2;
    rightNorm += right[index] ** 2;
  }
  const denominator = Math.sqrt(leftNorm * rightNorm);
  return denominator > 1e-12 ? dot / denominator : 0;
}

function normalizedAverage(
  values: readonly Float64Array[],
  weights?: readonly number[],
): Float64Array {
  if (values.length === 0)
    return Float64Array.from({ length: 52 }, () => 1 / 52);
  const output = new Float64Array(values[0].length);
  let weightTotal = 0;
  for (let row = 0; row < values.length; row += 1) {
    const weight = weights?.[row] ?? 1;
    weightTotal += weight;
    for (let index = 0; index < output.length; index += 1)
      output[index] += values[row][index] * weight;
  }
  let total = 0;
  for (let index = 0; index < output.length; index += 1) {
    output[index] /= Math.max(weightTotal, 1e-12);
    total += output[index];
  }
  for (let index = 0; index < output.length; index += 1)
    output[index] /= Math.max(total, 1e-12);
  return output;
}

function weightedPalette(
  hsv: Uint8Array,
  weights: Float64Array,
  left: number,
  top: number,
  right: number,
  bottom: number,
  uniformWhenEmpty: boolean,
): Float64Array {
  const bins = new Float64Array(52);
  let weightTotal = 0;
  for (let y = top; y < bottom; y += 1) {
    for (let x = left; x < right; x += 1) {
      const pixel = y * WIDTH + x;
      const weight = weights[pixel];
      if (weight <= 0) continue;
      const offset = pixel * CHANNELS;
      const hue = Math.min(11, Math.floor(hsv[offset] / 15));
      const saturation = Math.min(3, Math.floor(hsv[offset + 1] / 64));
      const value = Math.min(3, Math.floor(hsv[offset + 2] / 64));
      bins[hue * 4 + saturation] += weight;
      bins[48 + value] += weight;
      weightTotal += weight * 2;
    }
  }
  if (weightTotal < 1e-8) {
    if (uniformWhenEmpty) bins.fill(1 / bins.length);
    else {
      const uniform = new Float64Array(weights.length);
      uniform.fill(1);
      return weightedPalette(hsv, uniform, left, top, right, bottom, true);
    }
    return bins;
  }
  for (let index = 0; index < bins.length; index += 1)
    bins[index] /= weightTotal;
  return bins;
}

function normalizeCourtFrame(
  cv: CvRuntime,
  frame: ColorFrame,
  geometry: CourtGeometry,
): ColorFrame {
  const source = cv.matFromArray(HEIGHT, WIDTH, cv.CV_8UC3, frame);
  const output = new cv.Mat();
  const mapXData = new Float32Array(WIDTH * HEIGHT);
  const mapYData = new Float32Array(WIDTH * HEIGHT);
  const net = clamp(geometry.netYRatio, 0.2, 0.8);
  for (let y = 0; y < HEIGHT; y += 1) {
    const outputY = y / Math.max(HEIGHT - 1, 1);
    const sourceY =
      outputY <= CANONICAL_NET_Y
        ? (outputY * net) / CANONICAL_NET_Y
        : net +
          ((outputY - CANONICAL_NET_Y) * (1 - net)) / (1 - CANONICAL_NET_Y);
    for (let x = 0; x < WIDTH; x += 1) {
      const index = y * WIDTH + x;
      mapXData[index] = x;
      mapYData[index] = sourceY * (HEIGHT - 1);
    }
  }
  const mapX = cv.matFromArray(HEIGHT, WIDTH, cv.CV_32FC1, mapXData);
  const mapY = cv.matFromArray(HEIGHT, WIDTH, cv.CV_32FC1, mapYData);
  try {
    cv.remap(source, output, mapX, mapY, cv.INTER_LINEAR, cv.BORDER_REPLICATE);
    return Uint8Array.from(output.data);
  } finally {
    source.delete();
    output.delete();
    mapX.delete();
    mapY.delete();
  }
}

function grayAndHsv(
  cv: CvRuntime,
  frame: ColorFrame,
): { gray: Uint8Array; hsv: Uint8Array } {
  const source = cv.matFromArray(HEIGHT, WIDTH, cv.CV_8UC3, frame);
  const gray = new cv.Mat();
  const hsv = new cv.Mat();
  try {
    cv.cvtColor(source, gray, cv.COLOR_BGR2GRAY);
    cv.cvtColor(source, hsv, cv.COLOR_BGR2HSV);
    return { gray: Uint8Array.from(gray.data), hsv: Uint8Array.from(hsv.data) };
  } finally {
    source.delete();
    gray.delete();
    hsv.delete();
  }
}

function hanningWindowed(cv: CvRuntime, source: Mat): Mat {
  const values = new Float32Array(source.rows * source.cols);
  for (let y = 0; y < source.rows; y += 1) {
    const vertical =
      source.rows <= 1
        ? 1
        : 0.5 * (1 - Math.cos((2 * Math.PI * y) / (source.rows - 1)));
    for (let x = 0; x < source.cols; x += 1) {
      const horizontal =
        source.cols <= 1
          ? 1
          : 0.5 * (1 - Math.cos((2 * Math.PI * x) / (source.cols - 1)));
      values[y * source.cols + x] =
        source.data[y * source.cols + x] * Math.sqrt(vertical * horizontal);
    }
  }
  return cv.matFromArray(source.rows, source.cols, cv.CV_32FC1, values);
}

function alignFrames(
  cv: CvRuntime,
  frames: readonly ColorFrame[],
): { frames: ColorFrame[]; maximumShift: number; minimumResponse: number } {
  const calibrationHeight = Math.round(HEIGHT * 0.42);
  const referenceFrame = cv.matFromArray(
    HEIGHT,
    WIDTH,
    cv.CV_8UC3,
    frames[Math.floor(frames.length / 2)],
  );
  const referenceGray = new cv.Mat();
  const referenceBlur = new cv.Mat();
  cv.cvtColor(referenceFrame, referenceGray, cv.COLOR_BGR2GRAY);
  const referenceRoi = referenceGray.roi(
    new cv.Rect(0, 0, WIDTH, calibrationHeight),
  );
  cv.GaussianBlur(referenceRoi, referenceBlur, new cv.Size(7, 7), 0);
  const referenceWindowed = hanningWindowed(cv, referenceBlur);
  const aligned: ColorFrame[] = [];
  const shifts: number[] = [];
  const responses: number[] = [];
  try {
    for (const frame of frames) {
      const source = cv.matFromArray(HEIGHT, WIDTH, cv.CV_8UC3, frame);
      const gray = new cv.Mat();
      const blur = new cv.Mat();
      const output = new cv.Mat();
      let roi: Mat | null = null;
      let transform: Mat | null = null;
      let windowed: Mat | null = null;
      try {
        cv.cvtColor(source, gray, cv.COLOR_BGR2GRAY);
        roi = gray.roi(new cv.Rect(0, 0, WIDTH, calibrationHeight));
        cv.GaussianBlur(roi, blur, new cv.Size(7, 7), 0);
        windowed = hanningWindowed(cv, blur);
        const [dx, dy, response] = phaseCorrelate(
          cv,
          referenceWindowed,
          windowed,
        );
        const normalizedShift = Math.hypot(dx / WIDTH, dy / HEIGHT);
        if (
          Number.isFinite(normalizedShift) &&
          Number.isFinite(response) &&
          response >= 0.02 &&
          normalizedShift <= 0.12
        ) {
          transform = cv.matFromArray(2, 3, cv.CV_64FC1, [
            1,
            0,
            -dx,
            0,
            1,
            -dy,
          ]);
          cv.warpAffine(
            source,
            output,
            transform,
            new cv.Size(WIDTH, HEIGHT),
            cv.INTER_LINEAR,
            cv.BORDER_REFLECT,
          );
          shifts.push(normalizedShift);
          aligned.push(Uint8Array.from(output.data));
        } else {
          shifts.push(0);
          aligned.push(Uint8Array.from(source.data));
        }
        responses.push(Number.isFinite(response) ? response : 0);
      } finally {
        source.delete();
        gray.delete();
        blur.delete();
        output.delete();
        roi?.delete();
        transform?.delete();
        windowed?.delete();
      }
    }
  } finally {
    referenceFrame.delete();
    referenceGray.delete();
    referenceBlur.delete();
    referenceRoi.delete();
    referenceWindowed.delete();
  }
  return {
    frames: aligned,
    maximumShift: shifts.length > 0 ? Math.max(...shifts) : 0,
    minimumResponse: responses.length > 0 ? Math.min(...responses) : 0,
  };
}

function preparedSequence(
  cv: CvRuntime,
  frames: readonly ColorFrame[],
  geometry: CourtGeometry,
) {
  const normalized = frames.map((frame) =>
    normalizeCourtFrame(cv, frame, geometry),
  );
  const aligned = alignFrames(cv, normalized);
  const converted = aligned.frames.map((frame) => grayAndHsv(cv, frame));
  const median = new Float64Array(WIDTH * HEIGHT);
  const motion = new Float64Array(WIDTH * HEIGHT);
  for (let pixel = 0; pixel < median.length; pixel += 1) {
    const values = converted
      .map((frame) => frame.gray[pixel])
      .sort((left, right) => left - right);
    median[pixel] = quantile(values, 0.5);
    motion[pixel] = (values.at(-1)! - values[0]) / 255;
  }
  return { converted, median, motion, ...aligned };
}

function pooledRegion(
  converted: readonly { gray: Uint8Array; hsv: Uint8Array }[],
  median: Float64Array,
  motion: Float64Array,
  topRatio: number,
  bottomRatio: number,
): { palette: Float64Array; instability: number } {
  const top = Math.round(HEIGHT * topRatio);
  const bottom = Math.round(HEIGHT * bottomRatio);
  const left = Math.round(WIDTH * 0.06);
  const right = Math.round(WIDTH * 0.94);
  const palettes = converted.map((frame) => {
    const weights = new Float64Array(median.length);
    for (let pixel = 0; pixel < weights.length; pixel += 1) {
      weights[pixel] =
        Math.max(
          Math.abs(frame.gray[pixel] - median[pixel]) / 255 - 2 / 255,
          0,
        ) +
        0.2 * Math.max(motion[pixel] - 5 / 255, 0);
    }
    return weightedPalette(frame.hsv, weights, left, top, right, bottom, false);
  });
  const palette = normalizedAverage(palettes);
  return {
    palette,
    instability: quantile(
      palettes.map((value) => hellinger(value, palette)),
      0.5,
    ),
  };
}

function summarizeBroad(
  cv: CvRuntime,
  frames: readonly ColorFrame[],
  geometry: CourtGeometry,
): BroadSummary {
  const prepared = preparedSequence(cv, frames, geometry);
  const broadFar = pooledRegion(
    prepared.converted,
    prepared.median,
    prepared.motion,
    0.15,
    0.6,
  );
  const broadNear = pooledRegion(
    prepared.converted,
    prepared.median,
    prepared.motion,
    0.44,
    0.98,
  );
  const tightFar = pooledRegion(
    prepared.converted,
    prepared.median,
    prepared.motion,
    0.27,
    0.53,
  );
  const tightNear = pooledRegion(
    prepared.converted,
    prepared.median,
    prepared.motion,
    0.56,
    0.91,
  );
  const global = pooledRegion(
    prepared.converted,
    prepared.median,
    prepared.motion,
    0.08,
    0.98,
  );
  let active = 0;
  for (const value of prepared.motion) if (value > 10 / 255) active += 1;
  return {
    broad: {
      near: broadNear.palette,
      far: broadFar.palette,
      instability: (broadNear.instability + broadFar.instability) / 2,
    },
    tight: {
      near: tightNear.palette,
      far: tightFar.palette,
      instability: (tightNear.instability + tightFar.instability) / 2,
    },
    global: global.palette,
    foregroundCoverage: active / prepared.motion.length,
    maximumCameraShift: prepared.maximumShift,
    minimumAlignmentResponse: prepared.minimumResponse,
  };
}

function assignment(
  before: PaletteSummary,
  after: PaletteSummary,
): [number, number, number, number] {
  const same =
    (hellinger(before.near, after.near) + hellinger(before.far, after.far)) / 2;
  const swapped =
    (hellinger(before.near, after.far) + hellinger(before.far, after.near)) / 2;
  const beforeOrientation = Float64Array.from(
    before.near,
    (value, index) => value - before.far[index],
  );
  const afterOrientation = Float64Array.from(
    after.near,
    (value, index) => value - after.far[index],
  );
  return [
    same,
    swapped,
    same - swapped,
    -cosine(beforeOrientation, afterOrientation),
  ];
}

function broadFeatures(
  before: BroadSummary,
  after: BroadSummary,
): Record<string, number> {
  const broad = assignment(before.broad, after.broad);
  const tight = assignment(before.tight, after.tight);
  return {
    broadSameAssignmentCost: broad[0],
    tightSameAssignmentCost: tight[0],
    meanSwapMargin: (broad[2] + tight[2]) / 2,
    globalAppearanceChange: hellinger(before.global, after.global),
    maximumCameraShift: Math.max(
      before.maximumCameraShift,
      after.maximumCameraShift,
    ),
    minimumAlignmentResponse: Math.min(
      before.minimumAlignmentResponse,
      after.minimumAlignmentResponse,
    ),
  };
}

type Box = {
  x: number;
  y: number;
  width: number;
  height: number;
  area: number;
};

function closeMask(mask: Uint8Array): Uint8Array {
  const dilated = new Uint8Array(mask.length);
  const offsets = [
    [0, 0],
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1],
  ] as const;
  for (let y = 0; y < HEIGHT; y += 1)
    for (let x = 0; x < WIDTH; x += 1) {
      dilated[y * WIDTH + x] = offsets.some(([dx, dy]) => {
        const nextX = x + dx;
        const nextY = y + dy;
        return (
          nextX >= 0 &&
          nextX < WIDTH &&
          nextY >= 0 &&
          nextY < HEIGHT &&
          mask[nextY * WIDTH + nextX] > 0
        );
      })
        ? 1
        : 0;
    }
  const eroded = new Uint8Array(mask.length);
  for (let y = 0; y < HEIGHT; y += 1)
    for (let x = 0; x < WIDTH; x += 1) {
      eroded[y * WIDTH + x] = offsets.every(([dx, dy]) => {
        const nextX = x + dx;
        const nextY = y + dy;
        return (
          nextX >= 0 &&
          nextX < WIDTH &&
          nextY >= 0 &&
          nextY < HEIGHT &&
          dilated[nextY * WIDTH + nextX] > 0
        );
      })
        ? 1
        : 0;
    }
  return eroded;
}

function components(mask: Uint8Array): Box[] {
  const seen = new Uint8Array(mask.length);
  const queue = new Int32Array(mask.length);
  const result: Box[] = [];
  for (let origin = 0; origin < mask.length; origin += 1) {
    if (!mask[origin] || seen[origin]) continue;
    let head = 0;
    let tail = 0;
    let minimumX = WIDTH;
    let maximumX = 0;
    let minimumY = HEIGHT;
    let maximumY = 0;
    queue[tail++] = origin;
    seen[origin] = 1;
    while (head < tail) {
      const pixel = queue[head++];
      const y = Math.floor(pixel / WIDTH);
      const x = pixel - y * WIDTH;
      minimumX = Math.min(minimumX, x);
      maximumX = Math.max(maximumX, x);
      minimumY = Math.min(minimumY, y);
      maximumY = Math.max(maximumY, y);
      for (let dy = -1; dy <= 1; dy += 1)
        for (let dx = -1; dx <= 1; dx += 1) {
          if (dx === 0 && dy === 0) continue;
          const nextX = x + dx;
          const nextY = y + dy;
          if (nextX < 0 || nextX >= WIDTH || nextY < 0 || nextY >= HEIGHT)
            continue;
          const next = nextY * WIDTH + nextX;
          if (mask[next] && !seen[next]) {
            seen[next] = 1;
            queue[tail++] = next;
          }
        }
    }
    result.push({
      x: minimumX,
      y: minimumY,
      width: maximumX - minimumX + 1,
      height: maximumY - minimumY + 1,
      area: tail,
    });
  }
  return result;
}

function iou(first: Box, second: Box): number {
  const width = Math.max(
    0,
    Math.min(first.x + first.width, second.x + second.width) -
      Math.max(first.x, second.x),
  );
  const height = Math.max(
    0,
    Math.min(first.y + first.height, second.y + second.height) -
      Math.max(first.y, second.y),
  );
  const intersection = width * height;
  const union =
    first.width * first.height + second.width * second.height - intersection;
  return union ? intersection / union : 0;
}

function proposalBoxes(difference: Float64Array): Box[] {
  const numeric = Uint8Array.from(difference, (value) =>
    clamp(Math.floor(value * 255), 0, 255),
  );
  const threshold = Math.max(12, Math.floor(quantile(numeric, 0.94)));
  const mask = closeMask(
    Uint8Array.from(numeric, (value) => (value >= threshold ? 1 : 0)),
  );
  const candidates = components(mask)
    .flatMap((box) => {
      const bottom = box.y + box.height;
      const aspect = box.height / Math.max(box.width, 1);
      if (
        box.area < 10 ||
        box.area > WIDTH * HEIGHT * 0.035 ||
        box.width < 2 ||
        box.height < 4 ||
        box.width > WIDTH * 0.18 ||
        box.height > HEIGHT * 0.48 ||
        aspect < 0.45 ||
        aspect > 5.5 ||
        bottom < HEIGHT * 0.35 ||
        box.y > HEIGHT * 0.95
      )
        return [];
      const padX = Math.max(1, Math.round(box.width * 0.18));
      const padY = Math.max(1, Math.round(box.height * 0.1));
      const x = Math.max(0, box.x - padX);
      const y = Math.max(0, box.y - padY);
      const right = Math.min(WIDTH, box.x + box.width + padX);
      const lower = Math.min(HEIGHT, bottom + padY);
      const expanded = {
        x,
        y,
        width: right - x,
        height: lower - y,
        area: box.area,
      };
      const score =
        box.area * (0.5 + lower / HEIGHT) ** 2 * Math.min(aspect, 2.5);
      return [{ box: expanded, score }];
    })
    .sort((left, right) => right.score - left.score);
  const selected: Box[] = [];
  for (const candidate of candidates) {
    if (selected.every((box) => iou(candidate.box, box) < 0.35))
      selected.push(candidate.box);
    if (selected.length === 6) break;
  }
  return selected;
}

function pooledWeightedPalettes(
  values: readonly { palette: Float64Array; weight: number }[],
) {
  if (values.length === 0)
    return {
      palette: Float64Array.from({ length: 52 }, () => 1 / 52),
      instability: 0,
    };
  const palette = normalizedAverage(
    values.map((value) => value.palette),
    values.map((value) => value.weight),
  );
  return {
    palette,
    instability: quantile(
      values.map((value) => hellinger(value.palette, palette)),
      0.5,
    ),
  };
}

function summarizePlayers(
  cv: CvRuntime,
  frames: readonly ColorFrame[],
  geometry: CourtGeometry,
): PlayerSummary {
  const prepared = preparedSequence(cv, frames, geometry);
  const near: { palette: Float64Array; weight: number }[] = [];
  const far: { palette: Float64Array; weight: number }[] = [];
  const global: { palette: Float64Array; weight: number }[] = [];
  const coverages: number[] = [];
  const counts: number[] = [];
  let nearSupport = 0;
  let farSupport = 0;
  for (const frame of prepared.converted) {
    const difference = Float64Array.from(
      frame.gray,
      (value, pixel) => Math.abs(value - prepared.median[pixel]) / 255,
    );
    const boxes = proposalBoxes(difference);
    const covered = new Uint8Array(WIDTH * HEIGHT);
    counts.push(boxes.length);
    for (const box of boxes) {
      const weights = new Float64Array(WIDTH * HEIGHT);
      let support = 0;
      for (let y = box.y; y < box.y + box.height; y += 1)
        for (let x = box.x; x < box.x + box.width; x += 1) {
          const pixel = y * WIDTH + x;
          covered[pixel] = 1;
          const saturation = frame.hsv[pixel * 3 + 1] / 255;
          weights[pixel] =
            difference[pixel] +
            0.15 * prepared.motion[pixel] +
            0.01 * saturation;
          support += weights[pixel];
        }
      support = Math.max(support, 1e-8);
      const palette = weightedPalette(
        frame.hsv,
        weights,
        box.x,
        box.y,
        box.x + box.width,
        box.y + box.height,
        true,
      );
      const footY = (box.y + box.height) / HEIGHT;
      const nearProbability = 1 / (1 + Math.exp(-(footY - 0.63) / 0.06));
      const farProbability = 1 - nearProbability;
      if (nearProbability >= 0.08) {
        const weight = support * nearProbability;
        near.push({ palette, weight });
        nearSupport += weight;
      }
      if (farProbability >= 0.08) {
        const weight = support * farProbability;
        far.push({ palette, weight });
        farSupport += weight;
      }
      global.push({ palette, weight: support });
    }
    coverages.push(mean(covered));
  }
  const nearPalette = pooledWeightedPalettes(near);
  const farPalette = pooledWeightedPalettes(far);
  const globalPalette = pooledWeightedPalettes(global);
  const totalSupport = Math.max(nearSupport + farSupport, 1e-12);
  return {
    near: nearPalette.palette,
    far: farPalette.palette,
    global: globalPalette.palette,
    instability: (nearPalette.instability + farPalette.instability) / 2,
    proposalCoverage: mean(coverages),
    proposalCount: mean(counts),
    nearSupport: nearSupport / totalSupport,
    farSupport: farSupport / totalSupport,
  };
}

function playerFeatures(
  before: PlayerSummary,
  after: PlayerSummary,
  v4: Record<string, number>,
): Float64Array {
  const same =
    (hellinger(before.near, after.near) + hellinger(before.far, after.far)) / 2;
  const swapped =
    (hellinger(before.near, after.far) + hellinger(before.far, after.near)) / 2;
  const beforeOrientation = Float64Array.from(
    before.near,
    (value, index) => value - before.far[index],
  );
  const afterOrientation = Float64Array.from(
    after.near,
    (value, index) => value - after.far[index],
  );
  const beforeSeparation = hellinger(before.near, before.far);
  const afterSeparation = hellinger(after.near, after.far);
  return Float64Array.of(
    v4.broadSameAssignmentCost,
    v4.tightSameAssignmentCost,
    v4.meanSwapMargin,
    v4.globalAppearanceChange,
    v4.maximumCameraShift,
    v4.minimumAlignmentResponse,
    same,
    swapped,
    same - swapped,
    -cosine(beforeOrientation, afterOrientation),
    Math.min(beforeSeparation, afterSeparation),
    Math.abs(beforeSeparation - afterSeparation),
    before.instability,
    after.instability,
    hellinger(before.global, after.global),
    Math.min(before.proposalCoverage, after.proposalCoverage),
    Math.abs(before.proposalCoverage - after.proposalCoverage),
    Math.min(before.proposalCount, after.proposalCount),
    Math.abs(before.proposalCount - after.proposalCount),
    Math.min(before.nearSupport, after.nearSupport),
    Math.min(before.farSupport, after.farSupport),
    Math.abs(
      Math.abs(before.nearSupport - before.farSupport) -
        Math.abs(after.nearSupport - after.farSupport),
    ),
  );
}

function horizontalLineCandidate(
  cv: CvRuntime,
  frame: ColorFrame,
): { position: number; score: number } | null {
  const source = cv.matFromArray(HEIGHT, WIDTH, cv.CV_8UC3, frame);
  const gray = new cv.Mat();
  const blurred = new cv.Mat();
  const edges = new cv.Mat();
  const lines = new cv.Mat();
  try {
    cv.cvtColor(source, gray, cv.COLOR_BGR2GRAY);
    cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);
    cv.Canny(blurred, edges, 45, 110);
    cv.HoughLinesP(
      edges,
      lines,
      1,
      Math.PI / 180,
      Math.max(24, Math.floor(WIDTH / 10)),
      WIDTH * 0.32,
      WIDTH * 0.12,
    );
    let best: { position: number; score: number } | null = null;
    const data = lines.data32S;
    for (let index = 0; index + 3 < data.length; index += 4) {
      const [x1, y1, x2, y2] = data.subarray(index, index + 4);
      const length = Math.hypot(x2 - x1, y2 - y1);
      const coverage = Math.abs(x2 - x1) / WIDTH;
      const position = (y1 + y2) / (2 * HEIGHT);
      const slope = Math.abs(y2 - y1) / Math.max(Math.abs(x2 - x1), 1);
      if (position < 0.25 || position > 0.68 || slope > 0.08 || coverage < 0.3)
        continue;
      const score = coverage * (1 + 0.35 * position) + (0.1 * length) / WIDTH;
      if (!best || score > best.score)
        best = { position, score: Math.min(1, score) };
    }
    return best;
  } finally {
    source.delete();
    gray.delete();
    blurred.delete();
    edges.delete();
    lines.delete();
  }
}

function estimateCourtGeometry(
  cv: CvRuntime,
  frames: readonly ColorFrame[],
): CourtGeometry {
  const candidates = frames.flatMap((frame) => {
    const value = horizontalLineCandidate(cv, frame);
    return value ? [value] : [];
  });
  if (candidates.length === 0) return { netYRatio: 0.5, confidence: 0 };
  const positions = candidates.map((candidate) => candidate.position);
  let center = quantile(positions, 0.5);
  const deviations = positions.map((position) => Math.abs(position - center));
  const cutoff = Math.max(0.035, quantile(deviations, 0.5) * 2.5);
  const retained = candidates.filter((_, index) => deviations[index] <= cutoff);
  if (retained.length > 0) {
    const total = retained.reduce((sum, candidate) => sum + candidate.score, 0);
    center =
      retained.reduce(
        (sum, candidate) => sum + candidate.position * candidate.score,
        0,
      ) / total;
  }
  const dispersion = quantile(
    positions.map((position) => Math.abs(position - quantile(positions, 0.5))),
    0.5,
  );
  return {
    netYRatio: clamp(center, 0.25, 0.68),
    confidence: clamp(
      (candidates.length / frames.length) * Math.max(0, 1 - dispersion / 0.1),
      0,
      1,
    ),
  };
}

async function sampleFrames(
  cv: CvRuntime,
  media: OpenedMedia,
  roi: NormalizedRoi,
  requestedTimes: readonly number[],
  onProgress?: (progress: SideSwitchProgress) => void,
): Promise<Map<string, ColorFrame>> {
  const requested = [
    ...new Set(
      requestedTimes
        .map((time) => clampedTimestamp(time, media.info.duration))
        .map(timestampKey),
    ),
  ]
    .map(Number)
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
    width: WIDTH,
    height: HEIGHT,
    fit: "fill",
    poolSize: 1,
    decoderOptions: { hardwareAcceleration: "prefer-hardware" },
  });
  const result = new Map<string, ColorFrame>();
  const iterator = sink.canvasesAtTimestamps(requested);
  let index = 0;
  try {
    for await (const wrapped of iterator) {
      if (!wrapped)
        throw new Error(
          `No side-switch frame was available at ${requested[index]} seconds.`,
        );
      const rgba = cv.imread(wrapped.canvas as unknown as HTMLCanvasElement);
      const bgr = new cv.Mat();
      try {
        cv.cvtColor(rgba, bgr, cv.COLOR_RGBA2BGR);
        result.set(timestampKey(requested[index]), Uint8Array.from(bgr.data));
      } finally {
        rgba.delete();
        bgr.delete();
      }
      index += 1;
      if (index % 8 === 0 || index === requested.length) {
        onProgress?.({
          stage: "frames",
          completed: index,
          total: requested.length,
          detail: `Sampling team-side switch windows · ${index}/${requested.length} frames`,
        });
        await new Promise<void>((resolve) => setTimeout(resolve, 0));
      }
    }
  } finally {
    await iterator.return(undefined);
  }
  if (index !== requested.length)
    throw new Error("Side-switch frame sampling ended early.");
  return result;
}

function framesAt(
  sampled: ReadonlyMap<string, ColorFrame>,
  times: readonly number[],
  duration: number,
): ColorFrame[] {
  return times.map((time) => {
    const timestamp = clampedTimestamp(time, duration);
    const frame = sampled.get(timestampKey(timestamp));
    if (!frame)
      throw new Error(`Missing side-switch frame at ${timestamp} seconds.`);
    return frame;
  });
}

function comparisonFrames(
  sampled: ReadonlyMap<string, ColorFrame>,
  candidate: SideSwitchCandidateProposal,
  duration: number,
): [ColorFrame[], ColorFrame[]] {
  const times = sideSwitchCandidateSampleTimes(candidate);
  return [
    framesAt(sampled, times.slice(0, 7), duration),
    framesAt(sampled, times.slice(7), duration),
  ];
}

export async function inferSideSwitches(
  media: OpenedMedia,
  roi: NormalizedRoi,
  analysis: SideSwitchAnalysisInput,
  onProgress?: (progress: SideSwitchProgress) => void,
): Promise<OnDeviceSideSwitchOutput> {
  onProgress?.({
    stage: "loading",
    completed: 0,
    total: 1,
    detail: "Loading team-side switch model",
  });
  const [cv, runtime] = await Promise.all([
    loadOpenCv(),
    loadSideSwitchRuntime(),
  ]);
  const candidates = generateSideSwitchCandidates(analysis, runtime);
  if (candidates.length === 0) return emptySideSwitchOutput(runtime);
  const calibrationTimes = sideSwitchCalibrationTimes(analysis.intervals);
  const requested = [
    ...calibrationTimes,
    ...candidates.flatMap(sideSwitchCandidateSampleTimes),
  ];
  const sampled = await sampleFrames(cv, media, roi, requested, onProgress);
  const geometry = estimateCourtGeometry(
    cv,
    framesAt(sampled, calibrationTimes, media.info.duration),
  );
  const columns = SIDE_SWITCH_FEATURE_NAMES.length;
  const features = new Float64Array(candidates.length * columns);
  for (let row = 0; row < candidates.length; row += 1) {
    const candidate = candidates[row];
    const [beforeFrames, afterFrames] = comparisonFrames(
      sampled,
      candidate,
      media.info.duration,
    );
    const beforeBroad = summarizeBroad(cv, beforeFrames, geometry);
    const afterBroad = summarizeBroad(cv, afterFrames, geometry);
    const beforePlayers = summarizePlayers(cv, beforeFrames, geometry);
    const afterPlayers = summarizePlayers(cv, afterFrames, geometry);
    const visual = playerFeatures(
      beforePlayers,
      afterPlayers,
      broadFeatures(beforeBroad, afterBroad),
    );
    const state = sideSwitchStateFeatures(candidate, analysis);
    features.set(visual, row * columns);
    features.set(state, row * columns + visual.length);
    features[row * columns + visual.length + state.length] =
      candidate.kind === "internal-dead-state-peak" ? 1 : 0;
    features[row * columns + visual.length + state.length + 1] =
      candidate.generatorScore;
    onProgress?.({
      stage: "features",
      completed: row + 1,
      total: candidates.length,
      detail: `Comparing team sides · ${row + 1}/${candidates.length} candidates`,
    });
    if ((row + 1) % 2 === 0)
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
  }
  const probabilities = predictSideSwitchProbabilities(runtime, features);
  const selected = decodeSideSwitchCandidates(
    runtime,
    candidates,
    probabilities,
  );
  const output: OnDeviceSideSwitchOutput = {
    modelId: runtime.modelId,
    modelFingerprint: runtime.fingerprint,
    featureVersion: SIDE_SWITCH_FEATURE_VERSION,
    candidateContract: SIDE_SWITCH_CANDIDATE_CONTRACT,
    features: { rows: candidates.length, columns, values: features },
    candidates: selected.map((index) => ({
      id: candidates[index].id,
      timestamp: candidates[index].transitionTime,
      probability: probabilities[index],
      kind: candidates[index].kind,
      sourceRangeIds: [...candidates[index].sourceRangeIds],
    })),
  };
  onProgress?.({
    stage: "complete",
    completed: output.candidates.length,
    total: output.candidates.length,
    detail: `Team-side switch markers ready · ${output.candidates.length} predicted`,
  });
  return output;
}
