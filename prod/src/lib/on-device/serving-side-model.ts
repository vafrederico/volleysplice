import type {
  OnDeviceInterval,
  OnDeviceServeOutput,
  ServingSideCandidateVerdict,
  ServingSideDecisionSource,
  ServingSideHeadEvidence,
  ServingSideReviewReason,
  ServingSideSide,
} from "./types.ts";

import {
  SERVING_SIDE_ANCHOR_CONTRACT,
  SERVING_SIDE_FEATURE_VERSION,
  SERVING_SIDE_MODEL_FINGERPRINT,
  SERVING_SIDE_MODEL_ID,
} from "./serving-side-cache.ts";

export {
  isReusableServingSideOutput,
  SERVING_SIDE_ANCHOR_CONTRACT,
  SERVING_SIDE_FEATURE_COLUMNS,
  SERVING_SIDE_FEATURE_VERSION,
  SERVING_SIDE_MODEL_FINGERPRINT,
  SERVING_SIDE_MODEL_ID,
} from "./serving-side-cache.ts";

export const SERVING_SIDE_ASSET = "serving-side-85bc3325fbd4.json" as const;

export const COURT_FLOW_OFFSETS_SECONDS = [
  -1.25, -0.75, -0.35, -0.1, 0.1, 0.3, 0.55, 0.85,
] as const;
export const FLIGHT_OFFSETS_SECONDS = [
  -0.15, 0.05, 0.2, 0.35, 0.55, 0.8, 1.1, 1.4, 1.75,
] as const;
export const SERVING_SIDE_WIDTH = 192;
export const SERVING_SIDE_HEIGHT = 108;
export const FLIGHT_GRID_ROWS = 4;
export const FLIGHT_GRID_COLUMNS = 6;
export const SERVING_SIDE_THRESHOLD = 0.4783744762021848;
export const SERVING_SIDE_FAR_REVIEW_THRESHOLD = 0.3121748736511044;
export const SERVING_SIDE_NEAR_REVIEW_THRESHOLD = 0.5028396703865513;
export const SERVING_SIDE_SERVE_HEAD_THRESHOLD = 0.85;
export const SERVING_SIDE_SERVE_HEAD_WINDOW_SECONDS = 1;

const PHASES = ["pre", "contact", "post"] as const;
const ZONES = ["near", "far"] as const;
export const COURT_FLOW_STATISTICS = [
  "flowMean",
  "flowP90",
  "activeFraction",
  "largestComponentFraction",
  "componentCountDensity",
  "centroidX",
  "centroidY",
  "flowX",
  "flowY",
] as const;
const FLIGHT_PHASES = ["launch", "early", "late"] as const;
export const FLIGHT_GLOBAL_STATISTICS = [
  "energyMean",
  "activeFraction",
  "centroidX",
  "centroidY",
  "spreadX",
  "spreadY",
  "entropy",
  "largestComponentFraction",
  "flowX",
  "flowY",
  "divergence",
  "bottomMinusTop",
  "smallComponentEnergyFraction",
  "smallComponentCentroidY",
  "smallComponentFlowY",
] as const;
export const FLIGHT_TRAJECTORY_STATISTICS = [
  "centroidY",
  "spreadY",
  "entropy",
  "flowY",
  "divergence",
  "bottomMinusTop",
  "smallComponentEnergyFraction",
  "smallComponentCentroidY",
  "smallComponentFlowY",
] as const;

export function courtFlowFeatureNames(): string[] {
  const names: string[] = [];
  for (const phase of PHASES) {
    for (const zone of ZONES) {
      for (const statistic of COURT_FLOW_STATISTICS) {
        names.push(`${phase}:${zone}:${statistic}`);
      }
    }
    for (const statistic of COURT_FLOW_STATISTICS.slice(0, 5)) {
      names.push(`${phase}:nearMinusFar:${statistic}`);
    }
  }
  for (const zone of ZONES) {
    names.push(
      `contactMinusPre:${zone}:flowMean`,
      `contactMinusPre:${zone}:activeFraction`,
      `contactMinusPre:${zone}:largestComponentFraction`,
      `postMinusContact:${zone}:flowMean`,
      `postMinusContact:${zone}:activeFraction`,
    );
  }
  names.push(
    "contactDelta:nearMinusFar:flowMean",
    "contactDelta:nearMinusFar:activeFraction",
    "contactDelta:nearMinusFar:largestComponentFraction",
  );
  return names;
}

export function flightFeatureNames(
  rows = FLIGHT_GRID_ROWS,
  columns = FLIGHT_GRID_COLUMNS,
): string[] {
  if (rows < 2 || columns < 2) throw new Error("Flight grids require at least 2×2 cells.");
  const names: string[] = [];
  for (const phase of FLIGHT_PHASES) {
    for (let row = 0; row < rows; row += 1) {
      for (let column = 0; column < columns; column += 1) {
        names.push(`${phase}:grid:r${row}:c${column}:energy`);
      }
    }
    for (let row = 0; row < rows; row += 1) {
      names.push(`${phase}:row:r${row}:flowY`);
    }
    for (const statistic of FLIGHT_GLOBAL_STATISTICS) {
      names.push(`${phase}:${statistic}`);
    }
  }
  for (const transition of ["launchToEarly", "earlyToLate"] as const) {
    for (const statistic of FLIGHT_TRAJECTORY_STATISTICS) {
      names.push(`trajectory:${transition}:${statistic}`);
    }
    for (let row = 0; row < rows; row += 1) {
      names.push(`trajectory:${transition}:row:r${row}:energy`);
    }
  }
  return names;
}

export function servingSideFeatureNames(): string[] {
  return [
    ...courtFlowFeatureNames().map((name) => `v2:${name}`),
    ...flightFeatureNames().map((name) => `flight:${name}`),
  ];
}

/**
 * Confirms that a stored serving-side result belongs to the current frozen
 * model and the exact production candidate anchors. Ignored editor ranges are
 * intentionally absent from this identity: they filter results after inference.
 */
export type ServingSideRuntimeModel = {
  schemaVersion: 1;
  kind: "volleycut-serving-side-fixed-flight-runtime-v1";
  modelId: typeof SERVING_SIDE_MODEL_ID;
  fingerprint: typeof SERVING_SIDE_MODEL_FINGERPRINT;
  featureVersion: typeof SERVING_SIDE_FEATURE_VERSION;
  courtFlowOffsetsSeconds: number[];
  flightOffsetsSeconds: number[];
  resize: { width: number; height: number };
  flightGrid: { rows: number; columns: number };
  featureNames: string[];
  model: {
    family: "class-balanced-logistic";
    impute: number[];
    mean: number[];
    scale: number[];
    weights: number[];
    bias: number;
    l2: number;
    threshold: number;
  };
  sideThreshold: number;
  reviewBand: {
    farUpperExclusive: number;
    nearLowerInclusive: number;
  };
  gate: {
    serveHeadThreshold: number;
    serveHeadWindowSeconds: number;
    rallyRecoveryAgreement: "both-models";
    rallyRecoveryRequiresReview: true;
  };
};

function objectValue(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function finiteNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be finite.`);
  }
  return value;
}

function numericVector(value: unknown, length: number, label: string): number[] {
  if (!Array.isArray(value) || value.length !== length) {
    throw new Error(`${label} must contain ${length} values.`);
  }
  return value.map((item, index) => finiteNumber(item, `${label}[${index}]`));
}

function vectorsEqual(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

export function parseServingSideRuntime(value: unknown): ServingSideRuntimeModel {
  const payload = objectValue(value, "Serving-side runtime");
  const expectedNames = servingSideFeatureNames();
  if (
    payload.schemaVersion !== 1 ||
    payload.kind !== "volleycut-serving-side-fixed-flight-runtime-v1" ||
    payload.modelId !== SERVING_SIDE_MODEL_ID ||
    payload.fingerprint !== SERVING_SIDE_MODEL_FINGERPRINT ||
    payload.featureVersion !== SERVING_SIDE_FEATURE_VERSION
  ) throw new Error("Serving-side runtime identity does not match the frozen model.");
  if (!Array.isArray(payload.featureNames) ||
      payload.featureNames.length !== expectedNames.length ||
      !payload.featureNames.every((name, index) => name === expectedNames[index])) {
    throw new Error("Serving-side runtime feature signature changed.");
  }
  if (!vectorsEqual(numericVector(payload.courtFlowOffsetsSeconds, 8, "Court offsets"), COURT_FLOW_OFFSETS_SECONDS) ||
      !vectorsEqual(numericVector(payload.flightOffsetsSeconds, 9, "Flight offsets"), FLIGHT_OFFSETS_SECONDS)) {
    throw new Error("Serving-side runtime sampling offsets changed.");
  }
  const resize = objectValue(payload.resize, "Serving-side resize");
  const grid = objectValue(payload.flightGrid, "Serving-side grid");
  if (resize.width !== SERVING_SIDE_WIDTH || resize.height !== SERVING_SIDE_HEIGHT ||
      grid.rows !== FLIGHT_GRID_ROWS || grid.columns !== FLIGHT_GRID_COLUMNS) {
    throw new Error("Serving-side runtime image geometry changed.");
  }
  const model = objectValue(payload.model, "Serving-side logistic model");
  if (model.family !== "class-balanced-logistic") {
    throw new Error("Serving-side runtime must use class-balanced logistic regression.");
  }
  const length = expectedNames.length;
  const parsedModel = {
    family: "class-balanced-logistic" as const,
    impute: numericVector(model.impute, length, "Serving-side imputation"),
    mean: numericVector(model.mean, length, "Serving-side mean"),
    scale: numericVector(model.scale, length, "Serving-side scale"),
    weights: numericVector(model.weights, length, "Serving-side weights"),
    bias: finiteNumber(model.bias, "Serving-side bias"),
    l2: finiteNumber(model.l2, "Serving-side L2"),
    threshold: finiteNumber(model.threshold, "Serving-side model threshold"),
  };
  if (parsedModel.scale.some((item) => item <= 0)) {
    throw new Error("Serving-side scales must be positive.");
  }
  const reviewBand = objectValue(payload.reviewBand, "Serving-side review band");
  const gate = objectValue(payload.gate, "Serving-side gate");
  const sideThreshold = finiteNumber(payload.sideThreshold, "Serving-side threshold");
  const farUpperExclusive = finiteNumber(reviewBand.farUpperExclusive, "Far review threshold");
  const nearLowerInclusive = finiteNumber(reviewBand.nearLowerInclusive, "Near review threshold");
  if (
    parsedModel.threshold !== SERVING_SIDE_THRESHOLD ||
    sideThreshold !== SERVING_SIDE_THRESHOLD ||
    farUpperExclusive !== SERVING_SIDE_FAR_REVIEW_THRESHOLD ||
    nearLowerInclusive !== SERVING_SIDE_NEAR_REVIEW_THRESHOLD ||
    gate.rallyRecoveryAgreement !== "both-models" ||
    gate.rallyRecoveryRequiresReview !== true ||
    gate.serveHeadThreshold !== SERVING_SIDE_SERVE_HEAD_THRESHOLD ||
    gate.serveHeadWindowSeconds !== SERVING_SIDE_SERVE_HEAD_WINDOW_SECONDS
  ) throw new Error("Serving-side decision policy is malformed.");
  return {
    ...(payload as unknown as ServingSideRuntimeModel),
    featureNames: expectedNames,
    model: parsedModel,
    sideThreshold,
    reviewBand: { farUpperExclusive, nearLowerInclusive },
    gate: {
      serveHeadThreshold: finiteNumber(gate.serveHeadThreshold, "Serve-head threshold"),
      serveHeadWindowSeconds: finiteNumber(gate.serveHeadWindowSeconds, "Serve-head window"),
      rallyRecoveryAgreement: "both-models",
      rallyRecoveryRequiresReview: true,
    },
  };
}

/** Python/NumPy-compatible tied ranks: equal raw values receive their normalized midrank. */
export function tiedPercentileRanks(
  values: ArrayLike<number>,
  rows: number,
  columns: number,
): Float64Array {
  if (!Number.isInteger(rows) || !Number.isInteger(columns) || rows < 0 || columns < 0 ||
      values.length !== rows * columns) {
    throw new Error("Serving-side raw feature dimensions do not match.");
  }
  const result = new Float64Array(values.length);
  if (rows === 0) return result;
  if (rows === 1) return result.fill(0.5);
  const order = Array.from({ length: rows }, (_, index) => index);
  for (let column = 0; column < columns; column += 1) {
    order.sort((left, right) => {
      const delta = values[left * columns + column] - values[right * columns + column];
      return delta || left - right;
    });
    let start = 0;
    while (start < rows) {
      const value = values[order[start] * columns + column];
      if (!Number.isFinite(value)) throw new Error("Serving-side raw features must be finite.");
      let end = start + 1;
      while (end < rows && values[order[end] * columns + column] === value) end += 1;
      const rank = ((start + end - 1) / 2) / (rows - 1);
      for (let index = start; index < end; index += 1) {
        result[order[index] * columns + column] = rank;
      }
      start = end;
    }
  }
  return result;
}

export function servingSideNearProbability(
  rankedFeatures: ArrayLike<number>,
  runtime: ServingSideRuntimeModel,
): number {
  if (rankedFeatures.length !== runtime.featureNames.length) {
    throw new Error(`Serving-side classifier expects ${runtime.featureNames.length} features.`);
  }
  let logit = runtime.model.bias;
  for (let index = 0; index < rankedFeatures.length; index += 1) {
    const raw = rankedFeatures[index];
    const filled = Number.isFinite(raw) ? raw : runtime.model.impute[index];
    logit += (filled - runtime.model.mean[index]) /
      runtime.model.scale[index] * runtime.model.weights[index];
  }
  const clipped = Math.max(-30, Math.min(30, logit));
  return 1 / (1 + Math.exp(-clipped));
}

export function serveHeadEvidence(
  modelId: string,
  times: ArrayLike<number>,
  output: OnDeviceServeOutput,
  anchor: number,
  threshold: number,
  windowSeconds: number,
): ServingSideHeadEvidence {
  if (!Number.isFinite(anchor) || anchor < 0 || !Number.isFinite(windowSeconds) || windowSeconds < 0 ||
      times.length === 0 || output.probabilities.length !== times.length) {
    throw new Error("Serving-side serve evidence is not aligned.");
  }
  let nearestIndex = 0;
  let nearestDistance = Number.POSITIVE_INFINITY;
  const selected: number[] = [];
  for (let index = 0; index < times.length; index += 1) {
    const time = times[index];
    const probability = output.probabilities[index];
    if (!Number.isFinite(time) || !Number.isFinite(probability) || probability < 0 || probability > 1) {
      throw new Error("Serving-side serve evidence must be finite probabilities.");
    }
    const distance = Math.abs(time - anchor);
    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearestIndex = index;
    }
    if (distance <= windowSeconds + 1e-9) selected.push(index);
  }
  if (selected.length === 0) selected.push(nearestIndex);
  let peakIndex = selected[0];
  for (const index of selected.slice(1)) {
    if (output.probabilities[index] > output.probabilities[peakIndex]) peakIndex = index;
  }
  const nearestDetection = output.detections.length
    ? output.detections.reduce((nearest, candidate) =>
      Math.abs(candidate.time - anchor) < Math.abs(nearest.time - anchor) ? candidate : nearest)
    : null;
  const peakProbability = output.probabilities[peakIndex];
  return {
    modelId,
    threshold,
    peakProbability,
    peakTime: times[peakIndex],
    crossesThreshold: peakProbability >= threshold,
    nearestDetection: nearestDetection ? { ...nearestDetection } : null,
  };
}

export type HybridGateResult = {
  source: ServingSideDecisionSource;
  reviewReasons: ServingSideReviewReason[];
  isServe: boolean;
};

export function evaluateHybridGate(
  agreement: OnDeviceInterval["agreement"],
  headEvidence: readonly ServingSideHeadEvidence[],
): HybridGateResult {
  if (headEvidence.length !== 2) throw new Error("The serving-side gate requires two serve heads.");
  if (headEvidence.some((item) => item.crossesThreshold)) {
    return { source: "serve-head", reviewReasons: [], isServe: true };
  }
  if (agreement === "both-models") {
    return {
      source: "production-rally-recovery",
      reviewReasons: ["production-rally-recovery"],
      isServe: true,
    };
  }
  return { source: "none", reviewReasons: [], isServe: false };
}

export function composeServingSideVerdict(
  interval: OnDeviceInterval,
  rankedFeatures: ArrayLike<number>,
  times: ArrayLike<number>,
  allLabelsV2: OnDeviceServeOutput,
  previousProduction: OnDeviceServeOutput,
  runtime: ServingSideRuntimeModel,
): ServingSideCandidateVerdict {
  const anchor = interval.start;
  const allEvidence = serveHeadEvidence(
    "model-1ca43e38eefc", times, allLabelsV2, anchor,
    runtime.gate.serveHeadThreshold, runtime.gate.serveHeadWindowSeconds,
  );
  const previousEvidence = serveHeadEvidence(
    "model-9c92b8e9333f", times, previousProduction, anchor,
    runtime.gate.serveHeadThreshold, runtime.gate.serveHeadWindowSeconds,
  );
  const probability = servingSideNearProbability(rankedFeatures, runtime);
  const side: ServingSideSide = probability >= runtime.sideThreshold ? "near" : "far";
  const gate = evaluateHybridGate(interval.agreement, [allEvidence, previousEvidence]);
  const reviewReasons = [...gate.reviewReasons];
  if (probability >= runtime.reviewBand.farUpperExclusive &&
      probability < runtime.reviewBand.nearLowerInclusive) {
    reviewReasons.unshift("side-score");
  }
  return {
    id: interval.id,
    anchor,
    interval: { start: interval.start, end: interval.end, agreement: interval.agreement },
    nearProbability: probability,
    side,
    verdict: !gate.isServe ? "not-serve" : reviewReasons.length ? "review" : side,
    serveDecisionSource: gate.source,
    reviewReasons,
    serveEvidence: { allLabelsV2: allEvidence, previousProduction: previousEvidence },
  };
}
