import { runtimeAssetUrl } from "../runtime-assets.ts";
import type {
  OnDeviceInterval,
  OnDeviceSideSwitchOutput,
  ProductionStateOutputs,
  SideSwitchCandidateKind,
} from "./types.ts";

export const SIDE_SWITCH_ASSET = "side-switch-c2570481c30d.json" as const;
export const SIDE_SWITCH_MODEL_ID =
  "side-switch-hard-negative-mining-v1/union34-top2-x2";
export const SIDE_SWITCH_MODEL_FINGERPRINT =
  "sha256:c2570481c30dec62f56ac3284cd4028763ffd8e72a9ad07e9908fec10df387e3";
export const SIDE_SWITCH_FEATURE_VERSION = "SIDE-SWITCH-UNION34-V1" as const;
export const SIDE_SWITCH_CANDIDATE_CONTRACT =
  "range-boundaries-dead-peaks-v1" as const;

export const SIDE_SWITCH_VISUAL_FEATURE_NAMES = [
  "v4BroadSameAssignmentCost",
  "v4TightSameAssignmentCost",
  "v4MeanSwapMargin",
  "v4GlobalAppearanceChange",
  "v4MaximumCameraShift",
  "v4MinimumAlignmentResponse",
  "playerSameAssignmentCost",
  "playerSwappedAssignmentCost",
  "playerSwapMargin",
  "playerOrientationFlipEvidence",
  "minimumPlayerSideSeparation",
  "playerSideSeparationChange",
  "beforePlayerPaletteInstability",
  "afterPlayerPaletteInstability",
  "playerGlobalAppearanceChange",
  "minimumProposalCoverage",
  "proposalCoverageChange",
  "minimumProposalCount",
  "proposalCountChange",
  "minimumNearSupport",
  "minimumFarSupport",
  "sideSupportImbalanceChange",
] as const;

export const SIDE_SWITCH_STATE_FEATURE_NAMES = [
  "productionBeforeSupportCount",
  "productionAfterSupportCount",
  "productionMinimumAdjacentSupportCount",
  "productionMinimumAdjacentRallyPeak",
  "productionGapLiveFraction",
  "productionGapMeanRallyScore",
  "productionGapPeakRallyScore",
  "productionGapMeanDeadStateScore",
  "productionGapPeakDeadStateScore",
  "productionGapDurationSeconds",
] as const;

export const SIDE_SWITCH_FEATURE_NAMES = [
  ...SIDE_SWITCH_VISUAL_FEATURE_NAMES,
  ...SIDE_SWITCH_STATE_FEATURE_NAMES,
  "candidateIsInternalDeadStatePeak",
  "candidateGeneratorScore",
] as const;

type SideSwitchClassifier = {
  featureNames: string[];
  impute: number[];
  mean: number[];
  scale: number[];
  weights: number[];
  bias: number;
  threshold: number;
};

export type SideSwitchRuntimeModel = {
  modelId: string;
  fingerprint: string;
  featureVersion: typeof SIDE_SWITCH_FEATURE_VERSION;
  classifier: SideSwitchClassifier;
  candidateGenerator: {
    internalPeakThreshold: number;
    internalPeakMinimumSeparationSeconds: number;
    internalPeakRangeEdgeExclusionSeconds: number;
    internalPeakProposalHalfWidthSeconds: number;
  };
  decoder: {
    minimumCandidateIndexSeparation: number;
    minimumTimeSeparationSeconds: number;
    freePredictionsPerRecording: number;
    countPenaltyLogitPerExcessPrediction: number;
  };
};

export type SideSwitchCandidateProposal = {
  id: string;
  kind: SideSwitchCandidateKind;
  gapStart: number;
  gapEnd: number;
  transitionTime: number;
  generatorScore: number;
  sourceRangeIds: string[];
  beforeWindow: { start: number; end: number };
  afterWindow: { start: number; end: number };
};

export type SideSwitchAnalysisInput = {
  intervals: readonly OnDeviceInterval[];
  times: ArrayLike<number>;
  deadStateProbabilities: ArrayLike<number>;
  productionComponents: {
    allLabelsV2: readonly OnDeviceInterval[];
    previousProduction: readonly OnDeviceInterval[];
  };
  productionStateOutputs: ProductionStateOutputs;
};

let runtimePromise: Promise<SideSwitchRuntimeModel> | null = null;

function finiteArray(value: unknown, length: number): value is number[] {
  return (
    Array.isArray(value) &&
    value.length === length &&
    value.every((item) => typeof item === "number" && Number.isFinite(item))
  );
}

export function parseSideSwitchRuntime(value: unknown): SideSwitchRuntimeModel {
  if (!value || typeof value !== "object")
    throw new Error("Side-switch runtime is malformed.");
  const runtime = value as Record<string, unknown>;
  const classifier = runtime.classifier as Record<string, unknown> | undefined;
  const generator = runtime.candidateGenerator as
    | Record<string, unknown>
    | undefined;
  const decoder = runtime.decoder as Record<string, unknown> | undefined;
  const columns = SIDE_SWITCH_FEATURE_NAMES.length;
  if (
    runtime.modelId !== SIDE_SWITCH_MODEL_ID ||
    runtime.fingerprint !== SIDE_SWITCH_MODEL_FINGERPRINT ||
    runtime.featureVersion !== SIDE_SWITCH_FEATURE_VERSION ||
    !classifier ||
    !generator ||
    !decoder ||
    !Array.isArray(classifier.featureNames) ||
    classifier.featureNames.length !== columns ||
    classifier.featureNames.some(
      (name, index) => name !== SIDE_SWITCH_FEATURE_NAMES[index],
    ) ||
    !finiteArray(classifier.impute, columns) ||
    !finiteArray(classifier.mean, columns) ||
    !finiteArray(classifier.scale, columns) ||
    classifier.scale.some((item) => item <= 0) ||
    !finiteArray(classifier.weights, columns) ||
    typeof classifier.bias !== "number" ||
    !Number.isFinite(classifier.bias) ||
    typeof classifier.threshold !== "number" ||
    classifier.threshold < 0 ||
    classifier.threshold > 1
  )
    throw new Error("Side-switch classifier identity or parameters changed.");
  const numeric = (source: Record<string, unknown>, name: string) => {
    const item = source[name];
    if (typeof item !== "number" || !Number.isFinite(item)) {
      throw new Error(`Side-switch runtime field ${name} is invalid.`);
    }
    return item;
  };
  const parsed: SideSwitchRuntimeModel = {
    modelId: runtime.modelId,
    fingerprint: runtime.fingerprint,
    featureVersion: runtime.featureVersion,
    classifier: classifier as unknown as SideSwitchClassifier,
    candidateGenerator: {
      internalPeakThreshold: numeric(generator, "internalPeakThreshold"),
      internalPeakMinimumSeparationSeconds: numeric(
        generator,
        "internalPeakMinimumSeparationSeconds",
      ),
      internalPeakRangeEdgeExclusionSeconds: numeric(
        generator,
        "internalPeakRangeEdgeExclusionSeconds",
      ),
      internalPeakProposalHalfWidthSeconds: numeric(
        generator,
        "internalPeakProposalHalfWidthSeconds",
      ),
    },
    decoder: {
      minimumCandidateIndexSeparation: numeric(
        decoder,
        "minimumCandidateIndexSeparation",
      ),
      minimumTimeSeparationSeconds: numeric(
        decoder,
        "minimumTimeSeparationSeconds",
      ),
      freePredictionsPerRecording: numeric(
        decoder,
        "freePredictionsPerRecording",
      ),
      countPenaltyLogitPerExcessPrediction: numeric(
        decoder,
        "countPenaltyLogitPerExcessPrediction",
      ),
    },
  };
  return parsed;
}

export function loadSideSwitchRuntime(): Promise<SideSwitchRuntimeModel> {
  runtimePromise ??= fetch(runtimeAssetUrl(SIDE_SWITCH_ASSET))
    .then(async (response) => {
      if (!response.ok)
        throw new Error(
          `Could not load side-switch model (${response.status}).`,
        );
      return parseSideSwitchRuntime(await response.json());
    })
    .catch((cause) => {
      runtimePromise = null;
      throw cause;
    });
  return runtimePromise;
}

function orderedRanges(
  source: readonly OnDeviceInterval[],
): OnDeviceInterval[] {
  const result = source
    .filter(
      (range) =>
        range.included &&
        Number.isFinite(range.start) &&
        Number.isFinite(range.end) &&
        range.end > range.start,
    )
    .map((range) => ({ ...range }))
    .sort(
      (left, right) =>
        left.start - right.start ||
        left.end - right.end ||
        left.id.localeCompare(right.id),
    );
  if (new Set(result.map((range) => range.id)).size !== result.length) {
    throw new Error("Side-switch inference requires stable unique range IDs.");
  }
  return result;
}

function sampleTimes(start: number, end: number): number[] {
  const duration = end - start;
  let first = start + Math.min(0.2, duration * 0.08);
  let last = end - Math.min(0.15, duration * 0.08);
  if (last <= first) {
    first = start + duration * 0.2;
    last = start + duration * 0.8;
  }
  return Array.from(
    { length: 7 },
    (_, index) => first + ((last - first) * index) / 6,
  );
}

export function sideSwitchCalibrationTimes(
  intervals: readonly OnDeviceInterval[],
): number[] {
  return orderedRanges(intervals)
    .slice(0, 7)
    .flatMap((range) => sampleTimes(range.start, range.end));
}

export function generateSideSwitchCandidates(
  input: Pick<
    SideSwitchAnalysisInput,
    "intervals" | "times" | "deadStateProbabilities"
  >,
  runtime: SideSwitchRuntimeModel,
): SideSwitchCandidateProposal[] {
  const ranges = orderedRanges(input.intervals);
  const times = Array.from(input.times);
  const dead = Array.from(input.deadStateProbabilities);
  if (
    times.length === 0 ||
    times.length !== dead.length ||
    times.some(
      (time, index) =>
        !Number.isFinite(time) || (index > 0 && time <= times[index - 1]),
    ) ||
    dead.some((score) => !Number.isFinite(score) || score < 0 || score > 1)
  ) {
    throw new Error("Side-switch probability traces are invalid.");
  }
  const candidates: SideSwitchCandidateProposal[] = [];
  for (let index = 0; index + 1 < ranges.length; index += 1) {
    const before = ranges[index];
    const after = ranges[index + 1];
    if (after.start < before.end)
      throw new Error("Side-switch ranges overlap after production merge.");
    candidates.push({
      id: `switch:boundary:${before.id}:${after.id}`,
      kind: "adjacent-rally-boundary",
      gapStart: before.end,
      gapEnd: after.start,
      transitionTime: (before.end + after.start) / 2,
      generatorScore: 0,
      sourceRangeIds: [before.id, after.id],
      beforeWindow: { start: before.start, end: before.end },
      afterWindow: { start: after.start, end: after.end },
    });
  }
  const config = runtime.candidateGenerator;
  for (const range of ranges) {
    const eligible = times
      .map((time, index) => ({ time, index }))
      .filter(
        ({ time, index }) =>
          time >= range.start + config.internalPeakRangeEdgeExclusionSeconds &&
          time <= range.end - config.internalPeakRangeEdgeExclusionSeconds &&
          dead[index] >= config.internalPeakThreshold,
      )
      .sort(
        (left, right) =>
          dead[right.index] - dead[left.index] ||
          left.time - right.time ||
          left.index - right.index,
      );
    const selected: typeof eligible = [];
    for (const candidate of eligible) {
      if (
        selected.every(
          (other) =>
            Math.abs(candidate.time - other.time) >=
            config.internalPeakMinimumSeparationSeconds,
        )
      ) {
        selected.push(candidate);
      }
    }
    selected.sort((left, right) => left.time - right.time);
    for (const candidate of selected) {
      candidates.push({
        id: `switch:internal-dead-peak:${range.id}:${Math.round(candidate.time * 1000)}`,
        kind: "internal-dead-state-peak",
        gapStart: Math.max(
          range.start,
          candidate.time - config.internalPeakProposalHalfWidthSeconds,
        ),
        gapEnd: Math.min(
          range.end,
          candidate.time + config.internalPeakProposalHalfWidthSeconds,
        ),
        transitionTime: candidate.time,
        generatorScore: dead[candidate.index],
        sourceRangeIds: [range.id],
        beforeWindow: { start: candidate.time - 4, end: candidate.time - 1 },
        afterWindow: { start: candidate.time + 1, end: candidate.time + 4 },
      });
    }
  }
  return candidates.sort(
    (left, right) =>
      left.transitionTime - right.transitionTime ||
      left.kind.localeCompare(right.kind) ||
      left.id.localeCompare(right.id),
  );
}

export function sideSwitchCandidateSampleTimes(
  candidate: SideSwitchCandidateProposal,
): number[] {
  return [
    ...sampleTimes(candidate.beforeWindow.start, candidate.beforeWindow.end),
    ...sampleTimes(candidate.afterWindow.start, candidate.afterWindow.end),
  ];
}

function overlap(
  start: number,
  end: number,
  otherStart: number,
  otherEnd: number,
): number {
  return Math.max(0, Math.min(end, otherEnd) - Math.max(start, otherStart));
}

function valuesInWindow(
  times: readonly number[],
  values: ArrayLike<number>,
  start: number,
  end: number,
): number[] {
  const selected = times.flatMap((time, index) =>
    time >= start && time < end ? [values[index]] : [],
  );
  if (selected.length > 0) return selected;
  const center = (start + end) / 2;
  let nearest = 0;
  for (let index = 1; index < times.length; index += 1) {
    if (Math.abs(times[index] - center) < Math.abs(times[nearest] - center))
      nearest = index;
  }
  return [values[nearest]];
}

function unionDuration(
  start: number,
  end: number,
  ranges: readonly OnDeviceInterval[],
): number {
  const clipped = ranges
    .flatMap((range) => {
      const left = Math.max(start, range.start);
      const right = Math.min(end, range.end);
      return right > left ? [[left, right] as const] : [];
    })
    .sort((left, right) => left[0] - right[0] || left[1] - right[1]);
  if (clipped.length === 0) return 0;
  let total = 0;
  let activeStart = clipped[0][0];
  let activeEnd = clipped[0][1];
  for (const [left, right] of clipped.slice(1)) {
    if (left <= activeEnd) activeEnd = Math.max(activeEnd, right);
    else {
      total += activeEnd - activeStart;
      activeStart = left;
      activeEnd = right;
    }
  }
  return total + activeEnd - activeStart;
}

function evidence(
  range: OnDeviceInterval,
  input: SideSwitchAnalysisInput,
): { support: number; rallyPeak: number } {
  const sources = (["allLabelsV2", "previousProduction"] as const).filter(
    (source) =>
      input.productionComponents[source].some(
        (item) => overlap(range.start, range.end, item.start, item.end) > 0,
      ),
  );
  const times = Array.from(input.times);
  const peaks = sources.map((source) =>
    Math.max(
      ...valuesInWindow(
        times,
        input.productionStateOutputs[source].rallyProbabilities,
        range.start,
        range.end,
      ),
    ),
  );
  return {
    support: sources.length,
    rallyPeak: peaks.length > 0 ? Math.min(...peaks) : 0,
  };
}

export function sideSwitchStateFeatures(
  candidate: SideSwitchCandidateProposal,
  input: SideSwitchAnalysisInput,
): Float64Array {
  const ranges = orderedRanges(input.intervals);
  const byId = new Map(ranges.map((range) => [range.id, range]));
  const beforeRange = byId.get(candidate.sourceRangeIds[0]);
  const afterRange = byId.get(candidate.sourceRangeIds.at(-1)!);
  if (!beforeRange || !afterRange)
    throw new Error("Side-switch candidate lost its source range.");
  const before = evidence(beforeRange, input);
  const after = evidence(afterRange, input);
  const times = Array.from(input.times);
  const maximum = (left: ArrayLike<number>, right: ArrayLike<number>) =>
    Float64Array.from({ length: times.length }, (_, index) =>
      Math.max(left[index], right[index]),
    );
  const rally = maximum(
    input.productionStateOutputs.allLabelsV2.rallyProbabilities,
    input.productionStateOutputs.previousProduction.rallyProbabilities,
  );
  const dead = maximum(
    input.productionStateOutputs.allLabelsV2.deadStateProbabilities,
    input.productionStateOutputs.previousProduction.deadStateProbabilities,
  );
  const rallyWindow = valuesInWindow(
    times,
    rally,
    candidate.gapStart,
    candidate.gapEnd,
  );
  const deadWindow = valuesInWindow(
    times,
    dead,
    candidate.gapStart,
    candidate.gapEnd,
  );
  const duration = Math.max(candidate.gapEnd - candidate.gapStart, 1e-6);
  const sourceRanges = [
    ...input.productionComponents.allLabelsV2,
    ...input.productionComponents.previousProduction,
  ];
  const mean = (values: readonly number[]) =>
    values.reduce((sum, value) => sum + value, 0) / values.length;
  return Float64Array.of(
    before.support,
    after.support,
    Math.min(before.support, after.support),
    Math.min(before.rallyPeak, after.rallyPeak),
    unionDuration(candidate.gapStart, candidate.gapEnd, sourceRanges) /
      duration,
    mean(rallyWindow),
    Math.max(...rallyWindow),
    mean(deadWindow),
    Math.max(...deadWindow),
    candidate.gapEnd - candidate.gapStart,
  );
}

export function predictSideSwitchProbabilities(
  runtime: SideSwitchRuntimeModel,
  features: Float64Array,
): Float64Array {
  const columns = runtime.classifier.featureNames.length;
  if (features.length % columns !== 0)
    throw new Error("Side-switch feature matrix is ragged.");
  const rows = features.length / columns;
  const probabilities = new Float64Array(rows);
  for (let row = 0; row < rows; row += 1) {
    let logit = runtime.classifier.bias;
    for (let column = 0; column < columns; column += 1) {
      const raw = features[row * columns + column];
      const filled = Number.isFinite(raw)
        ? raw
        : runtime.classifier.impute[column];
      logit +=
        ((filled - runtime.classifier.mean[column]) /
          runtime.classifier.scale[column]) *
        runtime.classifier.weights[column];
    }
    logit = Math.max(-30, Math.min(30, logit));
    probabilities[row] = 1 / (1 + Math.exp(-logit));
  }
  return probabilities;
}

function logit(value: number): number {
  const clipped = Math.max(1e-9, Math.min(1 - 1e-9, value));
  return Math.log(clipped / (1 - clipped));
}

export function decodeSideSwitchCandidates(
  runtime: SideSwitchRuntimeModel,
  candidates: readonly SideSwitchCandidateProposal[],
  probabilities: ArrayLike<number>,
): number[] {
  if (probabilities.length !== candidates.length)
    throw new Error("Side-switch scores are misaligned.");
  const chronological = candidates
    .map((_, index) => index)
    .sort(
      (left, right) =>
        candidates[left].transitionTime - candidates[right].transitionTime ||
        candidates[left].id.localeCompare(candidates[right].id),
    );
  const ordinal = new Map(chronological.map((index, order) => [index, order]));
  const ranked = candidates
    .map((_, index) => index)
    .sort(
      (left, right) =>
        probabilities[right] - probabilities[left] ||
        candidates[left].transitionTime - candidates[right].transitionTime ||
        candidates[left].id.localeCompare(candidates[right].id),
    );
  const selected: number[] = [];
  const thresholdLogit = logit(runtime.classifier.threshold);
  for (const index of ranked) {
    const score = probabilities[index];
    if (!Number.isFinite(score) || score < 0 || score > 1)
      throw new Error("Side-switch score is invalid.");
    if (
      selected.some(
        (other) =>
          (runtime.decoder.minimumCandidateIndexSeparation > 0 &&
            Math.abs(ordinal.get(index)! - ordinal.get(other)!) <
              runtime.decoder.minimumCandidateIndexSeparation) ||
          (runtime.decoder.minimumTimeSeparationSeconds > 0 &&
            Math.abs(
              candidates[index].transitionTime -
                candidates[other].transitionTime,
            ) < runtime.decoder.minimumTimeSeparationSeconds),
      )
    )
      continue;
    const excess = Math.max(
      0,
      selected.length + 1 - runtime.decoder.freePredictionsPerRecording,
    );
    const margin =
      logit(score) -
      thresholdLogit -
      runtime.decoder.countPenaltyLogitPerExcessPrediction * excess;
    if (margin >= -1e-12) selected.push(index);
  }
  return selected.sort(
    (left, right) =>
      candidates[left].transitionTime - candidates[right].transitionTime ||
      candidates[left].id.localeCompare(candidates[right].id),
  );
}

export function emptySideSwitchOutput(
  runtime: SideSwitchRuntimeModel,
): OnDeviceSideSwitchOutput {
  return {
    modelId: runtime.modelId,
    modelFingerprint: runtime.fingerprint,
    featureVersion: SIDE_SWITCH_FEATURE_VERSION,
    candidateContract: SIDE_SWITCH_CANDIDATE_CONTRACT,
    features: {
      rows: 0,
      columns: SIDE_SWITCH_FEATURE_NAMES.length,
      values: new Float64Array(0),
    },
    candidates: [],
  };
}

export function isReusableSideSwitchOutput(
  value: unknown,
): value is OnDeviceSideSwitchOutput {
  if (!value || typeof value !== "object") return false;
  const output = value as Partial<OnDeviceSideSwitchOutput>;
  const candidates = Array.isArray(output.candidates) ? output.candidates : [];
  return (
    output.modelId === SIDE_SWITCH_MODEL_ID &&
    output.modelFingerprint === SIDE_SWITCH_MODEL_FINGERPRINT &&
    output.featureVersion === SIDE_SWITCH_FEATURE_VERSION &&
    output.candidateContract === SIDE_SWITCH_CANDIDATE_CONTRACT &&
    Boolean(output.features) &&
    output.features!.columns === SIDE_SWITCH_FEATURE_NAMES.length &&
    Number.isInteger(output.features!.rows) &&
    output.features!.rows >= 0 &&
    output.features!.values instanceof Float64Array &&
    output.features!.values.length ===
      output.features!.rows * output.features!.columns &&
    !output.features!.values.some((item) => !Number.isFinite(item)) &&
    candidates.length <= output.features!.rows &&
    new Set(candidates.map((candidate) => candidate.id)).size ===
      candidates.length &&
    candidates.every(
      (candidate) =>
        typeof candidate.id === "string" &&
        candidate.id.length > 0 &&
        Number.isFinite(candidate.timestamp) &&
        candidate.timestamp >= 0 &&
        Number.isFinite(candidate.probability) &&
        candidate.probability >= 0 &&
        candidate.probability <= 1 &&
        (candidate.kind === "adjacent-rally-boundary" ||
          candidate.kind === "internal-dead-state-peak") &&
        Array.isArray(candidate.sourceRangeIds) &&
        candidate.sourceRangeIds.every(
          (id) => typeof id === "string" && id.length > 0,
        ),
    )
  );
}
