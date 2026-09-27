import {
  decodeProbabilities,
  predictLogistic,
  type DecodedInterval,
  type LogisticHead,
  type ProbabilityDecoderConfig,
} from "./model.ts";

export const SUPPRESSION_MODEL_ID =
  "suppression-overlap-exclusion-retrained" as const;
export const SUPPRESSION_ARTIFACT_SHA256 =
  "39eddf58163901930434ea422a802686ae921ae1e8fe23c20a3c12e5f453da93" as const;
export const SUPPRESSION_WEIGHTS_SHA256 =
  "a943749b69c98a1bc926f8efc9fe60c67c1226534fe09a892c632519217aa3bb" as const;
export const SUPPRESSION_DECODER_VERSION =
  "held-production-suppression-decoder-v1" as const;
export const SUPPRESSION_BROWSER_ASSET_SHA256 =
  "ef0ad4eb93fa61ce1d403f083d91f7578cf9ff0f31fac797fde9ab8b73f42794" as const;

export const HELD_SUPPRESSION_DECODER: ProbabilityDecoderConfig = {
  smoothing_seconds: 1,
  enter_threshold: 0.75,
  exit_threshold: 0.65,
  min_live_seconds: 0.5,
  bridge_gap_seconds: 0.5,
  short_event_min_seconds: 0.25,
  short_event_threshold: 0.9,
};

export type SuppressionModelBundle = {
  schemaVersion: 1;
  modelId: typeof SUPPRESSION_MODEL_ID;
  artifactSha256: typeof SUPPRESSION_ARTIFACT_SHA256;
  weightsSha256: typeof SUPPRESSION_WEIGHTS_SHA256;
  decoderVersion: typeof SUPPRESSION_DECODER_VERSION;
  analysisFps: number;
  featureNames: readonly string[];
  head: LogisticHead & { decoder: ProbabilityDecoderConfig };
};

export type SuppressionInferenceResult = {
  probabilities: Float32Array;
  intervals: DecodedInterval[];
};

type UnknownRecord = Record<string, unknown>;

function objectValue(value: unknown, label: string): UnknownRecord {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value as UnknownRecord;
}

function finite(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be finite.`);
  }
  return value;
}

function vector(value: unknown, length: number, label: string): Float32Array {
  if (
    !Array.isArray(value) ||
    value.length !== length ||
    value.some((item) => typeof item !== "number" || !Number.isFinite(item))
  ) {
    throw new Error(`${label} must contain ${length} finite values.`);
  }
  return Float32Array.from(value as number[]);
}

function decoderMatches(value: unknown): value is ProbabilityDecoderConfig {
  if (!value || typeof value !== "object") return false;
  const decoder = value as Partial<ProbabilityDecoderConfig>;
  return Object.entries(HELD_SUPPRESSION_DECODER).every(
    ([key, expected]) => decoder[key as keyof ProbabilityDecoderConfig] === expected,
  );
}

export function loadSuppressionModelBundle(value: unknown): SuppressionModelBundle {
  const object = objectValue(value, "suppression model");
  if (
    object.schemaVersion !== 1 ||
    object.modelId !== SUPPRESSION_MODEL_ID ||
    object.artifactSha256 !== SUPPRESSION_ARTIFACT_SHA256 ||
    object.weightsSha256 !== SUPPRESSION_WEIGHTS_SHA256 ||
    object.decoderVersion !== SUPPRESSION_DECODER_VERSION
  ) {
    throw new Error("Suppression model identity does not match the held product artifact.");
  }
  if (
    !Array.isArray(object.featureNames) ||
    object.featureNames.length === 0 ||
    object.featureNames.some((name) => typeof name !== "string" || name.length === 0) ||
    new Set(object.featureNames).size !== object.featureNames.length
  ) {
    throw new Error("Suppression feature signature is invalid.");
  }
  const featureNames = [...object.featureNames] as string[];
  const sourceHead = objectValue(object.head, "suppression head");
  if (!decoderMatches(sourceHead.decoder)) {
    throw new Error("Suppression model does not contain the held decoder constants.");
  }
  const head = {
    mean: vector(sourceHead.mean, featureNames.length, "suppression mean"),
    scale: vector(sourceHead.scale, featureNames.length, "suppression scale"),
    weights: vector(sourceHead.weights, featureNames.length, "suppression weights"),
    bias: finite(sourceHead.bias, "suppression bias"),
    decoder: { ...HELD_SUPPRESSION_DECODER },
  };
  if (head.scale.some((scale) => !(scale > 0))) {
    throw new Error("Suppression normalization scales must be positive.");
  }
  const analysisFps = finite(object.analysisFps, "suppression analysisFps");
  if (!(analysisFps > 0)) throw new Error("Suppression analysisFps must be positive.");
  return {
    schemaVersion: 1,
    modelId: SUPPRESSION_MODEL_ID,
    artifactSha256: SUPPRESSION_ARTIFACT_SHA256,
    weightsSha256: SUPPRESSION_WEIGHTS_SHA256,
    decoderVersion: SUPPRESSION_DECODER_VERSION,
    analysisFps,
    featureNames,
    head,
  };
}

export function runSuppressionModel(
  bundle: SuppressionModelBundle,
  times: ArrayLike<number>,
  contextualFeatures: ArrayLike<number>,
  duration: number,
): SuppressionInferenceResult {
  const probabilities = predictLogistic(bundle.head, contextualFeatures, times.length);
  const decoded = decodeProbabilities(
    times,
    probabilities,
    duration,
    bundle.head.decoder,
    bundle.analysisFps,
  );
  return { probabilities, intervals: decoded.intervals };
}
