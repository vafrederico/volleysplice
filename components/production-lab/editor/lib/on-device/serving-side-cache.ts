import type { OnDeviceInterval, OnDeviceServingSideOutput } from "./types.ts";

export const SERVING_SIDE_MODEL_ID = "serving-side-fixed-flight-v3";
export const SERVING_SIDE_MODEL_FINGERPRINT =
  "85bc3325fbd43abba6ba3726ac091dc6a3a1eafc68d63d979cb2a7450eb49e06";
export const SERVING_SIDE_FEATURE_VERSION = "SERVSIDE237-FLIGHT" as const;
export const SERVING_SIDE_ANCHOR_CONTRACT =
  "merged-production-interval-start-v1" as const;
export const SERVING_SIDE_FEATURE_COLUMNS = 237;

/**
 * Confirms that a stored serving-side result belongs to the current frozen
 * model and the exact production candidate anchors. Ignored editor ranges are
 * intentionally absent from this identity: they filter results after inference.
 */
export function isReusableServingSideOutput(
  value: unknown,
  intervals: readonly OnDeviceInterval[],
): value is OnDeviceServingSideOutput {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const output = value as Partial<OnDeviceServingSideOutput>;
  if (
    output.modelId !== SERVING_SIDE_MODEL_ID ||
    output.modelFingerprint !== SERVING_SIDE_MODEL_FINGERPRINT ||
    output.featureVersion !== SERVING_SIDE_FEATURE_VERSION ||
    output.anchorContract !== SERVING_SIDE_ANCHOR_CONTRACT ||
    !output.features ||
    output.features.rows !== output.candidates?.length ||
    output.features.columns !== SERVING_SIDE_FEATURE_COLUMNS ||
    !(output.features.values instanceof Float64Array) ||
    output.features.values.length !== output.features.rows * output.features.columns ||
    !Array.isArray(output.candidates)
  ) return false;

  const expected = intervals
    .filter((interval) =>
      interval.included &&
      Number.isFinite(interval.start) &&
      Number.isFinite(interval.end) &&
      interval.start >= 0 &&
      interval.end > interval.start
    )
    .sort((left, right) =>
      left.start - right.start || left.end - right.end || left.id.localeCompare(right.id)
    );
  if (expected.length !== output.candidates.length) return false;
  const ids = new Set<string>();
  return output.candidates.every((candidate, index) => {
    const interval = expected[index];
    const valid =
      candidate &&
      typeof candidate.id === "string" &&
      candidate.id.length > 0 &&
      !ids.has(candidate.id) &&
      candidate.id === interval.id &&
      candidate.anchor === interval.start &&
      candidate.interval?.start === interval.start &&
      candidate.interval?.end === interval.end &&
      candidate.interval?.agreement === interval.agreement &&
      Number.isFinite(candidate.nearProbability) &&
      candidate.nearProbability >= 0 &&
      candidate.nearProbability <= 1 &&
      (candidate.side === "near" || candidate.side === "far") &&
      (candidate.verdict === "near" ||
        candidate.verdict === "far" ||
        candidate.verdict === "review" ||
        candidate.verdict === "not-serve");
    if (valid) ids.add(candidate.id);
    return valid;
  });
}
