import type { OnDeviceInterval } from "./types";

export const ALL_LABELS_V2_MODEL_ID = "model-1ca43e38eefc";
export const PREVIOUS_PRODUCTION_MODEL_ID = "model-9c92b8e9333f";
export const PRODUCTION_ENSEMBLE_MODEL_ID =
  "ensemble-1ca43e38eefc-9c92b8e9333f";

const DISAGREEMENT_CONFIDENCE_CEILING = 0.49;
const DISAGREEMENT_CONFIDENCE_SCALE = 0.6;

type Candidate = Omit<OnDeviceInterval, "id" | "agreement"> & {
  source: "all-labels-v2" | "previous-production";
};

/**
 * Unions overlapping ranges from the current and previous production models.
 * Single-model ranges remain export candidates, but receive a deliberately
 * conservative review confidence so they are surfaced by the editor.
 */
export function mergeProductionModelIntervals(
  allLabelsV2: readonly OnDeviceInterval[],
  previousProduction: readonly OnDeviceInterval[],
): OnDeviceInterval[] {
  const candidates: Candidate[] = [
    ...allLabelsV2.map((interval) => ({ ...interval, source: "all-labels-v2" as const })),
    ...previousProduction.map((interval) => ({
      ...interval,
      source: "previous-production" as const,
    })),
  ]
    .filter((interval) => interval.included && interval.end > interval.start)
    .sort((left, right) => left.start - right.start || left.end - right.end);

  const clusters = candidates.reduce<Candidate[][]>((result, candidate) => {
    const current = result.at(-1);
    const currentEnd = current
      ? Math.max(...current.map((interval) => interval.end))
      : Number.NEGATIVE_INFINITY;
    if (!current || candidate.start >= currentEnd) result.push([candidate]);
    else current.push(candidate);
    return result;
  }, []);

  return clusters.map((cluster, index) => {
    const sources = new Set(cluster.map((candidate) => candidate.source));
    const sourceConfidences = [...sources].map((source) =>
      Math.max(
        ...cluster
          .filter((candidate) => candidate.source === source)
          .map((candidate) => candidate.confidence),
      ),
    );
    const rawConfidence = sourceConfidences.reduce(
      (sum, confidence) => sum + confidence,
      0,
    ) / sourceConfidences.length;
    const agreement = sources.size === 2
      ? "both-models" as const
      : sources.has("all-labels-v2")
        ? "all-labels-v2-only" as const
        : "previous-production-only" as const;
    const confidence = agreement === "both-models"
      ? rawConfidence
      : Math.min(
          DISAGREEMENT_CONFIDENCE_CEILING,
          rawConfidence * DISAGREEMENT_CONFIDENCE_SCALE,
        );
    return {
      id: `R${String(index + 1).padStart(3, "0")}`,
      start: Math.min(...cluster.map((candidate) => candidate.start)),
      end: Math.max(...cluster.map((candidate) => candidate.end)),
      confidence,
      included: true,
      agreement,
    };
  });
}
