import type { ModelAgreement, Rally } from "./edit-list.ts";

const DISAGREEMENT_CONFIDENCE_CEILING = 0.49;
const DISAGREEMENT_CONFIDENCE_SCALE = 0.6;

type Candidate = Omit<Rally, "id" | "agreement"> & {
  source: "all-labels-v2" | "previous-production";
};

/** Mirrors the production app's overlap-union-disagreement-v1 merge. */
export function mergeProductionModelIntervals(
  allLabelsV2: readonly Rally[],
  previousProduction: readonly Rally[],
): Rally[] {
  const candidates: Candidate[] = [
    ...allLabelsV2.map((interval) => ({
      ...interval,
      source: "all-labels-v2" as const,
    })),
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
    const agreement: ModelAgreement = sources.size === 2
      ? "both-models"
      : sources.has("all-labels-v2")
        ? "all-labels-v2-only"
        : "previous-production-only";
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

export function isProductionModelDisagreement(
  value: Pick<Rally, "agreement">,
): boolean {
  return value.agreement === "all-labels-v2-only" ||
    value.agreement === "previous-production-only";
}

export function productionModelAgreementLabel(
  agreement: ModelAgreement | undefined,
): string {
  if (agreement === "both-models") return "Both production models agree";
  if (agreement === "all-labels-v2-only") return "Disagreement · all-labels v2 only";
  if (agreement === "previous-production-only") {
    return "Disagreement · previous production only";
  }
  return "Model prediction";
}
