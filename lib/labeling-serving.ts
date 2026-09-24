import type { ServeMarker } from "./annotations.ts";

export type ServingPrediction = {
  id: string;
  anchor: number;
  side: "near" | "far";
  verdict: "near" | "far" | "review" | "not-serve";
  nearProbability: number;
  serveDecisionSource: string;
  reviewReasons: string[];
  evidence: { label: string; peakProbability: number; peakTime: number; threshold: number; crossesThreshold: boolean }[];
};

const object = (value: unknown): Record<string, unknown> | null =>
  value !== null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : null;
const probability = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;

export function parseServingPredictions(value: unknown, duration: number): ServingPrediction[] {
  const output = object(value);
  if (!Array.isArray(output?.candidates)) return [];
  const result = output.candidates.map((item): ServingPrediction => {
    const row = object(item);
    if (!row || typeof row.id !== "string" || !row.id || typeof row.anchor !== "number" || !Number.isFinite(row.anchor)
      || row.anchor < 0 || row.anchor > duration || !probability(row.nearProbability)
      || (row.side !== "near" && row.side !== "far")
      || !["near", "far", "review", "not-serve"].includes(String(row.verdict))) {
      throw new Error("Invalid production serving-side candidate");
    }
    const rawEvidence = object(row.serveEvidence);
    const evidence: ServingPrediction["evidence"] = [];
    for (const [key, label] of [["allLabelsV2", "All labels v2"], ["previousProduction", "Previous production"]]) {
      const head = object(rawEvidence?.[key]);
      if (!head) continue;
      if (!probability(head.peakProbability) || !probability(head.threshold) || typeof head.peakTime !== "number"
        || !Number.isFinite(head.peakTime) || head.peakTime < 0 || head.peakTime > duration || typeof head.crossesThreshold !== "boolean") {
        throw new Error("Invalid production serve-head evidence");
      }
      evidence.push({ label, peakProbability: head.peakProbability, peakTime: head.peakTime, threshold: head.threshold, crossesThreshold: head.crossesThreshold });
    }
    return { id: row.id, anchor: row.anchor, side: row.side, verdict: row.verdict as ServingPrediction["verdict"],
      nearProbability: row.nearProbability, serveDecisionSource: typeof row.serveDecisionSource === "string" ? row.serveDecisionSource : "unknown",
      reviewReasons: Array.isArray(row.reviewReasons) ? row.reviewReasons.filter((v): v is string => typeof v === "string") : [], evidence };
  }).sort((a, b) => a.anchor - b.anchor || a.id.localeCompare(b.id));
  if (new Set(result.map(r => r.id)).size !== result.length) throw new Error("Duplicate production serve candidate");
  return result;
}

export function servingDecisionExplanation(row: ServingPrediction): string {
  const reasons = row.reviewReasons.map(reason => reason === "side-score"
    ? "The near/far score is in the model's review band."
    : reason === "production-rally-recovery"
      ? "Both rally models support this candidate, but neither serve head crossed its threshold."
      : reason.replaceAll("-", " "));
  if (row.verdict === "not-serve") return "Neither serve head passed, and this candidate lacks agreement from both rally models. The serve gate rejected it; this does not remove the rally.";
  if (row.serveDecisionSource === "serve-head") reasons.push("At least one production serve head crossed its threshold near the rally start.");
  return reasons.join(" ") || "Production serving-side decision.";
}

export function servingPredictionsToMarkers(rows: ServingPrediction[], modelId: string): ServeMarker[] {
  return rows.filter(row => row.verdict !== "not-serve").map(row => ({
    time: row.anchor, side: row.verdict === "review" ? "review" : row.side,
    origin: "model", modelSide: row.side,
    modelConfidence: row.side === "near" ? row.nearProbability : 1 - row.nearProbability,
    modelId, rallyId: row.id, notes: servingDecisionExplanation(row),
  }));
}
