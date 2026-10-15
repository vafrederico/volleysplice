import pinnedManifest from "../../../../models/serving-side/web-manifest.json" with { type: "json" };
import type { NeuralAsset, NeuralBundle, NeuralRallySelection } from "./rally-model.ts";
import type { ServingSideCandidateVerdict, ServingSideReviewReason, ServingSideSide } from "./types.ts";

export type NeuralServingBundle = {
  id: string;
  fingerprint: string;
  rallyModelId: string;
  directory: string;
  encoderSha256: string;
  encoderWeightsSha256: string;
  file: NeuralAsset;
  seeds: { seed: number; epoch: number; weightsSha256: string; farCutoff: number; nearCutoff: number }[];
};
export const NEURAL_SERVING_MANIFEST = pinnedManifest;
export function neuralServingSelection(modelId: string | undefined): NeuralRallySelection | null {
  for (const selection of ["high-f1", "high-recall"] as const) {
    if (pinnedManifest.variants[selection].rallyModelId === modelId) return selection;
  }
  return null;
}
export function neuralServingBundle(modelId: string | undefined): NeuralServingBundle | null {
  const selection = neuralServingSelection(modelId);
  return selection ? pinnedManifest.variants[selection] : null;
}
export function parseNeuralServingManifest(value: unknown, rally: NeuralBundle): NeuralServingBundle {
  if (JSON.stringify(value) !== JSON.stringify(pinnedManifest)) {
    throw new Error("The serving-side model catalog does not match the frozen testing models.");
  }
  const bundle = neuralServingBundle(rally.id);
  if (!bundle || bundle.encoderSha256 !== rally.files.encoder.sha256) {
    throw new Error("The serving-side model does not match the selected rally encoder.");
  }
  return bundle;
}

/** The exact source-held precision policy: all seeds must accept the same side. */
export function neuralServingDecision(logits: ArrayLike<number>, bundle: NeuralServingBundle): {
  nearProbability: number; side: ServingSideSide; accepted: boolean;
} {
  if (logits.length !== 6 || !Array.from(logits).every(Number.isFinite)) {
    throw new Error("The serving-side model returned invalid scores.");
  }
  const routes: number[] = [], decisions: (ServingSideSide | null)[] = [];
  for (let seed = 0; seed < 3; seed++) {
    const sigmoid = (logit: number) => 1 / (1 + Math.exp(-Math.max(-35, Math.min(35, logit))));
    const far = sigmoid(logits[seed * 2]), near = sigmoid(logits[seed * 2 + 1]);
    const mass = far + near;
    if (mass > 1 + 1e-12 || mass <= 0) throw new Error("Serving-side evidence exceeds its available mass.");
    const route = near / mass;
    const side = route >= .5 ? "near" : "far";
    const confidence = Math.min(mass, 1) * (side === "near" ? route : 1 - route);
    const cutoff = side === "near" ? bundle.seeds[seed].nearCutoff : bundle.seeds[seed].farCutoff;
    routes.push(route);
    decisions.push(confidence > 0 && confidence >= cutoff ? side : null);
  }
  const nearProbability = routes.reduce((sum, value) => sum + value, 0) / 3;
  return { nearProbability, side: nearProbability >= .5 ? "near" : "far",
    accepted: decisions[0] !== null && decisions.every(side => side === decisions[0]) };
}

export function applyNeuralServingDecision(candidate: ServingSideCandidateVerdict,
  decision: ReturnType<typeof neuralServingDecision>): ServingSideCandidateVerdict {
  const reviewReasons: ServingSideReviewReason[] = candidate.reviewReasons.filter(reason => reason !== "side-score");
  if (!decision.accepted) reviewReasons.unshift("side-score");
  return { ...candidate, nearProbability: decision.nearProbability, side: decision.side, reviewReasons,
    verdict: candidate.verdict === "not-serve" ? "not-serve" : reviewReasons.length ? "review" : decision.side };
}

/** Mirrors the training sampler on the global half-second grid; ties choose earlier. */
export function servingEmbeddingWindow(times: Float64Array, tokens: Float32Array, anchor: number, duration: number): Float32Array {
  if (!Number.isFinite(anchor) || anchor < 0 || anchor >= duration || tokens.length !== times.length * 3840
      || !times.length || !times.every((time, index) => Number.isFinite(time) && (index === 0 || time > times[index - 1]))) {
    throw new Error("Invalid serving-side image sequence or anchor.");
  }
  const values = new Float32Array(13 * 3843);
  for (let frame = 0; frame < 13; frame++) {
    const offset = -3 + frame * .5, requested = anchor + offset, destination = frame * 3843;
    values[destination + 3841] = offset / 3;
    if (requested < 0 || requested >= duration) continue;
    let low = 0, high = times.length;
    while (low < high) { const mid = (low + high) >>> 1; if (times[mid] < requested) low = mid + 1; else high = mid; }
    const later = Math.min(low, times.length - 1), earlier = Math.max(0, later - 1);
    const chosen = Math.abs(times[earlier] - requested) <= Math.abs(times[later] - requested) ? earlier : later;
    if (Math.abs(times[chosen] - requested) > .250001) continue;
    values.set(tokens.subarray(chosen * 3840, (chosen + 1) * 3840), destination);
    values[destination + 3840] = 1;
    values[destination + 3842] = (times[chosen] - requested) / .25;
  }
  if (values[6 * 3843 + 3840] !== 1) throw new Error("The serving anchor has no qualified image sample.");
  return values;
}

export function servingEmbeddingTimes(anchors: readonly number[], duration: number): Float64Array {
  const ticks = new Set<number>();
  for (const anchor of anchors) {
    if (!Number.isFinite(anchor) || anchor < 0 || anchor >= duration) throw new Error("Invalid serving anchor.");
    for (let frame = 0; frame < 13; frame++) {
      const requested = anchor - 3 + frame * .5;
      if (requested < 0 || requested >= duration) continue;
      const tick = Math.ceil(requested * 2 - .5);
      if (tick >= 0 && tick / 2 < duration) ticks.add(tick);
    }
  }
  return Float64Array.from([...ticks].sort((a, b) => a - b), tick => tick / 2);
}
