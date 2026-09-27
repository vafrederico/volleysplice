import type { OnDeviceServingSideOutput, ServingSideCandidateVerdict } from "./types.ts";

/** Legacy serve heads may miss a neural rally's start; retain a reviewable marker. */
export function retainNeuralServeCandidate(
  candidate: ServingSideCandidateVerdict,
): ServingSideCandidateVerdict {
  if (candidate.verdict !== "not-serve" || candidate.interval.agreement !== "neural") return candidate;
  return {
    ...candidate,
    verdict: "review",
    serveDecisionSource: "neural-rally-recovery",
    reviewReasons: [...new Set([...candidate.reviewReasons, "neural-rally-recovery" as const])],
  };
}

/** Reuses saved side predictions and features without decoding the video again. */
export function retainNeuralServeCandidates(output: OnDeviceServingSideOutput): OnDeviceServingSideOutput {
  const candidates = output.candidates.map(retainNeuralServeCandidate);
  return candidates.some((candidate, index) => candidate !== output.candidates[index])
    ? { ...output, candidates } : output;
}
