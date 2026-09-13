import { alignRallyServeMarkers, createCutDraft, type CutDraft, type CutDraftSeed } from "./cut-draft.ts";
import type { OnDeviceAnalysis } from "./on-device/types.ts";
import { scoreTrackingWithServingSideOutput, scoreTrackingWithSideSwitchOutput } from "./score-tracking-inference.ts";

type ModelScoreResults = Pick<OnDeviceAnalysis, "servingSide" | "sideSwitch">;

export function withModelScoreMarkers(draft: CutDraft, analysis: ModelScoreResults): CutDraft {
  let scoreTracking = draft.scoreTracking;
  if (analysis.servingSide) {
    const hadServeMarkers = scoreTracking.serveMarkers.length > 0;
    const generated = scoreTrackingWithServingSideOutput(scoreTracking, analysis.servingSide);
    scoreTracking = !hadServeMarkers && generated.serveMarkers.length > 0
      ? { ...generated, enabled: true } : generated;
  }
  if (analysis.sideSwitch) scoreTracking = scoreTrackingWithSideSwitchOutput(scoreTracking, analysis.sideSwitch);
  return alignRallyServeMarkers({ ...draft, scoreTracking });
}

// Rebuild from immutable inference outputs, never the edited/imported review draft.
export function createModelReviewDraft(seed: CutDraftSeed, analysis: ModelScoreResults): CutDraft {
  return withModelScoreMarkers(createCutDraft(seed), analysis);
}
