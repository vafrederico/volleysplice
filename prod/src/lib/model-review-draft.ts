import {
  alignRallyServeMarkers,
  createCutDraft,
  type CutDraft,
  type CutDraftSeed,
} from "./cut-draft.ts";
import type { OnDeviceAnalysis } from "./on-device/types.ts";
import {
  scoreTrackingWithServingSideOutput,
  scoreTrackingWithSideSwitchOutput,
} from "./score-tracking-inference.ts";

type ModelScoreResults = Pick<OnDeviceAnalysis, "servingSide" | "sideSwitch">;

export function withModelScoreMarkers(
  draft: CutDraft,
  analysis: ModelScoreResults,
): CutDraft {
  let scoreTracking = draft.scoreTracking;
  if (analysis.servingSide) {
    const hadServeMarkers = scoreTracking.serveMarkers.length > 0;
    const generated = scoreTrackingWithServingSideOutput(
      scoreTracking,
      analysis.servingSide,
    );
    // Generated evidence can fill missing markers, but saved sides, timestamps
    // and markers attached to split rallies belong to the user's review.
    const existingIds = new Set(
      scoreTracking.serveMarkers.map((marker) => marker.id),
    );
    const serveMarkers = [
      ...scoreTracking.serveMarkers,
      ...generated.serveMarkers.filter((marker) => !existingIds.has(marker.id)),
    ].sort(
      (left, right) =>
        left.timestamp - right.timestamp || left.id.localeCompare(right.id),
    );
    scoreTracking =
      !hadServeMarkers && generated.serveMarkers.length > 0
        ? { ...generated, serveMarkers, enabled: true }
        : { ...generated, serveMarkers };
  }
  if (analysis.sideSwitch) {
    const generated = scoreTrackingWithSideSwitchOutput(
      scoreTracking,
      analysis.sideSwitch,
    );
    const existingIds = new Set(
      scoreTracking.sideSwitchMarkers.map((marker) => marker.id),
    );
    scoreTracking = {
      ...generated,
      sideSwitchMarkers: [
        ...scoreTracking.sideSwitchMarkers,
        ...generated.sideSwitchMarkers.filter(
          (marker) => !existingIds.has(marker.id),
        ),
      ].sort(
        (left, right) =>
          left.timestamp - right.timestamp || left.id.localeCompare(right.id),
      ),
    };
  }
  return alignRallyServeMarkers({ ...draft, scoreTracking });
}

// Rebuild from immutable inference outputs, never the edited/imported review draft.
export function createModelReviewDraft(
  seed: CutDraftSeed,
  analysis: ModelScoreResults,
): CutDraft {
  return withModelScoreMarkers(createCutDraft(seed), analysis);
}
