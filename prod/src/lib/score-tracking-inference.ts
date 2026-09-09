import type {
  OnDeviceSideSwitchOutput,
  OnDeviceServingSideOutput,
} from "./on-device/types.ts";
import {
  addServeMarker,
  addSideSwitchMarker,
  type ScoreTracking,
} from "./score-tracking.ts";

export function scoreTrackingWithServingSideOutput(
  current: ScoreTracking,
  output: OnDeviceServingSideOutput,
): ScoreTracking {
  const existingByRally = new Map(
    current.serveMarkers
      .filter((marker) => marker.rallyId)
      .map((marker) => [marker.rallyId!, marker]),
  );
  let scoreTracking: ScoreTracking = {
    ...current,
    serveMarkers: current.serveMarkers.filter(
      (marker) => marker.origin === "manual",
    ),
  };
  for (const candidate of output.candidates) {
    const existing = existingByRally.get(candidate.id);
    const wasCorrected = Boolean(
      existing?.modelSide && existing.side !== existing.modelSide,
    );
    if (candidate.verdict === "not-serve" && !wasCorrected) continue;
    const modelSide =
      candidate.verdict === "review" || candidate.verdict === "not-serve"
        ? "review"
        : candidate.side;
    scoreTracking = addServeMarker(
      scoreTracking,
      existing?.timestamp ?? candidate.anchor,
      wasCorrected ? existing!.side : modelSide,
      {
        id: `serve-${candidate.id}`,
        origin: "model",
        modelSide,
        rallyId: candidate.id,
      },
    );
    if (existing?.ignorePreviousPoint) {
      scoreTracking = {
        ...scoreTracking,
        serveMarkers: scoreTracking.serveMarkers.map((marker) =>
          marker.rallyId === candidate.id
            ? { ...marker, ignorePreviousPoint: true }
            : marker,
        ),
      };
    }
  }
  return scoreTracking;
}

export function scoreTrackingWithSideSwitchOutput(
  current: ScoreTracking,
  output: OnDeviceSideSwitchOutput,
): ScoreTracking {
  let scoreTracking: ScoreTracking = {
    ...current,
    sideSwitchMarkers: current.sideSwitchMarkers.filter(
      (marker) => marker.origin === "manual",
    ),
  };
  for (const candidate of output.candidates) {
    scoreTracking = addSideSwitchMarker(
      scoreTracking,
      candidate.timestamp,
      {
        id: `switch-${candidate.id}`,
        origin: "model",
        modelConfidence: candidate.probability,
        modelEventId: candidate.id,
        rallyIds: candidate.sourceRangeIds,
      },
    );
  }
  return scoreTracking;
}
