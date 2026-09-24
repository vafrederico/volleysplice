import type { ScoreTracking } from "./score-tracking.ts";

/** Preserve reviewed serves while giving a genuinely new rally an unresolved start. */
export function splitLabServeMarkers(
  scoreTracking: ScoreTracking,
  parentId: string,
  left: { id: string; start: number },
  right: { id: string; start: number },
): ScoreTracking {
  const serveMarkers = scoreTracking.serveMarkers.map(marker => marker.rallyId === parentId
    ? { ...marker, rallyId: left.id, timestamp: left.start } : marker);
  const rightMarkerId = "lab-split-serve:" + right.id;
  if (!serveMarkers.some(marker => marker.rallyId === right.id || marker.id === rightMarkerId)
    && !scoreTracking.sideSwitchMarkers.some(marker => marker.id === rightMarkerId)
    && !scoreTracking.removedModelMarkerIds.includes(rightMarkerId)) {
    serveMarkers.push({ id: rightMarkerId, rallyId: right.id, timestamp: right.start,
      side: "review", modelSide: "review", origin: "model", ignorePreviousPoint: false });
  }
  return { ...scoreTracking,
    serveMarkers: serveMarkers.sort((a, b) => a.timestamp - b.timestamp || a.id.localeCompare(b.id)) };
}
