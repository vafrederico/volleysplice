import type { HardNegative, RallyLabel, ServeMarker } from "./annotations.ts";

const serveMarkerAssociationSeconds = 2;

export type RallyMergeResult =
  | {
      ok: true;
      rallies: RallyLabel[];
      serveMarkers: ServeMarker[];
      mergedIndex: number;
      mergedRally: RallyLabel;
      removedServeMarkerCount: number;
      movedServeMarkerToStart: boolean;
    }
  | { ok: false; error: string };

function overlaps(
  start: number,
  end: number,
  rows: Array<{ start: number; end: number }>,
): boolean {
  return rows.some((row) => start < row.end && row.start < end);
}

export function findServeMarkerRallyIndex(
  rallies: RallyLabel[],
  time: number,
  maxDistanceSeconds = serveMarkerAssociationSeconds,
): number {
  let closestIndex = -1;
  let closestDistance = Number.POSITIVE_INFINITY;
  rallies.forEach((rally, index) => {
    const distance = Math.abs(rally.start - time);
    if (distance < closestDistance) {
      closestDistance = distance;
      closestIndex = index;
    }
  });
  return closestDistance <= maxDistanceSeconds ? closestIndex : -1;
}

export function findServeMarkerIndexForRally(
  rallies: RallyLabel[],
  serveMarkers: ServeMarker[],
  rallyIndex: number,
): number {
  const rally = rallies[rallyIndex];
  if (!rally) return -1;
  let closestMarkerIndex = -1;
  let closestDistance = Number.POSITIVE_INFINITY;
  serveMarkers.forEach((marker, markerIndex) => {
    const distance = Math.abs(marker.time - rally.start);
    if (
      findServeMarkerRallyIndex(rallies, marker.time) === rallyIndex &&
      distance < closestDistance
    ) {
      closestMarkerIndex = markerIndex;
      closestDistance = distance;
    }
  });
  return closestMarkerIndex;
}

export function findClosestNextRallyIndex(
  rallies: RallyLabel[],
  time: number,
  epsilon = 0.0005,
): number {
  let nextIndex = -1;
  let nextStart = Number.POSITIVE_INFINITY;
  rallies.forEach((rally, index) => {
    if (rally.start > time + epsilon && rally.start < nextStart) {
      nextIndex = index;
      nextStart = rally.start;
    }
  });
  return nextIndex;
}

export function mergeSelectedRallies({
  rallies,
  serveMarkers,
  hardNegatives,
  selectedIndexes,
}: {
  rallies: RallyLabel[];
  serveMarkers: ServeMarker[];
  hardNegatives: HardNegative[];
  selectedIndexes: number[];
}): RallyMergeResult {
  const indexes = [...new Set(selectedIndexes)].sort((left, right) => left - right);
  if (indexes.length < 2) {
    return { ok: false, error: "Shift-click at least two rallies to merge." };
  }
  if (indexes.some((index) => !Number.isInteger(index) || !rallies[index])) {
    return { ok: false, error: "The rally merge selection is no longer valid." };
  }
  if (indexes.some((index, position) => position > 0 && index !== indexes[position - 1] + 1)) {
    return { ok: false, error: "Only consecutive rallies can be merged." };
  }

  const firstIndex = indexes[0];
  const lastIndex = indexes[indexes.length - 1];
  const selected = rallies.slice(firstIndex, lastIndex + 1);
  const first = selected[0];
  const last = selected[selected.length - 1];

  if (overlaps(first.start, last.end, hardNegatives)) {
    return {
      ok: false,
      error: "The merged rally would overlap a hard negative. Move or remove that hard negative first.",
    };
  }

  const internalBoundaryTracklet = selected.some((rally, index) =>
    (rally.playerTracklets ?? []).some(
      (tracklet) =>
        (tracklet.window === "serve" && index !== 0) ||
        (tracklet.window === "rally-end" && index !== selected.length - 1),
    ),
  );
  if (internalBoundaryTracklet) {
    return {
      ok: false,
      error:
        "Remove player tracks attached to the internal rally boundaries before merging.",
    };
  }

  const notes = [...new Set(selected.map((rally) => rally.notes?.trim()).filter(Boolean))];
  const playerTracklets = [
    ...(first.playerTracklets ?? []).filter((tracklet) => tracklet.window === "serve"),
    ...(last.playerTracklets ?? []).filter((tracklet) => tracklet.window === "rally-end"),
  ];
  const mergedRally: RallyLabel = {
    ...first,
    end: last.end,
    tags: [...new Set(selected.flatMap((rally) => rally.tags))],
    notes: notes.length > 0 ? notes.join("\n") : undefined,
    collectiveStandDownTime: last.collectiveStandDownTime,
    terminalCue: last.terminalCue,
    endObservability: last.endObservability,
    endConfidence: last.endConfidence,
    verifiedImmediateResult: last.verifiedImmediateResult,
    playerTracklets: playerTracklets.length > 0 ? playerTracklets : undefined,
  };

  const selectedIndexSet = new Set(indexes);
  const associatedMarkerIndexes = serveMarkers.flatMap((marker, markerIndex) =>
    selectedIndexSet.has(findServeMarkerRallyIndex(rallies, marker.time))
      ? [markerIndex]
      : [],
  );
  const earliestMarkerIndex = associatedMarkerIndexes.reduce(
    (earliest, markerIndex) =>
      earliest < 0 || serveMarkers[markerIndex].time < serveMarkers[earliest].time
        ? markerIndex
        : earliest,
    -1,
  );
  const movedServeMarkerToStart =
    earliestMarkerIndex >= 0 &&
    findServeMarkerRallyIndex(rallies, serveMarkers[earliestMarkerIndex].time) !== firstIndex;
  const associatedMarkerIndexSet = new Set(associatedMarkerIndexes);
  const nextServeMarkers = serveMarkers
    .flatMap((marker, markerIndex) => {
      if (associatedMarkerIndexSet.has(markerIndex) && markerIndex !== earliestMarkerIndex) {
        return [];
      }
      return [
        markerIndex === earliestMarkerIndex && movedServeMarkerToStart
          ? { ...marker, time: first.start }
          : marker,
      ];
    })
    .sort((left, right) => left.time - right.time);

  return {
    ok: true,
    rallies: [
      ...rallies.slice(0, firstIndex),
      mergedRally,
      ...rallies.slice(lastIndex + 1),
    ],
    serveMarkers: nextServeMarkers,
    mergedIndex: firstIndex,
    mergedRally,
    removedServeMarkerCount: associatedMarkerIndexes.length > 0
      ? associatedMarkerIndexes.length - 1
      : 0,
    movedServeMarkerToStart,
  };
}
