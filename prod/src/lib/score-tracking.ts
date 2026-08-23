export const SCORE_TRACKING_SCHEMA_VERSION = 2 as const;

export const SCORE_TEAM_IDS = ["team-1", "team-2"] as const;
export type ScoreTeamId = (typeof SCORE_TEAM_IDS)[number];

export const SERVING_SIDES = ["near", "far", "review"] as const;
export type ServingSide = (typeof SERVING_SIDES)[number];

export type ServeMarkerOrigin = "model" | "manual";

export type ServeMarker = {
  id: string;
  timestamp: number;
  side: ServingSide;
  origin: ServeMarkerOrigin;
  /** The immutable model verdict, retained when a user corrects `side`. */
  modelSide?: ServingSide;
  /** This serve follows a replay/ignored rally, so it does not award a point. */
  ignorePreviousPoint: boolean;
  rallyId?: string;
};

export type SideSwitchMarker = {
  id: string;
  timestamp: number;
};

/**
 * Team identity is anchored to the first court layout: Team 1 starts near and
 * Team 2 starts far. Side switches change the physical mapping, not identity.
 */
export type ScoreTracking = {
  version: typeof SCORE_TRACKING_SCHEMA_VERSION;
  enabled: boolean;
  team1Name: string;
  team2Name: string;
  serveMarkers: ServeMarker[];
  sideSwitchMarkers: SideSwitchMarker[];
  /** Persisted tombstones prevent deleted model predictions from reappearing. */
  removedModelMarkerIds: string[];
};

export type ScoreIgnoredInterval = {
  start: number;
  end: number;
};

export type ScoreRallyRange = {
  coreStart: number;
  coreEnd: number;
  keepStart: number;
  keepEnd: number;
};

export type ScoreMergedRange = {
  start: number;
  end: number;
};

export type ScorePointStatus = "counted" | "ignored" | "review";

export type DerivedScorePoint = {
  serveMarkerId: string;
  timestamp: number;
  servingSide: ServingSide;
  winnerTeamId: ScoreTeamId | null;
  status: ScorePointStatus;
  team1ScoreAfter: number;
  team2ScoreAfter: number;
};

export type DerivedScore = {
  team1Score: number;
  team2Score: number;
  servingTeamId: ScoreTeamId | null;
  servingSide: ServingSide | null;
  points: DerivedScorePoint[];
  ignoredPointCount: number;
  reviewPointCount: number;
};

export function createScoreTracking(enabled = true): ScoreTracking {
  return {
    version: SCORE_TRACKING_SCHEMA_VERSION,
    enabled,
    team1Name: "Team 1",
    team2Name: "Team 2",
    serveMarkers: [],
    sideSwitchMarkers: [],
    removedModelMarkerIds: [],
  };
}

function finiteTimestamp(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function validServeMarker(
  value: unknown,
  duration: number,
): value is ServeMarker {
  if (!isRecord(value)) return false;
  return (
    typeof value.id === "string" &&
    value.id.length > 0 &&
    finiteTimestamp(value.timestamp) &&
    value.timestamp <= duration &&
    SERVING_SIDES.includes(value.side as ServingSide) &&
    (value.origin === "model" || value.origin === "manual") &&
    (value.modelSide === undefined ||
      SERVING_SIDES.includes(value.modelSide as ServingSide)) &&
    typeof value.ignorePreviousPoint === "boolean" &&
    (value.rallyId === undefined ||
      (typeof value.rallyId === "string" && value.rallyId.length > 0))
  );
}

function validSideSwitchMarker(
  value: unknown,
  duration: number,
): value is SideSwitchMarker {
  if (!isRecord(value)) return false;
  return (
    typeof value.id === "string" &&
    value.id.length > 0 &&
    finiteTimestamp(value.timestamp) &&
    value.timestamp <= duration
  );
}

export function isValidScoreTracking(
  value: unknown,
  duration: number,
): value is ScoreTracking {
  if (!isRecord(value) || !Number.isFinite(duration) || duration < 0)
    return false;
  const removedModelMarkerIds = value.removedModelMarkerIds;
  if (
    value.version !== SCORE_TRACKING_SCHEMA_VERSION ||
    typeof value.enabled !== "boolean" ||
    typeof value.team1Name !== "string" ||
    value.team1Name.trim().length === 0 ||
    typeof value.team2Name !== "string" ||
    value.team2Name.trim().length === 0 ||
    !Array.isArray(value.serveMarkers) ||
    !value.serveMarkers.every((marker) => validServeMarker(marker, duration)) ||
    !Array.isArray(value.sideSwitchMarkers) ||
    !value.sideSwitchMarkers.every((marker) =>
      validSideSwitchMarker(marker, duration),
    ) ||
    !Array.isArray(removedModelMarkerIds) ||
    !removedModelMarkerIds.every(
      (id) => typeof id === "string" && id.length > 0,
    ) ||
    new Set(removedModelMarkerIds).size !== removedModelMarkerIds.length
  )
    return false;

  const ids = [
    ...value.serveMarkers.map((marker) => marker.id),
    ...value.sideSwitchMarkers.map((marker) => marker.id),
  ];
  return (
    new Set(ids).size === ids.length &&
    !ids.some((id) => removedModelMarkerIds.includes(id))
  );
}

export function migrateScoreTracking(
  value: unknown,
  duration: number,
): ScoreTracking | null {
  if (!isRecord(value)) return null;
  const candidate = value.version === 1
    ? {
        ...value,
        version: SCORE_TRACKING_SCHEMA_VERSION,
        removedModelMarkerIds: [],
      }
    : value;
  return isValidScoreTracking(candidate, duration) ? candidate : null;
}

function compareTimestampAndId(
  left: { timestamp: number; id: string },
  right: { timestamp: number; id: string },
): number {
  return left.timestamp - right.timestamp || left.id.localeCompare(right.id);
}

export function orderedServeMarkers(
  scoreTracking: Pick<ScoreTracking, "serveMarkers">,
): ServeMarker[] {
  return [...scoreTracking.serveMarkers].sort(compareTimestampAndId);
}

export function orderedSideSwitchMarkers(
  scoreTracking: Pick<ScoreTracking, "sideSwitchMarkers">,
): SideSwitchMarker[] {
  return [...scoreTracking.sideSwitchMarkers].sort(compareTimestampAndId);
}

/**
 * During dead time and a standalone rally's leading padding, show the score
 * state at the upcoming visible serve. When multiple raw rally fragments are
 * retained as one merged range, their internal padding/gaps stay on the
 * current score unless an actual serve marker lies inside that bridge.
 */
export function scoreBoundaryTimestamp(
  playbackTimestamp: number,
  rallyRanges: readonly ScoreRallyRange[],
  scoreTracking: Pick<ScoreTracking, "serveMarkers">,
  mergedRanges: readonly ScoreMergedRange[] = [],
): number {
  const timestamp = Number.isFinite(playbackTimestamp)
    ? Math.max(0, playbackTimestamp)
    : 0;
  const serves = orderedServeMarkers(scoreTracking);
  const nextServeTimestamp = () =>
    serves.find((marker) => marker.timestamp >= timestamp)?.timestamp ??
    timestamp;
  const paddingBoundaryTimestamp = (start: number, end: number) => {
    const paddingServes = serves.filter(
      (marker) => start <= marker.timestamp && marker.timestamp < end,
    );
    if (paddingServes.length === 0) return nextServeTimestamp();
    return paddingServes.some((marker) => marker.timestamp <= timestamp)
      ? timestamp
      : paddingServes[0].timestamp;
  };
  const mergedRange = mergedRanges.find(
    (range) => range.start <= timestamp && timestamp < range.end,
  );
  if (mergedRange) {
    const mergedRallies = rallyRanges
      .filter(
        (range) =>
          range.keepStart < mergedRange.end &&
          mergedRange.start < range.keepEnd,
      )
      .sort(
        (left, right) =>
          left.coreStart - right.coreStart || left.coreEnd - right.coreEnd,
    );
    if (mergedRallies.length === 0) return timestamp;
    if (timestamp < mergedRallies[0].coreStart) {
      const previousMergedEnd = mergedRanges.reduce(
        (latest, range) =>
          range.end <= mergedRange.start ? Math.max(latest, range.end) : latest,
        0,
      );
      return paddingBoundaryTimestamp(
        previousMergedEnd,
        mergedRallies[0].coreStart,
      );
    }
    for (let index = 1; index < mergedRallies.length; index += 1) {
      const previous = mergedRallies[index - 1];
      const next = mergedRallies[index];
      if (timestamp < previous.coreEnd) return timestamp;
      if (timestamp < next.coreStart) {
        const bridgeServes = serves.filter(
          (marker) =>
            previous.coreEnd <= marker.timestamp &&
            marker.timestamp < next.coreStart,
        );
        if (bridgeServes.length === 0) return timestamp;
        return bridgeServes.some((marker) => marker.timestamp <= timestamp)
          ? timestamp
          : bridgeServes[0].timestamp;
      }
    }
    return timestamp;
  }
  const insideRallyRange = rallyRanges.some(
    (range) => range.keepStart <= timestamp && timestamp < range.keepEnd,
  );
  const leadingPaddingRange = rallyRanges.find(
    (range) => range.keepStart <= timestamp && timestamp < range.coreStart,
  );
  if (insideRallyRange && !leadingPaddingRange) return timestamp;
  if (leadingPaddingRange) {
    return paddingBoundaryTimestamp(
      leadingPaddingRange.keepStart,
      leadingPaddingRange.coreStart,
    );
  }
  return nextServeTimestamp();
}

export function isScoreTimestampIgnored(
  timestamp: number,
  ignoredIntervals: readonly ScoreIgnoredInterval[],
): boolean {
  return ignoredIntervals.some(
    (interval) => interval.start <= timestamp && timestamp < interval.end,
  );
}

/**
 * Ignored source time is an edit-layer filter. It never changes cached model
 * verdicts, so removing an ignored interval restores its markers immediately.
 */
export function scoreTrackingOutsideIgnoredIntervals(
  scoreTracking: ScoreTracking,
  ignoredIntervals: readonly ScoreIgnoredInterval[],
): ScoreTracking {
  if (ignoredIntervals.length === 0) return scoreTracking;
  return {
    ...scoreTracking,
    serveMarkers: scoreTracking.serveMarkers.filter(
      (marker) => !isScoreTimestampIgnored(marker.timestamp, ignoredIntervals),
    ),
  };
}

/**
 * An excluded rally hides its model-linked serve marker without deleting the
 * cached verdict. Keeping the rally again therefore restores the marker.
 */
export function scoreTrackingOutsideExcludedRallies(
  scoreTracking: ScoreTracking,
  excludedRallyIds: ReadonlySet<string>,
): ScoreTracking {
  if (excludedRallyIds.size === 0) return scoreTracking;
  return {
    ...scoreTracking,
    serveMarkers: scoreTracking.serveMarkers.filter(
      (marker) => !marker.rallyId || !excludedRallyIds.has(marker.rallyId),
    ),
  };
}

export function teamForServingSide(
  side: ServingSide,
  sideSwitchCount: number,
): ScoreTeamId | null {
  if (side === "review") return null;
  const switched = Math.abs(Math.trunc(sideSwitchCount)) % 2 === 1;
  if (side === "near") return switched ? "team-2" : "team-1";
  return switched ? "team-1" : "team-2";
}

/**
 * Resolves source-timeline score state. A switch whose timestamp equals a
 * serve marker applies to that serve. The first serve establishes the initial
 * server and never represents a completed point.
 */
export function deriveScoreAt(
  scoreTracking: Pick<ScoreTracking, "serveMarkers" | "sideSwitchMarkers">,
  sourceTimestamp = Number.POSITIVE_INFINITY,
): DerivedScore {
  const maximumTimestamp = Number.isFinite(sourceTimestamp)
    ? Math.max(0, sourceTimestamp)
    : Number.POSITIVE_INFINITY;
  const serves = orderedServeMarkers(scoreTracking).filter(
    (marker) => marker.timestamp <= maximumTimestamp,
  );
  const switches = orderedSideSwitchMarkers(scoreTracking);
  let switchIndex = 0;
  let team1Score = 0;
  let team2Score = 0;
  let servingTeamId: ScoreTeamId | null = null;
  let servingSide: ServingSide | null = null;
  let ignoredPointCount = 0;
  let reviewPointCount = 0;
  const points: DerivedScorePoint[] = [];

  for (let serveIndex = 0; serveIndex < serves.length; serveIndex += 1) {
    const serve = serves[serveIndex];
    while (
      switchIndex < switches.length &&
      switches[switchIndex].timestamp <= serve.timestamp
    )
      switchIndex += 1;

    servingSide = serve.side;
    servingTeamId = teamForServingSide(serve.side, switchIndex);
    if (serveIndex === 0) continue;

    let status: ScorePointStatus = "counted";
    if (serve.ignorePreviousPoint) {
      status = "ignored";
      ignoredPointCount += 1;
    } else if (servingTeamId === null) {
      status = "review";
      reviewPointCount += 1;
    } else if (servingTeamId === "team-1") {
      team1Score += 1;
    } else {
      team2Score += 1;
    }
    points.push({
      serveMarkerId: serve.id,
      timestamp: serve.timestamp,
      servingSide: serve.side,
      winnerTeamId: servingTeamId,
      status,
      team1ScoreAfter: team1Score,
      team2ScoreAfter: team2Score,
    });
  }

  return {
    team1Score,
    team2Score,
    servingTeamId,
    servingSide,
    points,
    ignoredPointCount,
    reviewPointCount,
  };
}

function nextMarkerId(
  prefix: "S" | "X",
  scoreTracking: Pick<
    ScoreTracking,
    "serveMarkers" | "sideSwitchMarkers" | "removedModelMarkerIds"
  >,
): string {
  const used = new Set([
    ...scoreTracking.serveMarkers.map((marker) => marker.id),
    ...scoreTracking.sideSwitchMarkers.map((marker) => marker.id),
    ...scoreTracking.removedModelMarkerIds,
  ]);
  for (let index = 1; index < 10_000; index += 1) {
    const id = `${prefix}${String(index).padStart(3, "0")}`;
    if (!used.has(id)) return id;
  }
  return `${prefix}${Date.now()}`;
}

export function addServeMarker(
  scoreTracking: ScoreTracking,
  timestamp: number,
  side: ServingSide,
  options: {
    id?: string;
    origin?: ServeMarkerOrigin;
    modelSide?: ServingSide;
    rallyId?: string;
  } = {},
): ScoreTracking {
  if (!finiteTimestamp(timestamp) || !SERVING_SIDES.includes(side)) {
    return scoreTracking;
  }
  const id = options.id ?? nextMarkerId("S", scoreTracking);
  if (
    id.length === 0 ||
    scoreTracking.serveMarkers.some((marker) => marker.id === id) ||
    scoreTracking.sideSwitchMarkers.some((marker) => marker.id === id) ||
    scoreTracking.removedModelMarkerIds.includes(id)
  )
    return scoreTracking;
  const origin = options.origin ?? "manual";
  const marker: ServeMarker = {
    id,
    timestamp: Math.round(timestamp * 1000) / 1000,
    side,
    origin,
    ...(origin === "model"
      ? { modelSide: options.modelSide ?? side }
      : options.modelSide === undefined
        ? {}
        : { modelSide: options.modelSide }),
    ignorePreviousPoint: false,
    ...(options.rallyId ? { rallyId: options.rallyId } : {}),
  };
  return {
    ...scoreTracking,
    serveMarkers: [...scoreTracking.serveMarkers, marker].sort(
      compareTimestampAndId,
    ),
  };
}

export function setServeMarkerSide(
  scoreTracking: ScoreTracking,
  markerId: string,
  side: ServingSide,
): ScoreTracking {
  if (!SERVING_SIDES.includes(side)) return scoreTracking;
  return {
    ...scoreTracking,
    serveMarkers: scoreTracking.serveMarkers.map((marker) =>
      marker.id === markerId ? { ...marker, side } : marker,
    ),
  };
}

export function setPreviousPointIgnored(
  scoreTracking: ScoreTracking,
  markerId: string,
  ignored: boolean,
): ScoreTracking {
  return {
    ...scoreTracking,
    serveMarkers: scoreTracking.serveMarkers.map((marker) =>
      marker.id === markerId
        ? { ...marker, ignorePreviousPoint: ignored }
        : marker,
    ),
  };
}

export function removeServeMarker(
  scoreTracking: ScoreTracking,
  markerId: string,
): ScoreTracking {
  const removed = scoreTracking.serveMarkers.find(
    (marker) => marker.id === markerId,
  );
  return {
    ...scoreTracking,
    serveMarkers: scoreTracking.serveMarkers.filter(
      (marker) => marker.id !== markerId,
    ),
    removedModelMarkerIds: removed?.origin === "model"
      ? [...new Set([...scoreTracking.removedModelMarkerIds, markerId])]
      : scoreTracking.removedModelMarkerIds,
  };
}

export function addSideSwitchMarker(
  scoreTracking: ScoreTracking,
  timestamp: number,
  id = nextMarkerId("X", scoreTracking),
): ScoreTracking {
  if (
    !finiteTimestamp(timestamp) ||
    id.length === 0 ||
    scoreTracking.serveMarkers.some((marker) => marker.id === id) ||
    scoreTracking.sideSwitchMarkers.some((marker) => marker.id === id) ||
    scoreTracking.removedModelMarkerIds.includes(id)
  )
    return scoreTracking;
  return {
    ...scoreTracking,
    sideSwitchMarkers: [
      ...scoreTracking.sideSwitchMarkers,
      { id, timestamp: Math.round(timestamp * 1000) / 1000 },
    ].sort(compareTimestampAndId),
  };
}

export function removeSideSwitchMarker(
  scoreTracking: ScoreTracking,
  markerId: string,
): ScoreTracking {
  return {
    ...scoreTracking,
    sideSwitchMarkers: scoreTracking.sideSwitchMarkers.filter(
      (marker) => marker.id !== markerId,
    ),
  };
}
