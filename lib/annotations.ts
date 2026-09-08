export const annotationPolicyId = "serve-contact-to-dead-ball-v1";

export const terminalCueValues = [
  "ball-down-or-out",
  "whistle-or-stoppage",
  "no-recovery",
  "unobservable",
] as const;

export const endObservabilityValues = [
  "observable",
  "partially-observable",
  "unobservable",
] as const;

export const playerTrackletWindowValues = ["serve", "rally-end"] as const;

export const playerTeamValues = ["team-a", "team-b", "unknown"] as const;

export const playerCourtSideValues = ["near", "far", "outside", "unknown"] as const;

export const playerStateValues = [
  "ready",
  "playing",
  "jumping",
  "stand-down",
  "walking",
] as const;

export const servingSideValues = ["near", "far", "review"] as const;

export const hardNegativeCategories = [
  "adjacent-court",
  "camera-motion",
  "celebration",
  "celebration-huddle",
  "foreground-crossing",
  "model-false-positive",
  "random-dead-control",
  "setup-between-points",
  "timeout",
  "walking-ball-retrieval",
  "warmup",
  "other",
] as const;

export type NormalizedPoint = { x: number; y: number };

export type NormalizedBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type PlayerTrackletObservation = {
  time: number;
  footpoint?: NormalizedPoint;
  box?: NormalizedBox;
  state?: (typeof playerStateValues)[number];
};

export type PlayerTracklet = {
  trackId: string;
  window: (typeof playerTrackletWindowValues)[number];
  team: (typeof playerTeamValues)[number];
  courtSide: (typeof playerCourtSideValues)[number];
  observations: PlayerTrackletObservation[];
  notes?: string;
};

export type CourtGeometry = {
  corners: Partial<
    Record<"nearLeft" | "nearRight" | "farLeft" | "farRight", NormalizedPoint>
  >;
  netAnchors?: Partial<Record<"left" | "right", NormalizedPoint>>;
  serviceZoneAnchors?: Partial<Record<"near" | "far", NormalizedPoint>>;
};

export type RallyLabel = {
  start: number;
  end: number;
  tags: string[];
  notes?: string;
  receiverReactionTime?: number;
  collectiveStandDownTime?: number;
  terminalCue?: (typeof terminalCueValues)[number];
  endObservability?: (typeof endObservabilityValues)[number];
  startConfidence?: number;
  endConfidence?: number;
  verifiedImmediateResult?: boolean;
  playerTracklets?: PlayerTracklet[];
};

export type IgnoredInterval = {
  start: number;
  end: number;
  reason: string;
  notes?: string;
};

export type HardNegative = {
  start: number;
  end: number;
  category: string;
  notes?: string;
};

export type SideSwitch = {
  time: number;
  notes?: string;
  origin?: "model" | "manual";
  modelConfidence?: number;
  modelId?: string;
  modelEventId?: string;
};

export type ServeMarker = {
  time: number;
  side: (typeof servingSideValues)[number];
  notes?: string;
  origin?: "model" | "manual";
  modelSide?: (typeof servingSideValues)[number];
  modelConfidence?: number;
  modelId?: string;
  rallyId?: string;
};

export type LabelDocument = {
  schemaVersion: 1;
  kind: "volleycut-rally-labels";
  createdAt: string;
  recording: {
    id: string;
    video: string;
    videoFilename: string;
    contentSha256: string;
    durationSeconds: number;
    sourceGroup: string;
    split: "train" | "validation" | "test" | "challenge";
    environment: "indoor" | "beach" | "grass" | "broadcast" | "unknown";
    game: {
      playersPerTeam: number | null;
      targetPoints: number | null;
      format: string | null;
      scoringRule?: string | null;
    };
    capture: Record<string, unknown>;
    roi: { x: number; y: number; width: number; height: number } | null;
    courtGeometry?: CourtGeometry;
  };
  annotationPolicy: {
    id: typeof annotationPolicyId;
    rallyStart: string;
    rallyEnd: string;
    intervalConvention: string;
  };
  annotation: {
    status: "not-started" | "in-progress" | "complete";
    annotator: string;
    continuousVideoReviewed: boolean;
    reviewedAt: string | null;
    notes: string;
  };
  prelabel?: {
    analysisMethod: string;
    candidateFile: string;
    analyzedAt: string;
    ambiguities: unknown[];
  };
  rallies: RallyLabel[];
  ignoredIntervals: IgnoredInterval[];
  hardNegatives: HardNegative[];
  serveMarkers: ServeMarker[];
  sideSwitches: SideSwitch[];
};

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function assertOptionalString(value: unknown, where: string): void {
  if (value !== undefined && typeof value !== "string") {
    throw new Error(`${where} must be a string when present`);
  }
}

function assertIntervals(value: unknown, name: string): asserts value is Array<Record<string, unknown>> {
  if (!Array.isArray(value)) throw new Error(`${name} must be an array`);
  let previousEnd = -1;
  value.forEach((row, index) => {
    if (!isObject(row) || !isFiniteNumber(row.start) || !isFiniteNumber(row.end)) {
      throw new Error(`${name}[${index}] must contain numeric start and end`);
    }
    if (row.start < 0 || row.end <= row.start || row.start < previousEnd) {
      throw new Error(`${name} must be ordered and satisfy 0 ≤ start < end`);
    }
    previousEnd = row.end;
  });
}

function validModelMetadata(row: Record<string, unknown>): boolean {
  return (
    (row.origin === undefined || row.origin === "model" || row.origin === "manual") &&
    (row.modelConfidence === undefined ||
      (isFiniteNumber(row.modelConfidence) &&
        row.modelConfidence >= 0 &&
        row.modelConfidence <= 1)) &&
    (row.modelId === undefined || typeof row.modelId === "string")
  );
}

function readServeMarkers(value: unknown, duration: number): ServeMarker[] {
  if (value === undefined) return [];
  if (!Array.isArray(value)) throw new Error("serveMarkers must be an array");
  const sides = new Set<string>(servingSideValues);
  let previousTime = -1;
  value.forEach((row, index) => {
    if (
      !isObject(row) ||
      !isFiniteNumber(row.time) ||
      row.time < 0 ||
      row.time > duration ||
      row.time <= previousTime ||
      typeof row.side !== "string" ||
      !sides.has(row.side) ||
      (row.modelSide !== undefined &&
        (typeof row.modelSide !== "string" || !sides.has(row.modelSide))) ||
      (row.notes !== undefined && typeof row.notes !== "string") ||
      (row.rallyId !== undefined && typeof row.rallyId !== "string") ||
      !validModelMetadata(row)
    ) {
      throw new Error(
        `serveMarkers[${index}] must have an ordered in-range time, a valid side, and valid optional model metadata`,
      );
    }
    previousTime = row.time;
  });
  return value as ServeMarker[];
}

function readSideSwitches(value: unknown, duration: number): SideSwitch[] {
  if (value === undefined) return [];
  if (!Array.isArray(value)) throw new Error("sideSwitches must be an array");
  let previousTime = -1;
  value.forEach((row, index) => {
    if (
      !isObject(row) ||
      !isFiniteNumber(row.time) ||
      row.time < 0 ||
      row.time > duration ||
      row.time <= previousTime ||
      (row.notes !== undefined && typeof row.notes !== "string") ||
      (row.modelEventId !== undefined && typeof row.modelEventId !== "string") ||
      !validModelMetadata(row)
    ) {
      throw new Error(
        `sideSwitches[${index}] must have an ordered, non-negative finite time and optional notes`,
      );
    }
    previousTime = row.time;
  });
  return value as SideSwitch[];
}

function readPoint(value: unknown, where: string): NormalizedPoint {
  if (isObject(value)) assertOnlyKeys(value, ["x", "y"], where);
  if (
    !isObject(value) ||
    !isFiniteNumber(value.x) ||
    !isFiniteNumber(value.y) ||
    value.x < 0 ||
    value.x > 1 ||
    value.y < 0 ||
    value.y > 1
  ) {
    throw new Error(`${where} must contain normalized finite x and y coordinates`);
  }
  return { x: value.x, y: value.y };
}

function assertOnlyKeys(
  value: Record<string, unknown>,
  allowed: readonly string[],
  where: string,
): void {
  const recognized = new Set(allowed);
  const unknown = Object.keys(value).find((key) => !recognized.has(key));
  if (unknown) throw new Error(`${where}.${unknown} is not recognized`);
}

function readNormalizedBox(value: unknown, where: string): NormalizedBox {
  if (!isObject(value)) throw new Error(`${where} must be an object`);
  assertOnlyKeys(value, ["x", "y", "width", "height"], where);
  const { x, y, width, height } = value;
  if (
    !isFiniteNumber(x) ||
    !isFiniteNumber(y) ||
    !isFiniteNumber(width) ||
    !isFiniteNumber(height) ||
    x < 0 ||
    y < 0 ||
    width <= 0 ||
    height <= 0 ||
    x + width > 1 ||
    y + height > 1
  ) {
    throw new Error(`${where} must be a positive normalized frame box`);
  }
  return { x, y, width, height };
}

function validatePlayerTracklets(
  value: unknown,
  rallyIndex: number,
  rallyStart: number,
  rallyEnd: number,
  duration: number,
): void {
  if (value === undefined) return;
  if (!Array.isArray(value)) {
    throw new Error(`rallies[${rallyIndex}].playerTracklets must be an array`);
  }
  const windows = new Set<string>(playerTrackletWindowValues);
  const teams = new Set<string>(playerTeamValues);
  const courtSides = new Set<string>(playerCourtSideValues);
  const states = new Set<string>(playerStateValues);
  const seenKeys = new Set<string>();
  value.forEach((tracklet, trackletIndex) => {
    const where = `rallies[${rallyIndex}].playerTracklets[${trackletIndex}]`;
    if (!isObject(tracklet)) throw new Error(`${where} must be an object`);
    assertOnlyKeys(
      tracklet,
      ["trackId", "window", "team", "courtSide", "observations", "notes"],
      where,
    );
    if (
      typeof tracklet.trackId !== "string" ||
      tracklet.trackId !== tracklet.trackId.trim() ||
      !/^[A-Z]{0,2}[0-9]{1,3}$/.test(tracklet.trackId)
    ) {
      throw new Error(`${where}.trackId must be an anonymous token such as P1 or A02`);
    }
    if (typeof tracklet.window !== "string" || !windows.has(tracklet.window)) {
      throw new Error(`${where}.window is invalid`);
    }
    if (typeof tracklet.team !== "string" || !teams.has(tracklet.team)) {
      throw new Error(`${where}.team is invalid`);
    }
    if (typeof tracklet.courtSide !== "string" || !courtSides.has(tracklet.courtSide)) {
      throw new Error(`${where}.courtSide is invalid`);
    }
    assertOptionalString(tracklet.notes, `${where}.notes`);
    const uniqueKey = `${tracklet.window}:${tracklet.trackId}`;
    if (seenKeys.has(uniqueKey)) {
      throw new Error(`${where} duplicates track ${tracklet.trackId} in the same window`);
    }
    seenKeys.add(uniqueKey);
    if (!Array.isArray(tracklet.observations) || tracklet.observations.length === 0) {
      throw new Error(`${where}.observations must contain at least one labeled frame`);
    }
    const windowStart =
      tracklet.window === "serve"
        ? Math.max(0, rallyStart - 2)
        : Math.max(0, rallyEnd - 3);
    const windowEnd =
      tracklet.window === "serve"
        ? Math.min(duration, rallyStart + 3)
        : Math.min(duration, rallyEnd + 2);
    let previousTime = -1;
    tracklet.observations.forEach((observation, observationIndex) => {
      const observationWhere = `${where}.observations[${observationIndex}]`;
      if (!isObject(observation)) throw new Error(`${observationWhere} must be an object`);
      assertOnlyKeys(observation, ["time", "footpoint", "box", "state"], observationWhere);
      if (
        !isFiniteNumber(observation.time) ||
        observation.time < windowStart ||
        observation.time > windowEnd ||
        observation.time <= previousTime
      ) {
        throw new Error(
          `${observationWhere}.time must be strictly ordered inside its boundary window`,
        );
      }
      previousTime = observation.time;
      if (observation.footpoint === undefined && observation.box === undefined) {
        throw new Error(`${observationWhere} must contain a footpoint or box`);
      }
      if (observation.footpoint !== undefined) {
        readPoint(observation.footpoint, `${observationWhere}.footpoint`);
      }
      if (observation.box !== undefined) {
        readNormalizedBox(observation.box, `${observationWhere}.box`);
      }
      if (observation.state !== undefined && !states.has(String(observation.state))) {
        throw new Error(`${observationWhere}.state is invalid`);
      }
    });
  });
}

function readPointGroup(
  value: unknown,
  where: string,
  names: readonly string[],
): Record<string, NormalizedPoint> {
  if (!isObject(value)) throw new Error(`${where} must be an object`);
  const allowed = new Set(names);
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) throw new Error(`${where}.${key} is not a recognized anchor`);
  }
  const points: Record<string, NormalizedPoint> = {};
  for (const name of names) {
    if (value[name] !== undefined) points[name] = readPoint(value[name], `${where}.${name}`);
  }
  return points;
}

function readCourtGeometry(value: unknown): CourtGeometry | undefined {
  if (value === undefined) return undefined;
  if (!isObject(value)) throw new Error("recording.courtGeometry must be an object");
  const allowed = new Set(["corners", "netAnchors", "serviceZoneAnchors"]);
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) {
      throw new Error(`recording.courtGeometry.${key} is not recognized`);
    }
  }
  const corners = readPointGroup(
    value.corners,
    "recording.courtGeometry.corners",
    ["nearLeft", "nearRight", "farLeft", "farRight"],
  ) as CourtGeometry["corners"];
  const courtGeometry: CourtGeometry = { corners };
  if (value.netAnchors !== undefined) {
    courtGeometry.netAnchors = readPointGroup(
      value.netAnchors,
      "recording.courtGeometry.netAnchors",
      ["left", "right"],
    ) as CourtGeometry["netAnchors"];
  }
  if (value.serviceZoneAnchors !== undefined) {
    courtGeometry.serviceZoneAnchors = readPointGroup(
      value.serviceZoneAnchors,
      "recording.courtGeometry.serviceZoneAnchors",
      ["near", "far"],
    ) as CourtGeometry["serviceZoneAnchors"];
  }
  return courtGeometry;
}

function validateCompleteCourtGeometry(geometry: CourtGeometry | undefined): void {
  if (!geometry) return;
  if (
    !geometry.corners.nearLeft ||
    !geometry.corners.nearRight ||
    !geometry.corners.farLeft ||
    !geometry.corners.farRight
  ) {
    throw new Error("completed court geometry must contain all four named corners");
  }
  if (!!geometry.netAnchors?.left !== !!geometry.netAnchors?.right) {
    throw new Error("completed court geometry must contain both net anchors or neither");
  }
  if (!!geometry.serviceZoneAnchors?.near !== !!geometry.serviceZoneAnchors?.far) {
    throw new Error(
      "completed court geometry must contain both service-zone anchors or neither",
    );
  }
}

function validateRallyMetadata(value: unknown, duration: number): void {
  if (!Array.isArray(value)) return;
  const terminalCues = new Set<string>(terminalCueValues);
  const observability = new Set<string>(endObservabilityValues);
  value.forEach((row, index) => {
    if (!isObject(row)) return;
    if (!Array.isArray(row.tags) || row.tags.some((tag) => typeof tag !== "string" || !tag)) {
      throw new Error(`rallies[${index}].tags must be an array of non-empty strings`);
    }
    assertOptionalString(row.notes, `rallies[${index}].notes`);
    for (const confidence of ["startConfidence", "endConfidence"] as const) {
      const confidenceValue = row[confidence];
      if (
        confidenceValue !== undefined &&
        (!isFiniteNumber(confidenceValue) || confidenceValue < 0 || confidenceValue > 1)
      ) {
        throw new Error(`rallies[${index}].${confidence} must be between 0 and 1`);
      }
    }
    if (row.terminalCue !== undefined && !terminalCues.has(String(row.terminalCue))) {
      throw new Error(`rallies[${index}].terminalCue is invalid`);
    }
    if (
      row.endObservability !== undefined &&
      !observability.has(String(row.endObservability))
    ) {
      throw new Error(`rallies[${index}].endObservability is invalid`);
    }
    if (
      row.verifiedImmediateResult !== undefined &&
      typeof row.verifiedImmediateResult !== "boolean"
    ) {
      throw new Error(`rallies[${index}].verifiedImmediateResult must be boolean`);
    }
    const reaction = row.receiverReactionTime;
    if (
      reaction !== undefined &&
      (!isFiniteNumber(reaction) ||
        reaction < Number(row.start) ||
        reaction > Math.min(Number(row.end), Number(row.start) + 5) ||
        reaction > duration)
    ) {
      throw new Error(
        `rallies[${index}].receiverReactionTime must be within five seconds after rally start`,
      );
    }
    const standDown = row.collectiveStandDownTime;
    if (
      standDown !== undefined &&
      (!isFiniteNumber(standDown) ||
        standDown < Math.max(Number(row.start), Number(row.end) - 5) ||
        standDown > Math.min(duration, Number(row.end) + 5))
    ) {
      throw new Error(
        `rallies[${index}].collectiveStandDownTime must be within five seconds of rally end`,
      );
    }
    if (
      isFiniteNumber(reaction) &&
      isFiniteNumber(standDown) &&
      reaction > standDown
    ) {
      throw new Error(
        `rallies[${index}].receiverReactionTime must not follow collectiveStandDownTime`,
      );
    }
    validatePlayerTracklets(
      row.playerTracklets,
      index,
      Number(row.start),
      Number(row.end),
      duration,
    );
  });
}

function validateIntervalMetadata(value: unknown, kind: "ignored" | "negative"): void {
  if (!Array.isArray(value)) return;
  const categories = new Set<string>(hardNegativeCategories);
  value.forEach((row, index) => {
    if (!isObject(row)) return;
    assertOptionalString(row.notes, `${kind === "ignored" ? "ignoredIntervals" : "hardNegatives"}[${index}].notes`);
    if (kind === "ignored") {
      if (typeof row.reason !== "string" || !row.reason) {
        throw new Error(`ignoredIntervals[${index}].reason must be a non-empty string`);
      }
    } else if (typeof row.category !== "string" || !categories.has(row.category)) {
      throw new Error(`hardNegatives[${index}].category is invalid`);
    }
  });
}

export function parseLabelDocument(value: unknown): LabelDocument {
  if (!isObject(value) || value.schemaVersion !== 1 || value.kind !== "volleycut-rally-labels") {
    throw new Error("This is not a VolleySplice rally-label document (schema version 1)");
  }
  if (!isObject(value.recording)) throw new Error("recording must be an object");
  const recording = value.recording;
  if (typeof recording.id !== "string" || !recording.id.trim()) {
    throw new Error("recording.id is missing");
  }
  if (typeof recording.videoFilename !== "string" || !recording.videoFilename) {
    throw new Error("recording.videoFilename is missing");
  }
  if (!isFiniteNumber(recording.durationSeconds) || recording.durationSeconds <= 0) {
    throw new Error("recording.durationSeconds must be positive");
  }
  const duration = recording.durationSeconds;
  if (!isObject(recording.game)) throw new Error("recording.game must be an object");
  if (!isObject(value.annotation) || !isObject(value.annotationPolicy)) {
    throw new Error("annotation metadata is missing");
  }
  if (value.annotationPolicy.id !== annotationPolicyId) {
    throw new Error(`annotationPolicy.id must be ${annotationPolicyId}`);
  }
  if (
    value.prelabel !== undefined &&
    (!isObject(value.prelabel) ||
      typeof value.prelabel.analysisMethod !== "string" ||
      typeof value.prelabel.candidateFile !== "string" ||
      typeof value.prelabel.analyzedAt !== "string" ||
      !Array.isArray(value.prelabel.ambiguities))
  ) {
    throw new Error("prelabel metadata is invalid");
  }
  assertIntervals(value.rallies, "rallies");
  assertIntervals(value.ignoredIntervals, "ignoredIntervals");
  assertIntervals(value.hardNegatives, "hardNegatives");
  for (const [name, intervals] of [
    ["rallies", value.rallies],
    ["ignoredIntervals", value.ignoredIntervals],
    ["hardNegatives", value.hardNegatives],
  ] as const) {
    if (intervals.some((row) => Number(row.end) > duration)) {
      throw new Error(`${name} cannot exceed recording.durationSeconds`);
    }
  }
  validateRallyMetadata(value.rallies, duration);
  validateIntervalMetadata(value.ignoredIntervals, "ignored");
  validateIntervalMetadata(value.hardNegatives, "negative");
  const serveMarkers = readServeMarkers(value.serveMarkers, duration);
  const sideSwitches = readSideSwitches(value.sideSwitches, duration);
  const courtGeometry = readCourtGeometry(recording.courtGeometry);
  if (value.annotation.status === "complete") validateCompleteCourtGeometry(courtGeometry);
  return {
    ...(value as unknown as LabelDocument),
    recording: {
      ...(recording as unknown as LabelDocument["recording"]),
      ...(courtGeometry === undefined ? {} : { courtGeometry }),
    },
    serveMarkers,
    sideSwitches,
  };
}

export function roundTime(value: number): number {
  return Math.round(value * 1000) / 1000;
}

export function formatPreciseTime(seconds: number): string {
  const safe = Math.max(0, seconds);
  const minutes = Math.floor(safe / 60);
  const remainder = safe - minutes * 60;
  return `${minutes}:${remainder.toFixed(3).padStart(6, "0")}`;
}

export function downloadLabels(document: LabelDocument, complete: boolean): void {
  const annotation = {
    ...document.annotation,
    status: complete ? ("complete" as const) : ("in-progress" as const),
    continuousVideoReviewed: complete,
    reviewedAt: complete ? new Date().toISOString() : document.annotation.reviewedAt,
  };
  const payload = { ...document, annotation };
  const blob = new Blob([`${JSON.stringify(payload, null, 2)}\n`], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = window.document.createElement("a");
  anchor.href = url;
  anchor.download = `${document.recording.id}.labels.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}
