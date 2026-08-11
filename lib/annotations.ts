export const annotationPolicyId = "serve-contact-to-dead-ball-v1";

export type RallyLabel = {
  start: number;
  end: number;
  tags: string[];
  notes?: string;
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
  sideSwitches: SideSwitch[];
};

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
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

function readSideSwitches(value: unknown): SideSwitch[] {
  if (value === undefined) return [];
  if (!Array.isArray(value)) throw new Error("sideSwitches must be an array");
  let previousTime = -1;
  value.forEach((row, index) => {
    if (
      !isObject(row) ||
      !isFiniteNumber(row.time) ||
      row.time < 0 ||
      row.time <= previousTime ||
      (row.notes !== undefined && typeof row.notes !== "string")
    ) {
      throw new Error(
        `sideSwitches[${index}] must have an ordered, non-negative finite time and optional notes`,
      );
    }
    previousTime = row.time;
  });
  return value as SideSwitch[];
}

export function parseLabelDocument(value: unknown): LabelDocument {
  if (!isObject(value) || value.schemaVersion !== 1 || value.kind !== "volleycut-rally-labels") {
    throw new Error("This is not a VolleyCut rally-label document (schema version 1)");
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
  const sideSwitches = readSideSwitches(value.sideSwitches);
  return { ...(value as unknown as LabelDocument), sideSwitches };
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
