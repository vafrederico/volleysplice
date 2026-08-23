import { randomUUID } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";

const DEFAULT_LABELING_WORKSPACE =
  "/mnt/freenas/volleycut/labeling-v1-2026-08-09";
const DEFAULT_REPORT_PATH = path.join(
  DEFAULT_LABELING_WORKSPACE,
  "reports",
  "side-switch",
  "appearance-diagnostic-full-nas-v1.json",
);
const DEFAULT_DECISIONS_FILENAME =
  "appearance-review-decisions-full-nas-v1.json";
const DEFAULT_MARKERS_FILENAME =
  "full-video-side-switch-markers-full-nas-v1.json";
const MARKER_STORE_KIND = "volleycut-full-video-side-switch-markers-v1";
const MAX_MARKERS = 5_000;

const DECISION_VALUES = new Set(["switch", "no-switch", "unclear"]);

export type StoredSideSwitchDecision = "switch" | "no-switch" | "unclear";

export type SideSwitchReviewState = {
  schemaVersion: 1;
  reportKind: string | null;
  reportCreatedAt: string | null;
  savedAt: string | null;
  decisions: Record<string, StoredSideSwitchDecision>;
};

export type StoredFullVideoSideSwitchMarker = {
  id: string;
  recordingId: string;
  time: number;
  createdAt: string;
};

export type FullVideoSideSwitchMarkerState = {
  schemaVersion: 1;
  kind: typeof MARKER_STORE_KIND;
  reportKind: string | null;
  reportCreatedAt: string | null;
  savedAt: string | null;
  markers: StoredFullVideoSideSwitchMarker[];
  reviewedRecordingIds: string[];
};

export class SideSwitchReviewStoreError extends Error {}
export class SideSwitchReviewValidationError extends SideSwitchReviewStoreError {}

type AppearanceReportIdentity = {
  kind: string;
  createdAt: string;
  eventIds: Set<string>;
  recordingDurations: Map<string, number>;
};

function configuredPath(value: string | undefined, fallback: string): string {
  const candidate = path.resolve(value?.trim() || fallback);
  if (candidate === path.parse(candidate).root) {
    throw new SideSwitchReviewStoreError(
      "The side-switch review store cannot be a filesystem root",
    );
  }
  return candidate;
}

export function getSideSwitchReviewReportPath(): string {
  return configuredPath(
    process.env.VOLLEYCUT_SIDE_SWITCH_REPORT,
    DEFAULT_REPORT_PATH,
  );
}

export function getSideSwitchReviewDecisionPath(): string {
  const configured = process.env.VOLLEYCUT_SIDE_SWITCH_DECISIONS?.trim();
  return configured
    ? configuredPath(configured, DEFAULT_DECISIONS_FILENAME)
    : path.join(
        path.dirname(getSideSwitchReviewReportPath()),
        DEFAULT_DECISIONS_FILENAME,
      );
}

export function getFullVideoSideSwitchMarkerPath(): string {
  const configured = process.env.VOLLEYCUT_SIDE_SWITCH_MARKERS?.trim();
  return configured
    ? configuredPath(configured, DEFAULT_MARKERS_FILENAME)
    : path.join(
        path.dirname(getSideSwitchReviewReportPath()),
        DEFAULT_MARKERS_FILENAME,
      );
}

function emptyState(): SideSwitchReviewState {
  return {
    schemaVersion: 1,
    reportKind: null,
    reportCreatedAt: null,
    savedAt: null,
    decisions: {},
  };
}

function emptyMarkerState(): FullVideoSideSwitchMarkerState {
  return {
    schemaVersion: 1,
    kind: MARKER_STORE_KIND,
    reportKind: null,
    reportCreatedAt: null,
    savedAt: null,
    markers: [],
    reviewedRecordingIds: [],
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function storedState(value: unknown): SideSwitchReviewState {
  if (
    !isRecord(value) ||
    value.schemaVersion !== 1 ||
    !isRecord(value.decisions)
  ) {
    throw new SideSwitchReviewStoreError(
      "The side-switch decision file has an invalid schema",
    );
  }
  const decisions: Record<string, StoredSideSwitchDecision> = {};
  for (const [eventId, decision] of Object.entries(value.decisions)) {
    if (
      !/^[A-Za-z0-9:_-]+$/.test(eventId) ||
      typeof decision !== "string" ||
      !DECISION_VALUES.has(decision)
    ) {
      throw new SideSwitchReviewStoreError(
        "The side-switch decision file contains an invalid decision",
      );
    }
    decisions[eventId] = decision as StoredSideSwitchDecision;
  }
  return {
    schemaVersion: 1,
    reportKind: typeof value.reportKind === "string" ? value.reportKind : null,
    reportCreatedAt:
      typeof value.reportCreatedAt === "string" ? value.reportCreatedAt : null,
    savedAt: typeof value.savedAt === "string" ? value.savedAt : null,
    decisions,
  };
}

function validMarkerId(value: unknown): value is string {
  return (
    typeof value === "string" &&
    value.length > 0 &&
    value.length <= 240 &&
    /^[A-Za-z0-9:_-]+$/.test(value)
  );
}

function storedMarkerState(value: unknown): FullVideoSideSwitchMarkerState {
  if (
    !isRecord(value) ||
    value.schemaVersion !== 1 ||
    value.kind !== MARKER_STORE_KIND ||
    !Array.isArray(value.markers) ||
    !Array.isArray(value.reviewedRecordingIds)
  ) {
    throw new SideSwitchReviewStoreError(
      "The full-video side-switch marker file has an invalid schema",
    );
  }
  if (value.markers.length > MAX_MARKERS) {
    throw new SideSwitchReviewStoreError(
      "The full-video side-switch marker file contains too many markers",
    );
  }
  const ids = new Set<string>();
  const markers = value.markers.map((marker) => {
    if (
      !isRecord(marker) ||
      !validMarkerId(marker.id) ||
      typeof marker.recordingId !== "string" ||
      marker.recordingId.length === 0 ||
      typeof marker.time !== "number" ||
      !Number.isFinite(marker.time) ||
      marker.time < 0 ||
      typeof marker.createdAt !== "string" ||
      !Number.isFinite(Date.parse(marker.createdAt)) ||
      ids.has(marker.id)
    ) {
      throw new SideSwitchReviewStoreError(
        "The full-video side-switch marker file contains an invalid marker",
      );
    }
    ids.add(marker.id);
    return {
      id: marker.id,
      recordingId: marker.recordingId,
      time: marker.time,
      createdAt: marker.createdAt,
    };
  });
  const reviewedRecordingIds = value.reviewedRecordingIds.map((recordingId) => {
    if (typeof recordingId !== "string" || recordingId.length === 0) {
      throw new SideSwitchReviewStoreError(
        "The full-video side-switch marker file contains an invalid reviewed recording",
      );
    }
    return recordingId;
  });
  if (new Set(reviewedRecordingIds).size !== reviewedRecordingIds.length) {
    throw new SideSwitchReviewStoreError(
      "The full-video side-switch marker file repeats a reviewed recording",
    );
  }
  return {
    schemaVersion: 1,
    kind: MARKER_STORE_KIND,
    reportKind: typeof value.reportKind === "string" ? value.reportKind : null,
    reportCreatedAt:
      typeof value.reportCreatedAt === "string" ? value.reportCreatedAt : null,
    savedAt: typeof value.savedAt === "string" ? value.savedAt : null,
    markers,
    reviewedRecordingIds,
  };
}

async function readReportIdentity(): Promise<AppearanceReportIdentity> {
  let payload: unknown;
  try {
    payload = JSON.parse(
      await readFile(
        /* turbopackIgnore: true */ getSideSwitchReviewReportPath(),
        "utf8",
      ),
    ) as unknown;
  } catch (error) {
    throw new SideSwitchReviewStoreError(
      `The side-switch diagnostic report could not be read: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  if (
    !isRecord(payload) ||
    typeof payload.kind !== "string" ||
    typeof payload.createdAt !== "string" ||
    !isRecord(payload.summary) ||
    !Array.isArray(payload.events)
  ) {
    throw new SideSwitchReviewStoreError(
      "The side-switch diagnostic report has an invalid schema",
    );
  }
  const eventIds = new Set<string>();
  const fallbackDurations = new Map<string, number>();
  for (const event of payload.events) {
    if (
      !isRecord(event) ||
      typeof event.eventId !== "string" ||
      typeof event.recordingId !== "string"
    ) {
      throw new SideSwitchReviewStoreError(
        "The side-switch diagnostic report contains an invalid event",
      );
    }
    eventIds.add(event.eventId);
    const end =
      typeof event.gapEnd === "number" && Number.isFinite(event.gapEnd)
        ? event.gapEnd
        : typeof event.transitionTime === "number" &&
            Number.isFinite(event.transitionTime)
          ? event.transitionTime
          : 0;
    fallbackDurations.set(
      event.recordingId,
      Math.max(fallbackDurations.get(event.recordingId) ?? 0, end + 5),
    );
  }
  if (!isRecord(payload.labels) || !Array.isArray(payload.labels.files)) {
    throw new SideSwitchReviewStoreError(
      "The side-switch diagnostic report has invalid recording metadata",
    );
  }
  const recordingDurations = new Map<string, number>();
  for (const file of payload.labels.files) {
    if (!isRecord(file) || typeof file.recordingId !== "string") {
      throw new SideSwitchReviewStoreError(
        "The side-switch diagnostic report contains an invalid recording",
      );
    }
    const duration =
      typeof file.durationSeconds === "number" &&
      Number.isFinite(file.durationSeconds) &&
      file.durationSeconds > 0
        ? file.durationSeconds
        : (fallbackDurations.get(file.recordingId) ?? 0);
    if (duration <= 0) {
      throw new SideSwitchReviewStoreError(
        `The side-switch diagnostic report has no duration for ${file.recordingId}`,
      );
    }
    recordingDurations.set(file.recordingId, duration);
  }
  return {
    kind: payload.kind,
    createdAt: payload.createdAt,
    eventIds,
    recordingDurations,
  };
}

export async function loadSideSwitchReviewState(): Promise<SideSwitchReviewState> {
  try {
    return storedState(
      JSON.parse(
        await readFile(getSideSwitchReviewDecisionPath(), "utf8"),
      ) as unknown,
    );
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return emptyState();
    if (error instanceof SideSwitchReviewStoreError) throw error;
    throw new SideSwitchReviewStoreError(
      "The side-switch decision file could not be read",
    );
  }
}

export async function loadFullVideoSideSwitchMarkerState(): Promise<FullVideoSideSwitchMarkerState> {
  try {
    return storedMarkerState(
      JSON.parse(
        await readFile(getFullVideoSideSwitchMarkerPath(), "utf8"),
      ) as unknown,
    );
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT")
      return emptyMarkerState();
    if (error instanceof SideSwitchReviewStoreError) throw error;
    throw new SideSwitchReviewStoreError(
      "The full-video side-switch marker file could not be read",
    );
  }
}

async function atomicReplace(
  destination: string,
  content: string,
): Promise<void> {
  const directory = path.dirname(destination);
  await mkdir(directory, { recursive: true });
  const temporary = path.join(
    directory,
    `.${path.basename(destination)}.${randomUUID()}.tmp`,
  );
  try {
    await writeFile(temporary, content, {
      encoding: "utf8",
      flag: "wx",
      mode: 0o600,
    });
    await rename(temporary, destination);
  } finally {
    await rm(temporary, { force: true });
  }
}

export async function saveSideSwitchReviewDecisions(
  value: unknown,
): Promise<SideSwitchReviewState> {
  if (
    !isRecord(value) ||
    value.schemaVersion !== 1 ||
    !isRecord(value.decisions)
  ) {
    throw new SideSwitchReviewValidationError(
      "The decision request has an invalid schema",
    );
  }
  const identity = await readReportIdentity();
  if (
    value.reportKind !== identity.kind ||
    value.reportCreatedAt !== identity.createdAt
  ) {
    throw new SideSwitchReviewValidationError(
      "The decision request belongs to a different diagnostic report",
    );
  }
  const decisions: Record<string, StoredSideSwitchDecision> = {};
  for (const [eventId, decision] of Object.entries(value.decisions)) {
    if (!identity.eventIds.has(eventId)) {
      throw new SideSwitchReviewValidationError(
        `Unknown side-switch event: ${eventId}`,
      );
    }
    if (typeof decision !== "string" || !DECISION_VALUES.has(decision)) {
      throw new SideSwitchReviewValidationError(
        `Invalid decision for side-switch event: ${eventId}`,
      );
    }
    decisions[eventId] = decision as StoredSideSwitchDecision;
  }
  const state: SideSwitchReviewState = {
    schemaVersion: 1,
    reportKind: identity.kind,
    reportCreatedAt: identity.createdAt,
    savedAt: new Date().toISOString(),
    decisions,
  };
  await atomicReplace(
    getSideSwitchReviewDecisionPath(),
    `${JSON.stringify(state, null, 2)}\n`,
  );
  return state;
}

export async function saveFullVideoSideSwitchMarkers(
  value: unknown,
): Promise<FullVideoSideSwitchMarkerState> {
  if (
    !isRecord(value) ||
    value.schemaVersion !== 1 ||
    value.kind !== MARKER_STORE_KIND ||
    !Array.isArray(value.markers) ||
    !Array.isArray(value.reviewedRecordingIds) ||
    value.markers.length > MAX_MARKERS
  ) {
    throw new SideSwitchReviewValidationError(
      "The full-video marker request has an invalid schema",
    );
  }
  const identity = await readReportIdentity();
  if (
    value.reportKind !== identity.kind ||
    value.reportCreatedAt !== identity.createdAt
  ) {
    throw new SideSwitchReviewValidationError(
      "The full-video marker request belongs to a different diagnostic report",
    );
  }
  const ids = new Set<string>();
  const markers: StoredFullVideoSideSwitchMarker[] = [];
  for (const marker of value.markers) {
    if (
      !isRecord(marker) ||
      !validMarkerId(marker.id) ||
      typeof marker.recordingId !== "string" ||
      !identity.recordingDurations.has(marker.recordingId) ||
      typeof marker.time !== "number" ||
      !Number.isFinite(marker.time) ||
      marker.time < 0 ||
      marker.time >
        (identity.recordingDurations.get(marker.recordingId) ?? 0) ||
      typeof marker.createdAt !== "string" ||
      !Number.isFinite(Date.parse(marker.createdAt)) ||
      ids.has(marker.id)
    ) {
      throw new SideSwitchReviewValidationError(
        "The full-video marker request contains an invalid marker",
      );
    }
    ids.add(marker.id);
    markers.push({
      id: marker.id,
      recordingId: marker.recordingId,
      time: Math.round(marker.time * 1000) / 1000,
      createdAt: marker.createdAt,
    });
  }
  markers.sort(
    (left, right) =>
      left.recordingId.localeCompare(right.recordingId) ||
      left.time - right.time ||
      left.id.localeCompare(right.id),
  );
  const reviewedRecordingIds = value.reviewedRecordingIds.map((recordingId) => {
    if (
      typeof recordingId !== "string" ||
      !identity.recordingDurations.has(recordingId)
    ) {
      throw new SideSwitchReviewValidationError(
        "The full-video marker request contains an unknown reviewed recording",
      );
    }
    return recordingId;
  });
  if (new Set(reviewedRecordingIds).size !== reviewedRecordingIds.length) {
    throw new SideSwitchReviewValidationError(
      "The full-video marker request repeats a reviewed recording",
    );
  }
  reviewedRecordingIds.sort((left, right) => left.localeCompare(right));
  const state: FullVideoSideSwitchMarkerState = {
    schemaVersion: 1,
    kind: MARKER_STORE_KIND,
    reportKind: identity.kind,
    reportCreatedAt: identity.createdAt,
    savedAt: new Date().toISOString(),
    markers,
    reviewedRecordingIds,
  };
  await atomicReplace(
    getFullVideoSideSwitchMarkerPath(),
    `${JSON.stringify(state, null, 2)}\n`,
  );
  return state;
}
