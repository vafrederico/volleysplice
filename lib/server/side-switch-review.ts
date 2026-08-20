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
const DEFAULT_DECISIONS_FILENAME = "appearance-review-decisions-full-nas-v1.json";

const DECISION_VALUES = new Set(["switch", "no-switch", "unclear"]);

export type StoredSideSwitchDecision = "switch" | "no-switch" | "unclear";

export type SideSwitchReviewState = {
  schemaVersion: 1;
  reportKind: string | null;
  reportCreatedAt: string | null;
  savedAt: string | null;
  decisions: Record<string, StoredSideSwitchDecision>;
};

export class SideSwitchReviewStoreError extends Error {}
export class SideSwitchReviewValidationError extends SideSwitchReviewStoreError {}

type AppearanceReportIdentity = {
  kind: string;
  createdAt: string;
  eventIds: Set<string>;
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

function emptyState(): SideSwitchReviewState {
  return {
    schemaVersion: 1,
    reportKind: null,
    reportCreatedAt: null,
    savedAt: null,
    decisions: {},
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function storedState(value: unknown): SideSwitchReviewState {
  if (!isRecord(value) || value.schemaVersion !== 1 || !isRecord(value.decisions)) {
    throw new SideSwitchReviewStoreError(
      "The side-switch decision file has an invalid schema",
    );
  }
  const decisions: Record<string, StoredSideSwitchDecision> = {};
  for (const [eventId, decision] of Object.entries(value.decisions)) {
    if (!/^[A-Za-z0-9:_-]+$/.test(eventId) || typeof decision !== "string" || !DECISION_VALUES.has(decision)) {
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

async function readReportIdentity(): Promise<AppearanceReportIdentity> {
  let payload: unknown;
  try {
    payload = JSON.parse(
      await readFile(getSideSwitchReviewReportPath(), "utf8"),
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
  for (const event of payload.events) {
    if (!isRecord(event) || typeof event.eventId !== "string") {
      throw new SideSwitchReviewStoreError(
        "The side-switch diagnostic report contains an invalid event",
      );
    }
    eventIds.add(event.eventId);
  }
  return { kind: payload.kind, createdAt: payload.createdAt, eventIds };
}

export async function loadSideSwitchReviewState(): Promise<SideSwitchReviewState> {
  try {
    return storedState(
      JSON.parse(await readFile(getSideSwitchReviewDecisionPath(), "utf8")) as unknown,
    );
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return emptyState();
    if (error instanceof SideSwitchReviewStoreError) throw error;
    throw new SideSwitchReviewStoreError(
      "The side-switch decision file could not be read",
    );
  }
}

async function atomicReplace(destination: string, content: string): Promise<void> {
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
  if (!isRecord(value) || value.schemaVersion !== 1 || !isRecord(value.decisions)) {
    throw new SideSwitchReviewValidationError("The decision request has an invalid schema");
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
