import { randomUUID } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";

const DEFAULT_LABELING_WORKSPACE =
  "/mnt/freenas/volleycut/labeling-v1-2026-08-09";
const DEFAULT_REPORT_PATH = path.join(
  DEFAULT_LABELING_WORKSPACE,
  "reports",
  "serving-side",
  "serving-side-existing-label-variants-full-nas-v1.json",
);
const DEFAULT_DECISIONS_FILENAME =
  "serving-side-review-decisions-full-nas-v1.json";
const DECISION_VALUES = new Set(["near", "far", "unclear"]);

export type StoredServingSideDecision = "near" | "far" | "unclear";

export type ServingSideReviewState = {
  schemaVersion: 1;
  reportKind: string | null;
  reportCreatedAt: string | null;
  savedAt: string | null;
  decisions: Record<string, StoredServingSideDecision>;
};

export class ServingSideReviewStoreError extends Error {}
export class ServingSideReviewValidationError extends ServingSideReviewStoreError {}

type ServingReportIdentity = {
  kind: string;
  createdAt: string;
  rallyIds: Set<string>;
};

function configuredPath(value: string | undefined, fallback: string): string {
  const candidate = path.resolve(value?.trim() || fallback);
  if (candidate === path.parse(candidate).root) {
    throw new ServingSideReviewStoreError(
      "The serving-side review store cannot be a filesystem root",
    );
  }
  return candidate;
}

export function getServingSideReviewReportPath(): string {
  return configuredPath(
    process.env.VOLLEYCUT_SERVING_SIDE_REPORT,
    DEFAULT_REPORT_PATH,
  );
}

export function getServingSideReviewDecisionPath(): string {
  const configured = process.env.VOLLEYCUT_SERVING_SIDE_DECISIONS?.trim();
  return configured
    ? configuredPath(configured, DEFAULT_DECISIONS_FILENAME)
    : path.join(
        path.dirname(getServingSideReviewReportPath()),
        DEFAULT_DECISIONS_FILENAME,
      );
}

function emptyState(): ServingSideReviewState {
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

function storedState(value: unknown): ServingSideReviewState {
  if (
    !isRecord(value) ||
    value.schemaVersion !== 1 ||
    !isRecord(value.decisions)
  ) {
    throw new ServingSideReviewStoreError(
      "The serving-side decision file has an invalid schema",
    );
  }
  const decisions: Record<string, StoredServingSideDecision> = {};
  for (const [rallyId, decision] of Object.entries(value.decisions)) {
    if (
      !/^[A-Za-z0-9:_-]+$/.test(rallyId) ||
      typeof decision !== "string" ||
      !DECISION_VALUES.has(decision)
    ) {
      throw new ServingSideReviewStoreError(
        "The serving-side decision file contains an invalid decision",
      );
    }
    decisions[rallyId] = decision as StoredServingSideDecision;
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

async function readReportIdentity(): Promise<ServingReportIdentity> {
  let payload: unknown;
  try {
    payload = JSON.parse(
      await readFile(getServingSideReviewReportPath(), "utf8"),
    ) as unknown;
  } catch (error) {
    throw new ServingSideReviewStoreError(
      `The serving-side report could not be read: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  if (
    !isRecord(payload) ||
    typeof payload.kind !== "string" ||
    typeof payload.createdAt !== "string" ||
    !Array.isArray(payload.rallies)
  ) {
    throw new ServingSideReviewStoreError(
      "The serving-side report has an invalid schema",
    );
  }
  const rallyIds = new Set<string>();
  for (const rally of payload.rallies) {
    if (!isRecord(rally) || typeof rally.rallyId !== "string") {
      throw new ServingSideReviewStoreError(
        "The serving-side report contains an invalid rally",
      );
    }
    rallyIds.add(rally.rallyId);
  }
  return { kind: payload.kind, createdAt: payload.createdAt, rallyIds };
}

export async function loadServingSideReviewState(): Promise<ServingSideReviewState> {
  try {
    return storedState(
      JSON.parse(
        await readFile(getServingSideReviewDecisionPath(), "utf8"),
      ) as unknown,
    );
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return emptyState();
    if (error instanceof ServingSideReviewStoreError) throw error;
    throw new ServingSideReviewStoreError(
      "The serving-side decision file could not be read",
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

export async function saveServingSideReviewDecisions(
  value: unknown,
): Promise<ServingSideReviewState> {
  if (
    !isRecord(value) ||
    value.schemaVersion !== 1 ||
    !isRecord(value.decisions)
  ) {
    throw new ServingSideReviewValidationError(
      "The decision request has an invalid schema",
    );
  }
  const identity = await readReportIdentity();
  if (
    value.reportKind !== identity.kind ||
    value.reportCreatedAt !== identity.createdAt
  ) {
    throw new ServingSideReviewValidationError(
      "The decision request belongs to a different serving-side report",
    );
  }
  const decisions: Record<string, StoredServingSideDecision> = {};
  for (const [rallyId, decision] of Object.entries(value.decisions)) {
    if (!identity.rallyIds.has(rallyId)) {
      throw new ServingSideReviewValidationError(
        `Unknown serving-side rally: ${rallyId}`,
      );
    }
    if (typeof decision !== "string" || !DECISION_VALUES.has(decision)) {
      throw new ServingSideReviewValidationError(
        `Invalid decision for serving-side rally: ${rallyId}`,
      );
    }
    decisions[rallyId] = decision as StoredServingSideDecision;
  }
  const state: ServingSideReviewState = {
    schemaVersion: 1,
    reportKind: identity.kind,
    reportCreatedAt: identity.createdAt,
    savedAt: new Date().toISOString(),
    decisions,
  };
  await atomicReplace(
    getServingSideReviewDecisionPath(),
    `${JSON.stringify(state, null, 2)}\n`,
  );
  return state;
}
