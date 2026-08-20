import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";

import {
  getServingSideReviewDecisionPath,
  getServingSideReviewReportPath,
} from "./serving-side-review.ts";

const DEFAULT_CORRECTIONS_FILENAME =
  "serving-side-result-label-corrections-v1.json";
const CORRECTION_VALUES = new Set(["near", "far", "not-serve"]);

export type ServingSideCorrection = "near" | "far" | "not-serve";
type ServingSideDecision = "near" | "far";

export type ServingSideCorrectionState = {
  schemaVersion: 1;
  reportKind: string | null;
  reportCreatedAt: string | null;
  baseDecisionSha256: string | null;
  savedAt: string | null;
  corrections: Record<string, ServingSideCorrection>;
};

export class ServingSideCorrectionStoreError extends Error {}
export class ServingSideCorrectionValidationError extends ServingSideCorrectionStoreError {}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function configuredPath(value: string | undefined, fallback: string): string {
  const candidate = path.resolve(value?.trim() || fallback);
  if (candidate === path.parse(candidate).root) {
    throw new ServingSideCorrectionStoreError(
      "The serving-side correction store cannot be a filesystem root",
    );
  }
  return candidate;
}

export function getServingSideCorrectionPath(): string {
  const configured = process.env.VOLLEYCUT_SERVING_SIDE_CORRECTIONS?.trim();
  return configured
    ? configuredPath(configured, DEFAULT_CORRECTIONS_FILENAME)
    : path.join(
        path.dirname(getServingSideReviewReportPath()),
        DEFAULT_CORRECTIONS_FILENAME,
      );
}

function emptyState(): ServingSideCorrectionState {
  return {
    schemaVersion: 1,
    reportKind: null,
    reportCreatedAt: null,
    baseDecisionSha256: null,
    savedAt: null,
    corrections: {},
  };
}

function parseState(value: unknown): ServingSideCorrectionState {
  if (
    !isRecord(value) ||
    value.schemaVersion !== 1 ||
    !isRecord(value.corrections)
  ) {
    throw new ServingSideCorrectionStoreError(
      "The serving-side correction file has an invalid schema",
    );
  }
  const corrections: Record<string, ServingSideCorrection> = {};
  for (const [rallyId, correction] of Object.entries(value.corrections)) {
    if (
      !/^[A-Za-z0-9:_-]+$/.test(rallyId) ||
      typeof correction !== "string" ||
      !CORRECTION_VALUES.has(correction)
    ) {
      throw new ServingSideCorrectionStoreError(
        "The serving-side correction file contains an invalid correction",
      );
    }
    corrections[rallyId] = correction as ServingSideCorrection;
  }
  return {
    schemaVersion: 1,
    reportKind: typeof value.reportKind === "string" ? value.reportKind : null,
    reportCreatedAt:
      typeof value.reportCreatedAt === "string" ? value.reportCreatedAt : null,
    baseDecisionSha256:
      typeof value.baseDecisionSha256 === "string"
        ? value.baseDecisionSha256
        : null,
    savedAt: typeof value.savedAt === "string" ? value.savedAt : null,
    corrections,
  };
}

function sha256(content: string): string {
  return createHash("sha256").update(content).digest("hex");
}

async function readBaseIdentity(): Promise<{
  reportKind: string;
  reportCreatedAt: string;
  baseDecisionSha256: string;
  clearDecisions: Map<string, ServingSideDecision>;
}> {
  let reportText: string;
  let decisionsText: string;
  try {
    [reportText, decisionsText] = await Promise.all([
      readFile(
        /* turbopackIgnore: true */ getServingSideReviewReportPath(),
        "utf8",
      ),
      readFile(
        /* turbopackIgnore: true */ getServingSideReviewDecisionPath(),
        "utf8",
      ),
    ]);
  } catch (error) {
    throw new ServingSideCorrectionStoreError(
      `The frozen serving-side labels could not be read: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  let report: unknown;
  let decisions: unknown;
  try {
    report = JSON.parse(reportText) as unknown;
    decisions = JSON.parse(decisionsText) as unknown;
  } catch {
    throw new ServingSideCorrectionStoreError(
      "A frozen serving-side label artifact contains invalid JSON",
    );
  }
  if (
    !isRecord(report) ||
    typeof report.kind !== "string" ||
    typeof report.createdAt !== "string" ||
    !isRecord(decisions) ||
    !isRecord(decisions.decisions)
  ) {
    throw new ServingSideCorrectionStoreError(
      "A frozen serving-side label artifact has an invalid schema",
    );
  }
  const clearDecisions = new Map<string, ServingSideDecision>();
  for (const [rallyId, decision] of Object.entries(decisions.decisions)) {
    if (decision === "near" || decision === "far") {
      clearDecisions.set(rallyId, decision);
    }
  }
  return {
    reportKind: report.kind,
    reportCreatedAt: report.createdAt,
    baseDecisionSha256: sha256(decisionsText),
    clearDecisions,
  };
}

export async function loadServingSideCorrectionState(): Promise<ServingSideCorrectionState> {
  try {
    return parseState(
      JSON.parse(
        await readFile(
          /* turbopackIgnore: true */ getServingSideCorrectionPath(),
          "utf8",
        ),
      ) as unknown,
    );
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return emptyState();
    if (error instanceof ServingSideCorrectionStoreError) throw error;
    throw new ServingSideCorrectionStoreError(
      "The serving-side correction file could not be read",
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

export async function saveServingSideCorrections(
  value: unknown,
): Promise<ServingSideCorrectionState> {
  if (
    !isRecord(value) ||
    value.schemaVersion !== 1 ||
    !isRecord(value.corrections)
  ) {
    throw new ServingSideCorrectionValidationError(
      "The correction request has an invalid schema",
    );
  }
  const identity = await readBaseIdentity();
  if (
    value.reportKind !== identity.reportKind ||
    value.reportCreatedAt !== identity.reportCreatedAt ||
    value.baseDecisionSha256 !== identity.baseDecisionSha256
  ) {
    throw new ServingSideCorrectionValidationError(
      "The correction request belongs to a different frozen label revision",
    );
  }
  const corrections: Record<string, ServingSideCorrection> = {};
  for (const [rallyId, correction] of Object.entries(value.corrections)) {
    const original = identity.clearDecisions.get(rallyId);
    if (!original) {
      throw new ServingSideCorrectionValidationError(
        `Unknown or unclear serving-side rally: ${rallyId}`,
      );
    }
    if (typeof correction !== "string" || !CORRECTION_VALUES.has(correction)) {
      throw new ServingSideCorrectionValidationError(
        `Invalid correction for serving-side rally: ${rallyId}`,
      );
    }
    if (correction !== original) {
      corrections[rallyId] = correction as ServingSideCorrection;
    }
  }
  const state: ServingSideCorrectionState = {
    schemaVersion: 1,
    reportKind: identity.reportKind,
    reportCreatedAt: identity.reportCreatedAt,
    baseDecisionSha256: identity.baseDecisionSha256,
    savedAt: new Date().toISOString(),
    corrections,
  };
  await atomicReplace(
    getServingSideCorrectionPath(),
    `${JSON.stringify(state, null, 2)}\n`,
  );
  return state;
}
