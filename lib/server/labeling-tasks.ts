import { randomUUID } from "node:crypto";
import { mkdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";

import { parseLabelDocument, type LabelDocument } from "@/lib/annotations";

const DEFAULT_MEDIA_ROOT = "/mnt/freenas/volleycut";
const DEFAULT_LABELING_WORKSPACE =
  "/mnt/freenas/volleycut/labeling-v1-2026-08-09";

const mediaRoot = path.resolve(
  /* turbopackIgnore: true */
  process.env.VOLLEYCUT_MEDIA_ROOT ?? DEFAULT_MEDIA_ROOT,
);
const labelingWorkspace = path.resolve(
  /* turbopackIgnore: true */
  process.env.VOLLEYCUT_LABELING_WORKSPACE ?? DEFAULT_LABELING_WORKSPACE,
);
const pilotIndexPath = path.join(
  labelingWorkspace,
  "manifests",
  "pilot-task-index.json",
);

type PilotIndexEntry = {
  priority: number;
  task: string;
  proxy: string;
};

type PilotIndex = {
  schemaVersion: number;
  tasks: PilotIndexEntry[];
};

export type PreparedLabelingTask = {
  id: string;
  priority: number;
  document: LabelDocument;
  draftPath: string;
  originalFilename: string;
  taskPath: string;
  proxyPath: string;
  proxySize: number;
};

export class LabelingTaskNotFoundError extends Error {}
export class LabelingDraftValidationError extends Error {}

export type SavedLabelingDocument = {
  document: LabelDocument;
  savedAt: string | null;
};

function isWithin(parent: string, candidate: string): boolean {
  const relative = path.relative(parent, candidate);
  return (
    relative === "" ||
    (!relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative))
  );
}

function resolveRestrictedPath(
  base: string,
  value: unknown,
  allowedRoot: string,
  field: string,
): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${field} must be a non-empty path`);
  }
  const resolved = path.resolve(base, value);
  if (!isWithin(allowedRoot, resolved)) {
    throw new Error(`${field} resolves outside its allowed media root`);
  }
  return resolved;
}

function isMissingFile(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    (error as { code?: unknown }).code === "ENOENT"
  );
}

function intervalsOverlap(
  left: Array<{ start: number; end: number }>,
  right: Array<{ start: number; end: number }>,
): boolean {
  return left.some((first) =>
    right.some((second) => first.start < second.end && second.start < first.end),
  );
}

function validateDraftContent(document: LabelDocument, task: PreparedLabelingTask): void {
  const base = task.document;
  const immutableValues: Array<[unknown, unknown, string]> = [
    [document.createdAt, base.createdAt, "createdAt"],
    [document.recording.id, base.recording.id, "recording.id"],
    [document.recording.video, base.recording.video, "recording.video"],
    [document.recording.videoFilename, base.recording.videoFilename, "recording.videoFilename"],
    [document.recording.contentSha256, base.recording.contentSha256, "recording.contentSha256"],
    [document.recording.durationSeconds, base.recording.durationSeconds, "recording.durationSeconds"],
    [document.recording.sourceGroup, base.recording.sourceGroup, "recording.sourceGroup"],
    [document.recording.split, base.recording.split, "recording.split"],
    [document.recording.environment, base.recording.environment, "recording.environment"],
    [JSON.stringify(document.recording.capture), JSON.stringify(base.recording.capture), "recording.capture"],
    [JSON.stringify(document.recording.roi), JSON.stringify(base.recording.roi), "recording.roi"],
    [JSON.stringify(document.annotationPolicy), JSON.stringify(base.annotationPolicy), "annotationPolicy"],
  ];
  const changed = immutableValues.find(([actual, expected]) => actual !== expected);
  if (changed) throw new LabelingDraftValidationError(`${changed[2]} cannot be changed`);

  const annotation = document.annotation;
  if (
    !["not-started", "in-progress"].includes(annotation.status) ||
    typeof annotation.annotator !== "string" ||
    typeof annotation.notes !== "string" ||
    typeof annotation.continuousVideoReviewed !== "boolean" ||
    (annotation.reviewedAt !== null && typeof annotation.reviewedAt !== "string")
  ) {
    throw new LabelingDraftValidationError("annotation metadata is invalid for a draft");
  }
  const game = document.recording.game;
  if (
    (game.playersPerTeam !== null &&
      (!Number.isInteger(game.playersPerTeam) || game.playersPerTeam < 1 || game.playersPerTeam > 6)) ||
    (game.targetPoints !== null &&
      (!Number.isInteger(game.targetPoints) || game.targetPoints < 1 || game.targetPoints > 100)) ||
    (game.format !== null && typeof game.format !== "string")
  ) {
    throw new LabelingDraftValidationError("game metadata is invalid");
  }
  const allIntervals = [
    ...document.rallies,
    ...document.ignoredIntervals,
    ...document.hardNegatives,
  ];
  if (allIntervals.some((row) => row.end > document.recording.durationSeconds)) {
    throw new LabelingDraftValidationError("an interval exceeds the video duration");
  }
  if (
    intervalsOverlap(document.rallies, document.ignoredIntervals) ||
    intervalsOverlap(document.rallies, document.hardNegatives) ||
    intervalsOverlap(document.ignoredIntervals, document.hardNegatives)
  ) {
    throw new LabelingDraftValidationError("rallies, ignored spans, and hard negatives cannot overlap");
  }
  if (
    document.rallies.some(
      (row) => !Array.isArray(row.tags) || row.tags.some((tag) => typeof tag !== "string"),
    ) ||
    document.ignoredIntervals.some((row) => typeof row.reason !== "string" || !row.reason) ||
    document.hardNegatives.some((row) => typeof row.category !== "string" || !row.category)
  ) {
    throw new LabelingDraftValidationError("interval metadata is invalid");
  }
}

async function readPilotIndex(): Promise<PilotIndex> {
  const raw = JSON.parse(await readFile(pilotIndexPath, "utf8")) as unknown;
  if (
    typeof raw !== "object" ||
    raw === null ||
    Array.isArray(raw) ||
    (raw as { schemaVersion?: unknown }).schemaVersion !== 1 ||
    !Array.isArray((raw as { tasks?: unknown }).tasks)
  ) {
    throw new Error("pilot task index has an invalid schema");
  }
  return raw as PilotIndex;
}

async function loadEntry(entry: PilotIndexEntry): Promise<PreparedLabelingTask> {
  if (!Number.isInteger(entry.priority) || entry.priority < 1) {
    throw new Error("pilot task priority must be a positive integer");
  }
  const indexDirectory = path.dirname(pilotIndexPath);
  const taskPath = resolveRestrictedPath(
    indexDirectory,
    entry.task,
    labelingWorkspace,
    "task",
  );
  const proxyPath = resolveRestrictedPath(
    indexDirectory,
    entry.proxy,
    mediaRoot,
    "proxy",
  );
  const provenancePath = resolveRestrictedPath(
    mediaRoot,
    `${proxyPath}.provenance.json`,
    mediaRoot,
    "proxy provenance",
  );
  const document = parseLabelDocument(
    JSON.parse(await readFile(taskPath, "utf8")) as unknown,
  );
  const provenance = JSON.parse(await readFile(provenancePath, "utf8")) as unknown;
  const originalFilename =
    typeof provenance === "object" &&
    provenance !== null &&
    !Array.isArray(provenance) &&
    typeof (provenance as { source?: { filename?: unknown } }).source?.filename === "string"
      ? (provenance as { source: { filename: string } }).source.filename
      : "";
  if (!originalFilename || path.basename(originalFilename) !== originalFilename) {
    throw new Error(`task ${document.recording.id} provenance has no valid source filename`);
  }
  const referencedVideo = resolveRestrictedPath(
    path.dirname(taskPath),
    document.recording.video,
    mediaRoot,
    "recording.video",
  );
  if (referencedVideo !== proxyPath) {
    throw new Error(`task ${document.recording.id} does not reference its indexed proxy`);
  }
  if (document.recording.videoFilename !== path.basename(proxyPath)) {
    throw new Error(`task ${document.recording.id} has an unexpected proxy filename`);
  }
  const proxyMetadata = await stat(proxyPath);
  if (!proxyMetadata.isFile() || proxyMetadata.size === 0) {
    throw new Error(`task ${document.recording.id} proxy is unavailable`);
  }
  return {
    id: document.recording.id,
    priority: entry.priority,
    document,
    draftPath: path.join(labelingWorkspace, "labels", "pilot", `${document.recording.id}.labels.json`),
    originalFilename,
    taskPath,
    proxyPath,
    proxySize: proxyMetadata.size,
  };
}

export async function listPreparedLabelingTasks(): Promise<PreparedLabelingTask[]> {
  const index = await readPilotIndex();
  const tasks = await Promise.all(index.tasks.map(loadEntry));
  const ids = new Set<string>();
  for (const task of tasks) {
    if (ids.has(task.id)) throw new Error(`duplicate pilot task id: ${task.id}`);
    ids.add(task.id);
  }
  return tasks.sort((left, right) => left.priority - right.priority);
}

export async function getPreparedLabelingTask(id: string): Promise<PreparedLabelingTask> {
  if (!/^[A-Za-z0-9_-]+$/.test(id)) throw new LabelingTaskNotFoundError();
  const index = await readPilotIndex();
  const expectedFilename = `${id}.labels.json`;
  const entry = index.tasks.find(
    (candidate) =>
      typeof candidate.task === "string" && path.basename(candidate.task) === expectedFilename,
  );
  if (!entry) throw new LabelingTaskNotFoundError();
  const task = await loadEntry(entry);
  if (task.id !== id) throw new LabelingTaskNotFoundError();
  return task;
}

export async function getSavedLabelingDocument(
  task: PreparedLabelingTask,
): Promise<SavedLabelingDocument> {
  try {
    const metadata = await stat(task.draftPath);
    if (!metadata.isFile()) throw new Error("saved draft is not a file");
    const document = parseLabelDocument(
      JSON.parse(await readFile(task.draftPath, "utf8")) as unknown,
    );
    validateDraftContent(document, task);
    return { document, savedAt: metadata.mtime.toISOString() };
  } catch (error) {
    if (isMissingFile(error)) return { document: task.document, savedAt: null };
    throw error;
  }
}

export async function saveLabelingDraft(
  task: PreparedLabelingTask,
  value: unknown,
): Promise<SavedLabelingDocument> {
  let document: LabelDocument;
  try {
    document = parseLabelDocument(value);
  } catch (error) {
    throw new LabelingDraftValidationError(
      error instanceof Error ? error.message : "draft document is invalid",
    );
  }
  validateDraftContent(document, task);
  const draftDirectory = path.dirname(task.draftPath);
  if (!isWithin(labelingWorkspace, task.draftPath)) {
    throw new Error("draft destination is outside the labeling workspace");
  }
  await mkdir(draftDirectory, { recursive: true });
  const temporaryPath = path.join(
    draftDirectory,
    `.${task.id}.${randomUUID()}.labels.json.tmp`,
  );
  try {
    await writeFile(temporaryPath, `${JSON.stringify(document, null, 2)}\n`, {
      encoding: "utf8",
      flag: "wx",
      mode: 0o600,
    });
    await rename(temporaryPath, task.draftPath);
  } finally {
    await rm(temporaryPath, { force: true });
  }
  const metadata = await stat(task.draftPath);
  return { document, savedAt: metadata.mtime.toISOString() };
}
