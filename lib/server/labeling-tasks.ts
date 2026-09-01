import { randomUUID } from "node:crypto";
import { mkdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";

import {
  endObservabilityValues,
  hardNegativeCategories,
  parseLabelDocument,
  terminalCueValues,
  type LabelDocument,
  type NormalizedPoint,
  type RallyLabel,
  type ServeMarker,
  type SideSwitch,
} from "@/lib/annotations";
import {
  buildProductionEnsembleLabelSeed,
  buildProductionLabelSeed,
  type ProductionLabelSeed,
} from "@/lib/production-label-seed";
import {
  PREVIOUS_PRODUCTION_MODEL_ID,
  PREVIOUS_PRODUCTION_MODEL_LABEL,
  PRODUCTION_ENSEMBLE_MODEL_DESCRIPTION,
  PRODUCTION_ENSEMBLE_MODEL_ID,
  PRODUCTION_MODEL_LABEL,
  PRODUCTION_MODEL_DESCRIPTION,
  PRODUCTION_MODEL_ID,
} from "@/lib/production-model";
import { ENVIRONMENT_EXPERIMENT_MODELS } from "@/lib/experiment-models";
import {
  getAnalysesRoot,
  getIntakeAnalysesRoot,
  getIntakeWorkspaces,
} from "@/lib/storage";

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
const manifestsDirectory = path.join(labelingWorkspace, "manifests");
const pilotIndexPath = path.join(manifestsDirectory, "pilot-task-index.json");
const fullPlanPath = path.join(manifestsDirectory, "full-corpus-plan.json");
const suppressionExperiment = "feedback-suppression-v3-corrected-2026-08-18";
const suppressionVariant = "production-plus-suppression-zero-non-exempt-misses";
const servingSideInferencePath = path.resolve(
  /* turbopackIgnore: true */
  process.env.VOLLEYCUT_SERVING_SIDE_INFERENCE_PATH ??
    path.join(
      DEFAULT_LABELING_WORKSPACE,
      "reports/serving-side/serving-side-flight-v3-hybrid-serve-gate-all-video-inference-v2.json",
    ),
);

export type LabelingBatch = "pilot" | "full";

type PilotIndexEntry = {
  priority: number;
  task: string;
  proxy: string;
};

type PilotIndex = {
  schemaVersion: number;
  tasks: PilotIndexEntry[];
};

type FullPlanRow = {
  id: string;
  environment: LabelDocument["recording"]["environment"];
};

type FullPlan = {
  schemaVersion: number;
  recordings: FullPlanRow[];
};

type LabelingTaskEntry = {
  id: string;
  batch: LabelingBatch;
  priority: number;
  taskPath: string;
  proxyPath: string;
  workspaceRoot: string;
  draftPath: string;
  prelabelPath: string;
  completedPath: string;
};

export type PreparedLabelingTask = {
  id: string;
  batch: LabelingBatch;
  priority: number;
  document: LabelDocument;
  workspaceRoot: string;
  draftPath: string;
  prelabelPath: string;
  completedPath: string;
  originalFilename: string;
  taskPath: string;
  proxyPath: string;
  proxySize: number;
};

export type PreparedLabelingCatalog = {
  tasks: PreparedLabelingTask[];
  totals: Record<LabelingBatch, number>;
};

export class LabelingTaskNotFoundError extends Error {}
export class LabelingDraftValidationError extends Error {}

export type SavedLabelingDocument = {
  document: LabelDocument;
  source: "draft" | "completed" | "production-model" | "prelabel" | "task";
  savedAt: string | null;
};

export type SolReferenceLabels = {
  rallies: LabelDocument["rallies"];
};

export type ProductionReferenceLabels = SolReferenceLabels & {
  modelId: string;
  modelLabel: string;
  description?: string;
  serveMarkers?: ServeMarker[];
  humanServeMarkers?: ServeMarker[];
  serveModelLabel?: string;
  sideSwitches?: SideSwitch[];
  sideSwitchModelLabel?: string;
  suppressedRanges?: RallyLabel[];
};

export type ExperimentModelReferenceLabels = ProductionReferenceLabels & {
  description: string;
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

type JsonObject = Record<string, unknown>;

function jsonObject(value: unknown): JsonObject | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as JsonObject
    : null;
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

async function isFile(filePath: string): Promise<boolean> {
  try {
    return (await stat(filePath)).isFile();
  } catch (error) {
    if (isMissingFile(error)) return false;
    throw error;
  }
}

function taskIdFromPath(taskPath: string): string {
  const suffix = ".labels.json";
  const filename = path.basename(taskPath);
  if (!filename.endsWith(suffix)) throw new Error("task path must end in .labels.json");
  const id = filename.slice(0, -suffix.length);
  if (!/^[A-Za-z0-9_-]+$/.test(id)) throw new Error("task id contains unsafe characters");
  return id;
}

function intervalsOverlap(
  left: Array<{ start: number; end: number }>,
  right: Array<{ start: number; end: number }>,
): boolean {
  return left.some((first) =>
    right.some((second) => first.start < second.end && second.start < first.end),
  );
}

const hardNegativeCategorySet = new Set<string>(hardNegativeCategories);
const terminalCueSet = new Set<string>(terminalCueValues);
const endObservabilitySet = new Set<string>(endObservabilityValues);

function isNormalizedPoint(point: NormalizedPoint | undefined): boolean {
  return (
    point === undefined ||
    (Number.isFinite(point.x) &&
      Number.isFinite(point.y) &&
      point.x >= 0 &&
      point.x <= 1 &&
      point.y >= 0 &&
      point.y <= 1)
  );
}

function validateDraftContent(
  document: LabelDocument,
  task: PreparedLabelingTask,
  expectedStatus: "draft" | "complete" = "draft",
): void {
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
    (expectedStatus === "draft"
      ? !["not-started", "in-progress"].includes(annotation.status)
      : annotation.status !== "complete") ||
    typeof annotation.annotator !== "string" ||
    typeof annotation.notes !== "string" ||
    typeof annotation.continuousVideoReviewed !== "boolean" ||
    (expectedStatus === "complete"
      ? annotation.continuousVideoReviewed !== true ||
        typeof annotation.reviewedAt !== "string" ||
        annotation.reviewedAt.trim().length === 0
      : annotation.reviewedAt !== null && typeof annotation.reviewedAt !== "string")
  ) {
    throw new LabelingDraftValidationError(
      `annotation metadata is invalid for a ${expectedStatus === "complete" ? "completed label" : "draft"}`,
    );
  }
  const geometry = document.recording.courtGeometry;
  if (
    geometry &&
    [
      ...Object.values(geometry.corners),
      ...Object.values(geometry.netAnchors ?? {}),
      ...Object.values(geometry.serviceZoneAnchors ?? {}),
    ].some((point) => !isNormalizedPoint(point))
  ) {
    throw new LabelingDraftValidationError(
      "court geometry anchors must be finite normalized frame points",
    );
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
    document.serveMarkers.some(
      (marker, index) =>
        !Number.isFinite(marker.time) ||
        marker.time < 0 ||
        marker.time > document.recording.durationSeconds ||
        (index > 0 && marker.time <= document.serveMarkers[index - 1].time) ||
        !["near", "far", "review"].includes(marker.side) ||
        (marker.notes !== undefined && typeof marker.notes !== "string") ||
        (marker.modelConfidence !== undefined &&
          (!Number.isFinite(marker.modelConfidence) ||
            marker.modelConfidence < 0 ||
            marker.modelConfidence > 1)),
    )
  ) {
    throw new LabelingDraftValidationError(
      "serve markers must be finite, in range, strictly ordered points with valid serving sides",
    );
  }
  if (
    expectedStatus === "complete" &&
    document.serveMarkers.some((marker) => marker.side === "review")
  ) {
    throw new LabelingDraftValidationError(
      "completed labels must resolve every serving-side marker to near or far",
    );
  }
  if (
    document.sideSwitches.some(
      (marker, index) =>
        !Number.isFinite(marker.time) ||
        marker.time < 0 ||
        marker.time > document.recording.durationSeconds ||
        (index > 0 && marker.time <= document.sideSwitches[index - 1].time) ||
        (marker.notes !== undefined && typeof marker.notes !== "string"),
    )
  ) {
    throw new LabelingDraftValidationError(
      "side switches must be finite, in range, strictly ordered points with optional notes",
    );
  }
  if (intervalsOverlap(document.rallies, document.hardNegatives)) {
    throw new LabelingDraftValidationError("rallies and hard negatives cannot overlap");
  }
  if (
    document.rallies.some(
      (row) => !Array.isArray(row.tags) || row.tags.some((tag) => typeof tag !== "string"),
    ) ||
    document.ignoredIntervals.some((row) => typeof row.reason !== "string" || !row.reason) ||
    document.hardNegatives.some(
      (row) =>
        typeof row.category !== "string" ||
        !hardNegativeCategorySet.has(row.category),
    )
  ) {
    throw new LabelingDraftValidationError("interval metadata is invalid");
  }
  if (
    document.rallies.some((row) => {
      const reaction = row.receiverReactionTime;
      const standDown = row.collectiveStandDownTime;
      return (
        (reaction !== undefined &&
          (!Number.isFinite(reaction) ||
            reaction < row.start ||
            reaction > Math.min(row.end, row.start + 5))) ||
        (standDown !== undefined &&
          (!Number.isFinite(standDown) ||
            standDown < Math.max(row.start, row.end - 5) ||
            standDown > Math.min(document.recording.durationSeconds, row.end + 5))) ||
        (reaction !== undefined && standDown !== undefined && reaction > standDown) ||
        (row.startConfidence !== undefined &&
          (!Number.isFinite(row.startConfidence) ||
            row.startConfidence < 0 ||
            row.startConfidence > 1)) ||
        (row.endConfidence !== undefined &&
          (!Number.isFinite(row.endConfidence) ||
            row.endConfidence < 0 ||
            row.endConfidence > 1)) ||
        (row.terminalCue !== undefined && !terminalCueSet.has(row.terminalCue)) ||
        (row.endObservability !== undefined &&
          !endObservabilitySet.has(row.endObservability)) ||
        (row.verifiedImmediateResult !== undefined &&
          typeof row.verifiedImmediateResult !== "boolean")
      );
    })
  ) {
    throw new LabelingDraftValidationError("rally transition metadata is invalid");
  }
}

async function readPilotEntries(): Promise<LabelingTaskEntry[]> {
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
  const index = raw as PilotIndex;
  const indexDirectory = path.dirname(pilotIndexPath);
  return index.tasks.map((entry) => {
    if (!Number.isInteger(entry.priority) || entry.priority < 1) {
      throw new Error("pilot task priority must be a positive integer");
    }
    const taskPath = resolveRestrictedPath(
      indexDirectory,
      entry.task,
      labelingWorkspace,
      "pilot task",
    );
    return {
      id: taskIdFromPath(taskPath),
      batch: "pilot",
      priority: entry.priority,
      taskPath,
      proxyPath: resolveRestrictedPath(
        indexDirectory,
        entry.proxy,
        mediaRoot,
        "pilot proxy",
      ),
      workspaceRoot: labelingWorkspace,
      draftPath: path.join(
        labelingWorkspace,
        "labels",
        "pilot",
        `${taskIdFromPath(taskPath)}.labels.json`,
      ),
      prelabelPath: path.join(
        labelingWorkspace,
        "prelabels",
        "sol-xhigh",
        `${taskIdFromPath(taskPath)}.labels.json`,
      ),
      completedPath: path.join(
        labelingWorkspace,
        "completed",
        "pilot-v1",
        `${taskIdFromPath(taskPath)}.labels.json`,
      ),
    };
  });
}

type FullWorkspace = {
  root: string;
  planPath: string;
  prelabelsDirectory: string;
};

async function readFullEntries(workspace: FullWorkspace): Promise<LabelingTaskEntry[]> {
  const raw = JSON.parse(await readFile(workspace.planPath, "utf8")) as unknown;
  if (
    typeof raw !== "object" ||
    raw === null ||
    Array.isArray(raw) ||
    (raw as { schemaVersion?: unknown }).schemaVersion !== 1 ||
    !Array.isArray((raw as { recordings?: unknown }).recordings)
  ) {
    throw new Error("full-corpus plan has an invalid schema");
  }
  const plan = raw as FullPlan;
  return plan.recordings.map((row, index) => {
    if (
      typeof row.id !== "string" ||
      !/^[A-Za-z0-9_-]+$/.test(row.id) ||
      !["indoor", "beach", "grass", "broadcast", "unknown"].includes(row.environment)
    ) {
      throw new Error(`full-corpus plan recording ${index + 1} is invalid`);
    }
    const proxyFilename = row.id.endsWith("-full") ? `${row.id}.mp4` : `${row.id}-full.mp4`;
    return {
      id: row.id,
      batch: "full",
      priority: index + 1,
      taskPath: resolveRestrictedPath(
        workspace.root,
        path.join("tasks", "full", `${row.id}.labels.json`),
        workspace.root,
        "full task",
      ),
      proxyPath: resolveRestrictedPath(
        workspace.root,
        path.join("proxies", row.environment, proxyFilename),
        mediaRoot,
        "full proxy",
      ),
      workspaceRoot: workspace.root,
      draftPath: path.join(workspace.root, "labels", "full", `${row.id}.labels.json`),
      prelabelPath: path.join(workspace.prelabelsDirectory, `${row.id}.labels.json`),
      completedPath: path.join(
        workspace.root,
        "completed",
        "full-v1",
        `${row.id}.labels.json`,
      ),
    };
  });
}

async function readAllEntries(): Promise<LabelingTaskEntry[]> {
  const intakeWorkspaces = getIntakeWorkspaces();
  const [pilot, full, intakePlans] = await Promise.all([
    readPilotEntries(),
    readFullEntries({
      root: labelingWorkspace,
      planPath: fullPlanPath,
      prelabelsDirectory: path.join(labelingWorkspace, "prelabels", "sol-xhigh"),
    }),
    Promise.all(intakeWorkspaces.map(async (root) => {
      const standard = path.join(root, "manifests", "intake-plan.json");
      const inferenceOnly = path.join(root, "manifests", "inference-only.json");
      return {
        root,
        planPath: await isFile(standard)
          ? standard
          : await isFile(inferenceOnly)
            ? inferenceOnly
            : null,
      };
    })),
  ]);
  const intake = (await Promise.all(intakePlans.map(async ({ root, planPath }) =>
    planPath
      ? readFullEntries({
          root,
          planPath,
          prelabelsDirectory: path.join(root, "blind-sol", "prelabels"),
        })
      : []
  ))).flat();
  const entries = [...full, ...intake, ...pilot];
  const ids = new Set<string>();
  for (const entry of entries) {
    if (ids.has(entry.id)) throw new Error(`duplicate prepared task id: ${entry.id}`);
    ids.add(entry.id);
  }
  return entries;
}

async function loadEntry(entry: LabelingTaskEntry): Promise<PreparedLabelingTask> {
  const provenancePath = resolveRestrictedPath(
    mediaRoot,
    `${entry.proxyPath}.provenance.json`,
    mediaRoot,
    "proxy provenance",
  );
  const document = parseLabelDocument(
    JSON.parse(await readFile(entry.taskPath, "utf8")) as unknown,
  );
  if (document.recording.id !== entry.id) {
    throw new Error(`task ${entry.id} document has an unexpected recording id`);
  }
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
    path.dirname(entry.taskPath),
    document.recording.video,
    mediaRoot,
    "recording.video",
  );
  if (referencedVideo !== entry.proxyPath) {
    throw new Error(`task ${document.recording.id} does not reference its indexed proxy`);
  }
  if (document.recording.videoFilename !== path.basename(entry.proxyPath)) {
    throw new Error(`task ${document.recording.id} has an unexpected proxy filename`);
  }
  const proxyMetadata = await stat(entry.proxyPath);
  if (!proxyMetadata.isFile() || proxyMetadata.size === 0) {
    throw new Error(`task ${document.recording.id} proxy is unavailable`);
  }
  return {
    id: document.recording.id,
    batch: entry.batch,
    priority: entry.priority,
    document,
    workspaceRoot: entry.workspaceRoot,
    draftPath: entry.draftPath,
    prelabelPath: entry.prelabelPath,
    completedPath: entry.completedPath,
    originalFilename,
    taskPath: entry.taskPath,
    proxyPath: entry.proxyPath,
    proxySize: proxyMetadata.size,
  };
}

async function loadAvailableEntry(
  entry: LabelingTaskEntry,
): Promise<PreparedLabelingTask | null> {
  const provenancePath = `${entry.proxyPath}.provenance.json`;
  const [taskExists, proxyExists, provenanceExists] = await Promise.all([
    isFile(entry.taskPath),
    isFile(entry.proxyPath),
    isFile(provenancePath),
  ]);
  if (!taskExists) {
    if (entry.batch === "pilot") throw new Error(`pilot task is unavailable: ${entry.id}`);
    return null;
  }
  if (!proxyExists || !provenanceExists) {
    throw new Error(`prepared task ${entry.id} has incomplete proxy artifacts`);
  }
  return loadEntry(entry);
}

export async function getPreparedLabelingCatalog(): Promise<PreparedLabelingCatalog> {
  const entries = await readAllEntries();
  const loaded = await Promise.all(entries.map(loadAvailableEntry));
  return {
    tasks: loaded
      .filter((task): task is PreparedLabelingTask => task !== null)
      .sort((left, right) => {
        if (left.batch !== right.batch) return left.batch === "full" ? -1 : 1;
        return left.priority - right.priority;
      }),
    totals: {
      pilot: entries.filter((entry) => entry.batch === "pilot").length,
      full: entries.filter((entry) => entry.batch === "full").length,
    },
  };
}

export async function listPreparedLabelingTasks(): Promise<PreparedLabelingTask[]> {
  return (await getPreparedLabelingCatalog()).tasks;
}

export async function getPreparedLabelingTask(id: string): Promise<PreparedLabelingTask> {
  if (!/^[A-Za-z0-9_-]+$/.test(id)) throw new LabelingTaskNotFoundError();
  const entries = await readAllEntries();
  const entry = entries.find((candidate) => candidate.id === id);
  if (!entry) throw new LabelingTaskNotFoundError();
  try {
    const task = await loadAvailableEntry(entry);
    if (!task) throw new LabelingTaskNotFoundError();
    return task;
  } catch (error) {
    if (isMissingFile(error)) throw new LabelingTaskNotFoundError();
    throw error;
  }
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
    return { document, source: "draft", savedAt: metadata.mtime.toISOString() };
  } catch (error) {
    if (!isMissingFile(error)) throw error;
  }
  try {
    const metadata = await stat(task.completedPath);
    if (!metadata.isFile()) throw new Error("completed labels are not a file");
    const document = parseLabelDocument(
      JSON.parse(await readFile(task.completedPath, "utf8")) as unknown,
    );
    validateDraftContent(document, task, "complete");
    return { document, source: "completed", savedAt: metadata.mtime.toISOString() };
  } catch (error) {
    if (!isMissingFile(error)) throw error;
  }
  if (task.batch === "full") {
    const seed = await loadProductionLabelSeed(task);
    if (seed) {
      validateDraftContent(seed.document, task);
      return { document: seed.document, source: "production-model", savedAt: null };
    }
    try {
      const document = parseLabelDocument(
        JSON.parse(await readFile(task.prelabelPath, "utf8")) as unknown,
      );
      validateDraftContent(document, task);
      return { document, source: "prelabel", savedAt: null };
    } catch (error) {
      if (!isMissingFile(error)) throw error;
    }
  }
  return { document: task.document, source: "task", savedAt: null };
}

async function loadProductionLabelSeed(
  task: PreparedLabelingTask,
): Promise<ProductionLabelSeed | null> {
  if (task.batch !== "full") return null;
  try {
    const taskAnalysesRoot = task.workspaceRoot === labelingWorkspace
      ? getIntakeAnalysesRoot()
      : path.join(task.workspaceRoot, "analyses");
    const productionAnalysisPath = path.join(
      taskAnalysesRoot,
      `${PRODUCTION_MODEL_ID}--${task.id}`,
      "analysis.json",
    );
    const allLabelsV2Value = JSON.parse(
      await readFile(productionAnalysisPath, "utf8"),
    ) as unknown;
    const previousProductionPath = path.join(
      getAnalysesRoot("without-beach"),
      `${PREVIOUS_PRODUCTION_MODEL_ID}--${task.id}`,
      "analysis.json",
    );
    try {
      return buildProductionEnsembleLabelSeed(
        task.document,
        allLabelsV2Value,
        JSON.parse(await readFile(previousProductionPath, "utf8")) as unknown,
      );
    } catch (error) {
      if (!isMissingFile(error)) throw error;
      return buildProductionLabelSeed(
        task.document,
        allLabelsV2Value,
        PRODUCTION_MODEL_ID,
      );
    }
  } catch (error) {
    if (isMissingFile(error)) return null;
    throw error;
  }
}

function taskAnalysesRoot(task: PreparedLabelingTask): string {
  return task.workspaceRoot === labelingWorkspace
    ? getIntakeAnalysesRoot()
    : path.join(task.workspaceRoot, "analyses");
}

async function loadAnalysisReference(
  task: PreparedLabelingTask,
  modelId: string,
  modelLabel: string,
  description: string,
  analysesRoot: string,
): Promise<ExperimentModelReferenceLabels | null> {
  try {
    const seed = buildProductionLabelSeed(
      task.document,
      JSON.parse(
        await readFile(
          path.join(analysesRoot, `${modelId}--${task.id}`, "analysis.json"),
          "utf8",
        ),
      ) as unknown,
      modelId,
    );
    return {
      modelId,
      modelLabel,
      description,
      rallies: seed.document.rallies,
    };
  } catch (error) {
    if (isMissingFile(error)) return null;
    throw error;
  }
}

function inferenceRanges(
  value: unknown,
  task: PreparedLabelingTask,
): RallyLabel[] {
  const root = jsonObject(value);
  if (
    root?.recordingId !== task.id ||
    root.sourceFilename !== task.document.recording.videoFilename ||
    Math.abs((finiteNumber(root.duration) ?? -1) - task.document.recording.durationSeconds) > 0.1 ||
    !Array.isArray(root.ranges)
  ) {
    throw new Error("suppression inference does not match the labeling task");
  }
  return root.ranges.map((value, index) => {
    const range = jsonObject(value);
    const start = finiteNumber(range?.start);
    const end = finiteNumber(range?.end);
    if (
      start === null ||
      end === null ||
      start < 0 ||
      end <= start ||
      end > task.document.recording.durationSeconds
    ) {
      throw new Error(`suppression inference range ${index + 1} is invalid`);
    }
    return { start, end, tags: ["ai-reference", "suppression-v3"] };
  });
}

function subtractRanges(
  source: RallyLabel[],
  retained: RallyLabel[],
): RallyLabel[] {
  return source.flatMap((range) => {
    let segments = [{ start: range.start, end: range.end }];
    for (const keep of retained) {
      segments = segments.flatMap((segment) => {
        if (keep.end <= segment.start || keep.start >= segment.end) return [segment];
        return [
          ...(keep.start > segment.start
            ? [{ start: segment.start, end: Math.min(keep.start, segment.end) }]
            : []),
          ...(keep.end < segment.end
            ? [{ start: Math.max(keep.end, segment.start), end: segment.end }]
            : []),
        ];
      });
    }
    return segments.map((segment) => ({
      ...segment,
      tags: ["suppression-veto"],
    }));
  });
}

async function loadSuppressionReference(
  task: PreparedLabelingTask,
): Promise<ExperimentModelReferenceLabels | null> {
  const workspaceCandidates = [
    task.workspaceRoot,
    ...getIntakeWorkspaces().filter((root) => root !== task.workspaceRoot),
  ];
  for (const workspace of workspaceCandidates) {
    const inferenceRoot = path.join(
      workspace,
      "experiments",
      suppressionExperiment,
      "inference",
    );
    try {
      const [suppressedValue, productionValue] = await Promise.all([
        readFile(path.join(inferenceRoot, suppressionVariant, `${task.id}.json`), "utf8"),
        readFile(path.join(inferenceRoot, "current-production-ensemble", `${task.id}.json`), "utf8"),
      ]);
      const rallies = inferenceRanges(JSON.parse(suppressedValue) as unknown, task);
      const productionRanges = inferenceRanges(JSON.parse(productionValue) as unknown, task);
      return {
        modelId: `suppression-v3:${suppressionVariant}`,
        modelLabel: "Suppression-adjusted ensemble",
        description:
          "Corrected v3 suppression specialist applied to the held production ensemble with the zero-non-exempt-miss policy.",
        rallies,
        suppressedRanges: subtractRanges(productionRanges, rallies),
      };
    } catch (error) {
      if (isMissingFile(error)) continue;
      throw error;
    }
  }
  return null;
}

function scoreMarkerServeMarkers(value: unknown): ServeMarker[] | null {
  const root = jsonObject(value);
  if (!root || !Array.isArray(root.serveMarkers)) return null;
  const markers = root.serveMarkers.flatMap((value): ServeMarker[] => {
    const marker = jsonObject(value);
    const time = finiteNumber(marker?.time);
    const modelConfidence = finiteNumber(marker?.modelConfidence);
    const modelSide = marker?.modelSide;
    const side = modelSide === "near" || modelSide === "far"
      ? modelSide
      : marker?.side === "near" || marker?.side === "far"
        ? marker.side
        : null;
    if (time === null || side === null) return [];
    return [{
      time,
      side,
      origin: "model",
      modelSide: side,
      ...(modelConfidence === null ? {} : { modelConfidence }),
      ...(typeof marker?.modelId === "string" ? { modelId: marker.modelId } : {}),
      ...(typeof marker?.rallyId === "string" ? { rallyId: marker.rallyId } : {}),
    }];
  });
  return markers.sort((left, right) => left.time - right.time);
}

function scoreMarkerSideSwitches(value: unknown): SideSwitch[] {
  const root = jsonObject(value);
  if (!root || !Array.isArray(root.sideSwitches)) return [];
  return root.sideSwitches.flatMap((value): SideSwitch[] => {
    const marker = jsonObject(value);
    const time = finiteNumber(marker?.time);
    const modelConfidence = finiteNumber(marker?.modelConfidence);
    if (time === null) return [];
    return [{
      time,
      origin: "model",
      ...(modelConfidence === null ? {} : { modelConfidence }),
      ...(typeof marker?.modelId === "string" ? { modelId: marker.modelId } : {}),
      ...(typeof marker?.modelEventId === "string"
        ? { modelEventId: marker.modelEventId }
        : {}),
    }];
  }).sort((left, right) => left.time - right.time);
}

async function loadProductionServeMarkers(
  task: PreparedLabelingTask,
): Promise<{
  markers: ServeMarker[];
  humanMarkers?: ServeMarker[];
  label: string;
  sideSwitches?: SideSwitch[];
  sideSwitchLabel?: string;
} | null> {
  const workspaceCandidates = [
    task.workspaceRoot,
    ...getIntakeWorkspaces().filter((root) => root !== task.workspaceRoot),
  ];
  for (const workspace of workspaceCandidates) {
    try {
      const value = JSON.parse(
        await readFile(
          path.join(workspace, "predictions", "score-markers", `${task.id}.json`),
          "utf8",
        ),
      ) as unknown;
      const root = jsonObject(value);
      if (root?.recordingId !== task.id) continue;
      const markers = scoreMarkerServeMarkers(value);
      if (markers) {
        return {
          markers,
          label: "Serving-side fixed-flight v3",
          sideSwitches: scoreMarkerSideSwitches(value),
          sideSwitchLabel: "Side-switch hard-negative-mining v1",
        };
      }
    } catch (error) {
      if (!isMissingFile(error)) throw error;
    }
  }

  try {
    const report = jsonObject(
      JSON.parse(await readFile(servingSideInferencePath, "utf8")) as unknown,
    );
    if (!Array.isArray(report?.predictions)) return null;
    const markers = report.predictions.flatMap((value): ServeMarker[] => {
      const prediction = jsonObject(value);
      const evidence = jsonObject(prediction?.serveEvidence);
      const time = finiteNumber(evidence?.serveAnchor);
      const side = prediction?.finalPrediction;
      const nearProbability = finiteNumber(prediction?.nearProbability);
      if (
        prediction?.recordingId !== task.id ||
        prediction?.servePrediction !== "serve" ||
        time === null ||
        (side !== "near" && side !== "far")
      ) {
        return [];
      }
      return [{
        time,
        side,
        origin: "model",
        modelSide: side,
        ...(nearProbability === null
          ? {}
          : { modelConfidence: side === "near" ? nearProbability : 1 - nearProbability }),
        modelId: "serving-side-fixed-flight-v3",
        ...(typeof prediction.rallyId === "string" ? { rallyId: prediction.rallyId } : {}),
      }];
    }).sort((left, right) => left.time - right.time);
    const humanMarkers = report.predictions.flatMap((value): ServeMarker[] => {
      const prediction = jsonObject(value);
      const evidence = jsonObject(prediction?.serveEvidence);
      const time = finiteNumber(evidence?.serveAnchor);
      const side = prediction?.decision;
      if (
        prediction?.recordingId !== task.id ||
        time === null ||
        (side !== "near" && side !== "far")
      ) {
        return [];
      }
      return [{
        time,
        side,
        origin: "manual",
        notes: "Imported from the frozen serving-side human decision set.",
        ...(typeof prediction.rallyId === "string" ? { rallyId: prediction.rallyId } : {}),
      }];
    }).sort((left, right) => left.time - right.time);
    return markers.length > 0
      ? { markers, humanMarkers, label: "Serving-side fixed-flight v3" }
      : null;
  } catch (error) {
    if (isMissingFile(error)) return null;
    throw error;
  }
}

export async function getProductionReferenceLabels(
  task: PreparedLabelingTask,
): Promise<ProductionReferenceLabels | null> {
  const [seed, serveReference] = await Promise.all([
    loadProductionLabelSeed(task),
    loadProductionServeMarkers(task),
  ]);
  return seed || serveReference
    ? {
        modelId: seed?.modelId ?? PRODUCTION_ENSEMBLE_MODEL_ID,
        modelLabel: seed?.modelLabel ?? "Production ensemble · frozen score-marker run",
        description: seed
          ? seed.modelId === PRODUCTION_ENSEMBLE_MODEL_ID
            ? PRODUCTION_ENSEMBLE_MODEL_DESCRIPTION
            : PRODUCTION_MODEL_DESCRIPTION
          : "Frozen production rally ensemble and score-marker specialists used to initialize this labeling task.",
        rallies: seed?.document.rallies ?? task.document.rallies,
        ...(serveReference
          ? {
              serveMarkers: serveReference.markers,
              ...(serveReference.humanMarkers
                ? { humanServeMarkers: serveReference.humanMarkers }
                : {}),
              serveModelLabel: serveReference.label,
              ...(serveReference.sideSwitches?.length
                ? {
                    sideSwitches: serveReference.sideSwitches,
                    sideSwitchModelLabel: serveReference.sideSwitchLabel,
                  }
                : {}),
            }
          : {}),
      }
    : null;
}

export async function getModelBreakdownReferenceLabels(
  task: PreparedLabelingTask,
): Promise<ExperimentModelReferenceLabels[]> {
  if (task.batch !== "full") return [];
  const references = await Promise.all([
    loadAnalysisReference(
      task,
      PRODUCTION_MODEL_ID,
      PRODUCTION_MODEL_LABEL,
      PRODUCTION_MODEL_DESCRIPTION,
      taskAnalysesRoot(task),
    ),
    loadAnalysisReference(
      task,
      PREVIOUS_PRODUCTION_MODEL_ID,
      PREVIOUS_PRODUCTION_MODEL_LABEL,
      "The production model immediately preceding all-labels v2.",
      getAnalysesRoot("without-beach"),
    ),
    loadSuppressionReference(task),
  ]);
  return references.filter(
    (reference): reference is ExperimentModelReferenceLabels => reference !== null,
  );
}

export async function getExperimentModelReferenceLabels(
  task: PreparedLabelingTask,
): Promise<ExperimentModelReferenceLabels[]> {
  if (task.batch !== "full") return [];
  const references = await Promise.all(
    ENVIRONMENT_EXPERIMENT_MODELS.map(async (model): Promise<ExperimentModelReferenceLabels | null> => {
      try {
        const analysisPath = path.join(
          taskAnalysesRoot(task),
          `${model.id}--${task.id}`,
          "analysis.json",
        );
        const seed = buildProductionLabelSeed(
          task.document,
          JSON.parse(await readFile(analysisPath, "utf8")) as unknown,
          model.id,
        );
        return {
          modelId: seed.modelId,
          modelLabel: seed.modelLabel || model.label,
          description: model.description,
          rallies: seed.document.rallies,
        };
      } catch (error) {
        if (isMissingFile(error)) return null;
        throw error;
      }
    }),
  );
  return references.filter(
    (reference): reference is ExperimentModelReferenceLabels => reference !== null,
  );
}

export async function getSolReferenceLabels(
  task: PreparedLabelingTask,
): Promise<SolReferenceLabels | null> {
  if (task.batch !== "full") return null;
  try {
    const document = parseLabelDocument(
      JSON.parse(await readFile(task.prelabelPath, "utf8")) as unknown,
    );
    validateDraftContent(document, task);
    return { rallies: document.rallies };
  } catch (error) {
    if (isMissingFile(error)) return null;
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
  if (!isWithin(task.workspaceRoot, task.draftPath)) {
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
  return { document, source: "draft", savedAt: metadata.mtime.toISOString() };
}

export async function saveCompletedLabelingDocument(
  task: PreparedLabelingTask,
  value: unknown,
): Promise<SavedLabelingDocument> {
  let document: LabelDocument;
  try {
    document = parseLabelDocument(value);
  } catch (error) {
    throw new LabelingDraftValidationError(
      error instanceof Error ? error.message : "completed label document is invalid",
    );
  }
  validateDraftContent(document, task, "complete");
  const completedDirectory = path.dirname(task.completedPath);
  if (!isWithin(task.workspaceRoot, task.completedPath)) {
    throw new Error("completed-label destination is outside the labeling workspace");
  }
  await mkdir(completedDirectory, { recursive: true });
  const temporaryPath = path.join(
    completedDirectory,
    `.${task.id}.${randomUUID()}.labels.json.tmp`,
  );
  try {
    await writeFile(temporaryPath, `${JSON.stringify(document, null, 2)}\n`, {
      encoding: "utf8",
      flag: "wx",
      mode: 0o600,
    });
    await rename(temporaryPath, task.completedPath);
  } finally {
    await rm(temporaryPath, { force: true });
  }
  await rm(task.draftPath, { force: true });
  const metadata = await stat(task.completedPath);
  return { document, source: "completed", savedAt: metadata.mtime.toISOString() };
}
