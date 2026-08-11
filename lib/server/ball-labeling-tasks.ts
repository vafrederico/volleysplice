import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { isDeepStrictEqual } from "node:util";
import path from "node:path";

import {
  BallReviewValidationError,
  parseBallAnnotationTask,
  parseBallReviewDocument,
  reviewedFrameCount,
  toBallReviewDocument,
  type BallAnnotationTask,
  type BallComparisonLayers,
  type BallFrame,
  type BallFrameAnnotation,
  type BallProposalExposure,
  type BallProposalSource,
  type BallReviewDocument,
  type NormalizedBox,
} from "../ball-annotations.ts";

const defaultBallPilotRoot = "/mnt/freenas/volleycut/ball-presence-v1/round-01";
const taskSuffix = ".ball-presence.json";

type BallPilotIndexRow = {
  recordingId: string;
  split: "train" | "validation";
  task: string;
  taskId: string;
  initialTaskSha256: string;
  frameCount: number;
};

type BallPilotIndex = {
  schemaVersion: 1;
  artifactType: "volleycut-ball-presence-pilot-index";
  developmentOnly: true;
  recordingCount: number;
  frameCount: number;
  tasks: BallPilotIndexRow[];
};

type ProposalExposureAuditEntry = {
  proposalExposure: BallProposalExposure;
  assistedSources: BallProposalSource[];
  postDecisionSources: BallProposalSource[];
  firstAssistedAt: string | null;
  firstPostDecisionRevealAt: string | null;
  postDecisionHumanAnnotationSha256: string | null;
};

type ProposalExposureAudit = {
  schemaVersion: 1;
  kind: "volleycut-ball-proposal-exposure-audit";
  taskId: string;
  recordingId: string;
  frames: Record<string, ProposalExposureAuditEntry>;
};

export type BallLabelingTaskSummary = {
  id: string;
  taskId: string;
  split: "train" | "validation";
  environment: string;
  frameCount: number;
  reviewedFrameCount: number;
  windowCount: number;
  reviewStatus: BallReviewDocument["annotations"]["review"]["status"];
  assistedFrameCount: number;
  savedAt: string | null;
};

export type PreparedBallLabelingTask = {
  id: string;
  taskPath: string;
  reviewPath: string;
  baseSha256: string;
  base: BallAnnotationTask;
};

export type BallFrameImage = {
  bytes: Uint8Array;
  sha256: string;
  filename: string;
};

export class BallLabelingTaskNotFoundError extends Error {}
export class BallLabelingWorkspaceError extends Error {}
export class BallLabelingDraftValidationError extends Error {}
export class BallLabelingImageValidationError extends Error {}
export class BallComparisonAccessError extends Error {}

const taskMutationTails = new Map<string, Promise<void>>();
const preparedTaskCatalogs = new Map<string, Promise<PreparedBallLabelingTask[]>>();

async function withTaskMutationLock<T>(
  key: string,
  operation: () => Promise<T>,
): Promise<T> {
  const previous = taskMutationTails.get(key) ?? Promise.resolve();
  let release = () => {};
  const current = new Promise<void>((resolve) => {
    release = resolve;
  });
  const tail = previous.then(
    () => current,
    () => current,
  );
  taskMutationTails.set(key, tail);
  await previous.catch(() => undefined);
  try {
    return await operation();
  } finally {
    release();
    if (taskMutationTails.get(key) === tail) taskMutationTails.delete(key);
  }
}

export function getBallPilotRoot(): string {
  return path.resolve(
    /* turbopackIgnore: true */
    process.env.VOLLEYCUT_BALL_PILOT_ROOT ?? defaultBallPilotRoot,
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasExactKeys(value: Record<string, unknown>, expected: string[]): boolean {
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  return actual.length === wanted.length && actual.every((key, index) => key === wanted[index]);
}

function isWithin(parent: string, candidate: string): boolean {
  const relative = path.relative(parent, candidate);
  return (
    relative === "" ||
    (!relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative))
  );
}

function validSha256(value: unknown): value is string {
  return typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
}

function validNonzeroSha256(value: unknown): value is string {
  return validSha256(value) && !/^0{64}$/.test(value);
}

function sha256(bytes: Uint8Array | string): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value === "boolean" || typeof value === "number" || typeof value === "string") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (isRecord(value)) {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`)
      .join(",")}}`;
  }
  throw new BallLabelingWorkspaceError("task contains a non-JSON value");
}

type RawJsonNode =
  | { kind: "array"; values: RawJsonNode[] }
  | { kind: "object"; entries: Array<{ key: string; value: RawJsonNode }> }
  | { kind: "number"; raw: string }
  | { kind: "scalar"; value: string | boolean | null };

function parseRawJson(source: string): RawJsonNode {
  let offset = 0;
  const skipWhitespace = () => {
    while (offset < source.length && /[\t\n\r ]/.test(source[offset])) offset += 1;
  };
  const parseString = (): string => {
    const start = offset;
    offset += 1;
    let escaped = false;
    while (offset < source.length) {
      const character = source[offset];
      offset += 1;
      if (escaped) {
        escaped = false;
      } else if (character === "\\") {
        escaped = true;
      } else if (character === '"') {
        return JSON.parse(source.slice(start, offset)) as string;
      } else if (character.charCodeAt(0) < 0x20) {
        break;
      }
    }
    throw new BallLabelingWorkspaceError("task JSON contains an invalid string");
  };
  const parseNode = (): RawJsonNode => {
    skipWhitespace();
    const character = source[offset];
    if (character === '"') return { kind: "scalar", value: parseString() };
    if (character === "[") {
      offset += 1;
      const values: RawJsonNode[] = [];
      skipWhitespace();
      if (source[offset] === "]") {
        offset += 1;
        return { kind: "array", values };
      }
      while (true) {
        values.push(parseNode());
        skipWhitespace();
        if (source[offset] === "]") {
          offset += 1;
          return { kind: "array", values };
        }
        if (source[offset] !== ",") {
          throw new BallLabelingWorkspaceError("task JSON array is invalid");
        }
        offset += 1;
      }
    }
    if (character === "{") {
      offset += 1;
      const entries: Array<{ key: string; value: RawJsonNode }> = [];
      const keys = new Set<string>();
      skipWhitespace();
      if (source[offset] === "}") {
        offset += 1;
        return { kind: "object", entries };
      }
      while (true) {
        skipWhitespace();
        if (source[offset] !== '"') {
          throw new BallLabelingWorkspaceError("task JSON object key is invalid");
        }
        const key = parseString();
        if (keys.has(key)) {
          throw new BallLabelingWorkspaceError("task JSON contains duplicate object keys");
        }
        keys.add(key);
        skipWhitespace();
        if (source[offset] !== ":") {
          throw new BallLabelingWorkspaceError("task JSON object is invalid");
        }
        offset += 1;
        entries.push({ key, value: parseNode() });
        skipWhitespace();
        if (source[offset] === "}") {
          offset += 1;
          return { kind: "object", entries };
        }
        if (source[offset] !== ",") {
          throw new BallLabelingWorkspaceError("task JSON object is invalid");
        }
        offset += 1;
      }
    }
    for (const [literal, value] of [
      ["true", true],
      ["false", false],
      ["null", null],
    ] as const) {
      if (source.startsWith(literal, offset)) {
        offset += literal.length;
        return { kind: "scalar", value };
      }
    }
    const number = source.slice(offset).match(/^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?/);
    if (!number) throw new BallLabelingWorkspaceError("task JSON value is invalid");
    offset += number[0].length;
    return { kind: "number", raw: number[0] };
  };
  const root = parseNode();
  skipWhitespace();
  if (offset !== source.length) {
    throw new BallLabelingWorkspaceError("task JSON has trailing data");
  }
  return root;
}

function compareUnicodeCodePoints(left: string, right: string): number {
  const leftPoints = Array.from(left, (character) => character.codePointAt(0) ?? 0);
  const rightPoints = Array.from(right, (character) => character.codePointAt(0) ?? 0);
  const length = Math.min(leftPoints.length, rightPoints.length);
  for (let index = 0; index < length; index += 1) {
    if (leftPoints[index] !== rightPoints[index]) return leftPoints[index] - rightPoints[index];
  }
  return leftPoints.length - rightPoints.length;
}

function canonicalRawJson(node: RawJsonNode): string {
  if (node.kind === "number") return node.raw;
  if (node.kind === "scalar") return JSON.stringify(node.value);
  if (node.kind === "array") return `[${node.values.map(canonicalRawJson).join(",")}]`;
  return `{${[...node.entries]
    .sort((left, right) => compareUnicodeCodePoints(left.key, right.key))
    .map(({ key, value }) => `${JSON.stringify(key)}:${canonicalRawJson(value)}`)
    .join(",")}}`;
}

function immutableDigestFromSource(source: string): string {
  const root = parseRawJson(source);
  if (root.kind !== "object") {
    throw new BallLabelingWorkspaceError("task JSON root is invalid");
  }
  const immutable = root.entries.find((entry) => entry.key === "immutable")?.value;
  if (!immutable || immutable.kind !== "object") {
    throw new BallLabelingWorkspaceError("task immutable provenance is missing");
  }
  return sha256(
    canonicalRawJson({
      kind: "object",
      entries: immutable.entries.filter(
        (entry) => entry.key !== "taskId" && entry.key !== "digestSha256",
      ),
    }),
  );
}

function verifyImmutableDigest(task: BallAnnotationTask, source: string): void {
  const digest = immutableDigestFromSource(source);
  if (
    task.immutable.digestSha256 !== digest ||
    task.immutable.taskId !== `ball-presence-${digest.slice(0, 24)}`
  ) {
    throw new BallLabelingWorkspaceError("task immutable provenance digest is invalid");
  }
}

function safeRecordingId(value: unknown): value is string {
  return typeof value === "string" && /^[A-Za-z0-9_-]+$/.test(value);
}

async function isFile(filePath: string): Promise<boolean> {
  try {
    return (await stat(filePath)).isFile();
  } catch (error) {
    if (isRecord(error) && error.code === "ENOENT") return false;
    throw error;
  }
}

async function readIndex(): Promise<{ root: string; rows: BallPilotIndexRow[] }> {
  const root = getBallPilotRoot();
  const indexPath = path.join(root, "index.json");
  let value: unknown;
  try {
    value = JSON.parse(await readFile(indexPath, "utf8")) as unknown;
  } catch (error) {
    throw new BallLabelingWorkspaceError(
      `cannot read ball pilot index: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  if (
    !isRecord(value) ||
    value.schemaVersion !== 1 ||
    value.artifactType !== "volleycut-ball-presence-pilot-index" ||
    value.developmentOnly !== true ||
    !Number.isInteger(value.recordingCount) ||
    !Number.isInteger(value.frameCount) ||
    !Array.isArray(value.tasks)
  ) {
    throw new BallLabelingWorkspaceError("ball pilot index is invalid");
  }
  const index = value as unknown as BallPilotIndex;
  if (index.recordingCount !== index.tasks.length) {
    throw new BallLabelingWorkspaceError("ball pilot recording count is inconsistent");
  }
  const ids = new Set<string>();
  const rows = index.tasks.map((candidate, rowIndex) => {
    if (
      !isRecord(candidate) ||
      !safeRecordingId(candidate.recordingId) ||
      ids.has(candidate.recordingId) ||
      !["train", "validation"].includes(String(candidate.split)) ||
      typeof candidate.task !== "string" ||
      typeof candidate.taskId !== "string" ||
      !validSha256(candidate.initialTaskSha256) ||
      !Number.isInteger(candidate.frameCount) ||
      Number(candidate.frameCount) < 1
    ) {
      throw new BallLabelingWorkspaceError(`ball pilot index task ${rowIndex + 1} is invalid`);
    }
    ids.add(candidate.recordingId);
    return candidate as unknown as BallPilotIndexRow;
  });
  if (rows.reduce((total, row) => total + row.frameCount, 0) !== index.frameCount) {
    throw new BallLabelingWorkspaceError("ball pilot frame count is inconsistent");
  }
  return { root, rows };
}

async function loadBaseTask(
  root: string,
  row: BallPilotIndexRow,
): Promise<PreparedBallLabelingTask> {
  const tasksRoot = path.join(root, "tasks");
  const taskPath = path.resolve(root, row.task);
  const expectedFilename = `${row.recordingId}${taskSuffix}`;
  if (
    !isWithin(tasksRoot, taskPath) ||
    path.dirname(taskPath) !== tasksRoot ||
    path.basename(taskPath) !== expectedFilename
  ) {
    throw new BallLabelingWorkspaceError(`${row.recordingId} task path is outside tasks/`);
  }
  let encoded: Uint8Array;
  let base: BallAnnotationTask;
  try {
    encoded = await readFile(taskPath);
    if (sha256(encoded) !== row.initialTaskSha256) {
      throw new BallLabelingWorkspaceError(`${row.recordingId} pristine task SHA-256 changed`);
    }
    const source = Buffer.from(encoded).toString("utf8");
    base = parseBallAnnotationTask(JSON.parse(source) as unknown);
    verifyImmutableDigest(base, source);
  } catch (error) {
    if (error instanceof BallLabelingWorkspaceError) throw error;
    throw new BallLabelingWorkspaceError(
      `cannot load ${row.recordingId} pristine task: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  if (
    base.immutable.recording.id !== row.recordingId ||
    base.immutable.recording.split !== row.split ||
    base.immutable.taskId !== row.taskId ||
    base.immutable.frames.length !== row.frameCount
  ) {
    throw new BallLabelingWorkspaceError(`${row.recordingId} task does not match the pilot index`);
  }
  return {
    id: row.recordingId,
    taskPath,
    reviewPath: path.join(root, "reviews", expectedFilename),
    baseSha256: row.initialTaskSha256,
    base,
  };
}

async function loadReview(
  prepared: PreparedBallLabelingTask,
): Promise<{ document: BallReviewDocument; savedAt: string | null }> {
  if (!(await isFile(prepared.reviewPath))) {
    return { document: toBallReviewDocument(prepared.base), savedAt: null };
  }
  let reviewed: BallAnnotationTask;
  try {
    reviewed = parseBallAnnotationTask(
      JSON.parse(await readFile(prepared.reviewPath, "utf8")) as unknown,
    );
  } catch (error) {
    throw new BallLabelingWorkspaceError(
      `saved review ${prepared.id} is invalid: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  if (!isDeepStrictEqual(reviewed.immutable, prepared.base.immutable)) {
    throw new BallLabelingWorkspaceError(`${prepared.id} saved review changed immutable provenance`);
  }
  const metadata = await stat(prepared.reviewPath);
  return {
    document: toBallReviewDocument(reviewed),
    savedAt: metadata.mtime.toISOString(),
  };
}

async function writeReviewArtifact(
  prepared: PreparedBallLabelingTask,
  submission: BallReviewDocument,
): Promise<string> {
  const fullDocument: BallAnnotationTask = {
    ...submission,
    suggestions: { status: "empty", model: null, frames: {} },
  };
  try {
    parseBallAnnotationTask(fullDocument);
  } catch (error) {
    throw new BallLabelingDraftValidationError(
      error instanceof Error ? error.message : "review document is invalid",
    );
  }
  const root = getBallPilotRoot();
  const reviewsRoot = path.join(root, "reviews");
  if (!isWithin(reviewsRoot, prepared.reviewPath) || path.dirname(prepared.reviewPath) !== reviewsRoot) {
    throw new BallLabelingWorkspaceError("review destination is outside reviews/");
  }
  await mkdir(reviewsRoot, { recursive: true });
  const temporaryPath = path.join(reviewsRoot, `.${prepared.id}.${randomUUID()}.json.tmp`);
  try {
    await writeFile(temporaryPath, `${JSON.stringify(fullDocument, null, 2)}\n`, {
      encoding: "utf8",
      flag: "wx",
      mode: 0o600,
    });
    await rename(temporaryPath, prepared.reviewPath);
  } finally {
    await rm(temporaryPath, { force: true });
  }
  return (await stat(prepared.reviewPath)).mtime.toISOString();
}

function exposureAuditPath(prepared: PreparedBallLabelingTask): string {
  return path.join(
    path.dirname(prepared.reviewPath),
    `${prepared.id}.proposal-exposure.json`,
  );
}

function blankExposureAudit(prepared: PreparedBallLabelingTask): ProposalExposureAudit {
  return {
    schemaVersion: 1,
    kind: "volleycut-ball-proposal-exposure-audit",
    taskId: prepared.base.immutable.taskId,
    recordingId: prepared.id,
    frames: {},
  };
}

function validProposalSources(value: unknown): value is BallProposalSource[] {
  return (
    Array.isArray(value) &&
    new Set(value).size === value.length &&
    value.every((source) => source === "sol" || source === "detector")
  );
}

function exposureFromSources(sources: BallProposalSource[]): BallProposalExposure {
  if (sources.includes("sol") && sources.includes("detector")) return "both";
  if (sources.includes("sol")) return "sol";
  if (sources.includes("detector")) return "detector";
  return "blind";
}

function mergeSources(
  current: BallProposalSource[],
  additional: BallProposalSource[],
): BallProposalSource[] {
  return (["sol", "detector"] as const).filter(
    (source) => current.includes(source) || additional.includes(source),
  );
}

function humanAnnotationRevisionSha256(
  prepared: PreparedBallLabelingTask,
  frameId: string,
  annotation: BallFrameAnnotation,
): string {
  const semanticAnnotation = {
    status: annotation.status,
    primaryBallState: annotation.primaryBallState,
    objects: annotation.objects,
    notes: annotation.notes,
  };
  return sha256(
    canonicalJson({
      taskId: prepared.base.immutable.taskId,
      frameId,
      annotation: semanticAnnotation,
    }),
  );
}

async function readExposureAudit(
  prepared: PreparedBallLabelingTask,
): Promise<ProposalExposureAudit> {
  const auditPath = exposureAuditPath(prepared);
  if (!(await isFile(auditPath))) return blankExposureAudit(prepared);
  let value: unknown;
  try {
    value = JSON.parse(await readFile(auditPath, "utf8")) as unknown;
  } catch (error) {
    throw new BallLabelingWorkspaceError(
      `cannot read ${prepared.id} proposal-exposure audit: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  if (
    !isRecord(value) ||
    value.schemaVersion !== 1 ||
    value.kind !== "volleycut-ball-proposal-exposure-audit" ||
    value.taskId !== prepared.base.immutable.taskId ||
    value.recordingId !== prepared.id ||
    !isRecord(value.frames)
  ) {
    throw new BallLabelingWorkspaceError(`${prepared.id} proposal-exposure audit is invalid`);
  }
  const frameIds = new Set(prepared.base.immutable.frames.map((frame) => frame.id));
  const frames: Record<string, ProposalExposureAuditEntry> = {};
  for (const [frameId, entryValue] of Object.entries(value.frames)) {
    if (!frameIds.has(frameId) || !isRecord(entryValue)) {
      throw new BallLabelingWorkspaceError(`${prepared.id} proposal-exposure frame is invalid`);
    }
    if (
      !["blind", "sol", "detector", "both"].includes(String(entryValue.proposalExposure)) ||
      !validProposalSources(entryValue.assistedSources) ||
      !validProposalSources(entryValue.postDecisionSources) ||
      (entryValue.firstAssistedAt !== null && typeof entryValue.firstAssistedAt !== "string") ||
      (entryValue.firstPostDecisionRevealAt !== null &&
        typeof entryValue.firstPostDecisionRevealAt !== "string") ||
      (entryValue.postDecisionHumanAnnotationSha256 !== undefined &&
        entryValue.postDecisionHumanAnnotationSha256 !== null &&
        !validSha256(entryValue.postDecisionHumanAnnotationSha256)) ||
      exposureFromSources(entryValue.assistedSources) !== entryValue.proposalExposure
    ) {
      throw new BallLabelingWorkspaceError(`${prepared.id}/${frameId} exposure entry is invalid`);
    }
    frames[frameId] = {
      proposalExposure: entryValue.proposalExposure as BallProposalExposure,
      assistedSources: entryValue.assistedSources,
      postDecisionSources: entryValue.postDecisionSources,
      firstAssistedAt: entryValue.firstAssistedAt as string | null,
      firstPostDecisionRevealAt: entryValue.firstPostDecisionRevealAt as string | null,
      postDecisionHumanAnnotationSha256:
        (entryValue.postDecisionHumanAnnotationSha256 as string | null | undefined) ?? null,
    };
  }
  return {
    schemaVersion: 1,
    kind: "volleycut-ball-proposal-exposure-audit",
    taskId: prepared.base.immutable.taskId,
    recordingId: prepared.id,
    frames,
  };
}

async function writeExposureAudit(
  prepared: PreparedBallLabelingTask,
  audit: ProposalExposureAudit,
): Promise<void> {
  const destination = exposureAuditPath(prepared);
  const reviewsRoot = path.dirname(prepared.reviewPath);
  if (!isWithin(reviewsRoot, destination) || path.dirname(destination) !== reviewsRoot) {
    throw new BallLabelingWorkspaceError("proposal-exposure destination is outside reviews/");
  }
  await mkdir(reviewsRoot, { recursive: true });
  const temporaryPath = path.join(reviewsRoot, `.${prepared.id}.${randomUUID()}.exposure.tmp`);
  try {
    await writeFile(temporaryPath, `${JSON.stringify(audit, null, 2)}\n`, {
      encoding: "utf8",
      flag: "wx",
      mode: 0o600,
    });
    await rename(temporaryPath, destination);
  } finally {
    await rm(temporaryPath, { force: true });
  }
}

function validNormalizedBox(value: unknown): value is NormalizedBox {
  if (!isRecord(value)) return false;
  const coordinates = [value.x, value.y, value.width, value.height];
  if (!coordinates.every((item) => typeof item === "number" && Number.isFinite(item))) {
    return false;
  }
  const { x, y, width, height } = value as NormalizedBox;
  return x >= 0 && y >= 0 && width > 0 && height > 0 && x + width <= 1 && y + height <= 1;
}

async function loadSolLayer(
  prepared: PreparedBallLabelingTask,
  frameId: string,
): Promise<BallComparisonLayers["layers"]["sol"]> {
  const configuredRoot = process.env.VOLLEYCUT_BALL_SOL_LABELS_ROOT;
  const solRoot = path.resolve(
    /* turbopackIgnore: true */
    configuredRoot ?? path.join(getBallPilotRoot(), "sol-labels"),
  );
  const artifactPath = path.join(solRoot, `${prepared.id}${taskSuffix}`);
  if (!isWithin(solRoot, artifactPath) || !(await isFile(artifactPath))) return null;
  const encoded = await readFile(artifactPath);
  const source = Buffer.from(encoded).toString("utf8");
  let value: unknown;
  let solTask: BallAnnotationTask;
  try {
    value = JSON.parse(source) as unknown;
  } catch (error) {
    throw new BallLabelingWorkspaceError(
      `Sol layer ${prepared.id} is invalid: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "schemaVersion",
      "taskType",
      "immutable",
      "suggestions",
      "annotations",
      "solReviewProvenance",
    ]) ||
    !isRecord(value.solReviewProvenance)
  ) {
    throw new BallLabelingWorkspaceError(
      `Sol layer ${prepared.id} must contain exact detector-blind provenance`,
    );
  }
  const provenance = value.solReviewProvenance;
  if (
    !hasExactKeys(provenance, [
      "schemaVersion",
      "kind",
      "preparedAt",
      "sourceTask",
      "preparationReceipt",
      "reviewer",
      "detectorSuggestionsAbsent",
      "immutableDigestSha256",
      "implementationSha256",
    ]) ||
    provenance.schemaVersion !== 1 ||
    provenance.kind !== "volleycut-detector-blind-sol-ball-review" ||
    provenance.detectorSuggestionsAbsent !== true ||
    provenance.immutableDigestSha256 !== prepared.base.immutable.digestSha256 ||
    !validNonzeroSha256(provenance.implementationSha256) ||
    typeof provenance.preparedAt !== "string" ||
    Number.isNaN(Date.parse(provenance.preparedAt)) ||
    !/(?:Z|[+-][0-9]{2}:[0-9]{2})$/.test(provenance.preparedAt) ||
    !isRecord(provenance.reviewer) ||
    !isRecord(provenance.sourceTask) ||
    !isRecord(provenance.preparationReceipt)
  ) {
    throw new BallLabelingWorkspaceError(`Sol layer ${prepared.id} provenance is invalid`);
  }
  const reviewer = provenance.reviewer;
  const sourceTask = provenance.sourceTask;
  const receiptPointer = provenance.preparationReceipt;
  if (
    !hasExactKeys(reviewer, ["kind", "agentId", "modelId", "runId"]) ||
    reviewer.kind !== "detector-blind-sol-agent" ||
    ["agentId", "modelId", "runId"].some(
      (field) => typeof reviewer[field] !== "string" || !(reviewer[field] as string).trim(),
    ) ||
    !hasExactKeys(sourceTask, ["pathHint", "sha256", "requiredState"]) ||
    sourceTask.requiredState !== "unreviewed annotations with empty suggestions" ||
    sourceTask.sha256 !== prepared.baseSha256 ||
    typeof sourceTask.pathHint !== "string" ||
    path.resolve(sourceTask.pathHint) !== prepared.taskPath
  ) {
    throw new BallLabelingWorkspaceError(`Sol layer ${prepared.id} provenance is invalid`);
  }
  if (
    path.dirname(solRoot) !== path.dirname(path.dirname(prepared.taskPath)) ||
    path.dirname(prepared.taskPath) === solRoot
  ) {
    throw new BallLabelingWorkspaceError(
      `Sol layer ${prepared.id} must remain in a sibling-depth directory`,
    );
  }

  const receiptPath = `${artifactPath}.preparation-receipt.json`;
  if (
    !hasExactKeys(receiptPointer, ["pathHint", "sha256"]) ||
    typeof receiptPointer.pathHint !== "string" ||
    path.resolve(receiptPointer.pathHint) !== receiptPath ||
    !validNonzeroSha256(receiptPointer.sha256)
  ) {
    throw new BallLabelingWorkspaceError(
      `Sol layer ${prepared.id} preparation receipt pointer is invalid`,
    );
  }
  let receiptEncoded: Uint8Array;
  let receipt: unknown;
  try {
    receiptEncoded = await readFile(receiptPath);
    if (sha256(receiptEncoded) !== receiptPointer.sha256) {
      throw new BallLabelingWorkspaceError(
        `Sol layer ${prepared.id} preparation receipt SHA-256 differs`,
      );
    }
    receipt = JSON.parse(Buffer.from(receiptEncoded).toString("utf8")) as unknown;
  } catch (error) {
    if (error instanceof BallLabelingWorkspaceError) throw error;
    throw new BallLabelingWorkspaceError(
      `Sol layer ${prepared.id} preparation receipt is unreadable: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  if (
    !isRecord(receipt) ||
    !hasExactKeys(receipt, [
      "schemaVersion",
      "kind",
      "preparedAt",
      "outputTask",
      "sourceTask",
      "pilotIndex",
      "reviewer",
      "detectorSuggestionsAbsent",
      "immutableDigestSha256",
      "implementationSha256",
    ]) ||
    receipt.schemaVersion !== 1 ||
    receipt.kind !== "volleycut-sol-ball-review-preparation-receipt" ||
    receipt.preparedAt !== provenance.preparedAt ||
    receipt.detectorSuggestionsAbsent !== true ||
    receipt.immutableDigestSha256 !== provenance.immutableDigestSha256 ||
    receipt.implementationSha256 !== provenance.implementationSha256 ||
    !isDeepStrictEqual(receipt.sourceTask, sourceTask) ||
    !isDeepStrictEqual(receipt.reviewer, reviewer) ||
    !isRecord(receipt.outputTask) ||
    !hasExactKeys(receipt.outputTask, ["pathHint", "requiredState"]) ||
    typeof receipt.outputTask.pathHint !== "string" ||
    path.resolve(receipt.outputTask.pathHint) !== artifactPath ||
    receipt.outputTask.requiredState !== "unreviewed annotations with empty suggestions" ||
    !isRecord(receipt.pilotIndex) ||
    !hasExactKeys(receipt.pilotIndex, ["pathHint", "sha256"]) ||
    !validNonzeroSha256(receipt.pilotIndex.sha256)
  ) {
    throw new BallLabelingWorkspaceError(
      `Sol layer ${prepared.id} preparation receipt provenance is invalid`,
    );
  }
  const indexPath = path.join(getBallPilotRoot(), "index.json");
  if (
    typeof receipt.pilotIndex.pathHint !== "string" ||
    path.resolve(receipt.pilotIndex.pathHint) !== indexPath
  ) {
    throw new BallLabelingWorkspaceError(
      `Sol layer ${prepared.id} preparation receipt pilot index path differs`,
    );
  }
  let receiptIndex: unknown;
  try {
    const indexEncoded = await readFile(indexPath);
    if (sha256(indexEncoded) !== receipt.pilotIndex.sha256) {
      throw new BallLabelingWorkspaceError(
        `Sol layer ${prepared.id} preparation receipt pilot index SHA-256 differs`,
      );
    }
    receiptIndex = JSON.parse(Buffer.from(indexEncoded).toString("utf8")) as unknown;
  } catch (error) {
    if (error instanceof BallLabelingWorkspaceError) throw error;
    throw new BallLabelingWorkspaceError(
      `Sol layer ${prepared.id} preparation receipt pilot index is unreadable`,
    );
  }
  const matchingRows =
    isRecord(receiptIndex) && Array.isArray(receiptIndex.tasks)
      ? receiptIndex.tasks.filter(
          (row) => isRecord(row) && row.recordingId === prepared.id,
        )
      : [];
  if (
    !isRecord(receiptIndex) ||
    receiptIndex.schemaVersion !== 1 ||
    receiptIndex.artifactType !== "volleycut-ball-presence-pilot-index" ||
    receiptIndex.developmentOnly !== true ||
    !isRecord(receiptIndex.manifest) ||
    receiptIndex.manifest.sha256 !== prepared.base.immutable.manifest.sha256 ||
    receiptIndex.samplingPolicyId !== prepared.base.immutable.sampling.policyId ||
    receiptIndex.round !== prepared.base.immutable.sampling.round ||
    matchingRows.length !== 1 ||
    !isRecord(matchingRows[0]) ||
    matchingRows[0].taskId !== prepared.base.immutable.taskId ||
    matchingRows[0].initialTaskSha256 !== prepared.baseSha256 ||
    matchingRows[0].frameCount !== prepared.base.immutable.frames.length ||
    matchingRows[0].split !== prepared.base.immutable.recording.split
  ) {
    throw new BallLabelingWorkspaceError(
      `Sol layer ${prepared.id} preparation receipt pilot index binding is invalid`,
    );
  }

  if (
    !isRecord(value.annotations) ||
    !hasExactKeys(value.annotations, ["review", "frames"]) ||
    !isRecord(value.annotations.review) ||
    !hasExactKeys(value.annotations.review, ["status", "annotator", "reviewedAt", "notes"]) ||
    !isRecord(value.annotations.frames) ||
    Object.values(value.annotations.frames).some(
      (annotation) =>
        !isRecord(annotation) ||
        !hasExactKeys(annotation, ["status", "primaryBallState", "objects", "notes"]),
    )
  ) {
    throw new BallLabelingWorkspaceError(
      `Sol layer ${prepared.id} completed annotations contain unsupported fields`,
    );
  }
  try {
    solTask = parseBallAnnotationTask({
      schemaVersion: value.schemaVersion,
      taskType: value.taskType,
      immutable: value.immutable,
      suggestions: value.suggestions,
      annotations: value.annotations,
    });
  } catch (error) {
    throw new BallLabelingWorkspaceError(
      `Sol layer ${prepared.id} labels are invalid: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  verifyImmutableDigest(solTask, source);
  if (
    !isDeepStrictEqual(solTask.immutable, prepared.base.immutable) ||
    solTask.annotations.review.status !== "complete"
  ) {
    throw new BallLabelingWorkspaceError(
      `Sol layer ${prepared.id} is not a complete label set for the pristine task`,
    );
  }
  const annotation = solTask.annotations.frames[frameId];
  const review = solTask.annotations.review;
  const annotator = review.annotator;
  const reviewedAtValue = review.reviewedAt;
  const reviewedAt = typeof reviewedAtValue === "string" ? Date.parse(reviewedAtValue) : NaN;
  const preparedAt = Date.parse(provenance.preparedAt);
  if (
    !annotation ||
    annotation.status !== "reviewed" ||
    typeof annotator !== "string" ||
    annotator !== reviewer.agentId ||
    typeof reviewedAtValue !== "string" ||
    !/(?:Z|[+-][0-9]{2}:[0-9]{2})$/.test(reviewedAtValue) ||
    Number.isNaN(reviewedAt) ||
    reviewedAt < preparedAt
  ) {
    throw new BallLabelingWorkspaceError(`Sol layer ${prepared.id}/${frameId} is incomplete`);
  }
  return {
    provenance: {
      artifactSha256: sha256(encoded),
      annotator,
      reviewedAt: reviewedAtValue,
    },
    annotation,
  };
}

async function loadDetectorLayer(
  prepared: PreparedBallLabelingTask,
  frameId: string,
): Promise<BallComparisonLayers["layers"]["detector"]> {
  const suggestionsRoot = path.join(getBallPilotRoot(), "suggestions");
  const artifactPath = path.join(suggestionsRoot, `${prepared.id}${taskSuffix}`);
  if (!isWithin(suggestionsRoot, artifactPath) || !(await isFile(artifactPath))) return null;
  const encoded = await readFile(artifactPath);
  let value: unknown;
  try {
    value = JSON.parse(Buffer.from(encoded).toString("utf8")) as unknown;
  } catch (error) {
    throw new BallLabelingWorkspaceError(
      `detector layer ${prepared.id} is invalid: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  if (!isRecord(value) || !isDeepStrictEqual(value.immutable, prepared.base.immutable)) {
    throw new BallLabelingWorkspaceError(
      `detector layer ${prepared.id} does not match pristine task provenance`,
    );
  }
  const suggestions = value.suggestions;
  if (
    !isRecord(suggestions) ||
    suggestions.status !== "complete" ||
    !isRecord(suggestions.model) ||
    !isRecord(suggestions.frames)
  ) {
    throw new BallLabelingWorkspaceError(`detector layer ${prepared.id} is incomplete`);
  }
  const sourceTask = suggestions.model.sourceTask;
  if (!isRecord(sourceTask) || sourceTask.sha256 !== prepared.baseSha256) {
    throw new BallLabelingWorkspaceError(`detector layer ${prepared.id} source task SHA-256 differs`);
  }
  const frame = suggestions.frames[frameId];
  if (
    !isRecord(frame) ||
    typeof frame.ballPresenceProbability !== "number" ||
    !Number.isFinite(frame.ballPresenceProbability) ||
    frame.ballPresenceProbability < 0 ||
    frame.ballPresenceProbability > 1 ||
    !Array.isArray(frame.detections)
  ) {
    throw new BallLabelingWorkspaceError(`detector layer ${prepared.id}/${frameId} is invalid`);
  }
  const detections = frame.detections.map((detectionValue, index) => {
    if (
      !isRecord(detectionValue) ||
      typeof detectionValue.confidence !== "number" ||
      !Number.isFinite(detectionValue.confidence) ||
      detectionValue.confidence < 0 ||
      detectionValue.confidence > 1 ||
      !validNormalizedBox(detectionValue.bbox)
    ) {
      throw new BallLabelingWorkspaceError(
        `detector layer ${prepared.id}/${frameId} detection ${index + 1} is invalid`,
      );
    }
    return { confidence: detectionValue.confidence, bbox: detectionValue.bbox };
  });
  return {
    provenance: {
      artifactSha256: sha256(encoded),
      modelId:
        typeof suggestions.model.modelId === "string" ? suggestions.model.modelId : "unknown",
      modelSha256: validSha256(suggestions.model.modelSha256)
        ? suggestions.model.modelSha256
        : null,
    },
    ballPresenceProbability: frame.ballPresenceProbability,
    detections,
  };
}

async function allPreparedTasks(): Promise<PreparedBallLabelingTask[]> {
  const root = getBallPilotRoot();
  const existing = preparedTaskCatalogs.get(root);
  if (existing) return existing;
  const pending = (async () => {
    const indexed = await readIndex();
    return Promise.all(indexed.rows.map((row) => loadBaseTask(indexed.root, row)));
  })();
  preparedTaskCatalogs.set(root, pending);
  try {
    return await pending;
  } catch (error) {
    if (preparedTaskCatalogs.get(root) === pending) preparedTaskCatalogs.delete(root);
    throw error;
  }
}

export async function getBallLabelingCatalog(): Promise<BallLabelingTaskSummary[]> {
  const tasks = await allPreparedTasks();
  return Promise.all(
    tasks.map(async (prepared) => {
      const saved = await loadReview(prepared);
      return {
        id: prepared.id,
        taskId: prepared.base.immutable.taskId,
        split: prepared.base.immutable.recording.split,
        environment: prepared.base.immutable.recording.environment,
        frameCount: prepared.base.immutable.frames.length,
        reviewedFrameCount: reviewedFrameCount(saved.document),
        windowCount: prepared.base.immutable.windows.length,
        reviewStatus: saved.document.annotations.review.status,
        assistedFrameCount: Object.values(saved.document.annotations.frames).filter(
          (frame) => frame.proposalExposure === "shown_before_label_finalized",
        ).length,
        savedAt: saved.savedAt,
      };
    }),
  );
}

export async function getPreparedBallLabelingTask(id: string): Promise<PreparedBallLabelingTask> {
  if (!safeRecordingId(id)) throw new BallLabelingTaskNotFoundError();
  const tasks = await allPreparedTasks();
  const prepared = tasks.find((candidate) => candidate.id === id);
  if (!prepared) throw new BallLabelingTaskNotFoundError();
  return prepared;
}

export async function getBallReviewDocument(
  id: string,
): Promise<{ document: BallReviewDocument; savedAt: string | null }> {
  return loadReview(await getPreparedBallLabelingTask(id));
}

export async function getBallReviewBundle(id: string): Promise<{
  document: BallReviewDocument;
  savedAt: string | null;
}> {
  return getBallReviewDocument(id);
}

export async function saveBallReviewDocument(
  id: string,
  value: unknown,
): Promise<{ document: BallReviewDocument; savedAt: string }> {
  const prepared = await getPreparedBallLabelingTask(id);
  let submission: BallReviewDocument;
  try {
    submission = parseBallReviewDocument(value);
  } catch (error) {
    throw new BallLabelingDraftValidationError(
      error instanceof Error ? error.message : "review document is invalid",
    );
  }
  if (!isDeepStrictEqual(submission.immutable, prepared.base.immutable)) {
    throw new BallLabelingDraftValidationError("immutable source and image provenance cannot change");
  }
  return withTaskMutationLock(prepared.reviewPath, async () => {
    const [existing, audit] = await Promise.all([
      loadReview(prepared),
      readExposureAudit(prepared),
    ]);
    let auditChanged = false;
    const now = new Date().toISOString();
    for (const frame of prepared.base.immutable.frames) {
      const before = existing.document.annotations.frames[frame.id].proposalExposure;
      const after = submission.annotations.frames[frame.id].proposalExposure;
      const entry = audit.frames[frame.id];
      const auditedAssistance = (entry?.assistedSources.length ?? 0) > 0;
      if (
        (before === "shown_before_label_finalized" || auditedAssistance) &&
        after !== "shown_before_label_finalized"
      ) {
        throw new BallLabelingDraftValidationError(
          `${frame.id} proposal exposure is irreversible and cannot be marked blind`,
        );
      }

      if (!entry?.postDecisionSources.length) continue;
      const submittedRevision = humanAnnotationRevisionSha256(
        prepared,
        frame.id,
        submission.annotations.frames[frame.id],
      );
      const changedAfterReveal =
        entry.postDecisionHumanAnnotationSha256 === null ||
        submittedRevision !== entry.postDecisionHumanAnnotationSha256;
      if (after !== "shown_before_label_finalized" && !changedAfterReveal) continue;

      submission.annotations.frames[frame.id] = {
        ...submission.annotations.frames[frame.id],
        proposalExposure: "shown_before_label_finalized",
      };
      const assistedSources = mergeSources(entry.assistedSources, entry.postDecisionSources);
      audit.frames[frame.id] = {
        ...entry,
        proposalExposure: exposureFromSources(assistedSources),
        assistedSources,
        firstAssistedAt: entry.firstAssistedAt ?? now,
      };
      auditChanged = true;
    }
    const savedAt = await writeReviewArtifact(prepared, submission);
    if (auditChanged) await writeExposureAudit(prepared, audit);
    return { document: submission, savedAt };
  });
}

export async function getBallComparisonLayers(
  id: string,
  frameId: string,
  accessMode: "post-decision" | "assisted",
  requestedSources: BallProposalSource[],
): Promise<BallComparisonLayers> {
  if (
    !["post-decision", "assisted"].includes(accessMode) ||
    !validProposalSources(requestedSources) ||
    requestedSources.length === 0
  ) {
    throw new BallComparisonAccessError("comparison request is invalid");
  }
  const prepared = await getPreparedBallLabelingTask(id);
  return withTaskMutationLock(prepared.reviewPath, async () => {
    const frame = findFrame(prepared.base, frameId);
    if (!frame) throw new BallLabelingTaskNotFoundError();
    const human = await loadReview(prepared);
    if (
      accessMode === "post-decision" &&
      human.document.annotations.frames[frameId].status !== "reviewed"
    ) {
      throw new BallComparisonAccessError(
        "blind comparison is unavailable until this human decision is saved",
      );
    }

    const [sol, detector] = await Promise.all([
      requestedSources.includes("sol") ? loadSolLayer(prepared, frameId) : Promise.resolve(null),
      requestedSources.includes("detector")
        ? loadDetectorLayer(prepared, frameId)
        : Promise.resolve(null),
    ]);
    const availableSources: BallProposalSource[] = [
      ...(sol ? (["sol"] as const) : []),
      ...(detector ? (["detector"] as const) : []),
    ];
    const audit = await readExposureAudit(prepared);
    const previous = audit.frames[frameId] ?? {
      proposalExposure: "blind" as const,
      assistedSources: [],
      postDecisionSources: [],
      firstAssistedAt: null,
      firstPostDecisionRevealAt: null,
      postDecisionHumanAnnotationSha256: null,
    };
    const now = new Date().toISOString();
    const assistedSources =
      accessMode === "assisted"
        ? mergeSources(previous.assistedSources, availableSources)
        : previous.assistedSources;
    const postDecisionSources =
      accessMode === "post-decision"
        ? mergeSources(previous.postDecisionSources, availableSources)
        : previous.postDecisionSources;
    const entry: ProposalExposureAuditEntry = {
      proposalExposure: exposureFromSources(assistedSources),
      assistedSources,
      postDecisionSources,
      firstAssistedAt:
        previous.firstAssistedAt ??
        (accessMode === "assisted" && availableSources.length ? now : null),
      firstPostDecisionRevealAt:
        previous.firstPostDecisionRevealAt ??
        (accessMode === "post-decision" && availableSources.length ? now : null),
      postDecisionHumanAnnotationSha256:
        previous.postDecisionHumanAnnotationSha256 ??
        (accessMode === "post-decision" && availableSources.length
          ? humanAnnotationRevisionSha256(
              prepared,
              frameId,
              human.document.annotations.frames[frameId],
            )
          : null),
    };

    if (accessMode === "assisted" && availableSources.length) {
      human.document.annotations.frames[frameId] = {
        ...human.document.annotations.frames[frameId],
        proposalExposure: "shown_before_label_finalized",
      };
      await writeReviewArtifact(prepared, human.document);
    }
    if (availableSources.length) {
      audit.frames[frameId] = entry;
      await writeExposureAudit(prepared, audit);
    }

    return {
      schemaVersion: 1,
      taskId: prepared.base.immutable.taskId,
      frameId,
      accessMode,
      proposalExposure: entry.proposalExposure,
      layers: { sol, detector },
    };
  });
}

function findFrame(task: BallAnnotationTask, frameId: string): BallFrame | undefined {
  if (!/^f[0-9]{9}$/.test(frameId)) return undefined;
  return task.immutable.frames.find((frame) => frame.id === frameId);
}

export async function getBallFrameImage(id: string, frameId: string): Promise<BallFrameImage> {
  const prepared = await getPreparedBallLabelingTask(id);
  const frame = findFrame(prepared.base, frameId);
  if (!frame) throw new BallLabelingTaskNotFoundError();

  const root = getBallPilotRoot();
  const imagesRoot = path.join(root, "images");
  const recordingImagesRoot = path.join(imagesRoot, prepared.id);
  const imagePath = path.resolve(path.dirname(prepared.taskPath), frame.image.path);
  if (!isWithin(recordingImagesRoot, imagePath)) {
    throw new BallLabelingImageValidationError("frame image path is not bound to its source task");
  }
  let bytes: Uint8Array;
  try {
    bytes = await readFile(imagePath);
  } catch (error) {
    throw new BallLabelingImageValidationError(
      `cannot read frame image: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  const actualSha256 = sha256(bytes);
  if (actualSha256 !== frame.image.sha256) {
    throw new BallLabelingImageValidationError("frame image SHA-256 does not match its task");
  }
  return { bytes, sha256: actualSha256, filename: path.basename(imagePath) };
}

export { BallReviewValidationError };
