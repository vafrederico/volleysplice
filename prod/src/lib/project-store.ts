import type {
  NormalizedRoi,
  OnDeviceAnalysis,
  OnDeviceMediaInfo,
} from "./on-device/types.ts";
import {
  fullAnalysisWindow,
  isFullAnalysisWindow,
  normalizeAnalysisWindow,
  type AnalysisWindow,
} from "./on-device/analysis-window.ts";
import { PRODUCTION_ENSEMBLE_MODEL_ID } from "./on-device/ensemble.ts";

const DATABASE_NAME = "volleycut-projects";
const DATABASE_VERSION = 1;
const PROJECT_STORE = "projects";

export const SELECTED_PROJECT_STORAGE_KEY = "volleycut:selected-project:v1";

export type ProjectSource = {
  name: string;
  size: number;
  lastModified: number;
  type: string;
  fingerprint?: string;
};

export type ProjectStatus =
  | "queued"
  | "analyzing"
  | "waiting"
  | "ready"
  | "error";

export type VolleyCutProject = {
  schemaVersion: 1;
  id: string;
  source: ProjectSource;
  info: OnDeviceMediaInfo;
  analysisWindow: AnalysisWindow;
  roi: NormalizedRoi;
  status: ProjectStatus;
  analysis: OnDeviceAnalysis | null;
  error: string | null;
  createdAt: string;
  updatedAt: string;
};

function requestResult<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.addEventListener("success", () => resolve(request.result), {
      once: true,
    });
    request.addEventListener("error", () => reject(request.error), {
      once: true,
    });
  });
}

function transactionComplete(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.addEventListener("complete", () => resolve(), { once: true });
    transaction.addEventListener("abort", () => reject(transaction.error), {
      once: true,
    });
    transaction.addEventListener("error", () => reject(transaction.error), {
      once: true,
    });
  });
}

function openProjectDatabase(): Promise<IDBDatabase> {
  if (!("indexedDB" in globalThis)) {
    return Promise.reject(new Error("IndexedDB is unavailable."));
  }
  const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
  request.addEventListener("upgradeneeded", () => {
    const database = request.result;
    if (!database.objectStoreNames.contains(PROJECT_STORE)) {
      database.createObjectStore(PROJECT_STORE, { keyPath: "id" });
    }
  });
  return requestResult(request);
}

function hashText(value: string): string {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(36);
}

export function projectSource(file: File): ProjectSource {
  return {
    name: file.name,
    size: file.size,
    lastModified: file.lastModified,
    type: file.type,
  };
}

const SOURCE_FINGERPRINT_SAMPLE_BYTES = 1024 * 1024;

function hex(bytes: ArrayBuffer): string {
  return Array.from(new Uint8Array(bytes), (value) =>
    value.toString(16).padStart(2, "0")
  ).join("");
}

/**
 * A transfer-stable identity without reading a multi-gigabyte source in full.
 * The file size and up to 1 MiB from each end are hashed together.
 */
export async function sourceFileFingerprint(file: File): Promise<string> {
  const wholeFile = file.size <= SOURCE_FINGERPRINT_SAMPLE_BYTES * 2;
  const first = new Uint8Array(
    await file.slice(
      0,
      wholeFile ? file.size : SOURCE_FINGERPRINT_SAMPLE_BYTES,
    ).arrayBuffer(),
  );
  const last = wholeFile
    ? new Uint8Array(0)
    : new Uint8Array(
        await file.slice(file.size - SOURCE_FINGERPRINT_SAMPLE_BYTES).arrayBuffer(),
      );
  const payload = new Uint8Array(8 + first.length + last.length);
  new DataView(payload.buffer).setBigUint64(0, BigInt(file.size), true);
  payload.set(first, 8);
  payload.set(last, 8 + first.length);
  const digest = await crypto.subtle.digest("SHA-256", payload);
  return `sampled-sha256-v1:${hex(digest)}`;
}

export function projectId(
  source: ProjectSource,
  info: OnDeviceMediaInfo,
  requestedWindow: AnalysisWindow = fullAnalysisWindow(info.duration),
): string {
  const analysisWindow = normalizeAnalysisWindow(requestedWindow, info.duration);
  const windowIdentity = isFullAnalysisWindow(analysisWindow, info.duration)
    ? ""
    : `\u0000${analysisWindow.start}\u0000${analysisWindow.end}`;
  return `project-${hashText(
    `${source.name}\u0000${source.size}\u0000${source.lastModified}\u0000${info.duration}${windowIdentity}`,
  )}`;
}

export function projectAnalysisId(project: VolleyCutProject): string | null {
  return project.analysis
    ? `${project.id}-${project.analysis.modelId}-${project.analysis.featurePath}`
    : null;
}

export function sourceMatchesFile(source: ProjectSource, file: File): boolean {
  return (
    source.name === file.name &&
    source.size === file.size &&
    source.lastModified === file.lastModified
  );
}

export async function sourceCanReconnectFile(
  source: ProjectSource,
  file: File,
): Promise<boolean> {
  if (sourceMatchesFile(source, file)) return true;
  if (!source.fingerprint || source.size !== file.size) return false;
  try {
    return source.fingerprint === await sourceFileFingerprint(file);
  } catch {
    return false;
  }
}

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function validInfo(value: unknown): value is OnDeviceMediaInfo {
  if (!value || typeof value !== "object") return false;
  const info = value as Partial<OnDeviceMediaInfo>;
  return (
    finite(info.duration) &&
    info.duration > 0 &&
    finite(info.width) &&
    info.width > 0 &&
    finite(info.height) &&
    info.height > 0 &&
    typeof info.mimeType === "string" &&
    typeof info.videoCodec === "string" &&
    typeof info.canDecodeVideo === "boolean" &&
    typeof info.hasAudio === "boolean" &&
    typeof info.canDecodeAudio === "boolean"
  );
}

function validRoi(value: unknown): value is NormalizedRoi {
  if (!value || typeof value !== "object") return false;
  const roi = value as Partial<NormalizedRoi>;
  return (
    finite(roi.x) &&
    finite(roi.y) &&
    finite(roi.width) &&
    finite(roi.height) &&
    roi.x >= 0 &&
    roi.y >= 0 &&
    roi.width > 0 &&
    roi.height > 0 &&
    roi.x + roi.width <= 1.000_001 &&
    roi.y + roi.height <= 1.000_001
  );
}

function validAnalysisWindow(value: unknown, duration: number): value is AnalysisWindow {
  if (!value || typeof value !== "object") return false;
  const window = value as Partial<AnalysisWindow>;
  return (
    finite(window.start) &&
    finite(window.end) &&
    window.start >= 0 &&
    window.start < window.end &&
    window.end <= duration + 0.000_001
  );
}

function validAnalysis(value: unknown): value is OnDeviceAnalysis {
  if (!value || typeof value !== "object") return false;
  const analysis = value as Partial<OnDeviceAnalysis>;
  const featureNames = analysis.featureNames;
  const featureValues = analysis.featureValues;
  const featuresMissing = featureNames === undefined && featureValues === undefined;
  const featuresValid =
    Array.isArray(featureNames) &&
    featureNames.length > 0 &&
    featureNames.every((name) => typeof name === "string" && name.length > 0) &&
    new Set(featureNames).size === featureNames.length &&
    featureValues instanceof Float32Array &&
    analysis.times instanceof Float64Array &&
    featureValues.length === analysis.times.length * featureNames.length;
  return (
    typeof analysis.modelId === "string" &&
    analysis.modelId.length > 0 &&
    analysis.featurePath === "local-source" &&
    Array.isArray(analysis.intervals) &&
    analysis.intervals.every(
      (interval) =>
        interval &&
        typeof interval.id === "string" &&
        finite(interval.start) &&
        finite(interval.end) &&
        finite(interval.confidence) &&
        typeof interval.included === "boolean" &&
        (interval.agreement === undefined ||
          interval.agreement === "both-models" ||
          interval.agreement === "all-labels-v2-only" ||
          interval.agreement === "previous-production-only"),
    ) &&
    analysis.times instanceof Float64Array &&
    (featuresMissing || featuresValid) &&
    analysis.rallyProbabilities instanceof Float32Array &&
    analysis.serveProbabilities instanceof Float32Array &&
    analysis.deadStateProbabilities instanceof Float32Array &&
    analysis.rallyProbabilities.length === analysis.times.length &&
    analysis.serveProbabilities.length === analysis.times.length &&
    analysis.deadStateProbabilities.length === analysis.times.length
  );
}

function validProject(value: unknown): value is VolleyCutProject {
  if (!value || typeof value !== "object") return false;
  const project = value as Partial<VolleyCutProject>;
  const statuses: ProjectStatus[] = [
    "queued",
    "analyzing",
    "waiting",
    "ready",
    "error",
  ];
  return (
    project.schemaVersion === 1 &&
    typeof project.id === "string" &&
    project.id.length > 0 &&
    Boolean(project.source) &&
    typeof project.source?.name === "string" &&
    finite(project.source?.size) &&
    finite(project.source?.lastModified) &&
    typeof project.source?.type === "string" &&
    (project.source?.fingerprint === undefined ||
      (typeof project.source.fingerprint === "string" &&
        /^sampled-sha256-v1:[0-9a-f]{64}$/.test(project.source.fingerprint))) &&
    validInfo(project.info) &&
    (project.analysisWindow === undefined ||
      validAnalysisWindow(project.analysisWindow, project.info.duration)) &&
    validRoi(project.roi) &&
    typeof project.status === "string" &&
    statuses.includes(project.status as ProjectStatus) &&
    (project.analysis === null || validAnalysis(project.analysis)) &&
    (project.error === null || typeof project.error === "string") &&
    typeof project.createdAt === "string" &&
    typeof project.updatedAt === "string" &&
    (project.status !== "ready" || project.analysis !== null)
  );
}

export function normalizeStoredProject(
  project: VolleyCutProject,
): VolleyCutProject {
  const analysisWindow = normalizeAnalysisWindow(
    project.analysisWindow,
    project.info.duration,
  );
  const normalizedProject = project.analysisWindow &&
      project.analysisWindow.start === analysisWindow.start &&
      project.analysisWindow.end === analysisWindow.end
    ? project
    : { ...project, analysisWindow };
  if (
    normalizedProject.analysis &&
    (normalizedProject.analysis.modelId !== PRODUCTION_ENSEMBLE_MODEL_ID ||
      normalizedProject.analysis.intervals.some((interval) => !interval.agreement))
  ) {
    return {
      ...normalizedProject,
      status: "waiting",
      analysis: null,
      error:
        "The production model ensemble changed. Reconnect the source to run current inference; compatible cached features will be reused.",
    };
  }
  if (normalizedProject.status === "queued" || normalizedProject.status === "analyzing") {
    return {
      ...normalizedProject,
      status: "waiting",
      error: "Reconnect the source file to resume local inference.",
    };
  }
  return normalizedProject;
}

export async function listProjects(): Promise<VolleyCutProject[]> {
  const database = await openProjectDatabase();
  try {
    const transaction = database.transaction(PROJECT_STORE, "readonly");
    const complete = transactionComplete(transaction);
    const values = await requestResult(
      transaction.objectStore(PROJECT_STORE).getAll() as IDBRequest<unknown[]>,
    );
    await complete;
    return values
      .filter(validProject)
      .map(normalizeStoredProject)
      .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
  } finally {
    database.close();
  }
}

export async function putProject(project: VolleyCutProject): Promise<void> {
  if (!validProject(project)) throw new Error("The project record is invalid.");
  const database = await openProjectDatabase();
  try {
    const transaction = database.transaction(PROJECT_STORE, "readwrite");
    const complete = transactionComplete(transaction);
    transaction.objectStore(PROJECT_STORE).put(project);
    await complete;
  } finally {
    database.close();
  }
}

export async function deleteProject(projectIdToDelete: string): Promise<void> {
  const database = await openProjectDatabase();
  try {
    const transaction = database.transaction(PROJECT_STORE, "readwrite");
    const complete = transactionComplete(transaction);
    transaction.objectStore(PROJECT_STORE).delete(projectIdToDelete);
    await complete;
  } finally {
    database.close();
  }
}
