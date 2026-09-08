import {
  type AnalysisWindow,
  fullAnalysisWindow,
  isFullAnalysisWindow,
  normalizeAnalysisWindow,
} from "./on-device/analysis-window.ts";
import { PRODUCTION_ENSEMBLE_MODEL_ID } from "./on-device/ensemble.ts";
import { isReusableServingSideOutput } from "./on-device/serving-side-cache.ts";
import { isReusableSideSwitchOutput } from "./on-device/side-switch-model.ts";
import {
  SUPPRESSION_ARTIFACT_SHA256,
  SUPPRESSION_DECODER_VERSION,
  SUPPRESSION_MODEL_ID,
  SUPPRESSION_WEIGHTS_SHA256,
} from "./on-device/suppression-model.ts";
import { SUPPRESSION_POLICY_CONTRACT_VERSION } from "./on-device/suppression-policy.ts";
import type {
  NormalizedRoi,
  OnDeviceAnalysis,
  OnDeviceMediaInfo,
} from "./on-device/types.ts";
import type { CutDraft } from "./cut-draft.ts";

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

export type VolleySpliceProject = {
  schemaVersion: 1;
  id: string;
  source: ProjectSource;
  info: OnDeviceMediaInfo;
  analysisWindow: AnalysisWindow;
  roi: NormalizedRoi;
  /** Legacy score-tracking preference retained for stored-project compatibility. */
  servingSideEnabled?: boolean;
  /** Whether inference should generate team side-switch markers. */
  sideSwitchEnabled?: boolean;
  status: ProjectStatus;
  analysis: OnDeviceAnalysis | null;
  error: string | null;
  importedFeedback?: {
    schemaVersion: 1 | 2 | 3;
    generatedAt: string;
    importedAt: string;
    originalProjectId: string;
    originalAnalysisId: string;
    runtimeVariant: string;
    warnings: string[];
    initialDraft: CutDraft;
  };
  /** Latest editor state, mirrored from localStorage for durable project restore. */
  reviewDraft?: CutDraft;
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
    value.toString(16).padStart(2, "0"),
  ).join("");
}

/**
 * A transfer-stable identity without reading a multi-gigabyte source in full.
 * The file size and up to 1 MiB from each end are hashed together.
 */
export async function sourceFileFingerprint(file: File): Promise<string> {
  const wholeFile = file.size <= SOURCE_FINGERPRINT_SAMPLE_BYTES * 2;
  const first = new Uint8Array(
    await file
      .slice(0, wholeFile ? file.size : SOURCE_FINGERPRINT_SAMPLE_BYTES)
      .arrayBuffer(),
  );
  const last = wholeFile
    ? new Uint8Array(0)
    : new Uint8Array(
        await file
          .slice(file.size - SOURCE_FINGERPRINT_SAMPLE_BYTES)
          .arrayBuffer(),
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
  const analysisWindow = normalizeAnalysisWindow(
    requestedWindow,
    info.duration,
  );
  const windowIdentity = isFullAnalysisWindow(analysisWindow, info.duration)
    ? ""
    : `\u0000${analysisWindow.start}\u0000${analysisWindow.end}`;
  return `project-${hashText(
    `${source.name}\u0000${source.size}\u0000${source.lastModified}\u0000${info.duration}${windowIdentity}`,
  )}`;
}

export function projectAnalysisId(project: VolleySpliceProject): string | null {
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
    return source.fingerprint === (await sourceFileFingerprint(file));
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

function validAnalysisWindow(
  value: unknown,
  duration: number,
): value is AnalysisWindow {
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

function validInterval(value: unknown): boolean {
  if (!value || typeof value !== "object") return false;
  const interval = value as Record<string, unknown>;
  return (
    typeof interval.id === "string" &&
    finite(interval.start) &&
    finite(interval.end) &&
    interval.end > interval.start &&
    finite(interval.confidence) &&
    typeof interval.included === "boolean"
  );
}

function validServeOutput(value: unknown, rows: number): boolean {
  if (!value || typeof value !== "object") return false;
  const output = value as Record<string, unknown>;
  return (
    output.probabilities instanceof Float32Array &&
    output.probabilities.length === rows &&
    !output.probabilities.some(
      (probability) =>
        !Number.isFinite(probability) || probability < 0 || probability > 1,
    ) &&
    Array.isArray(output.detections) &&
    output.detections.every((candidate) => {
      if (!candidate || typeof candidate !== "object") return false;
      const detection = candidate as Record<string, unknown>;
      return (
        finite(detection.time) &&
        detection.time >= 0 &&
        finite(detection.confidence) &&
        detection.confidence >= 0 &&
        detection.confidence <= 1
      );
    })
  );
}

function validProductionServeOutputs(value: unknown, rows: number): boolean {
  if (!value || typeof value !== "object") return false;
  const outputs = value as Record<string, unknown>;
  return (
    validServeOutput(outputs.allLabelsV2, rows) &&
    validServeOutput(outputs.previousProduction, rows)
  );
}

function validProbabilityVector(value: unknown, rows: number): boolean {
  return (
    value instanceof Float32Array &&
    value.length === rows &&
    !value.some(
      (probability) =>
        !Number.isFinite(probability) || probability < 0 || probability > 1,
    )
  );
}

function validProductionStateOutputs(value: unknown, rows: number): boolean {
  if (!value || typeof value !== "object") return false;
  const outputs = value as Record<string, unknown>;
  return (["allLabelsV2", "previousProduction"] as const).every((source) => {
    const output = outputs[source];
    if (!output || typeof output !== "object") return false;
    const state = output as Record<string, unknown>;
    return (
      validProbabilityVector(state.rallyProbabilities, rows) &&
      validProbabilityVector(state.deadStateProbabilities, rows)
    );
  });
}

function validSuppression(
  value: unknown,
  rows: number,
  allowHistoricalArtifact: boolean,
): boolean {
  if (!value || typeof value !== "object") return false;
  const suppression = value as Record<string, unknown>;
  return (
    (allowHistoricalArtifact ||
      (suppression.modelId === SUPPRESSION_MODEL_ID &&
        suppression.artifactSha256 === SUPPRESSION_ARTIFACT_SHA256 &&
        suppression.weightsSha256 === SUPPRESSION_WEIGHTS_SHA256 &&
        suppression.decoderVersion === SUPPRESSION_DECODER_VERSION &&
        suppression.policyContractVersion ===
          SUPPRESSION_POLICY_CONTRACT_VERSION)) &&
    suppression.probabilities instanceof Float32Array &&
    suppression.probabilities.length === rows &&
    Array.isArray(suppression.decodedIntervals) &&
    suppression.decodedIntervals.every(validInterval) &&
    Array.isArray(suppression.suggestions) &&
    suppression.suggestions.every((candidate) => {
      if (!candidate || typeof candidate !== "object") return false;
      const suggestion = candidate as Record<string, unknown>;
      return (
        typeof suggestion.id === "string" &&
        typeof suggestion.logicalId === "string" &&
        typeof suggestion.suppressionEventId === "string" &&
        finite(suggestion.start) &&
        finite(suggestion.end) &&
        suggestion.end > suggestion.start &&
        finite(suggestion.score) &&
        Array.isArray(suggestion.sourceProductionIds) &&
        suggestion.sourceProductionIds.every((id) => typeof id === "string") &&
        Array.isArray(suggestion.eligiblePolicyIds) &&
        suggestion.eligiblePolicyIds.every(
          (id) =>
            id === "conservative" || id === "balanced" || id === "aggressive",
        )
      );
    }) &&
    typeof suppression.identicalPolicyResults === "boolean"
  );
}

function validAnalysis(
  value: unknown,
  allowHistoricalArtifacts = false,
): value is OnDeviceAnalysis {
  if (!value || typeof value !== "object") return false;
  const analysis = value as Partial<OnDeviceAnalysis>;
  const featureNames = analysis.featureNames;
  const featureValues = analysis.featureValues;
  const featuresMissing =
    featureNames === undefined && featureValues === undefined;
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
    analysis.deadStateProbabilities.length === analysis.times.length &&
    (analysis.productionComponents === undefined ||
      (Array.isArray(analysis.productionComponents.allLabelsV2) &&
        analysis.productionComponents.allLabelsV2.every(validInterval) &&
        Array.isArray(analysis.productionComponents.previousProduction) &&
        analysis.productionComponents.previousProduction.every(
          validInterval,
        ))) &&
    (analysis.productionServeOutputs === undefined ||
      validProductionServeOutputs(
        analysis.productionServeOutputs,
        analysis.times.length,
      )) &&
    (analysis.productionStateOutputs === undefined ||
      validProductionStateOutputs(
        analysis.productionStateOutputs,
        analysis.times.length,
      )) &&
    (analysis.sideSwitch === undefined ||
      (analysis.sideSwitch !== null &&
        typeof analysis.sideSwitch === "object")) &&
    (analysis.suppression === undefined ||
      validSuppression(
        analysis.suppression,
        analysis.times.length,
        allowHistoricalArtifacts,
      ))
  );
}

function validImportedFeedback(
  value: unknown,
  projectIdValue: string,
  analysisId: string | null,
): value is NonNullable<VolleySpliceProject["importedFeedback"]> {
  if (!value || typeof value !== "object" || !analysisId) return false;
  const feedback = value as Partial<
    NonNullable<VolleySpliceProject["importedFeedback"]>
  >;
  const draft = feedback.initialDraft as Partial<CutDraft> | undefined;
  return (
    (feedback.schemaVersion === 1 ||
      feedback.schemaVersion === 2 ||
      feedback.schemaVersion === 3) &&
    typeof feedback.generatedAt === "string" &&
    Number.isFinite(Date.parse(feedback.generatedAt)) &&
    typeof feedback.importedAt === "string" &&
    Number.isFinite(Date.parse(feedback.importedAt)) &&
    typeof feedback.originalProjectId === "string" &&
    feedback.originalProjectId.length > 0 &&
    typeof feedback.originalAnalysisId === "string" &&
    feedback.originalAnalysisId.length > 0 &&
    typeof feedback.runtimeVariant === "string" &&
    feedback.runtimeVariant.length > 0 &&
    Array.isArray(feedback.warnings) &&
    feedback.warnings.every((warning) => typeof warning === "string") &&
    Boolean(draft) &&
    draft?.analysisId === analysisId &&
    draft?.recordingId === projectIdValue &&
    Array.isArray(draft?.cuts) &&
    Array.isArray(draft?.ignoredIntervals)
  );
}

function validProject(value: unknown): value is VolleySpliceProject {
  if (!value || typeof value !== "object") return false;
  const project = value as Partial<VolleySpliceProject>;
  const statuses: ProjectStatus[] = [
    "queued",
    "analyzing",
    "waiting",
    "ready",
    "error",
  ];
  const importedFeedbackPresent = project.importedFeedback !== undefined;
  const analysisId =
    typeof project.id === "string" &&
    project.analysis &&
    typeof project.analysis.modelId === "string" &&
    typeof project.analysis.featurePath === "string"
      ? `${project.id}-${project.analysis.modelId}-${project.analysis.featurePath}`
      : null;
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
    (project.servingSideEnabled === undefined ||
      typeof project.servingSideEnabled === "boolean") &&
    (project.sideSwitchEnabled === undefined ||
      typeof project.sideSwitchEnabled === "boolean") &&
    typeof project.status === "string" &&
    statuses.includes(project.status as ProjectStatus) &&
    (project.analysis === null ||
      validAnalysis(project.analysis, importedFeedbackPresent)) &&
    (project.error === null || typeof project.error === "string") &&
    (!importedFeedbackPresent ||
      validImportedFeedback(
        project.importedFeedback,
        project.id,
        analysisId,
      )) &&
    (project.reviewDraft === undefined ||
      (project.reviewDraft !== null && typeof project.reviewDraft === "object")) &&
    typeof project.createdAt === "string" &&
    typeof project.updatedAt === "string" &&
    (project.status !== "ready" || project.analysis !== null)
  );
}

export function normalizeStoredProject(
  project: VolleySpliceProject,
): VolleySpliceProject {
  const analysisWindow = normalizeAnalysisWindow(
    project.analysisWindow,
    project.info.duration,
  );
  const windowNormalizedProject =
    project.analysisWindow &&
    project.analysisWindow.start === analysisWindow.start &&
    project.analysisWindow.end === analysisWindow.end
      ? project
      : { ...project, analysisWindow };
  const servingSideNormalizedProject =
    !windowNormalizedProject.importedFeedback &&
    windowNormalizedProject.analysis?.servingSide &&
    !isReusableServingSideOutput(
      windowNormalizedProject.analysis.servingSide,
      windowNormalizedProject.analysis.intervals,
    )
      ? {
          ...windowNormalizedProject,
          analysis: {
            ...windowNormalizedProject.analysis,
            servingSide: undefined,
          },
        }
      : windowNormalizedProject;
  const normalizedProject =
    !servingSideNormalizedProject.importedFeedback &&
    servingSideNormalizedProject.analysis?.sideSwitch &&
    !isReusableSideSwitchOutput(
      servingSideNormalizedProject.analysis.sideSwitch,
    )
      ? {
          ...servingSideNormalizedProject,
          analysis: {
            ...servingSideNormalizedProject.analysis,
            sideSwitch: undefined,
          },
        }
      : servingSideNormalizedProject;
  if (
    normalizedProject.analysis &&
    !normalizedProject.importedFeedback &&
    (normalizedProject.analysis.modelId !== PRODUCTION_ENSEMBLE_MODEL_ID ||
      normalizedProject.analysis.intervals.some(
        (interval) => !interval.agreement,
      ))
  ) {
    return {
      ...normalizedProject,
      status: "waiting",
      analysis: null,
      error:
        "The production model ensemble changed. Reconnect the source to run current inference; compatible cached features will be reused.",
    };
  }
  if (
    normalizedProject.status === "queued" ||
    normalizedProject.status === "analyzing"
  ) {
    return {
      ...normalizedProject,
      status: "waiting",
      error: "Reconnect the source file to resume local inference.",
    };
  }
  return normalizedProject;
}

export async function listProjects(): Promise<VolleySpliceProject[]> {
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

export async function putProject(project: VolleySpliceProject): Promise<void> {
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

export async function putProjectReviewDraft(
  projectIdToUpdate: string,
  reviewDraft: CutDraft,
): Promise<void> {
  const database = await openProjectDatabase();
  try {
    const transaction = database.transaction(PROJECT_STORE, "readwrite");
    const complete = transactionComplete(transaction);
    const store = transaction.objectStore(PROJECT_STORE);
    const stored = await requestResult(
      store.get(projectIdToUpdate) as IDBRequest<unknown>,
    );
    if (!validProject(stored)) {
      throw new Error("The project record is unavailable or invalid.");
    }
    store.put({ ...stored, reviewDraft });
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
