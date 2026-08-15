import { timingSafeEqual } from "node:crypto";
import { copyFile, lstat, mkdtemp, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";

import { PRODUCTION_MODEL_ID } from "../production-model.ts";
import {
  DEFAULT_ON_DEVICE_RUNTIME_VARIANT,
  isOnDeviceRuntimeVariant,
  type OnDeviceRuntimeVariant,
} from "../on-device/runtime-variants.ts";
import type { PreparedLabelingTask } from "./labeling-tasks.ts";

export const ON_DEVICE_BATCH_MODEL_ID = PRODUCTION_MODEL_ID;
export const ON_DEVICE_BATCH_FEATURE_PATH = "training-proxy";
export const ON_DEVICE_BATCH_MODEL_VERSION =
  "environment-specialists-v2-all-labels";
export const ON_DEVICE_BATCH_BUNDLE_SHA256 =
  "d2c2c11e8fed8b6c6ad77d244b613e81d5bab101939a8f57be5166b45ebca78f";

const TOKEN_ENV = "VOLLEYCUT_ON_DEVICE_BATCH_TOKEN";
const OUTPUT_ROOT_ENV = "VOLLEYCUT_ON_DEVICE_BATCH_OUTPUT_ROOT";
const MAX_INTERVALS = 1_000;
const DURATION_TOLERANCE_SECONDS = 0.1;

const RUNTIME_VARIANT_CONFIG = {
  "linear-v1": {
    analysisIdPrefix: `model-browser-on-device-${PRODUCTION_MODEL_ID.replace("model-", "")}--`,
    method: "browser-on-device-webcodecs-opencv-wasm-v1",
    variantLabel: `Browser on-device · ${PRODUCTION_MODEL_ID}`,
    variantDescription:
      "Browser-native WebCodecs, OpenCV WASM, and CPU inference over the fixed training proxy; media-feature parity remains unvalidated.",
    audioResampler: "deterministic-linear-48khz-to-16khz",
    warning:
      "The browser uses a deterministic linear 48 kHz to 16 kHz audio resampler that does not match FFmpeg/libswresample; this comparison run is experimental.",
    titleSuffix: "browser on-device",
  },
  "libswresample-wasm-v1": {
    analysisIdPrefix:
      `model-browser-on-device-libswresample-wasm-${PRODUCTION_MODEL_ID.replace("model-", "")}--`,
    method: "browser-on-device-webcodecs-opencv-libswresample-wasm-v1",
    variantLabel:
      `Browser on-device · libswresample WASM · ${PRODUCTION_MODEL_ID}`,
    variantDescription:
      "Browser-native WebCodecs, OpenCV WASM, FFmpeg libswresample WASM audio conversion, and CPU inference over the fixed training proxy; end-to-end parity remains under evaluation.",
    audioResampler: "ffmpeg-libswresample-wasm",
    warning:
      "The browser uses the experimental FFmpeg/libswresample WASM path for 48 kHz to 16 kHz audio; exact end-to-end feature and interval parity remains under evaluation.",
    titleSuffix: "browser on-device · libswresample WASM",
  },
} as const satisfies Record<
  OnDeviceRuntimeVariant,
  {
    analysisIdPrefix: string;
    method: string;
    variantLabel: string;
    variantDescription: string;
    audioResampler: string;
    warning: string;
    titleSuffix: string;
  }
>;

const MODEL_PARTS = {
  rally: {
    version: "environment-specialists-v2-all-labels-rally",
    sha256: "d084247aa09fd60b10a45458d150c4c1e1132b79b262009d2295e6c6b5e75693",
  },
  serve: {
    version: "environment-specialists-v2-all-labels-serve",
    sha256: "0fc2f32e25d1784ec131ced2bb874ba90ba8e625726da0c728205e37d8ff11f6",
  },
  deadState: {
    version: "environment-specialists-v2-all-labels-dead-state",
    sha256: "1ca43e38eefc0a3b0554b8818fe5c77492dc8bd697301fc55d229bf4092b2329",
  },
} as const;

export const ON_DEVICE_BATCH_RECORDING_IDS = [
  "indoor-source-07",
  "beach-source-02",
  "beach-source-01",
  "grass-source-09",
  "grass-source-01",
  "grass-source-10",
  "grass-source-04",
  "indoor-source-01",
  "indoor-source-05",
] as const;

const recordingIds = new Set<string>(ON_DEVICE_BATCH_RECORDING_IDS);

type BatchTask = Pick<
  PreparedLabelingTask,
  "id" | "batch" | "document" | "originalFilename" | "proxySize"
>;

export type OnDeviceBatchTask = BatchTask;

type BrowserMedia = {
  duration: number;
  mimeType: string;
  width: 960;
  height: 540;
  rotation: 0;
  videoCodec: "avc";
  videoCodecString: string | null;
  canDecodeVideo: true;
  hasAudio: true;
  audioCodec: "aac";
  sampleRate: 48_000;
  channels: 2;
  canDecodeAudio: true;
};

type BrowserInterval = {
  id: string;
  start: number;
  end: number;
  confidence: number;
  included: true;
};

export type OnDeviceBatchSubmission = {
  recordingId: string;
  modelId: typeof ON_DEVICE_BATCH_MODEL_ID;
  featurePath: typeof ON_DEVICE_BATCH_FEATURE_PATH;
  runtimeVariant: OnDeviceRuntimeVariant;
  media: BrowserMedia;
  intervals: BrowserInterval[];
  provenance: {
    secureContext: true;
    userAgent: string;
    completedAt: string;
  };
};

export type OnDeviceBatchVideo = {
  id: string;
  filename: string;
  originalFilename: string;
  environment: "indoor" | "grass" | "beach";
  duration: number;
  size: number;
  roi: { x: number; y: number; width: number; height: number };
  mediaUrl: string;
  completed: boolean;
};

export class OnDeviceBatchConfigurationError extends Error {}
export class OnDeviceBatchAuthorizationError extends Error {}
export class OnDeviceBatchValidationError extends Error {}
export class OnDeviceBatchConflictError extends Error {}

function object(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function exactKeys(value: Record<string, unknown>, expected: readonly string[], field: string): void {
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (actual.length !== wanted.length || actual.some((key, index) => key !== wanted[index])) {
    throw new OnDeviceBatchValidationError(`${field} has unexpected fields`);
  }
}

function finite(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new OnDeviceBatchValidationError(`${field} must be finite`);
  }
  return value;
}

export function onDeviceBatchAnalysisId(
  recordingId: string,
  runtimeVariant: OnDeviceRuntimeVariant = DEFAULT_ON_DEVICE_RUNTIME_VARIANT,
): string {
  return `${RUNTIME_VARIANT_CONFIG[runtimeVariant].analysisIdPrefix}${recordingId}`;
}

export function onDeviceRuntimeVariantFromRequest(request: Request): OnDeviceRuntimeVariant {
  const variants = new URL(request.url).searchParams.getAll("variant");
  if (variants.length === 0) return DEFAULT_ON_DEVICE_RUNTIME_VARIANT;
  if (variants.length !== 1 || !isOnDeviceRuntimeVariant(variants[0])) {
    throw new OnDeviceBatchValidationError(
      "variant must be linear-v1 or libswresample-wasm-v1",
    );
  }
  return variants[0];
}

function configuredToken(): string {
  const token = process.env[TOKEN_ENV];
  if (typeof token !== "string" || token.length === 0) {
    throw new OnDeviceBatchConfigurationError(`${TOKEN_ENV} is required`);
  }
  return token;
}

export async function configuredOnDeviceBatchOutputRoot(): Promise<string> {
  const configured = process.env[OUTPUT_ROOT_ENV];
  if (typeof configured !== "string" || configured.length === 0) {
    throw new OnDeviceBatchConfigurationError(`${OUTPUT_ROOT_ENV} is required`);
  }
  if (!path.isAbsolute(configured)) {
    throw new OnDeviceBatchConfigurationError(`${OUTPUT_ROOT_ENV} must be absolute`);
  }
  const outputRoot = path.resolve(configured);
  if (outputRoot === path.parse(outputRoot).root) {
    throw new OnDeviceBatchConfigurationError(`${OUTPUT_ROOT_ENV} cannot be a filesystem root`);
  }
  try {
    const metadata = await stat(outputRoot);
    if (!metadata.isDirectory()) throw new Error("not a directory");
  } catch {
    throw new OnDeviceBatchConfigurationError(`${OUTPUT_ROOT_ENV} must be an existing directory`);
  }
  return outputRoot;
}

export function assertOnDeviceBatchAuthorized(request: Request): void {
  const expected = Buffer.from(`Bearer ${configuredToken()}`, "utf8");
  const supplied = Buffer.from(request.headers.get("authorization") ?? "", "utf8");
  if (expected.length !== supplied.length || !timingSafeEqual(expected, supplied)) {
    throw new OnDeviceBatchAuthorizationError("Unauthorized");
  }
}

function fixedTasks(tasks: readonly BatchTask[]): BatchTask[] {
  const fullTasks = tasks.filter((task) => task.batch === "full");
  const full = new Map(
    fullTasks.map((task) => [task.id, task] as const),
  );
  if (
    fullTasks.length !== ON_DEVICE_BATCH_RECORDING_IDS.length ||
    full.size !== ON_DEVICE_BATCH_RECORDING_IDS.length ||
    ON_DEVICE_BATCH_RECORDING_IDS.some((id) => !full.has(id))
  ) {
    throw new OnDeviceBatchConfigurationError("The fixed nine-recording batch is unavailable");
  }
  return ON_DEVICE_BATCH_RECORDING_IDS.map((id) => full.get(id)!);
}

async function isCompleted(
  outputRoot: string,
  recordingId: string,
  runtimeVariant: OnDeviceRuntimeVariant,
): Promise<boolean> {
  try {
    return (
      await stat(
        path.join(outputRoot, onDeviceBatchAnalysisId(recordingId, runtimeVariant), "analysis.json"),
      )
    ).isFile();
  } catch {
    return false;
  }
}

export async function buildOnDeviceBatchCatalog(
  tasks: readonly BatchTask[],
  outputRoot: string,
  runtimeVariant: OnDeviceRuntimeVariant = DEFAULT_ON_DEVICE_RUNTIME_VARIANT,
): Promise<{
  schemaVersion: 1;
  modelId: typeof ON_DEVICE_BATCH_MODEL_ID;
  featurePath: typeof ON_DEVICE_BATCH_FEATURE_PATH;
  runtimeVariant: OnDeviceRuntimeVariant;
  videos: OnDeviceBatchVideo[];
}> {
  const videos = await Promise.all(
    fixedTasks(tasks).map(async (task): Promise<OnDeviceBatchVideo> => {
      const recording = task.document.recording;
      if (!recording.roi) {
        throw new OnDeviceBatchConfigurationError(`Recording ${task.id} has no fixed ROI`);
      }
      if (
        recording.environment !== "indoor" &&
        recording.environment !== "grass" &&
        recording.environment !== "beach"
      ) {
        throw new OnDeviceBatchConfigurationError(
          `Recording ${task.id} has an unsupported environment`,
        );
      }
      const environment = recording.environment;
      return {
        id: task.id,
        filename: recording.videoFilename,
        originalFilename: task.originalFilename,
        environment,
        duration: recording.durationSeconds,
        size: task.proxySize,
        roi: { ...recording.roi },
        mediaUrl: `/api/on-device-batch/media/${encodeURIComponent(task.id)}`,
        completed: await isCompleted(outputRoot, task.id, runtimeVariant),
      };
    }),
  );
  return {
    schemaVersion: 1,
    modelId: ON_DEVICE_BATCH_MODEL_ID,
    featurePath: ON_DEVICE_BATCH_FEATURE_PATH,
    runtimeVariant,
    videos,
  };
}

export async function getOnDeviceBatchCatalog(
  runtimeVariant: OnDeviceRuntimeVariant = DEFAULT_ON_DEVICE_RUNTIME_VARIANT,
) {
  const { getPreparedLabelingCatalog } = await import("./labeling-tasks.ts");
  const [catalog, outputRoot] = await Promise.all([
    getPreparedLabelingCatalog(),
    configuredOnDeviceBatchOutputRoot(),
  ]);
  return buildOnDeviceBatchCatalog(catalog.tasks, outputRoot, runtimeVariant);
}

function validateMedia(value: unknown, task: BatchTask): BrowserMedia {
  const media = object(value);
  if (!media) throw new OnDeviceBatchValidationError("media must be an object");
  exactKeys(
    media,
    [
      "duration",
      "mimeType",
      "width",
      "height",
      "rotation",
      "videoCodec",
      "videoCodecString",
      "canDecodeVideo",
      "hasAudio",
      "audioCodec",
      "sampleRate",
      "channels",
      "canDecodeAudio",
    ],
    "media",
  );
  const duration = finite(media.duration, "media.duration");
  if (
    duration <= 0 ||
    Math.abs(duration - task.document.recording.durationSeconds) > DURATION_TOLERANCE_SECONDS
  ) {
    throw new OnDeviceBatchValidationError("media.duration does not match the fixed proxy");
  }
  const validMp4Mime =
    typeof media.mimeType === "string" &&
    /^video\/mp4(?:;\s*codecs="[^"\r\n]{1,256}")?$/.test(media.mimeType);
  if (
    !validMp4Mime ||
    media.width !== 960 ||
    media.height !== 540 ||
    media.rotation !== 0 ||
    media.videoCodec !== "avc" ||
    !(media.videoCodecString === null ||
      (typeof media.videoCodecString === "string" &&
        media.videoCodecString.length > 0 &&
        media.videoCodecString.length <= 128)) ||
    media.canDecodeVideo !== true ||
    media.hasAudio !== true ||
    media.audioCodec !== "aac" ||
    media.sampleRate !== 48_000 ||
    media.channels !== 2 ||
    media.canDecodeAudio !== true
  ) {
    throw new OnDeviceBatchValidationError("media metadata does not match the fixed proxy format");
  }
  return media as BrowserMedia;
}

function validateIntervals(value: unknown, duration: number): BrowserInterval[] {
  if (!Array.isArray(value) || value.length > MAX_INTERVALS) {
    throw new OnDeviceBatchValidationError(`intervals must contain at most ${MAX_INTERVALS} rows`);
  }
  let previousEnd = 0;
  return value.map((candidate, index) => {
    const interval = object(candidate);
    if (!interval) throw new OnDeviceBatchValidationError(`intervals[${index}] must be an object`);
    exactKeys(interval, ["id", "start", "end", "confidence", "included"], `intervals[${index}]`);
    const expectedId = `R${String(index + 1).padStart(3, "0")}`;
    const start = finite(interval.start, `intervals[${index}].start`);
    const end = finite(interval.end, `intervals[${index}].end`);
    const confidence = finite(interval.confidence, `intervals[${index}].confidence`);
    if (interval.id !== expectedId) {
      throw new OnDeviceBatchValidationError(`intervals[${index}].id must be ${expectedId}`);
    }
    if (start < 0 || end <= start || end > duration || (index > 0 && start < previousEnd)) {
      throw new OnDeviceBatchValidationError("intervals must be bounded, sorted, and nonoverlapping");
    }
    if (confidence < 0 || confidence > 1) {
      throw new OnDeviceBatchValidationError(`intervals[${index}].confidence is out of range`);
    }
    if (interval.included !== true) {
      throw new OnDeviceBatchValidationError(`intervals[${index}].included must be true`);
    }
    previousEnd = end;
    return { id: expectedId, start, end, confidence, included: true };
  });
}

export function validateOnDeviceBatchSubmission(
  value: unknown,
  task: BatchTask,
  expectedRuntimeVariant: OnDeviceRuntimeVariant = DEFAULT_ON_DEVICE_RUNTIME_VARIANT,
): OnDeviceBatchSubmission {
  const body = object(value);
  if (!body) throw new OnDeviceBatchValidationError("request body must be an object");
  exactKeys(
    body,
    [
      "recordingId",
      "modelId",
      "featurePath",
      "runtimeVariant",
      "media",
      "intervals",
      "provenance",
    ],
    "request body",
  );
  if (body.recordingId !== task.id || !recordingIds.has(task.id)) {
    throw new OnDeviceBatchValidationError("recordingId is not in the fixed batch");
  }
  if (body.modelId !== ON_DEVICE_BATCH_MODEL_ID) {
    throw new OnDeviceBatchValidationError("modelId does not match the browser bundle");
  }
  if (body.featurePath !== ON_DEVICE_BATCH_FEATURE_PATH) {
    throw new OnDeviceBatchValidationError("featurePath must be training-proxy");
  }
  if (!isOnDeviceRuntimeVariant(body.runtimeVariant)) {
    throw new OnDeviceBatchValidationError(
      "runtimeVariant must be linear-v1 or libswresample-wasm-v1",
    );
  }
  if (body.runtimeVariant !== expectedRuntimeVariant) {
    throw new OnDeviceBatchValidationError(
      "runtimeVariant does not match the requested batch variant",
    );
  }
  const media = validateMedia(body.media, task);
  const intervals = validateIntervals(body.intervals, media.duration);
  const provenance = object(body.provenance);
  if (!provenance) throw new OnDeviceBatchValidationError("provenance must be an object");
  exactKeys(provenance, ["secureContext", "userAgent", "completedAt"], "provenance");
  if (provenance.secureContext !== true) {
    throw new OnDeviceBatchValidationError("provenance.secureContext must be true");
  }
  if (
    typeof provenance.userAgent !== "string" ||
    provenance.userAgent.length === 0 ||
    provenance.userAgent.length > 512 ||
    /[\u0000-\u001f\u007f]/.test(provenance.userAgent)
  ) {
    throw new OnDeviceBatchValidationError("provenance.userAgent is invalid");
  }
  if (
    typeof provenance.completedAt !== "string" ||
    !Number.isFinite(Date.parse(provenance.completedAt)) ||
    new Date(provenance.completedAt).toISOString() !== provenance.completedAt
  ) {
    throw new OnDeviceBatchValidationError("provenance.completedAt must be an ISO timestamp");
  }
  return {
    recordingId: task.id,
    modelId: ON_DEVICE_BATCH_MODEL_ID,
    featurePath: ON_DEVICE_BATCH_FEATURE_PATH,
    runtimeVariant: expectedRuntimeVariant,
    media,
    intervals,
    provenance: {
      secureContext: true,
      userAgent: provenance.userAgent,
      completedAt: provenance.completedAt,
    },
  };
}

function buildArtifact(task: BatchTask, submission: OnDeviceBatchSubmission, createdAt: string) {
  const recording = task.document.recording;
  const runtime = RUNTIME_VARIANT_CONFIG[submission.runtimeVariant];
  const warnings = [
    "Browser/WebCodecs feature extraction has not been validated for exact interval parity with the canonical FFmpeg/OpenCV feature path.",
    runtime.warning,
  ];
  if (recording.environment === "beach") {
    warnings.push("Beach footage was excluded from this model's training corpus; these predictions are qualitative and out of distribution.");
  }
  return {
    schemaVersion: 1,
    id: onDeviceBatchAnalysisId(task.id, submission.runtimeVariant),
    recordingId: task.id,
    title: `${task.id} — ${runtime.titleSuffix}`,
    createdAt,
    source: {
      filename: recording.videoFilename,
      contentSha256: recording.contentSha256,
      sizeBytes: task.proxySize,
      duration: submission.media.duration,
      width: submission.media.width,
      height: submission.media.height,
      fps: 30,
      hasAudio: submission.media.hasAudio,
      videoCodec: submission.media.videoCodec,
      audioCodec: submission.media.audioCodec,
    },
    analysis: {
      method: runtime.method,
      modelVersion: ON_DEVICE_BATCH_MODEL_VERSION,
      modelSha256: MODEL_PARTS.deadState.sha256,
      variantLabel: runtime.variantLabel,
      variantDescription: runtime.variantDescription,
      producer: "volleycut-browser-on-device/0.1.0",
      analysisFps: 4,
      featureVersion: "audiovisual-noise-normalized-audio-v3",
      featurePath: ON_DEVICE_BATCH_FEATURE_PATH,
      modelBundleSha256: ON_DEVICE_BATCH_BUNDLE_SHA256,
      models: MODEL_PARTS,
      provenance: {
        kind: "browser-on-device-prediction",
        secureContext: submission.provenance.secureContext,
        userAgent: submission.provenance.userAgent,
        clientCompletedAt: submission.provenance.completedAt,
        serverReceivedAt: createdAt,
        inferenceLocation: "browser",
        mediaSource: "fixed-training-proxy",
        runtimeVariant: submission.runtimeVariant,
        audioResampler: runtime.audioResampler,
      },
      warnings,
      cameraStability: recording.capture.stationary === true ? 1 : 0,
      court: {
        confidence: recording.roi ? 1 : 0,
        source: recording.roi ? "manifest-camera-roi" : "full-frame-fallback",
        roi: recording.roi ?? { x: 0, y: 0, width: 1, height: 1 },
        lines: [],
      },
    },
    rallies: submission.intervals.map((interval) => ({ ...interval })),
  };
}

export async function persistOnDeviceBatchAnalysis(
  task: BatchTask,
  submission: OnDeviceBatchSubmission,
  outputRoot: string,
  createdAt = new Date().toISOString(),
): Promise<{ analysisId: string; recordingId: string; rallyCount: number }> {
  const id = onDeviceBatchAnalysisId(task.id, submission.runtimeVariant);
  const destination = path.join(outputRoot, id);
  const staging = await mkdtemp(path.join(outputRoot, `.${id}.staging-`));
  try {
    const previewSource = path.join(
      outputRoot,
      `${PRODUCTION_MODEL_ID}--${task.id}`,
      "court-preview.jpg",
    );
    try {
      if ((await stat(previewSource)).isFile()) {
        await copyFile(previewSource, path.join(staging, "court-preview.jpg"));
      }
    } catch (error) {
      if (
        typeof error !== "object" ||
        error === null ||
        !("code" in error) ||
        (error as { code?: unknown }).code !== "ENOENT"
      ) {
        throw error;
      }
      // The comparison artifact remains valid when no existing diagnostic exists.
    }
    const artifact = buildArtifact(task, submission, createdAt);
    await writeFile(path.join(staging, "analysis.json"), `${JSON.stringify(artifact, null, 2)}\n`, {
      encoding: "utf8",
      flag: "wx",
    });
    try {
      try {
        await lstat(destination);
        throw new OnDeviceBatchConflictError("Analysis already exists");
      } catch (error) {
        if (error instanceof OnDeviceBatchConflictError) throw error;
        if (
          typeof error !== "object" ||
          error === null ||
          !("code" in error) ||
          (error as { code?: unknown }).code !== "ENOENT"
        ) {
          throw error;
        }
      }
      await rename(staging, destination);
    } catch (error) {
      if (
        typeof error === "object" &&
        error !== null &&
        "code" in error &&
        ["EEXIST", "ENOTEMPTY"].includes(String((error as { code?: unknown }).code))
      ) {
        throw new OnDeviceBatchConflictError("Analysis already exists");
      }
      throw error;
    }
    return { analysisId: id, recordingId: task.id, rallyCount: submission.intervals.length };
  } catch (error) {
    throw error;
  } finally {
    await rm(staging, { recursive: true, force: true });
  }
}

export async function saveOnDeviceBatchSubmission(
  value: unknown,
  runtimeVariant: OnDeviceRuntimeVariant = DEFAULT_ON_DEVICE_RUNTIME_VARIANT,
) {
  const recordingId = object(value)?.recordingId;
  if (typeof recordingId !== "string" || !recordingIds.has(recordingId)) {
    throw new OnDeviceBatchValidationError("recordingId is not in the fixed batch");
  }
  const { getPreparedLabelingCatalog } = await import("./labeling-tasks.ts");
  const [catalog, outputRoot] = await Promise.all([
    getPreparedLabelingCatalog(),
    configuredOnDeviceBatchOutputRoot(),
  ]);
  const task = fixedTasks(catalog.tasks).find((candidate) => candidate.id === recordingId)!;
  const submission = validateOnDeviceBatchSubmission(value, task, runtimeVariant);
  return persistOnDeviceBatchAnalysis(task, submission, outputRoot);
}

export async function readPersistedOnDeviceBatchAnalysis(
  outputRoot: string,
  recordingId: string,
  runtimeVariant: OnDeviceRuntimeVariant = DEFAULT_ON_DEVICE_RUNTIME_VARIANT,
): Promise<unknown> {
  return JSON.parse(
    await readFile(
      path.join(outputRoot, onDeviceBatchAnalysisId(recordingId, runtimeVariant), "analysis.json"),
      "utf8",
    ),
  ) as unknown;
}
