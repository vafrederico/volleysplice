import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";

import type {
  BallFlightVisibility,
  ContactTiming,
  FlightReviewSide,
  MotionDirectionAssessment,
  ServerVisibility,
  ServingSideFlightAnnotation,
  ServingSideFlightAnnotationRequest,
  ServingSideFlightAnnotationState,
  ServingSideFlightReviewData,
  ServingSideFlightReviewRecording,
  ServingSideFlightReviewResult,
} from "@/app/serving-side-flight-review/types";

import {
  getServingSideCorrectionPath,
  loadServingSideCorrectionState,
} from "./serving-side-corrections.ts";
import {
  getServingSideReviewDecisionPath,
  getServingSideReviewReportPath,
} from "./serving-side-review.ts";

const DEFAULT_EVALUATION_PATH =
  "/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serving-side/serving-side-flight-v2-development.json";
const DEFAULT_SOURCE_EXCLUSIONS_PATH =
  "/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serving-side/serving-side-source-quality-exclusions-v1.json";
const DEFAULT_ANNOTATION_FILENAME =
  "serving-side-flight-error-annotations-v2.json";
const ANNOTATION_KIND =
  "volleycut-serving-side-flight-error-annotations-v1" as const;
const RALLY_ID_PATTERN = /^[A-Za-z0-9:_-]+$/;
const SERVER_VISIBILITY = new Set<ServerVisibility>([
  "visible",
  "partial",
  "offscreen",
  "unclear",
]);
const CONTACT_TIMING = new Set<ContactTiming>([
  "on-anchor",
  "before-anchor",
  "after-anchor",
  "unclear",
]);
const BALL_VISIBILITY = new Set<BallFlightVisibility>([
  "visible",
  "not-visible",
  "unclear",
]);
const MOTION_DIRECTION = new Set<MotionDirectionAssessment>([
  "matches-human-side",
  "opposes-human-side",
  "unclear",
]);
const FLIGHT_SAMPLE_OFFSETS_SECONDS = [
  -0.15, 0.05, 0.2, 0.35, 0.55, 0.8, 1.1, 1.4, 1.75,
] as const;

type SourceQualityInterval = { start: number; end: number; reason: string };

type ExperimentIdentity = {
  kind: string;
  createdAt: string;
  sha256: string;
  predictionDigest: string;
  reportKind: string;
  reportCreatedAt: string;
  baseDecisionSha256: string;
  validRallyIds: Set<string>;
  durationByRecording: Map<string, number>;
};

export class ServingSideFlightReviewError extends Error {}
export class ServingSideFlightReviewValidationError extends ServingSideFlightReviewError {}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function objectArray(value: unknown, label: string): Record<string, unknown>[] {
  if (!Array.isArray(value) || !value.every(isRecord)) {
    throw new ServingSideFlightReviewError(
      `${label} must be an array of objects`,
    );
  }
  return value;
}

function requiredString(value: unknown, label: string): string {
  if (typeof value !== "string" || !value) {
    throw new ServingSideFlightReviewError(`${label} is invalid`);
  }
  return value;
}

function finiteNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new ServingSideFlightReviewError(`${label} is invalid`);
  }
  return value;
}

function probability(value: unknown, label: string): number {
  const result = finiteNumber(value, label);
  if (result < 0 || result > 1) {
    throw new ServingSideFlightReviewError(`${label} is outside [0, 1]`);
  }
  return result;
}

function side(value: unknown, label: string): FlightReviewSide {
  if (value !== "near" && value !== "far") {
    throw new ServingSideFlightReviewError(`${label} is not near or far`);
  }
  return value;
}

function configuredPath(value: string | undefined, fallback: string): string {
  const candidate = path.resolve(value?.trim() || fallback);
  if (candidate === path.parse(candidate).root) {
    throw new ServingSideFlightReviewError(
      "A serving-side flight review path cannot be a filesystem root",
    );
  }
  return candidate;
}

export function getServingSideFlightEvaluationPath(): string {
  return configuredPath(
    process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_EVALUATION,
    DEFAULT_EVALUATION_PATH,
  );
}

export function getServingSideFlightAnnotationPath(): string {
  const configured = process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_ANNOTATIONS;
  return configured
    ? configuredPath(configured, DEFAULT_ANNOTATION_FILENAME)
    : path.join(
        path.dirname(getServingSideFlightEvaluationPath()),
        DEFAULT_ANNOTATION_FILENAME,
      );
}

export function getServingSideSourceExclusionsPath(): string {
  return configuredPath(
    process.env.VOLLEYCUT_SERVING_SIDE_SOURCE_EXCLUSIONS,
    DEFAULT_SOURCE_EXCLUSIONS_PATH,
  );
}

async function loadSourceQualityExclusions(): Promise<
  Map<string, SourceQualityInterval[]>
> {
  let value: unknown;
  try {
    value = JSON.parse(
      await readFile(
        /* turbopackIgnore: true */ getServingSideSourceExclusionsPath(),
        "utf8",
      ),
    ) as unknown;
  } catch {
    throw new ServingSideFlightReviewError(
      "The serving-side source-quality exclusions could not be read",
    );
  }
  if (
    !isRecord(value) ||
    value.schemaVersion !== 1 ||
    value.kind !== "volleycut-serving-side-source-quality-exclusions-v1"
  ) {
    throw new ServingSideFlightReviewError(
      "The serving-side source-quality exclusions are invalid",
    );
  }
  const result = new Map<string, SourceQualityInterval[]>();
  for (const [recordIndex, record] of objectArray(
    value.records,
    "source-quality records",
  ).entries()) {
    const recordingId = requiredString(
      record.recordingId,
      `source-quality record ${recordIndex} recording id`,
    );
    const duration = finiteNumber(
      record.durationSeconds,
      `source-quality record ${recordingId} duration`,
    );
    if (duration <= 0 || result.has(recordingId)) {
      throw new ServingSideFlightReviewError(
        `Source-quality record ${recordingId} is invalid or duplicated`,
      );
    }
    let previousEnd = -1;
    const intervals = objectArray(
      record.intervals,
      `source-quality record ${recordingId} intervals`,
    ).map((interval, intervalIndex) => {
      const start = finiteNumber(
        interval.start,
        `source-quality interval ${recordingId}:${intervalIndex} start`,
      );
      const end = finiteNumber(
        interval.end,
        `source-quality interval ${recordingId}:${intervalIndex} end`,
      );
      const reason = requiredString(
        interval.reason,
        `source-quality interval ${recordingId}:${intervalIndex} reason`,
      );
      if (start < 0 || start >= end || end > duration || start < previousEnd) {
        throw new ServingSideFlightReviewError(
          `Source-quality interval ${recordingId}:${intervalIndex} is invalid or overlaps`,
        );
      }
      previousEnd = end;
      return { start, end, reason };
    });
    result.set(recordingId, intervals);
  }
  return result;
}

function flightSamplesTouchSourceExclusion(
  exclusions: Map<string, SourceQualityInterval[]>,
  recordingId: string,
  anchor: number,
): boolean {
  return (exclusions.get(recordingId) ?? []).some((interval) =>
    FLIGHT_SAMPLE_OFFSETS_SECONDS.some((offset) => {
      const sample = anchor + offset;
      return interval.start <= sample && sample < interval.end;
    }),
  );
}

async function currentCorrectionSha256(): Promise<string | null> {
  try {
    return sha256(
      await readFile(
        /* turbopackIgnore: true */ getServingSideCorrectionPath(),
        "utf8",
      ),
    );
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return null;
    throw new ServingSideFlightReviewError(
      "The current serving-side correction revision could not be read",
    );
  }
}

function sha256(content: string | Buffer): string {
  return createHash("sha256").update(content).digest("hex");
}

function selectedCandidate(evaluation: Record<string, unknown>) {
  if (!isRecord(evaluation.selection)) {
    throw new ServingSideFlightReviewError(
      "The flight evaluation has no selection record",
    );
  }
  const candidate = evaluation.selection.selectedCandidate;
  if (!isRecord(candidate) || !isRecord(candidate.evaluation)) {
    throw new ServingSideFlightReviewError(
      "The flight evaluation has no selected candidate",
    );
  }
  return candidate;
}

function predictionDigest(candidate: Record<string, unknown>): string {
  const evaluation = candidate.evaluation;
  if (!isRecord(evaluation)) {
    throw new ServingSideFlightReviewError(
      "The selected candidate evaluation is unavailable",
    );
  }
  const digest = requiredString(
    evaluation.predictionDigest,
    "selected prediction digest",
  );
  if (!/^[a-f0-9]{64}$/.test(digest)) {
    throw new ServingSideFlightReviewError(
      "The selected prediction digest is invalid",
    );
  }
  return digest;
}

function parseAnnotation(
  value: unknown,
  label: string,
  durationSeconds?: number,
): Omit<ServingSideFlightAnnotation, "reviewedAt"> & { reviewedAt?: string } {
  if (!isRecord(value)) {
    throw new ServingSideFlightReviewValidationError(`${label} is invalid`);
  }
  if (
    typeof value.serverVisibility !== "string" ||
    !SERVER_VISIBILITY.has(value.serverVisibility as ServerVisibility) ||
    typeof value.contactTiming !== "string" ||
    !CONTACT_TIMING.has(value.contactTiming as ContactTiming) ||
    typeof value.ballFlightVisibility !== "string" ||
    !BALL_VISIBILITY.has(value.ballFlightVisibility as BallFlightVisibility) ||
    typeof value.motionDirection !== "string" ||
    !MOTION_DIRECTION.has(value.motionDirection as MotionDirectionAssessment)
  ) {
    throw new ServingSideFlightReviewValidationError(
      `${label} contains an invalid review choice`,
    );
  }
  let correctedServeAnchorSeconds: number | null = null;
  if (value.correctedServeAnchorSeconds !== null) {
    correctedServeAnchorSeconds = finiteNumber(
      value.correctedServeAnchorSeconds,
      `${label} corrected serve anchor`,
    );
    if (
      correctedServeAnchorSeconds < 0 ||
      (durationSeconds !== undefined &&
        correctedServeAnchorSeconds > durationSeconds)
    ) {
      throw new ServingSideFlightReviewValidationError(
        `${label} corrected serve anchor is outside the video`,
      );
    }
  }
  if (
    (value.contactTiming === "on-anchor" ||
      value.contactTiming === "unclear") &&
    correctedServeAnchorSeconds !== null
  ) {
    throw new ServingSideFlightReviewValidationError(
      `${label} cannot attach a corrected time to this timing choice`,
    );
  }
  if (typeof value.notes !== "string" || value.notes.length > 1000) {
    throw new ServingSideFlightReviewValidationError(
      `${label} notes are invalid or too long`,
    );
  }
  const reviewedAt =
    typeof value.reviewedAt === "string" &&
    Number.isFinite(Date.parse(value.reviewedAt))
      ? value.reviewedAt
      : undefined;
  return {
    serverVisibility: value.serverVisibility as ServerVisibility,
    contactTiming: value.contactTiming as ContactTiming,
    correctedServeAnchorSeconds,
    ballFlightVisibility: value.ballFlightVisibility as BallFlightVisibility,
    motionDirection: value.motionDirection as MotionDirectionAssessment,
    notes: value.notes.trim(),
    ...(reviewedAt ? { reviewedAt } : {}),
  };
}

function emptyAnnotationState(
  identity: ExperimentIdentity,
): ServingSideFlightAnnotationState {
  return {
    schemaVersion: 1,
    kind: ANNOTATION_KIND,
    experimentKind: identity.kind,
    experimentCreatedAt: identity.createdAt,
    experimentSha256: identity.sha256,
    predictionDigest: identity.predictionDigest,
    savedAt: null,
    annotations: {},
  };
}

function parseAnnotationState(
  value: unknown,
  identity: ExperimentIdentity,
): ServingSideFlightAnnotationState {
  if (
    !isRecord(value) ||
    value.schemaVersion !== 1 ||
    value.kind !== ANNOTATION_KIND ||
    value.experimentKind !== identity.kind ||
    value.experimentCreatedAt !== identity.createdAt ||
    value.experimentSha256 !== identity.sha256 ||
    value.predictionDigest !== identity.predictionDigest ||
    !isRecord(value.annotations)
  ) {
    throw new ServingSideFlightReviewError(
      "The flight review annotations belong to a different experiment revision",
    );
  }
  const annotations: Record<string, ServingSideFlightAnnotation> = {};
  for (const [rallyId, rawAnnotation] of Object.entries(value.annotations)) {
    if (
      !RALLY_ID_PATTERN.test(rallyId) ||
      !identity.validRallyIds.has(rallyId)
    ) {
      throw new ServingSideFlightReviewError(
        `Flight review annotation references an unknown rally: ${rallyId}`,
      );
    }
    const recordingId = rallyId.split(":rally:", 1)[0];
    const annotation = parseAnnotation(
      rawAnnotation,
      `Flight review annotation ${rallyId}`,
      identity.durationByRecording.get(recordingId),
    );
    if (!annotation.reviewedAt) {
      throw new ServingSideFlightReviewError(
        `Flight review annotation ${rallyId} has no valid review time`,
      );
    }
    annotations[rallyId] = annotation as ServingSideFlightAnnotation;
  }
  return {
    schemaVersion: 1,
    kind: ANNOTATION_KIND,
    experimentKind: identity.kind,
    experimentCreatedAt: identity.createdAt,
    experimentSha256: identity.sha256,
    predictionDigest: identity.predictionDigest,
    savedAt:
      typeof value.savedAt === "string" &&
      Number.isFinite(Date.parse(value.savedAt))
        ? value.savedAt
        : null,
    annotations,
  };
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

async function readReviewInputs(): Promise<{
  evaluation: Record<string, unknown>;
  report: Record<string, unknown>;
  identity: ExperimentIdentity;
}> {
  let evaluationText: string;
  let reportText: string;
  let decisionText: string;
  try {
    [evaluationText, reportText, decisionText] = await Promise.all([
      readFile(
        /* turbopackIgnore: true */ getServingSideFlightEvaluationPath(),
        "utf8",
      ),
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
    throw new ServingSideFlightReviewError(
      `A flight review source could not be read: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  let evaluation: unknown;
  let report: unknown;
  try {
    evaluation = JSON.parse(evaluationText) as unknown;
    report = JSON.parse(reportText) as unknown;
  } catch {
    throw new ServingSideFlightReviewError(
      "A flight review source contains invalid JSON",
    );
  }
  if (!isRecord(evaluation) || !isRecord(report)) {
    throw new ServingSideFlightReviewError(
      "A flight review source has an invalid schema",
    );
  }
  if (
    evaluation.schemaVersion !== 1 ||
    evaluation.kind !==
      "volleycut-serving-side-flight-development-evaluation-v1"
  ) {
    throw new ServingSideFlightReviewError(
      "The configured artifact is not a serving-side flight evaluation",
    );
  }
  if (!isRecord(report.labels)) {
    throw new ServingSideFlightReviewError(
      "The serving-side review report has no media index",
    );
  }
  const durationByRecording = new Map(
    objectArray(report.labels.files, "serving-side media files").map((file) => [
      requiredString(file.recordingId, "media recording id"),
      finiteNumber(file.durationSeconds, "media duration"),
    ]),
  );
  const candidate = selectedCandidate(evaluation);
  const predictions = objectArray(
    evaluation.selectedPredictions,
    "selected flight predictions",
  );
  const validRallyIds = new Set(
    predictions.map((prediction) =>
      requiredString(prediction.rallyId, "flight prediction rally id"),
    ),
  );
  return {
    evaluation,
    report,
    identity: {
      kind: requiredString(evaluation.kind, "flight evaluation kind"),
      createdAt: requiredString(
        evaluation.createdAt,
        "flight evaluation creation time",
      ),
      sha256: sha256(evaluationText),
      predictionDigest: predictionDigest(candidate),
      reportKind: requiredString(report.kind, "serving-side report kind"),
      reportCreatedAt: requiredString(
        report.createdAt,
        "serving-side report creation time",
      ),
      baseDecisionSha256: sha256(decisionText),
      validRallyIds,
      durationByRecording,
    },
  };
}

async function loadAnnotationStateForIdentity(
  identity: ExperimentIdentity,
): Promise<ServingSideFlightAnnotationState> {
  try {
    return parseAnnotationState(
      JSON.parse(
        await readFile(
          /* turbopackIgnore: true */ getServingSideFlightAnnotationPath(),
          "utf8",
        ),
      ) as unknown,
      identity,
    );
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return emptyAnnotationState(identity);
    }
    if (error instanceof ServingSideFlightReviewError) throw error;
    throw new ServingSideFlightReviewError(
      "The flight review annotation file could not be read",
    );
  }
}

export async function loadServingSideFlightAnnotationState(): Promise<ServingSideFlightAnnotationState> {
  const { identity } = await readReviewInputs();
  return loadAnnotationStateForIdentity(identity);
}

function selectedMetrics(candidate: Record<string, unknown>) {
  const evaluation = candidate.evaluation;
  if (!isRecord(evaluation) || !isRecord(evaluation.pooledMetrics)) {
    throw new ServingSideFlightReviewError(
      "The selected flight metrics are unavailable",
    );
  }
  const value = evaluation.pooledMetrics;
  return {
    rows: finiteNumber(value.rows, "selected metric rows"),
    accuracy: probability(value.accuracy, "selected accuracy"),
    balancedAccuracy: probability(
      value.balancedAccuracy,
      "selected balanced accuracy",
    ),
    nearPrecision: probability(value.nearPrecision, "selected near precision"),
    nearRecall: probability(value.nearRecall, "selected near recall"),
    farPrecision: probability(value.farPrecision, "selected far precision"),
    farRecall: probability(value.farRecall, "selected far recall"),
  };
}

function ratio(numerator: number, denominator: number): number {
  return denominator ? numerator / denominator : 0;
}

function correctedMetrics(results: ServingSideFlightReviewResult[]) {
  const nearNear = results.filter(
    (row) => row.human === "near" && row.prediction === "near",
  ).length;
  const nearFar = results.filter(
    (row) => row.human === "near" && row.prediction === "far",
  ).length;
  const farNear = results.filter(
    (row) => row.human === "far" && row.prediction === "near",
  ).length;
  const farFar = results.filter(
    (row) => row.human === "far" && row.prediction === "far",
  ).length;
  const nearRows = nearNear + nearFar;
  const farRows = farNear + farFar;
  const nearRecall = ratio(nearNear, nearRows);
  const farRecall = ratio(farFar, farRows);
  return {
    rows: results.length,
    accuracy: ratio(nearNear + farFar, results.length),
    balancedAccuracy: (nearRecall + farRecall) / 2,
    nearPrecision: ratio(nearNear, nearNear + farNear),
    nearRecall,
    farPrecision: ratio(farFar, farFar + nearFar),
    farRecall,
  };
}

export async function loadServingSideFlightReview(): Promise<ServingSideFlightReviewData> {
  const { evaluation, report, identity } = await readReviewInputs();
  const [annotationState, correctionState, correctionSha256, sourceExclusions] =
    await Promise.all([
      loadAnnotationStateForIdentity(identity),
      loadServingSideCorrectionState(),
      currentCorrectionSha256(),
      loadSourceQualityExclusions(),
    ]);
  const hasCorrectionIdentity =
    correctionState.reportKind !== null ||
    correctionState.reportCreatedAt !== null ||
    correctionState.baseDecisionSha256 !== null ||
    Object.keys(correctionState.corrections).length > 0;
  if (
    hasCorrectionIdentity &&
    (correctionState.reportKind !== identity.reportKind ||
      correctionState.reportCreatedAt !== identity.reportCreatedAt ||
      correctionState.baseDecisionSha256 !== identity.baseDecisionSha256)
  ) {
    throw new ServingSideFlightReviewError(
      "The current human-label corrections belong to another label revision",
    );
  }
  const candidate = selectedCandidate(evaluation);
  const evaluationSources = isRecord(evaluation.sources)
    ? evaluation.sources
    : {};
  const evaluationCorrectionSource = isRecord(
    evaluationSources.humanLabelCorrections,
  )
    ? evaluationSources.humanLabelCorrections
    : null;
  const labelCorrectionsBakedIn =
    evaluationCorrectionSource?.sha256 === correctionSha256
      ? Object.keys(correctionState.corrections).length
      : 0;
  const rallyById = new Map(
    objectArray(report.rallies, "serving-side rallies").map((rally) => [
      requiredString(rally.rallyId, "serving-side rally id"),
      rally,
    ]),
  );
  const fileByRecording = new Map(
    objectArray(
      (report.labels as Record<string, unknown>).files,
      "serving-side media files",
    ).map((file) => [
      requiredString(file.recordingId, "media recording id"),
      file,
    ]),
  );
  let correctedNotServesExcluded = 0;
  let labelCorrectionsApplied = 0;
  let sourceQualityExcluded = 0;
  const results: ServingSideFlightReviewResult[] = objectArray(
    evaluation.selectedPredictions,
    "selected flight predictions",
  ).flatMap((prediction) => {
    const rallyId = requiredString(prediction.rallyId, "prediction rally id");
    const recordingId = requiredString(
      prediction.recordingId,
      `prediction ${rallyId} recording id`,
    );
    const rally = rallyById.get(rallyId);
    if (!rally || rally.recordingId !== recordingId) {
      throw new ServingSideFlightReviewError(
        `Flight prediction ${rallyId} is not bound to its reviewed rally`,
      );
    }
    const originalHuman = side(
      prediction.decision,
      `prediction ${rallyId} human side`,
    );
    const model = side(
      prediction.prediction,
      `prediction ${rallyId} model side`,
    );
    if (prediction.correct !== (originalHuman === model)) {
      throw new ServingSideFlightReviewError(
        `Flight prediction ${rallyId} has an inconsistent outcome`,
      );
    }
    const start = finiteNumber(rally.start, `rally ${rallyId} start`);
    if (
      flightSamplesTouchSourceExclusion(sourceExclusions, recordingId, start)
    ) {
      sourceQualityExcluded += 1;
      return [];
    }
    const correction = correctionState.corrections[rallyId];
    if (correction === "not-serve") {
      correctedNotServesExcluded += 1;
      return [];
    }
    const human = correction ?? originalHuman;
    if (human !== originalHuman) labelCorrectionsApplied += 1;
    const environment = requiredString(
      prediction.environment,
      `prediction ${rallyId} environment`,
    );
    const sourceGroup = requiredString(
      prediction.sourceGroup,
      `prediction ${rallyId} source group`,
    );
    if (
      rally.environment !== environment ||
      rally.sourceGroup !== sourceGroup
    ) {
      throw new ServingSideFlightReviewError(
        `Flight prediction ${rallyId} disagrees with source metadata`,
      );
    }
    return [
      {
        rallyId,
        recordingId,
        environment,
        sourceGroup,
        split: requiredString(rally.split, `rally ${rallyId} split`),
        start,
        end: finiteNumber(rally.end, `rally ${rallyId} end`),
        human,
        originalHuman,
        humanCorrected: human !== originalHuman,
        prediction: model,
        probabilityNear: probability(
          prediction.probabilityNear,
          `prediction ${rallyId} near probability`,
        ),
        correct: human === model,
      },
    ];
  });
  const grouped = new Map<string, ServingSideFlightReviewResult[]>();
  for (const result of results) {
    const rows = grouped.get(result.recordingId) ?? [];
    rows.push(result);
    grouped.set(result.recordingId, rows);
  }
  const recordings: ServingSideFlightReviewRecording[] = [...grouped.entries()]
    .map(([recordingId, rows]) => {
      const file = fileByRecording.get(recordingId);
      if (!file) {
        throw new ServingSideFlightReviewError(
          `Flight review recording ${recordingId} has no media entry`,
        );
      }
      const first = rows[0];
      if (!first)
        throw new ServingSideFlightReviewError("Empty recording group");
      return {
        recordingId,
        environment: first.environment,
        sourceGroup: first.sourceGroup,
        split: first.split,
        durationSeconds: finiteNumber(
          file.durationSeconds,
          `recording ${recordingId} duration`,
        ),
        videoFilename: requiredString(
          file.videoFilename,
          `recording ${recordingId} filename`,
        ),
        rows: rows.length,
        errors: rows.filter((row) => !row.correct).length,
      };
    })
    .sort((left, right) => left.recordingId.localeCompare(right.recordingId));

  const frozenMetrics = selectedMetrics(candidate);
  return {
    kind: identity.kind,
    createdAt: identity.createdAt,
    experimentSha256: identity.sha256,
    predictionDigest: identity.predictionDigest,
    configuration: requiredString(
      candidate.configuration,
      "selected configuration",
    ),
    featureFamily: requiredString(
      candidate.featureFamily,
      "selected feature family",
    ),
    l2: finiteNumber(candidate.l2, "selected L2"),
    labelCorrectionsApplied,
    labelCorrectionsBakedIn,
    correctedNotServesExcluded,
    sourceQualityExcluded,
    metrics: correctedMetrics(results),
    frozenMetrics,
    annotationState,
    recordings,
    results,
  };
}

export async function saveServingSideFlightAnnotation(
  value: unknown,
): Promise<ServingSideFlightAnnotationState> {
  if (
    !isRecord(value) ||
    value.schemaVersion !== 1 ||
    value.experimentSha256 === undefined ||
    typeof value.rallyId !== "string" ||
    !RALLY_ID_PATTERN.test(value.rallyId) ||
    !("annotation" in value)
  ) {
    throw new ServingSideFlightReviewValidationError(
      "The flight annotation request has an invalid schema",
    );
  }
  const request = value as ServingSideFlightAnnotationRequest;
  const { identity } = await readReviewInputs();
  if (request.experimentSha256 !== identity.sha256) {
    throw new ServingSideFlightReviewValidationError(
      "The flight annotation request belongs to another experiment revision",
    );
  }
  if (!identity.validRallyIds.has(request.rallyId)) {
    throw new ServingSideFlightReviewValidationError(
      `Unknown flight review rally: ${request.rallyId}`,
    );
  }
  const current = await loadAnnotationStateForIdentity(identity);
  const annotations = { ...current.annotations };
  if (request.annotation === null) {
    delete annotations[request.rallyId];
  } else {
    const recordingId = request.rallyId.split(":rally:", 1)[0];
    const annotation = parseAnnotation(
      request.annotation,
      `Flight review annotation ${request.rallyId}`,
      identity.durationByRecording.get(recordingId),
    );
    annotations[request.rallyId] = {
      ...annotation,
      reviewedAt: new Date().toISOString(),
    };
  }
  const state: ServingSideFlightAnnotationState = {
    ...emptyAnnotationState(identity),
    savedAt: new Date().toISOString(),
    annotations,
  };
  await atomicReplace(
    getServingSideFlightAnnotationPath(),
    `${JSON.stringify(state, null, 2)}\n`,
  );
  return state;
}
