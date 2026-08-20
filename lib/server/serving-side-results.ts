import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";

import type {
  ServingSideResult,
  ServingSideResultMetrics,
  ServingSideResultRecording,
  ServingSideResultsData,
} from "@/app/serving-side-results/types";

import {
  loadServingSideCorrectionState,
  type ServingSideCorrectionState,
} from "./serving-side-corrections.ts";

import {
  getServingSideReviewDecisionPath,
  getServingSideReviewReportPath,
} from "./serving-side-review.ts";

const DEFAULT_EVALUATION_PATH =
  "/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serving-side/serving-side-specialist-v2-all-video-inference.json";

export class ServingSideResultsError extends Error {}

function configuredPath(value: string | undefined, fallback: string): string {
  const candidate = path.resolve(value?.trim() || fallback);
  if (candidate === path.parse(candidate).root) {
    throw new ServingSideResultsError(
      "The serving-side evaluation path cannot be a filesystem root",
    );
  }
  return candidate;
}

export function getServingSideResultsEvaluationPath(): string {
  return configuredPath(
    process.env.VOLLEYCUT_SERVING_SIDE_EVALUATION,
    DEFAULT_EVALUATION_PATH,
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function objectArray(value: unknown, label: string): Record<string, unknown>[] {
  if (!Array.isArray(value) || !value.every(isRecord)) {
    throw new ServingSideResultsError(`${label} must be an array of objects`);
  }
  return value;
}

function requiredString(
  value: unknown,
  label: string,
  pattern?: RegExp,
): string {
  if (
    typeof value !== "string" ||
    !value ||
    (pattern && !pattern.test(value))
  ) {
    throw new ServingSideResultsError(`${label} is invalid`);
  }
  return value;
}

function finiteNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new ServingSideResultsError(`${label} is invalid`);
  }
  return value;
}

function side(value: unknown, label: string): "near" | "far" {
  if (value !== "near" && value !== "far") {
    throw new ServingSideResultsError(`${label} is not near or far`);
  }
  return value;
}

function sha256(content: Buffer): string {
  return createHash("sha256").update(content).digest("hex");
}

function sourceHash(
  evaluation: Record<string, unknown>,
  source: "servingSideReport" | "reviewDecisions",
): string {
  const sources = evaluation.sources;
  if (!isRecord(sources) || !isRecord(sources[source])) {
    throw new ServingSideResultsError(
      `evaluation is missing its ${source} source binding`,
    );
  }
  return requiredString(
    sources[source].sha256,
    `evaluation ${source} SHA-256`,
    /^[a-f0-9]{64}$/,
  );
}

function metrics(value: unknown): ServingSideResultMetrics {
  if (!isRecord(value)) {
    throw new ServingSideResultsError("evaluation metrics are unavailable");
  }
  return {
    rows: finiteNumber(value.rows, "metric rows"),
    accuracy: finiteNumber(value.accuracy, "metric accuracy"),
    balancedAccuracy: finiteNumber(
      value.balancedAccuracy,
      "metric balanced accuracy",
    ),
    nearPrecision: finiteNumber(value.nearPrecision, "near precision"),
    nearRecall: finiteNumber(value.nearRecall, "near recall"),
    farPrecision: finiteNumber(value.farPrecision, "far precision"),
    farRecall: finiteNumber(value.farRecall, "far recall"),
  };
}

function numericFeatures(value: unknown): Record<string, number | null> {
  if (!isRecord(value)) return {};
  return Object.fromEntries(
    Object.entries(value).map(([name, feature]) => [
      name,
      typeof feature === "number" && Number.isFinite(feature) ? feature : null,
    ]),
  );
}

function ratio(numerator: number, denominator: number): number {
  return denominator ? numerator / denominator : 0;
}

function metricsForResults(
  results: ServingSideResult[],
): ServingSideResultMetrics {
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
  const nearRecall = ratio(nearNear, nearNear + nearFar);
  const farRecall = ratio(farFar, farFar + farNear);
  const rows = nearNear + nearFar + farNear + farFar;
  return {
    rows,
    accuracy: ratio(nearNear + farFar, rows),
    balancedAccuracy: (nearRecall + farRecall) / 2,
    nearPrecision: ratio(nearNear, nearNear + farNear),
    nearRecall,
    farPrecision: ratio(farFar, farFar + nearFar),
    farRecall,
  };
}

export async function loadServingSideResults(): Promise<ServingSideResultsData> {
  const evaluationPath = getServingSideResultsEvaluationPath();
  const reportPath = getServingSideReviewReportPath();
  const decisionPath = getServingSideReviewDecisionPath();
  let evaluationBuffer: Buffer;
  let reportBuffer: Buffer;
  let decisionBuffer: Buffer;
  try {
    [evaluationBuffer, reportBuffer, decisionBuffer] = await Promise.all([
      readFile(/* turbopackIgnore: true */ evaluationPath),
      readFile(/* turbopackIgnore: true */ reportPath),
      readFile(/* turbopackIgnore: true */ decisionPath),
    ]);
  } catch (error) {
    throw new ServingSideResultsError(
      `Serving-side result artifacts could not be read: ${error instanceof Error ? error.message : String(error)}`,
    );
  }

  let evaluation: unknown;
  let report: unknown;
  let review: unknown;
  try {
    evaluation = JSON.parse(evaluationBuffer.toString("utf8")) as unknown;
    report = JSON.parse(reportBuffer.toString("utf8")) as unknown;
    review = JSON.parse(decisionBuffer.toString("utf8")) as unknown;
  } catch {
    throw new ServingSideResultsError(
      "A serving-side result artifact contains invalid JSON",
    );
  }
  let storedCorrections: ServingSideCorrectionState;
  try {
    storedCorrections = await loadServingSideCorrectionState();
  } catch (error) {
    throw new ServingSideResultsError(
      `The human-label correction overlay could not be read: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  if (!isRecord(evaluation) || !isRecord(report) || !isRecord(review)) {
    throw new ServingSideResultsError(
      "A serving-side result artifact has an invalid schema",
    );
  }
  if (sha256(reportBuffer) !== sourceHash(evaluation, "servingSideReport")) {
    throw new ServingSideResultsError(
      "The serving-side report does not match the frozen evaluation",
    );
  }
  if (sha256(decisionBuffer) !== sourceHash(evaluation, "reviewDecisions")) {
    throw new ServingSideResultsError(
      "The human decisions do not match the frozen evaluation",
    );
  }
  if (
    review.reportKind !== report.kind ||
    review.reportCreatedAt !== report.createdAt ||
    !isRecord(review.decisions)
  ) {
    throw new ServingSideResultsError(
      "The human decisions belong to a different serving-side report",
    );
  }
  const decisions = review.decisions;
  const baseDecisionSha256 = sha256(decisionBuffer);
  const hasStoredCorrectionIdentity =
    storedCorrections.reportKind !== null ||
    storedCorrections.reportCreatedAt !== null ||
    storedCorrections.baseDecisionSha256 !== null ||
    Object.keys(storedCorrections.corrections).length > 0;
  if (
    hasStoredCorrectionIdentity &&
    (storedCorrections.reportKind !== report.kind ||
      storedCorrections.reportCreatedAt !== report.createdAt ||
      storedCorrections.baseDecisionSha256 !== baseDecisionSha256)
  ) {
    throw new ServingSideResultsError(
      "The human-label corrections belong to a different frozen label revision",
    );
  }

  const rallyById = new Map(
    objectArray(report.rallies, "report rallies").map((rally) => [
      requiredString(rally.rallyId, "report rally id"),
      rally,
    ]),
  );
  const labels = report.labels;
  if (!isRecord(labels)) {
    throw new ServingSideResultsError("The serving-side report has no labels");
  }
  const fileByRecording = new Map(
    objectArray(labels.files, "report label files").map((file) => [
      requiredString(file.recordingId, "recording id"),
      file,
    ]),
  );

  const predictions = objectArray(
    evaluation.predictions,
    "evaluation predictions",
  );
  for (const [rallyId, correction] of Object.entries(
    storedCorrections.corrections,
  )) {
    if (
      !rallyById.has(rallyId) ||
      (decisions[rallyId] !== "near" && decisions[rallyId] !== "far") ||
      (correction !== "near" &&
        correction !== "far" &&
        correction !== "not-serve")
    ) {
      throw new ServingSideResultsError(
        `Human-label correction ${rallyId} is not bound to a clear reviewed rally`,
      );
    }
  }
  const results: ServingSideResult[] = predictions.map((prediction) => {
    const rallyId = requiredString(prediction.rallyId, "prediction rally id");
    const recordingId = requiredString(
      prediction.recordingId,
      "prediction recording id",
    );
    const rally = rallyById.get(rallyId);
    if (!rally || rally.recordingId !== recordingId) {
      throw new ServingSideResultsError(
        `Prediction ${rallyId} is not bound to its source rally`,
      );
    }
    const originalHuman = side(decisions[rallyId], `human decision ${rallyId}`);
    if (
      side(prediction.decision, `evaluation decision ${rallyId}`) !==
      originalHuman
    ) {
      throw new ServingSideResultsError(
        `Prediction ${rallyId} disagrees with the frozen human decision`,
      );
    }
    const correction = storedCorrections.corrections[rallyId];
    const human = correction ?? originalHuman;
    const model = side(prediction.prediction, `model prediction ${rallyId}`);
    const nearProbability = finiteNumber(
      prediction.nearProbability,
      `near probability ${rallyId}`,
    );
    if (nearProbability < 0 || nearProbability > 1) {
      throw new ServingSideResultsError(
        `Near probability ${rallyId} is outside [0, 1]`,
      );
    }
    return {
      rallyId,
      recordingId,
      environment: requiredString(rally.environment, `environment ${rallyId}`),
      sourceGroup: requiredString(rally.sourceGroup, `source group ${rallyId}`),
      split: requiredString(rally.split, `split ${rallyId}`),
      sourceType:
        typeof rally.sourceType === "string" ? rally.sourceType : null,
      targetStatus:
        typeof rally.targetStatus === "string" ? rally.targetStatus : null,
      rallyIndex: finiteNumber(rally.rallyIndex, `rally index ${rallyId}`),
      start: finiteNumber(rally.start, `rally start ${rallyId}`),
      end: finiteNumber(rally.end, `rally end ${rallyId}`),
      human,
      originalHuman,
      humanCorrected: human !== originalHuman,
      prediction: model,
      nearProbability,
      correct: human !== "not-serve" && human === model,
      notes: typeof rally.notes === "string" ? rally.notes : null,
      tags: Array.isArray(rally.tags)
        ? rally.tags.filter((tag): tag is string => typeof tag === "string")
        : [],
      features: numericFeatures(rally.features),
    };
  });

  const resultsByRecording = new Map<string, ServingSideResult[]>();
  for (const result of results) {
    const rows = resultsByRecording.get(result.recordingId) ?? [];
    rows.push(result);
    resultsByRecording.set(result.recordingId, rows);
  }
  const recordings: ServingSideResultRecording[] = [...resultsByRecording]
    .map(([recordingId, rows]) => {
      const file = fileByRecording.get(recordingId);
      if (!file) {
        throw new ServingSideResultsError(
          `Evaluation recording ${recordingId} has no media entry`,
        );
      }
      const sideRows = rows.filter((row) => row.human !== "not-serve");
      const correct = sideRows.filter((row) => row.correct).length;
      return {
        recordingId,
        environment: rows[0].environment,
        sourceGroup: rows[0].sourceGroup,
        split: rows[0].split,
        sourceType: rows[0].sourceType,
        targetStatus: rows[0].targetStatus,
        durationSeconds: finiteNumber(
          file.durationSeconds,
          `duration ${recordingId}`,
        ),
        videoFilename:
          typeof file.videoFilename === "string"
            ? file.videoFilename
            : `${recordingId}.mp4`,
        rows: rows.length,
        correct,
        errors: sideRows.length - correct,
      };
    })
    .sort((left, right) => {
      if (left.split === "test" && right.split !== "test") return -1;
      if (right.split === "test" && left.split !== "test") return 1;
      const leftSideRows = left.correct + left.errors;
      const rightSideRows = right.correct + right.errors;
      const accuracyDelta =
        ratio(left.correct, leftSideRows) - ratio(right.correct, rightSideRows);
      return accuracyDelta || left.recordingId.localeCompare(right.recordingId);
    });

  const heldOutAll = evaluation.heldOutAll;
  const overall = isRecord(evaluation.metrics)
    ? evaluation.metrics
    : isRecord(heldOutAll)
      ? heldOutAll.overall
      : null;
  const artifactMetrics = metrics(overall);
  if (artifactMetrics.rows !== results.length) {
    throw new ServingSideResultsError(
      "Evaluation metrics and prediction rows have different counts",
    );
  }

  return {
    kind: requiredString(evaluation.kind, "evaluation kind"),
    createdAt: requiredString(evaluation.createdAt, "evaluation timestamp"),
    modelFingerprint: requiredString(
      evaluation.modelFingerprint,
      "model fingerprint",
      /^[a-f0-9]{64}$/,
    ),
    selectedFeatureSet: requiredString(
      evaluation.selectedFeatureSet ?? evaluation.featureFamily,
      "selected feature set",
    ),
    threshold: finiteNumber(evaluation.threshold, "model threshold"),
    metrics: metricsForResults(results),
    correctionState: {
      schemaVersion: 1,
      reportKind: requiredString(report.kind, "report kind"),
      reportCreatedAt: requiredString(report.createdAt, "report timestamp"),
      baseDecisionSha256,
      savedAt: storedCorrections.savedAt,
      corrections: storedCorrections.corrections,
    },
    recordings,
    results,
  };
}
