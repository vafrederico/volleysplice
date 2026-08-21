import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import {
  getServingSideCorrectionPath,
  saveServingSideCorrections,
} from "../../lib/server/serving-side-corrections.ts";
import {
  getServingSideResultsEvaluationPath,
  loadServingSideResults,
} from "../../lib/server/serving-side-results.ts";
import {
  getServingSideReviewDecisionPath,
  getServingSideReviewReportPath,
  loadServingSideReviewState,
  saveServingSideReviewDecisions,
} from "../../lib/server/serving-side-review.ts";

const DEFAULT_SERVING_SIDE_DIRECTORY =
  "/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serving-side";

test("serving-side review defaults to the NAS report and decision files", () => {
  const previousReport = process.env.VOLLEYCUT_SERVING_SIDE_REPORT;
  const previousDecisions = process.env.VOLLEYCUT_SERVING_SIDE_DECISIONS;
  const previousEvaluation = process.env.VOLLEYCUT_SERVING_SIDE_EVALUATION;
  const previousCorrections = process.env.VOLLEYCUT_SERVING_SIDE_CORRECTIONS;
  try {
    delete process.env.VOLLEYCUT_SERVING_SIDE_REPORT;
    delete process.env.VOLLEYCUT_SERVING_SIDE_DECISIONS;
    delete process.env.VOLLEYCUT_SERVING_SIDE_EVALUATION;
    delete process.env.VOLLEYCUT_SERVING_SIDE_CORRECTIONS;
    assert.equal(
      getServingSideReviewReportPath(),
      path.join(
        DEFAULT_SERVING_SIDE_DIRECTORY,
        "serving-side-existing-label-variants-full-nas-v2.json",
      ),
    );
    assert.equal(
      getServingSideReviewDecisionPath(),
      path.join(
        DEFAULT_SERVING_SIDE_DIRECTORY,
        "serving-side-review-decisions-full-nas-v1.json",
      ),
    );
    assert.equal(
      getServingSideResultsEvaluationPath(),
      path.join(
        DEFAULT_SERVING_SIDE_DIRECTORY,
        "serving-side-specialist-v3-dual-serve-gate-all-video-inference-v2.json",
      ),
    );
    assert.equal(
      getServingSideCorrectionPath(),
      path.join(
        DEFAULT_SERVING_SIDE_DIRECTORY,
        "serving-side-result-label-corrections-v1.json",
      ),
    );
  } finally {
    if (previousReport === undefined) {
      delete process.env.VOLLEYCUT_SERVING_SIDE_REPORT;
    } else {
      process.env.VOLLEYCUT_SERVING_SIDE_REPORT = previousReport;
    }
    if (previousDecisions === undefined) {
      delete process.env.VOLLEYCUT_SERVING_SIDE_DECISIONS;
    } else {
      process.env.VOLLEYCUT_SERVING_SIDE_DECISIONS = previousDecisions;
    }
    if (previousEvaluation === undefined) {
      delete process.env.VOLLEYCUT_SERVING_SIDE_EVALUATION;
    } else {
      process.env.VOLLEYCUT_SERVING_SIDE_EVALUATION = previousEvaluation;
    }
    if (previousCorrections === undefined) {
      delete process.env.VOLLEYCUT_SERVING_SIDE_CORRECTIONS;
    } else {
      process.env.VOLLEYCUT_SERVING_SIDE_CORRECTIONS = previousCorrections;
    }
  }
});

test("serving-side decisions round-trip through the configured durable store", async () => {
  const temporary = await mkdtemp(
    path.join(tmpdir(), "volleycut-serving-side-review-"),
  );
  const reportPath = path.join(temporary, "report.json");
  const decisionPath = path.join(temporary, "decisions.json");
  const previousReport = process.env.VOLLEYCUT_SERVING_SIDE_REPORT;
  const previousDecisions = process.env.VOLLEYCUT_SERVING_SIDE_DECISIONS;
  try {
    process.env.VOLLEYCUT_SERVING_SIDE_REPORT = reportPath;
    process.env.VOLLEYCUT_SERVING_SIDE_DECISIONS = decisionPath;
    await writeFile(
      reportPath,
      `${JSON.stringify({
        kind: "volleycut-serving-side-test-v1",
        createdAt: "2026-08-20T12:00:00.000Z",
        rallies: [
          { rallyId: "recording-one:rally-001" },
          { rallyId: "recording-one:rally-002" },
          { rallyId: "recording-two:rally-001" },
        ],
      })}\n`,
      "utf8",
    );

    assert.deepEqual((await loadServingSideReviewState()).decisions, {});

    const saved = await saveServingSideReviewDecisions({
      schemaVersion: 1,
      reportKind: "volleycut-serving-side-test-v1",
      reportCreatedAt: "2026-08-20T12:00:00.000Z",
      decisions: {
        "recording-one:rally-001": "near",
        "recording-one:rally-002": "far",
        "recording-two:rally-001": "unclear",
      },
    });

    assert.ok(saved.savedAt);
    assert.deepEqual(await loadServingSideReviewState(), saved);
    assert.deepEqual(JSON.parse(await readFile(decisionPath, "utf8")), saved);
  } finally {
    if (previousReport === undefined) {
      delete process.env.VOLLEYCUT_SERVING_SIDE_REPORT;
    } else {
      process.env.VOLLEYCUT_SERVING_SIDE_REPORT = previousReport;
    }
    if (previousDecisions === undefined) {
      delete process.env.VOLLEYCUT_SERVING_SIDE_DECISIONS;
    } else {
      process.env.VOLLEYCUT_SERVING_SIDE_DECISIONS = previousDecisions;
    }
    await rm(temporary, { recursive: true, force: true });
  }
});

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

test("serving-side result review joins frozen human and model decisions", async () => {
  const temporary = await mkdtemp(
    path.join(tmpdir(), "volleycut-serving-side-results-"),
  );
  const reportPath = path.join(temporary, "report.json");
  const decisionPath = path.join(temporary, "decisions.json");
  const evaluationPath = path.join(temporary, "evaluation.json");
  const correctionPath = path.join(temporary, "corrections.json");
  const previousReport = process.env.VOLLEYCUT_SERVING_SIDE_REPORT;
  const previousDecisions = process.env.VOLLEYCUT_SERVING_SIDE_DECISIONS;
  const previousEvaluation = process.env.VOLLEYCUT_SERVING_SIDE_EVALUATION;
  const previousCorrections = process.env.VOLLEYCUT_SERVING_SIDE_CORRECTIONS;
  try {
    process.env.VOLLEYCUT_SERVING_SIDE_REPORT = reportPath;
    process.env.VOLLEYCUT_SERVING_SIDE_DECISIONS = decisionPath;
    process.env.VOLLEYCUT_SERVING_SIDE_EVALUATION = evaluationPath;
    process.env.VOLLEYCUT_SERVING_SIDE_CORRECTIONS = correctionPath;
    assert.equal(getServingSideResultsEvaluationPath(), evaluationPath);

    const report = `${JSON.stringify({
      kind: "volleycut-serving-side-report-test-v1",
      createdAt: "2026-08-20T12:00:00.000Z",
      labels: {
        files: [
          {
            recordingId: "indoor-test-video",
            durationSeconds: 120,
            videoFilename: "indoor-test-video.mp4",
          },
        ],
      },
      rallies: [
        {
          rallyId: "indoor-test-video:rally:1",
          recordingId: "indoor-test-video",
          environment: "indoor",
          sourceGroup: "fixture-source",
          split: "test",
          sourceType: "completed-labels",
          targetStatus: "gold",
          rallyIndex: 0,
          start: 10,
          end: 15,
          notes: "Near-side server.",
          tags: ["fixture"],
          features: { pixelMotionMargin: 0.25 },
        },
        {
          rallyId: "indoor-test-video:rally:2",
          recordingId: "indoor-test-video",
          environment: "indoor",
          sourceGroup: "fixture-source",
          split: "test",
          sourceType: "completed-labels",
          targetStatus: "gold",
          rallyIndex: 1,
          start: 30,
          end: 35,
          notes: "Far-side server.",
          tags: [],
          features: { pixelMotionMargin: -0.2 },
        },
      ],
    })}\n`;
    const decisions = `${JSON.stringify({
      schemaVersion: 1,
      reportKind: "volleycut-serving-side-report-test-v1",
      reportCreatedAt: "2026-08-20T12:00:00.000Z",
      savedAt: "2026-08-20T13:00:00.000Z",
      decisions: {
        "indoor-test-video:rally:1": "near",
        "indoor-test-video:rally:2": "far",
      },
    })}\n`;
    const evaluation = `${JSON.stringify({
      kind: "volleycut-serving-side-specialist-v2-protected-test-evaluation",
      createdAt: "2026-08-20T14:00:00.000Z",
      modelFingerprint: "a".repeat(64),
      serveGateFingerprint: "b".repeat(64),
      featureFamily: "court-flow-recording-rank",
      threshold: 0.51,
      sources: {
        servingSideReport: { sha256: sha256(report) },
        reviewDecisions: { sha256: sha256(decisions) },
      },
      predictions: [
        {
          rallyId: "indoor-test-video:rally:1",
          recordingId: "indoor-test-video",
          decision: "near",
          prediction: "far",
          servePrediction: "not-serve",
          finalPrediction: "not-serve",
          serveEvidence: {
            serveAnchor: 10,
            heads: {
              allLabelsV2: {
                modelId: "model-v2",
                threshold: 0.85,
                peakProbability: 0.7,
                peakTime: 10,
                crossesThreshold: false,
                nearestDetection: null,
              },
              previousProduction: {
                modelId: "model-previous",
                threshold: 0.85,
                peakProbability: 0.8,
                peakTime: 10.25,
                crossesThreshold: false,
                nearestDetection: null,
              },
            },
          },
          nearProbability: 0.49,
        },
        {
          rallyId: "indoor-test-video:rally:2",
          recordingId: "indoor-test-video",
          decision: "far",
          prediction: "far",
          servePrediction: "serve",
          finalPrediction: "far",
          serveEvidence: {
            serveAnchor: 30,
            heads: {
              allLabelsV2: {
                modelId: "model-v2",
                threshold: 0.85,
                peakProbability: 0.9,
                peakTime: 30,
                crossesThreshold: true,
                nearestDetection: {
                  time: 30.25,
                  confidence: 0.9,
                  distanceSeconds: 0.25,
                },
              },
              previousProduction: {
                modelId: "model-previous",
                threshold: 0.85,
                peakProbability: 0.8,
                peakTime: 30,
                crossesThreshold: false,
                nearestDetection: null,
              },
            },
          },
          nearProbability: 0.2,
        },
      ],
      metrics: {
        rows: 2,
        accuracy: 0.5,
        balancedAccuracy: 0.5,
        nearPrecision: 0,
        nearRecall: 0,
        farPrecision: 0.5,
        farRecall: 1,
      },
    })}\n`;
    await Promise.all([
      writeFile(reportPath, report, "utf8"),
      writeFile(decisionPath, decisions, "utf8"),
      writeFile(evaluationPath, evaluation, "utf8"),
    ]);

    const result = await loadServingSideResults();
    assert.equal(result.results.length, 2);
    assert.equal(result.serveGateMetrics.missedServes, 1);
    assert.equal(result.results[0].finalPrediction, "not-serve");
    assert.equal(result.metrics.nearRecall, 0);
    assert.equal(result.metrics.accuracy, 0.5);
    assert.deepEqual(
      result.results.map((row) => ({
        human: row.human,
        prediction: row.prediction,
        correct: row.correct,
      })),
      [
        { human: "near", prediction: "far", correct: false },
        { human: "far", prediction: "far", correct: true },
      ],
    );
    assert.deepEqual(result.recordings[0], {
      recordingId: "indoor-test-video",
      environment: "indoor",
      sourceGroup: "fixture-source",
      split: "test",
      sourceType: "completed-labels",
      targetStatus: "gold",
      durationSeconds: 120,
      videoFilename: "indoor-test-video.mp4",
      rows: 2,
      correct: 1,
      errors: 1,
    });

    const correctedState = await saveServingSideCorrections({
      schemaVersion: 1,
      reportKind: "volleycut-serving-side-report-test-v1",
      reportCreatedAt: "2026-08-20T12:00:00.000Z",
      baseDecisionSha256: sha256(decisions),
      corrections: {
        "indoor-test-video:rally:1": "not-serve",
      },
    });
    assert.equal(
      correctedState.corrections["indoor-test-video:rally:1"],
      "not-serve",
    );
    const corrected = await loadServingSideResults();
    assert.deepEqual(
      corrected.results.map((row) => ({
        human: row.human,
        originalHuman: row.originalHuman,
        humanCorrected: row.humanCorrected,
      })),
      [
        {
          human: "not-serve",
          originalHuman: "near",
          humanCorrected: true,
        },
        {
          human: "far",
          originalHuman: "far",
          humanCorrected: false,
        },
      ],
    );
    assert.equal(corrected.metrics.rows, 1);
    assert.equal(corrected.metrics.accuracy, 1);

    const bakedEvaluation = JSON.parse(evaluation) as {
      sources: Record<string, unknown>;
      predictions: Array<Record<string, unknown>>;
    };
    bakedEvaluation.sources.humanLabelCorrections = {
      sha256: "c".repeat(64),
    };
    bakedEvaluation.predictions[0]!.decision = "far";
    await Promise.all([
      writeFile(
        evaluationPath,
        `${JSON.stringify(bakedEvaluation)}\n`,
        "utf8",
      ),
      saveServingSideCorrections({
        schemaVersion: 1,
        reportKind: "volleycut-serving-side-report-test-v1",
        reportCreatedAt: "2026-08-20T12:00:00.000Z",
        baseDecisionSha256: sha256(decisions),
        corrections: {
          "indoor-test-video:rally:1": "far",
          "indoor-test-video:rally:2": "not-serve",
        },
      }),
    ]);
    const extendedCorrections = await loadServingSideResults();
    assert.deepEqual(
      extendedCorrections.results.map((row) => row.human),
      ["far", "not-serve"],
    );

    await writeFile(
      decisionPath,
      decisions.replace(
        '"indoor-test-video:rally:2":"far"',
        '"indoor-test-video:rally:2":"near"',
      ),
      "utf8",
    );
    await assert.rejects(
      loadServingSideResults(),
      /human decisions do not match the frozen evaluation/i,
    );
  } finally {
    if (previousReport === undefined) {
      delete process.env.VOLLEYCUT_SERVING_SIDE_REPORT;
    } else {
      process.env.VOLLEYCUT_SERVING_SIDE_REPORT = previousReport;
    }
    if (previousDecisions === undefined) {
      delete process.env.VOLLEYCUT_SERVING_SIDE_DECISIONS;
    } else {
      process.env.VOLLEYCUT_SERVING_SIDE_DECISIONS = previousDecisions;
    }
    if (previousEvaluation === undefined) {
      delete process.env.VOLLEYCUT_SERVING_SIDE_EVALUATION;
    } else {
      process.env.VOLLEYCUT_SERVING_SIDE_EVALUATION = previousEvaluation;
    }
    if (previousCorrections === undefined) {
      delete process.env.VOLLEYCUT_SERVING_SIDE_CORRECTIONS;
    } else {
      process.env.VOLLEYCUT_SERVING_SIDE_CORRECTIONS = previousCorrections;
    }
    await rm(temporary, { recursive: true, force: true });
  }
});
