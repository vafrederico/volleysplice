import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import {
  getServingSideFlightAnnotationPath,
  getServingSideFlightEvaluationPath,
  loadServingSideFlightReview,
  saveServingSideFlightAnnotation,
} from "../../lib/server/serving-side-flight-review.ts";

const DEFAULT_SERVING_SIDE_DIRECTORY =
  "/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/serving-side";

test("flight review defaults to versioned NAS evaluation and annotations", () => {
  const previousEvaluation =
    process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_EVALUATION;
  const previousAnnotations =
    process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_ANNOTATIONS;
  try {
    delete process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_EVALUATION;
    delete process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_ANNOTATIONS;
    assert.equal(
      getServingSideFlightEvaluationPath(),
      path.join(
        DEFAULT_SERVING_SIDE_DIRECTORY,
        "serving-side-flight-v1-development.json",
      ),
    );
    assert.equal(
      getServingSideFlightAnnotationPath(),
      path.join(
        DEFAULT_SERVING_SIDE_DIRECTORY,
        "serving-side-flight-error-annotations-v1.json",
      ),
    );
  } finally {
    if (previousEvaluation === undefined) {
      delete process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_EVALUATION;
    } else {
      process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_EVALUATION = previousEvaluation;
    }
    if (previousAnnotations === undefined) {
      delete process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_ANNOTATIONS;
    } else {
      process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_ANNOTATIONS =
        previousAnnotations;
    }
  }
});

test("flight failure annotations round-trip and stay bound to predictions", async () => {
  const temporary = await mkdtemp(
    path.join(tmpdir(), "volleycut-serving-side-flight-review-"),
  );
  const evaluationPath = path.join(temporary, "evaluation.json");
  const reportPath = path.join(temporary, "report.json");
  const annotationPath = path.join(temporary, "annotations.json");
  const previousEvaluation =
    process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_EVALUATION;
  const previousAnnotations =
    process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_ANNOTATIONS;
  const previousReport = process.env.VOLLEYCUT_SERVING_SIDE_REPORT;
  try {
    process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_EVALUATION = evaluationPath;
    process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_ANNOTATIONS = annotationPath;
    process.env.VOLLEYCUT_SERVING_SIDE_REPORT = reportPath;
    const evaluation = {
      schemaVersion: 1,
      kind: "volleycut-serving-side-flight-development-evaluation-v1",
      createdAt: "2026-08-20T20:00:00.000Z",
      selection: {
        selectedCandidate: {
          configuration: "384x216-r4c6",
          featureFamily: "v2-plus-flight-recording-rank",
          l2: 0.1,
          evaluation: {
            predictionDigest: "a".repeat(64),
            pooledMetrics: {
              rows: 2,
              accuracy: 0.5,
              balancedAccuracy: 0.5,
              nearPrecision: 0,
              nearRecall: 0,
              farPrecision: 0.5,
              farRecall: 1,
            },
          },
        },
      },
      selectedPredictions: [
        {
          rallyId: "video-one:rally:1",
          recordingId: "video-one",
          environment: "indoor",
          sourceGroup: "fixture-source",
          decision: "near",
          probabilityNear: 0.2,
          prediction: "far",
          correct: false,
        },
        {
          rallyId: "video-one:rally:2",
          recordingId: "video-one",
          environment: "indoor",
          sourceGroup: "fixture-source",
          decision: "far",
          probabilityNear: 0.1,
          prediction: "far",
          correct: true,
        },
      ],
    };
    const report = {
      kind: "serving-side-fixture",
      createdAt: "2026-08-20T19:00:00.000Z",
      labels: {
        files: [
          {
            recordingId: "video-one",
            durationSeconds: 100,
            videoFilename: "video-one.mp4",
          },
        ],
      },
      rallies: [
        {
          rallyId: "video-one:rally:1",
          recordingId: "video-one",
          environment: "indoor",
          sourceGroup: "fixture-source",
          split: "train",
          start: 10,
          end: 15,
        },
        {
          rallyId: "video-one:rally:2",
          recordingId: "video-one",
          environment: "indoor",
          sourceGroup: "fixture-source",
          split: "train",
          start: 30,
          end: 35,
        },
      ],
    };
    await Promise.all([
      writeFile(evaluationPath, `${JSON.stringify(evaluation)}\n`, "utf8"),
      writeFile(reportPath, `${JSON.stringify(report)}\n`, "utf8"),
    ]);

    const loaded = await loadServingSideFlightReview();
    assert.equal(loaded.results.length, 2);
    assert.equal(loaded.results.filter((row) => !row.correct).length, 1);
    assert.equal(loaded.recordings[0]?.errors, 1);
    assert.deepEqual(loaded.annotationState.annotations, {});

    const saved = await saveServingSideFlightAnnotation({
      schemaVersion: 1,
      experimentSha256: loaded.experimentSha256,
      rallyId: "video-one:rally:1",
      annotation: {
        serverVisibility: "offscreen",
        contactTiming: "before-anchor",
        correctedServeAnchorSeconds: 8.75,
        ballFlightVisibility: "visible",
        motionDirection: "matches-human-side",
        notes: "Ball enters after an offscreen contact.",
      },
    });
    assert.ok(saved.savedAt);
    assert.ok(saved.annotations["video-one:rally:1"]?.reviewedAt);
    assert.equal(
      saved.annotations["video-one:rally:1"]?.correctedServeAnchorSeconds,
      8.75,
    );
    assert.deepEqual(
      (await loadServingSideFlightReview()).annotationState,
      saved,
    );
    assert.deepEqual(JSON.parse(await readFile(annotationPath, "utf8")), saved);

    await assert.rejects(
      saveServingSideFlightAnnotation({
        schemaVersion: 1,
        experimentSha256: loaded.experimentSha256,
        rallyId: "video-one:rally:1",
        annotation: {
          serverVisibility: "visible",
          contactTiming: "on-anchor",
          correctedServeAnchorSeconds: 11,
          ballFlightVisibility: "visible",
          motionDirection: "matches-human-side",
          notes: "",
        },
      }),
      /cannot attach a corrected time/i,
    );
    await assert.rejects(
      saveServingSideFlightAnnotation({
        schemaVersion: 1,
        experimentSha256: loaded.experimentSha256,
        rallyId: "video-one:rally:999",
        annotation: null,
      }),
      /unknown flight review rally/i,
    );

    const cleared = await saveServingSideFlightAnnotation({
      schemaVersion: 1,
      experimentSha256: loaded.experimentSha256,
      rallyId: "video-one:rally:1",
      annotation: null,
    });
    assert.deepEqual(cleared.annotations, {});
  } finally {
    if (previousEvaluation === undefined) {
      delete process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_EVALUATION;
    } else {
      process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_EVALUATION = previousEvaluation;
    }
    if (previousAnnotations === undefined) {
      delete process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_ANNOTATIONS;
    } else {
      process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_ANNOTATIONS =
        previousAnnotations;
    }
    if (previousReport === undefined) {
      delete process.env.VOLLEYCUT_SERVING_SIDE_REPORT;
    } else {
      process.env.VOLLEYCUT_SERVING_SIDE_REPORT = previousReport;
    }
    await rm(temporary, { recursive: true, force: true });
  }
});
