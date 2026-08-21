import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import {
  getServingSideCorrectControlCohortPath,
  getServingSideFlightAnnotationPath,
  getServingSideFlightEvaluationPath,
  getServingSideSourceExclusionsPath,
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
  const previousSourceExclusions =
    process.env.VOLLEYCUT_SERVING_SIDE_SOURCE_EXCLUSIONS;
  const previousControlCohort =
    process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_CONTROL_COHORT;
  try {
    delete process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_EVALUATION;
    delete process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_ANNOTATIONS;
    delete process.env.VOLLEYCUT_SERVING_SIDE_SOURCE_EXCLUSIONS;
    delete process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_CONTROL_COHORT;
    assert.equal(
      getServingSideFlightEvaluationPath(),
      path.join(
        DEFAULT_SERVING_SIDE_DIRECTORY,
        "serving-side-flight-v2-development.json",
      ),
    );
    assert.equal(
      getServingSideFlightAnnotationPath(),
      path.join(
        DEFAULT_SERVING_SIDE_DIRECTORY,
        "serving-side-flight-error-annotations-v2.json",
      ),
    );
    assert.equal(
      getServingSideSourceExclusionsPath(),
      path.join(
        DEFAULT_SERVING_SIDE_DIRECTORY,
        "serving-side-source-quality-exclusions-v1.json",
      ),
    );
    assert.equal(
      getServingSideCorrectControlCohortPath(),
      path.join(
        DEFAULT_SERVING_SIDE_DIRECTORY,
        "serving-side-flight-correct-control-cohort-v1.json",
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
    if (previousSourceExclusions === undefined) {
      delete process.env.VOLLEYCUT_SERVING_SIDE_SOURCE_EXCLUSIONS;
    } else {
      process.env.VOLLEYCUT_SERVING_SIDE_SOURCE_EXCLUSIONS =
        previousSourceExclusions;
    }
    if (previousControlCohort === undefined) {
      delete process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_CONTROL_COHORT;
    } else {
      process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_CONTROL_COHORT =
        previousControlCohort;
    }
  }
});

test("flight failure annotations round-trip and stay bound to predictions", async () => {
  const temporary = await mkdtemp(
    path.join(tmpdir(), "volleycut-serving-side-flight-review-"),
  );
  const evaluationPath = path.join(temporary, "evaluation.json");
  const reportPath = path.join(temporary, "report.json");
  const decisionPath = path.join(temporary, "decisions.json");
  const correctionPath = path.join(temporary, "corrections.json");
  const annotationPath = path.join(temporary, "annotations.json");
  const sourceExclusionPath = path.join(temporary, "source-exclusions.json");
  const controlCohortPath = path.join(temporary, "control-cohort.json");
  const previousEvaluation =
    process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_EVALUATION;
  const previousAnnotations =
    process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_ANNOTATIONS;
  const previousReport = process.env.VOLLEYCUT_SERVING_SIDE_REPORT;
  const previousDecisions = process.env.VOLLEYCUT_SERVING_SIDE_DECISIONS;
  const previousCorrections = process.env.VOLLEYCUT_SERVING_SIDE_CORRECTIONS;
  const previousSourceExclusions =
    process.env.VOLLEYCUT_SERVING_SIDE_SOURCE_EXCLUSIONS;
  const previousControlCohort =
    process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_CONTROL_COHORT;
  try {
    process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_EVALUATION = evaluationPath;
    process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_ANNOTATIONS = annotationPath;
    process.env.VOLLEYCUT_SERVING_SIDE_REPORT = reportPath;
    process.env.VOLLEYCUT_SERVING_SIDE_DECISIONS = decisionPath;
    process.env.VOLLEYCUT_SERVING_SIDE_CORRECTIONS = correctionPath;
    process.env.VOLLEYCUT_SERVING_SIDE_SOURCE_EXCLUSIONS = sourceExclusionPath;
    process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_CONTROL_COHORT =
      controlCohortPath;
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
    const decisionsText = `${JSON.stringify({
      schemaVersion: 1,
      reportKind: report.kind,
      reportCreatedAt: report.createdAt,
      savedAt: "2026-08-20T19:30:00.000Z",
      decisions: {
        "video-one:rally:1": "near",
        "video-one:rally:2": "far",
      },
    })}\n`;
    const evaluationText = `${JSON.stringify(evaluation)}\n`;
    const evaluationSha256 = createHash("sha256")
      .update(evaluationText)
      .digest("hex");
    await Promise.all([
      writeFile(evaluationPath, evaluationText, "utf8"),
      writeFile(reportPath, `${JSON.stringify(report)}\n`, "utf8"),
      writeFile(decisionPath, decisionsText, "utf8"),
      writeFile(
        sourceExclusionPath,
        `${JSON.stringify({
          schemaVersion: 1,
          kind: "volleycut-serving-side-source-quality-exclusions-v1",
          createdAt: "2026-08-20T19:45:00.000Z",
          records: [],
        })}\n`,
        "utf8",
      ),
      writeFile(
        controlCohortPath,
        `${JSON.stringify({
          schemaVersion: 1,
          kind: "volleycut-serving-side-flight-correct-control-cohort-v1",
          createdAt: "2026-08-20T19:50:00.000Z",
          experiment: {
            sha256: evaluationSha256,
            kind: evaluation.kind,
            createdAt: evaluation.createdAt,
            predictionDigest: "a".repeat(64),
          },
          sampling: {
            algorithm: "minimum-one-then-proportional-largest-remainder-v1",
            targetRows: 1,
            populationRows: 1,
            sampledRows: 1,
          },
          rows: [
            {
              rallyId: "video-one:rally:2",
              stratumKey: "indoor|fixture-source|far|high",
              samplingWeight: 1,
            },
          ],
        })}\n`,
        "utf8",
      ),
    ]);

    const loaded = await loadServingSideFlightReview();
    assert.equal(loaded.results.length, 2);
    assert.equal(loaded.results.filter((row) => !row.correct).length, 1);
    assert.equal(loaded.recordings[0]?.errors, 1);
    assert.equal(loaded.sourceQualityExcluded, 0);
    assert.deepEqual(loaded.correctControlCohort.rallyIds, [
      "video-one:rally:2",
    ]);
    assert.equal(loaded.correctControlCohort.populationRows, 1);
    assert.deepEqual(loaded.annotationState.annotations, {});

    await writeFile(
      correctionPath,
      `${JSON.stringify({
        schemaVersion: 1,
        reportKind: report.kind,
        reportCreatedAt: report.createdAt,
        baseDecisionSha256: createHash("sha256")
          .update(decisionsText)
          .digest("hex"),
        savedAt: "2026-08-20T20:30:00.000Z",
        corrections: { "video-one:rally:1": "far" },
      })}\n`,
      "utf8",
    );
    const corrected = await loadServingSideFlightReview();
    assert.equal(corrected.results[0]?.originalHuman, "near");
    assert.equal(corrected.results[0]?.human, "far");
    assert.equal(corrected.results[0]?.humanCorrected, true);
    assert.equal(corrected.results[0]?.correct, true);
    assert.equal(corrected.results.filter((row) => !row.correct).length, 0);
    assert.equal(corrected.labelCorrectionsApplied, 1);
    assert.equal(corrected.metrics.accuracy, 1);

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

    await writeFile(
      correctionPath,
      `${JSON.stringify({
        schemaVersion: 1,
        reportKind: report.kind,
        reportCreatedAt: report.createdAt,
        baseDecisionSha256: createHash("sha256")
          .update(decisionsText)
          .digest("hex"),
        savedAt: "2026-08-20T20:45:00.000Z",
        corrections: {
          "video-one:rally:1": "far",
          "video-one:rally:2": "not-serve",
        },
      })}\n`,
      "utf8",
    );
    const withNotServe = await loadServingSideFlightReview();
    assert.equal(withNotServe.results.length, 1);
    assert.equal(withNotServe.correctedNotServesExcluded, 1);
    assert.equal(withNotServe.metrics.rows, 1);
    assert.equal(withNotServe.correctControlCohort.rows, 0);
    assert.equal(withNotServe.correctControlCohort.excludedByCurrentLabels, 1);

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

    await writeFile(
      sourceExclusionPath,
      `${JSON.stringify({
        schemaVersion: 1,
        kind: "volleycut-serving-side-source-quality-exclusions-v1",
        createdAt: "2026-08-20T21:00:00.000Z",
        records: [
          {
            recordingId: "video-one",
            durationSeconds: 100,
            intervals: [
              { start: 29, end: 36, reason: "camera-hit-rotated-view" },
            ],
          },
        ],
      })}\n`,
      "utf8",
    );
    const sourceFiltered = await loadServingSideFlightReview();
    assert.equal(sourceFiltered.sourceQualityExcluded, 1);
    assert.equal(sourceFiltered.results.length, 1);
    assert.equal(sourceFiltered.correctedNotServesExcluded, 0);
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
    if (previousDecisions === undefined) {
      delete process.env.VOLLEYCUT_SERVING_SIDE_DECISIONS;
    } else {
      process.env.VOLLEYCUT_SERVING_SIDE_DECISIONS = previousDecisions;
    }
    if (previousCorrections === undefined) {
      delete process.env.VOLLEYCUT_SERVING_SIDE_CORRECTIONS;
    } else {
      process.env.VOLLEYCUT_SERVING_SIDE_CORRECTIONS = previousCorrections;
    }
    if (previousSourceExclusions === undefined) {
      delete process.env.VOLLEYCUT_SERVING_SIDE_SOURCE_EXCLUSIONS;
    } else {
      process.env.VOLLEYCUT_SERVING_SIDE_SOURCE_EXCLUSIONS =
        previousSourceExclusions;
    }
    if (previousControlCohort === undefined) {
      delete process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_CONTROL_COHORT;
    } else {
      process.env.VOLLEYCUT_SERVING_SIDE_FLIGHT_CONTROL_COHORT =
        previousControlCohort;
    }
    await rm(temporary, { recursive: true, force: true });
  }
});
