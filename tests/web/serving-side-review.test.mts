import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

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
  try {
    delete process.env.VOLLEYCUT_SERVING_SIDE_REPORT;
    delete process.env.VOLLEYCUT_SERVING_SIDE_DECISIONS;
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
