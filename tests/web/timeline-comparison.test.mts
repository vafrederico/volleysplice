import assert from "node:assert/strict";
import test from "node:test";

import {
  buildLiveTimeComparisonSegments,
  buildPaddingSegments,
  calculateDurationDeltaPercent,
  calculateF1,
  calculateLiveTimeMetrics,
  compareRalliesToHumanLabels,
  excludeIgnoredTime,
  markModelPaddingOrigins,
  padAndMergeRallies,
  totalRallySeconds,
} from "../../lib/timeline-comparison.ts";
import type { Rally } from "../../lib/edit-list.ts";

function rally(id: string, start: number, end: number): Rally {
  return { id, start, end, confidence: 1, included: true };
}

test("timeline comparison separates human matches, additions, and misses", () => {
  const comparison = compareRalliesToHumanLabels(
    [rally("P1", 10, 20), rally("P2", 40, 50)],
    [rally("H1", 11, 20), rally("H2", 70, 80)],
  );

  assert.deepEqual([...comparison.matchedPredictionIds], ["P1"]);
  assert.deepEqual(
    comparison.unmatchedPredictionRallies.map((item) => item.id),
    ["P2"],
  );
  assert.deepEqual(comparison.missedHumanRallies.map((item) => item.id), ["H2"]);
});

test("timeline comparison uses strict one-to-one chronological matches", () => {
  const comparison = compareRalliesToHumanLabels(
    [rally("P1", 15, 30), rally("P2", 20, 30)],
    [rally("H1", 10, 20), rally("H2", 20, 30)],
    0.2,
  );

  assert.deepEqual([...comparison.matchedPredictionIds], ["P1", "P2"]);
  assert.deepEqual(comparison.missedHumanRallies, []);
});

test("timeline comparison treats overlap below the IoU threshold as an addition and a miss", () => {
  const comparison = compareRalliesToHumanLabels(
    [rally("P1", 18, 28)],
    [rally("H1", 10, 20)],
  );

  assert.deepEqual([...comparison.matchedPredictionIds], []);
  assert.deepEqual(comparison.missedHumanRallies.map((item) => item.id), ["H1"]);
});

test("live-time metrics use merged interval unions", () => {
  const metrics = calculateLiveTimeMetrics(
    [rally("P1", 4, 8), rally("P2", 6, 14)],
    [rally("H1", 0, 10), rally("H2", 8, 12)],
  );

  assert.equal(metrics.precision, 0.8);
  assert.equal(metrics.recall, 8 / 12);
  assert.ok(Math.abs(metrics.f1 - 8 / 11) < 1e-12);
});

test("live-time metrics report zero precision and recall for an empty prediction", () => {
  const metrics = calculateLiveTimeMetrics([], [rally("H1", 0, 10)]);

  assert.deepEqual(metrics, { precision: 0, recall: 0, f1: 0 });
});

test("ignored time is removed from precision rather than counted as a false positive", () => {
  const predictions = [rally("P1", 0, 20)];
  const human = [rally("H1", 0, 10)];
  const ignored = [{ start: 10, end: 20 }];

  assert.deepEqual(calculateLiveTimeMetrics(predictions, human, ignored), {
    precision: 1,
    recall: 1,
    f1: 1,
  });
  assert.deepEqual(
    excludeIgnoredTime([rally("P2", 5, 25)], [{ start: 10, end: 20 }])
      .map(({ start, end }) => [start, end]),
    [[5, 10], [20, 25]],
  );
});

test("ignored time is absent from model comparison coloring", () => {
  const segments = buildLiveTimeComparisonSegments(
    [rally("P1", 0, 20)],
    [rally("H1", 0, 10)],
    [{ start: 10, end: 20 }],
  );

  assert.deepEqual(
    segments.map(({ start, end, kind }) => ({ start, end, kind })),
    [{ start: 0, end: 10, kind: "match" }],
  );
});

test("model timeline splits core, before-padding, and after-padding sources", () => {
  const segments = markModelPaddingOrigins(
    [{ id: "S1", start: 7, end: 22, kind: "added", predictionId: "P1" }],
    [rally("P1", 10, 20)],
    3,
    2,
    60,
  );

  assert.deepEqual(
    segments.map(({ start, end, paddingOrigin }) => ({ start, end, paddingOrigin })),
    [
      { start: 7, end: 10, paddingOrigin: "before" },
      { start: 10, end: 20, paddingOrigin: undefined },
      { start: 20, end: 22, paddingOrigin: "after" },
    ],
  );
});

test("merged before and after padding uses crosshatch provenance", () => {
  const segments = markModelPaddingOrigins(
    [{ id: "S1", start: 5, end: 35, kind: "match", predictionId: "P1" }],
    [rally("P1", 10, 20), rally("P2", 24, 30)],
    5,
    5,
    60,
  );

  assert.deepEqual(
    segments.map(({ start, end, paddingOrigin }) => ({ start, end, paddingOrigin })),
    [
      { start: 5, end: 10, paddingOrigin: "before" },
      { start: 10, end: 20, paddingOrigin: undefined },
      { start: 20, end: 24, paddingOrigin: "both" },
      { start: 24, end: 30, paddingOrigin: undefined },
      { start: 30, end: 35, paddingOrigin: "after" },
    ],
  );
});

test("live-time comparison splits a partial prediction into matched, added, and missed ranges", () => {
  const segments = buildLiveTimeComparisonSegments(
    [rally("P1", 5, 15)],
    [rally("H1", 10, 20)],
  );

  assert.deepEqual(
    segments.map(({ start, end, kind, predictionId }) => ({ start, end, kind, predictionId })),
    [
      { start: 5, end: 10, kind: "added", predictionId: "P1" },
      { start: 10, end: 15, kind: "match", predictionId: "P1" },
      { start: 15, end: 20, kind: "missed", predictionId: undefined },
    ],
  );
});

test("live-time comparison handles multiple human overlaps inside one prediction", () => {
  const segments = buildLiveTimeComparisonSegments(
    [rally("P1", 5, 25)],
    [rally("H1", 10, 15), rally("H2", 18, 20)],
  );

  assert.deepEqual(
    segments.filter((segment) => segment.predictionId).map(({ start, end, kind }) => ({ start, end, kind })),
    [
      { start: 5, end: 10, kind: "added" },
      { start: 10, end: 15, kind: "match" },
      { start: 15, end: 18, kind: "added" },
      { start: 18, end: 20, kind: "match" },
      { start: 20, end: 25, kind: "added" },
    ],
  );
});

test("activity padding merges model rallies before live-time coloring", () => {
  const padded = padAndMergeRallies(
    [rally("P1", 10, 20), rally("P2", 24, 30)],
    3,
    2,
    60,
  );

  assert.equal(padded.length, 1);
  assert.equal(padded[0].start, 7);
  assert.equal(padded[0].end, 32);
  assert.deepEqual(padded[0].rallyIds, ["P1", "P2"]);
});

test("symmetric 2s and 3s model padding counts overlapping exports once", () => {
  const predictions = [rally("P1", 10, 20), rally("P2", 23, 30)];

  const twoSeconds = padAndMergeRallies(predictions, 2, 2, 60);
  assert.deepEqual(
    twoSeconds.map(({ start, end, rallyIds }) => ({ start, end, rallyIds })),
    [{ start: 8, end: 32, rallyIds: ["P1", "P2"] }],
  );
  assert.equal(totalRallySeconds(twoSeconds), 24);

  const threeSeconds = padAndMergeRallies(predictions, 3, 3, 60);
  assert.deepEqual(
    threeSeconds.map(({ start, end, rallyIds }) => ({ start, end, rallyIds })),
    [{ start: 7, end: 33, rallyIds: ["P1", "P2"] }],
  );
  assert.equal(totalRallySeconds(threeSeconds), 26);
});

test("reference padding excludes opaque core intervals", () => {
  const padding = buildPaddingSegments(
    [rally("H1", 10, 20), rally("H2", 30, 40)],
    3,
    2,
    60,
  );

  assert.deepEqual(
    padding.map(({ start, end, kind }) => ({ start, end, kind })),
    [
      { start: 7, end: 10, kind: "added" },
      { start: 20, end: 22, kind: "added" },
      { start: 27, end: 30, kind: "added" },
      { start: 40, end: 42, kind: "added" },
    ],
  );
});

test("overlapping reference padding is drawn as one translucent gap", () => {
  const padding = buildPaddingSegments(
    [rally("H1", 10, 20), rally("H2", 25, 30)],
    2,
    6,
    60,
  );

  assert.deepEqual(
    padding.map(({ start, end }) => [start, end]),
    [[8, 10], [20, 25], [30, 36]],
  );
});

test("model summaries distinguish core-human and equally padded-human coverage", () => {
  const coreHuman = [rally("H1", 10, 20)];
  const paddedModel = padAndMergeRallies(
    [rally("P1", 10, 20)],
    2,
    2,
    60,
  );
  const paddedHuman = padAndMergeRallies(coreHuman, 2, 2, 60);

  const coreMetrics = calculateLiveTimeMetrics(paddedModel, coreHuman);
  const paddedMetrics = calculateLiveTimeMetrics(paddedModel, paddedHuman);
  assert.equal(paddedModel[0].end - paddedModel[0].start, 14);
  assert.equal(coreMetrics.precision, 10 / 14);
  assert.equal(coreMetrics.recall, 1);
  assert.deepEqual(paddedMetrics, { precision: 1, recall: 1, f1: 1 });
});

test("hybrid F1 combines padded-label precision with core-label recall", () => {
  const paddedPrecision = 0.75;
  const coreRecall = 0.6;

  assert.ok(Math.abs(calculateF1(paddedPrecision, coreRecall) - 2 / 3) < 1e-12);
  assert.equal(calculateF1(0, 0), 0);
});

test("export summaries compare merged model duration with the padded human export", () => {
  const paddedHuman = [rally("H1", 0, 20), rally("H2", 15, 30)];
  const humanSeconds = totalRallySeconds(paddedHuman);

  assert.equal(humanSeconds, 30);
  assert.equal(calculateDurationDeltaPercent(36, humanSeconds), 20);
  assert.equal(calculateDurationDeltaPercent(24, humanSeconds), -20);
  assert.equal(calculateDurationDeltaPercent(24, 0), null);
});
