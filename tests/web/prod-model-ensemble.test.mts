import assert from "node:assert/strict";
import test from "node:test";

import { mergeProductionModelIntervals } from "../../prod/src/lib/on-device/ensemble.ts";

function interval(id: string, start: number, end: number, confidence: number) {
  return { id, start, end, confidence, included: true };
}

test("production ensemble unions overlaps and flags every single-model range", () => {
  const merged = mergeProductionModelIntervals(
    [
      interval("new-1", 10, 18, 0.9),
      interval("new-2", 30, 35, 0.8),
      interval("new-3", 50, 55, 0.7),
    ],
    [
      interval("old-1", 12, 20, 0.8),
      interval("old-2", 35, 40, 0.95),
      interval("old-3", 70, 75, 0.6),
    ],
  );

  assert.deepEqual(
    merged.map(({ id, start, end, agreement }) => ({ id, start, end, agreement })),
    [
      { id: "R001", start: 10, end: 20, agreement: "both-models" },
      { id: "R002", start: 30, end: 35, agreement: "all-labels-v2-only" },
      { id: "R003", start: 35, end: 40, agreement: "previous-production-only" },
      { id: "R004", start: 50, end: 55, agreement: "all-labels-v2-only" },
      { id: "R005", start: 70, end: 75, agreement: "previous-production-only" },
    ],
  );
  assert.ok(Math.abs(merged[0].confidence - 0.85) < 1e-12);
  assert.ok(merged.slice(1).every(({ confidence }) => confidence <= 0.49));
});

test("production ensemble treats transitively overlapping detections as one agreement range", () => {
  const merged = mergeProductionModelIntervals(
    [interval("new-1", 10, 15, 0.9), interval("new-2", 18, 22, 0.8)],
    [interval("old-1", 14, 19, 0.7)],
  );

  assert.equal(merged.length, 1);
  assert.deepEqual(
    { start: merged[0].start, end: merged[0].end, agreement: merged[0].agreement },
    { start: 10, end: 22, agreement: "both-models" },
  );
});
