import assert from "node:assert/strict";
import test from "node:test";

import {
  clipSampleToInterval,
  normalizeExportIntervals,
  timelinesHaveMatchingDuration,
} from "../../lib/on-device/export-math.ts";

test("export intervals are clamped, sorted, and merged", () => {
  assert.deepEqual(
    normalizeExportIntervals(
      [
        { start: 7, end: 12 },
        { start: -2, end: 2 },
        { start: 1.5, end: 4 },
        { start: 8, end: 7 },
        { start: Number.NaN, end: 5 },
      ],
      10,
    ),
    [
      { start: 0, end: 4 },
      { start: 7, end: 10 },
    ],
  );
});

test("the frame covering a cut start is trimmed without overlapping its successor", () => {
  const first = clipSampleToInterval(9.99, 0.02, { start: 10, end: 11 });
  const second = clipSampleToInterval(10.01, 0.02, { start: 10, end: 11 });
  assert.deepEqual(first, { timestamp: 0, duration: 0.009999999999999787 });
  assert.deepEqual(second, { timestamp: 0.009999999999999787, duration: 0.019999999999999574 });
  assert.ok(first.timestamp + first.duration <= second.timestamp);
  assert.equal(clipSampleToInterval(11, 0.02, { start: 10, end: 11 }), null);
});

test("alternate raw masters require the same timeline duration", () => {
  assert.equal(timelinesHaveMatchingDuration(868.487, 868.5), true);
  assert.equal(timelinesHaveMatchingDuration(867.7, 868.5), false);
  assert.equal(timelinesHaveMatchingDuration(Number.NaN, 868.5), false);
});
