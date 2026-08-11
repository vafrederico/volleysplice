import assert from "node:assert/strict";
import test from "node:test";
import { buildEditList, formatTime, timelinePercent, timelineTicks, type Rally } from "../../lib/edit-list.ts";

const rallies: Rally[] = [
  { id: "late", start: 21, end: 30, confidence: 0.8, included: true },
  { id: "early", start: 10, end: 20, confidence: 0.8, included: true },
];

test("buildEditList sorts and merges overlapping padded intervals", () => {
  const intervals = buildEditList(rallies, 3, 2, 60);
  assert.equal(intervals.length, 1);
  assert.equal(intervals[0].keptStart, 7);
  assert.equal(intervals[0].keptEnd, 32);
  assert.deepEqual(intervals[0].rallyIds, ["early", "late"]);
});

test("buildEditList clamps boundaries and ignores invalid or excluded rallies", () => {
  const intervals = buildEditList([
    { id: "start", start: 1, end: 4, confidence: 0.8, included: true },
    { id: "bad", start: 8, end: 7, confidence: 0.8, included: true },
    { id: "excluded", start: 6, end: 8, confidence: 0.8, included: false },
    { id: "end", start: 9, end: 12, confidence: 0.8, included: true },
  ], 3, 3, 10);
  assert.deepEqual(intervals.map(({ keptStart, keptEnd }) => [keptStart, keptEnd]), [[0, 10]]);
});

test("time helpers are bounded and support hour-long recordings", () => {
  assert.equal(formatTime(Number.NaN), "0:00");
  assert.equal(formatTime(-2), "0:00");
  assert.equal(formatTime(3661.9), "1:01:01");
  assert.equal(timelinePercent(120, 60), 100);
  assert.equal(timelinePercent(-1, 60), 0);
  assert.deepEqual(timelineTicks(100, 3), [0, 50, 100]);
});
