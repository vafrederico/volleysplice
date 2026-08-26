import assert from "node:assert/strict";
import test from "node:test";

import {
  findClosestNextRallyIndex,
  findServeMarkerIndexForRally,
  mergeSelectedRallies,
} from "../../lib/rally-label-editing.ts";
import type { RallyLabel, ServeMarker } from "../../lib/annotations.ts";

const rallies: RallyLabel[] = [
  { start: 10, end: 20, tags: ["first"], notes: "First note", startConfidence: 0.9 },
  { start: 25, end: 30, tags: ["second"] },
  {
    start: 35,
    end: 42,
    tags: ["third"],
    notes: "Last note",
    terminalCue: "ball-down-or-out",
    endConfidence: 0.8,
  },
  { start: 50, end: 55, tags: [] },
];

test("merges consecutive rallies and keeps only their earliest serve marker", () => {
  const serveMarkers: ServeMarker[] = [
    { time: 10.1, side: "near" },
    { time: 25.1, side: "far" },
    { time: 35.1, side: "near" },
    { time: 50.1, side: "far" },
  ];
  const result = mergeSelectedRallies({
    rallies,
    serveMarkers,
    hardNegatives: [],
    selectedIndexes: [2, 0, 1],
  });

  assert.equal(result.ok, true);
  if (!result.ok) return;
  assert.deepEqual(
    result.rallies.map(({ start, end }) => ({ start, end })),
    [
      { start: 10, end: 42 },
      { start: 50, end: 55 },
    ],
  );
  assert.deepEqual(result.mergedRally.tags, ["first", "second", "third"]);
  assert.equal(result.mergedRally.notes, "First note\nLast note");
  assert.equal(result.mergedRally.startConfidence, 0.9);
  assert.equal(result.mergedRally.endConfidence, 0.8);
  assert.equal(result.mergedRally.terminalCue, "ball-down-or-out");
  assert.deepEqual(result.serveMarkers, [serveMarkers[0], serveMarkers[3]]);
  assert.equal(result.removedServeMarkerCount, 2);
  assert.equal(result.movedServeMarkerToStart, false);
});

test("moves the retained serve marker to the merged start when the first rally has none", () => {
  const result = mergeSelectedRallies({
    rallies,
    serveMarkers: [
      { time: 25.1, side: "far", origin: "model" },
      { time: 50.1, side: "near" },
    ],
    hardNegatives: [],
    selectedIndexes: [0, 1],
  });

  assert.equal(result.ok, true);
  if (!result.ok) return;
  assert.deepEqual(result.serveMarkers, [
    { time: 10, side: "far", origin: "model" },
    { time: 50.1, side: "near" },
  ]);
  assert.equal(result.movedServeMarkerToStart, true);
});

test("does not remove a neighboring rally's closer serve marker", () => {
  const neighboringMarker = { time: 48.5, side: "far" as const };
  const result = mergeSelectedRallies({
    rallies,
    serveMarkers: [
      { time: 10, side: "near" },
      { time: 25, side: "far" },
      neighboringMarker,
    ],
    hardNegatives: [],
    selectedIndexes: [0, 1],
  });

  assert.equal(result.ok, true);
  if (!result.ok) return;
  assert.deepEqual(result.serveMarkers, [{ time: 10, side: "near" }, neighboringMarker]);
});

test("rejects nonconsecutive selections and hard negatives in the merged gap", () => {
  const nonconsecutive = mergeSelectedRallies({
    rallies,
    serveMarkers: [],
    hardNegatives: [],
    selectedIndexes: [0, 2],
  });
  assert.deepEqual(nonconsecutive, {
    ok: false,
    error: "Only consecutive rallies can be merged.",
  });

  const blockedGap = mergeSelectedRallies({
    rallies,
    serveMarkers: [],
    hardNegatives: [{ start: 21, end: 22, category: "setup-between-points" }],
    selectedIndexes: [0, 1],
  });
  assert.equal(blockedGap.ok, false);
  if (!blockedGap.ok) assert.match(blockedGap.error, /hard negative/);
});

test("finds the first rally start strictly after the playhead", () => {
  assert.equal(findClosestNextRallyIndex(rallies, 20), 1);
  assert.equal(findClosestNextRallyIndex(rallies, 25), 2);
  assert.equal(findClosestNextRallyIndex(rallies, 60), -1);
  assert.equal(findClosestNextRallyIndex([rallies[2], rallies[1]], 20), 1);
});

test("finds the serve marker associated with a rally start", () => {
  const markers: ServeMarker[] = [
    { time: 10.4, side: "near" },
    { time: 24.2, side: "far" },
    { time: 25.1, side: "review" },
  ];
  assert.equal(findServeMarkerIndexForRally(rallies, markers, 0), 0);
  assert.equal(findServeMarkerIndexForRally(rallies, markers, 1), 2);
  assert.equal(findServeMarkerIndexForRally(rallies, markers, 3), -1);
});
