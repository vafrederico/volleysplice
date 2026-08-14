import assert from "node:assert/strict";
import test from "node:test";

import {
  buildFinalCutIntervals,
  createCutDraft,
  cutSourceRevision,
  parseCutDraft,
  totalFinalCutSeconds,
  type CutDraftSeed,
} from "../../lib/cut-draft.ts";

const seed: CutDraftSeed = {
  analysisId: "model--recording",
  recordingId: "recording",
  duration: 60,
  rallies: [
    { id: "R1", start: 1, end: 4, confidence: 0.8, included: true },
    { id: "R2", start: 20, end: 30, confidence: 0.7, included: true },
  ],
  ignoredIntervals: [{ start: 22, end: 24, reason: "camera-gap" }],
};

test("createCutDraft seeds clamped per-cut padding from cached labels", () => {
  const draft = createCutDraft(seed);

  assert.equal(draft.sourceRevision, cutSourceRevision(seed));
  assert.deepEqual(
    draft.cuts.map(({ coreStart, coreEnd, keepStart, keepEnd, origin }) => ({
      coreStart,
      coreEnd,
      keepStart,
      keepEnd,
      origin,
    })),
    [
      { coreStart: 1, coreEnd: 4, keepStart: 0, keepEnd: 6, origin: "cached-label" },
      { coreStart: 20, coreEnd: 30, keepStart: 17, keepEnd: 32, origin: "cached-label" },
    ],
  );
});

test("parseCutDraft restores matching drafts and rejects stale source labels", () => {
  const draft = createCutDraft(seed);
  draft.updatedAt = new Date().toISOString();

  assert.deepEqual(parseCutDraft(JSON.stringify(draft), seed), draft);
  assert.equal(
    parseCutDraft(
      JSON.stringify(draft),
      { ...seed, rallies: [{ ...seed.rallies[0], end: 5 }] },
    ),
    null,
  );
});

test("final edit list merges touching cuts and subtracts ignored source time", () => {
  const draft = createCutDraft(seed);
  draft.cuts = [
    { ...draft.cuts[0], id: "A", keepStart: 0, keepEnd: 10 },
    {
      ...draft.cuts[1],
      id: "B",
      coreStart: 10,
      coreEnd: 15,
      keepStart: 10,
      keepEnd: 18,
    },
    {
      ...draft.cuts[1],
      id: "C",
      coreStart: 40,
      coreEnd: 45,
      keepStart: 38,
      keepEnd: 47,
      included: false,
    },
  ];
  draft.ignoredIntervals = [
    { id: "I1", start: 4, end: 6, reason: "camera-gap" },
    { id: "I2", start: 14, end: 20, reason: "non-game-content" },
  ];

  const intervals = buildFinalCutIntervals(draft);
  assert.deepEqual(intervals, [
    { start: 0, end: 4, cutIds: ["A", "B"] },
    { start: 6, end: 14, cutIds: ["A", "B"] },
  ]);
  assert.equal(totalFinalCutSeconds(intervals), 12);
});
