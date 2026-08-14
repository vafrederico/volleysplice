import assert from "node:assert/strict";
import test from "node:test";

import {
  applyPaddingToCachedCuts,
  buildFinalCutIntervals,
  createCutDraft,
  cutSourceRevision,
  nextFinalCutTime,
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
  assert.equal(draft.beforePaddingSeconds, 2);
  assert.equal(draft.afterPaddingSeconds, 2);
  assert.equal(draft.playbackRate, 1);
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
      { coreStart: 20, coreEnd: 30, keepStart: 18, keepEnd: 32, origin: "cached-label" },
    ],
  );
});

test("global before and after padding update cached cuts and leave manual cuts alone", () => {
  const draft = createCutDraft(seed);
  draft.cuts.push({
    id: "M001",
    coreStart: 40,
    coreEnd: 45,
    keepStart: 40,
    keepEnd: 45,
    confidence: 1,
    included: true,
    origin: "manual",
  });

  const updated = applyPaddingToCachedCuts(draft, 4.5, 1.5, seed.duration);

  assert.equal(updated.beforePaddingSeconds, 4.5);
  assert.equal(updated.afterPaddingSeconds, 1.5);
  assert.deepEqual(
    updated.cuts.map(({ keepStart, keepEnd }) => ({ keepStart, keepEnd })),
    [
      { keepStart: 0, keepEnd: 5.5 },
      { keepStart: 15.5, keepEnd: 31.5 },
      { keepStart: 40, keepEnd: 45 },
    ],
  );
});

test("parseCutDraft restores matching drafts and rejects stale source labels", () => {
  const draft = createCutDraft(seed);
  draft.updatedAt = new Date().toISOString();
  draft.pendingManualStart = 12.5;
  draft.ignoreReason = "camera-gap";
  draft.cutPreviewEnabled = true;
  draft.playbackRate = 4;

  assert.deepEqual(parseCutDraft(JSON.stringify(draft), seed), draft);
  assert.equal(
    parseCutDraft(
      JSON.stringify(draft),
      { ...seed, rallies: [{ ...seed.rallies[0], end: 5 }] },
    ),
    null,
  );
});

test("parseCutDraft migrates the original on-device draft without losing edits", () => {
  const draft = createCutDraft(seed);
  draft.cuts[0].included = false;
  const legacy = { ...draft } as Record<string, unknown>;
  legacy.version = 1;
  delete legacy.beforePaddingSeconds;
  delete legacy.afterPaddingSeconds;
  delete legacy.pendingManualStart;
  delete legacy.pendingIgnoreStart;
  delete legacy.ignoreReason;
  delete legacy.cutPreviewEnabled;
  delete legacy.playbackRate;

  const migrated = parseCutDraft(JSON.stringify(legacy), seed);

  assert.equal(migrated?.cuts[0].included, false);
  assert.equal(migrated?.beforePaddingSeconds, 3);
  assert.equal(migrated?.afterPaddingSeconds, 2);
  assert.equal(migrated?.playbackRate, 1);
});

test("playback-rate migration preserves the prior session state", () => {
  const prior = createCutDraft(seed) as unknown as Record<string, unknown>;
  prior.version = 4;
  prior.pendingManualStart = 12.5;
  prior.ignoreReason = "camera-gap";
  prior.cutPreviewEnabled = true;
  delete prior.playbackRate;

  const migrated = parseCutDraft(JSON.stringify(prior), seed);

  assert.equal(migrated?.pendingManualStart, 12.5);
  assert.equal(migrated?.ignoreReason, "camera-gap");
  assert.equal(migrated?.cutPreviewEnabled, true);
  assert.equal(migrated?.playbackRate, 1);
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

test("cut preview keeps playable time and jumps gaps to the next interval", () => {
  const intervals = [
    { start: 2, end: 5, cutIds: ["A"] },
    { start: 8, end: 12, cutIds: ["B"] },
  ];

  assert.equal(nextFinalCutTime(intervals, 0), 2);
  assert.equal(nextFinalCutTime(intervals, 3), 3);
  assert.equal(nextFinalCutTime(intervals, 5), 8);
  assert.equal(nextFinalCutTime(intervals, 20), null);
});
