import assert from "node:assert/strict";
import test from "node:test";

import {
  applyPaddingToCachedCuts,
  buildFinalCutIntervals,
  createCutDraft,
} from "../../prod/src/lib/cut-draft.ts";

test("production editor padding is clipped to the marked game window", () => {
  const draft = createCutDraft({
    analysisId: "trimmed-analysis",
    recordingId: "recording",
    duration: 120,
    analysisStart: 20,
    analysisEnd: 100,
    rallies: [
      { id: "R001", start: 20.5, end: 99.5, confidence: 0.9, included: true },
    ],
    ignoredIntervals: [
      { start: 0, end: 20, reason: "outside-game-window" },
      { start: 100, end: 120, reason: "outside-game-window" },
    ],
  });

  assert.equal(draft.cuts[0]?.keepStart, 20);
  assert.equal(draft.cuts[0]?.keepEnd, 100);

  const repadded = applyPaddingToCachedCuts(draft, 8, 8, 120, 20, 100);
  assert.equal(repadded.cuts[0]?.keepStart, 20);
  assert.equal(repadded.cuts[0]?.keepEnd, 100);
  assert.deepEqual(buildFinalCutIntervals(repadded), [
    { start: 20, end: 100, cutIds: ["R001"] },
  ]);
});

test("project creation can defer score-tracking analysis", () => {
  const seed = {
    analysisId: "score-tracking-option",
    recordingId: "recording",
    duration: 120,
    rallies: [],
    ignoredIntervals: [],
  };

  assert.equal(createCutDraft(seed).scoreTracking.enabled, true);
  assert.equal(
    createCutDraft({ ...seed, scoreTrackingEnabled: false }).scoreTracking
      .enabled,
    false,
  );
});
