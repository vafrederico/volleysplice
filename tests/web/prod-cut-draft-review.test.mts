import assert from "node:assert/strict";
import test from "node:test";

import {
  applyPaddingToCachedCuts,
  createCutDraft,
  parseCutDraft,
  setCutCoreEnd,
  setCutCoreRange,
  setCutCoreStart,
  splitCutAt,
  type CutDraftSeed,
} from "../../prod/src/lib/cut-draft.ts";

const seed: CutDraftSeed = {
  analysisId: "review-analysis",
  recordingId: "review-recording",
  duration: 60,
  rallies: [
    { id: "R001", start: 5, end: 10, confidence: 0.4, included: true },
    { id: "R002", start: 20, end: 25, confidence: 0.9, included: true },
  ],
  ignoredIntervals: [],
};

test("production drafts persist reviewed model ranges", () => {
  const draft = createCutDraft(seed);
  draft.reviewedCutIds = ["R001"];

  const restored = parseCutDraft(JSON.stringify(draft), seed);

  assert.deepEqual(restored?.reviewedCutIds, ["R001"]);
  assert.deepEqual(restored?.cuts, JSON.parse(JSON.stringify(draft.cuts)));
});

test("version seven production drafts migrate to an empty review history", () => {
  const legacy = createCutDraft(seed) as unknown as Record<string, unknown>;
  legacy.version = 7;
  delete legacy.reviewedCutIds;

  assert.deepEqual(parseCutDraft(JSON.stringify(legacy), seed)?.reviewedCutIds, []);
});

test("production drafts reject review history for unknown ranges", () => {
  const draft = createCutDraft(seed);
  draft.reviewedCutIds = ["missing"];

  assert.equal(parseCutDraft(JSON.stringify(draft), seed), null);
});

test("production rally edge edits preserve padding and accept later global padding", () => {
  const draft = createCutDraft(seed);

  const ranged = setCutCoreRange(draft, "R001", 6, 9);
  assert.deepEqual(
    (({ coreStart, coreEnd, keepStart, keepEnd }) => ({
      coreStart,
      coreEnd,
      keepStart,
      keepEnd,
    }))(ranged.cuts[0]),
    { coreStart: 6, coreEnd: 9, keepStart: 4, keepEnd: 11 },
  );
  assert.deepEqual(ranged.userTouchedCutIds, ["R001"]);

  const repadded = applyPaddingToCachedCuts(ranged, 3, 1, seed.duration);
  assert.equal(repadded.cuts[0].keepStart, 3);
  assert.equal(repadded.cuts[0].keepEnd, 10);

  const movedStart = setCutCoreStart(repadded, "R001", 7);
  const movedEnd = setCutCoreEnd(movedStart, "R001", 8);
  assert.equal(movedEnd.cuts[0].coreStart, 7);
  assert.equal(movedEnd.cuts[0].coreEnd, 8);
  assert.equal(movedEnd.cuts[0].keepStart, 4);
  assert.equal(movedEnd.cuts[0].keepEnd, 9);
});

test("production rally split creates two independently editable padded ranges", () => {
  const draft = createCutDraft(seed);
  const split = splitCutAt(draft, "R002", 22);

  assert.ok(split);
  assert.deepEqual(split.draft.cuts.map((cut) => cut.id), ["R001", "R002", "R003"]);
  assert.deepEqual(
    split.draft.cuts.slice(1).map(({ coreStart, coreEnd, keepStart, keepEnd }) => ({
      coreStart,
      coreEnd,
      keepStart,
      keepEnd,
    })),
    [
      { coreStart: 20, coreEnd: 22, keepStart: 18, keepEnd: 24 },
      { coreStart: 22, coreEnd: 25, keepStart: 20, keepEnd: 27 },
    ],
  );
  assert.deepEqual(split.draft.userTouchedCutIds, ["R002", "R003"]);
  const restored = parseCutDraft(JSON.stringify(split.draft), seed);
  assert.ok(restored);
  assert.deepEqual(restored.cuts.map((cut) => cut.id), ["R001", "R002", "R003"]);
  assert.deepEqual(restored.userTouchedCutIds, ["R002", "R003"]);

  const shortenedLeft = setCutCoreEnd(split.draft, "R002", 21);
  const shortenedRight = setCutCoreStart(shortenedLeft, "R003", 23);
  assert.equal(shortenedRight.cuts[1].coreEnd, 21);
  assert.equal(shortenedRight.cuts[2].coreStart, 23);
  assert.equal(splitCutAt(draft, "R002", 20.05), null);
  assert.equal(splitCutAt(draft, "R002", 24.95), null);
});
