import assert from "node:assert/strict";
import test from "node:test";

import {
  createCutDraft,
  parseCutDraft,
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
