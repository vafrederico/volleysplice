import assert from "node:assert/strict";
import test from "node:test";
import { createModelReviewDraft } from "../../prod/src/lib/model-review-draft.ts";
import { createCutDraft, parseCutDraft, rallySuppressionDecisionKey } from "../../prod/src/lib/cut-draft.ts";
import { recordEdit, moveHistory, readEditHistory } from "../../prod/src/designs/taste/edit-history.ts";

const seed = { analysisId: "model-reset", recordingId: "project-reset", duration: 60, analysisStart: 5, analysisEnd: 55,
  rallies: [{ id: "R001", start: 10, end: 20, included: true, confidence: 0.9 }],
  ignoredIntervals: [{ start: 0, end: 5, reason: "outside-game-window" }, { start: 55, end: 60, reason: "outside-game-window" }],
};
const analysis = {
  servingSide: { candidates: [{ id: "R001", anchor: 10, verdict: "near", side: "near" }] },
  sideSwitch: { candidates: [{ id: "side1", timestamp: 30, probability: 0.95, sourceRangeIds: ["R001"] }] },
} as Parameters<typeof createModelReviewDraft>[1];

test("reset reconstructs the original rallies and model markers without edited drafts", () => {
  const baseline = createModelReviewDraft(seed, analysis);
  assert.equal(baseline.cuts[0].coreStart, 10);
  assert.equal(baseline.cuts[0].included, true);
  assert.equal(baseline.scoreTracking.serveMarkers[0].side, "near");
  assert.equal(baseline.scoreTracking.serveMarkers[0].timestamp, 10);
  assert.equal(baseline.scoreTracking.sideSwitchMarkers[0].timestamp, 30);
  assert.equal(baseline.scoreTracking.serveMarkers[0].origin, "model");
  assert.equal(baseline.scoreTracking.sideSwitchMarkers[0].origin, "model");
  assert.deepEqual(baseline.ignoredIntervals.map(({ start, end }) => [start, end]), [[0, 5], [55, 60]]);
  assert.deepEqual(baseline.suppressionDecisionOverrides, {});
  assert.deepEqual(baseline.userTouchedCutIds, []);
  assert.deepEqual(baseline.reviewedCutIds, []);
  assert.deepEqual(baseline.scoreTracking.removedModelMarkerIds, []);
  assert.ok(parseCutDraft(JSON.stringify(baseline), seed));
});

test("reset is one persisted, reversible action even after manual edits and marker removals", () => {
  const baseline = createModelReviewDraft(seed, analysis);
  const edited = structuredClone(baseline);
  edited.cuts[0].included = false;
  edited.reviewedCutIds = ["R001"];
  edited.pendingIgnoreStart = 35;
  edited.beforePaddingSeconds = 1;
  edited.suppressionDecisionOverrides[rallySuppressionDecisionKey("R001")] = "suppress";
  edited.scoreTracking.team1Name = "Custom team";
  edited.scoreTracking.serveMarkers = [];
  edited.scoreTracking.removedModelMarkerIds = ["serve-R001"];
  const history = recordEdit(readEditHistory(null, edited, seed), createModelReviewDraft(seed, analysis));
  assert.equal(history.past.length, 1);
  const reloaded = readEditHistory(JSON.stringify({ version: 1, ...history }), history.present, seed);
  assert.deepEqual(moveHistory(reloaded, "undo").present, JSON.parse(JSON.stringify(edited)));
  assert.deepEqual(moveHistory(moveHistory(reloaded, "undo"), "redo").present, baseline);
});

test("projects without specialist results reset without inventing serve or switch markers", () => {
  const baseline = createModelReviewDraft(seed, {});
  assert.deepEqual(baseline.scoreTracking.serveMarkers, []);
  assert.deepEqual(baseline.scoreTracking.sideSwitchMarkers, []);
  assert.deepEqual(baseline.cuts, createCutDraft(seed).cuts);
});
