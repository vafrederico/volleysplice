import assert from "node:assert/strict";
import test from "node:test";
import { alignRallyServeMarkers, createCutDraft, parseCutDraft, setCutCoreStart, setCutCoreEnd, setCutCoreRange, applyPaddingToCachedCuts } from "../../prod/src/lib/cut-draft.ts";
import { addServeMarker, addSideSwitchMarker, alignServeMarkersToRallyStarts } from "../../prod/src/lib/score-tracking.ts";
import { scoreTrackingWithServingSideOutput } from "../../prod/src/lib/score-tracking-inference.ts";

const seed = {
  analysisId: "marker-test", recordingId: "recording", duration: 60,
  rallies: [{ id: "R001", start: 5, end: 15, confidence: 0.9, included: true }],
  ignoredIntervals: [],
};
function fixture() {
  const draft = createCutDraft(seed);
  draft.scoreTracking = addServeMarker(draft.scoreTracking, 5, "far", {
    id: "serve-R001", origin: "model", rallyId: "R001", modelSide: "near",
  });
  draft.scoreTracking.serveMarkers[0].ignorePreviousPoint = true;
  draft.scoreTracking = addServeMarker(draft.scoreTracking, 6, "near", { id: "manual" });
  draft.scoreTracking = addSideSwitchMarker(draft.scoreTracking, 7, "switch");
  return draft;
}

test("rally start moves linked serves in either direction and preserves independent markers", () => {
  const original = fixture();
  for (const start of [3, 8]) {
    const changed = setCutCoreStart(original, "R001", start);
    const linked = changed.scoreTracking.serveMarkers.find((marker) => marker.rallyId);
    assert.deepEqual(linked, { ...original.scoreTracking.serveMarkers[0], timestamp: start });
    assert.equal(changed.scoreTracking.serveMarkers.find((marker) => marker.id === "manual")?.timestamp, 6);
    assert.deepEqual(changed.scoreTracking.sideSwitchMarkers, original.scoreTracking.sideSwitchMarkers);
    assert.deepEqual(changed.scoreTracking.serveMarkers.map((marker) => marker.timestamp), start === 3 ? [3, 6] : [6, 8]);
    assert.equal(parseCutDraft(JSON.stringify(changed), seed)?.scoreTracking.serveMarkers.find((marker) => marker.rallyId)?.timestamp, start);
  }
});

test("range edits move serves, while end and padding edits keep their core timestamp", () => {
  const draft = fixture();
  assert.equal(setCutCoreRange(draft, "R001", 4, 14).scoreTracking.serveMarkers[0].timestamp, 4);
  assert.deepEqual(setCutCoreEnd(draft, "R001", 12, 60).scoreTracking, draft.scoreTracking);
  assert.deepEqual(applyPaddingToCachedCuts(draft, 3, 3, 60).scoreTracking, draft.scoreTracking);
});

test("older saved drafts repair linked timestamps, including disabled score tracking", () => {
  const draft = fixture();
  draft.cuts[0].coreStart = 8;
  draft.cuts[0].keepStart = 6;
  draft.scoreTracking.enabled = false;
  const restored = parseCutDraft(JSON.stringify(draft), seed)!;
  assert.equal(restored.scoreTracking.serveMarkers.find((marker) => marker.rallyId)?.timestamp, 8);
  assert.equal(restored.scoreTracking.enabled, false);
  assert.equal(alignRallyServeMarkers(restored), restored);
});

test("production desk uses the corrected clip starts and leaves unknown links alone", () => {
  const markers = fixture().scoreTracking.serveMarkers;
  const aligned = alignServeMarkersToRallyStarts(markers, [{ id: "R001", start: 9 }]);
  assert.equal(aligned.find((marker) => marker.rallyId)?.timestamp, 9);
  assert.equal(alignServeMarkersToRallyStarts(markers, [{ id: "other", start: 9 }]), markers);
});

test("model refresh retains corrected timestamps and deleted model markers", () => {
  const draft = setCutCoreStart(fixture(), "R001", 8);
  const output = { candidates: [{ id: "R001", anchor: 5, verdict: "near", side: "near" }] };
  const refreshed = scoreTrackingWithServingSideOutput(draft.scoreTracking, output as Parameters<typeof scoreTrackingWithServingSideOutput>[1]);
  assert.equal(refreshed.serveMarkers.find((marker) => marker.rallyId)?.timestamp, 8);
  assert.equal(refreshed.serveMarkers.find((marker) => marker.rallyId)?.side, "far");
  const removed = { ...draft.scoreTracking, serveMarkers: [], removedModelMarkerIds: ["serve-R001"] };
  assert.deepEqual(scoreTrackingWithServingSideOutput(removed, output as Parameters<typeof scoreTrackingWithServingSideOutput>[1]).serveMarkers, []);
});

test("markers follow clamped core starts and refresh without a side correction", () => {
  const draft = fixture();
  draft.scoreTracking.serveMarkers[0].side = "near";
  const changed = setCutCoreStart(draft, "R001", 100);
  const coreStart = changed.cuts[0].coreStart;
  assert.equal(coreStart, 14.9);
  assert.equal(changed.scoreTracking.serveMarkers.find((marker) => marker.rallyId)?.timestamp, coreStart);
  const output = { candidates: [{ id: "R001", anchor: 5, verdict: "near", side: "near" }] };
  const refreshed = scoreTrackingWithServingSideOutput(
    changed.scoreTracking,
    output as Parameters<typeof scoreTrackingWithServingSideOutput>[1],
  );
  assert.equal(refreshed.serveMarkers.find((marker) => marker.rallyId)?.timestamp, coreStart);
});
