import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { loadBindings, transform } from "next/dist/build/swc/index.js";
import { createCutDraft, splitCutAt, splitCutValidationMessage } from "../../components/production-lab/editor/lib/cut-draft.ts";
import { splitLabServeMarkers } from "../../components/production-lab/editor/lib/lab-split-markers.ts";
import { createScoreTracking, isValidScoreTracking, removeServeMarker, scoreTrackingOutsideExcludedRallies } from "../../components/production-lab/editor/lib/score-tracking.ts";

async function loadSplitEvent() {
  const source = await readFile(new URL("../../components/production-lab/editor/designs/taste/index.tsx", import.meta.url), "utf8");
  const from = source.indexOf("  function splitSelectedAtPlayhead() {");
  const to = source.indexOf("  function markManualBoundary()", from);
  assert.ok(from >= 0 && to > from);
  const body = `export function splitEvent(context: any) {
    const formatTime = (value: number) => String(value);
    const { selected, playhead, clips, suppressionDecisions, workingDraft, splitLabServeMarkers, splitCutAt, splitCutValidationMessage, gameStart, gameEnd,
      setReviewMessage, setClips, setScoreMarkers, setRemovedModelMarkerIds, setSuppressionDecisions, setSelectedId } = context;
    ${source.slice(from, to)}
    splitSelectedAtPlayhead();
  }`;
  await loadBindings();
  const compiled = await transform(body, { filename: "native-split-event.ts", jsc: { parser: { syntax: "typescript" }, target: "es2022" }, module: { type: "commonjs" } });
  const module = { exports: {} as { splitEvent: (context: unknown) => void } };
  new Function("module", "exports", compiled.code)(module, module.exports);
  return module.exports.splitEvent;
}

test("native rally split skips ignored IDs, keeps the corrected first serve and requests a new serving-side decision", async () => {
  const scoreTracking = createScoreTracking(true);
  scoreTracking.serveMarkers = [
    { id: "lab-serve:R1", rallyId: "R001", timestamp: 10, side: "far", modelSide: "near", origin: "model", ignorePreviousPoint: true },
    { id: "manual-serve", timestamp: 25, side: "near", origin: "manual", ignorePreviousPoint: false },
  ];
  const original = structuredClone(scoreTracking);
  const splitEvent = await loadSplitEvent();
  const originalClip = { id: "R001", start: 10, end: 20, included: true, origin: "model", label: "Detected rally", confidence: .9, reviewed: false };
  let clips = [originalClip];
  let markers = scoreTracking.serveMarkers;
  let removed = scoreTracking.removedModelMarkerIds;
  let selected = "R001";
  splitEvent({ selected: originalClip, playhead: 15, clips, suppressionDecisions: {}, workingDraft: { ...createCutDraft({ analysisId: "fixture", recordingId: "fixture", duration: 100,
      rallies: [{ id: "R001", start: 10, end: 20, confidence: .9, included: true }], ignoredIntervals: [] }),
      ignoredIntervals: [{ id: "R002", start: 1, end: 2, reason: "other" }], scoreTracking },
    splitLabServeMarkers, splitCutAt, splitCutValidationMessage, gameStart: 0, gameEnd: 100,
    setClips: (update: (value: typeof clips) => typeof clips) => { clips = update(clips); },
    setScoreMarkers: (value: typeof markers) => { markers = value; },
    setRemovedModelMarkerIds: (value: string[]) => { removed = value; },
    setSuppressionDecisions: () => {}, setSelectedId: (value: string) => { selected = value; }, setReviewMessage: () => {},
  });
  assert.deepEqual(clips.map(clip => [clip.id, clip.start, clip.end]), [["R001", 10, 15], ["R003", 15, 20]]);
  assert.equal(selected, "R003");
  assert.deepEqual(markers.find(marker => marker.id === "lab-serve:R1"), { ...original.serveMarkers[0], rallyId: "R001" });
  assert.deepEqual(markers.find(marker => marker.rallyId === "R003"), { id: "lab-split-serve:R003", rallyId: "R003", timestamp: 15,
    side: "review", modelSide: "review", origin: "model", ignorePreviousPoint: false });
  assert.equal(markers.filter(marker => marker.rallyId === "R001").length, 1);
  assert.deepEqual(scoreTracking, original, "the immutable model score input must stay unchanged");
  const updated = { ...scoreTracking, serveMarkers: markers, removedModelMarkerIds: removed };
  assert.equal(isValidScoreTracking(updated, 100), true);
  assert.ok(!scoreTrackingOutsideExcludedRallies(updated, new Set(["R001"])).serveMarkers.some(marker => marker.id === "lab-serve:R1"), "removing the left child must hide its preserved serve");
});

test("lab split action explains wire limits before changing clips, scores, selection or cleanup", async () => {
  const splitEvent = await loadSplitEvent();
  const base = createCutDraft({ analysisId: "synthetic-limits", recordingId: "synthetic-limits", duration: 40,
    rallies: [{ id: "R001", start: 10, end: 20, included: true, confidence: .9 }], ignoredIntervals: [] });
  for (const [count, id, message] of [[2048, "R001", /2,048/], [1, "L".repeat(257), /256/]] as const) {
    const workingDraft = { ...base, cuts: Array.from({ length: count }, (_, index) => ({
      ...base.cuts[0], id: index ? `S${index}` : id,
      ...(count > 1 ? { editableGroupId: "split:siblings" } : {}),
    })) };
    const before = JSON.stringify(workingDraft);
    const mutations: unknown[] = [];
    const recordMutation = (...args: unknown[]) => { mutations.push(args); };
    let shown = "";
    splitEvent({ selected: { id, start: 10, end: 20 }, playhead: 15, workingDraft,
      splitCutValidationMessage, splitCutAt, splitLabServeMarkers, gameStart: 0, gameEnd: 40,
      suppressionDecisions: {}, clips: [], setReviewMessage: (value: string) => { shown = value; },
      setClips: recordMutation, setScoreMarkers: recordMutation, setRemovedModelMarkerIds: recordMutation,
      setSuppressionDecisions: recordMutation, setSelectedId: recordMutation,
    });
    assert.match(shown, message);
    assert.deepEqual(mutations, []);
    assert.equal(JSON.stringify(workingDraft), before);
  }
});

test("split review does not resurrect rejected or deleted first serves, and honors right-marker tombstones", () => {
  const noFirstServe = createScoreTracking(true);
  noFirstServe.removedModelMarkerIds = ["lab-serve:R1"];
  const once = splitLabServeMarkers(noFirstServe, "R1", { id: "R1A", start: 10 }, { id: "R1B", start: 15 });
  assert.deepEqual(once.serveMarkers.map(marker => [marker.rallyId, marker.side]), [["R1B", "review"]]);
  assert.deepEqual(once.removedModelMarkerIds, ["lab-serve:R1"]);
  const deleted = removeServeMarker(once, "lab-split-serve:R1B");
  const repeated = splitLabServeMarkers(deleted, "R1", { id: "R1A", start: 10 }, { id: "R1B", start: 15 });
  assert.deepEqual(repeated.serveMarkers, []);
  assert.equal(isValidScoreTracking(repeated, 100), true);
});
