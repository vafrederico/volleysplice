import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { loadBindings, transform } from "next/dist/build/swc/index.js";
import { splitLabServeMarkers } from "../../components/production-lab/editor/lib/lab-split-markers.ts";
import { createScoreTracking, isValidScoreTracking, removeServeMarker, scoreTrackingOutsideExcludedRallies } from "../../components/production-lab/editor/lib/score-tracking.ts";

test("native rally split keeps the corrected first serve and requests a new serving-side decision", async () => {
  const scoreTracking = createScoreTracking(true);
  scoreTracking.serveMarkers = [
    { id: "lab-serve:R1", rallyId: "R1", timestamp: 10, side: "far", modelSide: "near", origin: "model", ignorePreviousPoint: true },
    { id: "manual-serve", timestamp: 25, side: "near", origin: "manual", ignorePreviousPoint: false },
  ];
  const original = structuredClone(scoreTracking);
  const source = await readFile(new URL("../../components/production-lab/editor/designs/taste/index.tsx", import.meta.url), "utf8");
  const from = source.indexOf("  function splitSelectedAtPlayhead() {");
  const to = source.indexOf("  function markManualBoundary()", from);
  assert.ok(from >= 0 && to > from);
  const body = `export function splitEvent(context: any) {
    const formatTime = (value: number) => String(value);
    const { selected, playhead, clips, suppressionDecisions, workingDraft, splitLabServeMarkers,
      setReviewMessage, setClips, setScoreMarkers, setRemovedModelMarkerIds, setSuppressionDecisions, setSelectedId } = context;
    ${source.slice(from, to)}
    splitSelectedAtPlayhead();
  }`;
  await loadBindings();
  const compiled = await transform(body, { filename: "native-split-event.ts", jsc: { parser: { syntax: "typescript" }, target: "es2022" }, module: { type: "commonjs" } });
  const module = { exports: {} as { splitEvent: (context: unknown) => void } };
  new Function("module", "exports", compiled.code)(module, module.exports);
  const originalClip = { id: "R1", start: 10, end: 20, included: true, origin: "model", label: "Detected rally", confidence: .9, reviewed: false };
  let clips = [originalClip];
  let markers = scoreTracking.serveMarkers;
  let removed = scoreTracking.removedModelMarkerIds;
  let selected = "R1";
  module.exports.splitEvent({ selected: originalClip, playhead: 15, clips, suppressionDecisions: {}, workingDraft: { scoreTracking }, splitLabServeMarkers,
    setClips: (update: (value: typeof clips) => typeof clips) => { clips = update(clips); },
    setScoreMarkers: (value: typeof markers) => { markers = value; },
    setRemovedModelMarkerIds: (value: string[]) => { removed = value; },
    setSuppressionDecisions: () => {}, setSelectedId: (value: string) => { selected = value; }, setReviewMessage: () => {},
  });
  assert.deepEqual(clips.map(clip => [clip.id, clip.start, clip.end]), [["R1A", 10, 15], ["R1B", 15, 20]]);
  assert.equal(selected, "R1B");
  assert.deepEqual(markers.find(marker => marker.id === "lab-serve:R1"), { ...original.serveMarkers[0], rallyId: "R1A" });
  assert.deepEqual(markers.find(marker => marker.rallyId === "R1B"), { id: "lab-split-serve:R1B", rallyId: "R1B", timestamp: 15,
    side: "review", modelSide: "review", origin: "model", ignorePreviousPoint: false });
  assert.ok(!markers.some(marker => marker.rallyId === "R1"));
  assert.deepEqual(scoreTracking, original, "the immutable model score input must stay unchanged");
  const updated = { ...scoreTracking, serveMarkers: markers, removedModelMarkerIds: removed };
  assert.equal(isValidScoreTracking(updated, 100), true);
  assert.ok(!scoreTrackingOutsideExcludedRallies(updated, new Set(["R1A"])).serveMarkers.some(marker => marker.id === "lab-serve:R1"), "removing the left child must hide its preserved serve");
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
