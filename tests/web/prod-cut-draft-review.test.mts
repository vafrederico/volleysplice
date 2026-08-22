import assert from "node:assert/strict";
import test from "node:test";

import {
  applyPaddingToCachedCuts,
  type CutDraftSeed,
  createCutDraft,
  parseCutDraft,
  setCutCoreEnd,
  setCutCoreRange,
  setCutCoreStart,
  splitCutAt,
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

test("version ten production drafts migrate to enabled score tracking", () => {
  const legacy = createCutDraft(seed) as unknown as Record<string, unknown>;
  legacy.version = 10;
  delete legacy.scoreTracking;

  assert.deepEqual(parseCutDraft(JSON.stringify(legacy), seed)?.scoreTracking, {
    version: 2,
    enabled: true,
    team1Name: "Team 1",
    team2Name: "Team 2",
    serveMarkers: [],
    sideSwitchMarkers: [],
    removedModelMarkerIds: [],
  });
});

test("version eleven score tracking migrates model-removal tombstones", () => {
  const legacy = createCutDraft(seed) as unknown as Record<string, unknown>;
  legacy.version = 11;
  const scoreTracking = legacy.scoreTracking as Record<string, unknown>;
  scoreTracking.version = 1;
  delete scoreTracking.removedModelMarkerIds;

  assert.deepEqual(
    parseCutDraft(JSON.stringify(legacy), seed)?.scoreTracking.removedModelMarkerIds,
    [],
  );
});

test("production drafts persist valid score tracking and reject invalid markers", () => {
  const draft = createCutDraft(seed);
  draft.scoreTracking = {
    ...draft.scoreTracking,
    enabled: true,
    team1Name: "Falcons",
    team2Name: "Tigers",
    serveMarkers: [
      {
        id: "serve-R001",
        timestamp: 5,
        side: "near",
        origin: "model",
        modelSide: "near",
        ignorePreviousPoint: false,
        rallyId: "R001",
      },
    ],
    sideSwitchMarkers: [{ id: "switch-1", timestamp: 15 }],
  };

  assert.deepEqual(
    parseCutDraft(JSON.stringify(draft), seed)?.scoreTracking,
    draft.scoreTracking,
  );

  draft.scoreTracking.serveMarkers[0].timestamp = 61;
  assert.equal(parseCutDraft(JSON.stringify(draft), seed), null);
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
