import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { feedbackUiFromDraft, parseFeedbackUi } from "../../prod/src/lib/feedback-ui.ts";
import { feedbackUiFromDraft as labFeedbackUiFromDraft, parseFeedbackUi as parseLabFeedbackUi } from "../../components/production-lab/editor/lib/feedback-ui.ts";
import {
  createCutDraft as createLabCutDraft,
  parseCutDraft as parseLabCutDraft,
} from "../../components/production-lab/editor/lib/cut-draft.ts";

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
const portableUi = JSON.parse(readFileSync(
  new URL("../fixtures/portable-review-preferences.json", import.meta.url), "utf8",
)).portableUi;

for (const [name, create, parse] of [
  ["production", createCutDraft, parseCutDraft],
  ["lab", createLabCutDraft, parseLabCutDraft],
] as const) {
  test(`${name} v16 and current drafts retain bounded UI extensions and exact imported playback rates`, () => {
    const draft = { ...create(seed), portableUi, playbackRate: 1.75 };
    const restored = parse(JSON.stringify(draft), seed);
    assert.equal(restored?.version, 17);
    assert.equal(restored?.playbackRate, 1.75);
    assert.deepEqual(restored?.portableUi, portableUi);
    const migrated = parse(JSON.stringify({ ...draft, version: 16 }), seed);
    assert.equal(migrated?.version, 17);
    assert.equal(migrated?.playbackRate, 1.75);
    assert.deepEqual(migrated?.portableUi, portableUi);
    for (const invalid of [null, [], { preferences: [] }, { preferences: { playbackRate: "fast" } }]) {
      assert.equal(parse(JSON.stringify({ ...draft, portableUi: invalid }), seed), null);
    }
    for (const invalid of [0, 8.01, null, "1.5"]) {
      assert.equal(parse(JSON.stringify({ ...draft, playbackRate: invalid }), seed), null);
    }
  });
  test(`${name} final-video score preference persists independently of preview`, () => {
    assert.equal(create(seed).renderScoreOnFinalVideo, true);
    for (const finalVideo of [false, true]) {
      for (const preview of [false, true]) {
        const draft = { ...create(seed), renderScoreOnFinalVideo: finalVideo, renderScoreOverlay: preview };
        const restored = parse(JSON.stringify(draft), seed);
        assert.equal(restored?.renderScoreOnFinalVideo, finalVideo);
        assert.equal(restored?.renderScoreOverlay, preview);
      }
    }
  });

  test(`${name} migrates old final-video defaults and rejects malformed current preferences`, () => {
    for (const version of [12, 13, 14, 15]) {
      const legacy = { ...create(seed), version, renderScoreOverlay: false } as Record<string, unknown>;
      delete legacy.renderScoreOnFinalVideo;
      assert.equal(parse(JSON.stringify(legacy), seed)?.renderScoreOnFinalVideo, true);
    }
    for (const invalid of [undefined, null, "false", 0]) {
      assert.equal(parse(JSON.stringify({ ...create(seed), renderScoreOnFinalVideo: invalid }), seed), null);
    }
  });
}

test("retained UI accepts only safe JSON, even before serialization", () => {
  for (const invalid of [NaN, Infinity, () => {}, new Date(), new Map(), new Uint8Array([1])]) {
    assert.throws(() => parseFeedbackUi({ extension: invalid }));
  }
  assert.throws(() => parseFeedbackUi({ schemaVersion: 1.5 }));
  assert.throws(() => parseFeedbackUi({ schemaVersion: Number.MAX_SAFE_INTEGER + 1 }));
  const original = structuredClone(portableUi);
  const copied = parseFeedbackUi(original);
  (copied.preferences as Record<string, unknown>).playbackRate = 2;
  assert.equal(original.preferences.playbackRate, 1.5, "retention never aliases untrusted objects");
});

for (const [name, parse, compose] of [
  ["production", parseFeedbackUi, feedbackUiFromDraft],
  ["lab", parseLabFeedbackUi, labFeedbackUiFromDraft],
] as const) {
  test(`${name} future UI distinguishes absent IDs from malformed or stale wire IDs`, () => {
    const inferred = new Set(["R001"]);
    for (const preferences of [undefined, {}, { reviewedCutIds: [] }, { reviewedCutIds: ["R001"] }]) {
      const ui = { schemaVersion: 2, ...(preferences === undefined ? {} : { preferences }) };
      assert.deepEqual(parse(ui, inferred), ui);
    }
    for (const ids of [null, "R001", {}, [false], [""], ["R001", "R001"]]) {
      assert.throws(() => parse({ schemaVersion: 2, preferences: { reviewedCutIds: ids } }), /reviewedCutIds/);
    }
    const opaque = { schemaVersion: 2, preferences: { reviewedCutIds: ["missing-cut"],
      chapterOptions: { future: [null, { retained: true }] } }, extension: "unchanged" };
    assert.deepEqual(parse(opaque), opaque, "local retention has no wire identity context");
    const draft = { ...createCutDraft(seed), portableUi: opaque, playbackRate: 1.75 };
    const before = JSON.stringify(draft);
    assert.throws(() => compose(draft), /reviewedCutIds.*inferred cuts/);
    assert.equal(JSON.stringify(draft), before);
  });
}

test("production drafts persist reviewed model ranges", () => {
  const draft = createCutDraft(seed);
  draft.reviewedCutIds = ["R001"];
  draft.renderScoreOverlay = true;
  draft.renderScoreTimeline = true;
  draft.fadeScoreOverlay = true;

  const restored = parseCutDraft(JSON.stringify(draft), seed);

  assert.deepEqual(restored?.reviewedCutIds, ["R001"]);
  assert.equal(restored?.renderScoreOverlay, true);
  assert.equal(restored?.renderScoreTimeline, true);
  assert.equal(restored?.fadeScoreOverlay, true);
  assert.deepEqual(restored?.cuts, JSON.parse(JSON.stringify(draft.cuts)));
});

test("score overlay defaults on with an always-visible scoreboard and migrates older drafts", () => {
  const priorVisibilityDraft = { ...createCutDraft(seed), version: 14 } as Record<string, unknown>;
  delete priorVisibilityDraft.fadeScoreOverlay;
  assert.equal(parseCutDraft(JSON.stringify(priorVisibilityDraft), seed)?.fadeScoreOverlay, false);
  assert.equal(createCutDraft(seed).renderScoreOverlay, true);
  assert.equal(createCutDraft(seed).renderScoreTimeline, false);

  const legacy = createCutDraft(seed) as unknown as Record<string, unknown>;
  legacy.version = 12;
  delete legacy.renderScoreOverlay;

  assert.equal(
    parseCutDraft(JSON.stringify(legacy), seed)?.renderScoreOverlay,
    false,
  );
  assert.equal(
    parseCutDraft(JSON.stringify(legacy), seed)?.renderScoreTimeline,
    false,
  );

  const priorOverlayDraft = createCutDraft(seed) as unknown as Record<string, unknown>;
  priorOverlayDraft.version = 13;
  priorOverlayDraft.renderScoreOverlay = true;
  delete priorOverlayDraft.renderScoreTimeline;
  assert.equal(
    parseCutDraft(JSON.stringify(priorOverlayDraft), seed)?.renderScoreTimeline,
    true,
    "version 13 preserves the timeline behavior that was bundled with score rendering",
  );
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
    version: 3,
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
    sideSwitchMarkers: [{ id: "switch-1", timestamp: 15, origin: "manual" }],
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
