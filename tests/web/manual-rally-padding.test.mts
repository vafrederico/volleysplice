import assert from "node:assert/strict";
import test from "node:test";
import {
  applyPaddingToCachedCuts,
  buildFinalCutIntervals,
  createCutDraft,
  newManualCut,
  parseCutDraft,
  setCutCoreRange,
  setCutKeepBoundary,
} from "../../prod/src/lib/cut-draft.ts";
import {
  applyPaddingToCachedCuts as applyLabPadding,
  createCutDraft as createLabDraft,
} from "../../components/production-lab/editor/lib/cut-draft.ts";
import { addServeMarker } from "../../prod/src/lib/score-tracking.ts";

const seed = {
  analysisId: "manual-padding", recordingId: "synthetic", duration: 40,
  analysisStart: 1, analysisEnd: 39,
  rallies: [], ignoredIntervals: [],
};

test("manual rally marks use current padding in preview, export and saved drafts", () => {
  const base = createCutDraft(seed);
  const defaultCut = newManualCut(base, 10, 20);
  assert.equal(defaultCut.keepStart, 8);
  assert.equal(defaultCut.keepEnd, 22);
  const configured = applyPaddingToCachedCuts(base, 3, 1.5, 40, 1, 39);
  const manual = newManualCut(configured, 10, 20);
  const draft = { ...configured, cuts: [manual] };
  assert.deepEqual(buildFinalCutIntervals(draft), [{ start: 7, end: 21.5, cutIds: [manual.id] }]);
  assert.deepEqual(parseCutDraft(JSON.stringify(draft), seed)?.cuts, [manual]);
  const edited = setCutCoreRange(draft, manual.id, 12, 18);
  assert.equal(edited.cuts[0].keepStart, 9);
  assert.equal(edited.cuts[0].keepEnd, 19.5);
  const repadded = applyPaddingToCachedCuts(edited, 1, 4, 40, 1, 39);
  assert.deepEqual(buildFinalCutIntervals(repadded), [{ start: 11, end: 22, cutIds: [manual.id] }]);
});

for (const [name, create, apply] of [
  ["production", createCutDraft, applyPaddingToCachedCuts],
  ["lab", createLabDraft, applyLabPadding],
] as const) {
  test(`${name} padding changes include manual rallies and preserve their core`, () => {
    const base = create(seed);
    const cut = newManualCut(base, 1.5, 38.5);
    const draft = { ...base, cuts: [cut] };
    assert.equal(cut.keepStart, 1);
    assert.equal(cut.keepEnd, 39);
    const padded = apply(draft, 10, 10, 40, 1, 39);
    assert.deepEqual(padded.cuts, [cut]);
    const zero = apply(draft, 0, 0, 40, 1, 39);
    assert.deepEqual(zero.cuts, [{ ...cut, keepStart: 1.5, keepEnd: 38.5 }]);
    assert.deepEqual(newManualCut({ ...zero, cuts: [] }, 1.5, 38.5), zero.cuts[0]);
  });
}

test("manual outer handles change padding without moving core or serve markers", () => {
  const base = createCutDraft(seed);
  const cut = newManualCut(base, 10, 20);
  const draft = {
    ...base, cuts: [cut],
    scoreTracking: addServeMarker(base.scoreTracking, cut.coreStart, "near", { id: "serve-M001", rallyId: cut.id }),
  };
  const start = setCutKeepBoundary(draft, cut.id, 7, true);
  const both = setCutKeepBoundary(start, cut.id, 23, false);
  assert.deepEqual(both.cuts, [{ ...cut, keepStart: 7, keepEnd: 23 }]);
  assert.deepEqual(both.scoreTracking, draft.scoreTracking);
  const clampedStart = setCutKeepBoundary(both, cut.id, 30, true);
  const clamped = setCutKeepBoundary(clampedStart, cut.id, 0, false);
  assert.deepEqual(clamped.cuts, [{ ...cut, keepStart: 10, keepEnd: 20 }]);
});

test("manual padding joins nearby export footage while ignored spans remain excluded", () => {
  const base = createCutDraft({ ...seed, rallies: [
    { id: "R001", start: 5, end: 10, included: true, confidence: 0.9 },
  ] });
  const cut = newManualCut(base, 13, 20);
  const draft = { ...base, cuts: [...base.cuts, cut] };
  assert.deepEqual(buildFinalCutIntervals(draft), [{ start: 3, end: 22, cutIds: ["R001", cut.id] }]);
  draft.ignoredIntervals.push({ id: "I001", start: 11.5, end: 12, reason: "camera-gap" });
  assert.deepEqual(buildFinalCutIntervals(draft).map(({ start, end }) => [start, end]), [[3, 11.5], [12, 22]]);
});
