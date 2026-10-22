import assert from "node:assert/strict";
import test from "node:test";
import { applyPaddingToCachedCuts, createCutDraft, materializeFinalCutIntervals, parseCutDraft, setCutCoreStart, setCutCoreEnd, setCutCoreRange, splitCutAt } from "../../prod/src/lib/cut-draft.ts";
import { createModelFeedbackBundle } from "../../prod/src/lib/model-feedback.ts";
import { importModelFeedbackProject } from "../../prod/src/lib/model-feedback-import.ts";
import { PRODUCTION_ENSEMBLE_MODEL_ID } from "../../prod/src/lib/on-device/ensemble.ts";
import type { ProductAnalysis } from "../../prod/src/lib/product-analysis.ts";
import * as labDrafts from "../../components/production-lab/editor/lib/cut-draft.ts";
import { createModelFeedbackBundle as createLabFeedbackBundle } from "../../components/production-lab/editor/lib/model-feedback.ts";
import type { SuppressionSuggestion } from "../../prod/src/lib/on-device/suppression-policy.ts";

const duration = 1.0026666666666666;
const analysis: ProductAnalysis = {
  id: "synthetic-fractional-analysis", recordingId: "synthetic-fractional-project", kind: "model",
  modelId: PRODUCTION_ENSEMBLE_MODEL_ID, duration, width: 320, height: 240,
  sourceFilename: "synthetic-fractional.mp4",
  source: { name: "synthetic-fractional.mp4", size: 1000, lastModified: 100, type: "video/mp4" },
  mediaInfo: { duration, width: 320, height: 240, rotation: 0, mimeType: "video/mp4", videoCodec: "avc", videoCodecString: null,
    canDecodeVideo: true, hasAudio: false, audioCodec: null, sampleRate: null, channels: null, canDecodeAudio: false },
  analysisWindow: { start: 0, end: duration }, roi: { x: 0, y: 0, width: 1, height: 1 },
  runtimeVariant: "synthetic-test", videoUrl: null, features: null,
  rallies: [{ id: "R001", start: .1, end: .9, confidence: .9, included: true, agreement: "both-models" }],
  ignoredIntervals: [], inferenceTimes: new Float64Array([0]),
  probabilities: { rally: new Float32Array([.9]), serve: new Float32Array([.1]), deadState: new Float32Array([.1]) },
};

test("millisecond export rounding stays inside exact fractional source and game bounds", () => {
  for (const start of [0, .0003333333333333333]) {
    const seed = { analysisId: analysis.id, recordingId: analysis.recordingId, duration,
      analysisStart: start, analysisEnd: duration, rallies: analysis.rallies, ignoredIntervals: [] };
    const draft = createCutDraft(seed);
    const materialized = materializeFinalCutIntervals(draft);
    assert.deepEqual(materialized.intervals, [{ start, end: duration, cutIds: ["R001"] }]);
    assert.ok(materialized.provenance.every(p => p.start >= start && p.end <= duration));
    const bundle = createModelFeedbackBundle({ ...analysis, analysisWindow: { start, end: duration } }, draft, materialized.intervals);
    const restored = importModelFeedbackProject(JSON.stringify(bundle)).project;
    assert.deepEqual(materializeFinalCutIntervals(restored.importedFeedback!.initialDraft).intervals, materialized.intervals,
      "Unmodified output imports and re-materializes without drifting its source edges");
  }
});

test("interior edges keep millisecond rounding and ignored fractional edges remain barriers", () => {
  let draft = createCutDraft({ analysisId: analysis.id, recordingId: analysis.recordingId, duration,
    rallies: [{ ...analysis.rallies[0], start: .123456, end: .987654 }], ignoredIntervals: [] });
  draft = applyPaddingToCachedCuts(draft, 0, 0, duration);
  assert.deepEqual(materializeFinalCutIntervals(draft).intervals, [{ start: .123, end: .988, cutIds: ["R001"] }]);
  draft.ignoredIntervals = [{ id: "synthetic-exclusion", start: .400444, end: .500555, reason: "non-game-content" }];
  assert.deepEqual(materializeFinalCutIntervals(draft).intervals.map(({ start, end }) => [start, end]), [[.123, .400444], [.500555, .988]]);
});

test("padding, boundary edits and splits retain valid drafts at fractional source/core edges", () => {
  const seed = { analysisId: analysis.id, recordingId: analysis.recordingId, duration,
    analysisStart: .0003333333333333333, analysisEnd: duration,
    rallies: [{ ...analysis.rallies[0], start: .1236, end: .9874 }], ignoredIntervals: [] };
  const draft = createCutDraft(seed);
  const variants = [
    applyPaddingToCachedCuts(draft, 0, 0, duration, seed.analysisStart),
    applyPaddingToCachedCuts(draft, 3, 3, duration, seed.analysisStart),
    setCutCoreStart(draft, "R001", 0), setCutCoreEnd(draft, "R001", 10),
    setCutCoreRange(draft, "R001", 0, 10), splitCutAt(draft, "R001", .5)!.draft,
  ];
  for (const updated of variants) {
    assert.ok(parseCutDraft(JSON.stringify(updated), seed), "Edited geometry must remain persistable");
    for (const cut of updated.cuts) {
      assert.ok(cut.keepStart >= seed.analysisStart && cut.keepStart <= cut.coreStart);
      assert.ok(cut.keepEnd <= duration && cut.coreEnd <= cut.keepEnd);
    }
  }
});

for (const runtime of [
  { name: "production", createCutDraft, applyPaddingToCachedCuts, materializeFinalCutIntervals, createModelFeedbackBundle },
  { name: "lab", ...labDrafts, createModelFeedbackBundle: createLabFeedbackBundle },
]) {
  for (const [start, end, expectedStart, expectedEnd] of [
    [.1236, .9874, .1236, .9874],
    [.5001, .5004, .5, .5004],
  ]) test(`${runtime.name}: zero padding retains fractional core ${start}–${end} through project roundtrip`, () => {
    const rallies = [{ ...analysis.rallies[0], start, end }];
    const seed = { analysisId: analysis.id, recordingId: analysis.recordingId, duration, rallies, ignoredIntervals: [] };
    const draft = runtime.applyPaddingToCachedCuts(runtime.createCutDraft(seed), 0, 0, duration);
    const materialized = runtime.materializeFinalCutIntervals(draft);
    assert.deepEqual(materialized.intervals, [{ start: expectedStart, end: expectedEnd, cutIds: ["R001"] }]);
    assert.ok(materialized.provenance.some(p => p.kind === "inferred-core" && p.start === start && p.end === end));
    const bundle = runtime.createModelFeedbackBundle({ ...analysis, rallies }, draft, materialized.intervals);
    assert.deepEqual(bundle.corrections.scoreTracking.excludedRallyIds, [], "An included short core must remain score-eligible");
    const restored = importModelFeedbackProject(JSON.stringify(bundle)).project;
    assert.deepEqual(runtime.materializeFinalCutIntervals(restored.importedFeedback!.initialDraft).intervals, materialized.intervals);
  });

  test(`${runtime.name}: a submillisecond core retains its serve and completed point in feedback`, () => {
    const rallies = [{ ...analysis.rallies[0], start: .5001, end: .5004 },
      { ...analysis.rallies[0], id: "R002", start: .8, end: .9 }];
    let draft = runtime.createCutDraft({ analysisId: analysis.id, recordingId: analysis.recordingId, duration, rallies, ignoredIntervals: [] });
    draft = runtime.applyPaddingToCachedCuts(draft, 0, 0, duration);
    draft.joinGapSeconds = 0;
    draft.scoreTracking.serveMarkers = rallies.map(r => ({ id: `serve-${r.id}`, rallyId: r.id, timestamp: r.start,
      side: "near", origin: "manual", ignorePreviousPoint: false }));
    const bundle = runtime.createModelFeedbackBundle({ ...analysis, rallies }, draft, runtime.materializeFinalCutIntervals(draft).intervals);
    assert.deepEqual(bundle.corrections.scoreTracking.excludedRallyIds, []);
    assert.equal(bundle.corrections.scoreTracking.derivedFinalScore.team1Score, 1);
    assert.equal(bundle.corrections.scoreTracking.derivedFinalScore.points.length, 1);
    draft.cuts[0].included = false;
    const removed = runtime.createModelFeedbackBundle({ ...analysis, rallies }, draft, runtime.materializeFinalCutIntervals(draft).intervals);
    assert.deepEqual(removed.corrections.scoreTracking.excludedRallyIds, ["R001"]);
    assert.equal(removed.corrections.scoreTracking.derivedFinalScore.team1Score, 0, "The point disappears only when the rally is explicitly excluded");
  });

  test(`${runtime.name}: exact window, ignored and veto edges still bound a fractional core`, () => {
    for (const removal of ["ignored", "veto"] as const) {
      const rallies = [{ ...analysis.rallies[0], start: .5001, end: .5004 }];
      let draft = runtime.createCutDraft({ analysisId: analysis.id, recordingId: analysis.recordingId, duration,
        analysisStart: .50005, analysisEnd: .50045, rallies, ignoredIntervals: [] });
      draft = runtime.applyPaddingToCachedCuts(draft, 0, 0, duration, .50005, .50045);
      draft.joinGapSeconds = 3;
      const veto: SuppressionSuggestion = { id: "synthetic-veto", logicalId: "synthetic-veto", suppressionEventId: "synthetic-event",
        start: .5002, end: .5003, score: .9, sourceProductionIds: ["R001"], eligiblePolicyIds: ["aggressive"] };
      if (removal === "ignored") draft.ignoredIntervals = [{ id: "synthetic-exclusion", start: veto.start, end: veto.end, reason: "non-game-content" }];
      else draft.suppressionScopeOverrides[veto.logicalId] = "veto-region";
      const result = runtime.materializeFinalCutIntervals(draft, removal === "veto" ? { suggestions: [veto] } : undefined);
      assert.deepEqual(result.intervals.map(({ start, end }) => [start, end]), [[.50005, .5002], [.5003, .5004]], removal);
      assert.ok(result.intervals.every(r => r.start >= draft.analysisStart && r.end <= draft.analysisEnd && !r.joinedGaps?.length));
    }
  });
}
