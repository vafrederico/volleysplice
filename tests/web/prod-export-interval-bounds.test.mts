import assert from "node:assert/strict";
import test from "node:test";
import { applyPaddingToCachedCuts, createCutDraft, materializeFinalCutIntervals, parseCutDraft, setCutCoreStart, setCutCoreEnd, setCutCoreRange, splitCutAt } from "../../prod/src/lib/cut-draft.ts";
import { createModelFeedbackBundle } from "../../prod/src/lib/model-feedback.ts";
import { importModelFeedbackProject } from "../../prod/src/lib/model-feedback-import.ts";
import { PRODUCTION_ENSEMBLE_MODEL_ID } from "../../prod/src/lib/on-device/ensemble.ts";
import type { ProductAnalysis } from "../../prod/src/lib/product-analysis.ts";

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
