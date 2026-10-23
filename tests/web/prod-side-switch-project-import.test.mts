import assert from "node:assert/strict";
import test from "node:test";
import { sideSwitchProjectFixture } from "../fixtures/web-side-switch-project.ts";
import { importModelFeedbackProject } from "../../prod/src/lib/model-feedback-import.ts";
import { parseModelFeedback, ModelFeedbackValidationError } from "../../prod/src/lib/model-feedback-validation.ts";
import { createModelFeedbackBundle } from "../../prod/src/lib/model-feedback.ts";
import { createModelReviewDraft } from "../../prod/src/lib/model-review-draft.ts";
import { restoreProjectReviewDraft } from "../../prod/src/lib/review-draft-persistence.ts";
import { deriveScoreAt } from "../../prod/src/lib/score-tracking.ts";
import { encodeNumericArray } from "../../prod/src/lib/model-feedback.ts";

test("side-switch raw evidence and corrected review independently survive import/re-export/reset", () => {
  const { analysis, bundle } = sideSwitchProjectFixture();
  const parsed = parseModelFeedback(bundle);
  assert.deepEqual(parsed.initialInference.sideSwitch, analysis.sideSwitch);
  const { project } = importModelFeedbackProject(JSON.stringify(bundle));
  assert.equal(project.sideSwitchEnabled, true);
  assert.deepEqual(project.analysis!.sideSwitch, analysis.sideSwitch);
  const imported = project.importedFeedback!.initialDraft;
  const seed = {
    analysisId: imported.analysisId, recordingId: project.id, duration: project.info.duration,
    rallies: project.analysis!.intervals, ignoredIntervals: [],
  };
  const restored = restoreProjectReviewDraft(project, seed, null);
  assert.deepEqual(restored.scoreTracking, bundle.corrections.scoreTracking.state);
  const reexport = createModelFeedbackBundle({
    ...analysis, id: seed.analysisId, recordingId: project.id,
    sideSwitch: project.analysis!.sideSwitch, sideSwitchEnabled: project.sideSwitchEnabled,
  }, restored, bundle.finalExportIntervals);
  assert.deepEqual(reexport.initialInference.sideSwitch, bundle.initialInference.sideSwitch);
  assert.deepEqual(reexport.corrections.scoreTracking, bundle.corrections.scoreTracking);
  const reset = createModelReviewDraft(seed, { ...project.analysis!, sideSwitchEnabled: project.sideSwitchEnabled });
  assert.deepEqual(reset.scoreTracking.sideSwitchMarkers.map(m => [m.id, m.timestamp, m.origin]), [
    ["switch-boundary-1", 25, "model"], ["switch-internal-2", 45, "model"],
  ]);
  assert.deepEqual(reset.scoreTracking.removedModelMarkerIds, []);
  assert.notDeepEqual(deriveScoreAt(restored.scoreTracking, 60), deriveScoreAt(reset.scoreTracking, 60));
  const resetExport = createModelFeedbackBundle(analysis, reset, []);
  const resetImport = importModelFeedbackProject(JSON.stringify(resetExport)).project;
  assert.deepEqual(resetImport.importedFeedback!.initialDraft.scoreTracking, reset.scoreTracking);
});

test("disabled preference survives independently of absent, empty or populated side-switch output", () => {
  for (const presence of ["absent", "empty", "populated"]) {
    const { analysis, bundle } = sideSwitchProjectFixture(false);
    if (presence === "absent") bundle.initialInference.sideSwitch = null;
    if (presence === "empty") {
      bundle.initialInference.sideSwitch!.candidates = [];
      // Valid completed output can retain proposal evidence and select no events.
    }
    const { project } = importModelFeedbackProject(JSON.stringify(bundle));
    assert.equal(project.sideSwitchEnabled, false);
    const initialDraft = project.importedFeedback!.initialDraft;
    const seed = { analysisId: initialDraft.analysisId, recordingId: project.id, duration: 60,
      rallies: project.analysis!.intervals, ignoredIntervals: [] };
    const restored = restoreProjectReviewDraft(project, seed, null);
    assert.deepEqual(restored.scoreTracking, initialDraft.scoreTracking, "keep saved corrections until reset");
    const reset = createModelReviewDraft(seed, { ...project.analysis!, sideSwitchEnabled: project.sideSwitchEnabled });
    assert.deepEqual(reset.scoreTracking.sideSwitchMarkers, []);
    const exported = createModelFeedbackBundle({ ...analysis, sideSwitch: project.analysis!.sideSwitch,
      sideSwitchEnabled: project.sideSwitchEnabled }, restored, []);
    assert.equal(exported.source.generateSideSwitchMarkers, false);
    assert.deepEqual(exported.initialInference.sideSwitch, bundle.initialInference.sideSwitch);
  }
});

test("legacy absent and explicit-null raw output remain compatible without invented switch evidence", () => {
  for (const version of [1, 2, 3]) {
    for (const absent of [true, false]) {
      const { bundle } = sideSwitchProjectFixture();
      const legacy = JSON.parse(JSON.stringify(bundle));
      legacy.schemaVersion = version;
      delete legacy.source.generateSideSwitchMarkers;
      if (absent) delete legacy.initialInference.sideSwitch;
      else legacy.initialInference.sideSwitch = null;
      const { project } = importModelFeedbackProject(JSON.stringify(legacy));
      assert.equal(project.sideSwitchEnabled, true, "retain the historical web default");
      assert.equal(project.analysis!.sideSwitch, undefined);
      const draft = project.importedFeedback!.initialDraft;
      assert.equal(draft.scoreTracking.sideSwitchMarkers[0].timestamp, 35);
    }
  }
});

test("zero-row empty specialist result stays present and round-trips", () => {
  const { bundle } = sideSwitchProjectFixture();
  const output = bundle.initialInference.sideSwitch!;
  output.features.rows = 0;
  output.features.values = encodeNumericArray(new Float64Array(), [0, output.features.columns]);
  output.candidates = [];
  assert.equal(importModelFeedbackProject(JSON.stringify(bundle)).project.analysis!.sideSwitch!.features.rows, 0);
});

test("malformed side-switch payloads fail clearly at the offending field", () => {
  const cases: Array<[string, (bundle: any) => void]> = [
    ["generateSideSwitchMarkers", b => { b.source.generateSideSwitchMarkers = "false"; }],
    ["generateSideSwitchMarkers", b => { b.source.generateSideSwitchMarkers = null; }],
    ["sideSwitch", b => { b.initialInference.sideSwitch = []; }],
    ["modelId", b => { b.initialInference.sideSwitch.modelId = ""; }],
    ["modelFingerprint", b => { b.initialInference.sideSwitch.modelFingerprint = null; }],
    ["featureVersion", b => { b.initialInference.sideSwitch.featureVersion = "future"; }],
    ["candidateContract", b => { b.initialInference.sideSwitch.candidateContract = "future"; }],
    ["columns", b => { b.initialInference.sideSwitch.features.columns = 1; }],
    ["rows", b => { b.initialInference.sideSwitch.features.rows = -1; }],
    ["shape", b => { b.initialInference.sideSwitch.features.values.shape = [2, 34]; }],
    ["shape", b => { b.initialInference.sideSwitch.features.rows = 25_000_001; b.initialInference.sideSwitch.features.values.shape = [25_000_001, 34]; }],
    ["data", b => { b.initialInference.sideSwitch.features.values.data = "bad"; }],
    ["finite", b => { const out = b.initialInference.sideSwitch; const bytes = new Float64Array(3 * 34); bytes[0] = Infinity; out.features.values = encodeNumericArray(bytes, [3, 34]); }],
    ["candidates", b => { b.initialInference.sideSwitch.candidates = {}; }],
    ["candidates", b => { b.initialInference.sideSwitch.candidates.push(...b.initialInference.sideSwitch.candidates); }],
    ["unique", b => { b.initialInference.sideSwitch.candidates[1].id = b.initialInference.sideSwitch.candidates[0].id; }],
    ["timestamp", b => { b.initialInference.sideSwitch.candidates[0].timestamp = 61; }],
    ["timestamp", b => { b.initialInference.sideSwitch.candidates[0].timestamp = -1; }],
    ["probability", b => { b.initialInference.sideSwitch.candidates[0].probability = 1.1; }],
    ["kind", b => { b.initialInference.sideSwitch.candidates[0].kind = "other"; }],
    ["sourceRangeIds", b => { b.initialInference.sideSwitch.candidates[0].sourceRangeIds = [""]; }],
  ];
  for (const [field, mutate] of cases) {
    const { bundle } = sideSwitchProjectFixture();
    mutate(bundle);
    assert.throws(() => importModelFeedbackProject(JSON.stringify(bundle)), error =>
      error instanceof ModelFeedbackValidationError && error.message.includes(field), field);
  }
});
