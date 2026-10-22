import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";

import { ModelFeedbackValidationError } from "../../lib/model-feedback.ts";
import { createCutDraft, materializeFinalCutIntervals, parseCutDraft } from "../../prod/src/lib/cut-draft.ts";
import {
  importModelFeedbackProject,
} from "../../prod/src/lib/model-feedback-import.ts";
import { createModelFeedbackBundle } from "../../prod/src/lib/model-feedback.ts";
import { ALL_LABELS_V2_MODEL_ID, PRODUCTION_ENSEMBLE_MODEL_ID } from "../../prod/src/lib/on-device/ensemble.ts";
import { RALLY_MODEL_OPTIONS } from "../../prod/src/lib/on-device/rally-model.ts";
import {
  SERVING_SIDE_ANCHOR_CONTRACT,
  SERVING_SIDE_FEATURE_COLUMNS,
  SERVING_SIDE_FEATURE_VERSION,
  SERVING_SIDE_MODEL_FINGERPRINT,
  SERVING_SIDE_MODEL_ID,
} from "../../prod/src/lib/on-device/serving-side-model.ts";
import {
  SUPPRESSION_ARTIFACT_SHA256,
  SUPPRESSION_DECODER_VERSION,
  SUPPRESSION_MODEL_ID,
  SUPPRESSION_WEIGHTS_SHA256,
} from "../../prod/src/lib/on-device/suppression-model.ts";
import { SUPPRESSION_POLICY_CONTRACT_VERSION } from "../../prod/src/lib/on-device/suppression-policy.ts";
import type { ProductAnalysis } from "../../prod/src/lib/product-analysis.ts";
import { normalizeStoredProject } from "../../prod/src/lib/project-store.ts";
import { addServeMarker, deriveScoreAt } from "../../prod/src/lib/score-tracking.ts";
import { scoreTrackingWithServingSideOutput } from "../../prod/src/lib/score-tracking-inference.ts";

const analysis: ProductAnalysis = {
  id: "analysis-original",
  recordingId: "project-original",
  kind: "model",
  modelId: PRODUCTION_ENSEMBLE_MODEL_ID,
  duration: 30,
  analysisWindow: { start: 2, end: 28 },
  width: 1920,
  height: 1080,
  sourceFilename: "match.mp4",
  source: {
    name: "match.mp4",
    size: 6_000,
    lastModified: 1_786_800_000_000,
    type: "video/mp4",
    fingerprint: `sampled-sha256-v1:${"a".repeat(64)}`,
  },
  mediaInfo: {
    duration: 30,
    mimeType: "video/mp4",
    width: 1920,
    height: 1080,
    rotation: 0,
    videoCodec: "avc",
    videoCodecString: "avc1.640028",
    canDecodeVideo: true,
    hasAudio: true,
    audioCodec: "aac",
    sampleRate: 48_000,
    channels: 2,
    canDecodeAudio: true,
  },
  roi: { x: 0.05, y: 0.1, width: 0.9, height: 0.85 },
  runtimeVariant: "libswresample-wasm-v1",
  videoUrl: null,
  rallies: [
    {
      id: "R001",
      start: 5,
      end: 8,
      confidence: 0.9,
      included: true,
      agreement: "both-models",
    },
  ],
  ignoredIntervals: [
    { start: 0, end: 2, reason: "outside-game-window" },
    { start: 28, end: 30, reason: "outside-game-window" },
  ],
  features: {
    times: new Float64Array([2, 2.25]),
    values: new Float32Array([1, 2, 3, 4]),
    rows: 2,
    columns: 2,
    names: ["motion", "audio"],
  },
  inferenceTimes: new Float64Array([2, 2.25]),
  probabilities: {
    rally: new Float32Array([0.2, 0.8]),
    serve: new Float32Array([0.1, 0.9]),
    deadState: new Float32Array([0.7, 0.2]),
  },
  productionComponents: {
    allLabelsV2: [
      {
        id: "A001",
        start: 5,
        end: 8,
        confidence: 0.91,
        included: true,
        agreement: "both-models",
      },
    ],
    previousProduction: [
      {
        id: "P001",
        start: 5.1,
        end: 7.9,
        confidence: 0.88,
        included: true,
        agreement: "both-models",
      },
    ],
  },
  productionServeOutputs: {
    allLabelsV2: {
      probabilities: new Float32Array([0.1, 0.9]),
      detections: [{ time: 5.1, confidence: 0.9 }],
    },
    previousProduction: {
      probabilities: new Float32Array([0.2, 0.8]),
      detections: [{ time: 5.2, confidence: 0.8 }],
    },
  },
  servingSide: {
    modelId: SERVING_SIDE_MODEL_ID,
    modelFingerprint: SERVING_SIDE_MODEL_FINGERPRINT,
    featureVersion: SERVING_SIDE_FEATURE_VERSION,
    anchorContract: SERVING_SIDE_ANCHOR_CONTRACT,
    features: {
      rows: 1,
      columns: SERVING_SIDE_FEATURE_COLUMNS,
      values: new Float64Array(SERVING_SIDE_FEATURE_COLUMNS),
    },
    candidates: [
      {
        id: "R001",
        anchor: 5,
        interval: { start: 5, end: 8, agreement: "both-models" },
        nearProbability: 0.75,
        side: "near",
        verdict: "near",
        serveDecisionSource: "serve-head",
        reviewReasons: [],
        serveEvidence: {
          allLabelsV2: {
            modelId: "all-labels-v2",
            threshold: 0.85,
            peakProbability: 0.9,
            peakTime: 5.1,
            crossesThreshold: true,
            nearestDetection: { time: 5.1, confidence: 0.9 },
          },
          previousProduction: {
            modelId: "previous-production",
            threshold: 0.85,
            peakProbability: 0.8,
            peakTime: 5.2,
            crossesThreshold: false,
            nearestDetection: { time: 5.2, confidence: 0.8 },
          },
        },
      },
    ],
  },
  suppression: {
    modelId: SUPPRESSION_MODEL_ID,
    artifactSha256: SUPPRESSION_ARTIFACT_SHA256,
    weightsSha256: SUPPRESSION_WEIGHTS_SHA256,
    decoderVersion: SUPPRESSION_DECODER_VERSION,
    policyContractVersion: SUPPRESSION_POLICY_CONTRACT_VERSION,
    probabilities: new Float32Array([0.1, 0.95]),
    decodedIntervals: [
      {
        id: "S001",
        start: 6,
        end: 7,
        confidence: 0.95,
        included: true,
      },
    ],
    suggestions: [
      {
        id: "suggestion-1",
        logicalId: "logical-1",
        suppressionEventId: "S001",
        start: 6,
        end: 7,
        score: 0.95,
        sourceProductionIds: ["P001"],
        eligiblePolicyIds: ["conservative", "balanced", "aggressive"],
      },
    ],
    identicalPolicyResults: true,
  },
};

function feedbackText(keepRally = false) {
  const draft = createCutDraft({
    analysisId: analysis.id,
    recordingId: analysis.recordingId,
    duration: analysis.duration,
    analysisStart: analysis.analysisWindow.start,
    analysisEnd: analysis.analysisWindow.end,
    rallies: analysis.rallies,
    ignoredIntervals: analysis.ignoredIntervals,
    suppressionContractVersion: SUPPRESSION_POLICY_CONTRACT_VERSION,
  });
  draft.updatedAt = "2026-08-20T20:00:00.000Z";
  draft.renderScoreOnFinalVideo = false;
  draft.cuts[0].included = keepRally;
  draft.selectedSuppressionPolicy = keepRally ? "none" : "aggressive";
  draft.suppressionDecisionOverrides = { "logical-1": "suppress" };
  draft.scoreTracking = {
    ...addServeMarker(draft.scoreTracking, 5, "near", {
      id: "serve-R001",
      origin: "model",
      modelSide: "near",
      rallyId: "R001",
    }),
    team1Name: "Falcons",
    team2Name: "Owls",
  };
  const bundle = createModelFeedbackBundle(
    analysis,
    draft,
    materializeFinalCutIntervals(draft, analysis.suppression).intervals,
    "2026-08-20T21:00:00.000Z",
  );
  return JSON.stringify(bundle);
}

test("imported moved serves survive the editor's model refresh in both directions", () => {
  for (const movedStart of [6, 4]) {
    const bundle = JSON.parse(feedbackText());
    const corrected = bundle.corrections.correctedRanges[0];
    corrected.coreStart = movedStart;
    corrected.keepStart = movedStart - 2;
    const savedMarker = bundle.corrections.scoreTracking.state.serveMarkers[0];
    savedMarker.timestamp = movedStart;
    savedMarker.ignorePreviousPoint = true;
    const { project } = importModelFeedbackProject(JSON.stringify(bundle));
    const draft = project.importedFeedback!.initialDraft;
    assert.equal(draft.scoreTracking.serveMarkers[0].timestamp, movedStart);
    const refreshed = scoreTrackingWithServingSideOutput(draft.scoreTracking, project.analysis!.servingSide!);
    assert.equal(refreshed.serveMarkers[0].timestamp, movedStart);
    assert.equal(refreshed.serveMarkers[0].timestamp, draft.cuts[0].coreStart);
    assert.equal(refreshed.serveMarkers[0].ignorePreviousPoint, true);
    assert.equal(deriveScoreAt(refreshed, movedStart - .001).servingSide, null);
    assert.equal(deriveScoreAt(refreshed, movedStart).servingSide, "near");
    assert.equal(project.analysis!.servingSide!.candidates[0].anchor, 5);
  }
});

test("feedback without cached numeric features imports its corrected markers", () => {
  const bundle = JSON.parse(feedbackText());
  bundle.features = null;
  bundle.initialInference.rows = 0;
  bundle.initialInference.timestamps = { ...bundle.initialInference.timestamps, shape: [0], data: "" };
  for (const value of Object.values(bundle.initialInference.probabilities) as { shape: number[]; data: string }[]) {
    value.shape = [0];
    value.data = "";
  }
  bundle.initialInference.suppression = null;
  bundle.initialInference.componentServeOutputs = undefined;
  const { project } = importModelFeedbackProject(JSON.stringify(bundle));
  assert.equal(project.analysis!.times.length, 0);
  assert.equal(project.importedFeedback!.initialDraft.scoreTracking.serveMarkers[0].timestamp, 5);
  bundle.initialInference.timestamps.shape = [1];
  bundle.initialInference.rows = 1;
  assert.throws(() => importModelFeedbackProject(JSON.stringify(bundle)), ModelFeedbackValidationError);
});

test("model feedback imports as a distinct ready project with its editor state", () => {
  const imported = importModelFeedbackProject(feedbackText(), {
    importedAt: "2026-08-21T00:00:00.000Z",
  });

  assert.equal(imported.project.status, "ready");
  assert.match(imported.project.id, /^feedback-/);
  assert.equal(imported.project.source.name, "match.mp4");
  assert.equal(imported.project.source.fingerprint, analysis.source.fingerprint);
  assert.equal(imported.project.analysis?.featureValues?.length, 4);
  assert.equal(imported.project.analysis?.productionComponents?.allLabelsV2[0].id, "A001");
  assert.equal(imported.project.analysis?.suppression?.suggestions[0].id, "suggestion-1");
  assert.equal(imported.project.analysis?.servingSide?.candidates[0].side, "near");
  assert.equal(imported.project.importedFeedback?.originalProjectId, "project-original");
  assert.equal(imported.project.importedFeedback?.initialDraft.cuts[0].included, false);
  assert.equal(imported.project.importedFeedback?.initialDraft.selectedSuppressionPolicy, "aggressive");
  assert.equal(imported.project.importedFeedback?.initialDraft.scoreTracking.team1Name, "Falcons");
  assert.equal(imported.project.importedFeedback?.initialDraft.renderScoreOnFinalVideo, false);
  assert.equal(normalizeStoredProject(imported.project), imported.project);
});

test("portable final-video score preference defaults on for old projects and validates booleans", () => {
  const bundle = JSON.parse(feedbackText());
  assert.equal(bundle.editorPreferences.renderScoreOnFinalVideo, false);
  delete bundle.corrections.ui;
  for (const preferences of [undefined, {}, { futurePreference: true }, { renderScoreOnFinalVideo: true }]) {
    bundle.editorPreferences = preferences;
    const { project } = importModelFeedbackProject(JSON.stringify(bundle));
    assert.equal(project.importedFeedback?.initialDraft.renderScoreOnFinalVideo, true);
  }
  for (const preferences of [null, [], { renderScoreOnFinalVideo: "false" }, { renderScoreOnFinalVideo: null }]) {
    bundle.editorPreferences = preferences;
    assert.throws(() => importModelFeedbackProject(JSON.stringify(bundle)), ModelFeedbackValidationError);
  }
});

const uiFixture = JSON.parse(readFileSync(
  new URL("../fixtures/portable-review-preferences.json", import.meta.url), "utf8",
));

function portableRoundTrip(ui: unknown, alias = false) {
  const bundle = JSON.parse(feedbackText(true));
  if (ui === undefined) delete bundle.corrections.ui;
  else bundle.corrections.ui = ui;
  bundle.editorPreferences = { renderScoreOnFinalVideo: alias };
  const project = normalizeStoredProject(importModelFeedbackProject(JSON.stringify(bundle)).project);
  const draft = project.importedFeedback!.initialDraft;
  const seed = { analysisId: draft.analysisId, recordingId: draft.recordingId,
    duration: analysis.duration, analysisStart: analysis.analysisWindow.start,
    analysisEnd: analysis.analysisWindow.end, rallies: analysis.rallies,
    ignoredIntervals: analysis.ignoredIntervals };
  const restored = parseCutDraft(JSON.stringify(draft), seed);
  assert.ok(restored, "the imported UI envelope survives local draft validation");
  const intervals = materializeFinalCutIntervals(restored, analysis.suppression).intervals;
  const reexported = createModelFeedbackBundle(analysis, restored, intervals);
  const reimported = importModelFeedbackProject(JSON.stringify(reexported)).project.importedFeedback!.initialDraft;
  return { bundle, project, draft: restored, reexported, reimported };
}

test("full feedback import, draft restore, export and reimport retain portable UI and unknown nested extensions", () => {
  for (const preview of [false, true]) for (const final of [false, true]) {
    const ui = structuredClone(uiFixture.portableUi);
    ui.preferences.renderScoreOnPreviewVideo = preview;
    ui.preferences.renderScoreOnFinalVideo = final;
    const { bundle, draft, reexported, reimported } = portableRoundTrip(ui, !final);
    for (const value of [draft, reimported]) {
      assert.equal(value.renderScoreOverlay, preview);
      assert.equal(value.renderScoreOnFinalVideo, final, "canonical UI wins over the PR8 alias");
      assert.equal(value.renderScoreTimeline, true);
      assert.equal(value.fadeScoreOverlay, true);
      assert.equal(value.cutPreviewEnabled, true);
      assert.equal(value.playbackRate, 1.5);
      assert.equal(value.confidenceReviewThreshold, .73);
      assert.deepEqual(value.reviewedCutIds, ["R001"]);
    }
    assert.deepEqual(reexported.corrections.ui, ui);
    assert.equal(reexported.editorPreferences?.renderScoreOnFinalVideo, final);
    assert.deepEqual(reexported.corrections.correctedRanges, bundle.corrections.correctedRanges);
    assert.deepEqual(reexported.corrections.scoreTracking, bundle.corrections.scoreTracking);
    assert.deepEqual(reexported.finalExportIntervals, bundle.finalExportIntervals);
    assert.deepEqual(reexported.finalExportProvenance, bundle.finalExportProvenance);
    // Export current edits while retaining unknown preferences and siblings.
    const edited = createModelFeedbackBundle(analysis, { ...draft, renderScoreOverlay: !preview,
      renderScoreOnFinalVideo: !final, playbackRate: 1.75, reviewedCutIds: [] }, []);
    const expected = structuredClone(ui);
    Object.assign(expected.preferences, { renderScoreOnPreviewVideo: !preview,
      renderScoreOnFinalVideo: !final, playbackRate: 1.75, reviewedCutIds: [] });
    assert.deepEqual(edited.corrections.ui, expected);
    assert.equal(edited.editorPreferences?.renderScoreOnFinalVideo, !final);
  }
});

test("legacy absent UI retains historical web defaults and unversioned iOS chapter options survive", () => {
  for (const ui of [undefined, {}, uiFixture.legacyIosUi]) {
    const { draft, reexported } = portableRoundTrip(ui, true);
    assert.equal(draft.renderScoreOverlay, true);
    assert.equal(draft.renderScoreOnFinalVideo, true);
    assert.equal(draft.renderScoreTimeline, false);
    assert.equal(draft.fadeScoreOverlay, false);
    assert.equal(draft.cutPreviewEnabled, false);
    assert.equal(draft.playbackRate, 1);
    assert.equal(draft.confidenceReviewThreshold, .7);
    assert.deepEqual(draft.reviewedCutIds, []);
    if (ui?.preferences) assert.deepEqual(
      (reexported.corrections.ui!.preferences as any).chapterOptions, ui.preferences.chapterOptions,
    );
  }
});

test("future UI versions remain opaque through web edits and re-export", () => {
  const { draft, reexported } = portableRoundTrip(uiFixture.futureUi);
  assert.equal(draft.playbackRate, 1);
  assert.equal(draft.renderScoreOverlay, true);
  assert.deepEqual(reexported.corrections.ui, uiFixture.futureUi);
  const edited = createModelFeedbackBundle(analysis, { ...draft, playbackRate: 2 }, []);
  assert.deepEqual(edited.corrections.ui, uiFixture.futureUi, "never downgrade or overwrite a future envelope");
});

test("bounded playback multipliers activate exactly and timeline off remains off", () => {
  for (const rate of [.25, 1, 1.5, 1.75, 8]) {
    const ui = structuredClone(uiFixture.portableUi);
    ui.preferences.playbackRate = rate;
    ui.preferences.renderScoreTimeline = false;
    const { draft, reimported } = portableRoundTrip(ui);
    assert.equal(draft.playbackRate, rate);
    assert.equal(reimported.playbackRate, rate);
    assert.equal(reimported.renderScoreTimeline, false);
  }
});

test("invalid known values and unsafe or unbounded UI fail before project creation", () => {
  let deep: any = {};
  for (let index = 0; index < 17; index++) deep = { child: deep };
  const invalid = [...uiFixture.invalidUi,
    { preferences: { reviewedCutIds: ["missing-cut"] } },
    { preferences: { reviewedCutIds: ["x".repeat(257)] } },
    { preferences: { confidenceReviewThreshold: -.1 } },
    { schemaVersion: 2, preferences: { playbackRate: "fast" } },
    { extension: deep }, { extension: Array(4096).fill(null) },
    { extension: "x".repeat(16385) },
    { extension: Array(5).fill("x".repeat(16384)) },
  ];
  for (const ui of invalid) {
    const bundle = JSON.parse(feedbackText());
    bundle.corrections.ui = ui;
    assert.throws(() => importModelFeedbackProject(JSON.stringify(bundle)), ModelFeedbackValidationError);
  }
  assert.equal(({} as any).polluted, undefined);
});

test("reimporting the same feedback allocates a new project ID", () => {
  const first = importModelFeedbackProject(feedbackText());
  const second = importModelFeedbackProject(feedbackText(), {
    occupiedProjectIds: new Set([first.project.id]),
  });

  assert.equal(second.project.id, `${first.project.id}-2`);
});

test("invalid model feedback does not create a project", () => {
  assert.throws(
    () => importModelFeedbackProject("{not-json"),
    ModelFeedbackValidationError,
  );
});


test("Android neural feedback preserves provenance, corrected cuts and serving side through restore", () => {
  for (const modelId of ["distilled-large-recall-v1", "distilled-large-f1-v1"]) {
    const bundle = JSON.parse(feedbackText());
    bundle.source.runtimeVariant = "native-android-distilled-large-v1";
    bundle.initialInference.modelId = modelId;
    bundle.initialInference.ranges[0].agreement = "neural";
    bundle.initialInference.servingSide.candidates[0].interval.agreement = "neural";
    bundle.initialInference.suppression = null;
    const corrected = bundle.corrections.correctedRanges[0];
    corrected.agreement = "neural";
    corrected.coreStart = 5.5;
    corrected.keepStart = 3.5;
    bundle.corrections.scoreTracking.state.serveMarkers[0].timestamp = 5.5;
    const { project } = importModelFeedbackProject(JSON.stringify(bundle));
    assert.equal(project.analysis!.modelId, modelId);
    assert.equal(project.analysis!.intervals[0].agreement, "neural");
    assert.equal(project.analysis!.servingSide!.candidates[0].interval.agreement, "neural");
    assert.equal(project.analysis!.servingSide!.candidates[0].side, "near");
    const restored = normalizeStoredProject(project);
    assert.equal(restored.status, "ready");
    assert.equal(restored.analysis!.modelId, modelId);
    const draft = restored.importedFeedback!.initialDraft;
    assert.equal(draft.cuts[0].agreement, "neural");
    assert.equal(draft.cuts[0].coreStart, 5.5);
    assert.equal(draft.cuts[0].included, false);
    assert.equal(draft.scoreTracking.serveMarkers[0].timestamp, 5.5);
  }
});

test("neural recovery provenance survives feedback and old hidden serves reuse saved side predictions", () => {
  for (const oldGate of [false, true]) {
    const bundle = JSON.parse(feedbackText());
    bundle.initialInference.modelId = "distilled-large-f1-v1";
    bundle.initialInference.ranges[0].agreement = "neural";
    bundle.initialInference.suppression = null;
    const candidate = bundle.initialInference.servingSide.candidates[0];
    candidate.interval.agreement = "neural";
    candidate.verdict = oldGate ? "not-serve" : "review";
    candidate.serveDecisionSource = oldGate ? "none" : "neural-rally-recovery";
    candidate.reviewReasons = oldGate ? [] : ["neural-rally-recovery"];
    const { project } = importModelFeedbackProject(JSON.stringify(bundle));
    const restored = project.analysis!.servingSide!.candidates[0];
    assert.equal(restored.verdict, "review");
    assert.equal(restored.serveDecisionSource, "neural-rally-recovery");
    assert.equal(restored.side, candidate.side);
    assert.equal(restored.nearProbability, candidate.nearProbability);
    assert.deepEqual(restored.reviewReasons, ["neural-rally-recovery"]);
  }
});

test("model selection and mixed probability sources survive project export, import and re-export", () => {
  for (const option of RALLY_MODEL_OPTIONS) for (const runtimeVariant of [analysis.runtimeVariant, "native-android-dsp-v1"]) {
    const neural = option.value !== "ensemble";
    const selected = { ...analysis, runtimeVariant,
      modelId: neural ? option.id : PRODUCTION_ENSEMBLE_MODEL_ID,
      rallies: analysis.rallies.map(range => ({ ...range, agreement: neural ? "neural" as const : range.agreement })),
      suppression: undefined,
      neuralScores: neural ? { modelId: option.id, heads: ["live", "serve", "end", "keep"],
        timestamps: new Float64Array([2, 2.25]),
        probabilities: new Float32Array([.1, .2, .3, .4, .5, .6, .7, .8]) } : undefined,
    };
    const draft = createCutDraft({ analysisId: selected.id, recordingId: selected.recordingId,
      duration: selected.duration, rallies: selected.rallies, ignoredIntervals: [] });
    const bundle = createModelFeedbackBundle(selected, draft, []);
    const expectedRallySource = neural && !runtimeVariant.startsWith("native-android")
      ? selected.modelId : ALL_LABELS_V2_MODEL_ID;
    assert.equal(bundle.initialInference.modelId, selected.modelId);
    assert.equal(bundle.initialInference.modelLabel, option.label);
    assert.equal(bundle.initialInference.modelSelection, option.value);
    assert.equal(bundle.initialInference.probabilityModelId, expectedRallySource);
    assert.deepEqual(bundle.initialInference.probabilityModelIds, {
      rally: expectedRallySource, serve: ALL_LABELS_V2_MODEL_ID, deadState: ALL_LABELS_V2_MODEL_ID,
    });
    const project = normalizeStoredProject(importModelFeedbackProject(JSON.stringify(bundle)).project);
    assert.equal(project.status, "ready");
    assert.equal(project.analysis!.modelId, selected.modelId);
    assert.deepEqual(project.analysis!.featureValues, selected.features!.values);
    assert.deepEqual(project.analysis!.neuralScores, selected.neuralScores);
    assert.ok(!JSON.stringify(bundle).includes("embeddings"));
    const reexported = createModelFeedbackBundle({ ...selected,
      probabilityModelIds: project.analysis!.probabilityModelIds,
      neuralScores: project.analysis!.neuralScores }, draft, []);
    assert.equal(reexported.initialInference.modelId, selected.modelId);
    assert.deepEqual(reexported.initialInference.probabilityModelIds, bundle.initialInference.probabilityModelIds);
    assert.deepEqual(reexported.initialInference.neuralScores, bundle.initialInference.neuralScores);
  }
});

test("neural score imports reject wrong identity, head order, shape and invalid values", () => {
  const model = RALLY_MODEL_OPTIONS[0];
  const selected: ProductAnalysis = { ...analysis, modelId: model.id,
    rallies: analysis.rallies.map(range => ({ ...range, agreement: "neural" })), suppression: undefined,
    neuralScores: { modelId: model.id, heads: ["live", "serve", "end", "keep"],
      timestamps: new Float64Array([2, 2.25]), probabilities: new Float32Array([.1, .2, .3, .4, .5, .6, .7, .8]) } };
  const draft = createCutDraft({ analysisId: selected.id, recordingId: selected.recordingId,
    duration: selected.duration, rallies: selected.rallies, ignoredIntervals: [] });
  for (const mutate of [
    (scores: any) => { scores.modelId = RALLY_MODEL_OPTIONS[1].id; },
    (scores: any) => { scores.heads.reverse(); },
    (scores: any) => { scores.probabilities.shape = [4, 2]; },
    (scores: any) => { scores.probabilities.data = Buffer.from(new Float32Array(8).fill(1.1).buffer).toString("base64"); },
    (scores: any) => { scores.timestamps.data = Buffer.from(new Float64Array([2.25, 2]).buffer).toString("base64"); },
  ]) {
    const bundle = createModelFeedbackBundle(selected, draft, []);
    mutate(bundle.initialInference.neuralScores);
    assert.throws(() => importModelFeedbackProject(JSON.stringify(bundle)), ModelFeedbackValidationError);
  }
  const bundle = createModelFeedbackBundle({ ...selected, features: null }, draft, []);
  assert.deepEqual(importModelFeedbackProject(JSON.stringify(bundle)).project.analysis?.neuralScores,
    selected.neuralScores, "scores survive imports even when old AV features are absent");
});
