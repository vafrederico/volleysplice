import assert from "node:assert/strict";
import test from "node:test";

import { ModelFeedbackValidationError } from "../../lib/model-feedback.ts";
import { createCutDraft } from "../../prod/src/lib/cut-draft.ts";
import {
  importModelFeedbackProject,
} from "../../prod/src/lib/model-feedback-import.ts";
import { createModelFeedbackBundle } from "../../prod/src/lib/model-feedback.ts";
import { PRODUCTION_ENSEMBLE_MODEL_ID } from "../../prod/src/lib/on-device/ensemble.ts";
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
import { addServeMarker } from "../../prod/src/lib/score-tracking.ts";

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

function feedbackText() {
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
  draft.cuts[0].included = false;
  draft.selectedSuppressionPolicy = "aggressive";
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
    [],
    "2026-08-20T21:00:00.000Z",
  );
  return JSON.stringify(bundle);
}

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
  assert.equal(normalizeStoredProject(imported.project), imported.project);
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
