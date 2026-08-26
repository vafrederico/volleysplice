import assert from "node:assert/strict";
import test from "node:test";

import { parseModelFeedback } from "../../lib/model-feedback.ts";
import {
  buildFinalCutIntervals,
  createCutDraft,
} from "../../prod/src/lib/cut-draft.ts";
import {
  createModelFeedbackBundle,
  modelFeedbackFilename,
} from "../../prod/src/lib/model-feedback.ts";
import {
  ALL_LABELS_V2_MODEL_ID,
  PREVIOUS_PRODUCTION_MODEL_ID,
  PRODUCTION_ENSEMBLE_MODEL_ID,
} from "../../prod/src/lib/on-device/ensemble.ts";
import {
  SUPPRESSION_ARTIFACT_SHA256,
  SUPPRESSION_DECODER_VERSION,
  SUPPRESSION_MODEL_ID,
  SUPPRESSION_WEIGHTS_SHA256,
} from "../../prod/src/lib/on-device/suppression-model.ts";
import { SUPPRESSION_POLICY_CONTRACT_VERSION } from "../../prod/src/lib/on-device/suppression-policy.ts";
import type { ProductAnalysis } from "../../prod/src/lib/product-analysis.ts";
import {
  addServeMarker,
  addSideSwitchMarker,
  removeServeMarker,
  setPreviousPointIgnored,
  setServeMarkerSide,
} from "../../prod/src/lib/score-tracking.ts";

const analysis: ProductAnalysis = {
  id: "analysis-1",
  recordingId: "project-1",
  kind: "model",
  modelId: PRODUCTION_ENSEMBLE_MODEL_ID,
  duration: 30,
  analysisWindow: { start: 2, end: 28 },
  width: 1920,
  height: 1080,
  sourceFilename: "Match One.mp4",
  source: {
    name: "Match One.mp4",
    size: 1234,
    lastModified: 1_786_752_000_000,
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
  roi: { x: 0.03, y: 0.12, width: 0.94, height: 0.86 },
  runtimeVariant: "libswresample-wasm-v1",
  videoUrl: null,
  rallies: [
    {
      id: "R001",
      start: 5,
      end: 8,
      confidence: 0.8,
      included: true,
      agreement: "both-models",
    },
    {
      id: "R002",
      start: 12,
      end: 15,
      confidence: 0.6,
      included: true,
      agreement: "all-labels-v2-only",
    },
  ],
  ignoredIntervals: [],
  features: {
    times: new Float64Array([2, 2.25]),
    values: new Float32Array([1.25, -2.5, 3.75, 4.5]),
    rows: 2,
    columns: 2,
    names: ["motion", "audio"],
  },
  inferenceTimes: new Float64Array([2, 2.25]),
  probabilities: {
    rally: new Float32Array([0.1, 0.9]),
    serve: new Float32Array([0.2, 0.8]),
    deadState: new Float32Array([0.3, 0.7]),
  },
  productionComponents: {
    allLabelsV2: [
      {
        id: "AV2-001",
        start: 5,
        end: 8,
        confidence: 0.8,
        included: true,
      },
    ],
    previousProduction: [
      {
        id: "PP-001",
        start: 12,
        end: 15,
        confidence: 0.6,
        included: true,
      },
    ],
  },
  productionServeOutputs: {
    allLabelsV2: {
      probabilities: new Float32Array([0.2, 0.8]),
      detections: [{ time: 5.25, confidence: 0.91 }],
    },
    previousProduction: {
      probabilities: new Float32Array([0.4, 0.7]),
      detections: [{ time: 5.5, confidence: 0.88 }],
    },
  },
  servingSide: {
    modelId: "serving-side-fixed-flight-v3",
    modelFingerprint: "a".repeat(64),
    featureVersion: "SERVSIDE237-FLIGHT",
    anchorContract: "merged-production-interval-start-v1",
    features: {
      rows: 2,
      columns: 237,
      values: Float64Array.from(
        { length: 2 * 237 },
        (_, index) => index / 1000,
      ),
    },
    candidates: [
      {
        id: "R001",
        anchor: 5,
        interval: { start: 5, end: 8, agreement: "both-models" },
        nearProbability: 0.8,
        side: "near",
        verdict: "near",
        serveDecisionSource: "serve-head",
        reviewReasons: [],
        serveEvidence: {
          allLabelsV2: {
            modelId: ALL_LABELS_V2_MODEL_ID,
            threshold: 0.5,
            peakProbability: 0.91,
            peakTime: 5.25,
            crossesThreshold: true,
            nearestDetection: { time: 5.25, confidence: 0.91 },
          },
          previousProduction: {
            modelId: PREVIOUS_PRODUCTION_MODEL_ID,
            threshold: 0.5,
            peakProbability: 0.88,
            peakTime: 5.5,
            crossesThreshold: true,
            nearestDetection: { time: 5.5, confidence: 0.88 },
          },
        },
      },
      {
        id: "R002",
        anchor: 12,
        interval: {
          start: 12,
          end: 15,
          agreement: "all-labels-v2-only",
        },
        nearProbability: 0.44,
        side: "far",
        verdict: "review",
        serveDecisionSource: "production-rally-recovery",
        reviewReasons: ["side-score", "production-rally-recovery"],
        serveEvidence: {
          allLabelsV2: {
            modelId: ALL_LABELS_V2_MODEL_ID,
            threshold: 0.5,
            peakProbability: 0.2,
            peakTime: 12,
            crossesThreshold: false,
            nearestDetection: null,
          },
          previousProduction: {
            modelId: PREVIOUS_PRODUCTION_MODEL_ID,
            threshold: 0.5,
            peakProbability: 0.7,
            peakTime: 12.25,
            crossesThreshold: true,
            nearestDetection: { time: 12.25, confidence: 0.7 },
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
    probabilities: new Float32Array([0.2, 0.95]),
    decodedIntervals: [
      {
        id: "S001",
        start: 12.5,
        end: 13.5,
        confidence: 0.95,
        included: true,
      },
    ],
    suggestions: [
      {
        id: "suppression-logical-12500-13500",
        logicalId: "suppression-logical",
        suppressionEventId: "S001-12500-13500",
        start: 12.5,
        end: 13.5,
        score: 0.95,
        sourceProductionIds: ["PP-001"],
        eligiblePolicyIds: ["conservative", "balanced", "aggressive"],
      },
    ],
    identicalPolicyResults: true,
  },
};

test("model feedback preserves initial inference and classifies corrections", () => {
  const draft = createCutDraft({
    analysisId: analysis.id,
    recordingId: analysis.recordingId,
    duration: analysis.duration,
    analysisStart: analysis.analysisWindow.start,
    analysisEnd: analysis.analysisWindow.end,
    rallies: analysis.rallies,
    ignoredIntervals: [],
  });
  draft.updatedAt = "2026-08-15T01:00:00.000Z";
  draft.cuts[0].included = false;
  draft.cuts.push(
    {
      id: "M001",
      coreStart: 18,
      coreEnd: 20,
      keepStart: 18,
      keepEnd: 20,
      confidence: 1,
      included: true,
      origin: "manual",
    },
    {
      id: "M002",
      coreStart: 22,
      coreEnd: 23,
      keepStart: 22,
      keepEnd: 23,
      confidence: 1,
      included: false,
      origin: "manual",
    },
  );
  let scoreTracking = addServeMarker(draft.scoreTracking, 5, "near", {
    id: "serve-R001",
    origin: "model",
    rallyId: "R001",
  });
  scoreTracking = addServeMarker(scoreTracking, 12, "review", {
    id: "serve-R002",
    origin: "model",
    rallyId: "R002",
  });
  scoreTracking = setServeMarkerSide(scoreTracking, "serve-R002", "far");
  scoreTracking = addSideSwitchMarker(scoreTracking, 16, "X001");
  scoreTracking = addServeMarker(scoreTracking, 20, "near", { id: "S001" });
  scoreTracking = setPreviousPointIgnored(scoreTracking, "S001", true);
  scoreTracking = removeServeMarker(scoreTracking, "serve-R001");
  draft.scoreTracking = {
    ...scoreTracking,
    team1Name: "Falcons",
    team2Name: "Owls",
  };

  const bundle = createModelFeedbackBundle(
    analysis,
    draft,
    buildFinalCutIntervals(draft),
    "2026-08-15T02:00:00.000Z",
  );

  assert.equal(bundle.schema, "volleycut-model-feedback");
  assert.equal(bundle.schemaVersion, 3);
  assert.equal(bundle.source.videoBytesIncluded, false);
  assert.equal(
    bundle.source.file.sampledFingerprint,
    analysis.source.fingerprint,
  );
  assert.deepEqual(
    bundle.initialInference.ranges.map(({ id, included }) => ({
      id,
      included,
    })),
    [
      { id: "R001", included: true },
      { id: "R002", included: true },
    ],
  );
  assert.deepEqual(
    bundle.corrections.labels.falsePositives.map(({ id }) => id),
    ["R001"],
  );
  assert.deepEqual(
    bundle.corrections.labels.falseNegatives.map(({ id }) => id),
    ["M001"],
  );
  assert.deepEqual(
    bundle.corrections.labels.confirmedModelRanges.map(({ id }) => id),
    ["R002"],
  );
  assert.deepEqual(
    bundle.corrections.labels.discardedManualRanges.map(({ id }) => id),
    ["M002"],
  );
  assert.equal(
    bundle.initialInference.suppression?.artifactSha256,
    SUPPRESSION_ARTIFACT_SHA256,
  );
  assert.equal(bundle.initialInference.servingSide?.features.rows, 2);
  assert.deepEqual(
    bundle.initialInference.servingSide?.features.values.shape,
    [2, 237],
  );
  assert.equal(
    bundle.initialInference.servingSide?.candidates[1].verdict,
    "review",
  );
  assert.equal(bundle.corrections.scoreTracking.state.team1Name, "Falcons");
  assert.deepEqual(
    bundle.corrections.scoreTracking.state.removedModelMarkerIds,
    ["serve-R001"],
  );
  assert.equal(
    bundle.corrections.scoreTracking.state.serveMarkers[0].modelSide,
    "review",
  );
  assert.equal(
    bundle.corrections.scoreTracking.state.serveMarkers[0].side,
    "far",
  );
  assert.deepEqual(bundle.corrections.scoreTracking.excludedRallyIds, [
    "R001",
    "R002",
    "M002",
  ]);
  assert.equal(
    bundle.corrections.scoreTracking.derivedFinalScore.ignoredPointCount,
    0,
  );
  const imported = parseModelFeedback(bundle);
  assert.equal(imported.initialInference.servingSide?.features.columns, 237);
  assert.equal(
    imported.corrections.scoreTracking?.state.serveMarkers[0].side,
    "far",
  );
  assert.deepEqual(bundle.corrections.suppression.decisions, [
    {
      suggestionId: "suppression-logical-12500-13500",
      logicalId: "suppression-logical",
      state: "suppressed",
      scope: "whole-rally",
    },
  ]);
  assert.ok(bundle.finalExportProvenance.length > 0);
});

test("model feedback encodes source-aligned features as little-endian base64", () => {
  const draft = createCutDraft({
    analysisId: analysis.id,
    recordingId: analysis.recordingId,
    duration: analysis.duration,
    analysisStart: analysis.analysisWindow.start,
    analysisEnd: analysis.analysisWindow.end,
    rallies: analysis.rallies,
    ignoredIntervals: [],
  });
  const bundle = createModelFeedbackBundle(analysis, draft, []);
  assert.deepEqual(bundle.features?.timestamps.shape, [2]);
  assert.deepEqual(bundle.features?.values.shape, [2, 2]);
  assert.deepEqual(bundle.initialInference.timestamps.shape, [2]);
  assert.equal(
    bundle.initialInference.componentServeOutputs?.allLabelsV2.modelId,
    ALL_LABELS_V2_MODEL_ID,
  );
  assert.equal(
    bundle.initialInference.componentServeOutputs?.previousProduction.modelId,
    PREVIOUS_PRODUCTION_MODEL_ID,
  );
  assert.deepEqual(
    bundle.initialInference.componentServeOutputs?.allLabelsV2.probabilities
      .shape,
    [2],
  );
  assert.deepEqual(
    bundle.initialInference.componentServeOutputs?.previousProduction
      .detections,
    [{ time: 5.5, confidence: 0.88 }],
  );
  const bytes = Buffer.from(bundle.features!.values.data, "base64");
  assert.equal(bytes.readFloatLE(0), 1.25);
  assert.equal(bytes.readFloatLE(4), -2.5);
  assert.equal(bytes.readFloatLE(8), 3.75);
  assert.equal(bytes.readFloatLE(12), 4.5);
  assert.equal(
    modelFeedbackFilename("Match One.mp4"),
    "Match-One.model-feedback.json",
  );
});

test("legacy analyses export useful feedback with an explicit feature warning", () => {
  const legacy = { ...analysis, features: null };
  const draft = createCutDraft({
    analysisId: legacy.id,
    recordingId: legacy.recordingId,
    duration: legacy.duration,
    rallies: legacy.rallies,
    ignoredIntervals: [],
  });
  const bundle = createModelFeedbackBundle(legacy, draft, []);
  assert.equal(bundle.features, null);
  assert.match(bundle.warnings[0], /predates model-feedback capture/);
});
