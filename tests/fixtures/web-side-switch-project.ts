// Synthetic evidence only: no source media or model execution is needed.
import type { ProductAnalysis } from "../../prod/src/lib/product-analysis.ts";
import { createModelReviewDraft } from "../../prod/src/lib/model-review-draft.ts";
import { createModelFeedbackBundle } from "../../prod/src/lib/model-feedback.ts";
import { materializeFinalCutIntervals } from "../../prod/src/lib/cut-draft.ts";
import { PRODUCTION_ENSEMBLE_MODEL_ID } from "../../prod/src/lib/on-device/ensemble.ts";
import {
  SIDE_SWITCH_MODEL_ID,
  SIDE_SWITCH_MODEL_FINGERPRINT,
  SIDE_SWITCH_FEATURE_VERSION,
  SIDE_SWITCH_CANDIDATE_CONTRACT,
  SIDE_SWITCH_FEATURE_NAMES,
} from "../../prod/src/lib/on-device/side-switch-model.ts";
import {
  SERVING_SIDE_MODEL_ID,
  SERVING_SIDE_MODEL_FINGERPRINT,
  SERVING_SIDE_FEATURE_VERSION,
  SERVING_SIDE_ANCHOR_CONTRACT,
  SERVING_SIDE_FEATURE_COLUMNS,
} from "../../prod/src/lib/on-device/serving-side-model.ts";

export function sideSwitchProjectFixture(enabled = true) {
  const rallies = [10, 20, 30, 50].map((start, index) => ({
    id: `R00${index + 1}`, start, end: start + 5, confidence: 0.9,
    included: true, agreement: "both-models" as const,
  }));
  const evidence = {
    modelId: "synthetic-serve-head", threshold: 0.85, peakProbability: 0.9,
    peakTime: 10, crossesThreshold: true, nearestDetection: null,
  };
  const analysis: ProductAnalysis = {
    id: "synthetic-analysis", recordingId: "synthetic-project", kind: "model",
    modelId: PRODUCTION_ENSEMBLE_MODEL_ID, duration: 60, analysisWindow: { start: 0, end: 60 },
    width: 640, height: 360, sourceFilename: "synthetic-match.mp4",
    source: { name: "synthetic-match.mp4", size: 1000, lastModified: 100, type: "video/mp4" },
    mediaInfo: {
      duration: 60, width: 640, height: 360, rotation: 0, mimeType: "video/mp4",
      videoCodec: "avc", videoCodecString: null, canDecodeVideo: true,
      hasAudio: false, audioCodec: null, sampleRate: null, channels: null, canDecodeAudio: false,
    },
    roi: { x: 0, y: 0, width: 1, height: 1 }, runtimeVariant: "synthetic-web-test",
    videoUrl: null, rallies, ignoredIntervals: [], features: null,
    inferenceTimes: new Float64Array([0, 1]),
    probabilities: {
      rally: new Float32Array([0.5, 0.9]), serve: new Float32Array([0.2, 0.8]),
      deadState: new Float32Array([0.5, 0.1]),
    },
    servingSide: {
      modelId: SERVING_SIDE_MODEL_ID, modelFingerprint: SERVING_SIDE_MODEL_FINGERPRINT,
      featureVersion: SERVING_SIDE_FEATURE_VERSION, anchorContract: SERVING_SIDE_ANCHOR_CONTRACT,
      features: { rows: 4, columns: SERVING_SIDE_FEATURE_COLUMNS, values: new Float64Array(4 * SERVING_SIDE_FEATURE_COLUMNS) },
      candidates: rallies.map((rally, index) => ({
        id: rally.id, anchor: rally.start, interval: { start: rally.start, end: rally.end },
        nearProbability: index % 2 ? 0.1 : 0.9,
        side: index % 2 ? "far" : "near", verdict: index % 2 ? "far" : "near",
        serveDecisionSource: "serve-head", reviewReasons: [],
        serveEvidence: {
          allLabelsV2: { ...evidence, peakTime: rally.start },
          previousProduction: { ...evidence, peakTime: rally.start },
        },
      })),
    },
    sideSwitchEnabled: enabled,
    sideSwitch: {
      modelId: SIDE_SWITCH_MODEL_ID, modelFingerprint: SIDE_SWITCH_MODEL_FINGERPRINT,
      featureVersion: SIDE_SWITCH_FEATURE_VERSION, candidateContract: SIDE_SWITCH_CANDIDATE_CONTRACT,
      features: {
        rows: 3, columns: SIDE_SWITCH_FEATURE_NAMES.length,
        values: Float64Array.from({ length: 3 * SIDE_SWITCH_FEATURE_NAMES.length }, (_, i) => (i - 50) / 8),
      },
      candidates: [
        { id: "boundary-1", timestamp: 25, probability: 0.95, kind: "adjacent-rally-boundary", sourceRangeIds: ["R002", "R003"] },
        { id: "internal-2", timestamp: 45, probability: 0.87, kind: "internal-dead-state-peak", sourceRangeIds: ["R003"] },
      ],
    },
  };
  const seed = {
    analysisId: analysis.id, recordingId: analysis.recordingId, duration: analysis.duration,
    rallies: analysis.rallies, ignoredIntervals: [],
  };
  // The saved review differs from immutable inference in timestamps and removals.
  const draft = createModelReviewDraft(seed, { ...analysis, sideSwitchEnabled: true });
  draft.scoreTracking.sideSwitchMarkers[0].timestamp = 35;
  draft.scoreTracking.sideSwitchMarkers.splice(1, 1);
  draft.scoreTracking.sideSwitchMarkers.push({ id: "manual-switch", timestamp: 40, origin: "manual" });
  draft.scoreTracking.removedModelMarkerIds = ["switch-internal-2"];
  draft.scoreTracking.team1Name = "Team A";
  draft.scoreTracking.team2Name = "Team B";
  draft.updatedAt = "2026-01-01T01:00:00.000Z";
  const bundle = createModelFeedbackBundle(analysis, draft, materializeFinalCutIntervals(draft).intervals, "2026-01-01T02:00:00.000Z");
  return { analysis, seed, draft, bundle };
}
