// Synthetic score evidence; no source media or identifying recording data.
export function recoveryFixture({ modelId = "distilled-large-recall-v1", empty = false, switches = false } = {}) {
  const interval = { id: "R001", start: 10, end: 20, confidence: .9, included: true,
    agreement: modelId.startsWith("distilled-") ? "neural" : "both-models" };
  const head = () => ({ probabilities: new Float32Array([.8]), detections: [] });
  const state = () => ({ rallyProbabilities: new Float32Array([.8]), deadStateProbabilities: new Float32Array([.2]) });
  return {
    schemaVersion: 1, id: "synthetic-score-project", source: { name: "match.mp4", size: 5, lastModified: 100, type: "video/mp4" },
    info: { duration: 60, width: 640, height: 360, mimeType: "video/mp4", videoCodec: "avc", canDecodeVideo: true, hasAudio: false, canDecodeAudio: false },
    roi: { x: 0, y: 0, width: 1, height: 1 }, analysisWindow: { start: 0, end: 60 }, sideSwitchEnabled: switches,
    status: "ready", error: null, createdAt: "2026-01-01T00:00:00.000Z", updatedAt: "2026-01-01T00:00:00.000Z",
    analysis: { modelId, featurePath: "local-source", intervals: empty ? [] : [interval], times: new Float64Array([0]),
      rallyProbabilities: new Float32Array([.8]), serveProbabilities: new Float32Array([.8]), deadStateProbabilities: new Float32Array([.2]),
      productionServeOutputs: { allLabelsV2: head(), previousProduction: head() },
      productionComponents: { allLabelsV2: empty ? [] : [interval], previousProduction: empty ? [] : [interval] },
      productionStateOutputs: { allLabelsV2: state(), previousProduction: state() } },
  };
}

export function servingFixture(analysis, serving, neural) {
  const bundle = neural.neuralServingBundle(analysis.modelId);
  const evidence = { probability: .9, nearestDetection: null };
  return { modelId: bundle?.id ?? serving.SERVING_SIDE_MODEL_ID,
    modelFingerprint: bundle?.fingerprint ?? serving.SERVING_SIDE_MODEL_FINGERPRINT,
    featureVersion: serving.SERVING_SIDE_FEATURE_VERSION, anchorContract: serving.SERVING_SIDE_ANCHOR_CONTRACT,
    features: { rows: analysis.intervals.length, columns: 237, values: new Float64Array(analysis.intervals.length * 237) },
    candidates: analysis.intervals.map((interval) => ({ id: interval.id, anchor: interval.start, interval: { ...interval },
      nearProbability: .9, side: "near", verdict: "near", serveDecisionSource: "serve-head", reviewReasons: [],
      serveEvidence: { allLabelsV2: evidence, previousProduction: evidence } })) };
}

export function switchFixture(model, candidates = []) {
  return { modelId: model.SIDE_SWITCH_MODEL_ID, modelFingerprint: model.SIDE_SWITCH_MODEL_FINGERPRINT,
    featureVersion: model.SIDE_SWITCH_FEATURE_VERSION, candidateContract: model.SIDE_SWITCH_CANDIDATE_CONTRACT,
    features: { rows: candidates.length, columns: model.SIDE_SWITCH_FEATURE_NAMES.length,
      values: new Float64Array(candidates.length * model.SIDE_SWITCH_FEATURE_NAMES.length) }, candidates };
}
