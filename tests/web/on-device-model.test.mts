import assert from "node:assert/strict";
import test from "node:test";
import {
  composeServeAnchoredIntervals,
  decodeDeadStateAfterServe,
  decodeProbabilities,
  decodeServeProbabilities,
  loadOnDeviceModelBundle,
  predictLogistic,
  refineDeadStateEnds,
  runOnDeviceModel,
  type OnDeviceModelBundle,
} from "../../lib/on-device/model.ts";

const rallyDecoder = {
  smoothing_seconds: 0,
  enter_threshold: 0.5,
  exit_threshold: 0.4,
  min_live_seconds: 0.5,
  bridge_gap_seconds: 0,
  short_event_min_seconds: 0.25,
  short_event_threshold: 0.9,
};

const permissiveDecoder = {
  ...rallyDecoder,
  min_live_seconds: 0.25,
};

const serveDecoder = {
  threshold: 0.85,
  minSeparationSeconds: 10,
  timeOffsetSeconds: 0.25,
};

const composition = {
  method: "serve-anchor-permissive-live-fallback-v1",
  associationSeconds: 0.5,
  fallbackSeconds: 2.5,
  maxRescueSeconds: 5,
  permissiveDecoder,
};

const deadDecoder = {
  deadThreshold: 0.9,
  liveResetThreshold: 0.4,
  minimumLiveSamples: 1,
  minimumDeadSamples: 1,
  minAfterServeSeconds: 0,
  maxAfterServeSeconds: 1.5,
  timeOffsetSeconds: 0,
};

const refinement = {
  method: "refine-v5-end",
  endWindowSeconds: 0.75,
};

function rawBundle(dimensions = 2) {
  const head = (weights = Array.from({ length: dimensions }, () => 0)) => ({
    mean: Array.from({ length: dimensions }, () => 0),
    scale: Array.from({ length: dimensions }, () => 1),
    weights,
    bias: 0,
  });
  return {
    schemaVersion: 1,
    analysisFps: 4,
    featureNames: Array.from({ length: dimensions }, (_, index) => `feature_${index}`),
    rally: { ...head(), decoder: { ...rallyDecoder } },
    serve: {
      ...head(),
      decoder: { ...serveDecoder },
      composition: {
        ...composition,
        permissiveDecoder: { ...composition.permissiveDecoder },
      },
    },
    deadState: {
      ...head(),
      decoder: { ...deadDecoder },
      refinement: { ...refinement },
    },
  };
}

test("model bundle loader normalizes artifact decoder keys and validates dimensions", () => {
  const raw = rawBundle();
  // The model artifacts use snake_case for rally decoders.
  delete (raw.rally.decoder as Partial<typeof rallyDecoder>).short_event_min_seconds;
  delete (raw.rally.decoder as Partial<typeof rallyDecoder>).short_event_threshold;
  const loaded = loadOnDeviceModelBundle(raw);

  assert.ok(loaded.rally.weights instanceof Float32Array);
  assert.deepEqual(loaded.featureNames, ["feature_0", "feature_1"]);
  assert.equal(loaded.rally.decoder.short_event_min_seconds, rallyDecoder.min_live_seconds);
  assert.equal(loaded.rally.decoder.short_event_threshold, 1);

  const bad = rawBundle();
  bad.serve.weights.pop();
  assert.throws(() => loadOnDeviceModelBundle(bad), /serve\.weights must contain 2 values/);

  const badScale = rawBundle();
  badScale.deadState.scale[1] = 0;
  assert.throws(() => loadOnDeviceModelBundle(badScale), /deadState\.scale\[1\] must be positive/);
});

test("logistic prediction applies float32 normalization and clipped sigmoid", () => {
  const bundle = loadOnDeviceModelBundle(rawBundle()) as OnDeviceModelBundle;
  const head = {
    mean: new Float32Array([1, 2]),
    scale: new Float32Array([2, 4]),
    weights: new Float32Array([2, -1]),
    bias: 0.5,
  };
  const result = predictLogistic(head, new Float32Array([3, 6, 1, 2]));
  assert.equal(result.length, 2);
  assert.ok(Math.abs(result[0] - 1 / (1 + Math.exp(-1.5))) < 1e-6);
  assert.ok(Math.abs(result[1] - 1 / (1 + Math.exp(-0.5))) < 1e-6);
  assert.throws(() => predictLogistic(bundle.rally, [1, 2, 3]), /flat \(\*, 2\) matrix/);
});

test("rally decoder preserves hysteresis intervals, bridges gaps, and keeps strong short events", () => {
  const times = new Float64Array([0.125, 0.375, 0.625, 0.875]);
  const decoded = decodeProbabilities(
    times,
    new Float32Array([0.1, 0.6, 0.55, 0.2]),
    1,
    rallyDecoder,
    4,
  );
  assert.equal(decoded.intervals.length, 1);
  assert.deepEqual(
    { start: decoded.intervals[0].start, end: decoded.intervals[0].end },
    { start: 0.25, end: 0.75 },
  );
  assert.ok(Math.abs(decoded.intervals[0].confidence - 0.575) < 1e-6);

  const withGap = decodeProbabilities(
    new Float64Array([0.125, 0.375, 0.625, 0.875, 1.125]),
    new Float32Array([0.95, 0.2, 0.95, 0.1, 0.1]),
    1.25,
    { ...rallyDecoder, min_live_seconds: 1, bridge_gap_seconds: 0.25 },
    4,
  );
  assert.equal(withGap.intervals.length, 1);
  assert.equal(withGap.intervals[0].end, 0.75);
});

test("serve decoder uses island peaks and score-first non-maximum suppression", () => {
  const detections = decodeServeProbabilities(
    new Float64Array([0, 1, 2, 3, 4, 5]),
    new Float32Array([0.1, 0.9, 0.91, 0.1, 0.97, 0.1]),
    { threshold: 0.85, minSeparationSeconds: 3, timeOffsetSeconds: 0.25 },
    6,
  );
  // The stronger second island wins even though it occurs later.
  assert.deepEqual(detections.map((item) => item.time), [4.25]);
  assert.ok(Math.abs(detections[0].confidence - 0.97) < 1e-6);
});

test("serve composition anchors primary starts, rescues permissive events, and falls back locally", () => {
  const intervals = composeServeAnchoredIntervals(
    [{ start: 2, end: 5, confidence: 0.6 }],
    [{ start: 9.8, end: 11, confidence: 0.7 }],
    [
      { time: 1.75, confidence: 0.9 },
      { time: 10, confidence: 0.95 },
      { time: 20, confidence: 0.91 },
    ],
    30,
    composition,
    0.25,
  );
  assert.deepEqual(
    intervals.map(({ start, end }) => ({ start, end })),
    [
      { start: 1.75, end: 5 },
      { start: 10, end: 11 },
      { start: 20, end: 22.5 },
    ],
  );
  assert.equal(intervals[0].confidence, 0.9);
  assert.equal(intervals[1].confidence, 0.95);
});

test("dead-state decoder requires live evidence before a stable transition", () => {
  const times = new Float64Array([2.25, 2.5, 2.75, 3]);
  const detection = decodeDeadStateAfterServe(
    times,
    new Float32Array([0.2, 0.3, 0.95, 0.96]),
    2.25,
    { ...deadDecoder, minimumDeadSamples: 2 },
    { duration: 4 },
  );
  assert.equal(detection?.time, 2.75);
  assert.ok(Math.abs((detection?.confidence ?? 0) - 0.95) < 1e-6);

  const alreadyDead = decodeDeadStateAfterServe(
    times,
    new Float32Array([0.95, 0.96, 0.97, 0.98]),
    2.25,
    deadDecoder,
    { duration: 4 },
  );
  assert.equal(alreadyDead, null);
});

test("dead-state refinement is a local end adjustment and a missing transition is a no-op", () => {
  const times = new Float64Array([
    0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25,
  ]);
  const probabilities = Float32Array.from(times, (time) => (time < 2.75 ? 0.2 : 0.95));
  const source = [{ start: 1, end: 3, confidence: 0.7 }];
  const result = refineDeadStateEnds(
    times,
    probabilities,
    source,
    4,
    0.25,
    deadDecoder,
    refinement,
  );
  assert.equal(result.intervals[0].end, 2.75);
  assert.equal(result.transitions[0]?.time, 2.75);

  const noTransition = refineDeadStateEnds(
    times,
    new Float32Array(times.length).fill(0.2),
    source,
    4,
    0.25,
    deadDecoder,
    refinement,
  );
  assert.deepEqual(noTransition.intervals, source);
  assert.deepEqual(noTransition.transitions, [null]);
});

test("complete runtime returns edit-list-compatible rallies after all three heads", () => {
  const raw = rawBundle(3);
  raw.rally.weights = [10, 0, 0];
  raw.serve.weights = [0, 10, 0];
  raw.deadState.weights = [0, 0, 10];
  const bundle = loadOnDeviceModelBundle(raw);
  const times = Float64Array.from({ length: 10 }, (_, index) => 0.125 + index * 0.25);
  const features = new Float32Array(times.length * 3);
  for (let index = 0; index < times.length; index += 1) {
    features[index * 3] = index >= 1 && index <= 7 ? 1 : -1;
    features[index * 3 + 1] = index === 1 ? 1 : -1;
    features[index * 3 + 2] = index >= 6 ? 1 : -1;
  }

  const result = runOnDeviceModel(bundle, times, features, 2.5);
  assert.equal(result.rallies.length, 1);
  assert.deepEqual(
    {
      id: result.rallies[0].id,
      start: result.rallies[0].start,
      end: result.rallies[0].end,
      included: result.rallies[0].included,
    },
    { id: "R001", start: 0.25, end: 1.625, included: true },
  );
  assert.equal(result.serves.length, 1);
  assert.equal(result.probabilities.rally.length, times.length);
});
