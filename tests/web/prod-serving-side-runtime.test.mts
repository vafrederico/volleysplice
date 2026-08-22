import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  composeServingSideVerdict,
  courtFlowFeatureNames,
  evaluateHybridGate,
  flightFeatureNames,
  isReusableServingSideOutput,
  parseServingSideRuntime,
  serveHeadEvidence,
  servingSideFeatureNames,
  servingSideNearProbability,
  tiedPercentileRanks,
} from "../../prod/src/lib/on-device/serving-side-model.ts";

const artifact = JSON.parse(await readFile(
  new URL("../../prod/public/runtime/serving-side-85bc3325fbd4.json", import.meta.url),
  "utf8",
));
const runtime = parseServingSideRuntime(artifact);

test("SERVSIDE237-FLIGHT runtime preserves the frozen feature signature", () => {
  assert.equal(courtFlowFeatureNames().length, 82);
  assert.equal(flightFeatureNames().length, 155);
  assert.equal(servingSideFeatureNames().length, 237);
  assert.deepEqual(runtime.featureNames, servingSideFeatureNames());
  assert.equal(runtime.model.weights.length, 237);
  assert.equal(runtime.fingerprint, "85bc3325fbd43abba6ba3726ac091dc6a3a1eafc68d63d979cb2a7450eb49e06");
});

test("tied percentile ranks match NumPy stable midranks exactly", () => {
  const ranked = tiedPercentileRanks([
    1, 9,
    1, 2,
    3, 2,
    2, 4,
  ], 4, 2);
  assert.deepEqual(Array.from(ranked), [
    1 / 6, 1,
    1 / 6, 1 / 6,
    1, 1 / 6,
    2 / 3, 2 / 3,
  ]);
  assert.deepEqual(Array.from(tiedPercentileRanks([7, -3], 1, 2)), [0.5, 0.5]);
});

test("frozen logistic runner matches the Python score for an all-midrank row", () => {
  const score = servingSideNearProbability(new Float64Array(237).fill(0.5), runtime);
  assert.ok(Math.abs(score - 0.5428339645988896) < 1e-14);
});

test("serve evidence includes the ±1 second boundary and keeps first peak tie", () => {
  const evidence = serveHeadEvidence(
    "head",
    new Float64Array([3, 4, 5, 7]),
    {
      probabilities: new Float32Array([0.9, 0.2, 0.9, 1]),
      detections: [{ time: 4.4, confidence: 0.91 }, { time: 6, confidence: 0.99 }],
    },
    4,
    0.85,
    1,
  );
  assert.equal(evidence.peakTime, 3);
  assert.ok(Math.abs(evidence.peakProbability - new Float32Array([0.9])[0]) < 1e-12);
  assert.equal(evidence.crossesThreshold, true);
  assert.deepEqual(evidence.nearestDetection, { time: 4.4, confidence: 0.91 });
});

test("hybrid gate recovers only both-model intervals and always requests review", () => {
  const missed = [
    { modelId: "a", threshold: 0.85, peakProbability: 0.2, peakTime: 1, crossesThreshold: false, nearestDetection: null },
    { modelId: "b", threshold: 0.85, peakProbability: 0.3, peakTime: 1, crossesThreshold: false, nearestDetection: null },
  ];
  assert.deepEqual(evaluateHybridGate("both-models", missed), {
    source: "production-rally-recovery",
    reviewReasons: ["production-rally-recovery"],
    isServe: true,
  });
  assert.deepEqual(evaluateHybridGate("all-labels-v2-only", missed), {
    source: "none",
    reviewReasons: [],
    isServe: false,
  });
  assert.equal(evaluateHybridGate(undefined, [
    { ...missed[0], crossesThreshold: true }, missed[1],
  ]).source, "serve-head");
});

test("composed candidate uses merged interval start and keeps side under review", () => {
  const interval = {
    id: "R007",
    start: 12.5,
    end: 20,
    confidence: 0.8,
    included: true,
    agreement: "both-models" as const,
  };
  const lowHead = {
    probabilities: new Float32Array([0.1, 0.2, 0.3]),
    detections: [],
  };
  const result = composeServingSideVerdict(
    interval,
    new Float64Array(237).fill(0.5),
    new Float64Array([12, 12.5, 13]),
    lowHead,
    lowHead,
    runtime,
  );
  assert.equal(result.id, "R007");
  assert.equal(result.anchor, 12.5);
  assert.equal(result.side, "near");
  assert.equal(result.verdict, "review");
  assert.equal(result.serveDecisionSource, "production-rally-recovery");
  assert.deepEqual(result.reviewReasons, ["production-rally-recovery"]);
});

test("runtime parser rejects a reordered signature", () => {
  const changed = structuredClone(artifact);
  [changed.featureNames[0], changed.featureNames[1]] = [
    changed.featureNames[1], changed.featureNames[0],
  ];
  assert.throws(() => parseServingSideRuntime(changed), /feature signature/);
});

test("durable serving-side cache is bound to model, features, and candidate anchors", () => {
  const interval = {
    id: "R001",
    start: 4,
    end: 9,
    confidence: 0.9,
    included: true,
    agreement: "both-models" as const,
  };
  const head = {
    probabilities: new Float32Array([0.95]),
    detections: [{ time: 4, confidence: 0.95 }],
  };
  const candidate = composeServingSideVerdict(
    interval,
    new Float64Array(237).fill(0.5),
    new Float64Array([4]),
    head,
    head,
    runtime,
  );
  const output = {
    modelId: runtime.modelId,
    modelFingerprint: runtime.fingerprint,
    featureVersion: "SERVSIDE237-FLIGHT" as const,
    anchorContract: "merged-production-interval-start-v1" as const,
    features: { rows: 1, columns: 237, values: new Float64Array(237) },
    candidates: [candidate],
  };
  assert.equal(isReusableServingSideOutput(output, [interval]), true);
  assert.equal(
    isReusableServingSideOutput(output, [{ ...interval, start: 4.1 }]),
    false,
  );
  assert.equal(
    isReusableServingSideOutput(
      { ...output, features: { ...output.features, values: new Float64Array(236) } },
      [interval],
    ),
    false,
  );
});
