import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  decodeSideSwitchCandidates,
  generateSideSwitchCandidates,
  parseSideSwitchRuntime,
  predictSideSwitchProbabilities,
  SIDE_SWITCH_FEATURE_NAMES,
  sideSwitchStateFeatures,
} from "../../prod/src/lib/on-device/side-switch-model.ts";

const runtime = parseSideSwitchRuntime(
  JSON.parse(
    readFileSync(
      new URL(
        "../../prod/public/runtime/side-switch-c2570481c30d.json",
        import.meta.url,
      ),
      "utf8",
    ),
  ),
);

test("browser side-switch runtime retains the frozen model identity and feature order", () => {
  assert.equal(runtime.classifier.featureNames.length, 34);
  assert.deepEqual(runtime.classifier.featureNames, [
    ...SIDE_SWITCH_FEATURE_NAMES,
  ]);
  assert.equal(runtime.classifier.threshold, 0.39884973953581804);
  assert.equal(runtime.decoder.minimumCandidateIndexSeparation, 2);
  assert.equal(runtime.decoder.freePredictionsPerRecording, 6);
  assert.equal(runtime.decoder.countPenaltyLogitPerExcessPrediction, 0.5);
});

test("browser classifier matches a frozen Python full-union feature row", () => {
  const row = Float64Array.of(
    0.08008777,
    0.09610611,
    -0.12597968,
    0.0886536,
    0.00180008,
    0.52332121,
    0.13456592,
    0.15405941,
    -0.01949349,
    -0.80088749,
    0.10861685,
    0.02875743,
    0.35565329,
    0.24777471,
    0.1214941,
    0.04468161,
    0.0758231,
    3.28571429,
    2.14285714,
    0.77322658,
    0.2071726,
    0.03920164,
    2,
    2,
    2,
    0.99827433,
    0,
    0.12110786,
    0.88879615,
    0.78783953,
    0.99970311,
    7,
    0,
    0,
  );
  const probability = predictSideSwitchProbabilities(runtime, row)[0];
  assert.ok(Math.abs(probability - 0.20205971061674935) < 1e-14);
});

test("candidate union keeps adjacent boundaries and score-ranked separated dead peaks", () => {
  const times = Float64Array.from({ length: 121 }, (_, index) => index / 4);
  const dead = new Float32Array(times.length);
  dead[8] = 1; // too close to the range edge
  dead[40] = 0.99;
  dead[48] = 1; // suppresses the nearby lower score
  dead[108] = 1; // too close to the range edge
  const candidates = generateSideSwitchCandidates(
    {
      intervals: [
        { id: "R001", start: 0, end: 30, confidence: 1, included: true },
        { id: "R002", start: 35, end: 40, confidence: 1, included: true },
      ],
      times,
      deadStateProbabilities: dead,
    },
    runtime,
  );
  assert.deepEqual(
    candidates.map((candidate) => [candidate.kind, candidate.transitionTime]),
    [
      ["internal-dead-state-peak", 12],
      ["adjacent-rally-boundary", 32.5],
    ],
  );
  assert.deepEqual(candidates[0].beforeWindow, { start: 8, end: 11 });
  assert.deepEqual(candidates[0].afterWindow, { start: 13, end: 16 });
});

test("decoder applies chronological index NMS and the post-six logit cost", () => {
  const candidates = Array.from({ length: 14 }, (_, index) => ({
    id: `candidate-${String(index).padStart(2, "0")}`,
    kind: "adjacent-rally-boundary" as const,
    gapStart: index * 10,
    gapEnd: index * 10 + 2,
    transitionTime: index * 10 + 1,
    generatorScore: 0,
    sourceRangeIds: [`R${index}`, `R${index + 1}`],
    beforeWindow: { start: index * 10 - 4, end: index * 10 - 1 },
    afterWindow: { start: index * 10 + 1, end: index * 10 + 4 },
  }));
  const selected = decodeSideSwitchCandidates(
    runtime,
    candidates,
    Float64Array.from({ length: candidates.length }, () => 0.5),
  );
  assert.deepEqual(selected, [0, 2, 4, 6, 8, 10]);
});

test("state features pool both production bundles without suppression inputs", () => {
  const ranges = [
    { id: "R001", start: 0, end: 2, confidence: 1, included: true },
    { id: "R002", start: 4, end: 6, confidence: 1, included: true },
  ];
  const candidate = generateSideSwitchCandidates(
    {
      intervals: ranges,
      times: new Float64Array([0, 1, 2, 3, 4, 5, 6]),
      deadStateProbabilities: new Float32Array(7),
    },
    runtime,
  )[0];
  const state = sideSwitchStateFeatures(candidate, {
    intervals: ranges,
    times: new Float64Array([0, 1, 2, 3, 4, 5, 6]),
    deadStateProbabilities: new Float32Array(7),
    productionComponents: {
      allLabelsV2: ranges,
      previousProduction: ranges,
    },
    productionStateOutputs: {
      allLabelsV2: {
        rallyProbabilities: new Float32Array([
          0.8, 0.9, 0.2, 0.1, 0.7, 0.8, 0.1,
        ]),
        deadStateProbabilities: new Float32Array([0, 0, 0.8, 0.9, 0, 0, 0]),
      },
      previousProduction: {
        rallyProbabilities: new Float32Array([
          0.7, 0.85, 0.3, 0.2, 0.6, 0.75, 0.1,
        ]),
        deadStateProbabilities: new Float32Array([0, 0, 0.7, 0.95, 0, 0, 0]),
      },
    },
  });
  assert.deepEqual(Array.from(state.slice(0, 5)), [2, 2, 2, 0.75, 0]);
  assert.ok(Math.abs(state[5] - 0.25) < 1e-6);
  assert.ok(Math.abs(state[6] - 0.3) < 1e-6);
  assert.ok(Math.abs(state[7] - 0.875) < 1e-6);
  assert.ok(Math.abs(state[8] - 0.95) < 1e-6);
  assert.equal(state[9], 2);
});
