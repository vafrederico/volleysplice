import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { NEURAL_SERVING_MANIFEST, applyNeuralServingDecision, neuralServingBundle, neuralServingDecision,
  neuralServingSelection, parseNeuralServingManifest, servingEmbeddingTimes, servingEmbeddingWindow } from "../../prod/src/lib/on-device/neural-serving-model.ts";
import { neuralServingFeatures } from "../../prod/src/lib/on-device/neural-serving-features.ts";
import { composeServingSideVerdict, parseServingSideRuntime, servingSideFeatureNames } from "../../prod/src/lib/on-device/serving-side-model.ts";
import { isReusableServingSideOutput } from "../../prod/src/lib/on-device/serving-side-cache.ts";
import { parseNeuralManifest } from "../../prod/src/lib/on-device/rally-model.ts";
import type { OnDeviceServingSideOutput } from "../../prod/src/lib/on-device/types.ts";

const rally = parseNeuralManifest(JSON.parse(await readFile(new URL("../../models/distilled-large/web-manifest.json", import.meta.url), "utf8")));
const runtime = parseServingSideRuntime(JSON.parse(await readFile(new URL("../../prod/public/runtime/serving-side-85bc3325fbd4.json", import.meta.url), "utf8")));
const interval = { id: "N001", start: 4, end: 8, confidence: .9, included: true, agreement: "neural" as const };
const serveHead = { probabilities: new Float32Array([.99]), detections: [{ time: 4, confidence: .99 }] };
const candidate = composeServingSideVerdict(interval, new Float64Array(237).fill(.5), [4], serveHead, serveHead, runtime);
const logit = (probability: number) => Math.log(probability / (1 - probability));
const nearLogits = Array.from({ length: 3 }, () => [logit(.01), logit(.7)]).flat();

test("both serving bundles match their rally encoder and frozen three-seed weights", async () => {
  assert.equal(neuralServingSelection("ensemble"), null);
  assert.equal(neuralServingSelection(undefined), null);
  for (const selection of ["high-recall", "high-f1"] as const) {
    const bundle = parseNeuralServingManifest(NEURAL_SERVING_MANIFEST, rally.variants[selection]);
    assert.equal(neuralServingSelection(rally.variants[selection].id), selection);
    const bytes = await readFile(new URL(`../../prod/public/runtime/serving-models/${bundle.directory}/${bundle.file.name}`, import.meta.url));
    assert.equal(bytes.byteLength, bundle.file.sizeBytes);
    assert.equal(createHash("sha256").update(bytes).digest("hex"), bundle.file.sha256);
    assert.deepEqual(bundle.seeds.map(seed => seed.seed), [3407, 1729, 20260918]);
    assert.throws(() => parseNeuralServingManifest(NEURAL_SERVING_MANIFEST,
      { ...rally.variants[selection], files: { ...rally.variants[selection].files,
        encoder: rally.variants[selection === "high-recall" ? "high-f1" : "high-recall"].files.encoder } }), /encoder/);
  }
  const changed = structuredClone(NEURAL_SERVING_MANIFEST);
  changed.variants["high-f1"].seeds[0].nearCutoff = 0;
  assert.throws(() => parseNeuralServingManifest(changed, rally.variants["high-f1"]), /catalog/);
});

test("three-seed agreement respects encoder-specific cutoffs and vetoes disagreement", () => {
  const coverage = neuralServingBundle(rally.variants["high-recall"].id)!;
  const balanced = neuralServingBundle(rally.variants["high-f1"].id)!;
  assert.equal(neuralServingDecision(nearLogits, coverage).accepted, true);
  assert.equal(neuralServingDecision(nearLogits, balanced).side, "near");
  const marginal = Array.from({ length: 3 }, () => [logit(.01), logit(.46)]).flat();
  assert.equal(neuralServingDecision(marginal, coverage).accepted, true);
  assert.equal(neuralServingDecision(marginal, balanced).accepted, false);
  assert.equal(neuralServingDecision([...nearLogits.slice(0, 4), logit(.7), logit(.01)], coverage).accepted, false);
  assert.equal(neuralServingDecision(Array(6).fill(logit(.01)), coverage).accepted, false);
  assert.throws(() => neuralServingDecision([NaN, ...nearLogits.slice(1)], coverage), /scores/);
  assert.throws(() => neuralServingDecision(Array(6).fill(logit(.9)), coverage), /mass/);
});

test("new side evidence never clears a separate serve/no-serve review", () => {
  const decision = neuralServingDecision(nearLogits, neuralServingBundle(rally.variants["high-recall"].id)!);
  assert.equal(applyNeuralServingDecision({ ...candidate, verdict: "review", reviewReasons: ["side-score"] }, decision).verdict, "near");
  const recovered = applyNeuralServingDecision({ ...candidate, verdict: "review", reviewReasons: ["neural-rally-recovery", "side-score"] }, decision);
  assert.equal(recovered.verdict, "review");
  assert.deepEqual(recovered.reviewReasons, ["neural-rally-recovery"]);
  assert.equal(applyNeuralServingDecision({ ...candidate, verdict: "not-serve" }, decision).verdict, "not-serve");
});

test("restored serving output cannot cross encoders or retain legacy or stale scores", () => {
  const coverage = neuralServingBundle(rally.variants["high-recall"].id)!;
  const output: OnDeviceServingSideOutput = { modelId: coverage.id, modelFingerprint: coverage.fingerprint,
    featureVersion: "SERVSIDE237-FLIGHT", anchorContract: "merged-production-interval-start-v1",
    features: { rows: 1, columns: 237, values: new Float64Array(237) }, candidates: [candidate] };
  assert.equal(isReusableServingSideOutput(output, [interval], rally.variants["high-recall"].id), true);
  assert.equal(isReusableServingSideOutput(output, [interval], rally.variants["high-f1"].id), false);
  assert.equal(isReusableServingSideOutput(output, [interval], "ensemble"), false);
  assert.equal(isReusableServingSideOutput({ ...output, modelFingerprint: "stale" }, [interval], rally.variants["high-recall"].id), false);
  assert.equal(isReusableServingSideOutput(output, [{ ...interval, start: 4.1 }], rally.variants["high-recall"].id), false);
});

test("serving sampling matches an independent nearest-grid search and masks video edges", () => {
  const duration = 10.2, times = Float64Array.from({ length: 21 }, (_, i) => i / 2);
  const tokens = Float32Array.from({ length: times.length * 3840 }, (_, i) => Math.floor(i / 3840));
  for (const anchor of [0, .1, .25, .5, 1.25, 3.13, 4.25, 7.999, 10.1]) {
    const sampled = servingEmbeddingWindow(times, tokens, anchor, duration);
    const sparseTimes = servingEmbeddingTimes([anchor], duration);
    const sparseTokens = Float32Array.from({ length: sparseTimes.length * 3840 }, (_, i) => sparseTimes[Math.floor(i / 3840)] * 2);
    assert.deepEqual(servingEmbeddingWindow(sparseTimes, sparseTokens, anchor, duration), sampled);
    for (let frame = 0; frame < 13; frame++) {
      const requested = anchor - 3 + frame / 2;
      const nearest = Array.from(times).reduce((best, time) => Math.abs(time - requested) < Math.abs(best - requested) ? time : best);
      const valid = requested >= 0 && requested < duration && Math.abs(nearest - requested) <= .250001;
      assert.equal(sampled[frame * 3843 + 3840], Number(valid));
      assert.equal(sampled[frame * 3843], valid ? nearest * 2 : 0);
      assert.equal(sampled[frame * 3843 + 3841], Math.fround((frame - 6) / 6));
    }
  }
  assert.throws(() => servingEmbeddingWindow(times, tokens, -.1, duration), /anchor/);
  assert.throws(() => servingEmbeddingWindow(times, tokens, duration, duration), /anchor/);
});

test("neural features preserve recording ranks and unranked absolute motion quality", () => {
  const raw = Float64Array.from({ length: 474 }, (_, i) => i < 237 ? -2 : 4);
  const features = neuralServingFeatures(raw, 2);
  assert.deepEqual(Array.from(features.slice(0, 237)), Array(237).fill(0));
  assert.deepEqual(Array.from(features.slice(249, 486)), Array(237).fill(1));
  assert.deepEqual(Array.from(features.slice(237, 249)), [-2, -2, -2, 2, -2, -2, -2, 2, -2, -2, -2, 2]);
  assert.deepEqual(Array.from(features.slice(486)), Array(12).fill(4));
  assert.equal(servingSideFeatureNames().length, 237);
  assert.throws(() => neuralServingFeatures(new Float64Array([NaN]), 1), /features/);
});
