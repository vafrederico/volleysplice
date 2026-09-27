import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { parseNeuralManifest, parseNeuralConfig, DEFAULT_RALLY_MODEL } from "../../prod/src/lib/on-device/rally-model.ts";
import { neuralEmbeddingCacheKey } from "../../prod/src/lib/on-device/neural-cache.ts";
import { decodeRallies, fuseFeatures, neuralEmbeddingTimes, roundTokensToFloat16, type NeuralPipelineConfig } from "../../prod/src/lib/on-device/neural-contract.ts";
import * as reference from "../../scripts/browser-distilled-large-contract.mjs";
import { productionModelAgreementLabel, isProductionModelDisagreement } from "../../lib/production-ensemble.ts";
import type { OnDeviceMediaInfo } from "../../prod/src/lib/on-device/types.ts";

const manifest = parseNeuralManifest(JSON.parse(readFileSync(new URL("../../models/distilled-large/web-manifest.json", import.meta.url), "utf8")));
const config: NeuralPipelineConfig = { modelIdentity: "dino-distilled-mobilenet-v3-large-tcn", selectionMode: "recall", recallTargetPercent: 99,
  tokenDimension: 3840, mean: Array(112).fill(.2), scale: Array(112).fill(.7), decoder: { enter: .2, smoothing: .5, minimum: 1, boundary: true } };

test("recall is the default and swapping requires the matching encoder, temporal model and scalers", () => {
  assert.equal(DEFAULT_RALLY_MODEL, "high-recall");
  assert.notEqual(manifest.variants["high-recall"].files.encoder.sha256, manifest.variants["high-f1"].files.encoder.sha256);
  assert.notEqual(manifest.variants["high-recall"].files.temporal.sha256, manifest.variants["high-f1"].files.temporal.sha256);
  assert.notEqual(manifest.variants["high-recall"].files.pipeline.sha256, manifest.variants["high-f1"].files.pipeline.sha256);
  assert.equal(parseNeuralConfig(config, "high-recall"), config);
  assert.throws(() => parseNeuralConfig(config, "high-f1"), /incompatible/);
  assert.throws(() => parseNeuralConfig({ ...config, scale: Array(112).fill(0) }, "high-recall"), /incompatible/);
  assert.throws(() => parseNeuralManifest({ ...manifest, variants: { ...manifest.variants, "high-f1": manifest.variants["high-recall"] } }), /identity/);
});

test("fractional game windows include the preceding causal embedding and exclude the end", () => {
  assert.deepEqual(Array.from(neuralEmbeddingTimes(120, 3.2, 5)), [3, 3.5, 4, 4.5]);
  assert.deepEqual(Array.from(neuralEmbeddingTimes(120, 3.5, 5.01)), [3.5, 4, 4.5, 5]);
  assert.equal(neuralEmbeddingTimes(120, 0, 120).length, 240);
  assert.throws(() => neuralEmbeddingTimes(120, 10, 9), /window/);
});

test("browser rounding and chunked fusion retain the qualified numerical contract", () => {
  const tokens = roundTokensToFloat16(Float32Array.from({ length: 6 * 3840 }, (_, i) => Math.sin(i) * 3));
  assert.deepEqual(tokens, reference.roundTokensToFloat16(Float32Array.from({ length: 6 * 3840 }, (_, i) => Math.sin(i) * 3)));
  const embeddingTimes = neuralEmbeddingTimes(3, 0, 3);
  const times = Float64Array.from({ length: 12 }, (_, i) => i / 4);
  const av = Float32Array.from({ length: times.length * 104 }, (_, i) => i % 97 / 97);
  const quality = Float32Array.from({ length: embeddingTimes.length * 6 }, (_, i) => i % 6 / 10);
  const whole = reference.fuseFeatures(times, av, embeddingTimes, tokens, quality, config);
  for (const [left, right] of [[0, 5], [5, 12]]) {
    const chunk = fuseFeatures(times.slice(left, right), av.slice(left * 104, right * 104), embeddingTimes, tokens, quality, config);
    assert.deepEqual(chunk, whole.slice(left * 3952, right * 3952));
  }
  const probabilities = Float32Array.from({ length: times.length * 4 }, (_, i) => i % 4 === 0 ? (i >= 8 && i < 32 ? .8 : .02) : .1);
  assert.deepEqual(decodeRallies(times, probabilities, 3, config.decoder), reference.decodeRallies(times, probabilities, 3, config.decoder));
});

test("embedding cache is isolated by matched variant, preprocessing, crop and game window", () => {
  const source = { name: "synthetic.mp4", size: 1024, lastModified: 1 };
  const info = { duration: 120, width: 1920, height: 1080, rotation: 0, videoCodecString: "avc1" } as OnDeviceMediaInfo;
  const roi = { x: 0, y: 0, width: 1, height: 1 }, window = { start: 0, end: 120 };
  const recall = manifest.variants["high-recall"];
  const key = neuralEmbeddingCacheKey(source, info, roi, window, recall);
  assert.notEqual(key, neuralEmbeddingCacheKey(source, info, roi, window, manifest.variants["high-f1"]));
  assert.notEqual(key, neuralEmbeddingCacheKey(source, info, roi, { start: 3.2, end: 120 }, recall));
  assert.notEqual(key, neuralEmbeddingCacheKey(source, info, { ...roi, width: .8 }, window, recall));
  assert.match(key, /rgb-linear-letterbox224-nearest2hz-f16-v1/);
});


test("neural provenance is neither ensemble agreement nor ensemble disagreement", () => {
  assert.equal(productionModelAgreementLabel("neural"), "Neural model prediction");
  assert.equal(isProductionModelDisagreement({ agreement: "neural" }), false);
});


test("neural provenance survives both root and editor-lab draft restoration", async () => {
  for (const editor of [await import("../../lib/cut-draft.ts"), await import("../../components/production-lab/editor/lib/cut-draft.ts")]) {
    const seed = { analysisId: "neural-test", recordingId: "fixture", duration: 10,
      rallies: [{ id: "N001", start: 2, end: 5, confidence: .9, included: true, agreement: "neural" as const }], ignoredIntervals: [] };
    const draft = editor.createCutDraft(seed);
    const restored = editor.parseCutDraft(JSON.stringify(draft), seed);
    assert.equal(restored?.cuts[0].agreement, "neural");
  }
});
