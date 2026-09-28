/** Public synthetic graph oracle for iOS native ORT. No videos, private inputs or training.
 * Run: node ios/scripts/generate-neural-runtime-golden.mjs
 * Uses the existing browser WASM dependency as an independent execution backend.
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const require = createRequire(new URL("../../prod/package.json", import.meta.url));
const ort = require("onnxruntime-web");
ort.env.wasm.numThreads = 1;
ort.env.logLevel = "warning";
const root = new URL("../../", import.meta.url);
const manifest = JSON.parse(await readFile(new URL("models/distilled-large/ios-manifest.json", root), "utf8"));
const dimensions = { imageSize: 224, contentWidth: 224, contentHeight: 126, left: 0, top: 49,
  featureRows: 301, featureColumns: 3952, chunkRows: 128, contextRows: 62 };
const sha = bytes => createHash("sha256").update(bytes).digest("hex");
function floatBytes(values) {
  const bytes = Buffer.alloc(values.length * 4);
  for (let i = 0; i < values.length; i++) bytes.writeFloatLE(values[i], i * 4);
  return bytes;
}
function encoded(values) {
  const bytes = floatBytes(values);
  return { encoding: "base64", byteOrder: "little-endian", dataType: "float32", count: values.length,
    sha256: sha(bytes), data: bytes.toString("base64") };
}
// Image is fractional RGB, including a true black letterbox. These formulas are
// repeated in the native test, with checksums to expose preprocessing differences.
const image = new Float32Array(3 * 224 * 224);
const mean = new Float32Array([.485, .456, .406]), scale = new Float32Array([.229, .224, .225]);
for (let channel = 0; channel < 3; channel++) for (let y = 0; y < 224; y++) for (let x = 0; x < 224; x++) {
  const rgb = y >= 49 && y < 175 ? ((x * 17 + (y - 49) * 29 + channel * 41) % 1021) / 4 : 0;
  const unit = Math.fround(rgb / 255);
  image[channel * 224 * 224 + y * 224 + x] = Math.fround(Math.fround(unit - mean[channel]) / scale[channel]);
}
const pool = new Float32Array(196), top = 49 / 224, bottom = 175 / 224;
for (const [region, [low, high]] of [[0, 1], [.5, 1], [0, .5], [.4, .6]].entries()) {
  const a = top + low * (bottom - top), b = top + high * (bottom - top);
  const area = new Float64Array(49);
  let total = 0;
  for (let y = 0; y < 7; y++) for (let x = 0; x < 7; x++) {
    area[y * 7 + x] = Math.max(0, Math.min((x + 1) / 7, 1) - Math.max(x / 7, 0))
      * Math.max(0, Math.min((y + 1) / 7, b) - Math.max(y / 7, a));
    total += area[y * 7 + x];
  }
  for (let i = 0; i < 49; i++) pool[region * 49 + i] = area[i] / total;
}
const features = new Float32Array(301 * 3952);
for (let row = 0; row < 301; row++) for (let column = 0; column < 3952; column++) {
  features[row * 3952 + column] = ((row * 31 + column * 17 + Math.floor(column / 104) * 7) % 257 - 128) / 128;
}
const result = { schemaVersion: 1, description: "Public synthetic inputs; graph numerical parity only, not model-accuracy evidence.",
  reference: { backend: "onnxruntime-web/wasm", version: ort.env.versions.web, threads: 1 }, dimensions,
  inputSha256: { image: sha(floatBytes(image)), pool: sha(floatBytes(pool)), features: sha(floatBytes(features)) },
  tolerance: { absolute: 1e-4, relative: 1e-4 }, variants: {} };

async function session(bundle, asset) {
  const bytes = await readFile(new URL(`prod/public/runtime/rally-models/${bundle.directory}/${asset.name}`, root));
  assert.equal(bytes.length, asset.sizeBytes);
  assert.equal(sha(bytes), asset.sha256);
  return ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
}
function tensor(values, shape) { return new ort.Tensor("float32", values, shape); }
function values(output, expectedShape) {
  assert.deepEqual(output.dims, expectedShape);
  assert.ok(output.data instanceof Float32Array && output.data.every(Number.isFinite));
  return new Float32Array(output.data);
}
for (const key of ["high-recall", "high-f1"]) {
  const bundle = manifest.variants[key];
  const encoder = await session(bundle, bundle.files.encoder);
  const encoderOutput = await encoder.run({ image: tensor(image, [1, 3, 224, 224]), pool_weights: tensor(pool, [1, 4, 7, 7]) });
  const tokens = values(encoderOutput.tokens, [1, 4, 960]);
  for (const output of Object.values(encoderOutput)) output.dispose();
  await encoder.release();
  const temporal = await session(bundle, bundle.files.temporal);
  const full = await temporal.run({ features: tensor(features, [1, 301, 3952]) });
  const logits = values(full.logits, [1, 301, 4]);
  for (const output of Object.values(full)) output.dispose();
  const chunked = new Float32Array(logits.length);
  for (let core = 0; core < 301; core += 128) {
    const end = Math.min(301, core + 128), left = Math.max(0, core - 62), right = Math.min(301, end + 62);
    const output = await temporal.run({ features: tensor(features.slice(left * 3952, right * 3952), [1, right - left, 3952]) });
    const rows = values(output.logits, [1, right - left, 4]);
    chunked.set(rows.subarray((core - left) * 4, (end - left) * 4), core * 4);
    for (const value of Object.values(output)) value.dispose();
  }
  await temporal.release();
  let maximumChunkAbsoluteError = 0;
  for (let i = 0; i < logits.length; i++) {
    const error = Math.abs(logits[i] - chunked[i]);
    maximumChunkAbsoluteError = Math.max(maximumChunkAbsoluteError, error);
    assert.ok(error <= result.tolerance.absolute + result.tolerance.relative * Math.abs(logits[i]), `Temporal seam mismatch at ${i}`);
  }
  result.variants[key] = { modelId: bundle.id, encoderSha256: bundle.files.encoder.sha256, temporalSha256: bundle.files.temporal.sha256,
    tokens: encoded(tokens), logits: encoded(logits), maximumChunkAbsoluteError };
  console.log(`${key}: ${tokens.length} encoder tokens, ${logits.length} temporal logits; chunk max error ${maximumChunkAbsoluteError}`);
}
const output = new URL("../Tests/Fixtures/neural-runtime-golden.json", import.meta.url);
await writeFile(output, JSON.stringify(result, null, 2) + "\n");
console.log(`Wrote ${fileURLToPath(output).split(/[/\\]/).slice(-4).join("/")}`);
