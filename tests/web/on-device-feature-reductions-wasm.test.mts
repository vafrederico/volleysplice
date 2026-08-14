import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { ANALYSIS_HEIGHT, ANALYSIS_WIDTH, FRAME_FEATURE_NAMES } from "../../lib/on-device/feature-schema.ts";
import { computeVisualFeatureReductionsJavaScript } from "../../lib/on-device/visual-feature-reductions.ts";
import { WasmVisualFeatureReducer } from "../../lib/on-device/visual-feature-reductions-wasm.ts";

const pixels = ANALYSIS_WIDTH * ANALYSIS_HEIGHT;

function fixture(hasPrevious: boolean) {
  let state = 0x9e3779b9;
  const random = () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 0x1_0000_0000;
  };
  const gray = Uint8Array.from({ length: pixels }, () => Math.floor(random() * 256));
  const hsv = Uint8Array.from({ length: pixels * 3 }, () => Math.floor(random() * 256));
  const edges = Uint8Array.from({ length: pixels }, () => (random() > 0.78 ? 255 : 0));
  const laplacian = Float32Array.from({ length: pixels }, () => (random() - 0.5) * 80);
  const gradientX = Float32Array.from({ length: pixels }, () => (random() - 0.5) * 32);
  const gradientY = Float32Array.from({ length: pixels }, () => (random() - 0.5) * 32);
  const previousGray = hasPrevious
    ? Uint8Array.from(gray, (value) => Math.max(0, Math.min(255, value + Math.floor(random() * 21) - 10)))
    : null;
  const flow = Float32Array.from({ length: pixels * 2 }, () => (random() - 0.5) * 8);
  return {
    gray,
    hsv,
    edges,
    laplacian,
    gradientX,
    gradientY,
    previousGray,
    flow,
    shiftX: hasPrevious ? 2.375 : 0,
    shiftY: hasPrevious ? -1.625 : 0,
    shiftResponse: hasPrevious ? 0.81 : 0,
  };
}

const wasmBytes = await readFile(
  new URL("../../public/on-device/feature-reductions.wasm", import.meta.url),
);

for (const hasPrevious of [false, true]) {
  test(`WASM reductions preserve all JavaScript features (${hasPrevious ? "temporal" : "first"} frame)`, async () => {
    const input = fixture(hasPrevious);
    const expected = computeVisualFeatureReductionsJavaScript(input);
    const reducer = await WasmVisualFeatureReducer.instantiate(wasmBytes);
    const actual = reducer.reduce(input);
    assert.equal(actual.length, FRAME_FEATURE_NAMES.length);
    for (let index = 0; index < actual.length; index += 1) {
      const tolerance = 2e-5 + Math.abs(expected[index]) * 1e-5;
      assert.ok(
        Math.abs(actual[index] - expected[index]) <= tolerance,
        `${FRAME_FEATURE_NAMES[index]} differs: JS=${expected[index]} WASM=${actual[index]}`,
      );
    }
  });
}
