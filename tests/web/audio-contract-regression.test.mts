import assert from "node:assert/strict";
import test from "node:test";
import { AudioSample } from "mediabunny";
import * as lab from "../../lib/on-device/audio-features.ts";
import * as production from "../../prod/src/lib/on-device/audio-features.ts";

for (const [name, implementation] of [["lab", lab], ["production", production]] as const) {
  test(`${name}: integer rolling quantiles preserve the startup noise floor`, () => {
    const values = Float32Array.from({ length: 210 }, (_, i) => i + 1);
    const result = implementation.rollingPercentile(values, 200, 0.2);
    for (let i = 0; i < values.length; i++) {
      const expected = Math.max(0, i - 199) + 1 + 0.2 * (Math.min(i + 1, 200) - 1);
      assert.ok(Math.abs(result[i] - expected) < 1e-5, `frame ${i}: ${result[i]} != ${expected}`);
    }
  });

  test(`${name}: irregular first AAC timestamp preserves every subsequent PCM frame`, () => {
    let written = 0;
    let nonzero = 0;
    const resampler = {
      configure() {},
      push(planes: readonly Float32Array[]) {
        written += planes[0].length;
        for (const sample of planes[0]) if (sample !== 0) nonzero++;
        return new Int16Array(0);
      },
      flush() { return new Int16Array(0); },
      close() {},
    };
    const accumulator = new implementation.AudioAccumulator(resampler);
    const units = 2000;
    for (let i = 0; i < units; i++) {
      const sample = new AudioSample({
        data: new Float32Array(1024).fill(0.5), format: "f32",
        numberOfChannels: 1, sampleRate: 48000,
        timestamp: i === 0 ? 0 : Math.round((82 + i * 1024) * 1e6 / 48000) / 1e6,
      });
      try { accumulator.push(sample); } finally { sample.close(); }
    }
    accumulator.finish();
    assert.equal(nonzero, units * 1024);
    assert.equal(written, units * 1024 + 82);
  });
}
