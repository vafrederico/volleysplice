import assert from "node:assert/strict";
import test from "node:test";

import {
  finiteQuantile,
  finiteQuantileInPlace,
  quantile,
} from "../../lib/on-device/feature-math.ts";

test("finite float32 selection matches the existing sorted quantile", () => {
  const cases = [
    new Float32Array(),
    new Float32Array([7]),
    new Float32Array([3, 1, 2, 2, -4, 10]),
    new Float32Array(257).fill(4.5),
  ];
  let state = 0x1234_5678;
  const random = new Float32Array(192 * 108);
  for (let index = 0; index < random.length; index += 1) {
    state = (Math.imul(1_664_525, state) + 1_013_904_223) >>> 0;
    random[index] = ((state >>> 8) / 0x1_000000 - 0.5) * 40;
  }
  cases.push(random);

  for (const values of cases) {
    for (const percentile of [-1, 0, 0.1, 0.5, 0.9, 1, 2]) {
      const expected = quantile(values, percentile);
      assert.equal(finiteQuantile(values, percentile), expected);
      assert.equal(finiteQuantileInPlace(values.slice(), percentile), expected);
    }
  }
});

test("non-mutating finite quantile preserves pixel order", () => {
  const values = new Float32Array([5, 1, 4, 2, 3]);
  const before = values.slice();
  assert.equal(finiteQuantile(values, 0.5), 3);
  assert.deepEqual(values, before);
});
