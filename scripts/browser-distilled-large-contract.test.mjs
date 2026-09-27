import test from 'node:test';
import assert from 'node:assert/strict';
import { geometry, roundEven, regionalPoolWeights, normalizedLetterbox, roundTokensToFloat16,
  fuseFeatures, decodeRallies, FUSED_DIMENSION, TOKEN_DIMENSION } from './browser-distilled-large-contract.mjs';

test('ROI edge rounding and letterbox match the declared image coordinates', () => {
  assert.deepEqual([.5, 1.5, 2.5, 3.5, -.5, -1.5].map(roundEven), [0, 2, 2, 4, 0, -2]);
  const full = geometry(1920, 1080);
  assert.deepEqual(full.box, [0, 49 / 224, 1, 175 / 224]);
  assert.equal(full.resizedWidth, 224); assert.equal(full.resizedHeight, 126);
  const fractional = geometry(640, 360, [2.5 / 640, 4.5 / 360, 512 / 640, 280 / 360]);
  assert.deepEqual([fractional.x, fractional.y, fractional.cropWidth, fractional.cropHeight], [2, 4, 512, 280]);
  assert.throws(() => geometry(640, 360, [0, 0, 0, 1]), /Invalid/);
});

test('ImageNet normalization preserves channel order and black padding', () => {
  const shape = geometry(2, 1);
  const rgb = new Uint8Array(shape.resizedWidth * shape.resizedHeight * 3);
  for (let i = 0; i < rgb.length; i += 3) { rgb[i] = 255; rgb[i + 1] = 128; rgb[i + 2] = 0; }
  const image = normalizedLetterbox(rgb, shape), plane = 224 * 224, content = shape.top * 224;
  assert.equal(image.length, plane * 3);
  assert.equal(image[0], Math.fround(-Math.fround(.485) / Math.fround(.229)));
  assert.equal(image[content], Math.fround(Math.fround(1 - Math.fround(.485)) / Math.fround(.229)));
  assert.ok(image[plane + content] > 0 && image[plane + content] < .3);
  assert.ok(image[2 * plane + content] < -1);
});

test('regional pools exclude letterbox padding and normalize independently', () => {
  const weights = regionalPoolWeights(geometry(1920, 1080).box);
  for (let region = 0; region < 4; region++) {
    const values = weights.subarray(region * 49, (region + 1) * 49);
    assert.ok(Math.abs(values.reduce((a, b) => a + b) - 1) < 1e-7);
    assert.ok(values.every(v => v >= 0));
    assert.ok(values.subarray(0, 7).every(v => v === 0));
    assert.ok(values.subarray(42).every(v => v === 0));
  }
  assert.ok(weights[49 + 5 * 7] > weights[49 + 1 * 7]);
  assert.ok(weights[98 + 1 * 7] > weights[98 + 5 * 7]);
});

test('float16 cache rounding covers ties, subnormals, carry and finite rejection', () => {
  const values = [0, -0, 1, 1 + 2 ** -11, 1 + 3 * 2 ** -11, 2 ** -25,
    3 * 2 ** -25, 2 ** -14 - 2 ** -25, 65504];
  assert.deepEqual(Array.from(roundTokensToFloat16(values)), [0, -0, 1, 1, 1 + 2 ** -9,
    0, 2 ** -23, 2 ** -14, 65504]);
  for (const invalid of [65520, NaN, Infinity]) assert.throws(() => roundTokensToFloat16([invalid]), /finite float16/);
  // Every finite half value must survive a float32 -> half cache roundtrip.
  const all = [];
  for (let sign = 0; sign < 2; sign++) for (let exponent = 0; exponent < 31; exponent++)
    for (let mantissa = 0; mantissa < 1024; mantissa++)
      all.push((sign ? -1 : 1) * (exponent ? (1 + mantissa / 1024) * 2 ** (exponent - 15) : mantissa * 2 ** -24));
  assert.deepEqual(Array.from(roundTokensToFloat16(all)), all);
});

test('fusion uses causal nominal-grid hold, quality age and separate scalar scaling', () => {
  const times = new Float64Array([0, .25, .5, 1.25]);
  const av = new Float32Array(times.length * 104).fill(.75);
  const tokens = new Float32Array(TOKEN_DIMENSION * 2); tokens.fill(3, 0, TOKEN_DIMENSION); tokens.fill(7, TOKEN_DIMENSION);
  const quality = new Float32Array([.5, .4, .2, .1, 0, -.01, .6, .5, .3, .2, 0, .01]);
  const config = { mean: Array(112).fill(.5), scale: Array(112).fill(.25) };
  const fused = fuseFeatures(times, av, new Float64Array([0, .5]), tokens, quality, config);
  assert.equal(fused[0], 1); assert.equal(fused[104], 3);
  assert.equal(fused[FUSED_DIMENSION + 104], 3); assert.equal(fused[2 * FUSED_DIMENSION + 104], 7);
  assert.equal(fused[3 * FUSED_DIMENSION + 104], 0);
  assert.equal(fused[FUSED_DIMENSION + FUSED_DIMENSION - 2], -1);
  assert.equal(fused[FUSED_DIMENSION - 1], 2);
  assert.equal(fused[4 * FUSED_DIMENSION - 1], -2);
});

test('decoder bridges interior gaps, keeps strong short events, and snaps boundaries', () => {
  const times = Float64Array.from({ length: 16 }, (_, i) => i / 4);
  const scores = new Float32Array(16 * 4);
  for (const i of [2, 3, 6, 7]) scores[i * 4] = .8;
  scores[12 * 4] = .99;
  scores[1 * 4 + 1] = .9; scores[8 * 4 + 2] = .95;
  const output = decodeRallies(times, scores, 4, { smoothing: 0, enter: .5, minimum: 1, boundary: true });
  assert.deepEqual(output.map(v => [v.start, v.end]), [[.25, 2], [2.875, 3.125]]);
  const short = decodeRallies(times, new Float32Array(16 * 4), 4, { smoothing: .5, enter: .2, minimum: 1, boundary: true });
  assert.deepEqual(short, []);
});

test('decoder preserves NumPy float32 threshold promotion and double short-event guard', () => {
  const times = new Float64Array([0, .25, .5, .75]), scores = new Float32Array(16);
  scores[4] = .9; scores[5] = .65; scores[10] = .65;
  assert.deepEqual(decodeRallies(times, scores, 1,
    { smoothing: 0, enter: .9, minimum: .25, boundary: true }).map(v => [v.start, v.end]), [[.25, .5]]);
  assert.deepEqual(decodeRallies(times, scores, 1,
    { smoothing: 0, enter: .5, minimum: 1, boundary: false }), []);
});
