// Pure frozen-model contract helpers shared by the isolated browser experiment
// and its tests. No labels, model selection, or private input resolution occurs here.
export const TOKEN_DIMENSION = 3840;
export const FUSED_DIMENSION = 3952;
export const INPUT_SIZE = 224;
const mean = new Float32Array([.485, .456, .406]);
const std = new Float32Array([.229, .224, .225]);

export function roundEven(value) {
  const floor = Math.floor(value), fraction = value - floor;
  return fraction === .5 ? floor + (floor % 2 !== 0 ? 1 : 0) : Math.round(value);
}

export function geometry(width, height, roi = [0, 0, 1, 1]) {
  if (![width, height].every(v => Number.isInteger(v) && v > 0)
      || roi.length !== 4 || !roi.every(Number.isFinite)
      || roi[0] < 0 || roi[1] < 0 || roi[2] <= 0 || roi[3] <= 0
      || roi[0] + roi[2] > 1 + 1e-9 || roi[1] + roi[3] > 1 + 1e-9)
    throw new Error('Invalid source geometry');
  const x = roundEven(roi[0] * width), y = roundEven(roi[1] * height);
  const cropWidth = Math.min(width, roundEven((roi[0] + roi[2]) * width)) - x;
  const cropHeight = Math.min(height, roundEven((roi[1] + roi[3]) * height)) - y;
  if (cropWidth <= 0 || cropHeight <= 0) throw new Error('Empty source ROI');
  const scale = Math.min(INPUT_SIZE / cropWidth, INPUT_SIZE / cropHeight);
  const resizedWidth = Math.max(1, roundEven(cropWidth * scale));
  const resizedHeight = Math.max(1, roundEven(cropHeight * scale));
  const left = Math.floor((INPUT_SIZE - resizedWidth) / 2);
  const top = Math.floor((INPUT_SIZE - resizedHeight) / 2);
  return { x, y, cropWidth, cropHeight, resizedWidth, resizedHeight, left, top,
    box: [left, top, left + resizedWidth, top + resizedHeight].map(v => v / INPUT_SIZE) };
}

export function regionalPoolWeights(box) {
  const [left, top, right, bottom] = box;
  const weights = new Float32Array(4 * 7 * 7);
  for (const [region, [low, high]] of [[0, 1], [.5, 1], [0, .5], [.4, .6]].entries()) {
    const regionTop = top + low * (bottom - top), regionBottom = top + high * (bottom - top);
    const area = new Float64Array(49);
    let total = 0;
    for (let y = 0; y < 7; y++) for (let x = 0; x < 7; x++) {
      const value = Math.max(0, Math.min((x + 1) / 7, right) - Math.max(x / 7, left))
        * Math.max(0, Math.min((y + 1) / 7, regionBottom) - Math.max(y / 7, regionTop));
      area[y * 7 + x] = value; total += value;
    }
    if (!(total > 0)) throw new Error('Empty regional pool');
    for (let index = 0; index < 49; index++) weights[region * 49 + index] = area[index] / total;
  }
  return weights;
}

export function normalizedLetterbox(rgb, shape) {
  const { resizedWidth: width, resizedHeight: height, left, top } = shape;
  if (rgb.length !== width * height * 3) throw new Error('RGB content shape differs');
  const plane = INPUT_SIZE * INPUT_SIZE, output = new Float32Array(plane * 3);
  for (let channel = 0; channel < 3; channel++) {
    output.fill(Math.fround(-mean[channel] / std[channel]), channel * plane, (channel + 1) * plane);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const value = Math.fround(rgb[(y * width + x) * 3 + channel] / 255);
      output[channel * plane + (y + top) * INPUT_SIZE + x + left] =
        Math.fround(Math.fround(value - mean[channel]) / std[channel]);
    }
  }
  return output;
}

/** Exact binary16 round-to-nearest-even, returned as representable float32. */
export function roundTokensToFloat16(values) {
  const output = new Float32Array(values.length);
  for (let index = 0; index < values.length; index++) {
    const value = Math.fround(values[index]), magnitude = Math.abs(value);
    if (!Number.isFinite(value) || magnitude >= 65520) throw new Error('Token cannot be stored as finite float16');
    const quantum = magnitude < 2 ** -14 ? 2 ** -24 : 2 ** (Math.floor(Math.log2(magnitude)) - 10);
    output[index] = (value < 0 || Object.is(value, -0) ? -1 : 1) * roundEven(magnitude / quantum) * quantum;
  }
  return output;
}

export function fuseFeatures(times, rankedAv, embeddingTimes, tokens, quality, config) {
  if (rankedAv.length !== times.length * 104 || tokens.length !== embeddingTimes.length * TOKEN_DIMENSION
      || quality.length !== embeddingTimes.length * 6 || config.mean.length !== 112 || config.scale.length !== 112)
    throw new Error('Fusion tensor/scaler dimensions differ');
  const output = new Float32Array(times.length * FUSED_DIMENSION);
  const scale = (value, column) => Math.max(-10, Math.min(10,
    Math.fround(Math.fround(value - config.mean[column]) / config.scale[column])));
  let sample = -1;
  for (let row = 0; row < times.length; row++) {
    while (sample + 1 < embeddingTimes.length && embeddingTimes[sample + 1] <= times[row]) sample++;
    const age = sample < 0 ? 0 : times[row] - embeddingTimes[sample];
    const valid = sample >= 0 && age >= 0 && age < .5 + 1e-9;
    const dst = row * FUSED_DIMENSION;
    for (let column = 0; column < 104; column++) output[dst + column] = scale(rankedAv[row * 104 + column], column);
    if (valid) output.set(tokens.subarray(sample * TOKEN_DIMENSION, (sample + 1) * TOKEN_DIMENSION), dst + 104);
    for (let column = 0; column < 8; column++) {
      const value = !valid ? 0 : column < 6 ? quality[sample * 6 + column] : column === 6 ? Math.fround(age) : 1;
      output[dst + 104 + TOKEN_DIMENSION + column] = scale(value, 104 + column);
    }
  }
  return output;
}

export function decodeRallies(times, probabilities, duration, config) {
  if (probabilities.length !== times.length * 4) throw new Error('Probability shape differs');
  const count = times.length;
  if (!count) return [];
  const window = Math.min(count, Math.max(1, roundEven(config.smoothing * 4)));
  const minimum = Math.max(1, roundEven(config.minimum * 4));
  const smooth = new Float32Array(count), mask = new Uint8Array(count);
  const weight = Math.fround(1 / window);
  // NumPy 2 compares a float32 array scalar with a Python float using the
  // array scalar's precision. The short-event guard below explicitly promotes
  // its maximum to Python float, and therefore retains the double threshold.
  const enter = Math.fround(config.enter), exit = Math.fround(config.enter - .1);
  let live = false;
  for (let i = 0; i < count; i++) {
    for (let k = 0; k < window; k++) smooth[i] = Math.fround(smooth[i]
      + Math.fround(probabilities[Math.max(0, Math.min(count - 1, i + k - Math.floor(window / 2))) * 4] * weight));
    if (!live && smooth[i] >= enter) live = true;
    else if (live && smooth[i] < exit) live = false;
    mask[i] = Number(live);
  }
  for (let i = 0; i < count;) {
    let end = i + 1;
    while (end < count && mask[end] === mask[i]) end++;
    if (!mask[i] && i > 0 && end < count && end - i <= 2) mask.fill(1, i, end);
    i = end;
  }
  const rallies = [];
  for (let i = 0; i < count;) {
    if (!mask[i]) { i++; continue; }
    let end = i + 1;
    while (end < count && mask[end]) end++;
    let peak = 0, sum = 0;
    for (let k = i; k < end; k++) { peak = Math.max(peak, smooth[k]); sum += smooth[k]; }
    if (end - i >= minimum || peak >= .9) {
      let a = Math.max(0, times[i] - .125), b = Math.min(duration, times[end - 1] + .125);
      if (config.boundary) {
        const originalA = a, originalB = b;
        let bestA = -1, bestB = -1;
        for (let k = 0; k < count; k++) {
          if (Math.abs(times[k] - originalA) <= .75 && probabilities[k * 4 + 1] >= Math.fround(.65) && probabilities[k * 4 + 1] > bestA) {
            a = times[k]; bestA = probabilities[k * 4 + 1];
          }
          if (Math.abs(times[k] - originalB) <= .75 && probabilities[k * 4 + 2] >= Math.fround(.65) && probabilities[k * 4 + 2] > bestB) {
            b = times[k]; bestB = probabilities[k * 4 + 2];
          }
        }
      }
      if (b > a) rallies.push({ start: a, end: b, confidence: sum / (end - i) });
    }
    i = end;
  }
  return rallies;
}
