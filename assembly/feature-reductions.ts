const WIDTH = 192;
const HEIGHT = 108;
const PIXELS = WIDTH * HEIGHT;
const FEATURES = 73;
const DIAGONAL = Math.sqrt(<f64>(WIDTH * WIDTH + HEIGHT * HEIGHT));

const gray = new Uint8Array(PIXELS);
const hsv = new Uint8Array(PIXELS * 3);
const edges = new Uint8Array(PIXELS);
const laplacian = new Float32Array(PIXELS);
const gradientX = new Float32Array(PIXELS);
const gradientY = new Float32Array(PIXELS);
const previousGray = new Uint8Array(PIXELS);
const flow = new Float32Array(PIXELS * 2);
const output = new Float32Array(FEATURES);

const difference = new Float32Array(PIXELS);
const magnitude = new Float32Array(PIXELS);
const flowX = new Float32Array(PIXELS);
const flowY = new Float32Array(PIXELS);
const residual = new Float32Array(PIXELS);
const selection = new Float32Array(PIXELS);
const xWeights = new Float64Array(WIDTH);
const yWeights = new Float64Array(HEIGHT);

export function grayPointer(): usize { return gray.dataStart; }
export function hsvPointer(): usize { return hsv.dataStart; }
export function edgesPointer(): usize { return edges.dataStart; }
export function laplacianPointer(): usize { return laplacian.dataStart; }
export function gradientXPointer(): usize { return gradientX.dataStart; }
export function gradientYPointer(): usize { return gradientY.dataStart; }
export function previousGrayPointer(): usize { return previousGray.dataStart; }
export function flowPointer(): usize { return flow.dataStart; }
export function outputPointer(): usize { return output.dataStart; }

function meanU8(values: Uint8Array): f64 {
  let total = 0.0;
  for (let index = 0; index < PIXELS; index += 1) total += values[index];
  return total / PIXELS;
}

function standardDeviationU8(values: Uint8Array, average: f64): f64 {
  let sumSquares = 0.0;
  for (let index = 0; index < PIXELS; index += 1) {
    const delta = <f64>values[index] - average;
    sumSquares += delta * delta;
  }
  return Math.sqrt(sumSquares / PIXELS);
}

function meanF32(values: Float32Array): f64 {
  let total = 0.0;
  for (let index = 0; index < PIXELS; index += 1) total += values[index];
  return total / PIXELS;
}

function standardDeviationF32(values: Float32Array, average: f64): f64 {
  let sumSquares = 0.0;
  for (let index = 0; index < PIXELS; index += 1) {
    const delta = <f64>values[index] - average;
    sumSquares += delta * delta;
  }
  return Math.sqrt(sumSquares / PIXELS);
}

function gridMeans(values: Float32Array, outputOffset: i32): void {
  let cell = 0;
  for (let row = 0; row < 3; row += 1) {
    const top = row * 36;
    const bottom = (row + 1) * 36;
    for (let column = 0; column < 3; column += 1) {
      const left = column * 64;
      const right = (column + 1) * 64;
      let total = 0.0;
      for (let y = top; y < bottom; y += 1) {
        for (let x = left; x < right; x += 1) total += values[y * WIDTH + x];
      }
      output[outputOffset + cell] = <f32>(total / ((bottom - top) * (right - left)));
      cell += 1;
    }
  }
}

function gridMeansU8(values: Uint8Array, outputOffset: i32, scale: f64): void {
  let cell = 0;
  for (let row = 0; row < 3; row += 1) {
    const top = row * 36;
    const bottom = (row + 1) * 36;
    for (let column = 0; column < 3; column += 1) {
      const left = column * 64;
      const right = (column + 1) * 64;
      let total = 0.0;
      for (let y = top; y < bottom; y += 1) {
        for (let x = left; x < right; x += 1) total += values[y * WIDTH + x];
      }
      output[outputOffset + cell] =
        <f32>(total / ((bottom - top) * (right - left)) / scale);
      cell += 1;
    }
  }
}

function selectKth(values: Float32Array, target: i32): f32 {
  let left = 0;
  let right = values.length - 1;
  while (left < right) {
    const pivot = values[(left + right) >>> 1];
    let lower = left;
    let upper = right;
    while (lower <= upper) {
      while (values[lower] < pivot) lower += 1;
      while (values[upper] > pivot) upper -= 1;
      if (lower <= upper) {
        const temporary = values[lower];
        values[lower] = values[upper];
        values[upper] = temporary;
        lower += 1;
        upper -= 1;
      }
    }
    if (target <= upper) right = upper;
    else if (target >= lower) left = lower;
    else return values[target];
  }
  return values[target];
}

function quantileInPlace(values: Float32Array, percentile: f64): f64 {
  const position = (values.length - 1) * percentile;
  const lower = <i32>Math.floor(position);
  const upper = <i32>Math.ceil(position);
  const weight = position - lower;
  const lowerValue = selectKth(values, lower);
  const upperValue = upper == lower ? lowerValue : selectKth(values, upper);
  return lowerValue * (1.0 - weight) + upperValue * weight;
}

function copiedQuantile(values: Float32Array, percentile: f64): f64 {
  selection.set(values);
  return quantileInPlace(selection, percentile);
}

export function compute(
  hasPrevious: i32,
  shiftX: f64,
  shiftY: f64,
  shiftResponse: f64,
): void {
  const lumaMean = meanU8(gray);
  const lumaStd = standardDeviationU8(gray, lumaMean);
  let saturationTotal = 0.0;
  for (let index = 0; index < PIXELS; index += 1) saturationTotal += hsv[index * 3 + 1];
  const saturationMean = saturationTotal / PIXELS;
  let saturationSquares = 0.0;
  for (let index = 0; index < PIXELS; index += 1) {
    const delta = <f64>hsv[index * 3 + 1] - saturationMean;
    saturationSquares += delta * delta;
  }
  const saturationStd = Math.sqrt(saturationSquares / PIXELS);
  const laplacianMean = meanF32(laplacian);
  const laplacianStd = standardDeviationF32(laplacian, laplacianMean);
  const laplacianVariance = laplacianStd * laplacianStd;
  let edgeCount = 0;
  for (let index = 0; index < PIXELS; index += 1) {
    if (edges[index] > 0) edgeCount += 1;
  }
  output[0] = <f32>(lumaMean / 255.0);
  output[1] = <f32>(lumaStd / 255.0);
  output[2] = <f32>(saturationMean / 255.0);
  output[3] = <f32>(saturationStd / 255.0);
  output[4] = <f32>(<f64>edgeCount / PIXELS);
  output[5] = <f32>Math.min(laplacianVariance / 2000.0, 5.0);
  gridMeansU8(gray, 6, 255.0);

  let activeDifference = 0;
  for (let index = 0; index < PIXELS; index += 1) {
    const value = hasPrevious != 0
      ? Math.abs(<f64>gray[index] - previousGray[index])
      : 0.0;
    difference[index] = <f32>value;
    if (value >= 18.0) activeDifference += 1;
  }
  const differenceMean = meanF32(difference);
  const differenceStd = standardDeviationF32(difference, differenceMean);
  gridMeans(difference, 19);
  for (let index = 19; index < 28; index += 1) output[index] /= 255.0;
  const differenceP90 = quantileInPlace(difference, 0.9);
  output[15] = <f32>(differenceMean / 255.0);
  output[16] = <f32>(differenceStd / 255.0);
  output[17] = <f32>(differenceP90 / 255.0);
  output[18] = <f32>(<f64>activeDifference / PIXELS);

  const focusQuality = laplacianVariance / (laplacianVariance + 100.0);
  let dark = 0;
  let bright = 0;
  let lowTexture = 0;
  for (let index = 0; index < PIXELS; index += 1) {
    if (gray[index] <= 12) dark += 1;
    if (gray[index] >= 243) bright += 1;
    const gx = <f64>gradientX[index];
    const gy = <f64>gradientY[index];
    if (Math.sqrt(gx * gx + gy * gy) < 8.0) lowTexture += 1;
  }
  const darkFraction = <f64>dark / PIXELS;
  const brightFraction = <f64>bright / PIXELS;
  let occludedCells = 0;
  for (let row = 0; row < 3; row += 1) {
    const top = row * 36;
    const bottom = (row + 1) * 36;
    for (let column = 0; column < 3; column += 1) {
      const left = column * 64;
      const right = (column + 1) * 64;
      const count = (bottom - top) * (right - left);
      let total = 0.0;
      for (let y = top; y < bottom; y += 1) {
        for (let x = left; x < right; x += 1) total += gray[y * WIDTH + x];
      }
      const average = total / count;
      let squares = 0.0;
      for (let y = top; y < bottom; y += 1) {
        for (let x = left; x < right; x += 1) {
          const delta = <f64>gray[y * WIDTH + x] - average;
          squares += delta * delta;
        }
      }
      if ((average <= 16.0 || average >= 239.0) && Math.sqrt(squares / count) <= 8.0) {
        occludedCells += 1;
      }
    }
  }
  const occlusionFraction = <f64>occludedCells / 9.0;
  const exposureQuality = Math.max(0.0, 1.0 - Math.min(1.0, darkFraction + brightFraction));
  const contrastQuality = Math.min(1.0, lumaStd / 255.0 / 0.12);
  const visibilityQuality =
    Math.sqrt(Math.max(0.0, focusQuality * exposureQuality * contrastQuality)) *
    (1.0 - occlusionFraction);
  const cameraShiftMagnitude = Math.sqrt(shiftX * shiftX + shiftY * shiftY) / DIAGONAL;
  output[28] = <f32>focusQuality;
  output[29] = <f32>(1.0 - focusQuality);
  output[30] = <f32>darkFraction;
  output[31] = <f32>brightFraction;
  output[32] = <f32>(<f64>lowTexture / PIXELS);
  output[33] = <f32>occlusionFraction;
  output[34] = <f32>visibilityQuality;
  output[35] = <f32>(shiftX / WIDTH);
  output[36] = <f32>(shiftY / HEIGHT);
  output[37] = <f32>cameraShiftMagnitude;
  output[38] = <f32>shiftResponse;

  let activeFlow = 0;
  for (let index = 0; index < PIXELS; index += 1) {
    const x = flow[index * 2];
    const y = flow[index * 2 + 1];
    flowX[index] = x;
    flowY[index] = y;
    const value = Math.sqrt(<f64>x * x + <f64>y * y);
    magnitude[index] = <f32>value;
    if (value >= 1.0) activeFlow += 1;
  }
  const magnitudeMean = meanF32(magnitude);
  gridMeans(magnitude, 44);
  for (let index = 44; index < 53; index += 1) output[index] /= <f32>DIAGONAL;
  const magnitudeP90 = quantileInPlace(magnitude, 0.9);
  const medianX = copiedQuantile(flowX, 0.5);
  const medianY = copiedQuantile(flowY, 0.5);
  output[39] = <f32>(magnitudeMean / DIAGONAL);
  output[40] = <f32>(magnitudeP90 / DIAGONAL);
  output[41] = <f32>(<f64>activeFlow / PIXELS);
  output[42] = <f32>(medianX / WIDTH);
  output[43] = <f32>(medianY / HEIGHT);

  let activeResidual = 0;
  let activeVectorX = 0.0;
  let activeVectorY = 0.0;
  let activeMagnitude = 0.0;
  for (let index = 0; index < PIXELS; index += 1) {
    const x = <f64>flowX[index] - medianX;
    const y = <f64>flowY[index] - medianY;
    const value = Math.sqrt(x * x + y * y);
    residual[index] = <f32>value;
    if (value >= 1.0) {
      activeResidual += 1;
      activeVectorX += x;
      activeVectorY += y;
      activeMagnitude += value;
    }
  }
  const residualMean = meanF32(residual) / DIAGONAL;
  gridMeans(residual, 64);
  let gridTotal = 0.0;
  for (let index = 64; index < 73; index += 1) gridTotal += output[index];
  let entropy = 0.0;
  if (gridTotal > 1e-9) {
    for (let index = 64; index < 73; index += 1) {
      const probability = output[index] / gridTotal;
      if (probability > 0.0) entropy -= probability * Math.log(probability);
    }
    entropy /= Math.log(9.0);
  }
  xWeights.fill(0.0);
  yWeights.fill(0.0);
  let geometryTotal = 0.0;
  for (let y = 0; y < HEIGHT; y += 1) {
    for (let x = 0; x < WIDTH; x += 1) {
      const value = residual[y * WIDTH + x];
      if (value < 0.5) continue;
      xWeights[x] += value;
      yWeights[y] += value;
      geometryTotal += value;
    }
  }
  let centroidX = 0.5;
  let centroidY = 0.5;
  let spreadX = 0.0;
  let spreadY = 0.0;
  if (geometryTotal > 1e-9) {
    centroidX = 0.0;
    centroidY = 0.0;
    for (let x = 0; x < WIDTH; x += 1) {
      centroidX += xWeights[x] * (<f64>x + 0.5) / WIDTH;
    }
    for (let y = 0; y < HEIGHT; y += 1) {
      centroidY += yWeights[y] * (<f64>y + 0.5) / HEIGHT;
    }
    centroidX /= geometryTotal;
    centroidY /= geometryTotal;
    let varianceX = 0.0;
    let varianceY = 0.0;
    for (let x = 0; x < WIDTH; x += 1) {
      const delta = (<f64>x + 0.5) / WIDTH - centroidX;
      varianceX += xWeights[x] * delta * delta;
    }
    for (let y = 0; y < HEIGHT; y += 1) {
      const delta = (<f64>y + 0.5) / HEIGHT - centroidY;
      varianceY += yWeights[y] * delta * delta;
    }
    spreadX = Math.sqrt(varianceX / geometryTotal);
    spreadY = Math.sqrt(varianceY / geometryTotal);
  }
  const coherence = activeResidual > 0
    ? Math.sqrt(
        (activeVectorX / activeResidual) * (activeVectorX / activeResidual) +
        (activeVectorY / activeResidual) * (activeVectorY / activeResidual),
      ) / Math.max(activeMagnitude / activeResidual, 1e-6)
    : 0.0;
  const cameraGate = Math.max(0.0, 1.0 - Math.min(1.0, cameraShiftMagnitude / 0.03));
  let activeZones = 0;
  for (let index = 64; index < 73; index += 1) {
    if (output[index] >= 0.75) activeZones += 1;
  }
  const residualP90 = quantileInPlace(residual, 0.9);
  output[53] = <f32>residualMean;
  output[54] = <f32>(residualP90 / DIAGONAL);
  output[55] = <f32>(<f64>activeResidual / PIXELS);
  output[56] = <f32>(<f64>activeZones / 9.0);
  output[57] = <f32>entropy;
  output[58] = <f32>centroidX;
  output[59] = <f32>centroidY;
  output[60] = <f32>spreadX;
  output[61] = <f32>spreadY;
  output[62] = <f32>coherence;
  output[63] = <f32>(residualMean * visibilityQuality * cameraGate);
  for (let index = 64; index < 73; index += 1) output[index] /= <f32>DIAGONAL;
}
