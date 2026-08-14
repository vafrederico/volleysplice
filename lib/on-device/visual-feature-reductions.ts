import { ANALYSIS_HEIGHT, ANALYSIS_WIDTH, FRAME_FEATURE_NAMES } from "./feature-schema.ts";
import {
  finiteQuantile,
  finiteQuantileInPlace,
  mean,
  standardDeviation,
} from "./feature-math.ts";

export type VisualReductionInput = {
  gray: Uint8Array;
  hsv: Uint8Array;
  edges: Uint8Array;
  laplacian: Float32Array;
  gradientX: Float32Array;
  gradientY: Float32Array;
  previousGray: Uint8Array | null;
  flow: Float32Array;
  shiftX: number;
  shiftY: number;
  shiftResponse: number;
};

function gridMeans(values: ArrayLike<number>, width: number, height: number): number[] {
  const result: number[] = [];
  for (let row = 0; row < 3; row += 1) {
    const top = Math.round((row * height) / 3);
    const bottom = Math.round(((row + 1) * height) / 3);
    for (let column = 0; column < 3; column += 1) {
      const left = Math.round((column * width) / 3);
      const right = Math.round(((column + 1) * width) / 3);
      let total = 0;
      let count = 0;
      for (let y = top; y < bottom; y += 1) {
        for (let x = left; x < right; x += 1) {
          total += values[y * width + x];
          count += 1;
        }
      }
      result.push(count ? total / count : 0);
    }
  }
  return result;
}

function weightedMotionGeometry(
  magnitude: Float32Array,
  width: number,
  height: number,
): [number, number, number, number] {
  const xWeights = new Float64Array(width);
  const yWeights = new Float64Array(height);
  let total = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const value = magnitude[y * width + x];
      if (value < 0.5) continue;
      xWeights[x] += value;
      yWeights[y] += value;
      total += value;
    }
  }
  if (total <= 1e-9) return [0.5, 0.5, 0, 0];
  let centroidX = 0;
  let centroidY = 0;
  for (let x = 0; x < width; x += 1) centroidX += xWeights[x] * ((x + 0.5) / width);
  for (let y = 0; y < height; y += 1) centroidY += yWeights[y] * ((y + 0.5) / height);
  centroidX /= total;
  centroidY /= total;
  let varianceX = 0;
  let varianceY = 0;
  for (let x = 0; x < width; x += 1) {
    varianceX += xWeights[x] * ((x + 0.5) / width - centroidX) ** 2;
  }
  for (let y = 0; y < height; y += 1) {
    varianceY += yWeights[y] * ((y + 0.5) / height - centroidY) ** 2;
  }
  return [centroidX, centroidY, Math.sqrt(varianceX / total), Math.sqrt(varianceY / total)];
}

export function computeVisualFeatureReductionsJavaScript(
  input: VisualReductionInput,
): Float32Array {
  const {
    gray,
    hsv,
    edges,
    laplacian,
    gradientX,
    gradientY,
    previousGray,
    flow,
    shiftX,
    shiftY,
    shiftResponse,
  } = input;
  const pixels = gray;
  const saturation = new Uint8Array(pixels.length);
  for (let index = 0; index < saturation.length; index += 1) {
    saturation[index] = hsv[index * 3 + 1];
  }
  const lumaMean = mean(pixels);
  const lumaStd = standardDeviation(pixels, lumaMean);
  const laplacianVariance = standardDeviation(laplacian) ** 2;
  let edgeCount = 0;
  for (const value of edges) if (value > 0) edgeCount += 1;
  const values: number[] = [
    lumaMean / 255,
    lumaStd / 255,
    mean(saturation) / 255,
    standardDeviation(saturation) / 255,
    edgeCount / pixels.length,
    Math.min(laplacianVariance / 2000, 5),
    ...gridMeans(pixels, ANALYSIS_WIDTH, ANALYSIS_HEIGHT).map((value) => value / 255),
  ];

  const difference = new Float32Array(pixels.length);
  if (previousGray) {
    for (let index = 0; index < pixels.length; index += 1) {
      difference[index] = Math.abs(pixels[index] - previousGray[index]);
    }
  }
  let activeDifference = 0;
  for (const value of difference) if (value >= 18) activeDifference += 1;
  const differenceMean = mean(difference);
  const differenceStd = standardDeviation(difference, differenceMean);
  const differenceGrid = gridMeans(difference, ANALYSIS_WIDTH, ANALYSIS_HEIGHT);
  const differenceP90 = finiteQuantileInPlace(difference, 0.9);
  values.push(
    differenceMean / 255,
    differenceStd / 255,
    differenceP90 / 255,
    activeDifference / difference.length,
    ...differenceGrid.map((value) => value / 255),
  );

  const focusQuality = laplacianVariance / (laplacianVariance + 100);
  const blurProbability = 1 - focusQuality;
  let dark = 0;
  let bright = 0;
  let lowTexture = 0;
  for (let index = 0; index < pixels.length; index += 1) {
    if (pixels[index] <= 12) dark += 1;
    if (pixels[index] >= 243) bright += 1;
    if (Math.hypot(gradientX[index], gradientY[index]) < 8) lowTexture += 1;
  }
  const darkFraction = dark / pixels.length;
  const brightFraction = bright / pixels.length;
  let occludedCells = 0;
  for (let row = 0; row < 3; row += 1) {
    const top = Math.round((row * ANALYSIS_HEIGHT) / 3);
    const bottom = Math.round(((row + 1) * ANALYSIS_HEIGHT) / 3);
    for (let column = 0; column < 3; column += 1) {
      const left = Math.round((column * ANALYSIS_WIDTH) / 3);
      const right = Math.round(((column + 1) * ANALYSIS_WIDTH) / 3);
      const cell: number[] = [];
      for (let y = top; y < bottom; y += 1) {
        for (let x = left; x < right; x += 1) cell.push(pixels[y * ANALYSIS_WIDTH + x]);
      }
      const cellMean = mean(cell);
      if ((cellMean <= 16 || cellMean >= 239) && standardDeviation(cell, cellMean) <= 8) {
        occludedCells += 1;
      }
    }
  }
  const occlusionFraction = occludedCells / 9;
  const exposureQuality = Math.max(0, 1 - Math.min(1, darkFraction + brightFraction));
  const contrastQuality = Math.min(1, lumaStd / 255 / 0.12);
  const visibilityQuality =
    Math.sqrt(Math.max(0, focusQuality * exposureQuality * contrastQuality)) *
    (1 - occlusionFraction);
  const diagonal = Math.hypot(ANALYSIS_WIDTH, ANALYSIS_HEIGHT);
  const cameraShiftMagnitude = Math.hypot(shiftX, shiftY) / diagonal;
  values.push(
    focusQuality,
    blurProbability,
    darkFraction,
    brightFraction,
    lowTexture / pixels.length,
    occlusionFraction,
    visibilityQuality,
    shiftX / ANALYSIS_WIDTH,
    shiftY / ANALYSIS_HEIGHT,
    cameraShiftMagnitude,
    shiftResponse,
  );

  const magnitude = new Float32Array(pixels.length);
  const flowX = new Float32Array(pixels.length);
  const flowY = new Float32Array(pixels.length);
  let activeFlow = 0;
  for (let index = 0; index < pixels.length; index += 1) {
    const x = flow[index * 2];
    const y = flow[index * 2 + 1];
    flowX[index] = x;
    flowY[index] = y;
    magnitude[index] = Math.hypot(x, y);
    if (magnitude[index] >= 1) activeFlow += 1;
  }
  const magnitudeMean = mean(magnitude);
  const magnitudeGrid = gridMeans(magnitude, ANALYSIS_WIDTH, ANALYSIS_HEIGHT);
  const magnitudeP90 = finiteQuantileInPlace(magnitude, 0.9);
  const medianX = finiteQuantile(flowX, 0.5);
  const medianY = finiteQuantile(flowY, 0.5);
  values.push(
    magnitudeMean / diagonal,
    magnitudeP90 / diagonal,
    activeFlow / magnitude.length,
    medianX / ANALYSIS_WIDTH,
    medianY / ANALYSIS_HEIGHT,
    ...magnitudeGrid.map((value) => value / diagonal),
  );

  const residualMagnitude = new Float32Array(pixels.length);
  let activeResidual = 0;
  let activeVectorX = 0;
  let activeVectorY = 0;
  let activeMagnitude = 0;
  for (let index = 0; index < pixels.length; index += 1) {
    const x = flowX[index] - medianX;
    const y = flowY[index] - medianY;
    const value = Math.hypot(x, y);
    residualMagnitude[index] = value;
    if (value >= 1) {
      activeResidual += 1;
      activeVectorX += x;
      activeVectorY += y;
      activeMagnitude += value;
    }
  }
  const residualGridPixels = gridMeans(
    residualMagnitude,
    ANALYSIS_WIDTH,
    ANALYSIS_HEIGHT,
  );
  const gridTotal = residualGridPixels.reduce((total, value) => total + value, 0);
  let entropy = 0;
  if (gridTotal > 1e-9) {
    for (const value of residualGridPixels) {
      const probability = value / gridTotal;
      if (probability > 0) entropy -= probability * Math.log(probability);
    }
    entropy /= Math.log(residualGridPixels.length);
  }
  const [centroidX, centroidY, spreadX, spreadY] = weightedMotionGeometry(
    residualMagnitude,
    ANALYSIS_WIDTH,
    ANALYSIS_HEIGHT,
  );
  const coherence = activeResidual
    ? Math.hypot(activeVectorX / activeResidual, activeVectorY / activeResidual) /
      Math.max(activeMagnitude / activeResidual, 1e-6)
    : 0;
  const cameraGate = Math.max(0, 1 - Math.min(1, cameraShiftMagnitude / 0.03));
  const residualMean = mean(residualMagnitude) / diagonal;
  let activeZones = 0;
  for (const value of residualGridPixels) if (value >= 0.75) activeZones += 1;
  const residualP90 = finiteQuantileInPlace(residualMagnitude, 0.9);
  values.push(
    residualMean,
    residualP90 / diagonal,
    activeResidual / residualMagnitude.length,
    activeZones / residualGridPixels.length,
    entropy,
    centroidX,
    centroidY,
    spreadX,
    spreadY,
    coherence,
    residualMean * visibilityQuality * cameraGate,
    ...residualGridPixels.map((value) => value / diagonal),
  );

  if (values.length !== FRAME_FEATURE_NAMES.length) {
    throw new Error(`Visual feature signature mismatch: ${values.length}.`);
  }
  return Float32Array.from(values);
}
