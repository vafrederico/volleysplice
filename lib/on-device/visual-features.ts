import type { Mat } from "@techstark/opencv-js";

import { ANALYSIS_HEIGHT, ANALYSIS_WIDTH, FRAME_FEATURE_NAMES } from "./feature-schema";
import {
  finiteQuantile,
  finiteQuantileInPlace,
  mean,
  standardDeviation,
} from "./feature-math";

type CvRuntime = typeof import("@techstark/opencv-js");
type CvThenable = {
  then?: (ready: (runtime: CvRuntime) => void) => unknown;
} & Record<string, unknown>;

let runtimePromise: Promise<CvRuntime> | null = null;

function withoutThen(candidate: CvThenable): CvRuntime {
  // Emscripten exposes `cv.then(callback)` but isn't a Promise. Hiding that one
  // property prevents native Promise resolution from recursively assimilating it.
  return new Proxy(candidate, {
    get(target, property, receiver) {
      return property === "then" ? undefined : Reflect.get(target, property, receiver);
    },
  }) as unknown as CvRuntime;
}

export function loadOpenCv(): Promise<CvRuntime> {
  runtimePromise ??= new Promise<CvRuntime>((resolve, reject) => {
    const finish = () => {
      const candidate = (window as unknown as { cv?: CvThenable }).cv;
      if (!candidate) {
        reject(new Error("OpenCV loaded without publishing its browser runtime."));
        return;
      }
      if (typeof candidate.then === "function") {
        candidate.then(() => resolve(withoutThen(candidate)));
      } else resolve(candidate as unknown as CvRuntime);
    };
    const existing = document.querySelector<HTMLScriptElement>(
      'script[data-volleycut-opencv="true"]',
    );
    if (existing) {
      if ((window as unknown as { cv?: CvThenable }).cv) finish();
      else {
        existing.addEventListener("load", finish, { once: true });
        existing.addEventListener(
          "error",
          () => reject(new Error("Could not load the OpenCV browser asset.")),
          { once: true },
        );
      }
      return;
    }
    const script = document.createElement("script");
    script.src = "/on-device/opencv.js";
    script.async = true;
    script.dataset.volleycutOpencv = "true";
    script.addEventListener("load", finish, { once: true });
    script.addEventListener(
      "error",
      () => reject(new Error("Could not load the OpenCV browser asset.")),
      { once: true },
    );
    document.head.append(script);
  });
  return runtimePromise;
}

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

/** OpenCV.js omits phaseCorrelate, so this mirrors OpenCV's DFT/cross-power implementation. */
export function phaseCorrelate(
  cv: CvRuntime,
  previousGray: Mat,
  gray: Mat,
): [number, number, number] {
  const firstInput = new cv.Mat();
  const secondInput = new cv.Mat();
  const first = new cv.Mat();
  const second = new cv.Mat();
  const correlation = new cv.Mat();
  let cross: Mat | null = null;
  try {
    // OpenCV's native phaseCorrelate accepts only floating-point inputs. The
    // reference extractor explicitly casts both grayscale frames to float32.
    previousGray.convertTo(firstInput, cv.CV_32F);
    gray.convertTo(secondInput, cv.CV_32F);
    cv.dft(firstInput, first, cv.DFT_COMPLEX_OUTPUT);
    cv.dft(secondInput, second, cv.DFT_COMPLEX_OUTPUT);
    const left = first.data32F;
    const right = second.data32F;
    const crossData = new Float32Array(left.length);
    for (let index = 0; index < left.length; index += 2) {
      const real = left[index] * right[index] + left[index + 1] * right[index + 1];
      const imaginary = left[index + 1] * right[index] - left[index] * right[index + 1];
      const magnitude = Math.hypot(real, imaginary);
      if (magnitude > Number.EPSILON) {
        crossData[index] = real / magnitude;
        crossData[index + 1] = imaginary / magnitude;
      }
    }
    cross = cv.matFromArray(gray.rows, gray.cols, cv.CV_32FC2, crossData);
    cv.dft(cross, correlation, cv.DFT_INVERSE | cv.DFT_REAL_OUTPUT);

    const source = correlation.data32F;
    const shifted = new Float32Array(source.length);
    const halfWidth = Math.floor(gray.cols / 2);
    const halfHeight = Math.floor(gray.rows / 2);
    let peakValue = Number.NEGATIVE_INFINITY;
    let peakX = 0;
    let peakY = 0;
    for (let y = 0; y < gray.rows; y += 1) {
      const sourceY = (y + halfHeight) % gray.rows;
      for (let x = 0; x < gray.cols; x += 1) {
        const sourceX = (x + halfWidth) % gray.cols;
        const value = source[sourceY * gray.cols + sourceX];
        shifted[y * gray.cols + x] = value;
        if (value > peakValue) {
          peakValue = value;
          peakX = x;
          peakY = y;
        }
      }
    }

    let weightedX = 0;
    let weightedY = 0;
    let weight = 0;
    for (let y = Math.max(0, peakY - 2); y <= Math.min(gray.rows - 1, peakY + 2); y += 1) {
      for (let x = Math.max(0, peakX - 2); x <= Math.min(gray.cols - 1, peakX + 2); x += 1) {
        const value = shifted[y * gray.cols + x];
        weightedX += x * value;
        weightedY += y * value;
        weight += value;
      }
    }
    const centroidX = Math.abs(weight) > Number.EPSILON ? weightedX / weight : peakX;
    const centroidY = Math.abs(weight) > Number.EPSILON ? weightedY / weight : peakY;
    const response = Math.min(1, Math.max(0, weight / (gray.rows * gray.cols)));
    return [gray.cols / 2 - centroidX, gray.rows / 2 - centroidY, response];
  } finally {
    firstInput.delete();
    secondInput.delete();
    first.delete();
    second.delete();
    cross?.delete();
    correlation.delete();
  }
}

export type VisualFeatureResult = {
  values: Float32Array;
  gray: Mat;
  timing: {
    canvasReadbackMs: number;
    imageOperationsMs: number;
    phaseCorrelationMs: number;
    opticalFlowMs: number;
    javascriptMs: number;
  };
};

export function extractVisualFeatures(
  cv: CvRuntime,
  canvas: HTMLCanvasElement | OffscreenCanvas,
  previousGray: Mat | null,
): VisualFeatureResult {
  const readbackStartedAt = performance.now();
  const rgba = cv.imread(canvas as unknown as HTMLCanvasElement);
  const canvasReadbackMs = performance.now() - readbackStartedAt;
  const resized = new cv.Mat();
  const gray = new cv.Mat();
  const rgb = new cv.Mat();
  const hsv = new cv.Mat();
  const edges = new cv.Mat();
  const laplacian = new cv.Mat();
  const gradientX = new cv.Mat();
  const gradientY = new cv.Mat();
  const flow = new cv.Mat();
  let imageOperationsMs = 0;
  let phaseCorrelationMs = 0;
  let opticalFlowMs = 0;
  let javascriptMs = 0;
  try {
    const imageOperationsStartedAt = performance.now();
    if (rgba.cols !== ANALYSIS_WIDTH || rgba.rows !== ANALYSIS_HEIGHT) {
      cv.resize(
        rgba,
        resized,
        new cv.Size(ANALYSIS_WIDTH, ANALYSIS_HEIGHT),
        0,
        0,
        cv.INTER_AREA,
      );
    } else {
      rgba.copyTo(resized);
    }
    cv.cvtColor(resized, gray, cv.COLOR_RGBA2GRAY);
    cv.cvtColor(resized, rgb, cv.COLOR_RGBA2RGB);
    cv.cvtColor(rgb, hsv, cv.COLOR_RGB2HSV);
    cv.Canny(gray, edges, 60, 140);
    cv.Laplacian(gray, laplacian, cv.CV_32F);
    cv.Sobel(gray, gradientX, cv.CV_32F, 1, 0, 3);
    cv.Sobel(gray, gradientY, cv.CV_32F, 0, 1, 3);
    imageOperationsMs += performance.now() - imageOperationsStartedAt;

    let javascriptStartedAt = performance.now();
    const pixels = gray.data;
    const saturation = new Uint8Array(pixels.length);
    for (let index = 0; index < saturation.length; index += 1) {
      saturation[index] = hsv.data[index * 3 + 1];
    }
    const lumaMean = mean(pixels);
    const lumaStd = standardDeviation(pixels, lumaMean);
    const laplacianVariance = standardDeviation(laplacian.data32F) ** 2;
    let edgeCount = 0;
    for (const value of edges.data) if (value > 0) edgeCount += 1;
    const values: number[] = [
      lumaMean / 255,
      lumaStd / 255,
      mean(saturation) / 255,
      standardDeviation(saturation) / 255,
      edgeCount / pixels.length,
      Math.min(laplacianVariance / 2000, 5),
      ...gridMeans(pixels, gray.cols, gray.rows).map((value) => value / 255),
    ];

    const difference = new Float32Array(pixels.length);
    if (previousGray) {
      const previous = previousGray.data;
      for (let index = 0; index < pixels.length; index += 1) {
        difference[index] = Math.abs(pixels[index] - previous[index]);
      }
    }
    let activeDifference = 0;
    for (const value of difference) if (value >= 18) activeDifference += 1;
    const differenceMean = mean(difference);
    const differenceStd = standardDeviation(difference, differenceMean);
    const differenceGrid = gridMeans(difference, gray.cols, gray.rows);
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
      if (Math.hypot(gradientX.data32F[index], gradientY.data32F[index]) < 8) lowTexture += 1;
    }
    const darkFraction = dark / pixels.length;
    const brightFraction = bright / pixels.length;
    let occludedCells = 0;
    for (let row = 0; row < 3; row += 1) {
      const top = Math.round((row * gray.rows) / 3);
      const bottom = Math.round(((row + 1) * gray.rows) / 3);
      for (let column = 0; column < 3; column += 1) {
        const left = Math.round((column * gray.cols) / 3);
        const right = Math.round(((column + 1) * gray.cols) / 3);
        const cell: number[] = [];
        for (let y = top; y < bottom; y += 1) {
          for (let x = left; x < right; x += 1) cell.push(pixels[y * gray.cols + x]);
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
    let shiftX = 0;
    let shiftY = 0;
    let shiftResponse = 0;
    javascriptMs += performance.now() - javascriptStartedAt;
    if (previousGray) {
      const phaseCorrelationStartedAt = performance.now();
      try {
        [shiftX, shiftY, shiftResponse] = phaseCorrelate(cv, previousGray, gray);
      } catch {
        // Match the Python extractor's cv2.error fallback: a failed camera-motion
        // estimate should zero these optional channels, not abort the whole match.
      }
      phaseCorrelationMs += performance.now() - phaseCorrelationStartedAt;
    }
    javascriptStartedAt = performance.now();
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
    javascriptMs += performance.now() - javascriptStartedAt;

    const opticalFlowStartedAt = performance.now();
    if (previousGray) {
      cv.calcOpticalFlowFarneback(previousGray, gray, flow, 0.5, 2, 13, 2, 5, 1.1, 0);
    } else {
      const zeroFlow = cv.Mat.zeros(gray.rows, gray.cols, cv.CV_32FC2);
      zeroFlow.copyTo(flow);
      zeroFlow.delete();
    }
    opticalFlowMs += performance.now() - opticalFlowStartedAt;
    javascriptStartedAt = performance.now();
    const magnitude = new Float32Array(pixels.length);
    const flowX = new Float32Array(pixels.length);
    const flowY = new Float32Array(pixels.length);
    let activeFlow = 0;
    for (let index = 0; index < pixels.length; index += 1) {
      const x = flow.data32F[index * 2];
      const y = flow.data32F[index * 2 + 1];
      flowX[index] = x;
      flowY[index] = y;
      magnitude[index] = Math.hypot(x, y);
      if (magnitude[index] >= 1) activeFlow += 1;
    }
    const magnitudeMean = mean(magnitude);
    const magnitudeGrid = gridMeans(magnitude, gray.cols, gray.rows);
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
    const residualGridPixels = gridMeans(residualMagnitude, gray.cols, gray.rows);
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
      gray.cols,
      gray.rows,
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
    const result = {
      values: Float32Array.from(values),
      gray: gray.clone(),
      timing: {
        canvasReadbackMs,
        imageOperationsMs,
        phaseCorrelationMs,
        opticalFlowMs,
        javascriptMs: javascriptMs + performance.now() - javascriptStartedAt,
      },
    };
    return result;
  } finally {
    rgba.delete();
    resized.delete();
    gray.delete();
    rgb.delete();
    hsv.delete();
    edges.delete();
    laplacian.delete();
    gradientX.delete();
    gradientY.delete();
    flow.delete();
  }
}
