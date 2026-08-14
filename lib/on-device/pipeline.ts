import { CanvasSink } from "mediabunny";

import { extractAudioFeatures } from "./audio-features";
import {
  ANALYSIS_FPS,
  ANALYSIS_HEIGHT,
  ANALYSIS_WIDTH,
  AUDIO_FEATURE_NAMES,
  BASE_FEATURE_NAMES,
  FRAME_FEATURE_NAMES,
  TEMPORAL_FEATURE_NAMES,
} from "./feature-schema";
import { contextualizeFeatures, rollingMean } from "./feature-math";
import { analysisTimestamps, type OpenedMedia } from "./media";
import { loadOnDeviceModelBundle, runOnDeviceModel } from "./model";
import {
  DEFAULT_ON_DEVICE_RUNTIME_VARIANT,
  type OnDeviceRuntimeVariant,
} from "./runtime-variants";
import type {
  AnalysisProgress,
  BaseFeatureSequence,
  NormalizedRoi,
  OnDeviceAnalysis,
} from "./types";
import { extractVisualFeatures, loadOpenCv } from "./visual-features";

const MODEL_URL = "/on-device/model-9c92b8e9333f.json";

function column(values: Float32Array, rows: number, columns: number, index: number): Float32Array {
  const output = new Float32Array(rows);
  for (let row = 0; row < rows; row += 1) output[row] = values[row * columns + index];
  return output;
}

function temporalVisualFeatures(visual: Float32Array, rows: number): Float32Array {
  const columns = FRAME_FEATURE_NAMES.length;
  const byName = new Map(FRAME_FEATURE_NAMES.map((name, index) => [name, index]));
  const playerMotion = column(visual, rows, columns, byName.get("player_motion_mean")!);
  const activeZones = column(
    visual,
    rows,
    columns,
    byName.get("player_motion_active_zone_fraction")!,
  );
  const window = Math.max(1, Math.round(ANALYSIS_FPS));
  const shortWindow = Math.max(1, Math.round(0.5 * ANALYSIS_FPS));
  const pastMotion = rollingMean(playerMotion, window, false);
  const futureMotion = rollingMean(playerMotion, shortWindow, true);
  const pastZones = rollingMean(activeZones, window, false);
  const futureZones = rollingMean(activeZones, shortWindow, true);
  const output = new Float32Array(rows * TEMPORAL_FEATURE_NAMES.length);
  const geometryNames = [
    "player_motion_centroid_x",
    "player_motion_centroid_y",
    "player_motion_spread_x",
    "player_motion_spread_y",
  ] as const;
  const geometryChanges = geometryNames.map((name) => {
    const values = column(visual, rows, columns, byName.get(name)!);
    const past = rollingMean(values, window, false);
    const future = rollingMean(values, window, true);
    const change = new Float32Array(rows);
    for (let row = 0; row < rows; row += 1) change[row] = future[row] - past[row];
    return change;
  });
  for (let row = 0; row < rows; row += 1) {
    const onset = Math.max(futureMotion[row] - pastMotion[row], 0);
    const collapse = Math.max(pastMotion[row] - futureMotion[row], 0);
    const synchronized = collapse * Math.max(pastZones[row] - futureZones[row], 0);
    let geometrySquares = 0;
    for (const change of geometryChanges) geometrySquares += change[row] * change[row];
    const activityGate = Math.min(1, (pastMotion[row] + futureMotion[row]) / 0.015);
    const offset = row * TEMPORAL_FEATURE_NAMES.length;
    output[offset] = onset;
    output[offset + 1] = collapse;
    output[offset + 2] = synchronized;
    output[offset + 3] = Math.sqrt(geometrySquares) * activityGate;
  }
  return output;
}

function yieldToBrowser(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

export async function extractBrowserFeatures(
  media: OpenedMedia,
  roi: NormalizedRoi,
  runtimeVariant: OnDeviceRuntimeVariant = DEFAULT_ON_DEVICE_RUNTIME_VARIANT,
  onProgress?: (progress: AnalysisProgress) => void,
): Promise<BaseFeatureSequence> {
  onProgress?.({ stage: "video", completed: 0, total: media.info.duration, detail: "Loading OpenCV WASM" });
  const cv = await loadOpenCv();
  const left = Math.round(roi.x * media.info.width);
  const top = Math.round(roi.y * media.info.height);
  const right = Math.round((roi.x + roi.width) * media.info.width);
  const bottom = Math.round((roi.y + roi.height) * media.info.height);
  const sink = new CanvasSink(media.videoTrack, {
    crop: {
      left,
      top,
      width: Math.max(1, right - left),
      height: Math.max(1, bottom - top),
    },
    width: ANALYSIS_WIDTH,
    height: ANALYSIS_HEIGHT,
    fit: "fill",
    poolSize: 1,
    decoderOptions: { hardwareAcceleration: "prefer-hardware" },
  });

  const times: number[] = [];
  const rows: Float32Array[] = [];
  let previousGray: import("@techstark/opencv-js").Mat | null = null;
  let decoded = 0;
  try {
    for await (const wrapped of sink.canvasesAtTimestamps(
      analysisTimestamps(media.info.duration, ANALYSIS_FPS),
    )) {
      if (!wrapped) continue;
      if (times.length && wrapped.timestamp <= times[times.length - 1] + 1e-9) continue;
      const result = extractVisualFeatures(cv, wrapped.canvas, previousGray);
      previousGray?.delete();
      previousGray = result.gray;
      times.push(wrapped.timestamp);
      rows.push(result.values);
      decoded += 1;
      if (decoded % 8 === 0) {
        onProgress?.({
          stage: "video",
          completed: wrapped.timestamp,
          total: media.info.duration,
          detail: `Measuring motion · ${decoded.toLocaleString()} frames`,
        });
        await yieldToBrowser();
      }
    }
  } finally {
    previousGray?.delete();
  }
  if (rows.length === 0) throw new Error("The video decoder returned no analysis frames.");

  const timeValues = Float64Array.from(times);
  const visual = new Float32Array(rows.length * FRAME_FEATURE_NAMES.length);
  rows.forEach((row, index) => visual.set(row, index * FRAME_FEATURE_NAMES.length));
  const temporal = temporalVisualFeatures(visual, rows.length);
  const audio = await extractAudioFeatures(
    media.audioTrack,
    timeValues,
    media.info.duration,
    runtimeVariant,
    onProgress,
  );
  const base = new Float32Array(rows.length * BASE_FEATURE_NAMES.length);
  for (let row = 0; row < rows.length; row += 1) {
    const outputOffset = row * BASE_FEATURE_NAMES.length;
    base.set(
      visual.subarray(row * FRAME_FEATURE_NAMES.length, (row + 1) * FRAME_FEATURE_NAMES.length),
      outputOffset,
    );
    base.set(
      temporal.subarray(
        row * TEMPORAL_FEATURE_NAMES.length,
        (row + 1) * TEMPORAL_FEATURE_NAMES.length,
      ),
      outputOffset + FRAME_FEATURE_NAMES.length,
    );
    base.set(
      audio.subarray(row * AUDIO_FEATURE_NAMES.length, (row + 1) * AUDIO_FEATURE_NAMES.length),
      outputOffset + FRAME_FEATURE_NAMES.length + TEMPORAL_FEATURE_NAMES.length,
    );
  }
  return {
    times: timeValues,
    values: base,
    rows: rows.length,
    columns: BASE_FEATURE_NAMES.length,
    names: BASE_FEATURE_NAMES,
  };
}

export async function analyzeOpenedMedia(
  media: OpenedMedia,
  roi: NormalizedRoi,
  featurePath: OnDeviceAnalysis["featurePath"],
  runtimeVariantOrProgress:
    | OnDeviceRuntimeVariant
    | ((progress: AnalysisProgress) => void) = DEFAULT_ON_DEVICE_RUNTIME_VARIANT,
  onProgress?: (progress: AnalysisProgress) => void,
): Promise<OnDeviceAnalysis> {
  const runtimeVariant =
    typeof runtimeVariantOrProgress === "string"
      ? runtimeVariantOrProgress
      : DEFAULT_ON_DEVICE_RUNTIME_VARIANT;
  const progress =
    typeof runtimeVariantOrProgress === "function" ? runtimeVariantOrProgress : onProgress;
  const sequence = await extractBrowserFeatures(media, roi, runtimeVariant, progress);
  progress?.({
    stage: "normalizing",
    completed: 0,
    total: sequence.rows,
    detail: "Ranking whole-recording features and adding temporal context",
  });
  await yieldToBrowser();
  const contextual = contextualizeFeatures(sequence.times, sequence.values, sequence.names);
  const response = await fetch(MODEL_URL);
  if (!response.ok) throw new Error(`Could not load the on-device model (${response.status}).`);
  const rawBundle: unknown = await response.json();
  const bundle = loadOnDeviceModelBundle(rawBundle);
  if (
    contextual.names.length !== bundle.featureNames.length ||
    contextual.names.some((name, index) => name !== bundle.featureNames[index])
  ) {
    throw new Error("Extracted feature signature does not match model-9c92b8e9333f.");
  }
  progress?.({
    stage: "inference",
    completed: sequence.rows,
    total: sequence.rows,
    detail: "Running rally, serve, and dead-state heads on CPU",
  });
  const inference = runOnDeviceModel(
    bundle,
    sequence.times,
    contextual.values,
    media.info.duration,
  );
  const intervals = inference.rallies.map((rally) => ({ ...rally }));
  progress?.({
    stage: "complete",
    completed: media.info.duration,
    total: media.info.duration,
    detail: `${intervals.length} candidate rallies ready for review`,
  });
  return {
    modelId: "model-9c92b8e9333f",
    featurePath,
    intervals,
    times: sequence.times,
    rallyProbabilities: inference.probabilities.rally,
    serveProbabilities: inference.probabilities.serve,
    deadStateProbabilities: inference.probabilities.deadState,
  };
}
