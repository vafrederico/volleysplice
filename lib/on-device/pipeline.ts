import { CanvasSink, VideoSample, VideoSampleSink } from "mediabunny";

import { extractAudioFeatures } from "./audio-features";
import {
  FEATURE_CACHE_CHUNK_ROWS,
  markVisualFeatureCacheComplete,
  readVisualFeatureCache,
  type LocalFeatureSource,
  visualFeatureCacheKey,
  writeVisualFeatureChunk,
} from "./feature-cache";
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
import type {
  AnalysisProgress,
  BaseFeatureSequence,
  FeatureExtractionPerformance,
  NormalizedRoi,
  OnDeviceAnalysis,
} from "./types";
import { extractVisualFeatures, loadOpenCv } from "./visual-features";
import {
  VisualFeatureWorkerClient,
  type WorkerFeatureResult,
} from "./visual-feature-worker-client";

const MODEL_URL = "/on-device/model-9c92b8e9333f.json";

export const VIDEO_DECODER_HARDWARE_ACCELERATION = "prefer-hardware" as const;

type FeatureCacheState = NonNullable<AnalysisProgress["featureCache"]>;

type ExtractedBrowserFeatures = BaseFeatureSequence & {
  featureCache: FeatureCacheState;
  performance: FeatureExtractionPerformance;
};

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

type InstrumentedVideoSample = VideoSample & {
  _drawWithFitAndMipmapping?: (...args: unknown[]) => void;
};

function instrumentCanvasDraw(onDraw: (milliseconds: number) => void): () => void {
  // CanvasSink intentionally combines sample retrieval and rendering in one
  // await. Mediabunny's pinned runtime has no public timing hook, so wrap its
  // synchronous draw boundary while profiling and restore it after the run.
  const prototype = VideoSample.prototype as InstrumentedVideoSample;
  const original = prototype._drawWithFitAndMipmapping;
  if (typeof original !== "function") return () => undefined;
  const instrumented = function (this: VideoSample, ...args: unknown[]) {
    const startedAt = performance.now();
    try {
      Reflect.apply(original, this, args);
    } finally {
      onDraw(performance.now() - startedAt);
    }
  };
  prototype._drawWithFitAndMipmapping = instrumented;
  return () => {
    if (prototype._drawWithFitAndMipmapping === instrumented) {
      prototype._drawWithFitAndMipmapping = original;
    }
  };
}

function* analysisTimestampsFrom(duration: number, fps: number, start: number) {
  for (const timestamp of analysisTimestamps(duration, fps)) {
    if (timestamp + 1e-9 >= start) yield timestamp;
  }
}

function rowsFromValues(values: Float32Array, rows: number): Float32Array[] {
  return Array.from({ length: rows }, (_, row) =>
    values.slice(
      row * FRAME_FEATURE_NAMES.length,
      (row + 1) * FRAME_FEATURE_NAMES.length,
    ),
  );
}

export async function extractBrowserFeatures(
  media: OpenedMedia,
  roi: NormalizedRoi,
  onProgress?: (progress: AnalysisProgress) => void,
  cacheSource?: LocalFeatureSource,
): Promise<ExtractedBrowserFeatures> {
  const videoStartedAt = performance.now();
  const timingTotals: FeatureExtractionPerformance = {
    sampledFrames: 0,
    generatedFrames: 0,
    generatedVideoSeconds: 0,
    videoElapsedMs: 0,
    openCvLoadMs: 0,
    decoderCanvasMs: 0,
    decoderWaitMs: 0,
    decoderOverlapMs: 0,
    canvasDrawMs: 0,
    canvasDrawFrames: 0,
    workerActive: false,
    workerBlockingMs: 0,
    workerOverlapMs: 0,
    extractionMs: 0,
    canvasReadbackMs: 0,
    imageOperationsMs: 0,
    phaseCorrelationMs: 0,
    opticalFlowMs: 0,
    javascriptMs: 0,
    cacheIoMs: 0,
  };
  const performanceSnapshot = (): FeatureExtractionPerformance => ({
    ...timingTotals,
    videoElapsedMs: performance.now() - videoStartedAt,
  });
  const measureCacheIo = async <Value,>(operation: () => Promise<Value>): Promise<Value> => {
    const startedAt = performance.now();
    try {
      return await operation();
    } finally {
      timingTotals.cacheIoMs += performance.now() - startedAt;
    }
  };
  let cacheKey: string | null = null;
  let cacheEnabled = false;
  let cachedRows = 0;
  let savedRows = 0;
  let cacheChunkCount = 0;
  let cachedComplete = false;
  let cachedTimes: Float64Array<ArrayBufferLike> = new Float64Array(0);
  let cachedValues: Float32Array<ArrayBufferLike> = new Float32Array(0);
  if (cacheSource) {
    const resolvedCacheKey = visualFeatureCacheKey(cacheSource, media.info, roi);
    cacheKey = resolvedCacheKey;
    try {
      const cached = await measureCacheIo(() => readVisualFeatureCache(resolvedCacheKey));
      cacheEnabled = true;
      if (cached) {
        cachedRows = cached.rows;
        savedRows = cached.rows;
        cacheChunkCount = cached.chunkCount;
        cachedComplete = cached.complete;
        cachedTimes = cached.times;
        cachedValues = cached.values;
      }
    } catch {
      cacheEnabled = false;
    }
  }
  const featureCache = (): FeatureCacheState => ({
    enabled: cacheEnabled,
    complete: cachedComplete,
    resumedRows: cachedRows,
    savedRows,
  });
  onProgress?.({
    stage: "video",
    completed: cachedRows > 0 ? cachedTimes[cachedRows - 1] ?? 0 : 0,
    total: media.info.duration,
    detail: cachedRows > 0
      ? `Restored ${cachedRows.toLocaleString()} locally saved frames`
      : "Loading OpenCV WASM",
    featureCache: featureCache(),
    performance: performanceSnapshot(),
  });
  const left = Math.round(roi.x * media.info.width);
  const top = Math.round(roi.y * media.info.height);
  const right = Math.round((roi.x + roi.width) * media.info.width);
  const bottom = Math.round((roi.y + roi.height) * media.info.height);
  const times: number[] = Array.from(cachedTimes);
  const rows: Float32Array[] = rowsFromValues(cachedValues, cachedRows);
  let decoded = cachedRows;
  if (!cachedComplete) {
    const resumeAfter = cachedRows > 0 ? cachedTimes[cachedRows - 1] : -Infinity;
    const warmupStart = Math.max(0, resumeAfter - 1 / ANALYSIS_FPS);
    const crop = {
      left,
      top,
      width: Math.max(1, right - left),
      height: Math.max(1, bottom - top),
    };
    let pendingTimes: number[] = [];
    let pendingRows: Float32Array[] = [];
    const writePendingRows = async (complete: boolean) => {
      if (!cacheKey || !cacheEnabled) return;
      if (pendingRows.length === 0) {
        if (!complete) return;
        await measureCacheIo(() =>
          markVisualFeatureCacheComplete(cacheKey, cacheChunkCount, decoded),
        );
        savedRows = decoded;
        cachedComplete = true;
        return;
      }
      const chunkValues = new Float32Array(
        pendingRows.length * FRAME_FEATURE_NAMES.length,
      );
      pendingRows.forEach((row, index) =>
        chunkValues.set(row, index * FRAME_FEATURE_NAMES.length),
      );
      await measureCacheIo(() =>
        writeVisualFeatureChunk(
          cacheKey,
          cacheChunkCount,
          Float64Array.from(pendingTimes),
          chunkValues,
          decoded,
          complete,
        ),
      );
      cacheChunkCount += 1;
      savedRows = decoded;
      cachedComplete = complete;
      pendingTimes = [];
      pendingRows = [];
    };
    const appendGeneratedFrame = async (timestamp: number, values: Float32Array) => {
      if (timestamp <= resumeAfter + 1e-9) return;
      if (times.length && timestamp <= times[times.length - 1] + 1e-9) return;
      times.push(timestamp);
      rows.push(values);
      pendingTimes.push(timestamp);
      pendingRows.push(values);
      decoded += 1;
      timingTotals.generatedFrames += 1;
      timingTotals.generatedVideoSeconds = timingTotals.generatedFrames / ANALYSIS_FPS;
      if (pendingRows.length >= FEATURE_CACHE_CHUNK_ROWS) {
        try {
          await writePendingRows(false);
        } catch {
          cacheEnabled = false;
        }
      }
      if (timingTotals.generatedFrames % 8 !== 0) return;
      onProgress?.({
        stage: "video",
        completed: timestamp,
        total: media.info.duration,
        detail: cachedRows > 0
          ? `Measuring motion · ${timingTotals.generatedFrames.toLocaleString()} new · ${decoded.toLocaleString()} total frames`
          : `Measuring motion · ${timingTotals.generatedFrames.toLocaleString()} frames`,
        featureCache: featureCache(),
        performance: performanceSnapshot(),
      });
      await yieldToBrowser();
    };
    const collectTiming = (result: WorkerFeatureResult) => {
      timingTotals.sampledFrames += 1;
      timingTotals.extractionMs += result.workerElapsedMs;
      timingTotals.canvasDrawMs += result.canvasDrawMs;
      timingTotals.canvasDrawFrames += 1;
      timingTotals.canvasReadbackMs += result.timing.canvasReadbackMs;
      timingTotals.imageOperationsMs += result.timing.imageOperationsMs;
      timingTotals.phaseCorrelationMs += result.timing.phaseCorrelationMs;
      timingTotals.opticalFlowMs += result.timing.opticalFlowMs;
      timingTotals.javascriptMs += result.timing.javascriptMs;
    };
    try {
      const workerClient = await VisualFeatureWorkerClient.create().catch(() => null);
      if (workerClient) {
        timingTotals.workerActive = true;
        timingTotals.openCvLoadMs += workerClient.openCvLoadMs;
        const sink = new VideoSampleSink(media.videoTrack, {
          hardwareAcceleration: VIDEO_DECODER_HARDWARE_ACCELERATION,
        });
        const sampleIterator = sink.samplesAtTimestamps(
          analysisTimestampsFrom(media.info.duration, ANALYSIS_FPS, warmupStart),
        );
        const pendingFeatures: Promise<WorkerFeatureResult>[] = [];
        const consumeOldest = async () => {
          const pending = pendingFeatures.shift();
          if (!pending) return;
          const waitStartedAt = performance.now();
          const result = await pending;
          timingTotals.workerBlockingMs += performance.now() - waitStartedAt;
          collectTiming(result);
          timingTotals.workerOverlapMs = Math.max(
            0,
            timingTotals.extractionMs - timingTotals.workerBlockingMs,
          );
          await appendGeneratedFrame(result.timestamp, new Float32Array(result.values));
        };
        try {
          while (true) {
            const decoderStartedAt = performance.now();
            const next = await sampleIterator.next();
            const decoderMs = performance.now() - decoderStartedAt;
            timingTotals.decoderCanvasMs += decoderMs;
            timingTotals.decoderWaitMs += decoderMs;
            if (next.done) break;
            const sample = next.value;
            if (!sample) continue;
            const frame = sample.toVideoFrame();
            const metadata = {
              timestamp: sample.timestamp,
              duration: sample.duration,
              rotation: sample.rotation,
              crop,
            };
            sample.close();
            pendingFeatures.push(workerClient.extract(frame, metadata));
            if (pendingFeatures.length >= 2) await consumeOldest();
          }
          while (pendingFeatures.length > 0) await consumeOldest();
        } finally {
          const settling = Promise.allSettled(pendingFeatures);
          workerClient.dispose();
          await settling;
          await sampleIterator.return(undefined);
        }
      } else {
        const openCvStartedAt = performance.now();
        const cv = await loadOpenCv();
        timingTotals.openCvLoadMs += performance.now() - openCvStartedAt;
        const sink = new CanvasSink(media.videoTrack, {
          crop,
          width: ANALYSIS_WIDTH,
          height: ANALYSIS_HEIGHT,
          fit: "fill",
          poolSize: 1,
          decoderOptions: { hardwareAcceleration: VIDEO_DECODER_HARDWARE_ACCELERATION },
        });
        const canvasIterator = sink.canvasesAtTimestamps(
          analysisTimestampsFrom(media.info.duration, ANALYSIS_FPS, warmupStart),
        );
        const stopCanvasDrawTiming = instrumentCanvasDraw((milliseconds) => {
          timingTotals.canvasDrawMs += milliseconds;
          timingTotals.canvasDrawFrames += 1;
        });
        let previousGray: import("@techstark/opencv-js").Mat | null = null;
        let pendingNext: ReturnType<typeof canvasIterator.next> | null = null;
        try {
          let nextRequestedAt = performance.now();
          pendingNext = canvasIterator.next();
          while (true) {
            const canvasDrawBeforeWait = timingTotals.canvasDrawMs;
            const waitStartedAt = performance.now();
            const next = await pendingNext;
            const resolvedAt = performance.now();
            const blockingMs = resolvedAt - waitStartedAt;
            const requestSpanMs = resolvedAt - nextRequestedAt;
            const canvasDrawDuringWaitMs = timingTotals.canvasDrawMs - canvasDrawBeforeWait;
            timingTotals.decoderCanvasMs += blockingMs;
            timingTotals.decoderWaitMs += Math.max(0, blockingMs - canvasDrawDuringWaitMs);
            timingTotals.decoderOverlapMs += Math.max(0, requestSpanMs - blockingMs);
            if (next.done) break;
            const wrapped = next.value;
            nextRequestedAt = performance.now();
            pendingNext = canvasIterator.next();
            if (!wrapped) continue;
            const extractionStartedAt = performance.now();
            const result = extractVisualFeatures(cv, wrapped.canvas, previousGray);
            timingTotals.extractionMs += performance.now() - extractionStartedAt;
            timingTotals.sampledFrames += 1;
            timingTotals.canvasReadbackMs += result.timing.canvasReadbackMs;
            timingTotals.imageOperationsMs += result.timing.imageOperationsMs;
            timingTotals.phaseCorrelationMs += result.timing.phaseCorrelationMs;
            timingTotals.opticalFlowMs += result.timing.opticalFlowMs;
            timingTotals.javascriptMs += result.timing.javascriptMs;
            previousGray?.delete();
            previousGray = result.gray;
            await appendGeneratedFrame(wrapped.timestamp, result.values);
          }
        } finally {
          stopCanvasDrawTiming();
          const pendingResult = pendingNext;
          const iteratorReturn = canvasIterator.return(undefined);
          await pendingResult?.catch(() => undefined);
          await iteratorReturn;
          previousGray?.delete();
        }
      }
      try {
        await writePendingRows(true);
      } catch {
        cacheEnabled = false;
      }
    } catch (error) {
      await writePendingRows(false).catch(() => undefined);
      throw error;
    }
  }
  if (rows.length === 0) throw new Error("The video decoder returned no analysis frames.");

  const timeValues = Float64Array.from(times);
  const visual = new Float32Array(rows.length * FRAME_FEATURE_NAMES.length);
  rows.forEach((row, index) => visual.set(row, index * FRAME_FEATURE_NAMES.length));
  const temporal = temporalVisualFeatures(visual, rows.length);
  const finalPerformance = performanceSnapshot();
  const audio = await extractAudioFeatures(
    media.audioTrack,
    timeValues,
    media.info.duration,
    onProgress
      ? (progress) => onProgress({
          ...progress,
          featureCache: featureCache(),
          performance: finalPerformance,
        })
      : undefined,
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
    featureCache: featureCache(),
    performance: finalPerformance,
  };
}

export async function analyzeOpenedMedia(
  media: OpenedMedia,
  roi: NormalizedRoi,
  featurePath: OnDeviceAnalysis["featurePath"],
  onProgress?: (progress: AnalysisProgress) => void,
  cacheSource?: LocalFeatureSource,
): Promise<OnDeviceAnalysis> {
  const sequence = await extractBrowserFeatures(media, roi, onProgress, cacheSource);
  onProgress?.({
    stage: "normalizing",
    completed: 0,
    total: sequence.rows,
    detail: "Ranking whole-recording features and adding temporal context",
    featureCache: sequence.featureCache,
    performance: sequence.performance,
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
  onProgress?.({
    stage: "inference",
    completed: sequence.rows,
    total: sequence.rows,
    detail: "Running rally, serve, and dead-state heads on CPU",
    featureCache: sequence.featureCache,
    performance: sequence.performance,
  });
  const inference = runOnDeviceModel(
    bundle,
    sequence.times,
    contextual.values,
    media.info.duration,
  );
  const intervals = inference.rallies.map((rally) => ({ ...rally }));
  onProgress?.({
    stage: "complete",
    completed: media.info.duration,
    total: media.info.duration,
    detail: `${intervals.length} candidate rallies ready for review`,
    featureCache: sequence.featureCache,
    performance: sequence.performance,
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
