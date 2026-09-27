import { VideoSampleSink } from "mediabunny";
import { runtimeAssetUrl } from "../runtime-assets";
import { neuralEmbeddingTimes } from "./neural-contract";
import { contextualizeFeatures } from "./feature-math";
import { extractBrowserFeatures, type FeatureExtractionOptions } from "./pipeline";
import { type OpenedMedia } from "./media";
import { normalizeAnalysisWindow } from "./analysis-window";
import { runProductionInferenceFromFeatures } from "./production-inference";
import { nearestSamplesAtTimestampsFromSequentialPass } from "./sequential-samples";
import { parseNeuralManifest, parseNeuralConfig, verifiedAsset, type NeuralRallySelection } from "./rally-model";
import { neuralEmbeddingCacheKey, readNeuralEmbeddings, writeNeuralEmbeddings, type EmbeddingSequence } from "./neural-cache";
import { NeuralWorkerClient } from "./neural-worker-client";
import type { LocalFeatureSource } from "./feature-cache";
import type { OnDeviceRuntimeVariant } from "./runtime-variants";
import type { AnalysisProgress, NormalizedRoi, OnDeviceAnalysis } from "./types";

export async function analyzeNeuralMedia(media: OpenedMedia, roi: NormalizedRoi, selection: NeuralRallySelection,
  runtimeVariant: OnDeviceRuntimeVariant, onProgress?: (progress: AnalysisProgress) => void,
  source?: LocalFeatureSource, options: FeatureExtractionOptions = {}): Promise<OnDeviceAnalysis> {
  const window = normalizeAnalysisWindow(options.analysisWindow, media.info.duration);
  const duration = window.end - window.start;
  const progress = (value: AnalysisProgress) => { options.signal?.throwIfAborted(); onProgress?.(value); };
  progress({ stage: "video", completed: 0, total: duration, detail: "Loading the selected rally model" });
  const manifestUrl = runtimeAssetUrl("rally-models/manifest.json");
  const response = await fetch(manifestUrl, { signal: options.signal });
  if (!response.ok) throw new Error("The selected rally model is unavailable. Retry, or choose Production ensemble.");
  const bundle = parseNeuralManifest(await response.json()).variants[selection];
  const baseUrl = new URL(`${bundle.directory}/`, manifestUrl).href;
  const config = parseNeuralConfig(JSON.parse(new TextDecoder().decode(await verifiedAsset(
    new URL(bundle.files.pipeline.name, baseUrl).href, bundle.files.pipeline, options.signal))), selection);
  // Both AV and embeddings use the global source grid, clipped to the same game window.
  const times = neuralEmbeddingTimes(media.info.duration, window.start, window.end);
  if (!times.length) throw new Error("The game window contains no image samples.");
  const key = source ? neuralEmbeddingCacheKey(source, media.info, roi, window, bundle) : null;
  let embeddings: EmbeddingSequence | null = key ? await readNeuralEmbeddings(key, times).catch(() => null) : null;
  const restored = Boolean(embeddings);
  const worker = new NeuralWorkerClient(options.signal, (completed, total) => progress({
    stage: "inference", completed, total, detail: `Finding rallies - ${Math.round(completed / total * 100)}%`,
  }));
  try {
    const ready = await worker.request({ type: "initialize", baseUrl, bundle, config, selection,
      needEncoder: !restored, ortBaseUrl: new URL("ort/", runtimeAssetUrl("opencv-worker.js")).href,
      openCvUrl: runtimeAssetUrl("opencv-worker.js") }, "ready");
    if (!embeddings) {
      embeddings = { times, tokens: new Float32Array(times.length * 3840), quality: new Float32Array(times.length * 6) };
      const sink = new VideoSampleSink(media.videoTrack, { hardwareAcceleration: options.decoderAcceleration ?? "prefer-hardware" });
      let row = 0;
      for await (const sample of nearestSamplesAtTimestampsFromSequentialPass(sink, Array.from(times), () => {}, window.end)) {
        options.signal?.throwIfAborted();
        if (!sample) throw new Error("The video decoder omitted an image sample.");
        try {
          if (Math.abs(sample.timestamp - times[row]) > .25 + 1e-9) throw new Error("The video frame timing is too sparse for this rally model.");
          const frame = sample.toVideoFrame();
          let output;
          try {
            output = await worker.request({ type: "frame", frame, timestamp: sample.timestamp, duration: sample.duration,
              rotation: sample.rotation, width: media.info.width, height: media.info.height, roi, offset: sample.timestamp - times[row] }, "frame", [frame]);
          } finally { frame.close(); }
          embeddings.tokens.set(output.tokens, row * 3840);
          embeddings.quality.set(output.quality, row * 6);
          row++;
          progress({ stage: "video", completed: row / times.length * duration * .5, total: duration,
            detail: `Reading game images - ${row}/${times.length} - ${ready.provider === "webgpu" ? "GPU" : "CPU"}` });
        } finally { sample.close(); }
      }
      if (row !== times.length) throw new Error("Image feature extraction did not finish.");
      if (key) await writeNeuralEmbeddings(key, embeddings).catch(() => {});
    }
    options.signal?.throwIfAborted();
    const sequence = await extractBrowserFeatures(media, roi, runtimeVariant, update => progress({
      ...update, ...(update.stage === "video" ? { completed: duration * .5 + update.completed * .5,
        total: duration, detail: `${restored ? "Reused image features - " : ""}${update.detail}` } : {}),
    }), source, { ...options, decodeStrategy: "sequential", visualPreprocessing: "opencv-area-nearest-grid-v1" });
    progress({ stage: "normalizing", completed: 0, total: sequence.rows, detail: "Combining motion, sound and image features" });
    if (sequence.columns !== 104) throw new Error("The neural audio/video feature signature changed.");
    const contextual = contextualizeFeatures(sequence.times, sequence.values, sequence.names);
    const rankedAv = new Float32Array(sequence.rows * 104);
    for (let row = 0; row < sequence.rows; row++) rankedAv.set(contextual.values.subarray(row * 520 + 208, row * 520 + 312), row * 104);
    const modelTimes = new Float64Array(sequence.times);
    const result = await worker.request({ type: "temporal", times: modelTimes, rankedAv, embedding: embeddings, duration: window.end }, "temporal",
      [modelTimes.buffer, rankedAv.buffer, embeddings.times.buffer, embeddings.tokens.buffer, embeddings.quality.buffer]);
    // Keep the existing serve/state heads as evidence for score specialists. Their
    // rally proposals and ensemble suppression policy do not override neural cuts.
    const production = await runProductionInferenceFromFeatures(sequence, window);
    const intervals = result.rallies.map((rally, index) => ({ ...rally, start: Math.max(window.start, rally.start),
      end: Math.min(window.end, rally.end), id: `N${String(index + 1).padStart(3, "0")}`, included: true, agreement: "neural" as const })).filter(rally => rally.end > rally.start);
    const rallyProbabilities = Float32Array.from(sequence.times, (_, i) => result.probabilities[i * 4]);
    const output: OnDeviceAnalysis = { ...production, modelId: bundle.id, intervals, rallyProbabilities, suppression: undefined };
    progress({ stage: "complete", completed: duration, total: duration, detail: `${intervals.length} rallies ready - ${bundle.label}` });
    return output;
  } finally { worker.dispose(); }
}
