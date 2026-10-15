import { VideoSampleSink } from "mediabunny";
import { runtimeAssetUrl } from "../runtime-assets";
import { parseNeuralConfig, parseNeuralManifest, verifiedAsset } from "./rally-model";
import { applyNeuralServingDecision, neuralServingDecision, neuralServingSelection, parseNeuralServingManifest,
  servingEmbeddingTimes, servingEmbeddingWindow } from "./neural-serving-model";
import { takeServingEmbeddings } from "./neural-serving-context";
import { NeuralWorkerClient } from "./neural-worker-client";
import { neuralServingFeatures } from "./neural-serving-features";
import { nearestSamplesAtTimestampsFromSequentialPass } from "./sequential-samples";
import type { EmbeddingSequence } from "./neural-cache";
import type { OpenedMedia } from "./media";
import type { ServingSideAnalysisInput, ServingSideProgress } from "./serving-side";
import type { NormalizedRoi, OnDeviceServingSideOutput } from "./types";

/** Replace only side evidence; retain the independent production serve gate. */
export async function inferNeuralServingSideOutput(media: OpenedMedia, roi: NormalizedRoi,
  analysis: ServingSideAnalysisInput, specialist: OnDeviceServingSideOutput,
  onProgress?: (progress: ServingSideProgress) => void): Promise<OnDeviceServingSideOutput> {
  const selection = neuralServingSelection(analysis.modelId);
  if (!selection) throw new Error("The selected rally model has no neural serving-side pairing.");
  const rallyUrl = runtimeAssetUrl("rally-models/manifest.json");
  const [rallyResponse, servingResponse] = await Promise.all([
    fetch(rallyUrl), fetch(runtimeAssetUrl("serving-models/manifest.json")),
  ]);
  if (!rallyResponse.ok || !servingResponse.ok) throw new Error("The selected serving-side model is unavailable.");
  const rally = parseNeuralManifest(await rallyResponse.json()).variants[selection];
  const bundle = parseNeuralServingManifest(await servingResponse.json(), rally);
  const output: OnDeviceServingSideOutput = { ...specialist, modelId: bundle.id, modelFingerprint: bundle.fingerprint };
  if (!specialist.candidates.length) return output;
  const baseUrl = new URL(`${rally.directory}/`, rallyUrl).href;
  const config = parseNeuralConfig(JSON.parse(new TextDecoder().decode(await verifiedAsset(
    new URL(rally.files.pipeline.name, baseUrl).href, rally.files.pipeline))), selection);
  let embedding = takeServingEmbeddings(analysis);
  const worker = new NeuralWorkerClient();
  try {
    await worker.request({ type: "initialize", baseUrl, bundle: rally, config, selection, needEncoder: !embedding,
      ortBaseUrl: new URL("ort/", runtimeAssetUrl("opencv-worker.js")).href,
      openCvUrl: runtimeAssetUrl("opencv-worker.js") }, "ready");
    if (!embedding) {
      const times = servingEmbeddingTimes(specialist.candidates.map(candidate => candidate.anchor), media.info.duration);
      embedding = { times, tokens: new Float32Array(times.length * 3840), quality: new Float32Array(times.length * 6) } satisfies EmbeddingSequence;
      const sink = new VideoSampleSink(media.videoTrack, { hardwareAcceleration: "prefer-hardware" });
      let row = 0;
      for await (const sample of nearestSamplesAtTimestampsFromSequentialPass(sink, Array.from(times), () => {}, media.info.duration)) {
        if (!sample) throw new Error("The video decoder omitted a serving-side image sample.");
        try {
          if (Math.abs(sample.timestamp - times[row]) > .25 + 1e-9) throw new Error("Serving-side image timing is too sparse.");
          const frame = sample.toVideoFrame();
          try {
            const result = await worker.request({ type: "frame", frame, timestamp: sample.timestamp, duration: sample.duration,
              rotation: sample.rotation, width: media.info.width, height: media.info.height, roi, offset: sample.timestamp - times[row] }, "frame", [frame]);
            embedding.tokens.set(result.tokens, row * 3840);
            embedding.quality.set(result.quality, row * 6);
          } finally { frame.close(); }
          row++;
          onProgress?.({ stage: "frames", completed: row, total: times.length, detail: `Reading serve images · ${row}/${times.length}` });
        } finally { sample.close(); }
      }
      if (row !== times.length) throw new Error("Serving-side image extraction did not finish.");
    }
    const { rows, columns, values } = specialist.features;
    if (rows !== specialist.candidates.length || columns !== 237 || values.length !== rows * columns) {
      throw new Error("The neural serving-side specialist features changed.");
    }
    const allFeatures = neuralServingFeatures(values, rows);
    const candidates = [...specialist.candidates];
    for (let start = 0; start < rows; start += 8) {
      const count = Math.min(8, rows - start);
      const windows = new Float32Array(count * 13 * 3843), features = new Float32Array(count * 249);
      for (let row = 0; row < count; row++) {
        const source = start + row;
        windows.set(servingEmbeddingWindow(embedding.times, embedding.tokens, candidates[source].anchor, media.info.duration), row * 13 * 3843);
        features.set(allFeatures.subarray(source * 249, (source + 1) * 249), row * 249);
      }
      const result = await worker.request({ type: "serving-side", bundle,
        assetUrl: new URL(`${bundle.directory}/${bundle.file.name}`, runtimeAssetUrl("serving-models/manifest.json")).href,
        windows, features }, "serving-side", [windows.buffer, features.buffer]);
      for (let row = 0; row < count; row++) {
        const candidate = candidates[start + row];
        const decision = neuralServingDecision(result.logits.subarray(row * 6, row * 6 + 6), bundle);
        candidates[start + row] = applyNeuralServingDecision(candidate, decision);
      }
      onProgress?.({ stage: "features", completed: start + count, total: rows, detail: `Checking serve sides · ${start + count}/${rows} rallies` });
    }
    return { ...output, candidates };
  } finally { worker.dispose(); }
}
