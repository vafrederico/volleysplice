import type { LocalFeatureSource } from "./feature-cache";
import type { NormalizedRoi, OnDeviceMediaInfo } from "./types";
import type { AnalysisWindow } from "./analysis-window";
import type { NeuralBundle } from "./rally-model";

const DATABASE = "volleysplice-neural-embeddings";
const STORE = "embeddings";
export const NEURAL_IMAGE_CONTRACT = "rgb-linear-letterbox224-nearest2hz-f16-v1";
export type EmbeddingSequence = { times: Float64Array; tokens: Float32Array; quality: Float32Array };

export function neuralEmbeddingCacheKey(source: LocalFeatureSource, info: OnDeviceMediaInfo,
  roi: NormalizedRoi, window: AnalysisWindow, bundle: NeuralBundle): string {
  return JSON.stringify({ schema: 1, source: [source.name, source.size, source.lastModified],
    media: [info.duration, info.width, info.height, info.rotation, info.videoCodecString],
    roi: [roi.x, roi.y, roi.width, roi.height], window: [window.start, window.end],
    modelId: bundle.id, encoder: bundle.files.encoder.sha256, pipeline: bundle.files.pipeline.sha256,
    preprocessing: NEURAL_IMAGE_CONTRACT });
}
function database(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
export async function readNeuralEmbeddings(key: string, times: Float64Array): Promise<EmbeddingSequence | null> {
  const db = await database();
  try {
    const value = await new Promise<EmbeddingSequence | undefined>((resolve, reject) => {
      const request = db.transaction(STORE).objectStore(STORE).get(key);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    if (!value || !(value.times instanceof Float64Array) || !(value.tokens instanceof Float32Array)
      || !(value.quality instanceof Float32Array) || value.times.length !== times.length
      || !times.every((v, i) => v === value.times[i]) || value.tokens.length !== times.length * 3840
      || value.quality.length !== times.length * 6 || !value.tokens.every(Number.isFinite)
      || !value.quality.every(Number.isFinite)) return null;
    return value;
  } finally { db.close(); }
}
export async function writeNeuralEmbeddings(key: string, value: EmbeddingSequence): Promise<void> {
  const db = await database();
  try {
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite");
      tx.objectStore(STORE).put(value, key);
      tx.oncomplete = () => resolve();
      tx.onerror = tx.onabort = () => reject(tx.error);
    });
  } finally { db.close(); }
}
export async function deleteNeuralEmbeddingsForSource(source: LocalFeatureSource): Promise<void> {
  const db = await database();
  try {
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite");
      const cursor = tx.objectStore(STORE).openCursor();
      cursor.onsuccess = () => {
        const row = cursor.result;
        if (!row) return;
        try {
          const parsed = JSON.parse(String(row.key));
          if (parsed.source?.[0] === source.name && parsed.source?.[1] === source.size
              && parsed.source?.[2] === source.lastModified) row.delete();
        } catch { /* Ignore unknown future key schemas. */ }
        row.continue();
      };
      tx.oncomplete = () => resolve();
      tx.onerror = tx.onabort = () => reject(tx.error);
    });
  } finally { db.close(); }
}
