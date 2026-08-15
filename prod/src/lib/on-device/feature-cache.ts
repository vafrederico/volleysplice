import { ANALYSIS_FPS, AUDIO_FEATURE_NAMES, FRAME_FEATURE_NAMES } from "./feature-schema.ts";
import {
  isFullAnalysisWindow,
  normalizeAnalysisWindow,
  type AnalysisWindow,
} from "./analysis-window.ts";
import type { OnDeviceRuntimeVariant } from "./runtime-variants.ts";
import type {
  FeatureReductionKernel,
  NormalizedRoi,
  OnDeviceMediaInfo,
  VideoDecoderAcceleration,
  VideoDecodeStrategy,
} from "./types.ts";

const DATABASE_NAME = "volleycut-on-device-features";
const DATABASE_VERSION = 3;
const ENTRY_STORE = "entries";
const CHUNK_STORE = "chunks";
const AUDIO_STORE = "audio";
const CHUNK_CACHE_KEY_INDEX = "cacheKey";

export const FEATURE_CACHE_CHUNK_ROWS = 16;

export type LocalFeatureSource = {
  name: string;
  size: number;
  lastModified: number;
};

export type VisualFeatureCache = {
  key: string;
  times: Float64Array;
  values: Float32Array;
  rows: number;
  chunkCount: number;
  complete: boolean;
};

type FeatureCacheEntry = {
  id: string;
  schemaVersion: 1;
  columns: number;
  chunkCount: number;
  rowCount: number;
  complete: boolean;
  updatedAt: number;
};

type FeatureCacheChunk = {
  id: string;
  cacheKey: string;
  index: number;
  rows: number;
  times: ArrayBuffer;
  values: ArrayBuffer;
};

type AudioFeatureCacheEntry = {
  id: string;
  schemaVersion: 1;
  columns: number;
  rowCount: number;
  values: ArrayBuffer;
  updatedAt: number;
};

function requestResult<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.addEventListener("success", () => resolve(request.result), { once: true });
    request.addEventListener("error", () => reject(request.error), { once: true });
  });
}

function transactionComplete(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.addEventListener("complete", () => resolve(), { once: true });
    transaction.addEventListener("abort", () => reject(transaction.error), { once: true });
    transaction.addEventListener("error", () => reject(transaction.error), { once: true });
  });
}

function openFeatureDatabase(): Promise<IDBDatabase> {
  if (!("indexedDB" in globalThis)) {
    return Promise.reject(new Error("IndexedDB is unavailable."));
  }
  const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
  request.addEventListener("upgradeneeded", () => {
    const database = request.result;
    if (!database.objectStoreNames.contains(ENTRY_STORE)) {
      database.createObjectStore(ENTRY_STORE, { keyPath: "id" });
    }
    const chunks = database.objectStoreNames.contains(CHUNK_STORE)
      ? request.transaction!.objectStore(CHUNK_STORE)
      : database.createObjectStore(CHUNK_STORE, { keyPath: "id" });
    if (!chunks.indexNames.contains(CHUNK_CACHE_KEY_INDEX)) {
      chunks.createIndex(CHUNK_CACHE_KEY_INDEX, "cacheKey", { unique: false });
    }
    if (!database.objectStoreNames.contains(AUDIO_STORE)) {
      database.createObjectStore(AUDIO_STORE, { keyPath: "id" });
    }
  });
  return requestResult(request);
}

function hashTimes(times: Float64Array): string {
  let hash = 0x811c9dc5;
  const bytes = new Uint8Array(times.buffer, times.byteOffset, times.byteLength);
  for (const byte of bytes) {
    hash ^= byte;
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16);
}

export function audioFeatureCacheKey(
  source: LocalFeatureSource,
  info: OnDeviceMediaInfo,
  runtimeVariant: OnDeviceRuntimeVariant,
  times: Float64Array,
): string {
  return JSON.stringify({
    schema: 1,
    columns: AUDIO_FEATURE_NAMES,
    source: [source.name, source.size, source.lastModified],
    media: [info.duration, info.hasAudio, info.audioCodec, info.sampleRate, info.channels],
    runtimeVariant,
    rows: times.length,
    times: hashTimes(times),
  });
}

function chunkId(cacheKey: string, index: number): string {
  return `${cacheKey}\u0000${String(index).padStart(8, "0")}`;
}

export function visualFeatureCacheKey(
  source: LocalFeatureSource,
  info: OnDeviceMediaInfo,
  roi: NormalizedRoi,
  experiment?: {
    decodeStrategy: VideoDecodeStrategy;
    decoderAcceleration: VideoDecoderAcceleration;
    reductionKernel?: FeatureReductionKernel;
  },
  requestedWindow?: AnalysisWindow,
): string {
  let featureSignature = 0x811c9dc5;
  for (const character of FRAME_FEATURE_NAMES.join("|")) {
    featureSignature ^= character.charCodeAt(0);
    featureSignature = Math.imul(featureSignature, 0x01000193) >>> 0;
  }
  const analysisWindow = normalizeAnalysisWindow(requestedWindow, info.duration);
  return JSON.stringify({
    schema: 1,
    featureSchema: featureSignature.toString(16),
    analysisFps: ANALYSIS_FPS,
    source: [source.name, source.size, source.lastModified],
    media: [info.duration, info.width, info.height, info.rotation, info.videoCodecString],
    roi: [roi.x, roi.y, roi.width, roi.height],
    ...(isFullAnalysisWindow(analysisWindow, info.duration)
      ? {}
      : { analysisWindow: [analysisWindow.start, analysisWindow.end] }),
    ...(experiment ? { experiment } : {}),
  });
}

export async function readVisualFeatureCache(key: string): Promise<VisualFeatureCache | null> {
  const database = await openFeatureDatabase();
  try {
    const entryTransaction = database.transaction(ENTRY_STORE, "readonly");
    const entryComplete = transactionComplete(entryTransaction);
    const entry = await requestResult(
      entryTransaction.objectStore(ENTRY_STORE).get(key) as IDBRequest<FeatureCacheEntry | undefined>,
    );
    await entryComplete;
    if (
      !entry ||
      entry.schemaVersion !== 1 ||
      entry.columns !== FRAME_FEATURE_NAMES.length ||
      entry.rowCount < 0 ||
      entry.chunkCount < 0
    ) {
      return null;
    }

    const chunkTransaction = database.transaction(CHUNK_STORE, "readonly");
    const chunksComplete = transactionComplete(chunkTransaction);
    const store = chunkTransaction.objectStore(CHUNK_STORE);
    const chunks = await Promise.all(
      Array.from({ length: entry.chunkCount }, (_, index) =>
        requestResult(
          store.get(chunkId(key, index)) as IDBRequest<FeatureCacheChunk | undefined>,
        ),
      ),
    );
    await chunksComplete;
    if (chunks.some((chunk) => !chunk)) return null;

    const times = new Float64Array(entry.rowCount);
    const values = new Float32Array(entry.rowCount * FRAME_FEATURE_NAMES.length);
    let rowOffset = 0;
    for (const chunk of chunks as FeatureCacheChunk[]) {
      const chunkTimes = new Float64Array(chunk.times);
      const chunkValues = new Float32Array(chunk.values);
      if (
        chunk.rows !== chunkTimes.length ||
        chunkValues.length !== chunk.rows * FRAME_FEATURE_NAMES.length ||
        rowOffset + chunk.rows > entry.rowCount
      ) {
        return null;
      }
      times.set(chunkTimes, rowOffset);
      values.set(chunkValues, rowOffset * FRAME_FEATURE_NAMES.length);
      rowOffset += chunk.rows;
    }
    if (rowOffset !== entry.rowCount) return null;
    return {
      key,
      times,
      values,
      rows: entry.rowCount,
      chunkCount: entry.chunkCount,
      complete: entry.complete,
    };
  } finally {
    database.close();
  }
}

export async function writeVisualFeatureChunk(
  key: string,
  index: number,
  times: Float64Array,
  values: Float32Array,
  totalRows: number,
  complete: boolean,
): Promise<void> {
  if (times.length === 0 || values.length !== times.length * FRAME_FEATURE_NAMES.length) {
    throw new Error("Invalid visual feature checkpoint chunk.");
  }
  const database = await openFeatureDatabase();
  try {
    const transaction = database.transaction([ENTRY_STORE, CHUNK_STORE], "readwrite");
    const completeTransaction = transactionComplete(transaction);
    const entry: FeatureCacheEntry = {
      id: key,
      schemaVersion: 1,
      columns: FRAME_FEATURE_NAMES.length,
      chunkCount: index + 1,
      rowCount: totalRows,
      complete,
      updatedAt: Date.now(),
    };
    const chunk: FeatureCacheChunk = {
      id: chunkId(key, index),
      cacheKey: key,
      index,
      rows: times.length,
      times: new Float64Array(times).buffer,
      values: new Float32Array(values).buffer,
    };
    transaction.objectStore(CHUNK_STORE).put(chunk);
    transaction.objectStore(ENTRY_STORE).put(entry);
    await completeTransaction;
  } finally {
    database.close();
  }
}

export async function markVisualFeatureCacheComplete(
  key: string,
  chunkCount: number,
  totalRows: number,
): Promise<void> {
  const database = await openFeatureDatabase();
  try {
    const transaction = database.transaction(ENTRY_STORE, "readwrite");
    const completeTransaction = transactionComplete(transaction);
    const entry: FeatureCacheEntry = {
      id: key,
      schemaVersion: 1,
      columns: FRAME_FEATURE_NAMES.length,
      chunkCount,
      rowCount: totalRows,
      complete: true,
      updatedAt: Date.now(),
    };
    transaction.objectStore(ENTRY_STORE).put(entry);
    await completeTransaction;
  } finally {
    database.close();
  }
}

export async function readAudioFeatureCache(
  key: string,
): Promise<Float32Array | null> {
  const database = await openFeatureDatabase();
  try {
    const transaction = database.transaction(AUDIO_STORE, "readonly");
    const complete = transactionComplete(transaction);
    const entry = await requestResult(
      transaction.objectStore(AUDIO_STORE).get(key) as IDBRequest<
        AudioFeatureCacheEntry | undefined
      >,
    );
    await complete;
    if (
      !entry ||
      entry.schemaVersion !== 1 ||
      entry.columns !== AUDIO_FEATURE_NAMES.length ||
      entry.rowCount < 0
    ) {
      return null;
    }
    const values = new Float32Array(entry.values);
    return values.length === entry.rowCount * entry.columns ? values : null;
  } finally {
    database.close();
  }
}

export async function writeAudioFeatureCache(
  key: string,
  values: Float32Array,
  rowCount: number,
): Promise<void> {
  if (values.length !== rowCount * AUDIO_FEATURE_NAMES.length) {
    throw new Error("Invalid audio feature cache entry.");
  }
  const database = await openFeatureDatabase();
  try {
    const transaction = database.transaction(AUDIO_STORE, "readwrite");
    const complete = transactionComplete(transaction);
    const entry: AudioFeatureCacheEntry = {
      id: key,
      schemaVersion: 1,
      columns: AUDIO_FEATURE_NAMES.length,
      rowCount,
      values: new Float32Array(values).buffer,
      updatedAt: Date.now(),
    };
    transaction.objectStore(AUDIO_STORE).put(entry);
    await complete;
  } finally {
    database.close();
  }
}

function cacheKeyMatchesSource(
  key: IDBValidKey,
  source: LocalFeatureSource,
): boolean {
  if (typeof key !== "string") return false;
  try {
    const value = JSON.parse(key) as { source?: unknown };
    return (
      Array.isArray(value.source) &&
      value.source[0] === source.name &&
      value.source[1] === source.size &&
      value.source[2] === source.lastModified
    );
  } catch {
    return false;
  }
}

export async function deleteFeatureCachesForSource(
  source: LocalFeatureSource,
): Promise<void> {
  const database = await openFeatureDatabase();
  try {
    const transaction = database.transaction(
      [ENTRY_STORE, CHUNK_STORE, AUDIO_STORE],
      "readwrite",
    );
    const complete = transactionComplete(transaction);
    const entries = transaction.objectStore(ENTRY_STORE);
    const chunks = transaction.objectStore(CHUNK_STORE);
    const audio = transaction.objectStore(AUDIO_STORE);

    const [entryKeys, audioKeys] = await Promise.all([
      requestResult(entries.getAllKeys()),
      requestResult(audio.getAllKeys()),
    ]);
    const matchingEntryKeys = entryKeys.filter((key) =>
      cacheKeyMatchesSource(key, source),
    );
    const matchingAudioKeys = audioKeys.filter((key) =>
      cacheKeyMatchesSource(key, source),
    );
    const chunkIndex = chunks.index(CHUNK_CACHE_KEY_INDEX);
    const chunkKeyGroups = await Promise.all(
      matchingEntryKeys.map((key) => requestResult(chunkIndex.getAllKeys(key))),
    );

    for (const key of matchingEntryKeys) entries.delete(key);
    for (const key of matchingAudioKeys) audio.delete(key);
    for (const key of chunkKeyGroups.flat()) chunks.delete(key);

    await complete;
  } finally {
    database.close();
  }
}
