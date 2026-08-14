import { ANALYSIS_FPS, FRAME_FEATURE_NAMES } from "./feature-schema.ts";
import type { NormalizedRoi, OnDeviceMediaInfo } from "./types.ts";

const DATABASE_NAME = "volleycut-on-device-features";
const DATABASE_VERSION = 1;
const ENTRY_STORE = "entries";
const CHUNK_STORE = "chunks";

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
    if (!database.objectStoreNames.contains(CHUNK_STORE)) {
      database.createObjectStore(CHUNK_STORE, { keyPath: "id" });
    }
  });
  return requestResult(request);
}

function chunkId(cacheKey: string, index: number): string {
  return `${cacheKey}\u0000${String(index).padStart(8, "0")}`;
}

export function visualFeatureCacheKey(
  source: LocalFeatureSource,
  info: OnDeviceMediaInfo,
  roi: NormalizedRoi,
): string {
  let featureSignature = 0x811c9dc5;
  for (const character of FRAME_FEATURE_NAMES.join("|")) {
    featureSignature ^= character.charCodeAt(0);
    featureSignature = Math.imul(featureSignature, 0x01000193) >>> 0;
  }
  return JSON.stringify({
    schema: 1,
    featureSchema: featureSignature.toString(16),
    analysisFps: ANALYSIS_FPS,
    source: [source.name, source.size, source.lastModified],
    media: [info.duration, info.width, info.height, info.rotation, info.videoCodecString],
    roi: [roi.x, roi.y, roi.width, roi.height],
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
