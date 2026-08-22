import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  audioFeatureCacheKey,
  visualFeatureCacheKey,
} from "../../prod/src/lib/on-device/feature-cache.ts";
import {
  fullAnalysisWindow,
  normalizeAnalysisWindow,
} from "../../prod/src/lib/on-device/analysis-window.ts";
import {
  ALL_LABELS_V2_BUNDLE_SHA256,
  PREVIOUS_PRODUCTION_BUNDLE_SHA256,
  PRODUCTION_ENSEMBLE_ALGORITHM_VERSION,
  PRODUCTION_ENSEMBLE_MODEL_ID,
} from "../../prod/src/lib/on-device/ensemble.ts";
import { DEFAULT_ON_DEVICE_RUNTIME_VARIANT } from "../../prod/src/lib/on-device/runtime-variants.ts";
import {
  SERVING_SIDE_ANCHOR_CONTRACT,
  SERVING_SIDE_FEATURE_VERSION,
  SERVING_SIDE_MODEL_FINGERPRINT,
  SERVING_SIDE_MODEL_ID,
} from "../../prod/src/lib/on-device/serving-side-model.ts";
import {
  normalizeStoredProject,
  projectAnalysisId,
  projectId,
  sourceCanReconnectFile,
  sourceFileFingerprint,
  sourceMatchesFile,
  type ProjectSource,
  type VolleyCutProject,
} from "../../prod/src/lib/project-store.ts";
import type {
  OnDeviceAnalysis,
  OnDeviceMediaInfo,
} from "../../prod/src/lib/on-device/types.ts";

const source: ProjectSource = {
  name: "match.mp4",
  size: 123_456,
  lastModified: 1_786_752_000_000,
  type: "video/mp4",
};

const info: OnDeviceMediaInfo = {
  duration: 90,
  mimeType: "video/mp4",
  width: 1920,
  height: 1080,
  rotation: 0,
  videoCodec: "avc",
  videoCodecString: "avc1.640028",
  canDecodeVideo: true,
  hasAudio: true,
  audioCodec: "aac",
  sampleRate: 48_000,
  channels: 2,
  canDecodeAudio: true,
};

function cachedAnalysis(modelId: string, currentShape = false): OnDeviceAnalysis {
  return {
    modelId,
    featurePath: "local-source",
    intervals: [
      {
        id: "R001",
        start: 1,
        end: 2,
        confidence: 0.9,
        included: true,
        ...(currentShape ? { agreement: "both-models" as const } : {}),
      },
    ],
    times: new Float64Array([0]),
    rallyProbabilities: new Float32Array([0]),
    serveProbabilities: new Float32Array([0]),
    deadStateProbabilities: new Float32Array([0]),
  };
}

function storedProject(analysis: OnDeviceAnalysis): VolleyCutProject {
  return {
    schemaVersion: 1,
    id: "project-fixture",
    source,
    info,
    analysisWindow: fullAnalysisWindow(info.duration),
    roi: { x: 0, y: 0, width: 1, height: 1 },
    status: "ready",
    analysis,
    error: null,
    createdAt: "2026-08-15T00:00:00.000Z",
    updatedAt: "2026-08-15T00:00:00.000Z",
  };
}

test("project IDs are stable for the same local source fingerprint", () => {
  assert.equal(projectId(source, info), projectId({ ...source }, { ...info }));
  assert.match(projectId(source, info), /^project-[a-z0-9]+$/);
});

test("project IDs change when source identity or timeline changes", () => {
  const id = projectId(source, info);
  assert.notEqual(
    id,
    projectId({ ...source, lastModified: source.lastModified + 1 }, info),
  );
  assert.notEqual(
    id,
    projectId(source, { ...info, duration: info.duration + 0.001 }),
  );
});

test("project IDs and visual caches are isolated by marked game window", () => {
  const fullId = projectId(source, info);
  const firstGame = { start: 10, end: 80 };
  const secondGame = { start: 12, end: 80 };
  assert.notEqual(fullId, projectId(source, info, firstGame));
  assert.notEqual(
    projectId(source, info, firstGame),
    projectId(source, info, secondGame),
  );

  const roi = { x: 0, y: 0, width: 1, height: 1 };
  const localSource = {
    name: source.name,
    size: source.size,
    lastModified: source.lastModified,
  };
  assert.notEqual(
    visualFeatureCacheKey(localSource, info, roi, undefined, firstGame),
    visualFeatureCacheKey(localSource, info, roi, undefined, secondGame),
  );
});

test("legacy projects default to the full source window", () => {
  const legacy = storedProject(
    cachedAnalysis(PRODUCTION_ENSEMBLE_MODEL_ID, true),
  ) as VolleyCutProject & { analysisWindow?: undefined };
  delete legacy.analysisWindow;
  assert.deepEqual(
    normalizeStoredProject(legacy as VolleyCutProject).analysisWindow,
    { start: 0, end: info.duration },
  );
  assert.deepEqual(normalizeAnalysisWindow({ start: -5, end: 120 }, 90), {
    start: 0,
    end: 90,
  });
});

test("reconnected files must match name, size, and modification time", () => {
  const matching = { ...source } as File;
  assert.equal(sourceMatchesFile(source, matching), true);
  assert.equal(
    sourceMatchesFile(source, { ...matching, size: matching.size + 1 } as File),
    false,
  );
  assert.equal(
    sourceMatchesFile(source, { ...matching, name: "other.mp4" } as File),
    false,
  );
});

test("sampled source fingerprints reconnect identical transferred files", async () => {
  const bytes = new Uint8Array(2_200_000);
  bytes.fill(23, 0, 1_100_000);
  bytes.fill(91, 1_100_000);
  const original = new File([bytes], "original.mp4", {
    type: "video/mp4",
    lastModified: 100,
  });
  const transferred = new File([bytes], "renamed.mp4", {
    type: "video/mp4",
    lastModified: 200,
  });
  const changed = new File([bytes.slice(0, -1), new Uint8Array([92])], "renamed.mp4", {
    type: "video/mp4",
    lastModified: 200,
  });
  const fingerprint = await sourceFileFingerprint(original);
  const fingerprintSource: ProjectSource = {
    name: original.name,
    size: original.size,
    lastModified: original.lastModified,
    type: original.type,
    fingerprint,
  };
  assert.equal(await sourceCanReconnectFile(fingerprintSource, transferred), true);
  assert.equal(await sourceCanReconnectFile(fingerprintSource, changed), false);
});

test("audio feature caches are isolated by source and analysis timestamps", () => {
  const localSource = {
    name: source.name,
    size: source.size,
    lastModified: source.lastModified,
  };
  const times = new Float64Array([0, 0.25, 0.5]);
  const key = audioFeatureCacheKey(
    localSource,
    info,
    DEFAULT_ON_DEVICE_RUNTIME_VARIANT,
    times,
  );
  assert.equal(
    key,
    audioFeatureCacheKey(
      { ...localSource },
      { ...info },
      DEFAULT_ON_DEVICE_RUNTIME_VARIANT,
      new Float64Array(times),
    ),
  );
  assert.notEqual(
    key,
    audioFeatureCacheKey(
      localSource,
      info,
      DEFAULT_ON_DEVICE_RUNTIME_VARIANT,
      new Float64Array([0, 0.25, 0.75]),
    ),
  );
});

test("production inference cache identity includes both bundles and ensemble logic", () => {
  const assetHash = (filename: string) =>
    createHash("sha256")
      .update(
        readFileSync(
          new URL(`../../prod/public/runtime/${filename}`, import.meta.url),
        ),
      )
      .digest("hex");
  assert.equal(
    ALL_LABELS_V2_BUNDLE_SHA256,
    assetHash("model-1ca43e38eefc.json"),
  );
  assert.equal(
    PREVIOUS_PRODUCTION_BUNDLE_SHA256,
    assetHash("model-9c92b8e9333f.json"),
  );
  assert.match(
    PRODUCTION_ENSEMBLE_MODEL_ID,
    new RegExp(PRODUCTION_ENSEMBLE_ALGORITHM_VERSION),
  );
  assert.ok(PRODUCTION_ENSEMBLE_MODEL_ID.includes(ALL_LABELS_V2_BUNDLE_SHA256));
  assert.ok(
    PRODUCTION_ENSEMBLE_MODEL_ID.includes(
      PREVIOUS_PRODUCTION_BUNDLE_SHA256,
    ),
  );

  const project = storedProject(
    cachedAnalysis(PRODUCTION_ENSEMBLE_MODEL_ID, true),
  );
  assert.equal(normalizeStoredProject(project), project);
  assert.ok(projectAnalysisId(project)?.includes(PRODUCTION_ENSEMBLE_MODEL_ID));
});

test("stale single-model and prior-ensemble inference caches require re-inference", () => {
  for (const modelId of [
    "model-1ca43e38eefc",
    "ensemble-1ca43e38eefc-9c92b8e9333f",
  ]) {
    const normalized = normalizeStoredProject(
      storedProject(cachedAnalysis(modelId)),
    );
    assert.equal(normalized.status, "waiting");
    assert.equal(normalized.analysis, null);
    assert.match(normalized.error ?? "", /ensemble changed/);
  }
});

test("an ensemble cache without per-range agreement provenance is stale", () => {
  const normalized = normalizeStoredProject(
    storedProject(cachedAnalysis(PRODUCTION_ENSEMBLE_MODEL_ID)),
  );
  assert.equal(normalized.status, "waiting");
  assert.equal(normalized.analysis, null);
});

test("a serving-side cache with changed candidate anchors is discarded without invalidating the project", () => {
  const project = storedProject(
    cachedAnalysis(PRODUCTION_ENSEMBLE_MODEL_ID, true),
  );
  project.analysis!.servingSide = {
    modelId: SERVING_SIDE_MODEL_ID,
    modelFingerprint: SERVING_SIDE_MODEL_FINGERPRINT,
    featureVersion: SERVING_SIDE_FEATURE_VERSION,
    anchorContract: SERVING_SIDE_ANCHOR_CONTRACT,
    features: { rows: 0, columns: 237, values: new Float64Array(0) },
    candidates: [],
  };
  const normalized = normalizeStoredProject(project);
  assert.equal(normalized.status, "ready");
  assert.equal(normalized.analysis?.servingSide, undefined);
  assert.ok(normalized.analysis);
});
