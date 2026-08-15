import assert from "node:assert/strict";
import test from "node:test";

import { audioFeatureCacheKey } from "../../prod/src/lib/on-device/feature-cache.ts";
import { DEFAULT_ON_DEVICE_RUNTIME_VARIANT } from "../../prod/src/lib/on-device/runtime-variants.ts";
import {
  projectId,
  sourceMatchesFile,
  type ProjectSource,
} from "../../prod/src/lib/project-store.ts";
import type { OnDeviceMediaInfo } from "../../prod/src/lib/on-device/types.ts";

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
