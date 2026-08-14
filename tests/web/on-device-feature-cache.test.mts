import assert from "node:assert/strict";
import test from "node:test";

import { visualFeatureCacheKey } from "../../lib/on-device/feature-cache.ts";
import type { OnDeviceMediaInfo } from "../../lib/on-device/types.ts";

const source = { name: "match.mp4", size: 123_456, lastModified: 1_786_000_000_000 };
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
const roi = { x: 0.04, y: 0.14, width: 0.92, height: 0.84 };

test("visual feature checkpoints are stable for the exact same video and crop", () => {
  assert.equal(
    visualFeatureCacheKey(source, info, roi),
    visualFeatureCacheKey({ ...source }, { ...info }, { ...roi }),
  );
});

test("visual feature checkpoints do not cross files or crop configurations", () => {
  const key = visualFeatureCacheKey(source, info, roi);
  assert.notEqual(key, visualFeatureCacheKey({ ...source, size: source.size + 1 }, info, roi));
  assert.notEqual(key, visualFeatureCacheKey(source, info, { ...roi, x: roi.x + 0.01 }));
  assert.notEqual(
    key,
    visualFeatureCacheKey(source, { ...info, videoCodecString: "avc1.42c01f" }, roi),
  );
});

test("decode experiments have isolated checkpoints while the legacy sparse key stays stable", () => {
  const baseline = visualFeatureCacheKey(source, info, roi);
  assert.equal(
    baseline,
    visualFeatureCacheKey(source, info, roi, undefined),
  );
  const sequential = visualFeatureCacheKey(source, info, roi, {
    decodeStrategy: "sequential",
    decoderAcceleration: "prefer-hardware",
  });
  const browserDefault = visualFeatureCacheKey(source, info, roi, {
    decodeStrategy: "sparse",
    decoderAcceleration: "no-preference",
  });
  const wasmReductions = visualFeatureCacheKey(source, info, roi, {
    decodeStrategy: "sequential",
    decoderAcceleration: "prefer-hardware",
    reductionKernel: "wasm",
  });
  assert.notEqual(baseline, sequential);
  assert.notEqual(baseline, browserDefault);
  assert.notEqual(sequential, browserDefault);
  assert.notEqual(sequential, wasmReductions);
});
