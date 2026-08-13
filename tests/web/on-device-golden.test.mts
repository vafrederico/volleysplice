import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { gunzipSync } from "node:zlib";

import { BASE_FEATURE_NAMES } from "../../lib/on-device/feature-schema.ts";
import { contextualizeFeatures } from "../../lib/on-device/feature-math.ts";
import {
  loadOnDeviceModelBundle,
  runOnDeviceModel,
} from "../../lib/on-device/model.ts";

type GoldenMetadata = {
  schemaVersion: 1;
  duration: number;
  rows: number;
  columns: number;
  timesBytes: number;
  names: string[];
  rallies: Array<{
    id: string;
    start: number;
    end: number;
    confidence: number;
  }>;
};

const fixtureRoot = new URL("../fixtures/", import.meta.url);

test("canonical cached features reproduce all 37 model-9c92 intervals", () => {
  const metadata = JSON.parse(
    readFileSync(new URL("on-device-y9-golden.json", fixtureRoot), "utf8"),
  ) as GoldenMetadata;
  assert.equal(metadata.schemaVersion, 1);
  assert.deepEqual(metadata.names, [...BASE_FEATURE_NAMES]);
  assert.equal(metadata.rows, 3474);
  assert.equal(metadata.columns, 104);

  const inflated = gunzipSync(
    readFileSync(new URL("on-device-y9-base-features.bin.gz", fixtureRoot)),
  );
  const storage = inflated.buffer.slice(
    inflated.byteOffset,
    inflated.byteOffset + inflated.byteLength,
  );
  const times = new Float64Array(storage, 0, metadata.rows);
  const base = new Float32Array(
    storage,
    metadata.timesBytes,
    metadata.rows * metadata.columns,
  );
  const contextual = contextualizeFeatures(times, base, metadata.names);
  const rawBundle: unknown = JSON.parse(
    readFileSync(
      new URL("../../public/on-device/model-9c92b8e9333f.json", import.meta.url),
      "utf8",
    ),
  );
  const bundle = loadOnDeviceModelBundle(rawBundle);
  assert.deepEqual(contextual.names, bundle.featureNames);

  const result = runOnDeviceModel(bundle, times, contextual.values, metadata.duration);
  assert.equal(result.rallies.length, 37);
  assert.deepEqual(
    result.rallies.map(({ id, start, end }) => ({
      id,
      start: Number(start.toFixed(3)),
      end: Number(end.toFixed(3)),
    })),
    metadata.rallies.map(({ id, start, end }) => ({ id, start, end })),
  );
  for (let index = 0; index < result.rallies.length; index += 1) {
    assert.ok(
      Math.abs(result.rallies[index].confidence - metadata.rallies[index].confidence) < 2e-5,
      `confidence drift at ${metadata.rallies[index].id}`,
    );
  }
});
