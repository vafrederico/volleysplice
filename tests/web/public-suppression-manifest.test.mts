import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { publicSuppressionManifest, verifyPublicSuppressionManifest } from "../../prod/scripts/public-suppression-manifest.mjs";

const checkedManifest = JSON.parse(await readFile(new URL(
  "../../prod/public/runtime/suppression-39eddf581639.manifest.json", import.meta.url,
), "utf8"));

test("checked suppression provenance contains only public fields", () => {
  verifyPublicSuppressionManifest(checkedManifest);
  assert.deepEqual(publicSuppressionManifest(checkedManifest), checkedManifest);
});

test("preparation strips unapproved fields at every manifest level", () => {
  const input = structuredClone(checkedManifest);
  input.internalNote = "internal-only";
  input.source.modelDirectory = "internal-only";
  input.emitted.internalNote = "internal-only";
  input.decoder.internalNote = "internal-only";
  assert.deepEqual(publicSuppressionManifest(input), checkedManifest);
  assert.throws(() => verifyPublicSuppressionManifest(input), /not approved for publication/);
});

test("preparation rejects identifying strings in runtime fields", () => {
  for (const mutate of [
    value => { value.emitted.filename = "internal-only"; },
    value => { value.emitted.sha256 = "internal-only"; },
    value => { value.source.metadataSha256 = "internal-only"; },
    value => { value.decoderVersion = "internal-only"; },
    value => { value.decoder.enter_threshold = "internal-only"; },
    value => { value.decoder.enter_threshold = 2; },
    value => { value.decoder.bridge_gap_seconds = Number.POSITIVE_INFINITY; },
    value => { delete value.source; },
  ]) {
    const input = structuredClone(checkedManifest);
    mutate(input);
    assert.throws(() => publicSuppressionManifest(input));
  }
});
