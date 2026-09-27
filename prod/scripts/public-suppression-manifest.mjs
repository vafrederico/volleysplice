import { isDeepStrictEqual } from "node:util";

const decoderFields = [
  "bridge_gap_seconds", "enter_threshold", "exit_threshold", "min_live_seconds",
  "short_event_min_seconds", "short_event_threshold", "smoothing_seconds",
];
const hashFields = ["artifactSha256", "metadataSha256", "weightsSha256"];

// Public provenance needs content identities, never the location of training inputs.
export function publicSuppressionManifest(manifest) {
  if (manifest?.schemaVersion !== 1
    || manifest.decoderVersion !== "held-production-suppression-decoder-v1") {
    throw new Error("Unsupported suppression manifest schema or decoder.");
  }
  const decoder = Object.fromEntries(decoderFields.map(field => {
    const value = manifest.decoder?.[field];
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0
      || (field.includes("threshold") && value > 1)) {
      throw new Error(`Invalid public suppression decoder field: ${field}.`);
    }
    return [field, value];
  }));
  const publicHash = value => {
    if (typeof value !== "string" || !/^[a-f0-9]{64}$/.test(value)) {
      throw new Error("Invalid public suppression content hash.");
    }
    return value;
  };
  const filename = manifest.emitted?.filename;
  if (typeof filename !== "string" || !/^suppression-[a-f0-9]{12}\.json$/.test(filename)) {
    throw new Error("Invalid public suppression asset filename.");
  }
  return {
    decoder,
    decoderVersion: manifest.decoderVersion,
    emitted: { filename, sha256: publicHash(manifest.emitted?.sha256) },
    schemaVersion: 1,
    source: Object.fromEntries(hashFields.map(field => [field, publicHash(manifest.source?.[field])])),
  };
}

export function verifyPublicSuppressionManifest(manifest) {
  if (!isDeepStrictEqual(manifest, publicSuppressionManifest(manifest))) {
    throw new Error("Suppression manifest contains fields not approved for publication.");
  }
}
