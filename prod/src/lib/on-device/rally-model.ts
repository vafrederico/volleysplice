import type { NeuralPipelineConfig } from "./neural-contract";

export const RALLY_MODEL_OPTIONS = [
  { value: "high-f1", id: "distilled-large-f1-v1", label: "Balanced · BETA",
    description: "Balances retained play and extra footage with tighter cuts." },
  { value: "high-recall", id: "distilled-large-recall-v1", label: "Maximum coverage · BETA",
    description: "Keeps more possible play, with more extra footage to review." },
  { value: "ensemble", id: "ensemble", label: "Legacy model",
    description: "Uses the previous production detector." },
] as const;
export type RallyModelSelection = typeof RALLY_MODEL_OPTIONS[number]["value"];
export type NeuralRallySelection = Exclude<RallyModelSelection, "ensemble">;
export const DEFAULT_RALLY_MODEL: RallyModelSelection = "high-f1";
export function isRallyModelSelection(value: unknown): value is RallyModelSelection {
  return RALLY_MODEL_OPTIONS.some(option => option.value === value);
}
export function isNeuralModelId(value: string): boolean {
  return RALLY_MODEL_OPTIONS.some(option => option.value !== "ensemble" && option.id === value);
}
export type NeuralAsset = { name: string; sha256: string; sizeBytes: number };
export type NeuralBundle = {
  id: string;
  label: string;
  directory: string;
  files: { encoder: NeuralAsset; temporal: NeuralAsset; pipeline: NeuralAsset };
};
export type NeuralManifest = {
  schemaVersion: 1;
  defaultVariant: NeuralRallySelection;
  variants: Record<NeuralRallySelection, NeuralBundle>;
};

export function parseNeuralManifest(value: unknown): NeuralManifest {
  const manifest = value as NeuralManifest;
  if (manifest?.schemaVersion !== 1 || manifest.defaultVariant !== "high-f1") {
    throw new Error("The rally model catalog is incompatible.");
  }
  for (const selection of ["high-recall", "high-f1"] as const) {
    const bundle = manifest.variants?.[selection];
    if (!bundle || bundle.id !== RALLY_MODEL_OPTIONS.find(v => v.value === selection)?.id
        || bundle.directory !== (selection === "high-recall" ? "recall" : "f1")) {
      throw new Error("The selected rally bundle does not match its model identity.");
    }
    for (const [key, name] of [["encoder", "encoder.onnx"], ["temporal", "temporal.onnx"], ["pipeline", "pipeline.json"]] as const) {
      const asset = bundle.files?.[key];
      if (!asset || asset.name !== name || !/^[a-f0-9]{64}$/.test(asset.sha256)
          || !Number.isSafeInteger(asset.sizeBytes) || asset.sizeBytes <= 0) {
        throw new Error("The rally model asset manifest is incomplete.");
      }
    }
  }
  return manifest;
}

export function parseNeuralConfig(value: unknown, selection: NeuralRallySelection): NeuralPipelineConfig {
  const config = value as NeuralPipelineConfig;
  if (config?.modelIdentity !== "dino-distilled-mobilenet-v3-large-tcn"
      || config.selectionMode !== (selection === "high-recall" ? "recall" : "f1") || config.recallTargetPercent !== 99
      || config.tokenDimension !== 3840 || config.mean?.length !== 112 || config.scale?.length !== 112
      || !config.mean.every(Number.isFinite) || !config.scale.every(v => Number.isFinite(v) && v > 0)
      || !config.decoder || !Number.isFinite(config.decoder.enter) || config.decoder.enter <= .1
      || config.decoder.enter > 1 || !Number.isFinite(config.decoder.smoothing) || config.decoder.smoothing < 0
      || !Number.isFinite(config.decoder.minimum) || config.decoder.minimum <= 0
      || typeof config.decoder.boundary !== "boolean") {
    throw new Error("The rally model preprocessing or decoder configuration is incompatible.");
  }
  return config;
}

export async function verifiedAsset(url: string, asset: NeuralAsset, signal?: AbortSignal): Promise<ArrayBuffer> {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`The selected rally model could not be downloaded (${response.status}).`);
  const bytes = await response.arrayBuffer();
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  const actual = Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, "0")).join("");
  if (bytes.byteLength !== asset.sizeBytes || actual !== asset.sha256) {
    throw new Error("The rally model download failed its integrity check. Reload to retry.");
  }
  return bytes;
}
