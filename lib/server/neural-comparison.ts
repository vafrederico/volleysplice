import { readFile } from "node:fs/promises";
import path from "node:path";
import { createHash } from "node:crypto";
import { parseResearchReferences } from "../labeling-research.ts";
import type { LabelDocument } from "../annotations.ts";

type Entry = { id: string; name: string; file: string; tier: string; modelIds?: string[] };

async function index() {
  const configured = process.env.VOLLEYCUT_NEURAL_COMPARISON_INDEX_PATH?.trim();
  if (!configured) return null;
  const filename = path.resolve(/* turbopackIgnore: true */ configured);
  const value = JSON.parse(await readFile(filename, "utf8"));
  if (value.schemaVersion !== 1 || value.kind !== "volleycut-neural-comparison-index" || !Array.isArray(value.recordings)) {
    throw new Error("Invalid neural comparison index");
  }
  const modelIds: string[] | undefined = value.models?.map((model: { modelId: string }) => model.modelId);
  if (modelIds && (!modelIds.length || new Set(modelIds).size !== modelIds.length || modelIds.some(id => typeof id !== "string"))) {
    throw new Error("Invalid neural comparison model catalog");
  }
  const entries: Entry[] = value.recordings.map((entry: Entry) => {
    if (!entry || !/^[\w-]+$/.test(entry.id) || typeof entry.name !== "string" || typeof entry.tier !== "string"
      || entry.file !== `recordings/${entry.id}.json`) throw new Error("Invalid neural comparison recording");
    if (entry.modelIds !== undefined && (!Array.isArray(entry.modelIds) || !entry.modelIds.length
      || new Set(entry.modelIds).size !== entry.modelIds.length || entry.modelIds.some(id => typeof id !== "string")
      || (modelIds && entry.modelIds.some(id => !modelIds.includes(id))))) {
      throw new Error("Invalid neural comparison recording model catalog");
    }
    return entry;
  });
  if (new Set(entries.map(entry => entry.id)).size !== entries.length) throw new Error("Duplicate neural comparison recording");
  return { root: path.dirname(filename), entries, modelIds };
}

export async function neuralComparisonCatalog() {
  return (await index())?.entries.map(({ id, name, tier }) => ({ id, name, tier })) ?? [];
}

export async function loadNeuralComparison(recording: LabelDocument["recording"]) {
  const manifest = await index();
  const entry = manifest?.entries.find(item => item.id === recording.id);
  if (!manifest || !entry) return null;
  const raw = await readFile(path.join(manifest.root, entry.file), "utf8");
  const references = parseResearchReferences(JSON.parse(raw), recording);
  const expected = entry.modelIds ?? manifest.modelIds;
  if (!references.length || (expected && (references.length !== expected.length || references.some(reference => !expected.includes(reference.modelId))))
      || references.some(reference => !reference.research?.signals.times.length)) {
    throw new Error("Neural comparison predictions must match the model catalog and include signals");
  }
  return { references, revision: createHash("sha256").update(raw).digest("hex"), tier: entry.tier };
}
