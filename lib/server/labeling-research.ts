import { readFile } from "node:fs/promises";
import path from "node:path";
import type { LabelDocument } from "../annotations.ts";
import { parseResearchReferences, type ResearchReference } from "../labeling-research.ts";
import { loadNeuralComparison } from "./neural-comparison.ts";

export async function loadLabelingResearchReferences(document: LabelDocument): Promise<ResearchReference[]> {
  const configured = process.env.VOLLEYCUT_LABELING_RESEARCH_REFERENCES_PATH?.trim();
  const comparison = await loadNeuralComparison(document.recording);
  const legacy = configured ? parseResearchReferences(JSON.parse(await readFile(
    path.resolve(/* turbopackIgnore: true */ configured), "utf8")), document.recording) : [];
  return [...legacy, ...(comparison?.references ?? [])];
}
