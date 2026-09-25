import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";
import type { LabelDocument } from "../annotations.ts";
import { parseProductionEditorLabManifest, type LabConfiguration, type ProductionEditorLabTask } from "../production-editor-lab.ts";
import { loadNeuralComparison } from "./neural-comparison.ts";
import { loadEditorSuppression } from "./editor-lab-suppression.ts";

/** Human reference is appended after inference loading; never an input to model configurations. */
export function humanExportConfiguration(document: LabelDocument, source: "draft" | "completed" | "imported"): LabConfiguration {
  const events = document.rallies.map((rally, index) => {
    const sourceId = (rally as typeof rally & { id?: string }).id ?? `H${String(index + 1).padStart(3, "0")}`;
    return { id: `human:${sourceId}`, parentId: `human:${sourceId}`, start: rally.start, end: rally.end };
  });
  const ignoredIntervals = document.ignoredIntervals.map(({ start, end }) => ({ start, end }));
  const scoreTracking = {
    serveMarkers: document.serveMarkers.map((marker, index) => {
      const rally = events.find(event => event.id === `human:${marker.rallyId}`)
        ?? events.find(event => Math.abs(event.start - marker.time) < 1e-6);
      return { id: `human-serve:${index + 1}`, timestamp: marker.time, side: marker.side,
        origin: marker.origin ?? "manual" as const, ...(marker.modelSide ? { modelSide: marker.modelSide } : {}),
        ignorePreviousPoint: (marker as typeof marker & { ignorePreviousPoint?: boolean }).ignorePreviousPoint === true,
        ...(rally ? { rallyId: rally.id } : {}) };
    }),
    sideSwitchMarkers: document.sideSwitches.map((marker, index) => ({ id: `human-switch:${index + 1}`,
      timestamp: marker.time, origin: marker.origin ?? "manual" as const })),
  };
  const revision = createHash("sha256").update(JSON.stringify({ events, ignoredIntervals, scoreTracking, source })).digest("hex");
  return { id: "human-export", label: "Human export", sourcePolicy: "saved-human-labels",
    description: "Saved human export regions, with the same padding controls as the model modes. Lab edits do not change the human labels.",
    events, ignoredIntervals, humanReference: { revision, source, scoreTracking }, proposals: [], removals: [], splits: [] };
}

export async function loadProductionEditorLab(recording: LabelDocument["recording"]): Promise<ProductionEditorLabTask | null> {
  const configured = process.env.VOLLEYCUT_EDITOR_LAB_MANIFEST_PATH?.trim();
  let task: ProductionEditorLabTask | null = null;
  if (configured) {
    const filename = path.resolve(/* turbopackIgnore: true */ configured);
    const raw = await readFile(filename, "utf8");
    task = parseProductionEditorLabManifest(JSON.parse(raw), recording, createHash("sha256").update(raw).digest("hex"));
  }
  const comparison = await loadNeuralComparison(recording);
  if (!comparison) return task;
  const configurations: LabConfiguration[] = comparison.references.map(reference => ({
    id: reference.modelId, label: reference.modelLabel, description: reference.description,
    sourcePolicy: "frozen-recall-sweep", sourceRevision: /^[a-f0-9]{64}$/.test(String(reference.research?.provenance?.uiDraftRevision))
      ? String(reference.research!.provenance!.uiDraftRevision) : comparison.revision,
    signals: reference.research!.signals,
    events: reference.rallies.map((rally, i) => {
      const signals = reference.research!.signals;
      const samples = signals.times.flatMap((time, index) => time >= rally.start && time < rally.end ? [signals.live[index]] : []);
      return { id: `${reference.modelId}:${i + 1}`, parentId: `${reference.modelId}:${i + 1}`,
        start: rally.start, end: rally.end,
        // A descriptive mean live-head score, not a calibrated event probability.
        confidence: samples.length ? samples.reduce((sum, score) => sum + score, 0) / samples.length : 0 };
    }),
    proposals: [], removals: [], splits: [], servingPredictions: [],
  }));
  if (!task) task = { schemaVersion: 1, id: recording.id, name: recording.videoFilename,
    durationSeconds: recording.durationSeconds, mediaUrl: `/api/labeling/tasks/${encodeURIComponent(recording.id)}/video`,
    sourceRevision: comparison.revision, ignoredIntervals: [], configurations: [],
    signals: configurations[0].signals!, serving: [], provenance: { labelBlind: true, sourceHashes: {} } };
  task.configurations.push(...configurations);
  task.provenance.sourceHashes["neural-comparison.json"] = comparison.revision;
  const suppression = await loadEditorSuppression(recording);
  if (suppression) {
    task.suppressionSource = suppression.source;
    const existing = task.configurations.find(c => c.id === "production");
    if (!existing) task.configurations.unshift(suppression.production);
    else if (JSON.stringify(existing.events.map(e => [e.start, e.end])) !== JSON.stringify(suppression.production.events.map(e => [e.start, e.end]))) {
      task.configurations.push({ ...suppression.production, id: "production-replay", label: "Production ensemble · comparison replay",
        description: "Frozen production ensemble on the comparison study's cached features. This is the production agreement reference used for neural suppression combinations. The older Production ensemble mode and its saved edits remain separate." });
    }
  }
  return task;
}
