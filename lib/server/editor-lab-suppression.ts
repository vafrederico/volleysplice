import { readFile } from "node:fs/promises";
import path from "node:path";
import type { LabelDocument } from "../annotations.ts";
import type { LabConfiguration, LabEvent, ProductionEditorLabTask } from "../production-editor-lab.ts";

/** Private locations remain server-only. Copy only validated prediction fields into the response. */
export function parseEditorSuppression(value: unknown, recording: LabelDocument["recording"]): {
  source: NonNullable<ProductionEditorLabTask["suppressionSource"]>; production: LabConfiguration;
} | null {
  const fail = (): never => { throw new Error("Invalid editor suppression evidence"); };
  const doc = value as { schemaVersion?: number; kind?: string; labelsUsed?: boolean; records?: Array<Record<string, unknown>> };
  if (!doc || doc.schemaVersion !== 1 || doc.kind !== "volleycut-editor-suppression-index" || doc.labelsUsed !== false || !Array.isArray(doc.records)) fail();
  const matches = doc.records!.filter(r => r.recordingId === recording.id);
  if (!matches.length) return null;
  if (matches.length !== 1) fail();
  const row = matches[0];
  if (row.contentSha256 !== recording.contentSha256 || Math.abs(Number(row.durationSeconds) - recording.durationSeconds) > .01
      || !Number.isFinite(row.durationSeconds) || !/^[a-f0-9]{64}$/.test(String(row.revision))) fail();
  const interval = (v: unknown) => {
    const r = v as { start: number; end: number };
    if (!r || !Number.isFinite(r.start) || !Number.isFinite(r.end) || r.start < 0 || r.end <= r.start || r.end > recording.durationSeconds + .01) fail();
    return { start: r.start, end: Math.min(r.end, recording.durationSeconds) };
  };
  const score = (v: unknown) => { if (typeof v !== "number" || !Number.isFinite(v) || v < 0 || v > 1) fail(); return v as number; };
  const string = (v: unknown) => { if (typeof v !== "string" || !v.length) fail(); return v as string; };
  const array = (v: unknown): unknown[] => Array.isArray(v) ? v : fail();
  const gated = row.gated as NonNullable<LabConfiguration["suppression"]>;
  if (!gated || typeof gated.identicalPolicyResults !== "boolean") fail();
  const suggestions = array(gated.suggestions).map(v => {
    const s = v as NonNullable<LabConfiguration["suppression"]>["suggestions"][number];
    const policies = array(s.eligiblePolicyIds);
    if (!policies.length || policies.some(p => !["conservative", "balanced", "aggressive"].includes(String(p)))) fail();
    return { ...interval(s), id: string(s.id), logicalId: string(s.logicalId), suppressionEventId: string(s.suppressionEventId),
      score: score(s.score), sourceProductionIds: array(s.sourceProductionIds).map(string), eligiblePolicyIds: policies as typeof s.eligiblePolicyIds };
  });
  const decoded = array(row.decoded).map(v => ({ ...interval(v), score: score((v as { score: number }).score) }));
  const events: LabEvent[] = array(row.productionEvents).map(v => {
    const e = v as LabEvent;
    if (e.agreement && !["both-models", "all-labels-v2-only", "previous-production-only"].includes(e.agreement)) fail();
    return { ...interval(e), id: string(e.id), parentId: string(e.id), confidence: score(e.confidence), ...(e.agreement ? { agreement: e.agreement } : {}) };
  });
  if (new Set(events.map(e => e.id)).size !== events.length || new Set(suggestions.map(s => s.id)).size !== suggestions.length) fail();
  return { source: { revision: String(row.revision), gated: { identicalPolicyResults: gated.identicalPolicyResults, suggestions }, decoded },
    production: { id: "production", label: "Production ensemble", description: "Frozen production ensemble before suppression. Select a suppression combination to compare removals.",
      sourcePolicy: "frozen-production-union", sourceRevision: String(row.revision), events, proposals: [], removals: [], splits: [] } };
}

export async function loadEditorSuppression(recording: LabelDocument["recording"]) {
  const configured = process.env.VOLLEYCUT_EDITOR_LAB_SUPPRESSION_INDEX_PATH?.trim();
  if (!configured) return null;
  return parseEditorSuppression(JSON.parse(await readFile(path.resolve(/* turbopackIgnore: true */ configured), "utf8")), recording);
}
