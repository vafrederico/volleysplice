import { deriveLabRemovals, type LabConfiguration, type ProductionEditorLabTask } from "./production-editor-lab.ts";
import { hasVisibleTime } from "./production-editor-lab-draft.ts";

export const LAB_SUPPRESSION_OPTIONS = {
  none: "Off / original model",
  conservative: "Production guard · conservative",
  balanced: "Production guard · balanced",
  aggressive: "Production guard · aggressive",
  direct: "Head only · neural experiment",
} as const;
export type LabSuppressionMode = keyof typeof LAB_SUPPRESSION_OPTIONS;

export function canCombineSuppression(configuration: LabConfiguration): boolean {
  return !configuration.humanReference && !configuration.suppressionPolicy
    && (configuration.id === "production" || configuration.id === "production-replay" || configuration.id === "compact-standalone" || configuration.id.startsWith("neural-"));
}

/** Apply frozen model evidence only. Human truth is used by the separate comparison rail. */
export function withLabSuppression(task: ProductionEditorLabTask, base: LabConfiguration, mode: LabSuppressionMode): LabConfiguration {
  if (mode === "none" || !canCombineSuppression(base) || !task.suppressionSource
      || (mode === "direct" && base.id.startsWith("production"))) return base;
  const source = task.suppressionSource;
  const policy = mode === "direct" ? "aggressive" : mode;
  const ignored = base.ignoredIntervals ?? task.ignoredIntervals;
  // The historical editor recording has an older feature replay. Keep its original
  // production gate paired with that baseline, rather than mixing replay revisions.
  const gated = base.id === "production" ? task.configurations.find(c => c.id === "suppression-aggressive")?.suppression ?? source.gated : source.gated;
  const suggestions = mode === "direct" ? source.decoded.map((region, i) => ({
    ...region, id: `head-only:${i}`, logicalId: `head-only:${i}`, suppressionEventId: `head:${i}`,
    sourceProductionIds: [], eligiblePolicyIds: ["aggressive" as const],
  })) : gated.suggestions;
  // Ignore evidence entirely inside ignored footage, including when a rally crosses its edge.
  const active = suggestions.filter(s => s.eligiblePolicyIds.includes(policy));
  const removed = base.events.filter(event => active.some(s => {
    const intersection = { start: Math.max(event.start, s.start), end: Math.min(event.end, s.end) };
    return intersection.end > intersection.start && hasVisibleTime(intersection, ignored);
  }));
  // Clip evidence to the visible universe so materialization and the queue use the same rule.
  const visibleSuggestions = suggestions.flatMap(s => {
    let pieces = [{ start: s.start, end: s.end }];
    for (const mask of ignored) pieces = pieces.flatMap(p => mask.end <= p.start || mask.start >= p.end ? [p] : [
      ...(mask.start > p.start ? [{ start: p.start, end: mask.start }] : []),
      ...(mask.end < p.end ? [{ start: mask.end, end: p.end }] : []),
    ]);
    return pieces.map((p, i) => ({ ...s, ...p, id: `${s.id}:visible:${i}` }));
  });
  const events = base.events.filter(e => !removed.includes(e));
  return { ...base, id: `${base.id}--suppression-${mode}`, suppressionBaseId: base.id,
    sourceRevision: `${base.sourceRevision ?? task.sourceRevision}:${source.revision}:suppression-v1`,
    label: `${base.label} + ${LAB_SUPPRESSION_OPTIONS[mode]}`,
    description: `${base.description} ${mode === "direct"
      ? "Experimental direct suppression head: no production agreement protection."
      : `Suppression is eligible only in the frozen production ${base.id === "production" ? "baseline" : "comparison replay"}'s one-model-only regions; this is not neural-model agreement.`} Any eligible overlap removes the whole rally by default. Restore harmful removals in the review queue.`,
    draftEvents: base.events, events, suppressionPolicy: policy,
    suppression: { identicalPolicyResults: mode === "direct" || gated.identicalPolicyResults, suggestions: visibleSuggestions },
    removals: deriveLabRemovals(removed, [], ignored).map(r => ({ ...r, suppressionRemoval: true })),
  };
}
