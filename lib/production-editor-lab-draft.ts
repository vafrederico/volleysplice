import { createCutDraft, type CutDraft, type CutDraftSeed } from "../components/production-lab/editor/lib/cut-draft.ts";
import type { LabConfiguration, LabEvent, LabInterval, ProductionEditorLabTask } from "./production-editor-lab.ts";

export type LabDecision = { action: string; signature: string };
export type LabDraft = CutDraft & { labDecisions?: Record<string, LabDecision> };

export function labStorageKey(task: ProductionEditorLabTask, configuration: LabConfiguration): string {
  return `volleycut:production-lab:trial:v1:${task.id}:${configuration.humanReference?.revision ?? configuration.sourceRevision ?? task.sourceRevision}:${configuration.id}`;
}

export function hasVisibleTime(event: LabInterval, ignored: LabInterval[]): boolean {
  let ranges = [{ ...event }];
  for (const mask of ignored) ranges = ranges.flatMap(range => mask.end <= range.start || mask.start >= range.end ? [range] : [
    ...(mask.start > range.start ? [{ start: range.start, end: mask.start }] : []),
    ...(mask.end < range.end ? [{ start: mask.end, end: range.end }] : []),
  ]);
  return ranges.some(range => range.end > range.start);
}

export function labSeed(task: ProductionEditorLabTask, configuration: LabConfiguration): CutDraftSeed {
  const ignored = configuration.ignoredIntervals ?? task.ignoredIntervals;
  return {
    analysisId: labStorageKey(task, configuration), recordingId: task.id, duration: task.durationSeconds,
    scoreTrackingEnabled: true,
    rallies: (configuration.draftEvents ?? configuration.events).filter(event => hasVisibleTime(event, ignored)).map(event => ({
      id: event.id, start: event.start, end: event.end, confidence: event.confidence ?? 0,
      included: true, ...(event.agreement ? { agreement: event.agreement } : {}),
    })),
    ignoredIntervals: ignored.map(range => ({ ...range, reason: "ignored-source-footage" })),
  };
}

export function initialLabDraft(task: ProductionEditorLabTask, configuration: LabConfiguration): LabDraft {
  const draft = createCutDraft(labSeed(task, configuration));
  draft.selectedSuppressionPolicy = configuration.suppressionPolicy ?? "none";
  if (configuration.humanReference) {
    draft.reviewedCutIds = draft.cuts.map(cut => cut.id);
    draft.scoreTracking = { ...draft.scoreTracking, ...structuredClone(configuration.humanReference.scoreTracking) };
    return { ...draft, labDecisions: {} };
  }
  draft.scoreTracking.serveMarkers = (configuration.draftEvents ?? configuration.events).filter(event => event.serve && hasVisibleTime(event, task.ignoredIntervals)).map(event => ({
    id: `lab-serve:${event.id}`, rallyId: event.id, timestamp: event.start,
    side: event.serve!.side, modelSide: event.serve!.side, origin: "model", ignorePreviousPoint: false,
  }));
  return { ...draft, labDecisions: {} };
}

export function draftLabEvents(draft: CutDraft, configuration: LabConfiguration): LabEvent[] {
  const sources = [...(configuration.draftEvents ?? configuration.events), ...configuration.events, ...configuration.proposals.flatMap(proposal => [...proposal.before, ...proposal.after]),
    ...configuration.removals.map(removal => removal.parent)];
  return draft.cuts.filter(cut => cut.included).map(cut => {
    const source = sources.find(event => event.id === cut.id);
    const parent = cut.origin === "manual" ? undefined : source ?? sources.find(event => event.start <= cut.coreStart && event.end >= cut.coreEnd);
    const marker = draft.scoreTracking.serveMarkers.find(serve => serve.rallyId === cut.id);
    return { id: cut.id, parentId: parent?.parentId ?? cut.id, start: cut.coreStart, end: cut.coreEnd,
      confidence: cut.confidence, ...(cut.agreement ? { agreement: cut.agreement } : {}),
      ...(marker ? { serve: { time: marker.timestamp, side: marker.side, reason: "Current editor serve marker." } } : {}) };
  });
}

/** Update only changed event geometry; preserve padding, score edits, excluded cuts and all unrelated fields. */
export function draftWithLabEvents(draft: CutDraft, events: LabEvent[]): LabDraft {
  const previous = new Map(draft.cuts.map(cut => [cut.id, cut]));
  const retained = new Set(events.map(event => event.id));
  const cuts = events.map(event => {
    const old = previous.get(event.id);
    return { id: event.id, coreStart: event.start, coreEnd: event.end,
      keepStart: old && old.coreStart === event.start ? old.keepStart : Math.max(draft.analysisStart, event.start - draft.beforePaddingSeconds),
      keepEnd: old && old.coreEnd === event.end ? old.keepEnd : Math.min(draft.analysisEnd, event.end + draft.afterPaddingSeconds),
      included: true, confidence: old?.confidence ?? event.confidence ?? 0, origin: old?.origin ?? "cached-label" as const,
      ...(event.agreement ? { agreement: event.agreement } : {}) };
  });
  const excluded = draft.cuts.filter(cut => !cut.included && !retained.has(cut.id));
  const validIds = new Set([...retained, ...excluded.map(cut => cut.id)]);
  const markers = draft.scoreTracking.serveMarkers.filter(marker => !marker.rallyId || validIds.has(marker.rallyId)).map(marker => {
    const event = events.find(item => item.id === marker.rallyId);
    const old = event ? previous.get(event.id) : undefined;
    if (!event || !old || old.coreStart === event.start) return marker;
    return { ...marker, timestamp: event.start, side: event.serve?.side ?? "review" as const,
      modelSide: event.serve?.side ?? "review" as const };
  });
  for (const event of events) {
    if (!event.serve || markers.some(marker => marker.rallyId === event.id)
      || draft.scoreTracking.removedModelMarkerIds.includes(`lab-serve:${event.id}`)) continue;
    markers.push({ id: `lab-serve:${event.id}`, rallyId: event.id, timestamp: event.start,
      side: event.serve.side, modelSide: event.serve.side, origin: "model", ignorePreviousPoint: false });
  }
  return { ...draft, cuts: [...cuts, ...excluded].sort((a, b) => a.coreStart - b.coreStart),
    reviewedCutIds: draft.reviewedCutIds.filter(id => validIds.has(id)),
    userTouchedCutIds: [...new Set([...draft.userTouchedCutIds.filter(id => validIds.has(id)), ...events.filter(event => {
      const old = previous.get(event.id); return !old || old.coreStart !== event.start || old.coreEnd !== event.end;
    }).map(event => event.id)])],
    scoreTracking: { ...draft.scoreTracking, serveMarkers: markers } };
}

export function decisionSignature(draft: CutDraft, range: LabInterval): string {
  return JSON.stringify(draft.cuts.filter(cut => cut.coreStart <= range.end && cut.coreEnd >= range.start)
    .map(cut => [cut.id, cut.coreStart, cut.coreEnd, cut.included, draft.suppressionDecisionOverrides[`rally:${cut.id}`] ?? null]));
}

export function recordLabDecision(draft: CutDraft, id: string, range: LabInterval, action: string): LabDraft {
  const previous = (draft as LabDraft).labDecisions ?? {};
  return { ...draft, labDecisions: { ...previous, [id]: { action, signature: decisionSignature(draft, range) } } };
}

export function currentLabDecision(draft: CutDraft, id: string, range: LabInterval): string | undefined {
  const decision = (draft as LabDraft).labDecisions?.[id];
  return decision && decision.signature === decisionSignature(draft, range) ? decision.action : undefined;
}
