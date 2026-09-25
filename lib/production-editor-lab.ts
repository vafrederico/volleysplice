import type { ResearchSignals } from "./labeling-research.ts";
import type { ScoreTracking } from "../components/production-lab/editor/lib/score-tracking.ts";
import { parseServingPredictions, type ServingPrediction } from "./labeling-serving.ts";

export type LabInterval = { start: number; end: number };
export type LabServe = { time: number; side: "near" | "far" | "review"; confidence?: number; reason: string };
export type LabEvent = LabInterval & {
  id: string;
  parentId: string;
  confidence?: number;
  agreement?: "both-models" | "all-labels-v2-only" | "previous-production-only";
  serve?: LabServe;
};
export type LabBoundaryProposal = {
  id: string; parentId: string; before: LabEvent[]; after: LabEvent[]; reason: string;
  recommended?: boolean; priority?: number;
};
export type LabRemoval = LabInterval & {
  suppressionRemoval?: boolean;
  id: string; parentId: string; kind: "prefix" | "suffix" | "gap" | "whole";
  parent: LabEvent;
  leftEventId?: string; rightEventId?: string;
};
export type LabSplit = LabInterval & {
  id: string; parentId: string; leftEventId: string; rightEventId: string;
};
export type LabConfiguration = {
  suppressionBaseId?: string;
  id: string; label: string; description: string; sourcePolicy: string;
  events: LabEvent[]; proposals: LabBoundaryProposal[]; removals: LabRemoval[]; splits: LabSplit[];
  draftEvents?: LabEvent[];
  sourceRevision?: string;
  signals?: ResearchSignals;
  servingPredictions?: ServingPrediction[];
  ignoredIntervals?: LabInterval[];
  humanReference?: { revision: string; source: "draft" | "completed" | "imported"; scoreTracking: Pick<ScoreTracking, "serveMarkers" | "sideSwitchMarkers"> };
  suppressionPolicy?: "conservative" | "balanced" | "aggressive";
  suppression?: { identicalPolicyResults: boolean; suggestions: Array<LabInterval & {
    id: string; logicalId: string; suppressionEventId: string; score: number; sourceProductionIds: string[];
    eligiblePolicyIds: Array<"conservative" | "balanced" | "aggressive">;
  }> };
};
export type ProductionEditorLabTask = {
  suppressionSource?: {
    revision: string;
    gated: NonNullable<LabConfiguration["suppression"]>;
    decoded: Array<LabInterval & { score: number }>;
  };
  schemaVersion: 1; id: string; name: string; durationSeconds: number; mediaUrl: string;
  sourceRevision: string; ignoredIntervals: LabInterval[]; configurations: LabConfiguration[];
  signals: ResearchSignals; serving: ServingPrediction[];
  provenance: { labelBlind: true; labelBlindScope?: "model-configurations"; sourceHashes: Record<string, string> };
};
export type LabEditResult = { ok: true; events: LabEvent[] } | { ok: false; events: LabEvent[]; error: string };

const EPSILON = 1e-6;
const close = (a: number, b: number) => Math.abs(a - b) <= EPSILON;
const ordered = (events: LabEvent[]) => events.toSorted((a, b) => a.start - b.start || a.end - b.end);
const conflict = (events: LabEvent[], error: string): LabEditResult => ({ ok: false, events, error });
const reviewStart = (event: LabEvent, start: number): LabEvent => ({ ...event, start,
  serve: close(start, event.start) ? event.serve : { time: start, side: "review", reason: "Start changed; confirm serve and serving side." } });

/** Raw core differences. Padding and gap joining must never hide a review decision. */
export function deriveLabRemovals(parents: LabEvent[], next: LabEvent[], ignored: LabInterval[] = []): LabRemoval[] {
  const result: LabRemoval[] = [];
  for (const parent of parents) {
    const children = ordered(next.filter(event => event.parentId === parent.parentId && event.end > parent.start && event.start < parent.end));
    let cursor = parent.start;
    let left: LabEvent | undefined;
    const append = (start: number, end: number, right?: LabEvent) => {
      if (end - start <= EPSILON) return;
      let pieces: LabInterval[] = [{ start, end }];
      for (const mask of ignored) pieces = pieces.flatMap(piece => mask.end <= piece.start || mask.start >= piece.end ? [piece] : [
        ...(mask.start > piece.start ? [{ start: piece.start, end: mask.start }] : []),
        ...(mask.end < piece.end ? [{ start: mask.end, end: piece.end }] : []),
      ]);
      for (const piece of pieces) {
        const adjacentLeft = left && close(left.end, piece.start) ? left : undefined;
        const adjacentRight = right && close(right.start, piece.end) ? right : undefined;
        result.push({ ...piece, id: `removed:${parent.id}:${result.length + 1}`, parentId: parent.parentId,
          parent, kind: adjacentLeft && adjacentRight ? "gap" : adjacentLeft ? "suffix" : adjacentRight ? "prefix" : "whole",
          ...(adjacentLeft ? { leftEventId: adjacentLeft.id } : {}), ...(adjacentRight ? { rightEventId: adjacentRight.id } : {}) });
      }
    };
    for (const child of children) {
      append(cursor, Math.min(child.start, parent.end), child);
      cursor = Math.max(cursor, child.end); left = child;
    }
    append(cursor, parent.end);
  }
  return result;
}

export function deriveLabSplits(events: LabEvent[]): LabSplit[] {
  const parents = new Set(events.map(event => event.parentId));
  return [...parents].flatMap(parentId => {
    const children = ordered(events.filter(event => event.parentId === parentId));
    return children.slice(1).map((right, index) => ({ id: `split:${children[index].id}:${right.id}`, parentId,
      leftEventId: children[index].id, rightEventId: right.id, start: children[index].end, end: right.start }));
  });
}

/** Restore only the selected raw removal. Other endpoints and other rally IDs survive. */
export function restoreLabRemoval(events: LabEvent[], removal: LabRemoval): LabEditResult {
  if (events.some(event => event.start < removal.end - EPSILON && event.end > removal.start + EPSILON)) {
    return conflict(events, "This removed interval overlaps an edited rally. Adjust it manually or undo your last change first.");
  }
  const candidates = events.filter(event => event.parentId === removal.parentId);
  const left = candidates.find(event => close(event.end, removal.start));
  const right = candidates.find(event => close(event.start, removal.end));
  if ((removal.kind === "prefix" && !right) || (removal.kind === "suffix" && !left)
    || (removal.kind === "gap" && (!left || !right))) {
    return conflict(events, "The adjacent boundary has changed. Adjust this interval manually to preserve your edits.");
  }
  let restored: LabEvent;
  const removedIds = new Set<string>();
  if (removal.kind === "gap" && left && right) {
    restored = { ...left, end: right.end }; removedIds.add(left.id); removedIds.add(right.id);
  } else if (removal.kind === "prefix" && right) {
    restored = reviewStart(right, removal.start); removedIds.add(right.id);
    if (close(removal.start, removal.parent.start)) restored.serve = removal.parent.serve;
  } else if (removal.kind === "suffix" && left) {
    restored = { ...left, end: removal.end }; removedIds.add(left.id);
  } else {
    const id = close(removal.start, removal.parent.start) && close(removal.end, removal.parent.end) ? removal.parent.id : `restored:${removal.id}`;
    if (events.some(event => event.id === id)) return conflict(events, "The original rally ID is already in use. Adjust this interval manually.");
    restored = { ...reviewStart(removal.parent, removal.start), id, end: removal.end };
  }
  return { ok: true, events: ordered([...events.filter(event => !removedIds.has(event.id)), restored]) };
}

/** A guidance proposal replaces only an untouched original parent. */
export function applyLabProposal(events: LabEvent[], proposal: LabBoundaryProposal): LabEditResult {
  const current = events.filter(event => event.parentId === proposal.parentId);
  if (current.length !== proposal.before.length || proposal.before.some(before => !current.some(event => event.id === before.id
    && close(event.start, before.start) && close(event.end, before.end)))) {
    return conflict(events, "This rally has changed since the proposal. Apply boundaries manually to preserve your edits.");
  }
  const untouched = events.filter(event => event.parentId !== proposal.parentId);
  if (proposal.after.some(next => untouched.some(event => next.start < event.end - EPSILON && next.end > event.start + EPSILON))) {
    return conflict(events, "The proposal overlaps another edited rally. Apply boundaries manually.");
  }
  return { ok: true, events: ordered([...untouched, ...proposal.after.map(event => ({ ...event }))]) };
}

/** Undo an internal split even when padding makes its removed export gap invisible. */
export function undoLabSplit(events: LabEvent[], split: LabSplit): LabEditResult {
  const parent = events.find(event => event.parentId === split.parentId);
  if (!parent) return conflict(events, "This split no longer has an adjacent rally.");
  return restoreLabRemoval(events, { ...split, parent, kind: "gap" });
}

type RecordingIdentity = { id: string; videoFilename: string; durationSeconds: number; contentSha256?: string };
const object = (value: unknown): Record<string, unknown> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid editor lab manifest object");
  return value as Record<string, unknown>;
};
const list = (value: unknown): unknown[] => { if (!Array.isArray(value)) throw new Error("Invalid editor lab array"); return value; };
const text = (value: unknown): string => { if (typeof value !== "string" || !value) throw new Error("Invalid editor lab text"); return value; };
const number = (value: unknown, max = Infinity): number => {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > max) throw new Error("Invalid editor lab number"); return value;
};
const interval = (value: unknown, duration: number): LabInterval => {
  const row = object(value); const start = number(row.start, duration); const end = number(row.end, duration);
  if (end <= start) throw new Error("Invalid editor lab interval"); return { start, end };
};

/** Rebuild the sandbox exclusively from allowlisted prediction fields. Unknown fields never reach the client. */
export function parseProductionEditorLabManifest(value: unknown, recording: RecordingIdentity, sourceRevision: string): ProductionEditorLabTask | null {
  const root = object(value);
  if (root.schemaVersion !== 1 || root.kind !== "volleycut-production-editor-lab") throw new Error("Unknown editor lab manifest");
  const identity = object(root.recording);
  if (identity.id !== recording.id) return null;
  if (identity.videoFilename !== recording.videoFilename || Math.abs(number(identity.durationSeconds) - recording.durationSeconds) > .01
    || identity.contentSha256 !== recording.contentSha256) throw new Error("Editor lab recording identity mismatch");
  const duration = recording.durationSeconds;
  const ignoredIntervals = list(root.ignoredIntervals).map(row => interval(row, duration));
  const serving = parseServingPredictions(root.serving, duration);
  const serveFor = (id: string, start: number, predictions = serving): LabServe | undefined => {
    const prediction = predictions.find(row => row.id === id && close(row.anchor, start));
    if (!prediction) return { time: start, side: "review", reason: "New or moved start; confirm serve and serving side." };
    if (prediction.verdict === "not-serve") return undefined;
    return { time: start, side: prediction.verdict === "review" ? "review" : prediction.side,
      confidence: prediction.side === "near" ? prediction.nearProbability : 1 - prediction.nearProbability,
      reason: prediction.verdict === "review" ? "Production serving-side model requests review." : "Production serving-side prediction at this rally start." };
  };
  const events = (values: unknown, predictions?: ServingPrediction[]): LabEvent[] => {
    const parsed = list(values).map(value => {
      const row = object(value); const geometry = interval(row, duration); const id = text(row.id);
      const parentId = row.parentId === undefined ? id : text(row.parentId);
      const agreement = ["both-models", "all-labels-v2-only", "previous-production-only"].includes(String(row.agreement))
        ? row.agreement as LabEvent["agreement"] : undefined;
      if (predictions && !predictions.some(row => row.id === id && close(row.anchor, geometry.start))) throw new Error("Neural serving prediction does not match rally geometry");
      return { ...geometry, id, parentId, serve: serveFor(predictions ? id : parentId, geometry.start, predictions),
        ...(row.confidence === undefined ? {} : { confidence: number(row.confidence, 1) }), ...(agreement ? { agreement } : {}) };
    });
    if (new Set(parsed.map(event => event.id)).size !== parsed.length) throw new Error("Duplicate editor lab event ID");
    const sorted = ordered(parsed);
    if (sorted.some((event, i) => i > 0 && sorted[i - 1].end > event.start + EPSILON)) throw new Error("Overlapping editor lab event identities");
    return sorted;
  };
  const production = events(root.productionEvents);
  const policies = object(root.policies);
  const policyEvents = (id: string): LabEvent[] => list(object(policies[id]).core).map(value => {
    const core = interval(value, duration);
    const source = production.find(event => close(event.start, core.start) && close(event.end, core.end));
    if (!source) throw new Error("Suppression must retain original whole-rally identities");
    return source;
  });
  const aggressive = policyEvents("aggressive");
  const neuralServing = root.neuralServing === undefined ? null : object(root.neuralServing);
  const boundaryServing = neuralServing ? parseServingPredictions(neuralServing.boundary, duration) : undefined;
  const compactServing = neuralServing ? parseServingPredictions(neuralServing.compact, duration) : undefined;
  const boundary = events(root.boundaryEvents, boundaryServing);
  if (boundary.some(event => !aggressive.some(parent => parent.id === event.parentId && parent.start <= event.start + EPSILON && parent.end >= event.end - EPSILON))) {
    throw new Error("Boundary candidate outside its production parent");
  }
  const compact = events(root.compactEvents, compactServing);
  const guidanceReviewParents = root.guidanceReviewParents === undefined ? null : list(root.guidanceReviewParents).map(value => {
    const row = object(value);
    if (typeof row.recommended !== "boolean") throw new Error("Invalid guidance recommendation");
    return { parentId: text(row.parentId), priority: number(row.priority, 1), recommended: row.recommended };
  });
  const proposals: LabBoundaryProposal[] = aggressive.flatMap(parent => {
    const after = boundary.filter(event => event.parentId === parent.id);
    if (after.length === 1 && close(after[0].start, parent.start) && close(after[0].end, parent.end)) return [];
    const review = guidanceReviewParents?.find(row => row.parentId === parent.id);
    return [{ id: `proposal:${parent.id}`, parentId: parent.id, before: [parent], after,
      ...(review ? { recommended: review.recommended, priority: review.priority } : {}),
      reason: after.length > 1 ? "Compact proposes separate rallies with independent starts and ends." : "Compact proposes a revised start and end." }];
  }).filter(proposal => proposal.before.some(event => !ignoredIntervals.some(mask => mask.start <= event.start && mask.end >= event.end)));
  const config = (id: string, label: string, description: string, sourcePolicy: string, configured: LabEvent[]): LabConfiguration => ({
    id, label, description, sourcePolicy, events: configured, proposals: [], removals: [], splits: [],
  });
  const configurations: LabConfiguration[] = [config("production", "Production ensemble", "Current ensemble without suppression.", "none", production)];
  const rawSuppression = object(root.suppression);
  const suppression: NonNullable<LabConfiguration["suppression"]> = {
    identicalPolicyResults: rawSuppression.identicalPolicyResults === true,
    suggestions: list(rawSuppression.suggestions).map(value => {
      const row = object(value);
      const eligiblePolicyIds = list(row.eligiblePolicyIds).map(value => {
        if (value !== "conservative" && value !== "balanced" && value !== "aggressive") throw new Error("Invalid suppression policy");
        return value;
      });
      return { ...interval(row, duration), id: text(row.id), logicalId: text(row.logicalId), suppressionEventId: text(row.suppressionEventId),
        score: number(row.score, 1), sourceProductionIds: list(row.sourceProductionIds).map(text), eligiblePolicyIds };
    }),
  };
  const conservative = policyEvents("conservative");
  const balanced = policyEvents("balanced");
  for (const [id, label, selected] of [["conservative", "Suppression · conservative", conservative],
    ...(JSON.stringify(conservative) === JSON.stringify(balanced) ? [] : [["balanced", "Suppression · balanced", balanced]]),
    ["aggressive", "Suppression · aggressive", aggressive]] as Array<["conservative" | "balanced" | "aggressive", string, LabEvent[]]>) {
    configurations.push({ ...config(`suppression-${id}`, label, "Production suppression starts applied; keep any removed rally to restore it.", id, selected),
      draftEvents: production, suppressionPolicy: id, suppression, removals: deriveLabRemovals(production, selected, ignoredIntervals) });
  }
  configurations.push({ ...config("compact-guidance", "Compact review", "Production aggressive suppression cores stay in place until you apply a compact boundary proposal.", "head_refined-guidance", aggressive),
    proposals: guidanceReviewParents ? proposals.filter(proposal => guidanceReviewParents.some(row => row.parentId === proposal.parentId)) : proposals });
  configurations.push({ ...config("boundary-undo", "Boundary edits + removal review", "Compact boundaries start applied. Review every raw removal and proposed split, restoring only changes you reject.", "head_refined", boundary),
    removals: deriveLabRemovals(aggressive, boundary, ignoredIntervals), splits: deriveLabSplits(boundary), proposals });
  configurations.push(config("compact-standalone", "Compact standalone", "Compact predictions use their own rally boundaries. Confirm serving sides for these new anchors.", "short-boost-3407-outer0", compact));
  for (const configuration of configurations) {
    if (boundaryServing && ["compact-guidance", "boundary-undo"].includes(configuration.id)) configuration.servingPredictions = boundaryServing;
    if (compactServing && configuration.id === "compact-standalone") {
      configuration.servingPredictions = compactServing;
      configuration.description = "Compact's own rally boundaries, with serving-side predictions computed at those starts.";
    }
    if (typeof root.modelGeometryRevision === "string" && (configuration.id === "production" || configuration.suppressionPolicy)) configuration.sourceRevision = root.modelGeometryRevision;
  }
  const rawSignals = object(root.signals);
  const signals: ResearchSignals = { times: list(rawSignals.times).map(value => number(value, duration)), live: [], serve: [], end: [], keep: [] };
  if (!signals.times.length || signals.times.length > 200_000 || signals.times.some((value, i) => i > 0 && value <= signals.times[i - 1])) throw new Error("Invalid native signal timestamps");
  for (const head of ["live", "serve", "end", "keep"] as const) {
    signals[head] = list(rawSignals[head]).map(value => number(value, 1));
    if (signals[head].length !== signals.times.length) throw new Error("Unaligned native signal samples");
  }
  const provenance = object(root.provenance);
  if (provenance.labelBlind !== true) throw new Error("Editor lab inputs must be label blind");
  const sourceHashes = Object.fromEntries(Object.entries(object(provenance.sourceHashes)).map(([name, value]) => {
    if (!/^[\w./-]+$/.test(name) || name.includes("..") || name.startsWith("/") || !/^[a-f0-9]{64}$/.test(String(value))) throw new Error("Invalid model source fingerprint");
    return [name, text(value)];
  }));
  return { schemaVersion: 1, id: recording.id, name: recording.videoFilename, durationSeconds: duration,
    mediaUrl: `/api/labeling/tasks/${encodeURIComponent(recording.id)}/video`, sourceRevision, ignoredIntervals, configurations, signals, serving,
    provenance: { labelBlind: true, sourceHashes } };
}
