import type { RallyLabel } from "./annotations.ts";

export type ResearchSignals = {
  times: number[];
  live: number[];
  serve: number[];
  end: number[];
  keep: number[];
};

export type ResearchBoundaryDetails = {
  candidateId: string;
  componentId: string;
  candidateStart: number;
  candidateEnd: number;
  source: string;
  observed: boolean;
  comparison: "original_parent_start" | "original_parent_end" | "new_boundary";
  previousTime: number | null;
  shiftSeconds: number | null;
  originalNeuralTime: number;
  headShiftSeconds: number;
  sample: { index: number; time: number; live: number; serve: number; end: number; keep: number };
  previousProposedEnd?: number;
  nextProposedStart?: number;
  reviewInstruction: string;
};

export type ResearchBoundaryFlag = {
  id: string;
  kind: "initial_start" | "additional_start" | "end";
  time: number;
  parentId: string;
  priority: number;
  details?: ResearchBoundaryDetails;
};

export type ResearchReviewRegion = {
  id: string;
  parentId: string;
  start: number;
  end: number;
  priority: number;
  recommended: boolean;
  reasons: string[];
};

export type LabelingResearch = {
  recommendation: string;
  signals: ResearchSignals;
  boundaryFlags: ResearchBoundaryFlag[];
  reviewRegions: ResearchReviewRegion[];
  queue: { budgetFraction: number; reviewSeconds: number; selectedParentCount: number };
  provenance?: Record<string, unknown>;
};

export type ResearchExportPolicy = "fixed-production" | "model-predictions";

export type ResearchReference = {
  modelId: string;
  modelLabel: string;
  description: string;
  rallies: RallyLabel[];
  exportRallies: RallyLabel[];
  exportPolicy: ResearchExportPolicy;
  research?: LabelingResearch;
};

type RecordingIdentity = {
  id: string;
  videoFilename: string;
  durationSeconds: number;
  contentSha256?: string;
};

function object(value: unknown, field: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${field} must be an object`);
  return value as Record<string, unknown>;
}

function text(value: unknown, field: string): string {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${field} must be text`);
  return value;
}

function number(value: unknown, field: string, maximum = Number.POSITIVE_INFINITY): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > maximum) {
    throw new Error(`${field} is outside its numeric bounds`);
  }
  return value;
}

function array(value: unknown, field: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`${field} must be an array`);
  return value;
}

function unique(values: string[], field: string): void {
  if (new Set(values).size !== values.length) throw new Error(`${field} contains duplicate IDs`);
}

function signedNumber(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${field} must be a finite number`);
  return value;
}

function boundaryDetails(value: unknown, flag: { kind: string; time: number }, parent: ResearchReviewRegion,
  signals: ResearchSignals, duration: number): ResearchBoundaryDetails {
  const row = object(value, "boundary details");
  const candidateStart = number(row.candidateStart, "candidate start", parent.end);
  const candidateEnd = number(row.candidateEnd, "candidate end", parent.end);
  if (candidateStart < parent.start || candidateEnd <= candidateStart || Math.abs(flag.time - (flag.kind === "end" ? candidateEnd : candidateStart)) > 1e-8) {
    throw new Error("Boundary details do not match the flagged candidate endpoint");
  }
  const comparison = row.comparison;
  if (comparison !== "original_parent_start" && comparison !== "original_parent_end" && comparison !== "new_boundary") throw new Error("Unknown original boundary comparison");
  const previousTime = row.previousTime === null ? null : number(row.previousTime, "previous boundary", duration);
  const shiftSeconds = row.shiftSeconds === null ? null : signedNumber(row.shiftSeconds, "boundary shift");
  if (comparison === "new_boundary" ? previousTime !== null || shiftSeconds !== null
    : previousTime === null || shiftSeconds === null || Math.abs(flag.time - previousTime - shiftSeconds) > 1e-8) {
    throw new Error("Boundary shift does not match its original timestamp");
  }
  if ((comparison === "original_parent_start" && (flag.kind !== "initial_start" || previousTime !== parent.start))
    || (comparison === "original_parent_end" && (flag.kind !== "end" || previousTime !== parent.end))) throw new Error("Boundary comparison must reference the original production endpoint");
  const sample = object(row.sample, "boundary sample");
  const index = number(sample.index, "sample index");
  if (!Number.isInteger(index) || index !== nearestSignalIndex(signals.times, flag.time)) throw new Error("Boundary sample must be nearest actual native timestamp");
  const parsedSample = { index, time: number(sample.time, "sample time", duration), live: 0, serve: 0, end: 0, keep: 0 };
  if (parsedSample.time !== signals.times[index]) throw new Error("Boundary sample timestamp does not match signals");
  for (const head of ["live", "serve", "end", "keep"] as const) {
    parsedSample[head] = number(sample[head], `boundary ${head} score`, 1);
    if (parsedSample[head] !== signals[head][index]) throw new Error("Boundary scores do not match native signal sample");
  }
  const originalNeuralTime = number(row.originalNeuralTime, "original compact boundary", duration);
  const headShiftSeconds = signedNumber(row.headShiftSeconds, "head refinement shift");
  if (Math.abs(flag.time - originalNeuralTime - headShiftSeconds) > 1e-8 || typeof row.observed !== "boolean") throw new Error("Invalid model boundary provenance");
  const previousProposedEnd = row.previousProposedEnd === undefined ? undefined : number(row.previousProposedEnd, "previous proposed end", candidateStart);
  const nextProposedStart = row.nextProposedStart === undefined ? undefined : number(row.nextProposedStart, "next proposed start", parent.end);
  if ((previousProposedEnd !== undefined && previousProposedEnd < parent.start) || (nextProposedStart !== undefined && nextProposedStart < candidateEnd)) throw new Error("Adjacent proposed boundaries overlap");
  return { candidateId: text(row.candidateId, "candidate ID"), componentId: text(row.componentId, "component ID"), candidateStart, candidateEnd,
    source: text(row.source, "boundary source"), observed: row.observed, comparison, previousTime, shiftSeconds,
    originalNeuralTime, headShiftSeconds, sample: parsedSample,
    ...(previousProposedEnd === undefined ? {} : { previousProposedEnd }), ...(nextProposedStart === undefined ? {} : { nextProposedStart }),
    reviewInstruction: text(row.reviewInstruction, "boundary review instruction") };
}

function ranges(value: unknown, duration: number, field: string): RallyLabel[] {
  const result = array(value, field).map((entry, i) => {
    const row = object(entry, `${field}[${i}]`);
    const start = number(row.start, `${field}.start`, duration);
    const end = number(row.end, `${field}.end`, duration);
    if (end <= start) throw new Error(`${field} contains an empty interval`);
    return {
      start, end,
      tags: row.tags === undefined ? ["ai-reference", "research-preview"] : array(row.tags, `${field}.tags`).map(v => text(v, "tag")),
      ...(typeof row.notes === "string" ? { notes: row.notes } : {}),
    };
  }).sort((a, b) => a.start - b.start || a.end - b.end);
  if (result.some((r, i) => i > 0 && result[i - 1].end > r.start)) throw new Error(`${field} contains overlapping rally identities`);
  return result;
}

function research(value: unknown, duration: number): LabelingResearch {
  const row = object(value, "research");
  const raw = object(row.signals, "research.signals");
  const times = array(raw.times, "signals.times").map(v => number(v, "signal timestamp", duration));
  if (times.length > 200_000 || times.some((t, i) => i > 0 && t <= times[i - 1])) throw new Error("Signal timestamps must increase strictly");
  const signals: ResearchSignals = { times, live: [], serve: [], end: [], keep: [] };
  for (const head of ["live", "serve", "end", "keep"] as const) {
    signals[head] = array(raw[head], `signals.${head}`).map(v => number(v, `signals.${head}`, 1));
    if (signals[head].length !== times.length) throw new Error("Every signal head must align with actual timestamps");
  }
  const reviewRegions = array(row.reviewRegions, "reviewRegions").map(value => {
    const region = object(value, "review region");
    const start = number(region.start, "review start", duration);
    const end = number(region.end, "review end", duration);
    if (end <= start || typeof region.recommended !== "boolean") throw new Error("Invalid review region");
    return { id: text(region.id, "review ID"), parentId: text(region.parentId, "parent ID"), start, end,
      priority: number(region.priority, "review priority"), recommended: region.recommended,
      reasons: array(region.reasons, "review reasons").map(v => text(v, "review reason")) };
  });
  unique(reviewRegions.map(r => r.id), "Review regions");
  unique(reviewRegions.map(r => r.parentId), "Review parents");
  const boundaryFlags = array(row.boundaryFlags, "boundaryFlags").map(value => {
    const flag = object(value, "boundary flag");
    if (!["initial_start", "additional_start", "end"].includes(String(flag.kind))) throw new Error("Unknown boundary flag kind");
    const parentId = text(flag.parentId, "boundary parent");
    const time = number(flag.time, "boundary timestamp", duration);
    const parent = reviewRegions.find(r => r.parentId === parentId);
    if (!parent || time < parent.start || time > parent.end) throw new Error("Boundary flag lies outside its review parent");
    return { id: text(flag.id, "boundary ID"), kind: flag.kind as ResearchBoundaryFlag["kind"],
      time, parentId, priority: number(flag.priority, "boundary priority"),
      ...(flag.details === undefined ? {} : { details: boundaryDetails(flag.details, { kind: String(flag.kind), time }, parent, signals, duration) }) };
  });
  unique(boundaryFlags.map(f => f.id), "Boundary flags");
  const queue = object(row.queue, "review queue");
  const selectedParentCount = number(queue.selectedParentCount, "selected parent count");
  if (!Number.isInteger(selectedParentCount) || selectedParentCount !== reviewRegions.filter(r => r.recommended).length) {
    throw new Error("Recommended review parent count does not match the queue");
  }
  return { recommendation: text(row.recommendation, "recommendation"), signals, boundaryFlags, reviewRegions,
    queue: { budgetFraction: number(queue.budgetFraction, "review budget", 1),
      reviewSeconds: number(queue.reviewSeconds, "review duration", duration), selectedParentCount },
    ...(row.provenance === undefined ? {} : { provenance: object(row.provenance, "provenance") }) };
}

export function parseResearchReferences(value: unknown, recording: RecordingIdentity): ResearchReference[] {
  const root = object(value, "research manifest");
  if (root.schemaVersion !== 1 || root.kind !== "volleycut-labeling-research-references") throw new Error("Unknown research reference manifest");
  const recordings = array(root.recordings, "research recordings").map(v => object(v, "research recording"));
  const matches = recordings.filter(r => r.recordingId === recording.id);
  if (!matches.length) return [];
  if (matches.length !== 1) throw new Error("Duplicate research recording");
  const row = matches[0];
  if (row.videoFilename !== recording.videoFilename || Math.abs(number(row.durationSeconds, "research duration") - recording.durationSeconds) > .1
      || (row.contentSha256 !== undefined && row.contentSha256 !== recording.contentSha256)) {
    throw new Error("Research reference does not match this recording");
  }
  const references = array(row.references, "research references").map((value): ResearchReference => {
    const reference = object(value, "research reference");
    if (reference.exportPolicy !== "fixed-production" && reference.exportPolicy !== "model-predictions") throw new Error("Research reference must declare its export policy");
    const rallies = ranges(reference.rallies, recording.durationSeconds, "research rallies");
    const exportRallies = ranges(reference.exportRallies, recording.durationSeconds, "export cores");
    if (reference.exportPolicy === "model-predictions" && (rallies.length !== exportRallies.length
      || rallies.some((rally, index) => rally.start !== exportRallies[index].start || rally.end !== exportRallies[index].end))) {
      throw new Error("Standalone export cores must match its own model predictions");
    }
    return { modelId: text(reference.modelId, "research model ID"), modelLabel: text(reference.modelLabel, "research label"),
      description: text(reference.description, "research description"),
      rallies, exportRallies,
      exportPolicy: reference.exportPolicy,
      ...(reference.research === undefined ? {} : { research: research(reference.research, recording.durationSeconds) }) };
  });
  unique(references.map(r => r.modelId), "Research references");
  return references;
}

export function referenceExportCore(reference: { rallies: RallyLabel[]; exportRallies?: RallyLabel[] }): RallyLabel[] {
  return reference.exportRallies ?? reference.rallies;
}

export function nearestSignalIndex(times: readonly number[], time: number): number {
  if (!times.length) return -1;
  let low = 0;
  let high = times.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (times[middle] < time) low = middle + 1;
    else high = middle;
  }
  if (low === 0) return 0;
  if (low === times.length) return times.length - 1;
  return time - times[low - 1] <= times[low] - time ? low - 1 : low;
}
