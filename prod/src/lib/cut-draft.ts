import type { IgnoredInterval } from "./product-analysis.ts";
import type { Rally } from "./edit-list.ts";

export const CUT_DRAFT_VERSION = 6 as const;
export const DEFAULT_CUT_PADDING = { before: 2, after: 2 } as const;
export const DEFAULT_CONFIDENCE_REVIEW_THRESHOLD = 0.7;
export const MAX_CUT_PADDING_SECONDS = 10;
export const PLAYBACK_RATES = [1, 2, 4, 8] as const;

export type CutOrigin = "cached-label" | "manual";

export type EditableCut = {
  id: string;
  coreStart: number;
  coreEnd: number;
  keepStart: number;
  keepEnd: number;
  confidence: number;
  included: boolean;
  origin: CutOrigin;
};

export type IgnoredSourceInterval = IgnoredInterval & {
  id: string;
};

export type CutDraft = {
  version: typeof CUT_DRAFT_VERSION;
  analysisId: string;
  recordingId: string;
  sourceRevision: string;
  updatedAt: string;
  beforePaddingSeconds: number;
  afterPaddingSeconds: number;
  pendingManualStart: number | null;
  pendingIgnoreStart: number | null;
  ignoreReason: string;
  cutPreviewEnabled: boolean;
  playbackRate: (typeof PLAYBACK_RATES)[number];
  confidenceReviewThreshold: number;
  cuts: EditableCut[];
  ignoredIntervals: IgnoredSourceInterval[];
};

export type FinalCutInterval = {
  start: number;
  end: number;
  cutIds: string[];
};

export type CutDraftSeed = {
  analysisId: string;
  recordingId: string;
  duration: number;
  rallies: Rally[];
  ignoredIntervals: IgnoredInterval[];
};

function finiteTime(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function hashText(value: string): string {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(36);
}

export function cutSourceRevision(seed: CutDraftSeed): string {
  const source = JSON.stringify({
    duration: Number(seed.duration.toFixed(3)),
    rallies: seed.rallies.map((rally) => [
      rally.id,
      Number(rally.start.toFixed(3)),
      Number(rally.end.toFixed(3)),
    ]),
    ignoredIntervals: seed.ignoredIntervals.map((interval) => [
      Number(interval.start.toFixed(3)),
      Number(interval.end.toFixed(3)),
      interval.reason,
    ]),
  });
  return `fnv1a-${hashText(source)}`;
}

export function cutDraftStorageKey(analysisId: string): string {
  return `volleycut:cut-draft:v${CUT_DRAFT_VERSION}:${encodeURIComponent(analysisId)}`;
}

export function cutDraftStorageKeys(analysisId: string): string[] {
  return [CUT_DRAFT_VERSION, 5, 4, 3, 2, 1].map(
    (version) => `volleycut:cut-draft:v${version}:${encodeURIComponent(analysisId)}`,
  );
}

export function createCutDraft(seed: CutDraftSeed): CutDraft {
  const duration = Math.max(0, seed.duration);
  return {
    version: CUT_DRAFT_VERSION,
    analysisId: seed.analysisId,
    recordingId: seed.recordingId,
    sourceRevision: cutSourceRevision(seed),
    updatedAt: new Date(0).toISOString(),
    beforePaddingSeconds: DEFAULT_CUT_PADDING.before,
    afterPaddingSeconds: DEFAULT_CUT_PADDING.after,
    pendingManualStart: null,
    pendingIgnoreStart: null,
    ignoreReason: "non-game-content",
    cutPreviewEnabled: false,
    playbackRate: 1,
    confidenceReviewThreshold: DEFAULT_CONFIDENCE_REVIEW_THRESHOLD,
    cuts: seed.rallies.map((rally) => ({
      id: rally.id,
      coreStart: clamp(rally.start, 0, duration),
      coreEnd: clamp(rally.end, 0, duration),
      keepStart: clamp(rally.start - DEFAULT_CUT_PADDING.before, 0, duration),
      keepEnd: clamp(rally.end + DEFAULT_CUT_PADDING.after, 0, duration),
      confidence: clamp(rally.confidence, 0, 1),
      included: rally.included,
      origin: "cached-label",
    })),
    ignoredIntervals: seed.ignoredIntervals.map((interval, index) => ({
      id: `I${String(index + 1).padStart(3, "0")}`,
      start: clamp(interval.start, 0, duration),
      end: clamp(interval.end, 0, duration),
      reason: interval.reason,
    })),
  };
}

function validCut(value: unknown, duration: number): value is EditableCut {
  if (!value || typeof value !== "object") return false;
  const cut = value as Partial<EditableCut>;
  return (
    typeof cut.id === "string" &&
    cut.id.length > 0 &&
    finiteTime(cut.coreStart) &&
    finiteTime(cut.coreEnd) &&
    finiteTime(cut.keepStart) &&
    finiteTime(cut.keepEnd) &&
    finiteTime(cut.confidence) &&
    typeof cut.included === "boolean" &&
    (cut.origin === "cached-label" || cut.origin === "manual") &&
    cut.keepStart >= 0 &&
    cut.keepStart <= cut.coreStart &&
    cut.coreStart < cut.coreEnd &&
    cut.coreEnd <= cut.keepEnd &&
    cut.keepEnd <= duration &&
    cut.confidence >= 0 &&
    cut.confidence <= 1
  );
}

function validIgnoredInterval(
  value: unknown,
  duration: number,
): value is IgnoredSourceInterval {
  if (!value || typeof value !== "object") return false;
  const interval = value as Partial<IgnoredSourceInterval>;
  return (
    typeof interval.id === "string" &&
    interval.id.length > 0 &&
    finiteTime(interval.start) &&
    finiteTime(interval.end) &&
    interval.start >= 0 &&
    interval.start < interval.end &&
    interval.end <= duration &&
    typeof interval.reason === "string" &&
    interval.reason.length > 0
  );
}

export function parseCutDraft(raw: string, seed: CutDraftSeed): CutDraft | null {
  try {
    const persisted = JSON.parse(raw) as Partial<Omit<CutDraft, "version">> & {
      version?: unknown;
      paddingSeconds?: unknown;
    };
    const persistedVersion = persisted.version;
    if (
      typeof persistedVersion !== "number" ||
      ![1, 2, 3, 4, 5, CUT_DRAFT_VERSION].includes(persistedVersion)
    ) {
      return null;
    }
    const value: Partial<CutDraft> = {
      ...persisted,
      version: CUT_DRAFT_VERSION,
      beforePaddingSeconds: persistedVersion === 1
        ? 3
        : persistedVersion === 2
          ? persisted.paddingSeconds as number
          : persisted.beforePaddingSeconds,
      afterPaddingSeconds: persistedVersion === 1
        ? 2
        : persistedVersion === 2
          ? persisted.paddingSeconds as number
          : persisted.afterPaddingSeconds,
      pendingManualStart: persistedVersion >= 4 ? persisted.pendingManualStart : null,
      pendingIgnoreStart: persistedVersion >= 4 ? persisted.pendingIgnoreStart : null,
      ignoreReason: persistedVersion >= 4
        ? persisted.ignoreReason
        : "non-game-content",
      cutPreviewEnabled: persistedVersion >= 4 ? persisted.cutPreviewEnabled : false,
      playbackRate: persistedVersion >= 5 ? persisted.playbackRate : 1,
      confidenceReviewThreshold: persistedVersion >= 6
        ? persisted.confidenceReviewThreshold
        : DEFAULT_CONFIDENCE_REVIEW_THRESHOLD,
    };
    if (
      value.version !== CUT_DRAFT_VERSION ||
      value.analysisId !== seed.analysisId ||
      value.recordingId !== seed.recordingId ||
      value.sourceRevision !== cutSourceRevision(seed) ||
      typeof value.updatedAt !== "string" ||
      !finiteTime(value.beforePaddingSeconds) ||
      value.beforePaddingSeconds < 0 ||
      value.beforePaddingSeconds > MAX_CUT_PADDING_SECONDS ||
      !finiteTime(value.afterPaddingSeconds) ||
      value.afterPaddingSeconds < 0 ||
      value.afterPaddingSeconds > MAX_CUT_PADDING_SECONDS ||
      !(
        value.pendingManualStart === null ||
        (finiteTime(value.pendingManualStart) &&
          value.pendingManualStart >= 0 &&
          value.pendingManualStart <= seed.duration)
      ) ||
      !(
        value.pendingIgnoreStart === null ||
        (finiteTime(value.pendingIgnoreStart) &&
          value.pendingIgnoreStart >= 0 &&
          value.pendingIgnoreStart <= seed.duration)
      ) ||
      typeof value.ignoreReason !== "string" ||
      value.ignoreReason.length === 0 ||
      typeof value.cutPreviewEnabled !== "boolean" ||
      !PLAYBACK_RATES.includes(value.playbackRate as (typeof PLAYBACK_RATES)[number]) ||
      !finiteTime(value.confidenceReviewThreshold) ||
      value.confidenceReviewThreshold < 0 ||
      value.confidenceReviewThreshold > 1 ||
      (value.pendingManualStart !== null && value.pendingIgnoreStart !== null) ||
      !Array.isArray(value.cuts) ||
      !Array.isArray(value.ignoredIntervals) ||
      !value.cuts.every((cut) => validCut(cut, seed.duration)) ||
      !value.ignoredIntervals.every((interval) =>
        validIgnoredInterval(interval, seed.duration),
      )
    ) {
      return null;
    }
    const ids = [
      ...value.cuts.map((cut) => cut.id),
      ...value.ignoredIntervals.map((interval) => interval.id),
    ];
    if (new Set(ids).size !== ids.length) return null;
    return value as CutDraft;
  } catch {
    return null;
  }
}

export function applyPaddingToCachedCuts(
  draft: CutDraft,
  beforePaddingSeconds: number,
  afterPaddingSeconds: number,
  duration: number,
): CutDraft {
  const before = clamp(beforePaddingSeconds, 0, MAX_CUT_PADDING_SECONDS);
  const after = clamp(afterPaddingSeconds, 0, MAX_CUT_PADDING_SECONDS);
  return {
    ...draft,
    beforePaddingSeconds: before,
    afterPaddingSeconds: after,
    cuts: draft.cuts.map((cut) =>
      cut.origin === "cached-label"
        ? {
            ...cut,
            keepStart: roundTime(clamp(cut.coreStart - before, 0, duration)),
            keepEnd: roundTime(clamp(cut.coreEnd + after, 0, duration)),
          }
        : cut,
    ),
  };
}

export function playbackFocusCut(
  cuts: readonly EditableCut[],
  playbackTime: number,
): EditableCut | null {
  if (!Number.isFinite(playbackTime)) return null;
  let reached: EditableCut | null = null;
  for (const cut of cuts) {
    if (
      cut.keepStart <= playbackTime &&
      (reached === null || cut.keepStart >= reached.keepStart)
    ) {
      reached = cut;
    }
  }
  return reached;
}

function roundTime(seconds: number): number {
  return Math.round(seconds * 1000) / 1000;
}

function mergeIgnoredIntervals(
  intervals: IgnoredSourceInterval[],
): Array<{ start: number; end: number }> {
  return intervals
    .map(({ start, end }) => ({ start, end }))
    .sort((left, right) => left.start - right.start || left.end - right.end)
    .reduce<Array<{ start: number; end: number }>>((merged, interval) => {
      const previous = merged.at(-1);
      if (!previous || interval.start > previous.end) {
        merged.push({ ...interval });
      } else {
        previous.end = Math.max(previous.end, interval.end);
      }
      return merged;
    }, []);
}

export function effectiveKeptCutIds(draft: CutDraft): string[] {
  const ignored = mergeIgnoredIntervals(draft.ignoredIntervals);
  return draft.cuts
    .filter((cut) => cut.included && cut.keepEnd > cut.keepStart)
    .filter((cut) => {
      const ignoredSeconds = ignored.reduce((total, interval) => {
        const overlap = Math.max(
          0,
          Math.min(cut.keepEnd, interval.end) - Math.max(cut.keepStart, interval.start),
        );
        return total + overlap;
      }, 0);
      return cut.keepEnd - cut.keepStart - ignoredSeconds > 0.000_001;
    })
    .map((cut) => cut.id);
}

export function buildFinalCutIntervals(draft: CutDraft): FinalCutInterval[] {
  const merged = draft.cuts
    .filter((cut) => cut.included && cut.keepEnd > cut.keepStart)
    .sort((left, right) => left.keepStart - right.keepStart || left.keepEnd - right.keepEnd)
    .reduce<FinalCutInterval[]>((intervals, cut) => {
      const previous = intervals.at(-1);
      if (!previous || cut.keepStart > previous.end) {
        intervals.push({ start: cut.keepStart, end: cut.keepEnd, cutIds: [cut.id] });
      } else {
        previous.end = Math.max(previous.end, cut.keepEnd);
        previous.cutIds.push(cut.id);
      }
      return intervals;
  }, []);
  const ignored = mergeIgnoredIntervals(draft.ignoredIntervals);
  const remaining = ignored.reduce<FinalCutInterval[]>((intervals, excluded) => {
    return intervals.flatMap((interval) => {
      if (excluded.end <= interval.start || excluded.start >= interval.end) {
        return [interval];
      }
      const pieces: FinalCutInterval[] = [];
      if (excluded.start > interval.start) {
        pieces.push({
          ...interval,
          end: Math.min(interval.end, excluded.start),
          cutIds: [...interval.cutIds],
        });
      }
      if (excluded.end < interval.end) {
        pieces.push({
          ...interval,
          start: Math.max(interval.start, excluded.end),
          cutIds: [...interval.cutIds],
        });
      }
      return pieces;
    });
  }, merged);
  const cutsById = new Map(draft.cuts.map((cut) => [cut.id, cut]));
  return remaining.map((interval) => ({
    ...interval,
    cutIds: interval.cutIds.filter((id) => {
      const cut = cutsById.get(id);
      return Boolean(
        cut && Math.min(cut.keepEnd, interval.end) - Math.max(cut.keepStart, interval.start) > 0.000_001,
      );
    }),
  }));
}

export function totalFinalCutSeconds(intervals: FinalCutInterval[]): number {
  return intervals.reduce((total, interval) => total + interval.end - interval.start, 0);
}

export function nextFinalCutTime(
  intervals: FinalCutInterval[],
  playbackTime: number,
): number | null {
  for (const interval of intervals) {
    if (playbackTime >= interval.start && playbackTime < interval.end) {
      return playbackTime;
    }
    if (playbackTime < interval.start) return interval.start;
  }
  return null;
}
