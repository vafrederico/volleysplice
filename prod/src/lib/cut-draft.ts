import type { Rally } from "./edit-list.ts";
import {
  SUPPRESSION_POLICY_CONTRACT_VERSION,
  SUPPRESSION_POLICY_IDS,
  type SuppressionPolicyId,
  type SuppressionSuggestion,
} from "./on-device/suppression-policy.ts";
import type { OnDeviceSuppression } from "./on-device/types.ts";
import type { IgnoredInterval } from "./product-analysis.ts";
import {
  createScoreTracking,
  isValidScoreTracking,
  migrateScoreTracking,
  type ScoreTracking,
} from "./score-tracking.ts";

export const CUT_DRAFT_VERSION = 12 as const;
export const DEFAULT_CUT_PADDING = { before: 2, after: 2 } as const;
export const DEFAULT_JOIN_GAP_SECONDS = 3;
export const DEFAULT_CONFIDENCE_REVIEW_THRESHOLD = 0.7;
export const MAX_CUT_PADDING_SECONDS = 10;
export const MAX_JOIN_GAP_SECONDS = 10;
export const MIN_CUT_SECONDS = 0.1;
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
  agreement?: Rally["agreement"];
};

export type IgnoredSourceInterval = IgnoredInterval & {
  id: string;
};

export type CutDraft = {
  version: typeof CUT_DRAFT_VERSION;
  analysisId: string;
  recordingId: string;
  sourceRevision: string;
  analysisStart: number;
  analysisEnd: number;
  updatedAt: string;
  beforePaddingSeconds: number;
  afterPaddingSeconds: number;
  joinGapSeconds: number;
  pendingManualStart: number | null;
  pendingIgnoreStart: number | null;
  ignoreReason: string;
  cutPreviewEnabled: boolean;
  playbackRate: (typeof PLAYBACK_RATES)[number];
  confidenceReviewThreshold: number;
  reviewedCutIds: string[];
  selectedSuppressionPolicy: SuppressionPolicyId;
  suppressionDecisionOverrides: Record<string, "keep" | "suppress">;
  suppressionScopeOverrides: Record<string, SuppressionScope>;
  userTouchedCutIds: string[];
  suppressionContractVersion: number;
  scoreTracking: ScoreTracking;
  cuts: EditableCut[];
  ignoredIntervals: IgnoredSourceInterval[];
};

export type FinalCutInterval = {
  start: number;
  end: number;
  cutIds: string[];
  joinedGaps?: Array<{ start: number; end: number }>;
};

export type CutSplitResult = {
  draft: CutDraft;
  newCut: EditableCut;
};

export type FinalCutProvenanceSegment = {
  start: number;
  end: number;
  kind:
    | "inferred-core"
    | "padding"
    | "joined-gap"
    | "manual"
    | "suppression-whole-rally"
    | "suppression-veto-region";
  cutIds: string[];
  suggestionIds?: string[];
};

export const SUPPRESSION_SCOPE_IDS = ["whole-rally", "veto-region"] as const;
export type SuppressionScope = (typeof SUPPRESSION_SCOPE_IDS)[number];
export const DEFAULT_SUPPRESSION_SCOPE: SuppressionScope = "whole-rally";
export const SUPPRESSION_SCOPE_LABELS: Record<SuppressionScope, string> = {
  "whole-rally": "Whole rally",
  "veto-region": "Veto region",
};

export type FinalCutMaterialization = {
  intervals: FinalCutInterval[];
  provenance: FinalCutProvenanceSegment[];
};

export type CutDraftSeed = {
  analysisId: string;
  recordingId: string;
  duration: number;
  analysisStart?: number;
  analysisEnd?: number;
  rallies: Rally[];
  ignoredIntervals: IgnoredInterval[];
  suppressionContractVersion?: number;
};

function seedAnalysisBounds(seed: CutDraftSeed): { start: number; end: number } {
  const duration = Math.max(0, seed.duration);
  const start = clamp(seed.analysisStart ?? 0, 0, duration);
  const end = clamp(seed.analysisEnd ?? duration, start, duration);
  return { start, end };
}

function finiteTime(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function nextCutId(prefix: string, cuts: readonly EditableCut[]): string {
  const used = new Set(cuts.map((cut) => cut.id));
  for (let index = 1; index < 10_000; index += 1) {
    const candidate = `${prefix}${String(index).padStart(3, "0")}`;
    if (!used.has(candidate)) return candidate;
  }
  return `${prefix}${Date.now()}`;
}

function withTouchedCachedCuts(draft: CutDraft, ids: readonly string[]): CutDraft {
  const cachedIds = new Set(
    draft.cuts
      .filter((cut) => cut.origin === "cached-label")
      .map((cut) => cut.id),
  );
  const touched = ids.filter((id) => cachedIds.has(id));
  return touched.length === 0
    ? draft
    : {
        ...draft,
        userTouchedCutIds: [...new Set([...draft.userTouchedCutIds, ...touched])],
      };
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
  const bounds = seedAnalysisBounds(seed);
  const source = JSON.stringify({
    duration: Number(seed.duration.toFixed(3)),
    ...(bounds.start > 0 || bounds.end < seed.duration
      ? { analysisWindow: [Number(bounds.start.toFixed(3)), Number(bounds.end.toFixed(3))] }
      : {}),
    rallies: seed.rallies.map((rally) => [
      rally.id,
      Number(rally.start.toFixed(3)),
      Number(rally.end.toFixed(3)),
      rally.agreement ?? null,
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
  return [CUT_DRAFT_VERSION, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1].map(
    (version) => `volleycut:cut-draft:v${version}:${encodeURIComponent(analysisId)}`,
  );
}

export function createCutDraft(seed: CutDraftSeed): CutDraft {
  const duration = Math.max(0, seed.duration);
  const bounds = seedAnalysisBounds(seed);
  return {
    version: CUT_DRAFT_VERSION,
    analysisId: seed.analysisId,
    recordingId: seed.recordingId,
    sourceRevision: cutSourceRevision(seed),
    analysisStart: bounds.start,
    analysisEnd: bounds.end,
    updatedAt: new Date(0).toISOString(),
    beforePaddingSeconds: DEFAULT_CUT_PADDING.before,
    afterPaddingSeconds: DEFAULT_CUT_PADDING.after,
    joinGapSeconds: DEFAULT_JOIN_GAP_SECONDS,
    pendingManualStart: null,
    pendingIgnoreStart: null,
    ignoreReason: "non-game-content",
    cutPreviewEnabled: false,
    playbackRate: 1,
    confidenceReviewThreshold: DEFAULT_CONFIDENCE_REVIEW_THRESHOLD,
    reviewedCutIds: [],
    selectedSuppressionPolicy: "none",
    suppressionDecisionOverrides: {},
    suppressionScopeOverrides: {},
    userTouchedCutIds: [],
    suppressionContractVersion:
      seed.suppressionContractVersion ?? SUPPRESSION_POLICY_CONTRACT_VERSION,
    scoreTracking: createScoreTracking(),
    cuts: seed.rallies.map((rally) => ({
      id: rally.id,
      coreStart: clamp(rally.start, bounds.start, bounds.end),
      coreEnd: clamp(rally.end, bounds.start, bounds.end),
      keepStart: clamp(
        rally.start - DEFAULT_CUT_PADDING.before,
        bounds.start,
        bounds.end,
      ),
      keepEnd: clamp(
        rally.end + DEFAULT_CUT_PADDING.after,
        bounds.start,
        bounds.end,
      ),
      confidence: clamp(rally.confidence, 0, 1),
      included: rally.included,
      origin: "cached-label",
      agreement: rally.agreement,
    })),
    ignoredIntervals: seed.ignoredIntervals.map((interval, index) => ({
      id: `I${String(index + 1).padStart(3, "0")}`,
      start: clamp(interval.start, 0, duration),
      end: clamp(interval.end, 0, duration),
      reason: interval.reason,
    })),
  };
}

function validCut(
  value: unknown,
  minimum: number,
  maximum: number,
): value is EditableCut {
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
    (cut.agreement === undefined ||
      cut.agreement === "both-models" ||
      cut.agreement === "all-labels-v2-only" ||
      cut.agreement === "previous-production-only") &&
    cut.keepStart >= minimum &&
    cut.keepStart <= cut.coreStart &&
    cut.coreStart < cut.coreEnd &&
    cut.coreEnd <= cut.keepEnd &&
    cut.keepEnd <= maximum &&
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

function inferLegacyTouchedCutIds(
  cuts: readonly EditableCut[],
  seed: CutDraftSeed,
  beforePaddingSeconds: number,
  afterPaddingSeconds: number,
): string[] {
  const bounds = seedAnalysisBounds(seed);
  const rallies = new Map(seed.rallies.map((rally) => [rally.id, rally]));
  return cuts
    .filter((cut) => cut.origin === "cached-label")
    .filter((cut) => {
      const rally = rallies.get(cut.id);
      if (!rally) return true;
      const expectedStart = roundTime(clamp(
        rally.start - beforePaddingSeconds,
        bounds.start,
        bounds.end,
      ));
      const expectedEnd = roundTime(clamp(
        rally.end + afterPaddingSeconds,
        bounds.start,
        bounds.end,
      ));
      return (
        cut.included !== rally.included ||
        Math.abs(cut.coreStart - rally.start) > 0.000_5 ||
        Math.abs(cut.coreEnd - rally.end) > 0.000_5 ||
        Math.abs(cut.keepStart - expectedStart) > 0.000_5 ||
        Math.abs(cut.keepEnd - expectedEnd) > 0.000_5
      );
    })
    .map((cut) => cut.id);
}

export function parseCutDraft(raw: string, seed: CutDraftSeed): CutDraft | null {
  try {
    const bounds = seedAnalysisBounds(seed);
    const persisted = JSON.parse(raw) as Partial<Omit<CutDraft, "version">> & {
      version?: unknown;
      paddingSeconds?: unknown;
    };
    const persistedVersion = persisted.version;
    if (
      typeof persistedVersion !== "number" ||
      ![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, CUT_DRAFT_VERSION].includes(
        persistedVersion,
      )
    ) {
      return null;
    }
    const value: Partial<CutDraft> = {
      ...persisted,
      version: CUT_DRAFT_VERSION,
      analysisStart: persistedVersion >= 9 ? persisted.analysisStart : bounds.start,
      analysisEnd: persistedVersion >= 9 ? persisted.analysisEnd : bounds.end,
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
      joinGapSeconds: persistedVersion >= 7
        ? persisted.joinGapSeconds
        : DEFAULT_JOIN_GAP_SECONDS,
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
      reviewedCutIds: persistedVersion >= 8 ? persisted.reviewedCutIds : [],
      selectedSuppressionPolicy: persistedVersion >= 9
        ? persisted.selectedSuppressionPolicy
        : "none",
      suppressionDecisionOverrides: persistedVersion >= 9
        ? persisted.suppressionDecisionOverrides
        : {},
      suppressionScopeOverrides: persistedVersion >= 10
        ? persisted.suppressionScopeOverrides
        : {},
      userTouchedCutIds: persistedVersion >= 9
        ? persisted.userTouchedCutIds
        : Array.isArray(persisted.cuts) &&
            finiteTime(
              persistedVersion === 1
                ? 3
                : persistedVersion === 2
                  ? persisted.paddingSeconds
                  : persisted.beforePaddingSeconds,
            ) &&
            finiteTime(
              persistedVersion === 1
                ? 2
                : persistedVersion === 2
                  ? persisted.paddingSeconds
                  : persisted.afterPaddingSeconds,
            )
          ? inferLegacyTouchedCutIds(
              persisted.cuts as EditableCut[],
              seed,
              persistedVersion === 1
                ? 3
                : persistedVersion === 2
                  ? persisted.paddingSeconds as number
                  : persisted.beforePaddingSeconds as number,
              persistedVersion === 1
                ? 2
                : persistedVersion === 2
                  ? persisted.paddingSeconds as number
                  : persisted.afterPaddingSeconds as number,
            )
          : [],
      suppressionContractVersion: persistedVersion >= 9
        ? persisted.suppressionContractVersion
        : seed.suppressionContractVersion ?? SUPPRESSION_POLICY_CONTRACT_VERSION,
      scoreTracking: persistedVersion >= 11
        ? migrateScoreTracking(persisted.scoreTracking, seed.duration) ?? undefined
        : createScoreTracking(),
    };
    if (
      value.version !== CUT_DRAFT_VERSION ||
      value.analysisId !== seed.analysisId ||
      value.recordingId !== seed.recordingId ||
      value.sourceRevision !== cutSourceRevision(seed) ||
      value.analysisStart !== bounds.start ||
      value.analysisEnd !== bounds.end ||
      typeof value.updatedAt !== "string" ||
      !finiteTime(value.beforePaddingSeconds) ||
      value.beforePaddingSeconds < 0 ||
      value.beforePaddingSeconds > MAX_CUT_PADDING_SECONDS ||
      !finiteTime(value.afterPaddingSeconds) ||
      value.afterPaddingSeconds < 0 ||
      value.afterPaddingSeconds > MAX_CUT_PADDING_SECONDS ||
      !finiteTime(value.joinGapSeconds) ||
      value.joinGapSeconds < 0 ||
      value.joinGapSeconds > MAX_JOIN_GAP_SECONDS ||
      !(
        value.pendingManualStart === null ||
        (finiteTime(value.pendingManualStart) &&
          value.pendingManualStart >= bounds.start &&
          value.pendingManualStart <= bounds.end)
      ) ||
      !(
        value.pendingIgnoreStart === null ||
        (finiteTime(value.pendingIgnoreStart) &&
          value.pendingIgnoreStart >= bounds.start &&
          value.pendingIgnoreStart <= bounds.end)
      ) ||
      typeof value.ignoreReason !== "string" ||
      value.ignoreReason.length === 0 ||
      typeof value.cutPreviewEnabled !== "boolean" ||
      !PLAYBACK_RATES.includes(value.playbackRate as (typeof PLAYBACK_RATES)[number]) ||
      !finiteTime(value.confidenceReviewThreshold) ||
      value.confidenceReviewThreshold < 0 ||
      value.confidenceReviewThreshold > 1 ||
      !Array.isArray(value.reviewedCutIds) ||
      !value.reviewedCutIds.every((id) => typeof id === "string" && id.length > 0) ||
      new Set(value.reviewedCutIds).size !== value.reviewedCutIds.length ||
      !SUPPRESSION_POLICY_IDS.includes(
        value.selectedSuppressionPolicy as SuppressionPolicyId,
      ) ||
      !value.suppressionDecisionOverrides ||
      typeof value.suppressionDecisionOverrides !== "object" ||
      Array.isArray(value.suppressionDecisionOverrides) ||
      !Object.entries(value.suppressionDecisionOverrides).every(
        ([id, decision]) =>
          id.length > 0 && (decision === "keep" || decision === "suppress"),
      ) ||
      !value.suppressionScopeOverrides ||
      typeof value.suppressionScopeOverrides !== "object" ||
      Array.isArray(value.suppressionScopeOverrides) ||
      !Object.entries(value.suppressionScopeOverrides).every(
        ([id, scope]) =>
          id.length > 0 && SUPPRESSION_SCOPE_IDS.includes(scope as SuppressionScope),
      ) ||
      !Array.isArray(value.userTouchedCutIds) ||
      !value.userTouchedCutIds.every((id) => typeof id === "string" && id.length > 0) ||
      new Set(value.userTouchedCutIds).size !== value.userTouchedCutIds.length ||
      !Number.isInteger(value.suppressionContractVersion) ||
      value.suppressionContractVersion! < 1 ||
      !isValidScoreTracking(value.scoreTracking, seed.duration) ||
      (value.pendingManualStart !== null && value.pendingIgnoreStart !== null) ||
      !Array.isArray(value.cuts) ||
      !Array.isArray(value.ignoredIntervals) ||
      !value.cuts.every((cut) => validCut(cut, bounds.start, bounds.end)) ||
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
    const cachedCutIds = new Set(
      value.cuts.filter((cut) => cut.origin === "cached-label").map((cut) => cut.id),
    );
    if (
      value.reviewedCutIds.some((id) => !cachedCutIds.has(id)) ||
      value.userTouchedCutIds.some((id) => !cachedCutIds.has(id))
    ) return null;
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
  analysisStart = 0,
  analysisEnd = duration,
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
            keepStart: roundTime(
              clamp(cut.coreStart - before, analysisStart, analysisEnd),
            ),
            keepEnd: roundTime(
              clamp(cut.coreEnd + after, analysisStart, analysisEnd),
            ),
          }
        : cut,
    ),
  };
}

export function setCutCoreStart(
  draft: CutDraft,
  cutId: string,
  value: number,
  minimum = draft.analysisStart,
): CutDraft {
  if (!Number.isFinite(value)) return draft;
  const cuts = draft.cuts.map((cut) => {
    if (cut.id !== cutId) return cut;
    const maximumStart = cut.coreEnd - MIN_CUT_SECONDS;
    if (maximumStart < minimum) return cut;
    const start = roundTime(clamp(value, minimum, maximumStart));
    const beforePadding = cut.coreStart - cut.keepStart;
    return {
      ...cut,
      coreStart: start,
      keepStart: roundTime(Math.max(minimum, start - beforePadding)),
    };
  });
  return withTouchedCachedCuts({ ...draft, cuts }, [cutId]);
}

export function setCutCoreEnd(
  draft: CutDraft,
  cutId: string,
  value: number,
  maximum = draft.analysisEnd,
): CutDraft {
  if (!Number.isFinite(value)) return draft;
  const cuts = draft.cuts.map((cut) => {
    if (cut.id !== cutId) return cut;
    const minimumEnd = cut.coreStart + MIN_CUT_SECONDS;
    if (minimumEnd > maximum) return cut;
    const end = roundTime(clamp(value, minimumEnd, maximum));
    const afterPadding = cut.keepEnd - cut.coreEnd;
    return {
      ...cut,
      coreEnd: end,
      keepEnd: roundTime(Math.min(maximum, end + afterPadding)),
    };
  });
  return withTouchedCachedCuts({ ...draft, cuts }, [cutId]);
}

export function setCutCoreRange(
  draft: CutDraft,
  cutId: string,
  startValue: number,
  endValue: number,
  minimum = draft.analysisStart,
  maximum = draft.analysisEnd,
): CutDraft {
  if (
    !Number.isFinite(startValue) ||
    !Number.isFinite(endValue) ||
    maximum - minimum < MIN_CUT_SECONDS
  ) return draft;
  const cuts = draft.cuts.map((cut) => {
    if (cut.id !== cutId) return cut;
    const start = roundTime(clamp(startValue, minimum, maximum - MIN_CUT_SECONDS));
    const end = roundTime(clamp(endValue, start + MIN_CUT_SECONDS, maximum));
    const beforePadding = cut.coreStart - cut.keepStart;
    const afterPadding = cut.keepEnd - cut.coreEnd;
    return {
      ...cut,
      coreStart: start,
      coreEnd: end,
      keepStart: roundTime(Math.max(minimum, start - beforePadding)),
      keepEnd: roundTime(Math.min(maximum, end + afterPadding)),
    };
  });
  return withTouchedCachedCuts({ ...draft, cuts }, [cutId]);
}

export function splitCutAt(
  draft: CutDraft,
  cutId: string,
  value: number,
  minimum = draft.analysisStart,
  maximum = draft.analysisEnd,
): CutSplitResult | null {
  if (!Number.isFinite(value)) return null;
  const cutIndex = draft.cuts.findIndex((cut) => cut.id === cutId);
  if (cutIndex < 0) return null;
  const cut = draft.cuts[cutIndex];
  const position = roundTime(value);
  if (
    position < cut.coreStart + MIN_CUT_SECONDS ||
    position > cut.coreEnd - MIN_CUT_SECONDS
  ) return null;

  const beforePadding = cut.coreStart - cut.keepStart;
  const afterPadding = cut.keepEnd - cut.coreEnd;
  const prefix = cut.origin === "cached-label" ? "R" : "M";
  const left: EditableCut = {
    ...cut,
    coreEnd: position,
    keepEnd: roundTime(Math.min(maximum, position + afterPadding)),
  };
  const right: EditableCut = {
    ...cut,
    id: nextCutId(prefix, draft.cuts),
    coreStart: position,
    keepStart: roundTime(Math.max(minimum, position - beforePadding)),
  };
  const cuts = [...draft.cuts];
  cuts.splice(cutIndex, 1, left, right);
  const nextDraft = withTouchedCachedCuts({ ...draft, cuts }, [left.id, right.id]);
  return { draft: nextDraft, newCut: right };
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

export type SuppressionSuggestionState =
  | "dormant"
  | "suppressed"
  | "kept"
  | "edited-kept";

export function suppressionSuggestionState(
  suggestion: SuppressionSuggestion,
  draft: CutDraft,
): SuppressionSuggestionState {
  const policy = draft.selectedSuppressionPolicy;
  if (policy === "none" || !suggestion.eligiblePolicyIds.includes(policy)) {
    return "dormant";
  }
  const override = draft.suppressionDecisionOverrides[suggestion.logicalId];
  if (override === "suppress") return "suppressed";
  if (override === "keep") return "kept";
  const touched = new Set(draft.userTouchedCutIds);
  if (
    draft.cuts.some(
      (cut) =>
        cut.origin === "cached-label" &&
        touched.has(cut.id) &&
        cut.coreStart < suggestion.end &&
        suggestion.start < cut.coreEnd,
    )
  ) {
    return "edited-kept";
  }
  return "suppressed";
}

export function suppressionSuggestionScope(
  suggestion: SuppressionSuggestion,
  draft: CutDraft,
): SuppressionScope {
  return draft.suppressionScopeOverrides[suggestion.logicalId] ?? DEFAULT_SUPPRESSION_SCOPE;
}

export function activeSuppressionSuggestions(
  draft: CutDraft,
  suppression?: Pick<OnDeviceSuppression, "suggestions">,
): SuppressionSuggestion[] {
  if (!suppression || draft.selectedSuppressionPolicy === "none") return [];
  return suppression.suggestions.filter((suggestion) =>
    suggestion.eligiblePolicyIds.includes(draft.selectedSuppressionPolicy as Exclude<SuppressionPolicyId, "none">)
  );
}

function subtractRanges(
  source: { start: number; end: number },
  removals: readonly { start: number; end: number }[],
): Array<{ start: number; end: number }> {
  return mergeIgnoredIntervals(
    removals.map((interval, index) => ({
      ...interval,
      id: `removal-${index}`,
      reason: "suppression",
    })),
  ).reduce<Array<{ start: number; end: number }>>(
    (pieces, removal) => pieces.flatMap((piece) => {
      if (removal.end <= piece.start || removal.start >= piece.end) return [piece];
      return [
        ...(removal.start > piece.start
          ? [{ start: piece.start, end: Math.min(piece.end, removal.start) }]
          : []),
        ...(removal.end < piece.end
          ? [{ start: Math.max(piece.start, removal.end), end: piece.end }]
          : []),
      ];
    }),
    [{ ...source }],
  );
}

export function materializeFinalCutIntervals(
  draft: CutDraft,
  suppression?: Pick<OnDeviceSuppression, "suggestions">,
): FinalCutMaterialization {
  const appliedSuggestions = activeSuppressionSuggestions(draft, suppression)
    .filter((suggestion) => suppressionSuggestionState(suggestion, draft) === "suppressed");
  const wholeRallySuggestions = appliedSuggestions.filter(
    (suggestion) => suppressionSuggestionScope(suggestion, draft) === "whole-rally",
  );
  const vetoRegionSuggestions = appliedSuggestions.filter(
    (suggestion) => suppressionSuggestionScope(suggestion, draft) === "veto-region",
  );
  const wholeRallyCutIds = new Set(
    draft.cuts
      .filter((cut) => cut.origin === "cached-label")
      .filter((cut) => wholeRallySuggestions.some(
        (suggestion) => cut.coreStart < suggestion.end && suggestion.start < cut.coreEnd,
      ))
      .map((cut) => cut.id),
  );
  const suppressionBarriers = [
    ...vetoRegionSuggestions.map((suggestion) => ({
      start: suggestion.start,
      end: suggestion.end,
    })),
    ...draft.cuts
      .filter((cut) => wholeRallyCutIds.has(cut.id))
      .map((cut) => ({ start: cut.keepStart, end: cut.keepEnd })),
  ];
  const provenance: FinalCutProvenanceSegment[] = [];
  const sourceIntervals: FinalCutInterval[] = [];
  for (const cut of draft.cuts) {
    if (!cut.included || cut.keepEnd <= cut.keepStart) continue;
    if (cut.origin === "manual") {
      sourceIntervals.push({ start: cut.keepStart, end: cut.keepEnd, cutIds: [cut.id] });
      provenance.push({
        start: cut.keepStart,
        end: cut.keepEnd,
        kind: "manual",
        cutIds: [cut.id],
      });
      continue;
    }
    if (wholeRallyCutIds.has(cut.id)) continue;
    const fragments = subtractRanges(
      { start: cut.coreStart, end: cut.coreEnd },
      vetoRegionSuggestions,
    );
    for (const fragment of fragments) {
      const outerStart = Math.abs(fragment.start - cut.coreStart) < 0.000_5;
      const outerEnd = Math.abs(fragment.end - cut.coreEnd) < 0.000_5;
      const paddedStart = roundTime(clamp(
        outerStart ? cut.keepStart : fragment.start - draft.beforePaddingSeconds,
        draft.analysisStart,
        draft.analysisEnd,
      ));
      const paddedEnd = roundTime(clamp(
        outerEnd ? cut.keepEnd : fragment.end + draft.afterPaddingSeconds,
        draft.analysisStart,
        draft.analysisEnd,
      ));
      if (paddedEnd <= paddedStart) continue;
      const hardClipped = subtractRanges(
        { start: paddedStart, end: paddedEnd },
        vetoRegionSuggestions,
      );
      hardClipped.forEach((interval) => {
        if (interval.end > interval.start) {
          sourceIntervals.push({
            start: interval.start,
            end: interval.end,
            cutIds: [cut.id],
          });
        }
      });
      provenance.push({
        start: fragment.start,
        end: fragment.end,
        kind: "inferred-core",
        cutIds: [cut.id],
      });
      if (paddedStart < fragment.start) {
        provenance.push({
          start: paddedStart,
          end: fragment.start,
          kind: "padding",
          cutIds: [cut.id],
        });
      }
      if (fragment.end < paddedEnd) {
        provenance.push({
          start: fragment.end,
          end: paddedEnd,
          kind: "padding",
          cutIds: [cut.id],
        });
      }
    }
  }

  appliedSuggestions.forEach((suggestion) => {
    if (suppressionSuggestionScope(suggestion, draft) === "whole-rally") {
      draft.cuts
        .filter((cut) => wholeRallyCutIds.has(cut.id))
        .filter((cut) => cut.coreStart < suggestion.end && suggestion.start < cut.coreEnd)
        .forEach((cut) => {
          provenance.push({
            start: cut.keepStart,
            end: cut.keepEnd,
            kind: "suppression-whole-rally",
            cutIds: [cut.id],
            suggestionIds: [suggestion.logicalId],
          });
        });
    } else {
      provenance.push({
        start: suggestion.start,
        end: suggestion.end,
        kind: "suppression-veto-region",
        cutIds: [],
        suggestionIds: [suggestion.logicalId],
      });
    }
  });

  const joinGap = clamp(draft.joinGapSeconds, 0, MAX_JOIN_GAP_SECONDS);
  const merged = sourceIntervals
    .sort((left, right) => left.start - right.start || left.end - right.end)
    .reduce<FinalCutInterval[]>((intervals, interval) => {
      const previous = intervals.at(-1);
      const gap = previous ? interval.start - previous.end : Number.POSITIVE_INFINITY;
      const crossesSuppression = previous !== undefined && gap > 0 && suppressionBarriers.some(
        (barrier) => barrier.start < interval.start && previous.end < barrier.end,
      );
      if (!previous || (gap > 0 && (gap >= joinGap || crossesSuppression))) {
        intervals.push({ ...interval, cutIds: [...interval.cutIds] });
      } else {
        if (gap > 0) {
          const joinedGap = { start: previous.end, end: interval.start };
          previous.joinedGaps = [...(previous.joinedGaps ?? []), joinedGap];
          provenance.push({ ...joinedGap, kind: "joined-gap", cutIds: [] });
        }
        previous.end = Math.max(previous.end, interval.end);
        previous.cutIds = [...new Set([...previous.cutIds, ...interval.cutIds])];
      }
      return intervals;
    }, []);

  const ignored = mergeIgnoredIntervals(draft.ignoredIntervals);
  const remaining = ignored.reduce<FinalCutInterval[]>(
    (intervals, excluded) => intervals.flatMap((interval) => {
      if (excluded.end <= interval.start || excluded.start >= interval.end) return [interval];
      return [
        ...(excluded.start > interval.start
          ? [{ ...interval, end: Math.min(interval.end, excluded.start), cutIds: [...interval.cutIds] }]
          : []),
        ...(excluded.end < interval.end
          ? [{ ...interval, start: Math.max(interval.start, excluded.end), cutIds: [...interval.cutIds] }]
          : []),
      ];
    }),
    merged,
  );
  const intervals = remaining.map((interval) => {
    const joinedGaps = interval.joinedGaps
      ?.map((gap) => ({
        start: Math.max(interval.start, gap.start),
        end: Math.min(interval.end, gap.end),
      }))
      .filter((gap) => gap.end > gap.start);
    return {
      start: interval.start,
      end: interval.end,
      cutIds: interval.cutIds.filter((id) => provenance.some(
        (segment) =>
          segment.cutIds.includes(id) &&
          segment.start < interval.end &&
          interval.start < segment.end,
      )),
      ...(joinedGaps?.length ? { joinedGaps } : {}),
    };
  });
  const visibleProvenance = provenance.flatMap((segment) => {
    if (segment.kind.startsWith("suppression-")) return [segment];
    return intervals.flatMap((interval) => {
      const start = Math.max(segment.start, interval.start);
      const end = Math.min(segment.end, interval.end);
      return end > start ? [{ ...segment, start, end }] : [];
    });
  });
  return { intervals, provenance: visibleProvenance };
}

export function effectiveKeptCutIds(
  draft: CutDraft,
  suppression?: Pick<OnDeviceSuppression, "suggestions">,
): string[] {
  return [...new Set(
    materializeFinalCutIntervals(draft, suppression).intervals.flatMap(
      (interval) => interval.cutIds,
    ),
  )];
}

export function buildFinalCutIntervals(
  draft: CutDraft,
  suppression?: Pick<OnDeviceSuppression, "suggestions">,
): FinalCutInterval[] {
  return materializeFinalCutIntervals(draft, suppression).intervals;
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
