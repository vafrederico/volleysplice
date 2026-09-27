import type { OnDeviceInterval } from "./types.ts";

export const SUPPRESSION_POLICY_CONTRACT_VERSION = 1 as const;

export const SUPPRESSION_POLICY_IDS = [
  "none",
  "conservative",
  "balanced",
  "aggressive",
] as const;

export type SuppressionPolicyId = (typeof SUPPRESSION_POLICY_IDS)[number];
export type ActiveSuppressionPolicyId = Exclude<SuppressionPolicyId, "none">;

export type ProductionComponentSource =
  | "all-labels-v2"
  | "previous-production";

export type ProductionComponentInterval = OnDeviceInterval & {
  source: ProductionComponentSource;
};

export type SuppressionSuggestion = {
  id: string;
  logicalId: string;
  suppressionEventId: string;
  start: number;
  end: number;
  score: number;
  sourceProductionIds: string[];
  eligiblePolicyIds: ActiveSuppressionPolicyId[];
};

export type SuppressionPolicyResult = {
  suggestions: SuppressionSuggestion[];
  identicalPolicyResults: boolean;
};

type MillisecondInterval = { start: number; end: number };
type TaggedMillisecondInterval = MillisecondInterval & {
  id: string;
  source: ProductionComponentSource;
};

const POLICY_CONFIG: Record<
  ActiveSuppressionPolicyId,
  { agreementPaddingMs: number; joinGapMs: number; overlapOnly: boolean }
> = {
  conservative: { agreementPaddingMs: 2_000, joinGapMs: 500, overlapOnly: false },
  balanced: { agreementPaddingMs: 1_500, joinGapMs: 500, overlapOnly: false },
  aggressive: { agreementPaddingMs: 0, joinGapMs: 0, overlapOnly: true },
};

export const SUPPRESSION_POLICY_LABELS: Record<SuppressionPolicyId, string> = {
  none: "No suppression",
  conservative: "Conservative",
  balanced: "Balanced",
  aggressive: "Aggressive",
};

export const SUPPRESSION_POLICY_DIAGNOSTIC_NAMES: Record<
  ActiveSuppressionPolicyId,
  string
> = {
  conservative: "Zero non-exempt misses",
  balanced: "Aggressive intermediate",
  aggressive: "Raw connected",
};

function toMs(seconds: number): number {
  return Math.round(seconds * 1000);
}

function fromMs(milliseconds: number): number {
  return milliseconds / 1000;
}

function hashText(value: string): string {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(36);
}

function unionMs(intervals: readonly MillisecondInterval[]): MillisecondInterval[] {
  return intervals
    .filter((interval) => interval.end > interval.start)
    .map((interval) => ({ ...interval }))
    .sort((left, right) => left.start - right.start || left.end - right.end)
    .reduce<MillisecondInterval[]>((merged, interval) => {
      const previous = merged.at(-1);
      if (!previous || interval.start > previous.end) merged.push(interval);
      else previous.end = Math.max(previous.end, interval.end);
      return merged;
    }, []);
}

function intersectMs(
  left: readonly MillisecondInterval[],
  right: readonly MillisecondInterval[],
): MillisecondInterval[] {
  const first = unionMs(left);
  const second = unionMs(right);
  const intersections: MillisecondInterval[] = [];
  let leftIndex = 0;
  let rightIndex = 0;
  while (leftIndex < first.length && rightIndex < second.length) {
    const start = Math.max(first[leftIndex].start, second[rightIndex].start);
    const end = Math.min(first[leftIndex].end, second[rightIndex].end);
    if (end > start) intersections.push({ start, end });
    if (first[leftIndex].end <= second[rightIndex].end) leftIndex += 1;
    else rightIndex += 1;
  }
  return intersections;
}

function agreementComponents(
  raw: readonly TaggedMillisecondInterval[],
  durationMs: number,
  policy: ActiveSuppressionPolicyId,
): Array<MillisecondInterval & { sources: Set<ProductionComponentSource> }> {
  const config = POLICY_CONFIG[policy];
  const expanded = raw
    .map((interval) => ({
      start: Math.max(0, interval.start - config.agreementPaddingMs),
      end: Math.min(durationMs, interval.end + config.agreementPaddingMs),
      sources: new Set([interval.source]),
    }))
    .sort((left, right) => left.start - right.start || left.end - right.end);
  return expanded.reduce<
    Array<MillisecondInterval & { sources: Set<ProductionComponentSource> }>
  >((components, interval) => {
    const previous = components.at(-1);
    const gap = previous ? interval.start - previous.end : Number.POSITIVE_INFINITY;
    const connects = previous && (config.overlapOnly
      ? gap < 0
      : gap <= 0 || (gap > 0 && gap < config.joinGapMs));
    if (!previous || !connects) {
      components.push({ ...interval, sources: new Set(interval.sources) });
      return components;
    }
    previous.end = Math.max(previous.end, interval.end);
    for (const source of interval.sources) previous.sources.add(source);
    return components;
  }, []);
}

function candidateForPolicy(
  raw: readonly TaggedMillisecondInterval[],
  event: MillisecondInterval,
  durationMs: number,
  policy: ActiveSuppressionPolicyId,
): MillisecondInterval[] {
  const oneModelComponents = agreementComponents(raw, durationMs, policy)
    .filter((component) => component.sources.size === 1);
  const rawUnion = unionMs(raw);
  const eligibleRaw = intersectMs(rawUnion, oneModelComponents);
  return intersectMs([event], eligibleRaw);
}

function intervalContains(
  intervals: readonly MillisecondInterval[],
  start: number,
  end: number,
): boolean {
  return intervals.some((interval) => interval.start <= start && interval.end >= end);
}

function normalizedPolicyIntervals(
  suggestions: readonly SuppressionSuggestion[],
  policy: ActiveSuppressionPolicyId,
): string {
  return JSON.stringify(
    unionMs(
      suggestions
        .filter((suggestion) => suggestion.eligiblePolicyIds.includes(policy))
        .map((suggestion) => ({
          start: toMs(suggestion.start),
          end: toMs(suggestion.end),
        })),
    ),
  );
}

export function suggestionsForPolicy(
  suggestions: readonly SuppressionSuggestion[],
  policy: SuppressionPolicyId,
): SuppressionSuggestion[] {
  if (policy === "none") return [];
  return suggestions.filter((suggestion) =>
    suggestion.eligiblePolicyIds.includes(policy)
  );
}

export function nextSuppressionSuggestion(
  suggestions: readonly SuppressionSuggestion[],
  selectedId: string,
  offset: -1 | 1,
): SuppressionSuggestion | undefined {
  if (suggestions.length === 0) return undefined;
  const ordered = [...suggestions].sort(
    (left, right) => left.start - right.start || left.end - right.end ||
      left.id.localeCompare(right.id),
  );
  const selectedIndex = ordered.findIndex((suggestion) => suggestion.id === selectedId);
  if (selectedIndex < 0) return offset < 0 ? ordered.at(-1) : ordered[0];
  return ordered[(selectedIndex + offset + ordered.length) % ordered.length];
}

export function nextSuppressionAfterTime(
  suggestions: readonly SuppressionSuggestion[],
  time: number,
): SuppressionSuggestion | undefined {
  if (suggestions.length === 0) return undefined;
  const ordered = [...suggestions].sort(
    (left, right) => left.start - right.start || left.end - right.end ||
      left.id.localeCompare(right.id),
  );
  return ordered.find((suggestion) => suggestion.start > time + 0.000_5) ?? ordered[0];
}

export function buildSuppressionSuggestions(
  productionComponents: {
    allLabelsV2: readonly OnDeviceInterval[];
    previousProduction: readonly OnDeviceInterval[];
  },
  decodedSuppression: readonly OnDeviceInterval[],
  duration: number,
): SuppressionPolicyResult {
  const durationMs = Math.max(0, toMs(duration));
  const raw: TaggedMillisecondInterval[] = [
    ...productionComponents.allLabelsV2.map((interval) => ({
      id: interval.id,
      source: "all-labels-v2" as const,
      start: Math.max(0, toMs(interval.start)),
      end: Math.min(durationMs, toMs(interval.end)),
    })),
    ...productionComponents.previousProduction.map((interval) => ({
      id: interval.id,
      source: "previous-production" as const,
      start: Math.max(0, toMs(interval.start)),
      end: Math.min(durationMs, toMs(interval.end)),
    })),
  ].filter((interval) => interval.end > interval.start);

  const suggestions: SuppressionSuggestion[] = [];
  decodedSuppression.forEach((decoded, eventIndex) => {
    const event = {
      start: Math.max(0, toMs(decoded.start)),
      end: Math.min(durationMs, toMs(decoded.end)),
    };
    if (event.end <= event.start) return;
    const sourceProductionIds = raw
      .filter((interval) => interval.start < event.end && event.start < interval.end)
      .map((interval) => interval.id)
      .sort();
    if (sourceProductionIds.length === 0) return;
    const suppressionEventId =
      `S${String(eventIndex + 1).padStart(3, "0")}-${event.start}-${event.end}`;
    const logicalId = `suppression-${hashText(
      `${suppressionEventId}|${sourceProductionIds.join("|")}`,
    )}`;
    const byPolicy = Object.fromEntries(
      (["conservative", "balanced", "aggressive"] as const).map((policy) => [
        policy,
        candidateForPolicy(raw, event, durationMs, policy),
      ]),
    ) as Record<ActiveSuppressionPolicyId, MillisecondInterval[]>;
    const boundaries = [...new Set(
      Object.values(byPolicy).flatMap((intervals) =>
        intervals.flatMap((interval) => [interval.start, interval.end])
      ),
    )].sort((left, right) => left - right);
    const atoms: Array<{
      start: number;
      end: number;
      policies: ActiveSuppressionPolicyId[];
    }> = [];
    for (let index = 0; index < boundaries.length - 1; index += 1) {
      const start = boundaries[index];
      const end = boundaries[index + 1];
      if (end <= start) continue;
      const policies = (["conservative", "balanced", "aggressive"] as const)
        .filter((policy) => intervalContains(byPolicy[policy], start, end));
      if (policies.length === 0) continue;
      const previous = atoms.at(-1);
      if (
        previous &&
        previous.end === start &&
        previous.policies.join("|") === policies.join("|")
      ) {
        previous.end = end;
      } else {
        atoms.push({ start, end, policies: [...policies] });
      }
    }
    for (const atom of atoms) {
      suggestions.push({
        id: `${logicalId}-${atom.start}-${atom.end}`,
        logicalId,
        suppressionEventId,
        start: fromMs(atom.start),
        end: fromMs(atom.end),
        score: Math.max(0, Math.min(1, decoded.confidence)),
        sourceProductionIds: [...sourceProductionIds],
        eligiblePolicyIds: atom.policies,
      });
    }
  });
  suggestions.sort((left, right) => left.start - right.start || left.end - right.end);
  const normalized = (["conservative", "balanced", "aggressive"] as const)
    .map((policy) => normalizedPolicyIntervals(suggestions, policy));
  return {
    suggestions,
    identicalPolicyResults: normalized.every((value) => value === normalized[0]),
  };
}
