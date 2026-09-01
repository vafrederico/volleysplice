import { buildEditList, type JoinedGap, type Rally } from "./edit-list.ts";

export type RallyComparison = {
  matchedPredictionIds: Set<string>;
  unmatchedPredictionRallies: Rally[];
  missedHumanRallies: Rally[];
};

export type LiveTimeMetrics = {
  precision: number;
  recall: number;
  f1: number;
};

export type LiveTimeComparisonSegment = {
  id: string;
  start: number;
  end: number;
  kind: "match" | "added" | "missed";
  predictionId?: string;
  paddingOrigin?: "before" | "after" | "both";
};

export type PaddedRally = Rally & {
  rallyIds: string[];
  joinedGaps: JoinedGap[];
};

export function calculateF1(precision: number, recall: number): number {
  return precision + recall > 0
    ? (2 * precision * recall) / (precision + recall)
    : 0;
}

export function padAndMergeRallies(
  rallies: Rally[],
  beforeSeconds: number,
  afterSeconds: number,
  duration: number,
  joinGapSeconds?: number,
): PaddedRally[] {
  return buildEditList(
    rallies,
    beforeSeconds,
    afterSeconds,
    duration,
    joinGapSeconds,
  ).map(
    (interval) => ({
      id: interval.rallyIds[0],
      start: interval.keptStart,
      end: interval.keptEnd,
      confidence: interval.confidence,
      included: true,
      rallyIds: interval.rallyIds,
      joinedGaps: interval.joinedGaps,
    }),
  );
}

type TimeInterval = Pick<Rally, "start" | "end">;

function mergeIntervals(intervals: TimeInterval[]): TimeInterval[] {
  const merged: TimeInterval[] = [];
  for (const interval of [...intervals].sort(
    (left, right) => left.start - right.start || left.end - right.end,
  )) {
    if (interval.end <= interval.start) continue;
    const previous = merged.at(-1);
    if (previous && interval.start < previous.end) {
      previous.end = Math.max(previous.end, interval.end);
    } else {
      merged.push({ start: interval.start, end: interval.end });
    }
  }
  return merged;
}

function totalSeconds(intervals: TimeInterval[]): number {
  return intervals.reduce((total, interval) => total + interval.end - interval.start, 0);
}

export function totalRallySeconds(rallies: Rally[]): number {
  return totalSeconds(mergeIntervals(rallies));
}

export function calculateDurationDeltaPercent(
  duration: number,
  referenceDuration: number,
): number | null {
  if (!Number.isFinite(duration) || !Number.isFinite(referenceDuration) || referenceDuration <= 0) {
    return null;
  }
  return ((duration - referenceDuration) / referenceDuration) * 100;
}

function intersectionSeconds(
  firstIntervals: TimeInterval[],
  secondIntervals: TimeInterval[],
): number {
  let firstIndex = 0;
  let secondIndex = 0;
  let total = 0;
  while (firstIndex < firstIntervals.length && secondIndex < secondIntervals.length) {
    const first = firstIntervals[firstIndex];
    const second = secondIntervals[secondIndex];
    total += Math.max(0, Math.min(first.end, second.end) - Math.max(first.start, second.start));
    if (first.end <= second.end) firstIndex += 1;
    else secondIndex += 1;
  }
  return total;
}

export function excludeIgnoredTime(
  rallies: Rally[],
  ignoredIntervals: TimeInterval[],
): Rally[] {
  const ignored = mergeIntervals(ignoredIntervals);
  if (ignored.length === 0) return rallies;
  const kept: Rally[] = [];
  for (const rally of rallies) {
    let segments = [{ start: rally.start, end: rally.end }];
    for (const excluded of ignored) {
      const next: TimeInterval[] = [];
      for (const segment of segments) {
        if (excluded.end <= segment.start || excluded.start >= segment.end) {
          next.push(segment);
          continue;
        }
        if (excluded.start > segment.start) {
          next.push({ start: segment.start, end: Math.min(excluded.start, segment.end) });
        }
        if (excluded.end < segment.end) {
          next.push({ start: Math.max(excluded.end, segment.start), end: segment.end });
        }
      }
      segments = next;
      if (segments.length === 0) break;
    }
    segments.forEach((segment) => kept.push({ ...rally, ...segment }));
  }
  return kept;
}

export function calculateLiveTimeMetrics(
  predictions: Rally[],
  humanRallies: Rally[],
  ignoredIntervals: TimeInterval[] = [],
): LiveTimeMetrics {
  const normalizedPredictions = mergeIntervals(excludeIgnoredTime(predictions, ignoredIntervals));
  const normalizedHuman = mergeIntervals(excludeIgnoredTime(humanRallies, ignoredIntervals));
  const predictedSeconds = totalSeconds(normalizedPredictions);
  const humanSeconds = totalSeconds(normalizedHuman);
  const intersection = intersectionSeconds(normalizedPredictions, normalizedHuman);
  const precision = predictedSeconds > 0 ? intersection / predictedSeconds : 0;
  const recall = humanSeconds > 0 ? intersection / humanSeconds : 1;
  return {
    precision,
    recall,
    f1: calculateF1(precision, recall),
  };
}

export function buildLiveTimeComparisonSegments(
  predictions: Rally[],
  humanRallies: Rally[],
  ignoredIntervals: TimeInterval[] = [],
): LiveTimeComparisonSegment[] {
  const evaluatedPredictions = excludeIgnoredTime(predictions, ignoredIntervals);
  const normalizedHuman = mergeIntervals(excludeIgnoredTime(humanRallies, ignoredIntervals));
  const normalizedPredictions = mergeIntervals(evaluatedPredictions);
  const segments: LiveTimeComparisonSegment[] = [];
  let segmentIndex = 0;

  for (const prediction of evaluatedPredictions) {
    if (prediction.end <= prediction.start) continue;
    let cursor = prediction.start;
    for (const human of normalizedHuman) {
      if (human.end <= cursor) continue;
      if (human.start >= prediction.end) break;
      if (human.start > cursor) {
        segments.push({
          id: `${prediction.id}-segment-${segmentIndex += 1}`,
          start: cursor,
          end: Math.min(human.start, prediction.end),
          kind: "added",
          predictionId: prediction.id,
        });
      }
      const overlapStart = Math.max(cursor, human.start);
      const overlapEnd = Math.min(prediction.end, human.end);
      if (overlapEnd > overlapStart) {
        segments.push({
          id: `${prediction.id}-segment-${segmentIndex += 1}`,
          start: overlapStart,
          end: overlapEnd,
          kind: "match",
          predictionId: prediction.id,
        });
        cursor = overlapEnd;
      }
      if (cursor >= prediction.end) break;
    }
    if (cursor < prediction.end) {
      segments.push({
        id: `${prediction.id}-segment-${segmentIndex += 1}`,
        start: cursor,
        end: prediction.end,
        kind: "added",
        predictionId: prediction.id,
      });
    }
  }

  for (const human of normalizedHuman) {
    let cursor = human.start;
    for (const prediction of normalizedPredictions) {
      if (prediction.end <= cursor) continue;
      if (prediction.start >= human.end) break;
      if (prediction.start > cursor) {
        segments.push({
          id: `missed-segment-${segmentIndex += 1}`,
          start: cursor,
          end: Math.min(prediction.start, human.end),
          kind: "missed",
        });
      }
      cursor = Math.max(cursor, Math.min(prediction.end, human.end));
      if (cursor >= human.end) break;
    }
    if (cursor < human.end) {
      segments.push({
        id: `missed-segment-${segmentIndex += 1}`,
        start: cursor,
        end: human.end,
        kind: "missed",
      });
    }
  }

  return segments.sort(
    (left, right) => left.start - right.start || left.end - right.end || left.kind.localeCompare(right.kind),
  );
}

export function markModelPaddingOrigins(
  segments: LiveTimeComparisonSegment[],
  coreRallies: Rally[],
  beforeSeconds: number,
  afterSeconds: number,
  duration: number,
): LiveTimeComparisonSegment[] {
  const cores = mergeIntervals(coreRallies.filter((rally) => rally.included));
  const before = Math.max(0, beforeSeconds);
  const after = Math.max(0, afterSeconds);
  const beforeRanges = coreRallies
    .filter((rally) => rally.included)
    .map((rally) => ({
      start: Math.max(0, rally.start - before),
      end: Math.min(duration, rally.start),
    }))
    .filter((range) => range.end > range.start);
  const afterRanges = coreRallies
    .filter((rally) => rally.included)
    .map((rally) => ({
      start: Math.max(0, rally.end),
      end: Math.min(duration, rally.end + after),
    }))
    .filter((range) => range.end > range.start);
  const boundaries = [
    ...cores.flatMap((range) => [range.start, range.end]),
    ...beforeRanges.flatMap((range) => [range.start, range.end]),
    ...afterRanges.flatMap((range) => [range.start, range.end]),
  ];
  const contains = (ranges: TimeInterval[], time: number) =>
    ranges.some((range) => range.start <= time && time < range.end);
  let splitIndex = 0;

  const splitSegments = segments.flatMap((segment) => {
    if (segment.kind === "missed") return [segment];
    const points = [
      segment.start,
      ...boundaries.filter((boundary) => boundary > segment.start && boundary < segment.end),
      segment.end,
    ].sort((left, right) => left - right);
    return points.slice(0, -1).map((start, index) => {
      const end = points[index + 1];
      const midpoint = start + (end - start) / 2;
      let paddingOrigin: LiveTimeComparisonSegment["paddingOrigin"];
      if (!contains(cores, midpoint)) {
        const fromBefore = contains(beforeRanges, midpoint);
        const fromAfter = contains(afterRanges, midpoint);
        paddingOrigin = fromBefore && fromAfter
          ? "both"
          : fromBefore
            ? "before"
            : fromAfter
              ? "after"
              : undefined;
      }
      return {
        ...segment,
        id: `${segment.id}-origin-${splitIndex += 1}`,
        start,
        end,
        paddingOrigin,
      };
    });
  });
  return splitSegments.reduce<LiveTimeComparisonSegment[]>((merged, segment) => {
    const previous = merged.at(-1);
    if (
      previous &&
      previous.end === segment.start &&
      previous.kind === segment.kind &&
      previous.predictionId === segment.predictionId &&
      previous.paddingOrigin === segment.paddingOrigin
    ) {
      previous.end = segment.end;
    } else {
      merged.push(segment);
    }
    return merged;
  }, []);
}

export function buildPaddingSegments(
  rallies: Rally[],
  beforeSeconds: number,
  afterSeconds: number,
  duration: number,
): LiveTimeComparisonSegment[] {
  const padded = padAndMergeRallies(
    rallies,
    beforeSeconds,
    afterSeconds,
    duration,
    0,
  );
  return buildLiveTimeComparisonSegments(padded, rallies).filter(
    (segment) => segment.kind === "added",
  );
}

function intervalIou(first: Rally, second: Rally): number {
  const intersection = Math.max(
    0,
    Math.min(first.end, second.end) - Math.max(first.start, second.start),
  );
  if (intersection === 0) return 0;
  const union = first.end - first.start + second.end - second.start - intersection;
  return union > 0 ? intersection / union : 0;
}

export function compareRalliesToHumanLabels(
  predictions: Rally[],
  humanRallies: Rally[],
  minimumIou = 0.5,
): RallyComparison {
  const orderedPredictions = predictions
    .map((rally, originalIndex) => ({ rally, originalIndex }))
    .sort((left, right) => left.rally.start - right.rally.start || left.rally.end - right.rally.end);
  const orderedHuman = humanRallies
    .map((rally, originalIndex) => ({ rally, originalIndex }))
    .sort((left, right) => left.rally.start - right.rally.start || left.rally.end - right.rally.end);
  type Score = { matches: number; iou: number };
  type Decision = "prediction" | "human" | "match";
  const scores: Score[][] = Array.from(
    { length: orderedPredictions.length + 1 },
    () => Array.from({ length: orderedHuman.length + 1 }, () => ({ matches: 0, iou: 0 })),
  );
  const decisions: (Decision | null)[][] = Array.from(
    { length: orderedPredictions.length + 1 },
    () => Array<Decision | null>(orderedHuman.length + 1).fill(null),
  );
  const better = (left: Score, right: Score) =>
    left.matches > right.matches ||
    (left.matches === right.matches && left.iou > right.iou + Number.EPSILON);

  for (let predictionIndex = 1; predictionIndex <= orderedPredictions.length; predictionIndex += 1) {
    for (let humanIndex = 1; humanIndex <= orderedHuman.length; humanIndex += 1) {
      let best = scores[predictionIndex - 1][humanIndex];
      let decision: Decision = "prediction";
      const skipHuman = scores[predictionIndex][humanIndex - 1];
      if (better(skipHuman, best)) {
        best = skipHuman;
        decision = "human";
      }
      const iou = intervalIou(
        orderedPredictions[predictionIndex - 1].rally,
        orderedHuman[humanIndex - 1].rally,
      );
      if (iou >= minimumIou) {
        const previous = scores[predictionIndex - 1][humanIndex - 1];
        const match = { matches: previous.matches + 1, iou: previous.iou + iou };
        if (better(match, best)) {
          best = match;
          decision = "match";
        }
      }
      scores[predictionIndex][humanIndex] = best;
      decisions[predictionIndex][humanIndex] = decision;
    }
  }

  const matchedPredictionIndexes = new Set<number>();
  const matchedHumanIndexes = new Set<number>();
  let predictionIndex = orderedPredictions.length;
  let humanIndex = orderedHuman.length;
  while (predictionIndex > 0 && humanIndex > 0) {
    const decision = decisions[predictionIndex][humanIndex];
    if (decision === "match") {
      matchedPredictionIndexes.add(orderedPredictions[predictionIndex - 1].originalIndex);
      matchedHumanIndexes.add(orderedHuman[humanIndex - 1].originalIndex);
      predictionIndex -= 1;
      humanIndex -= 1;
    } else if (decision === "human") {
      humanIndex -= 1;
    } else {
      predictionIndex -= 1;
    }
  }

  return {
    matchedPredictionIds: new Set(
      predictions
        .filter((_, index) => matchedPredictionIndexes.has(index))
        .map((prediction) => prediction.id),
    ),
    unmatchedPredictionRallies: predictions.filter(
      (_, index) => !matchedPredictionIndexes.has(index),
    ),
    missedHumanRallies: humanRallies.filter(
      (_, index) => !matchedHumanIndexes.has(index),
    ),
  };
}
