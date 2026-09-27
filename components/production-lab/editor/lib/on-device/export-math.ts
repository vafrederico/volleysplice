export type ExportInterval = { start: number; end: number };

export type ClippedSampleTiming = {
  timestamp: number;
  duration: number;
};

export function normalizeExportIntervals(
  intervals: readonly ExportInterval[],
  duration: number,
): ExportInterval[] {
  const ordered = intervals
    .map(({ start, end }) => ({
      start: Math.max(0, Math.min(duration, start)),
      end: Math.max(0, Math.min(duration, end)),
    }))
    .filter(({ start, end }) => Number.isFinite(start) && Number.isFinite(end) && end > start)
    .sort((left, right) => left.start - right.start);
  const merged: ExportInterval[] = [];
  for (const interval of ordered) {
    const previous = merged[merged.length - 1];
    if (previous && interval.start <= previous.end) {
      previous.end = Math.max(previous.end, interval.end);
    } else {
      merged.push({ ...interval });
    }
  }
  return merged;
}

export function clipSampleToInterval(
  sampleTimestamp: number,
  sampleDuration: number,
  interval: ExportInterval,
): ClippedSampleTiming | null {
  const visibleStart = Math.max(interval.start, sampleTimestamp);
  const visibleEnd = Math.min(interval.end, sampleTimestamp + sampleDuration);
  if (visibleEnd <= visibleStart) return null;
  return {
    timestamp: visibleStart - interval.start,
    duration: visibleEnd - visibleStart,
  };
}

export function timelinesHaveMatchingDuration(
  actualDuration: number,
  expectedDuration: number,
  toleranceSeconds = 0.75,
): boolean {
  return (
    Number.isFinite(actualDuration) &&
    Number.isFinite(expectedDuration) &&
    Math.abs(actualDuration - expectedDuration) <= toleranceSeconds
  );
}
