export type Rally = {
  id: string;
  start: number;
  end: number;
  confidence: number;
  included: boolean;
};

export type KeptInterval = Rally & {
  keptStart: number;
  keptEnd: number;
  rallyIds: string[];
};

export function buildEditList(
  rallies: Rally[],
  preRoll: number,
  postRoll: number,
  duration: number,
): KeptInterval[] {
  const padded = rallies
    .filter((rally) => rally.included)
    .filter((rally) => [rally.start, rally.end].every(Number.isFinite) && rally.end > rally.start)
    .map((rally) => ({
      ...rally,
      keptStart: Math.max(0, Math.min(duration, rally.start - Math.max(0, preRoll))),
      keptEnd: Math.max(0, Math.min(duration, rally.end + Math.max(0, postRoll))),
      rallyIds: [rally.id],
    }))
    .filter((interval) => interval.keptEnd > interval.keptStart)
    .sort((a, b) => a.keptStart - b.keptStart || a.keptEnd - b.keptEnd);

  return padded.reduce<KeptInterval[]>((merged, interval) => {
    const previous = merged.at(-1);
    if (!previous || interval.keptStart > previous.keptEnd) {
      merged.push(interval);
      return merged;
    }
    previous.keptEnd = Math.max(previous.keptEnd, interval.keptEnd);
    previous.rallyIds.push(...interval.rallyIds);
    return merged;
  }, []);
}

export function formatTime(seconds: number): string {
  const safeSeconds = Number.isFinite(seconds) ? Math.max(0, Math.floor(seconds)) : 0;
  const hours = Math.floor(safeSeconds / 3600);
  const minutes = Math.floor((safeSeconds % 3600) / 60);
  const remainder = safeSeconds % 60;
  return hours
    ? `${hours}:${minutes.toString().padStart(2, "0")}:${remainder.toString().padStart(2, "0")}`
    : `${minutes}:${remainder.toString().padStart(2, "0")}`;
}

export function timelinePercent(seconds: number, duration: number): number {
  if (!Number.isFinite(seconds) || !Number.isFinite(duration) || duration <= 0) return 0;
  return Math.max(0, Math.min(100, (seconds / duration) * 100));
}

export function timelineTicks(duration: number, count = 6): number[] {
  if (!Number.isFinite(duration) || duration <= 0 || count < 2) return [0];
  return Array.from({ length: count }, (_, index) => (duration * index) / (count - 1));
}
