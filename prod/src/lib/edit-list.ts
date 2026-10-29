export type Rally = {
  id: string;
  start: number;
  end: number;
  confidence: number;
  included: boolean;
  agreement?:
    | "both-models"
    | "all-labels-v2-only"
    | "previous-production-only"
    | "neural";
};

export function timelinePercent(seconds: number, duration: number): number {
  if (!Number.isFinite(seconds) || !Number.isFinite(duration) || duration <= 0) return 0;
  return Math.max(0, Math.min(100, (seconds / duration) * 100));
}
