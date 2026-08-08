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
};

export function buildEditList(
  rallies: Rally[],
  preRoll: number,
  postRoll: number,
  duration: number,
): KeptInterval[] {
  return rallies
    .filter((rally) => rally.included)
    .map((rally) => ({
      ...rally,
      keptStart: Math.max(0, rally.start - preRoll),
      keptEnd: Math.min(duration, rally.end + postRoll),
    }));
}

export function formatTime(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.floor(seconds % 60);
  return `${minutes}:${remainder.toString().padStart(2, "0")}`;
}
