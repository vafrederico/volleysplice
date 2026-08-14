import type { Rally } from "./edit-list";

export type IgnoredInterval = {
  start: number;
  end: number;
  reason: string;
  notes?: string;
};

export type ProductAnalysis = {
  id: string;
  recordingId: string;
  kind: "model";
  modelId: string;
  duration: number;
  width: number;
  height: number;
  sourceFilename: string;
  videoUrl: string;
  rallies: Rally[];
  ignoredIntervals: IgnoredInterval[];
};
