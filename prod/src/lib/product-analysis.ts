import type { Rally } from "./edit-list";
import type { AnalysisWindow } from "./on-device/analysis-window";

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
  analysisWindow: AnalysisWindow;
  width: number;
  height: number;
  sourceFilename: string;
  videoUrl: string | null;
  rallies: Rally[];
  ignoredIntervals: IgnoredInterval[];
};
