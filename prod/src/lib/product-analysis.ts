import type { Rally } from "./edit-list";
import type { AnalysisWindow } from "./on-device/analysis-window";
import type { OnDeviceRuntimeVariant } from "./on-device/runtime-variants";
import type {
  BaseFeatureSequence,
  NormalizedRoi,
  OnDeviceAnalysis,
  OnDeviceMediaInfo,
} from "./on-device/types";
import type { ProjectSource } from "./project-store";

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
  source: ProjectSource;
  mediaInfo: OnDeviceMediaInfo;
  roi: NormalizedRoi;
  runtimeVariant: OnDeviceRuntimeVariant;
  videoUrl: string | null;
  rallies: Rally[];
  ignoredIntervals: IgnoredInterval[];
  features: BaseFeatureSequence | null;
  inferenceTimes: Float64Array;
  probabilities: {
    rally: Float32Array;
    serve: Float32Array;
    deadState: Float32Array;
  };
  productionComponents?: OnDeviceAnalysis["productionComponents"];
  productionServeOutputs?: OnDeviceAnalysis["productionServeOutputs"];
  servingSide?: OnDeviceAnalysis["servingSide"];
  suppression?: OnDeviceAnalysis["suppression"];
};
