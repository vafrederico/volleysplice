export type NormalizedRoi = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type RoiProfile = {
  id: string;
  label: string;
  roi: NormalizedRoi;
  source: "known-recording" | "camera-default" | "full-frame";
};

export type OnDeviceMediaInfo = {
  duration: number;
  mimeType: string;
  width: number;
  height: number;
  rotation: 0 | 90 | 180 | 270;
  videoCodec: string;
  videoCodecString: string | null;
  canDecodeVideo: boolean;
  hasAudio: boolean;
  audioCodec: string | null;
  sampleRate: number | null;
  channels: number | null;
  canDecodeAudio: boolean;
};

export type AnalysisStage =
  | "opening"
  | "video"
  | "audio"
  | "normalizing"
  | "inference"
  | "complete";

export type FeatureExtractionPerformance = {
  sampledFrames: number;
  generatedFrames: number;
  generatedVideoSeconds: number;
  videoElapsedMs: number;
  openCvLoadMs: number;
  decoderCanvasMs: number;
  decoderWaitMs: number;
  decoderOverlapMs: number;
  canvasDrawMs: number;
  canvasDrawFrames: number;
  extractionMs: number;
  canvasReadbackMs: number;
  imageOperationsMs: number;
  phaseCorrelationMs: number;
  opticalFlowMs: number;
  javascriptMs: number;
  cacheIoMs: number;
};

export type AnalysisProgress = {
  stage: AnalysisStage;
  completed: number;
  total: number;
  detail: string;
  featureCache?: {
    enabled: boolean;
    complete: boolean;
    resumedRows: number;
    savedRows: number;
  };
  performance?: FeatureExtractionPerformance;
};

export type BaseFeatureSequence = {
  times: Float64Array;
  values: Float32Array;
  rows: number;
  columns: number;
  names: readonly string[];
};

export type OnDeviceInterval = {
  id: string;
  start: number;
  end: number;
  confidence: number;
  included: boolean;
};

export type OnDeviceAnalysis = {
  modelId: string;
  featurePath: "raw-virtual-proxy" | "training-proxy";
  intervals: OnDeviceInterval[];
  times: Float64Array;
  rallyProbabilities: Float32Array;
  serveProbabilities: Float32Array;
  deadStateProbabilities: Float32Array;
};
