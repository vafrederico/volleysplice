export type NormalizedRoi = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type { AnalysisWindow } from "./analysis-window";

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

export type VideoDecodeStrategy = "sparse" | "sequential";

export type VideoDecoderAcceleration = "prefer-hardware" | "no-preference";

export type FeatureReductionKernel = "javascript" | "wasm";

export type FeatureExtractionPerformance = {
  profilingEnabled: boolean;
  decodeStrategy: VideoDecodeStrategy;
  decoderAcceleration: VideoDecoderAcceleration;
  reductionKernel: FeatureReductionKernel;
  decodedSourceFrames: number | null;
  sampledFrames: number;
  generatedFrames: number;
  generatedVideoSeconds: number;
  videoElapsedMs: number;
  openCvLoadMs: number;
  reductionKernelLoadMs: number;
  decoderCanvasMs: number;
  decoderWaitMs: number;
  decoderOverlapMs: number;
  canvasDrawMs: number;
  canvasDrawFrames: number;
  workerActive: boolean;
  workerBlockingMs: number;
  workerOverlapMs: number;
  extractionMs: number;
  canvasReadbackMs: number;
  imageOperationsMs: number;
  phaseCorrelationMs: number;
  opticalFlowMs: number;
  javascriptMs: number;
  wasmReductionMs: number;
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
  agreement?:
    | "both-models"
    | "all-labels-v2-only"
    | "previous-production-only";
};

export type OnDeviceAnalysis = {
  modelId: string;
  featurePath: "local-source";
  intervals: OnDeviceInterval[];
  times: Float64Array;
  rallyProbabilities: Float32Array;
  serveProbabilities: Float32Array;
  deadStateProbabilities: Float32Array;
};
