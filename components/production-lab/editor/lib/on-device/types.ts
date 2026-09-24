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

export type AudioExtractionPerformance = {
  phase: "decoding" | "features" | "alignment" | "complete";
  elapsedMs: number;
  decodeElapsedMs: number;
  decodedAudioSeconds: number;
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
  audioPerformance?: AudioExtractionPerformance;
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
  agreement?: "both-models" | "all-labels-v2-only" | "previous-production-only";
};

export type OnDeviceServeDetection = {
  time: number;
  confidence: number;
};

export type OnDeviceServeOutput = {
  probabilities: Float32Array;
  detections: OnDeviceServeDetection[];
};

export type ProductionServeOutputs = {
  allLabelsV2: OnDeviceServeOutput;
  previousProduction: OnDeviceServeOutput;
};

export type ProductionStateOutput = {
  rallyProbabilities: Float32Array;
  deadStateProbabilities: Float32Array;
};

export type ProductionStateOutputs = {
  allLabelsV2: ProductionStateOutput;
  previousProduction: ProductionStateOutput;
};

export type ServingSideSide = "near" | "far";
export type ServingSideVerdict = ServingSideSide | "review" | "not-serve";
export type ServingSideDecisionSource =
  | "serve-head"
  | "production-rally-recovery"
  | "none";
export type ServingSideReviewReason =
  | "side-score"
  | "production-rally-recovery";

export type ServingSideHeadEvidence = {
  modelId: string;
  threshold: number;
  peakProbability: number;
  peakTime: number;
  crossesThreshold: boolean;
  nearestDetection: OnDeviceServeDetection | null;
};

export type ServingSideCandidateVerdict = {
  /** Stable ID of the merged production interval used as the candidate. */
  id: string;
  /** Explicit production assumption: the merged interval start is the serve anchor. */
  anchor: number;
  interval: Pick<OnDeviceInterval, "start" | "end" | "agreement">;
  nearProbability: number;
  side: ServingSideSide;
  verdict: ServingSideVerdict;
  serveDecisionSource: ServingSideDecisionSource;
  reviewReasons: ServingSideReviewReason[];
  serveEvidence: {
    allLabelsV2: ServingSideHeadEvidence;
    previousProduction: ServingSideHeadEvidence;
  };
};

export type OnDeviceServingSideOutput = {
  modelId: string;
  modelFingerprint: string;
  featureVersion: "SERVSIDE237-FLIGHT";
  anchorContract: "merged-production-interval-start-v1";
  /** Raw, source-derived features aligned row-for-row with `candidates`. */
  features: {
    rows: number;
    columns: number;
    values: Float64Array;
  };
  candidates: ServingSideCandidateVerdict[];
};

export type SideSwitchCandidateKind =
  | "adjacent-rally-boundary"
  | "internal-dead-state-peak";

export type SideSwitchCandidateVerdict = {
  id: string;
  timestamp: number;
  probability: number;
  kind: SideSwitchCandidateKind;
  sourceRangeIds: string[];
};

export type OnDeviceSideSwitchOutput = {
  modelId: string;
  modelFingerprint: string;
  featureVersion: "SIDE-SWITCH-UNION34-V1";
  candidateContract: "range-boundaries-dead-peaks-v1";
  features: {
    rows: number;
    columns: number;
    values: Float64Array;
  };
  candidates: SideSwitchCandidateVerdict[];
};

export type OnDeviceSuppression = {
  modelId: string;
  artifactSha256: string;
  weightsSha256: string;
  decoderVersion: string;
  policyContractVersion: number;
  probabilities: Float32Array;
  decodedIntervals: OnDeviceInterval[];
  suggestions: import("./suppression-policy").SuppressionSuggestion[];
  identicalPolicyResults: boolean;
};

export type OnDeviceAnalysis = {
  modelId: string;
  featurePath: "local-source";
  intervals: OnDeviceInterval[];
  times: Float64Array;
  featureNames?: string[];
  featureValues?: Float32Array;
  rallyProbabilities: Float32Array;
  serveProbabilities: Float32Array;
  deadStateProbabilities: Float32Array;
  productionComponents?: {
    allLabelsV2: OnDeviceInterval[];
    previousProduction: OnDeviceInterval[];
  };
  productionServeOutputs?: ProductionServeOutputs;
  productionStateOutputs?: ProductionStateOutputs;
  servingSide?: OnDeviceServingSideOutput;
  sideSwitch?: OnDeviceSideSwitchOutput;
  suppression?: OnDeviceSuppression;
};
