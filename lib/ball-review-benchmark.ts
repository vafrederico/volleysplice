export const benchmarkConfigurationIds = [
  "sol-low",
  "sol-medium",
  "sol-high",
  "sol-xhigh",
  "sol-max",
  "terra-xhigh",
  "terra-max",
  "luna-xhigh",
  "luna-max",
] as const;

export type BenchmarkConfigurationId =
  (typeof benchmarkConfigurationIds)[number];

export type BenchmarkPrimaryBallState =
  | "localizable"
  | "fully_occluded"
  | "out_of_frame"
  | "indeterminate";

export type BenchmarkBallObject = {
  id: string;
  category: "volleyball";
  role: "primary-court" | "other-court" | "unknown";
  bbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  visibility: "clear" | "motion-blurred" | "partially-occluded";
  truncated: boolean;
};

export type BenchmarkFrameAnnotation = {
  status: "reviewed";
  primaryBallState: BenchmarkPrimaryBallState;
  objects: BenchmarkBallObject[];
  notes: string;
};

export type BenchmarkSimilarityMetrics = {
  frameCount: number;
  stateAccuracy: number;
  stateMacroF1: number;
  primaryPresenceF1: number;
  boxF1Iou25: number;
  boxF1Iou50: number;
  matchedPrimaryCount: number;
  matchedPrimaryMeanIou: number | null;
  matchedPrimaryMedianCenterErrorPixels: number | null;
  matchedPrimaryVisibilityAccuracyIou25: number | null;
  objectCountAccuracy: number;
};

export type BenchmarkUsage = {
  cache_write_input_tokens: number;
  cached_input_tokens: number;
  input_tokens: number;
  output_tokens: number;
  reasoning_output_tokens: number;
};

export type BenchmarkConfiguration = {
  model: "gpt-5.6-sol" | "gpt-5.6-terra" | "gpt-5.6-luna";
  effort: "low" | "medium" | "high" | "xhigh" | "max";
  elapsedSeconds: number;
  speedupVsFreshSolXhigh: number;
  usage: BenchmarkUsage;
  similarityToPriorSolXhigh: BenchmarkSimilarityMetrics;
};

export type BallReviewBenchmarkBundle = {
  schemaVersion: 1;
  benchmarkId: "ball-review-effort-screen12-v1";
  width: number;
  height: number;
  policy: string;
  reportSha256: string;
  sample: {
    frameCount: number;
    recordingCount: number;
    environments: string[];
    reference: string;
    sealedReferenceSha256: string;
  };
  frames: Array<{
    attachmentIndex: number;
    frameId: string;
    imageUrl: string;
    reference: BenchmarkFrameAnnotation;
    outputs: Record<BenchmarkConfigurationId, BenchmarkFrameAnnotation>;
  }>;
  configurations: Record<BenchmarkConfigurationId, BenchmarkConfiguration>;
  pairwise: Record<
    BenchmarkConfigurationId,
    Record<BenchmarkConfigurationId, BenchmarkSimilarityMetrics>
  >;
  metricSemantics: {
    similarityToPriorSolXhigh: string;
    pairwise: string;
  };
  limitations: string[];
};
