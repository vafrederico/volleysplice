export const APPEARANCE_FEATURES = [
  "fullFrameControl",
  "playerPaletteEqual",
  "playerPaletteArea",
  "playerPaletteAreaPlusGeometry",
] as const;

export type AppearanceFeature = (typeof APPEARANCE_FEATURES)[number];

export type AppearanceAggregate = {
  meanDetectionCount: number;
  meanTotalBoxAreaFraction: number;
  meanMedianBoxHeightFraction: number | null;
  meanDetectionScore: number | null;
  usableFrameCount: number;
};

export type AppearanceEvent = {
  eventId: string;
  recordingId: string;
  environment: string;
  label: 0 | 1 | null;
  source: string;
  targetStatus: string;
  sourceType: string;
  sourceGroup: string;
  split: string;
  candidateSource?: Record<string, unknown>;
  transitionTime: number;
  gapStart: number;
  gapEnd: number;
  gapSeconds: number;
  beforeTimes: number[];
  afterTimes: number[];
  status: "ok" | "insufficient-window" | "frame-error" | string;
  before?: AppearanceAggregate;
  after?: AppearanceAggregate;
  features: Partial<Record<AppearanceFeature, number | null>>;
  error?: string;
};

export type AppearanceMetric = {
  usableEvents: number;
  coverage: number | null;
  positives: number;
  negatives: number;
  rocAuc: number | null;
  averagePrecision: number | null;
  negative95thPercentileDiagnostic: {
    threshold: number | null;
    precision: number | null;
    recall: number | null;
    falsePositiveRate: number | null;
    truePositives?: number;
    falsePositives?: number;
  };
};

export type AppearanceScope = {
  events: number;
  positives: number;
  negatives: number;
  features: Record<AppearanceFeature, AppearanceMetric>;
};

export type AppearanceReport = {
  schemaVersion: number;
  kind: string;
  createdAt: string;
  labels: {
    directory: string;
    files: Array<{
      recordingId: string;
      environment: string;
      sourceGroup?: string;
      split?: string;
      sourceType?: string;
      targetStatus?: string;
      path: string | null;
      sha256: string | null;
      videoPath?: string;
      videoFilename?: string;
      durationSeconds?: number;
      candidateSource?: Record<string, unknown>;
      rallies: number;
      sideSwitches: number;
    }>;
  };
  protocol: {
    environments: string[];
    minimumGapSeconds: number;
    maximumNegativeEventsPerRecording: number;
    samplesPerSide: number;
    flankSeconds: number;
    edgeMarginSeconds: number;
    positiveSource: string;
    negativeSource: string;
    candidateSource?: string;
    evaluationTargetStatuses?: string[];
    corpusScope?: string;
    personProposal: string;
    colorRepresentation: string;
    thresholdsAreDiagnostic: boolean;
  };
  summary: {
    recordings: number;
    events: number;
    positives: number;
    negatives: number;
    candidateEvents?: number;
    targetStatusCounts?: Record<string, number>;
    sourceTypeCounts?: Record<string, number>;
    statuses: Record<string, number>;
    metrics: Record<string, AppearanceScope>;
    metricsByTargetStatus?: Record<string, Record<string, AppearanceScope>>;
  };
  events: AppearanceEvent[];
};

export type SideSwitchRecording = {
  recordingId: string;
  environment: string;
  durationSeconds: number;
  videoFilename: string;
  sourceType?: string;
  targetStatus?: string;
  timelineLoadError?: string;
  feedbackProducer: "android" | "production-web" | null;
  continuousVideoReviewed: boolean;
  gameWindow: { start: number; end: number } | null;
  sourceSideSwitches: Array<{ time: number; notes?: string }>;
  productionModelRanges: Array<{
    id: string;
    start: number;
    end: number;
    confidence: number | null;
    agreement?: string;
  }>;
  productionEditorRanges: Array<{
    id: string;
    coreStart: number;
    coreEnd: number;
    keepStart: number;
    keepEnd: number;
    confidence: number | null;
    included: boolean;
    origin?: string;
    agreement?: string;
  }>;
  productionIgnoredIntervals: Array<{
    id: string;
    start: number;
    end: number;
    reason?: string;
  }>;
  productionFinalIntervals: Array<{
    start: number;
    end: number;
    cutIds: string[];
  }>;
};

export type ReviewDecision = "switch" | "no-switch" | "unclear";

export type FullVideoSideSwitchMarker = {
  id: string;
  recordingId: string;
  time: number;
  createdAt: string;
};
