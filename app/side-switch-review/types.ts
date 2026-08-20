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
  label: 0 | 1;
  source: string;
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
      path: string;
      sha256: string;
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
    personProposal: string;
    colorRepresentation: string;
    thresholdsAreDiagnostic: boolean;
  };
  summary: {
    recordings: number;
    events: number;
    positives: number;
    negatives: number;
    statuses: Record<string, number>;
    metrics: Record<string, AppearanceScope>;
  };
  events: AppearanceEvent[];
};

export type SideSwitchRecording = {
  recordingId: string;
  environment: string;
  durationSeconds: number;
  videoFilename: string;
};

export type ReviewDecision = "switch" | "no-switch" | "unclear";
