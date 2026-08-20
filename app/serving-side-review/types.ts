export const SERVING_VARIANTS = [
  "pixelMotion",
  "paletteChange",
  "hogAreaChange",
  "motionPalette",
  "motionPaletteHog",
  "baselineMotion",
  "baselinePalette",
  "baselineHogArea",
  "baselineMotionPalette",
] as const;

export type ServingVariant = (typeof SERVING_VARIANTS)[number];
export type ServingSide = "near" | "far";
export type ServingDecision = ServingSide | "unclear";

export type ServingSideCue = {
  side: ServingSide | null;
  strength: string;
  reason: string;
  matches: string[];
};

export type ServingVariantEvidence = {
  score: number | null;
  direction: ServingSide | null;
  predictedSide: ServingSide | null;
  confidence: number | null;
};

export type ServingRally = {
  rallyId: string;
  recordingId: string;
  environment: string;
  sourceGroup: string;
  split: string;
  sourceType?: string;
  targetStatus?: string;
  rallyIndex: number;
  start: number;
  end: number;
  notes: string | null;
  tags: string[];
  weakTarget: ServingSideCue;
  preTimes: number[];
  actionTimes: number[];
  status: string;
  error?: string;
  features: Record<string, number | null>;
  variants: Partial<Record<ServingVariant, ServingVariantEvidence>>;
  hogBefore?: Record<string, number | null>;
  hogAfter?: Record<string, number | null>;
  hogBaselineBefore?: Record<string, number | null>;
  hogBaselineAfter?: Record<string, number | null>;
  candidateSource?: Record<string, unknown>;
};

export type ServingMetric = {
  knownTargets: number;
  usableScores: number;
  scoreCoverage: number | null;
  directionalAccuracy: number | null;
  directionalBalancedAccuracy: number | null;
  directionalConfusion: Record<string, Record<string, number>>;
  decisionMargin: number;
  decidedRows: number;
  decisionCoverage: number | null;
  decisionAccuracy: number | null;
  decisionBalancedAccuracy: number | null;
  weakTargetStrengths: Record<string, number>;
};

export type ServingScope = {
  rows: number;
  knownTargets: number;
  variants: Record<ServingVariant, ServingMetric>;
};

export type ServingReport = {
  schemaVersion: number;
  kind: string;
  createdAt: string;
  labels: {
    directory: string;
    manifest?: string | null;
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
    }>;
  };
  protocol: {
    rallyAnchor: string;
    preOffsetsSeconds: number[];
    actionOffsetsSeconds: number[];
    roi: string;
    sideSplitFraction: number;
    baselineBandFraction: number;
    nearSideDefinition: string;
    farSideDefinition: string;
    hogProposal: string;
    weakTargetSource: string;
    weakTargetExclusions: string[];
    decisionMargin: number;
    corpusScope?: string;
    evaluationTargetStatuses?: string[];
    thresholdsAreDiagnostic: boolean;
  };
  variants: Record<
    ServingVariant,
    { label: string; description: string; components: string[] }
  >;
  summary: {
    recordings: number;
    rallies: number;
    targetableRallies: number;
    unknownRallies: number;
    targetSides: Record<string, number>;
    targetStrengths: Record<string, number>;
    statuses: Record<string, number>;
    targetStatusCounts?: Record<string, number>;
    sourceTypeCounts?: Record<string, number>;
    metrics: Record<string, ServingScope>;
    metricsByTargetStatus?: Record<string, Record<string, ServingScope>>;
  };
  rallies: ServingRally[];
};

export type ServingRecording = {
  recordingId: string;
  environment: string;
  durationSeconds: number;
  videoFilename: string;
  sourceType?: string;
  targetStatus?: string;
};

export type ServingReviewDecision = ServingDecision;
