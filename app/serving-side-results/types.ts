export type ServingSideResultSide = "near" | "far";
export type ServingSideHumanLabel = ServingSideResultSide | "not-serve";
export type ServingSideServePrediction = "serve" | "not-serve";
export type ServingSideServeDecisionSource =
  | "serve-head"
  | "production-rally-recovery"
  | "none";
export type ServingSideReviewRecommendation = "near" | "far" | "review";

export type ServingSideReviewPolicy = {
  precisionTarget: number;
  farThreshold: number;
  nearThreshold: number;
  developmentCoverage: number;
  developmentReviewFraction: number;
  developmentSelectiveAccuracy: number;
};

export type ServingSideServeHeadEvidence = {
  modelId: string;
  threshold: number;
  peakProbability: number;
  peakTime: number;
  crossesThreshold: boolean;
  nearestDetection: null | {
    time: number;
    confidence: number;
    distanceSeconds: number;
  };
};

export type ServingSideProductionRallyEvidence = {
  anchorContained: boolean;
  bothModels: boolean;
  recoversServe: boolean;
  interval: null | {
    start: number;
    end: number;
    agreement:
      | "both-models"
      | "all-labels-v2-only"
      | "previous-production-only";
  };
};

export type ServingSideResult = {
  rallyId: string;
  recordingId: string;
  environment: string;
  sourceGroup: string;
  split: string;
  sourceType: string | null;
  targetStatus: string | null;
  rallyIndex: number;
  start: number;
  end: number;
  human: ServingSideHumanLabel;
  originalHuman: ServingSideResultSide;
  humanCorrected: boolean;
  prediction: ServingSideResultSide;
  servePrediction: ServingSideServePrediction;
  serveDecisionSource: ServingSideServeDecisionSource;
  serveReviewRecommended: boolean;
  finalPrediction: ServingSideHumanLabel;
  serveEvidence: {
    serveAnchor: number;
    allLabelsV2: ServingSideServeHeadEvidence;
    previousProduction: ServingSideServeHeadEvidence;
    productionRally: ServingSideProductionRallyEvidence | null;
  };
  nearProbability: number;
  reviewRecommendation: ServingSideReviewRecommendation | null;
  correct: boolean;
  notes: string | null;
  tags: string[];
  features: Record<string, number | null>;
};

export type ServingSideServeGateMetrics = {
  rows: number;
  humanServes: number;
  humanNotServes: number;
  trueServes: number;
  falseServes: number;
  missedServes: number;
  trueNotServes: number;
  serveHeadPasses: number;
  recoveredServes: number;
  recoveredTrueServes: number;
  recoveredFalseServes: number;
  precision: number;
  recall: number;
  specificity: number;
  accuracy: number;
};

export type ServingSideResultRecording = {
  recordingId: string;
  environment: string;
  sourceGroup: string;
  split: string;
  sourceType: string | null;
  targetStatus: string | null;
  durationSeconds: number;
  videoFilename: string;
  rows: number;
  correct: number;
  errors: number;
};

export type ServingSideResultMetrics = {
  rows: number;
  accuracy: number;
  balancedAccuracy: number;
  nearPrecision: number;
  nearRecall: number;
  farPrecision: number;
  farRecall: number;
};

export type ServingSideCorrectionState = {
  schemaVersion: 1;
  reportKind: string;
  reportCreatedAt: string;
  baseDecisionSha256: string;
  savedAt: string | null;
  corrections: Record<string, ServingSideHumanLabel>;
};

export type ServingSideResultsData = {
  kind: string;
  createdAt: string;
  modelFingerprint: string;
  serveGateFingerprint: string;
  selectedFeatureSet: string;
  threshold: number;
  reviewPolicy: ServingSideReviewPolicy | null;
  metrics: ServingSideResultMetrics;
  serveGateMetrics: ServingSideServeGateMetrics;
  correctionState: ServingSideCorrectionState;
  recordings: ServingSideResultRecording[];
  results: ServingSideResult[];
};
