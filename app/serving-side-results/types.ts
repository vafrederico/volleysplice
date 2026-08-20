export type ServingSideResultSide = "near" | "far";
export type ServingSideHumanLabel = ServingSideResultSide | "not-serve";

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
  nearProbability: number;
  correct: boolean;
  notes: string | null;
  tags: string[];
  features: Record<string, number | null>;
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
  selectedFeatureSet: string;
  threshold: number;
  metrics: ServingSideResultMetrics;
  correctionState: ServingSideCorrectionState;
  recordings: ServingSideResultRecording[];
  results: ServingSideResult[];
};
