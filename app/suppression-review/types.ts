export type ReviewInterval = {
  start: number;
  end: number;
  confidence?: number;
};

export type ReviewTrack = {
  raw: ReviewInterval[];
  padding: ReviewInterval[];
  joinedGaps: ReviewInterval[];
  padded: ReviewInterval[];
};

export type AffectedRally = {
  rallyNumber: number;
  start: number;
  end: number;
  duration: number;
  productionCoveredSeconds: number;
  candidateCoveredSeconds: number;
  lostCoreSeconds: number;
  completeMiss: boolean;
};

export type SuppressionReviewVideo = {
  recordingId: string;
  file: string;
  duration: number;
  environment: string;
  provenance: string;
  feedbackPartition: string | null;
  videoUrl: string;
  completeMisses: number;
  partialMisses: number;
  lostCoreSeconds: number;
  affectedRallies: AffectedRally[];
  focusRanges: ReviewInterval[];
  tracks: {
    affectedRallies: ReviewTrack;
    human: ReviewTrack;
    previousProduction: ReviewTrack;
    allLabelsV2: ReviewTrack;
    ensemble: ReviewTrack;
    ensembleSuppressed: ReviewTrack;
    suppressionApplied: ReviewTrack;
  };
};

type MetricSummary = {
  core: MetricTriplet;
  padded: MetricTriplet;
  P_pad: number;
  R_core: number;
  F1_padP_coreR: number;
  paddedModelExportSeconds: number;
};

export type MetricTriplet = {
  precision: number;
  recall: number;
  f1: number;
};

export type SuppressionReviewDataset = {
  schemaVersion: number;
  experiment: string;
  policyId: "pointwise" | "any-overlap" | "any-overlap-raw";
  policyLabel: string;
  modelVariantId?: string;
  modelVariantLabel?: string;
  suppressionModelPath?: string;
  suppressionModelSha256?: string | null;
  strategy: string;
  paddingSecondsBeforeAndAfter: number;
  joinGapSecondsStrictlyLessThan: number;
  agreementGrouping?: {
    paddingSecondsBeforeAndAfter: number;
    joinGapSecondsStrictlyLessThan: number;
    minimumRawSupportSecondsPerModel: number;
    effectiveRawGapSeconds: number;
  };
  confidenceNotes: {
    componentPredictions: string;
    suppressionApplied: string;
    derivedRanges: string;
  };
  summary: {
    videos: number;
    affectedRallies: number;
    completeMisses: number;
    partialMisses: number;
    lostCoreSeconds: number;
    exportTimeSavedSeconds: number;
    correctlyRemovedFalsePositivePredictions: number;
    correctlyRemovedRawSeconds: number;
    production: MetricSummary;
    candidate: MetricSummary;
  };
  videos: SuppressionReviewVideo[];
};
