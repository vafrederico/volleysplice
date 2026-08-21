export type FlightReviewSide = "near" | "far";

export type ServerVisibility = "visible" | "partial" | "offscreen" | "unclear";
export type ContactTiming =
  | "on-anchor"
  | "before-anchor"
  | "after-anchor"
  | "unclear";
export type BallFlightVisibility = "visible" | "not-visible" | "unclear";
export type MotionDirectionAssessment =
  | "matches-human-side"
  | "opposes-human-side"
  | "unclear";

export type ServingSideFlightAnnotation = {
  serverVisibility: ServerVisibility;
  contactTiming: ContactTiming;
  correctedServeAnchorSeconds: number | null;
  ballFlightVisibility: BallFlightVisibility;
  motionDirection: MotionDirectionAssessment;
  notes: string;
  reviewedAt: string;
};

export type ServingSideFlightAnnotationState = {
  schemaVersion: 1;
  kind: "volleycut-serving-side-flight-error-annotations-v1";
  experimentKind: string;
  experimentCreatedAt: string;
  experimentSha256: string;
  predictionDigest: string;
  savedAt: string | null;
  annotations: Record<string, ServingSideFlightAnnotation>;
};

export type ServingSideFlightReviewResult = {
  rallyId: string;
  recordingId: string;
  environment: string;
  sourceGroup: string;
  split: string;
  start: number;
  end: number;
  human: FlightReviewSide;
  prediction: FlightReviewSide;
  probabilityNear: number;
  correct: boolean;
};

export type ServingSideFlightReviewRecording = {
  recordingId: string;
  environment: string;
  sourceGroup: string;
  split: string;
  durationSeconds: number;
  videoFilename: string;
  rows: number;
  errors: number;
};

export type ServingSideFlightReviewData = {
  kind: string;
  createdAt: string;
  experimentSha256: string;
  predictionDigest: string;
  configuration: string;
  featureFamily: string;
  l2: number;
  metrics: {
    rows: number;
    accuracy: number;
    balancedAccuracy: number;
    nearPrecision: number;
    nearRecall: number;
    farPrecision: number;
    farRecall: number;
  };
  annotationState: ServingSideFlightAnnotationState;
  recordings: ServingSideFlightReviewRecording[];
  results: ServingSideFlightReviewResult[];
};

export type ServingSideFlightAnnotationRequest = {
  schemaVersion: 1;
  experimentSha256: string;
  rallyId: string;
  annotation: Omit<ServingSideFlightAnnotation, "reviewedAt"> | null;
};
