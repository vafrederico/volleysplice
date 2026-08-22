export const MODEL_FEEDBACK_SCHEMA = "volleycut-model-feedback" as const;
export const MODEL_FEEDBACK_SCHEMA_VERSION = 3 as const;
export const SUPPORTED_MODEL_FEEDBACK_SCHEMA_VERSIONS = [1, 2, 3] as const;

const MAX_NUMERIC_VALUES = 25_000_000;

export type FeedbackSource = "production-web" | "android";

export type FeedbackRange = {
  id: string;
  start: number;
  end: number;
  confidence: number;
  included?: boolean;
  agreement?: string;
};

export type CorrectedFeedbackRange = {
  id: string;
  coreStart: number;
  coreEnd: number;
  keepStart: number;
  keepEnd: number;
  confidence: number;
  included: boolean;
  origin: "cached-label" | "manual";
  agreement?: string;
};

export type IgnoredFeedbackRange = {
  id: string;
  start: number;
  end: number;
  reason: string;
};

export type FinalFeedbackRange = {
  start: number;
  end: number;
  cutIds: string[];
  joinedGaps: Array<{ start: number; end: number }>;
};

export type ParsedModelFeedback = {
  schema: typeof MODEL_FEEDBACK_SCHEMA;
  schemaVersion: (typeof SUPPORTED_MODEL_FEEDBACK_SCHEMA_VERSIONS)[number];
  generatedAt: string;
  producer: FeedbackSource;
  source: {
    projectId: string;
    analysisId: string;
    timelineCoordinates: "seconds-from-start-of-source";
    file: {
      name: string;
      sizeBytes: number;
      lastModifiedMs: number;
      mimeType: string;
      sampledFingerprint: string | null;
    };
    media: {
      duration: number;
      mimeType: string;
      width: number;
      height: number;
      rotation: number;
      videoCodec: string;
      videoCodecString: string | null;
      canDecodeVideo: boolean;
      hasAudio: boolean;
      audioCodec: string | null;
      sampleRate: number | null;
      channels: number | null;
      canDecodeAudio: boolean;
    };
    gameWindow: { start: number; end: number };
    featureRoi: { x: number; y: number; width: number; height: number };
    runtimeVariant: string;
  };
  features: {
    analysisFps: number;
    rows: number;
    columns: number;
    names: string[];
    timestamps: Float64Array;
    values: Float32Array;
  } | null;
  initialInference: {
    modelId: string;
    components: Array<{ modelId: string; bundleSha256: string }>;
    ensembleAlgorithmVersion: string;
    ranges: FeedbackRange[];
    probabilityModelId: string;
    timestamps: Float64Array;
    rallyProbabilities: Float32Array;
    serveProbabilities: Float32Array;
    deadStateProbabilities: Float32Array;
    productionComponents: null | {
      allLabelsV2: FeedbackRange[];
      previousProduction: FeedbackRange[];
    };
    componentServeOutputs: null | {
      allLabelsV2: ParsedServeOutput;
      previousProduction: ParsedServeOutput;
    };
    servingSide: ParsedServingSideOutput | null;
    suppression: ParsedSuppressionOutput | null;
  };
  corrections: {
    updatedAt: string;
    beforePaddingSeconds: number;
    afterPaddingSeconds: number;
    joinGapSeconds: number;
    correctedRanges: CorrectedFeedbackRange[];
    ignoredIntervals: IgnoredFeedbackRange[];
    suppression: ParsedSuppressionCorrections | null;
    scoreTracking: ParsedScoreTrackingFeedback | null;
    labels: {
      falsePositives: FeedbackRange[];
      falseNegatives: FeedbackRange[];
      confirmedModelRanges: FeedbackRange[];
      discardedManualRanges: FeedbackRange[];
    };
  };
  finalExportIntervals: FinalFeedbackRange[];
  warnings: string[];
};

export type ParsedSuppressionSuggestion = {
  id: string;
  logicalId: string;
  suppressionEventId: string;
  start: number;
  end: number;
  score: number;
  sourceProductionIds: string[];
  eligiblePolicyIds: Array<"conservative" | "balanced" | "aggressive">;
};

export type ParsedSuppressionOutput = {
  modelId: string;
  artifactSha256: string;
  weightsSha256: string;
  decoderVersion: string;
  policyContractVersion: number;
  probabilities: Float32Array;
  decodedIntervals: FeedbackRange[];
  suggestions: ParsedSuppressionSuggestion[];
  identicalPolicyResults: boolean;
};

export type ParsedSuppressionCorrections = {
  selectedPolicy: "none" | "conservative" | "balanced" | "aggressive";
  decisionOverrides: Record<string, "keep" | "suppress">;
  defaultSuppressionScope: "whole-rally" | "veto-region";
  suppressionScopeOverrides: Record<string, "whole-rally" | "veto-region">;
  userTouchedCutIds: string[];
  decisions: Array<{
    suggestionId: string;
    logicalId: string;
    state: "dormant" | "suppressed" | "kept" | "edited-kept";
    scope: "whole-rally" | "veto-region";
  }>;
};

export type ParsedServeOutput = {
  modelId: string;
  probabilities: Float32Array;
  detections: Array<{ time: number; confidence: number }>;
};

export type ParsedServingSideEvidence = {
  modelId: string;
  threshold: number;
  peakProbability: number;
  peakTime: number;
  crossesThreshold: boolean;
  nearestDetection: { time: number; confidence: number } | null;
};

export type ParsedServingSideOutput = {
  modelId: string;
  modelFingerprint: string;
  featureVersion: string;
  anchorContract: string;
  features: {
    rows: number;
    columns: number;
    values: Float64Array;
  };
  candidates: Array<{
    id: string;
    anchor: number;
    interval: { start: number; end: number; agreement?: string };
    nearProbability: number;
    side: "near" | "far";
    verdict: "near" | "far" | "review" | "not-serve";
    serveDecisionSource: "serve-head" | "production-rally-recovery" | "none";
    reviewReasons: Array<"side-score" | "production-rally-recovery">;
    serveEvidence: {
      allLabelsV2: ParsedServingSideEvidence;
      previousProduction: ParsedServingSideEvidence;
    };
  }>;
};

export type ParsedScoreTrackingFeedback = {
  state: {
    version: number;
    enabled: boolean;
    team1Name: string;
    team2Name: string;
    serveMarkers: Array<{
      id: string;
      timestamp: number;
      side: "near" | "far" | "review";
      origin: "model" | "manual";
      modelSide?: "near" | "far" | "review";
      ignorePreviousPoint: boolean;
      rallyId?: string;
    }>;
    sideSwitchMarkers: Array<{ id: string; timestamp: number }>;
    removedModelMarkerIds: string[];
  };
  excludedRallyIds: string[];
  derivedFinalScore: {
    team1Score: number;
    team2Score: number;
    servingTeamId: "team-1" | "team-2" | null;
    servingSide: "near" | "far" | "review" | null;
    ignoredPointCount: number;
    reviewPointCount: number;
    points: Array<{
      serveMarkerId: string;
      timestamp: number;
      servingSide: "near" | "far" | "review";
      winnerTeamId: "team-1" | "team-2" | null;
      status: "counted" | "ignored" | "review";
      team1ScoreAfter: number;
      team2ScoreAfter: number;
    }>;
  };
};

export class ModelFeedbackValidationError extends Error {}

function fail(field: string, message: string): never {
  throw new ModelFeedbackValidationError(`${field} ${message}`);
}

function object(value: unknown, field: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    fail(field, "must be an object");
  }
  return value as Record<string, unknown>;
}

function string(value: unknown, field: string): string {
  if (typeof value !== "string" || value.length === 0)
    fail(field, "must be a non-empty string");
  return value as string;
}

function text(value: unknown, field: string): string {
  if (typeof value !== "string") fail(field, "must be a string");
  return value as string;
}

function nullableString(value: unknown, field: string): string | null {
  if (value === null) return null;
  return string(value, field);
}

function number(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isFinite(value))
    fail(field, "must be finite");
  return value as number;
}

function nonNegative(value: unknown, field: string): number {
  const parsed = number(value, field);
  if (parsed < 0) fail(field, "must be non-negative");
  return parsed;
}

function integer(value: unknown, field: string): number {
  const parsed = nonNegative(value, field);
  if (!Number.isSafeInteger(parsed)) fail(field, "must be a safe integer");
  return parsed;
}

function boolean(value: unknown, field: string): boolean {
  if (typeof value !== "boolean") fail(field, "must be a boolean");
  return value as boolean;
}

function date(value: unknown, field: string): string {
  const parsed = string(value, field);
  if (!Number.isFinite(Date.parse(parsed))) fail(field, "must be an ISO date");
  return parsed;
}

function stringArray(value: unknown, field: string): string[] {
  if (!Array.isArray(value)) fail(field, "must be an array");
  return value.map((item, index) => string(item, `${field}[${index}]`));
}

function optionalNumber(value: unknown, field: string): number | null {
  return value === null ? null : number(value, field);
}

function decodeBase64(value: string, field: string): Uint8Array {
  if (
    value.length % 4 !== 0 ||
    !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(
      value,
    )
  ) {
    fail(field, "must be canonical base64");
  }
  try {
    const binary = atob(value);
    return Uint8Array.from(binary, (character) => character.charCodeAt(0));
  } catch {
    fail(field, "must be valid base64");
  }
}

function decodeNumericArray(
  value: unknown,
  field: string,
  dataType: "float32" | "float64",
  expectedShape?: number[],
): Float32Array | Float64Array {
  const payload = object(value, field);
  if (payload.encoding !== "base64")
    fail(`${field}.encoding`, "must be base64");
  if (payload.byteOrder !== "little-endian") {
    fail(`${field}.byteOrder`, "must be little-endian");
  }
  if (payload.dataType !== dataType)
    fail(`${field}.dataType`, `must be ${dataType}`);
  if (!Array.isArray(payload.shape) || payload.shape.length === 0) {
    fail(`${field}.shape`, "must be a non-empty array");
  }
  const shape = payload.shape.map((dimension, index) =>
    integer(dimension, `${field}.shape[${index}]`),
  );
  if (
    expectedShape &&
    (shape.length !== expectedShape.length ||
      shape.some((item, index) => item !== expectedShape[index]))
  ) {
    fail(`${field}.shape`, `must be [${expectedShape.join(", ")}]`);
  }
  const values = shape.reduce((product, dimension) => product * dimension, 1);
  if (!Number.isSafeInteger(values) || values > MAX_NUMERIC_VALUES) {
    fail(`${field}.shape`, "contains too many values");
  }
  const bytes = decodeBase64(
    string(payload.data, `${field}.data`),
    `${field}.data`,
  );
  const elementBytes = dataType === "float32" ? 4 : 8;
  if (bytes.byteLength !== values * elementBytes) {
    fail(`${field}.data`, "byte length does not match its shape");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (dataType === "float32") {
    const output = new Float32Array(values);
    for (let index = 0; index < values; index += 1) {
      const decoded = view.getFloat32(index * elementBytes, true);
      if (!Number.isFinite(decoded))
        fail(`${field}.data`, "must contain only finite values");
      output[index] = decoded;
    }
    return output;
  }
  const output = new Float64Array(values);
  for (let index = 0; index < values; index += 1) {
    const decoded = view.getFloat64(index * elementBytes, true);
    if (!Number.isFinite(decoded))
      fail(`${field}.data`, "must contain only finite values");
    output[index] = decoded;
  }
  return output;
}

function checkBounds(
  start: number,
  end: number,
  duration: number,
  field: string,
): void {
  if (start < 0 || end <= start || end > duration) {
    fail(field, `must satisfy 0 <= start < end <= ${duration}`);
  }
}

function range(value: unknown, field: string, duration: number): FeedbackRange {
  const item = object(value, field);
  const start = number(item.start, `${field}.start`);
  const end = number(item.end, `${field}.end`);
  checkBounds(start, end, duration, field);
  const confidence = number(item.confidence, `${field}.confidence`);
  if (confidence < 0 || confidence > 1)
    fail(`${field}.confidence`, "must be between 0 and 1");
  return {
    id: string(item.id, `${field}.id`),
    start,
    end,
    confidence,
    ...(item.included === undefined
      ? {}
      : { included: boolean(item.included, `${field}.included`) }),
    ...(item.agreement === undefined
      ? {}
      : { agreement: string(item.agreement, `${field}.agreement`) }),
  };
}

function ranges(
  value: unknown,
  field: string,
  duration: number,
): FeedbackRange[] {
  if (!Array.isArray(value)) fail(field, "must be an array");
  return value.map((item, index) =>
    range(item, `${field}[${index}]`, duration),
  );
}

function correctedRange(
  value: unknown,
  field: string,
  duration: number,
): CorrectedFeedbackRange {
  const item = object(value, field);
  const coreStart = number(item.coreStart, `${field}.coreStart`);
  const coreEnd = number(item.coreEnd, `${field}.coreEnd`);
  const keepStart = number(item.keepStart, `${field}.keepStart`);
  const keepEnd = number(item.keepEnd, `${field}.keepEnd`);
  checkBounds(coreStart, coreEnd, duration, field);
  checkBounds(keepStart, keepEnd, duration, field);
  if (keepStart > coreStart || keepEnd < coreEnd)
    fail(field, "padding must contain its core range");
  const confidence = number(item.confidence, `${field}.confidence`);
  if (confidence < 0 || confidence > 1)
    fail(`${field}.confidence`, "must be between 0 and 1");
  if (item.origin !== "cached-label" && item.origin !== "manual") {
    fail(`${field}.origin`, "must be cached-label or manual");
  }
  return {
    id: string(item.id, `${field}.id`),
    coreStart,
    coreEnd,
    keepStart,
    keepEnd,
    confidence,
    included: boolean(item.included, `${field}.included`),
    origin: item.origin,
    ...(item.agreement === undefined
      ? {}
      : { agreement: string(item.agreement, `${field}.agreement`) }),
  };
}

function validateTimestamps(
  values: Float64Array,
  duration: number,
  field: string,
): void {
  for (let index = 0; index < values.length; index += 1) {
    if (
      values[index] < 0 ||
      values[index] > duration ||
      (index > 0 && values[index] <= values[index - 1])
    ) {
      fail(field, "must be strictly increasing and inside the source duration");
    }
  }
}

function validateProbabilities(values: Float32Array, field: string): void {
  if (values.some((value) => value < 0 || value > 1)) {
    fail(field, "must contain values between 0 and 1");
  }
}

function sourceProducer(runtimeVariant: string): FeedbackSource {
  return runtimeVariant.startsWith("native-android")
    ? "android"
    : "production-web";
}

function choice<const Values extends readonly string[]>(
  value: unknown,
  values: Values,
  field: string,
): Values[number] {
  if (!values.includes(value as Values[number])) {
    fail(field, `must be one of ${values.join(", ")}`);
  }
  return value as Values[number];
}

function probability(value: unknown, field: string): number {
  const parsed = number(value, field);
  if (parsed < 0 || parsed > 1) fail(field, "must be between 0 and 1");
  return parsed;
}

function sourceTime(value: unknown, field: string, duration: number): number {
  const parsed = nonNegative(value, field);
  if (parsed > duration) fail(field, "must be within the source");
  return parsed;
}

function parseServingSideOutput(
  value: unknown,
  field: string,
  duration: number,
): ParsedServingSideOutput {
  const output = object(value, field);
  const features = object(output.features, `${field}.features`);
  const rows = integer(features.rows, `${field}.features.rows`);
  const columns = integer(features.columns, `${field}.features.columns`);
  if (columns <= 0) fail(`${field}.features.columns`, "must be positive");
  const values = decodeNumericArray(
    features.values,
    `${field}.features.values`,
    "float64",
    [rows, columns],
  ) as Float64Array;
  if (!Array.isArray(output.candidates)) {
    fail(`${field}.candidates`, "must be an array");
  }
  if (output.candidates.length !== rows) {
    fail(`${field}.candidates`, "length must match feature rows");
  }
  const candidateIds = new Set<string>();
  const parseEvidence = (
    value: unknown,
    evidenceField: string,
  ): ParsedServingSideEvidence => {
    const evidence = object(value, evidenceField);
    const nearestDetection =
      evidence.nearestDetection === null
        ? null
        : (() => {
            const detection = object(
              evidence.nearestDetection,
              `${evidenceField}.nearestDetection`,
            );
            return {
              time: sourceTime(
                detection.time,
                `${evidenceField}.nearestDetection.time`,
                duration,
              ),
              confidence: probability(
                detection.confidence,
                `${evidenceField}.nearestDetection.confidence`,
              ),
            };
          })();
    return {
      modelId: string(evidence.modelId, `${evidenceField}.modelId`),
      threshold: probability(evidence.threshold, `${evidenceField}.threshold`),
      peakProbability: probability(
        evidence.peakProbability,
        `${evidenceField}.peakProbability`,
      ),
      peakTime: sourceTime(
        evidence.peakTime,
        `${evidenceField}.peakTime`,
        duration,
      ),
      crossesThreshold: boolean(
        evidence.crossesThreshold,
        `${evidenceField}.crossesThreshold`,
      ),
      nearestDetection,
    };
  };
  const candidates = output.candidates.map((value, index) => {
    const candidateField = `${field}.candidates[${index}]`;
    const candidate = object(value, candidateField);
    const id = string(candidate.id, `${candidateField}.id`);
    if (candidateIds.has(id)) fail(`${candidateField}.id`, "must be unique");
    candidateIds.add(id);
    const interval = object(candidate.interval, `${candidateField}.interval`);
    const start = sourceTime(
      interval.start,
      `${candidateField}.interval.start`,
      duration,
    );
    const end = sourceTime(
      interval.end,
      `${candidateField}.interval.end`,
      duration,
    );
    if (end <= start)
      fail(`${candidateField}.interval`, "must have positive duration");
    const anchor = sourceTime(
      candidate.anchor,
      `${candidateField}.anchor`,
      duration,
    );
    if (anchor !== start)
      fail(`${candidateField}.anchor`, "must equal interval start");
    if (!Array.isArray(candidate.reviewReasons)) {
      fail(`${candidateField}.reviewReasons`, "must be an array");
    }
    const serveEvidence = object(
      candidate.serveEvidence,
      `${candidateField}.serveEvidence`,
    );
    return {
      id,
      anchor,
      interval: {
        start,
        end,
        ...(interval.agreement === undefined
          ? {}
          : {
              agreement: string(
                interval.agreement,
                `${candidateField}.interval.agreement`,
              ),
            }),
      },
      nearProbability: probability(
        candidate.nearProbability,
        `${candidateField}.nearProbability`,
      ),
      side: choice(
        candidate.side,
        ["near", "far"] as const,
        `${candidateField}.side`,
      ),
      verdict: choice(
        candidate.verdict,
        ["near", "far", "review", "not-serve"] as const,
        `${candidateField}.verdict`,
      ),
      serveDecisionSource: choice(
        candidate.serveDecisionSource,
        ["serve-head", "production-rally-recovery", "none"] as const,
        `${candidateField}.serveDecisionSource`,
      ),
      reviewReasons: candidate.reviewReasons.map((reason, reasonIndex) =>
        choice(
          reason,
          ["side-score", "production-rally-recovery"] as const,
          `${candidateField}.reviewReasons[${reasonIndex}]`,
        ),
      ),
      serveEvidence: {
        allLabelsV2: parseEvidence(
          serveEvidence.allLabelsV2,
          `${candidateField}.serveEvidence.allLabelsV2`,
        ),
        previousProduction: parseEvidence(
          serveEvidence.previousProduction,
          `${candidateField}.serveEvidence.previousProduction`,
        ),
      },
    };
  });
  return {
    modelId: string(output.modelId, `${field}.modelId`),
    modelFingerprint: string(
      output.modelFingerprint,
      `${field}.modelFingerprint`,
    ),
    featureVersion: string(output.featureVersion, `${field}.featureVersion`),
    anchorContract: string(output.anchorContract, `${field}.anchorContract`),
    features: { rows, columns, values },
    candidates,
  };
}

function parseScoreTrackingFeedback(
  value: unknown,
  field: string,
  duration: number,
): ParsedScoreTrackingFeedback {
  const feedback = object(value, field);
  const state = object(feedback.state, `${field}.state`);
  if (!Array.isArray(state.serveMarkers)) {
    fail(`${field}.state.serveMarkers`, "must be an array");
  }
  if (!Array.isArray(state.sideSwitchMarkers)) {
    fail(`${field}.state.sideSwitchMarkers`, "must be an array");
  }
  const markerIds = new Set<string>();
  const serveMarkers = state.serveMarkers.map((value, index) => {
    const markerField = `${field}.state.serveMarkers[${index}]`;
    const marker = object(value, markerField);
    const id = string(marker.id, `${markerField}.id`);
    if (markerIds.has(id)) fail(`${markerField}.id`, "must be unique");
    markerIds.add(id);
    return {
      id,
      timestamp: sourceTime(
        marker.timestamp,
        `${markerField}.timestamp`,
        duration,
      ),
      side: choice(
        marker.side,
        ["near", "far", "review"] as const,
        `${markerField}.side`,
      ),
      origin: choice(
        marker.origin,
        ["model", "manual"] as const,
        `${markerField}.origin`,
      ),
      ...(marker.modelSide === undefined
        ? {}
        : {
            modelSide: choice(
              marker.modelSide,
              ["near", "far", "review"] as const,
              `${markerField}.modelSide`,
            ),
          }),
      ignorePreviousPoint: boolean(
        marker.ignorePreviousPoint,
        `${markerField}.ignorePreviousPoint`,
      ),
      ...(marker.rallyId === undefined
        ? {}
        : { rallyId: string(marker.rallyId, `${markerField}.rallyId`) }),
    };
  });
  const sideSwitchMarkers = state.sideSwitchMarkers.map((value, index) => {
    const markerField = `${field}.state.sideSwitchMarkers[${index}]`;
    const marker = object(value, markerField);
    const id = string(marker.id, `${markerField}.id`);
    if (markerIds.has(id)) fail(`${markerField}.id`, "must be unique");
    markerIds.add(id);
    return {
      id,
      timestamp: sourceTime(
        marker.timestamp,
        `${markerField}.timestamp`,
        duration,
      ),
    };
  });
  const removedModelMarkerIds = stringArray(
    state.removedModelMarkerIds,
    `${field}.state.removedModelMarkerIds`,
  );
  if (new Set(removedModelMarkerIds).size !== removedModelMarkerIds.length) {
    fail(`${field}.state.removedModelMarkerIds`, "must contain unique IDs");
  }
  if (removedModelMarkerIds.some((id) => markerIds.has(id))) {
    fail(
      `${field}.state.removedModelMarkerIds`,
      "must not contain active marker IDs",
    );
  }
  const excludedRallyIds = stringArray(
    feedback.excludedRallyIds,
    `${field}.excludedRallyIds`,
  );
  if (new Set(excludedRallyIds).size !== excludedRallyIds.length) {
    fail(`${field}.excludedRallyIds`, "must contain unique IDs");
  }
  const derived = object(
    feedback.derivedFinalScore,
    `${field}.derivedFinalScore`,
  );
  if (!Array.isArray(derived.points)) {
    fail(`${field}.derivedFinalScore.points`, "must be an array");
  }
  const team = (value: unknown, teamField: string) =>
    value === null
      ? null
      : choice(value, ["team-1", "team-2"] as const, teamField);
  const side = (value: unknown, sideField: string) =>
    value === null
      ? null
      : choice(value, ["near", "far", "review"] as const, sideField);
  const points = derived.points.map((value, index) => {
    const pointField = `${field}.derivedFinalScore.points[${index}]`;
    const point = object(value, pointField);
    return {
      serveMarkerId: string(point.serveMarkerId, `${pointField}.serveMarkerId`),
      timestamp: sourceTime(
        point.timestamp,
        `${pointField}.timestamp`,
        duration,
      ),
      servingSide: choice(
        point.servingSide,
        ["near", "far", "review"] as const,
        `${pointField}.servingSide`,
      ),
      winnerTeamId: team(point.winnerTeamId, `${pointField}.winnerTeamId`),
      status: choice(
        point.status,
        ["counted", "ignored", "review"] as const,
        `${pointField}.status`,
      ),
      team1ScoreAfter: integer(
        point.team1ScoreAfter,
        `${pointField}.team1ScoreAfter`,
      ),
      team2ScoreAfter: integer(
        point.team2ScoreAfter,
        `${pointField}.team2ScoreAfter`,
      ),
    };
  });
  return {
    state: {
      version: integer(state.version, `${field}.state.version`),
      enabled: boolean(state.enabled, `${field}.state.enabled`),
      team1Name: string(state.team1Name, `${field}.state.team1Name`),
      team2Name: string(state.team2Name, `${field}.state.team2Name`),
      serveMarkers,
      sideSwitchMarkers,
      removedModelMarkerIds,
    },
    excludedRallyIds,
    derivedFinalScore: {
      team1Score: integer(
        derived.team1Score,
        `${field}.derivedFinalScore.team1Score`,
      ),
      team2Score: integer(
        derived.team2Score,
        `${field}.derivedFinalScore.team2Score`,
      ),
      servingTeamId: team(
        derived.servingTeamId,
        `${field}.derivedFinalScore.servingTeamId`,
      ),
      servingSide: side(
        derived.servingSide,
        `${field}.derivedFinalScore.servingSide`,
      ),
      ignoredPointCount: integer(
        derived.ignoredPointCount,
        `${field}.derivedFinalScore.ignoredPointCount`,
      ),
      reviewPointCount: integer(
        derived.reviewPointCount,
        `${field}.derivedFinalScore.reviewPointCount`,
      ),
      points,
    },
  };
}

export function parseModelFeedback(value: unknown): ParsedModelFeedback {
  const root = object(value, "bundle");
  if (root.schema !== MODEL_FEEDBACK_SCHEMA) {
    fail("bundle.schema", `must be ${MODEL_FEEDBACK_SCHEMA}`);
  }
  if (
    root.schemaVersion !== 1 &&
    root.schemaVersion !== 2 &&
    root.schemaVersion !== MODEL_FEEDBACK_SCHEMA_VERSION
  ) {
    fail("bundle.schemaVersion", "must be 1, 2, or 3");
  }
  const schemaVersion = root.schemaVersion;

  const source = object(root.source, "bundle.source");
  if (source.timelineCoordinates !== "seconds-from-start-of-source") {
    fail(
      "bundle.source.timelineCoordinates",
      "must be seconds-from-start-of-source",
    );
  }
  if (source.videoBytesIncluded !== false) {
    fail("bundle.source.videoBytesIncluded", "must be false");
  }
  const file = object(source.file, "bundle.source.file");
  const media = object(source.media, "bundle.source.media");
  const gameWindow = object(source.gameWindow, "bundle.source.gameWindow");
  const featureRoi = object(source.featureRoi, "bundle.source.featureRoi");
  const duration = nonNegative(media.duration, "bundle.source.media.duration");
  if (duration <= 0) fail("bundle.source.media.duration", "must be positive");
  const windowStart = nonNegative(
    gameWindow.start,
    "bundle.source.gameWindow.start",
  );
  const windowEnd = number(gameWindow.end, "bundle.source.gameWindow.end");
  if (windowEnd <= windowStart || windowEnd > duration) {
    fail("bundle.source.gameWindow", "must be inside the source duration");
  }
  const roi = {
    x: number(featureRoi.x, "bundle.source.featureRoi.x"),
    y: number(featureRoi.y, "bundle.source.featureRoi.y"),
    width: number(featureRoi.width, "bundle.source.featureRoi.width"),
    height: number(featureRoi.height, "bundle.source.featureRoi.height"),
  };
  if (
    roi.x < 0 ||
    roi.y < 0 ||
    roi.width <= 0 ||
    roi.height <= 0 ||
    roi.x + roi.width > 1.000001 ||
    roi.y + roi.height > 1.000001
  ) {
    fail("bundle.source.featureRoi", "must be a normalized rectangle");
  }
  const runtimeVariant = string(
    source.runtimeVariant,
    "bundle.source.runtimeVariant",
  );

  let features: ParsedModelFeedback["features"] = null;
  if (root.features !== null) {
    const feature = object(root.features, "bundle.features");
    const rows = integer(feature.rows, "bundle.features.rows");
    const columns = integer(feature.columns, "bundle.features.columns");
    const names = stringArray(feature.names, "bundle.features.names");
    if (columns !== names.length)
      fail("bundle.features.names", "length must match columns");
    features = {
      analysisFps: nonNegative(
        feature.analysisFps,
        "bundle.features.analysisFps",
      ),
      rows,
      columns,
      names,
      timestamps: decodeNumericArray(
        feature.timestamps,
        "bundle.features.timestamps",
        "float64",
        [rows],
      ) as Float64Array,
      values: decodeNumericArray(
        feature.values,
        "bundle.features.values",
        "float32",
        [rows, columns],
      ) as Float32Array,
    };
    validateTimestamps(
      features.timestamps,
      duration,
      "bundle.features.timestamps",
    );
  }

  const inference = object(root.initialInference, "bundle.initialInference");
  const inferenceTimesPayload = object(
    inference.timestamps,
    "bundle.initialInference.timestamps",
  );
  if (
    !Array.isArray(inferenceTimesPayload.shape) ||
    inferenceTimesPayload.shape.length !== 1
  ) {
    fail("bundle.initialInference.timestamps.shape", "must have one dimension");
  }
  const inferenceRows = integer(
    inferenceTimesPayload.shape[0],
    "bundle.initialInference.timestamps.shape[0]",
  );
  const probabilities = object(
    inference.probabilities,
    "bundle.initialInference.probabilities",
  );
  const componentsValue = inference.components;
  if (!Array.isArray(componentsValue))
    fail("bundle.initialInference.components", "must be an array");
  const components = componentsValue.map((component, index) => {
    const item = object(
      component,
      `bundle.initialInference.components[${index}]`,
    );
    return {
      modelId: string(
        item.modelId,
        `bundle.initialInference.components[${index}].modelId`,
      ),
      bundleSha256: string(
        item.bundleSha256,
        `bundle.initialInference.components[${index}].bundleSha256`,
      ),
    };
  });
  const inferenceTimestamps = decodeNumericArray(
    inference.timestamps,
    "bundle.initialInference.timestamps",
    "float64",
    [inferenceRows],
  ) as Float64Array;
  const rallyProbabilities = decodeNumericArray(
    probabilities.rally,
    "bundle.initialInference.probabilities.rally",
    "float32",
    [inferenceRows],
  ) as Float32Array;
  const serveProbabilities = decodeNumericArray(
    probabilities.serve,
    "bundle.initialInference.probabilities.serve",
    "float32",
    [inferenceRows],
  ) as Float32Array;
  const deadStateProbabilities = decodeNumericArray(
    probabilities.deadState,
    "bundle.initialInference.probabilities.deadState",
    "float32",
    [inferenceRows],
  ) as Float32Array;
  validateTimestamps(
    inferenceTimestamps,
    duration,
    "bundle.initialInference.timestamps",
  );
  validateProbabilities(
    rallyProbabilities,
    "bundle.initialInference.probabilities.rally",
  );
  validateProbabilities(
    serveProbabilities,
    "bundle.initialInference.probabilities.serve",
  );
  validateProbabilities(
    deadStateProbabilities,
    "bundle.initialInference.probabilities.deadState",
  );
  const componentServeOutputs = (() => {
    if (inference.componentServeOutputs === undefined) return null;
    const outputs = object(
      inference.componentServeOutputs,
      "bundle.initialInference.componentServeOutputs",
    );
    const componentModelIds = new Set(components.map(({ modelId }) => modelId));
    const parseOutput = (value: unknown, field: string): ParsedServeOutput => {
      const output = object(value, field);
      const modelId = string(output.modelId, `${field}.modelId`);
      if (!componentModelIds.has(modelId)) {
        fail(
          `${field}.modelId`,
          "must identify an initial-inference component",
        );
      }
      const probabilities = decodeNumericArray(
        output.probabilities,
        `${field}.probabilities`,
        "float32",
        [inferenceRows],
      ) as Float32Array;
      validateProbabilities(probabilities, `${field}.probabilities`);
      if (!Array.isArray(output.detections)) {
        fail(`${field}.detections`, "must be an array");
      }
      const detections = output.detections.map((value, index) => {
        const detection = object(value, `${field}.detections[${index}]`);
        const time = nonNegative(
          detection.time,
          `${field}.detections[${index}].time`,
        );
        const confidence = number(
          detection.confidence,
          `${field}.detections[${index}].confidence`,
        );
        if (time > duration) {
          fail(
            `${field}.detections[${index}].time`,
            "must be within the source",
          );
        }
        if (confidence < 0 || confidence > 1) {
          fail(
            `${field}.detections[${index}].confidence`,
            "must be between 0 and 1",
          );
        }
        return { time, confidence };
      });
      return { modelId, probabilities, detections };
    };
    return {
      allLabelsV2: parseOutput(
        outputs.allLabelsV2,
        "bundle.initialInference.componentServeOutputs.allLabelsV2",
      ),
      previousProduction: parseOutput(
        outputs.previousProduction,
        "bundle.initialInference.componentServeOutputs.previousProduction",
      ),
    };
  })();
  const servingSide = (() => {
    if (inference.servingSide === undefined) {
      if (schemaVersion >= 3) {
        fail("bundle.initialInference.servingSide", "is required");
      }
      return null;
    }
    if (inference.servingSide === null) return null;
    return parseServingSideOutput(
      inference.servingSide,
      "bundle.initialInference.servingSide",
      duration,
    );
  })();
  const productionComponents = (() => {
    if (inference.productionComponents === undefined) return null;
    const components = object(
      inference.productionComponents,
      "bundle.initialInference.productionComponents",
    );
    return {
      allLabelsV2: ranges(
        components.allLabelsV2,
        "bundle.initialInference.productionComponents.allLabelsV2",
        duration,
      ),
      previousProduction: ranges(
        components.previousProduction,
        "bundle.initialInference.productionComponents.previousProduction",
        duration,
      ),
    };
  })();
  const suppression = (() => {
    if (inference.suppression === undefined || inference.suppression === null) {
      return null;
    }
    const value = object(
      inference.suppression,
      "bundle.initialInference.suppression",
    );
    const probabilities = decodeNumericArray(
      value.probabilities,
      "bundle.initialInference.suppression.probabilities",
      "float32",
      [inferenceRows],
    ) as Float32Array;
    validateProbabilities(
      probabilities,
      "bundle.initialInference.suppression.probabilities",
    );
    const timestamps = decodeNumericArray(
      value.timestamps,
      "bundle.initialInference.suppression.timestamps",
      "float64",
      [inferenceRows],
    ) as Float64Array;
    validateTimestamps(
      timestamps,
      duration,
      "bundle.initialInference.suppression.timestamps",
    );
    if (!Array.isArray(value.suggestions)) {
      fail(
        "bundle.initialInference.suppression.suggestions",
        "must be an array",
      );
    }
    const suggestions = value.suggestions.map((item, index) => {
      const field = `bundle.initialInference.suppression.suggestions[${index}]`;
      const suggestion = object(item, field);
      const start = sourceTime(suggestion.start, `${field}.start`, duration);
      const end = sourceTime(suggestion.end, `${field}.end`, duration);
      if (end <= start) fail(field, "must have positive duration");
      if (!Array.isArray(suggestion.eligiblePolicyIds)) {
        fail(`${field}.eligiblePolicyIds`, "must be an array");
      }
      return {
        id: string(suggestion.id, `${field}.id`),
        logicalId: string(suggestion.logicalId, `${field}.logicalId`),
        suppressionEventId: string(
          suggestion.suppressionEventId,
          `${field}.suppressionEventId`,
        ),
        start,
        end,
        score: probability(suggestion.score, `${field}.score`),
        sourceProductionIds: stringArray(
          suggestion.sourceProductionIds,
          `${field}.sourceProductionIds`,
        ),
        eligiblePolicyIds: suggestion.eligiblePolicyIds.map(
          (policy, policyIndex) =>
            choice(
              policy,
              ["conservative", "balanced", "aggressive"] as const,
              `${field}.eligiblePolicyIds[${policyIndex}]`,
            ),
        ),
      };
    });
    const policySignature = (
      policy: "conservative" | "balanced" | "aggressive",
    ) =>
      suggestions
        .filter((suggestion) => suggestion.eligiblePolicyIds.includes(policy))
        .map((suggestion) => suggestion.id)
        .sort()
        .join("\u0000");
    return {
      modelId: string(
        value.modelId,
        "bundle.initialInference.suppression.modelId",
      ),
      artifactSha256: string(
        value.artifactSha256,
        "bundle.initialInference.suppression.artifactSha256",
      ),
      weightsSha256: string(
        value.weightsSha256,
        "bundle.initialInference.suppression.weightsSha256",
      ),
      decoderVersion: string(
        value.decoderVersion,
        "bundle.initialInference.suppression.decoderVersion",
      ),
      policyContractVersion: integer(
        value.policyContractVersion,
        "bundle.initialInference.suppression.policyContractVersion",
      ),
      probabilities,
      decodedIntervals: ranges(
        value.decodedIntervals,
        "bundle.initialInference.suppression.decodedIntervals",
        duration,
      ),
      suggestions,
      identicalPolicyResults:
        value.identicalPolicyResults === undefined
          ? policySignature("conservative") === policySignature("balanced") &&
            policySignature("balanced") === policySignature("aggressive")
          : boolean(
              value.identicalPolicyResults,
              "bundle.initialInference.suppression.identicalPolicyResults",
            ),
    };
  })();

  const corrections = object(root.corrections, "bundle.corrections");
  const correctedValue = corrections.correctedRanges;
  if (!Array.isArray(correctedValue))
    fail("bundle.corrections.correctedRanges", "must be an array");
  const ignoredValue = corrections.ignoredIntervals;
  if (!Array.isArray(ignoredValue))
    fail("bundle.corrections.ignoredIntervals", "must be an array");
  const labels = object(corrections.labels, "bundle.corrections.labels");
  const suppressionCorrections = (() => {
    if (corrections.suppression === undefined) return null;
    const value = object(
      corrections.suppression,
      "bundle.corrections.suppression",
    );
    const decisionOverridesValue = object(
      value.decisionOverrides ?? {},
      "bundle.corrections.suppression.decisionOverrides",
    );
    const decisionOverrides = Object.fromEntries(
      Object.entries(decisionOverridesValue).map(([id, decision]) => [
        id,
        choice(
          decision,
          ["keep", "suppress"] as const,
          `bundle.corrections.suppression.decisionOverrides.${id}`,
        ),
      ]),
    );
    const scopeOverridesValue = object(
      value.suppressionScopeOverrides ?? {},
      "bundle.corrections.suppression.suppressionScopeOverrides",
    );
    const suppressionScopeOverrides = Object.fromEntries(
      Object.entries(scopeOverridesValue).map(([id, scope]) => [
        id,
        choice(
          scope,
          ["whole-rally", "veto-region"] as const,
          `bundle.corrections.suppression.suppressionScopeOverrides.${id}`,
        ),
      ]),
    );
    const decisionsValue = value.decisions ?? [];
    if (!Array.isArray(decisionsValue)) {
      fail("bundle.corrections.suppression.decisions", "must be an array");
    }
    return {
      selectedPolicy: choice(
        value.selectedPolicy,
        ["none", "conservative", "balanced", "aggressive"] as const,
        "bundle.corrections.suppression.selectedPolicy",
      ),
      decisionOverrides,
      defaultSuppressionScope:
        value.defaultSuppressionScope === undefined
          ? ("whole-rally" as const)
          : choice(
              value.defaultSuppressionScope,
              ["whole-rally", "veto-region"] as const,
              "bundle.corrections.suppression.defaultSuppressionScope",
            ),
      suppressionScopeOverrides,
      userTouchedCutIds: stringArray(
        value.userTouchedCutIds ?? [],
        "bundle.corrections.suppression.userTouchedCutIds",
      ),
      decisions: decisionsValue.map((item, index) => {
        const field = `bundle.corrections.suppression.decisions[${index}]`;
        const decision = object(item, field);
        return {
          suggestionId: string(decision.suggestionId, `${field}.suggestionId`),
          logicalId: string(decision.logicalId, `${field}.logicalId`),
          state: choice(
            decision.state,
            ["dormant", "suppressed", "kept", "edited-kept"] as const,
            `${field}.state`,
          ),
          scope: choice(
            decision.scope,
            ["whole-rally", "veto-region"] as const,
            `${field}.scope`,
          ),
        };
      }),
    };
  })();
  const scoreTracking = (() => {
    if (corrections.scoreTracking === undefined) {
      if (schemaVersion >= 3) {
        fail("bundle.corrections.scoreTracking", "is required");
      }
      return null;
    }
    return parseScoreTrackingFeedback(
      corrections.scoreTracking,
      "bundle.corrections.scoreTracking",
      duration,
    );
  })();
  const finalValue = root.finalExportIntervals;
  if (!Array.isArray(finalValue))
    fail("bundle.finalExportIntervals", "must be an array");

  return {
    schema: MODEL_FEEDBACK_SCHEMA,
    schemaVersion,
    generatedAt: date(root.generatedAt, "bundle.generatedAt"),
    producer: sourceProducer(runtimeVariant),
    source: {
      projectId: string(source.projectId, "bundle.source.projectId"),
      analysisId: string(source.analysisId, "bundle.source.analysisId"),
      timelineCoordinates: "seconds-from-start-of-source",
      file: {
        name: string(file.name, "bundle.source.file.name"),
        sizeBytes: integer(file.sizeBytes, "bundle.source.file.sizeBytes"),
        lastModifiedMs: nonNegative(
          file.lastModifiedMs,
          "bundle.source.file.lastModifiedMs",
        ),
        mimeType: text(file.mimeType, "bundle.source.file.mimeType"),
        sampledFingerprint: (() => {
          if (file.sampledFingerprint === null) return null;
          const fingerprint = string(
            file.sampledFingerprint,
            "bundle.source.file.sampledFingerprint",
          );
          if (!/^sampled-sha256-v1:[0-9a-f]{64}$/.test(fingerprint)) {
            fail(
              "bundle.source.file.sampledFingerprint",
              "must use sampled-sha256-v1",
            );
          }
          return fingerprint;
        })(),
      },
      media: {
        duration,
        mimeType: string(media.mimeType, "bundle.source.media.mimeType"),
        width: integer(media.width, "bundle.source.media.width"),
        height: integer(media.height, "bundle.source.media.height"),
        rotation: number(media.rotation, "bundle.source.media.rotation"),
        videoCodec: string(media.videoCodec, "bundle.source.media.videoCodec"),
        videoCodecString: nullableString(
          media.videoCodecString,
          "bundle.source.media.videoCodecString",
        ),
        canDecodeVideo: boolean(
          media.canDecodeVideo,
          "bundle.source.media.canDecodeVideo",
        ),
        hasAudio: boolean(media.hasAudio, "bundle.source.media.hasAudio"),
        audioCodec: nullableString(
          media.audioCodec,
          "bundle.source.media.audioCodec",
        ),
        sampleRate: optionalNumber(
          media.sampleRate,
          "bundle.source.media.sampleRate",
        ),
        channels: optionalNumber(
          media.channels,
          "bundle.source.media.channels",
        ),
        canDecodeAudio: boolean(
          media.canDecodeAudio,
          "bundle.source.media.canDecodeAudio",
        ),
      },
      gameWindow: { start: windowStart, end: windowEnd },
      featureRoi: roi,
      runtimeVariant,
    },
    features,
    initialInference: {
      modelId: string(inference.modelId, "bundle.initialInference.modelId"),
      components,
      ensembleAlgorithmVersion: string(
        inference.ensembleAlgorithmVersion,
        "bundle.initialInference.ensembleAlgorithmVersion",
      ),
      ranges: ranges(
        inference.ranges,
        "bundle.initialInference.ranges",
        duration,
      ),
      probabilityModelId: string(
        inference.probabilityModelId,
        "bundle.initialInference.probabilityModelId",
      ),
      timestamps: inferenceTimestamps,
      rallyProbabilities,
      serveProbabilities,
      deadStateProbabilities,
      productionComponents,
      componentServeOutputs,
      servingSide,
      suppression,
    },
    corrections: {
      updatedAt: date(corrections.updatedAt, "bundle.corrections.updatedAt"),
      beforePaddingSeconds: nonNegative(
        corrections.beforePaddingSeconds,
        "bundle.corrections.beforePaddingSeconds",
      ),
      afterPaddingSeconds: nonNegative(
        corrections.afterPaddingSeconds,
        "bundle.corrections.afterPaddingSeconds",
      ),
      joinGapSeconds: nonNegative(
        corrections.joinGapSeconds,
        "bundle.corrections.joinGapSeconds",
      ),
      correctedRanges: correctedValue.map((item, index) =>
        correctedRange(
          item,
          `bundle.corrections.correctedRanges[${index}]`,
          duration,
        ),
      ),
      ignoredIntervals: ignoredValue.map((item, index) => {
        const ignored = object(
          item,
          `bundle.corrections.ignoredIntervals[${index}]`,
        );
        const start = number(
          ignored.start,
          `bundle.corrections.ignoredIntervals[${index}].start`,
        );
        const end = number(
          ignored.end,
          `bundle.corrections.ignoredIntervals[${index}].end`,
        );
        checkBounds(
          start,
          end,
          duration,
          `bundle.corrections.ignoredIntervals[${index}]`,
        );
        return {
          id: string(
            ignored.id,
            `bundle.corrections.ignoredIntervals[${index}].id`,
          ),
          start,
          end,
          reason: string(
            ignored.reason,
            `bundle.corrections.ignoredIntervals[${index}].reason`,
          ),
        };
      }),
      suppression: suppressionCorrections,
      scoreTracking,
      labels: {
        falsePositives: ranges(
          labels.falsePositives,
          "bundle.corrections.labels.falsePositives",
          duration,
        ),
        falseNegatives: ranges(
          labels.falseNegatives,
          "bundle.corrections.labels.falseNegatives",
          duration,
        ),
        confirmedModelRanges: ranges(
          labels.confirmedModelRanges,
          "bundle.corrections.labels.confirmedModelRanges",
          duration,
        ),
        discardedManualRanges: ranges(
          labels.discardedManualRanges,
          "bundle.corrections.labels.discardedManualRanges",
          duration,
        ),
      },
    },
    finalExportIntervals: finalValue.map((item, index) => {
      const interval = object(item, `bundle.finalExportIntervals[${index}]`);
      const start = number(
        interval.start,
        `bundle.finalExportIntervals[${index}].start`,
      );
      const end = number(
        interval.end,
        `bundle.finalExportIntervals[${index}].end`,
      );
      checkBounds(
        start,
        end,
        duration,
        `bundle.finalExportIntervals[${index}]`,
      );
      const gapValue = interval.joinedGaps ?? [];
      if (!Array.isArray(gapValue)) {
        fail(
          `bundle.finalExportIntervals[${index}].joinedGaps`,
          "must be an array",
        );
      }
      return {
        start,
        end,
        cutIds: stringArray(
          interval.cutIds,
          `bundle.finalExportIntervals[${index}].cutIds`,
        ),
        joinedGaps: gapValue.map((gap, gapIndex) => {
          const joined = object(
            gap,
            `bundle.finalExportIntervals[${index}].joinedGaps[${gapIndex}]`,
          );
          const gapStart = number(
            joined.start,
            `bundle.finalExportIntervals[${index}].joinedGaps[${gapIndex}].start`,
          );
          const gapEnd = number(
            joined.end,
            `bundle.finalExportIntervals[${index}].joinedGaps[${gapIndex}].end`,
          );
          if (gapStart < start || gapEnd > end || gapEnd <= gapStart) {
            fail(
              `bundle.finalExportIntervals[${index}].joinedGaps[${gapIndex}]`,
              "must be inside its export interval",
            );
          }
          return { start: gapStart, end: gapEnd };
        }),
      };
    }),
    warnings: stringArray(root.warnings, "bundle.warnings"),
  };
}

export function parseModelFeedbackText(text: string): ParsedModelFeedback {
  let value: unknown;
  try {
    value = JSON.parse(text) as unknown;
  } catch {
    throw new ModelFeedbackValidationError(
      "The selected file is not valid JSON",
    );
  }
  return parseModelFeedback(value);
}
