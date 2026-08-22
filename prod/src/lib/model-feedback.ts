import {
  type CutDraft,
  DEFAULT_SUPPRESSION_SCOPE,
  type EditableCut,
  type FinalCutInterval,
  materializeFinalCutIntervals,
  suppressionSuggestionScope,
  suppressionSuggestionState,
} from "./cut-draft.ts";
import {
  ALL_LABELS_V2_BUNDLE_SHA256,
  ALL_LABELS_V2_MODEL_ID,
  PREVIOUS_PRODUCTION_BUNDLE_SHA256,
  PREVIOUS_PRODUCTION_MODEL_ID,
  PRODUCTION_ENSEMBLE_ALGORITHM_VERSION,
} from "./on-device/ensemble.ts";
import { ANALYSIS_FPS } from "./on-device/feature-schema.ts";
import type { ProductAnalysis } from "./product-analysis";
import {
  deriveScoreAt,
  scoreTrackingOutsideExcludedRallies,
  scoreTrackingOutsideIgnoredIntervals,
  type DerivedScore,
} from "./score-tracking.ts";

export const MODEL_FEEDBACK_SCHEMA = "volleycut-model-feedback" as const;
export const MODEL_FEEDBACK_SCHEMA_VERSION = 3 as const;

export type EncodedNumericArray = {
  encoding: "base64";
  byteOrder: "little-endian";
  dataType: "float32" | "float64";
  shape: number[];
  data: string;
};

type EncodedServeOutput = {
  modelId: string;
  probabilities: EncodedNumericArray;
  detections: Array<{ time: number; confidence: number }>;
};

type FeedbackRange = {
  id: string;
  start: number;
  end: number;
  confidence: number;
  agreement?: EditableCut["agreement"];
};

export type ModelFeedbackBundle = {
  schema: typeof MODEL_FEEDBACK_SCHEMA;
  schemaVersion: typeof MODEL_FEEDBACK_SCHEMA_VERSION;
  generatedAt: string;
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
    media: ProductAnalysis["mediaInfo"];
    gameWindow: ProductAnalysis["analysisWindow"];
    featureRoi: ProductAnalysis["roi"];
    runtimeVariant: ProductAnalysis["runtimeVariant"];
    videoBytesIncluded: false;
  };
  features: {
    analysisFps: number;
    rows: number;
    columns: number;
    names: string[];
    timestamps: EncodedNumericArray;
    values: EncodedNumericArray;
  } | null;
  initialInference: {
    modelId: string;
    components: Array<{ modelId: string; bundleSha256: string }>;
    ensembleAlgorithmVersion: string;
    ranges: ProductAnalysis["rallies"];
    probabilityModelId: string;
    timestamps: EncodedNumericArray;
    probabilities: {
      rally: EncodedNumericArray;
      serve: EncodedNumericArray;
      deadState: EncodedNumericArray;
    };
    componentServeOutputs?: {
      allLabelsV2: EncodedServeOutput;
      previousProduction: EncodedServeOutput;
    };
    servingSide: null | {
      modelId: string;
      modelFingerprint: string;
      featureVersion: string;
      anchorContract: string;
      features: {
        rows: number;
        columns: number;
        values: EncodedNumericArray;
      };
      candidates: NonNullable<ProductAnalysis["servingSide"]>["candidates"];
    };
    productionComponents: ProductAnalysis["productionComponents"];
    suppression: null | {
      modelId: string;
      artifactSha256: string;
      weightsSha256: string;
      decoderVersion: string;
      policyContractVersion: number;
      timestamps: EncodedNumericArray;
      probabilities: EncodedNumericArray;
      decodedIntervals: NonNullable<
        ProductAnalysis["suppression"]
      >["decodedIntervals"];
      suggestions: NonNullable<ProductAnalysis["suppression"]>["suggestions"];
    };
  };
  corrections: {
    updatedAt: string;
    beforePaddingSeconds: number;
    afterPaddingSeconds: number;
    joinGapSeconds: number;
    correctedRanges: EditableCut[];
    ignoredIntervals: CutDraft["ignoredIntervals"];
    suppression: {
      selectedPolicy: CutDraft["selectedSuppressionPolicy"];
      decisionOverrides: CutDraft["suppressionDecisionOverrides"];
      defaultSuppressionScope: typeof DEFAULT_SUPPRESSION_SCOPE;
      suppressionScopeOverrides: CutDraft["suppressionScopeOverrides"];
      userTouchedCutIds: string[];
      decisions: Array<{
        suggestionId: string;
        logicalId: string;
        state: ReturnType<typeof suppressionSuggestionState>;
        scope: ReturnType<typeof suppressionSuggestionScope>;
      }>;
    };
    scoreTracking: {
      state: CutDraft["scoreTracking"];
      excludedRallyIds: string[];
      derivedFinalScore: DerivedScore;
    };
    labels: {
      falsePositives: FeedbackRange[];
      falseNegatives: FeedbackRange[];
      confirmedModelRanges: FeedbackRange[];
      discardedManualRanges: FeedbackRange[];
    };
  };
  finalExportIntervals: FinalCutInterval[];
  finalExportProvenance: ReturnType<
    typeof materializeFinalCutIntervals
  >["provenance"];
  warnings: string[];
};

function nativeIsLittleEndian(): boolean {
  const bytes = new Uint8Array(new Uint16Array([0x0102]).buffer);
  return bytes[0] === 0x02;
}

function littleEndianBytes(values: Float32Array | Float64Array): Uint8Array {
  const source = new Uint8Array(
    values.buffer,
    values.byteOffset,
    values.byteLength,
  );
  if (nativeIsLittleEndian()) return source;
  const output = new Uint8Array(source.length);
  const elementBytes = values.BYTES_PER_ELEMENT;
  for (let offset = 0; offset < source.length; offset += elementBytes) {
    for (let byte = 0; byte < elementBytes; byte += 1) {
      output[offset + byte] = source[offset + elementBytes - byte - 1];
    }
  }
  return output;
}

function base64(bytes: Uint8Array): string {
  const chunks: string[] = [];
  const chunkBytes = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkBytes) {
    chunks.push(
      String.fromCharCode(...bytes.subarray(offset, offset + chunkBytes)),
    );
  }
  return btoa(chunks.join(""));
}

export function encodeNumericArray(
  values: Float32Array | Float64Array,
  shape: number[],
): EncodedNumericArray {
  const expectedValues = shape.reduce(
    (product, dimension) => product * dimension,
    1,
  );
  if (
    shape.length === 0 ||
    shape.some((dimension) => !Number.isInteger(dimension) || dimension < 0) ||
    expectedValues !== values.length
  ) {
    throw new Error("Numeric-array shape does not match its data.");
  }
  return {
    encoding: "base64",
    byteOrder: "little-endian",
    dataType: values instanceof Float64Array ? "float64" : "float32",
    shape: [...shape],
    data: base64(littleEndianBytes(values)),
  };
}

function feedbackRange(cut: EditableCut): FeedbackRange {
  return {
    id: cut.id,
    start: cut.coreStart,
    end: cut.coreEnd,
    confidence: cut.confidence,
    ...(cut.agreement ? { agreement: cut.agreement } : {}),
  };
}

export function createModelFeedbackBundle(
  analysis: ProductAnalysis,
  draft: CutDraft,
  finalExportIntervals: FinalCutInterval[],
  generatedAt = new Date().toISOString(),
): ModelFeedbackBundle {
  const modelCuts = draft.cuts.filter((cut) => cut.origin === "cached-label");
  const manualCuts = draft.cuts.filter((cut) => cut.origin === "manual");
  const warnings: string[] = [];
  if (!analysis.features) {
    warnings.push(
      "Base features are unavailable because this analysis predates model-feedback capture; inference and corrections are still included.",
    );
  }
  if (!analysis.productionServeOutputs) {
    warnings.push(
      "Per-component serve-head outputs are unavailable because this analysis predates their retention; the V2 probability trace remains included.",
    );
  }
  if (!analysis.servingSide) {
    warnings.push(
      "Serving-side features and initial verdicts are unavailable because this analysis predates their retention; score-marker corrections are still included.",
    );
  }
  const materialized = materializeFinalCutIntervals(
    draft,
    analysis.suppression,
  );
  const allSuppressionSuggestions = analysis.suppression?.suggestions ?? [];
  const effectiveCutIds = new Set(
    materialized.intervals.flatMap((interval) => interval.cutIds),
  );
  const excludedRallyIds = draft.cuts
    .filter((cut) => !effectiveCutIds.has(cut.id))
    .map((cut) => cut.id);
  const scoringTracking = scoreTrackingOutsideExcludedRallies(
    scoreTrackingOutsideIgnoredIntervals(
      draft.scoreTracking,
      draft.ignoredIntervals,
    ),
    new Set(excludedRallyIds),
  );

  return {
    schema: MODEL_FEEDBACK_SCHEMA,
    schemaVersion: MODEL_FEEDBACK_SCHEMA_VERSION,
    generatedAt,
    source: {
      projectId: analysis.recordingId,
      analysisId: analysis.id,
      timelineCoordinates: "seconds-from-start-of-source",
      file: {
        name: analysis.source.name,
        sizeBytes: analysis.source.size,
        lastModifiedMs: analysis.source.lastModified,
        mimeType: analysis.source.type,
        sampledFingerprint: analysis.source.fingerprint ?? null,
      },
      media: analysis.mediaInfo,
      gameWindow: analysis.analysisWindow,
      featureRoi: analysis.roi,
      runtimeVariant: analysis.runtimeVariant,
      videoBytesIncluded: false,
    },
    features: analysis.features
      ? {
          analysisFps: ANALYSIS_FPS,
          rows: analysis.features.rows,
          columns: analysis.features.columns,
          names: [...analysis.features.names],
          timestamps: encodeNumericArray(analysis.features.times, [
            analysis.features.rows,
          ]),
          values: encodeNumericArray(analysis.features.values, [
            analysis.features.rows,
            analysis.features.columns,
          ]),
        }
      : null,
    initialInference: {
      modelId: analysis.modelId,
      components: [
        {
          modelId: ALL_LABELS_V2_MODEL_ID,
          bundleSha256: ALL_LABELS_V2_BUNDLE_SHA256,
        },
        {
          modelId: PREVIOUS_PRODUCTION_MODEL_ID,
          bundleSha256: PREVIOUS_PRODUCTION_BUNDLE_SHA256,
        },
      ],
      ensembleAlgorithmVersion: PRODUCTION_ENSEMBLE_ALGORITHM_VERSION,
      ranges: analysis.rallies.map((range) => ({ ...range })),
      probabilityModelId: ALL_LABELS_V2_MODEL_ID,
      timestamps: encodeNumericArray(analysis.inferenceTimes, [
        analysis.inferenceTimes.length,
      ]),
      probabilities: {
        rally: encodeNumericArray(analysis.probabilities.rally, [
          analysis.probabilities.rally.length,
        ]),
        serve: encodeNumericArray(analysis.probabilities.serve, [
          analysis.probabilities.serve.length,
        ]),
        deadState: encodeNumericArray(analysis.probabilities.deadState, [
          analysis.probabilities.deadState.length,
        ]),
      },
      componentServeOutputs: analysis.productionServeOutputs
        ? {
            allLabelsV2: {
              modelId: ALL_LABELS_V2_MODEL_ID,
              probabilities: encodeNumericArray(
                analysis.productionServeOutputs.allLabelsV2.probabilities,
                [
                  analysis.productionServeOutputs.allLabelsV2.probabilities
                    .length,
                ],
              ),
              detections:
                analysis.productionServeOutputs.allLabelsV2.detections.map(
                  (detection) => ({ ...detection }),
                ),
            },
            previousProduction: {
              modelId: PREVIOUS_PRODUCTION_MODEL_ID,
              probabilities: encodeNumericArray(
                analysis.productionServeOutputs.previousProduction
                  .probabilities,
                [
                  analysis.productionServeOutputs.previousProduction
                    .probabilities.length,
                ],
              ),
              detections:
                analysis.productionServeOutputs.previousProduction.detections.map(
                  (detection) => ({ ...detection }),
                ),
            },
          }
        : undefined,
      servingSide: analysis.servingSide
        ? {
            modelId: analysis.servingSide.modelId,
            modelFingerprint: analysis.servingSide.modelFingerprint,
            featureVersion: analysis.servingSide.featureVersion,
            anchorContract: analysis.servingSide.anchorContract,
            features: {
              rows: analysis.servingSide.features.rows,
              columns: analysis.servingSide.features.columns,
              values: encodeNumericArray(
                analysis.servingSide.features.values,
                [
                  analysis.servingSide.features.rows,
                  analysis.servingSide.features.columns,
                ],
              ),
            },
            candidates: analysis.servingSide.candidates.map((candidate) => ({
              ...candidate,
              interval: { ...candidate.interval },
              reviewReasons: [...candidate.reviewReasons],
              serveEvidence: {
                allLabelsV2: {
                  ...candidate.serveEvidence.allLabelsV2,
                  nearestDetection: candidate.serveEvidence.allLabelsV2.nearestDetection
                    ? { ...candidate.serveEvidence.allLabelsV2.nearestDetection }
                    : null,
                },
                previousProduction: {
                  ...candidate.serveEvidence.previousProduction,
                  nearestDetection:
                    candidate.serveEvidence.previousProduction.nearestDetection
                      ? { ...candidate.serveEvidence.previousProduction.nearestDetection }
                      : null,
                },
              },
            })),
          }
        : null,
      productionComponents: analysis.productionComponents
        ? {
            allLabelsV2: analysis.productionComponents.allLabelsV2.map(
              (range) => ({ ...range }),
            ),
            previousProduction:
              analysis.productionComponents.previousProduction.map((range) => ({
                ...range,
              })),
          }
        : undefined,
      suppression: analysis.suppression
        ? {
            modelId: analysis.suppression.modelId,
            artifactSha256: analysis.suppression.artifactSha256,
            weightsSha256: analysis.suppression.weightsSha256,
            decoderVersion: analysis.suppression.decoderVersion,
            policyContractVersion: analysis.suppression.policyContractVersion,
            timestamps: encodeNumericArray(analysis.inferenceTimes, [
              analysis.inferenceTimes.length,
            ]),
            probabilities: encodeNumericArray(
              analysis.suppression.probabilities,
              [analysis.suppression.probabilities.length],
            ),
            decodedIntervals: analysis.suppression.decodedIntervals.map(
              (range) => ({ ...range }),
            ),
            suggestions: analysis.suppression.suggestions.map((suggestion) => ({
              ...suggestion,
              sourceProductionIds: [...suggestion.sourceProductionIds],
              eligiblePolicyIds: [...suggestion.eligiblePolicyIds],
            })),
          }
        : null,
    },
    corrections: {
      updatedAt: draft.updatedAt,
      beforePaddingSeconds: draft.beforePaddingSeconds,
      afterPaddingSeconds: draft.afterPaddingSeconds,
      joinGapSeconds: draft.joinGapSeconds,
      correctedRanges: draft.cuts.map((cut) => ({ ...cut })),
      ignoredIntervals: draft.ignoredIntervals.map((interval) => ({
        ...interval,
      })),
      suppression: {
        selectedPolicy: draft.selectedSuppressionPolicy,
        decisionOverrides: { ...draft.suppressionDecisionOverrides },
        defaultSuppressionScope: DEFAULT_SUPPRESSION_SCOPE,
        suppressionScopeOverrides: { ...draft.suppressionScopeOverrides },
        userTouchedCutIds: [...draft.userTouchedCutIds],
        decisions: allSuppressionSuggestions.map((suggestion) => ({
          suggestionId: suggestion.id,
          logicalId: suggestion.logicalId,
          state: suppressionSuggestionState(suggestion, draft),
          scope: suppressionSuggestionScope(suggestion, draft),
        })),
      },
      scoreTracking: {
        state: {
          ...draft.scoreTracking,
          serveMarkers: draft.scoreTracking.serveMarkers.map((marker) => ({
            ...marker,
          })),
          sideSwitchMarkers: draft.scoreTracking.sideSwitchMarkers.map(
            (marker) => ({ ...marker }),
          ),
          removedModelMarkerIds: [
            ...draft.scoreTracking.removedModelMarkerIds,
          ],
        },
        excludedRallyIds,
        derivedFinalScore: deriveScoreAt(scoringTracking),
      },
      labels: {
        falsePositives: modelCuts
          .filter((cut) => !cut.included)
          .map(feedbackRange),
        falseNegatives: manualCuts
          .filter((cut) => cut.included)
          .map(feedbackRange),
        confirmedModelRanges: modelCuts
          .filter((cut) => cut.included)
          .map(feedbackRange),
        discardedManualRanges: manualCuts
          .filter((cut) => !cut.included)
          .map(feedbackRange),
      },
    },
    finalExportIntervals: finalExportIntervals.map((interval) => ({
      ...interval,
      cutIds: [...interval.cutIds],
      ...(interval.joinedGaps
        ? { joinedGaps: interval.joinedGaps.map((gap) => ({ ...gap })) }
        : {}),
    })),
    finalExportProvenance: materialized.provenance.map((segment) => ({
      ...segment,
      cutIds: [...segment.cutIds],
    })),
    warnings,
  };
}

export function modelFeedbackFilename(sourceName: string): string {
  const safe = sourceName
    .replace(/\.[^.]+$/, "")
    .replace(/[^A-Za-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return `${safe || "volleycut"}.model-feedback.json`;
}

export function modelFeedbackBlob(bundle: ModelFeedbackBundle): Blob {
  return new Blob([`${JSON.stringify(bundle)}\n`], {
    type: "application/json",
  });
}
