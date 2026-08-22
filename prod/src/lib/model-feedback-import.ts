import {
  ModelFeedbackValidationError,
  type ParsedModelFeedback,
  parseModelFeedbackText,
} from "../../../lib/model-feedback.ts";

import {
  createCutDraft,
  parseCutDraft,
  type CutDraft,
  type CutDraftSeed,
} from "./cut-draft.ts";
import {
  SERVING_SIDE_ANCHOR_CONTRACT,
  SERVING_SIDE_FEATURE_COLUMNS,
  SERVING_SIDE_FEATURE_VERSION,
} from "./on-device/serving-side-model.ts";
import { SUPPRESSION_POLICY_CONTRACT_VERSION } from "./on-device/suppression-policy.ts";
import type {
  OnDeviceAnalysis,
  OnDeviceInterval,
  OnDeviceMediaInfo,
  OnDeviceServingSideOutput,
} from "./on-device/types.ts";
import {
  projectAnalysisId,
  type VolleyCutProject,
} from "./project-store.ts";
import { migrateScoreTracking } from "./score-tracking.ts";

export type ImportedModelFeedbackProject = {
  project: VolleyCutProject;
  warnings: string[];
};

type ImportOptions = {
  occupiedProjectIds?: ReadonlySet<string>;
  importedAt?: string;
};

function hashText(value: string): string {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(36);
}

function nextProjectId(
  feedback: ParsedModelFeedback,
  occupiedProjectIds: ReadonlySet<string>,
): string {
  const base = `feedback-${hashText(
    `${feedback.source.projectId}\u0000${feedback.source.analysisId}\u0000${feedback.generatedAt}`,
  )}`;
  if (!occupiedProjectIds.has(base)) return base;
  for (let suffix = 2; suffix < 10_000; suffix += 1) {
    const candidate = `${base}-${suffix}`;
    if (!occupiedProjectIds.has(candidate)) return candidate;
  }
  return `${base}-${Date.now()}`;
}

function agreement(
  value: string | undefined,
  field: string,
): OnDeviceInterval["agreement"] {
  if (value === undefined) return undefined;
  if (
    value === "both-models" ||
    value === "all-labels-v2-only" ||
    value === "previous-production-only"
  ) {
    return value;
  }
  throw new ModelFeedbackValidationError(
    `${field} uses an unsupported production-model agreement`,
  );
}

function interval(
  value: ParsedModelFeedback["initialInference"]["ranges"][number],
  field: string,
): OnDeviceInterval {
  return {
    id: value.id,
    start: value.start,
    end: value.end,
    confidence: value.confidence,
    included: value.included ?? true,
    ...(agreement(value.agreement, field)
      ? { agreement: agreement(value.agreement, field) }
      : {}),
  };
}

function mediaInfo(feedback: ParsedModelFeedback): OnDeviceMediaInfo {
  const media = feedback.source.media;
  if (
    media.rotation !== 0 &&
    media.rotation !== 90 &&
    media.rotation !== 180 &&
    media.rotation !== 270
  ) {
    throw new ModelFeedbackValidationError(
      "bundle.source.media.rotation must be 0, 90, 180, or 270",
    );
  }
  return {
    ...media,
    rotation: media.rotation,
  };
}

function servingSideOutput(
  feedback: ParsedModelFeedback,
): OnDeviceServingSideOutput | undefined {
  const output = feedback.initialInference.servingSide;
  if (!output) return undefined;
  if (
    output.featureVersion !== SERVING_SIDE_FEATURE_VERSION ||
    output.anchorContract !== SERVING_SIDE_ANCHOR_CONTRACT ||
    output.features.columns !== SERVING_SIDE_FEATURE_COLUMNS
  ) {
    throw new ModelFeedbackValidationError(
      "The serving-side feature contract is not supported by this VolleyCut version",
    );
  }
  return {
    ...output,
    featureVersion: SERVING_SIDE_FEATURE_VERSION,
    anchorContract: SERVING_SIDE_ANCHOR_CONTRACT,
    features: {
      ...output.features,
      values: new Float64Array(output.features.values),
    },
    candidates: output.candidates.map((candidate) => {
      const { agreement: rawAgreement, ...candidateInterval } =
        candidate.interval;
      const normalizedAgreement = agreement(
        rawAgreement,
        `bundle.initialInference.servingSide.candidates.${candidate.id}.interval.agreement`,
      );
      return {
        ...candidate,
        interval: {
          ...candidateInterval,
          ...(normalizedAgreement
            ? { agreement: normalizedAgreement }
            : {}),
        },
        reviewReasons: [...candidate.reviewReasons],
        serveEvidence: {
          allLabelsV2: {
            ...candidate.serveEvidence.allLabelsV2,
            nearestDetection: candidate.serveEvidence.allLabelsV2
              .nearestDetection
              ? { ...candidate.serveEvidence.allLabelsV2.nearestDetection }
              : null,
          },
          previousProduction: {
            ...candidate.serveEvidence.previousProduction,
            nearestDetection: candidate.serveEvidence.previousProduction
              .nearestDetection
              ? {
                  ...candidate.serveEvidence.previousProduction
                    .nearestDetection,
                }
              : null,
          },
        },
      };
    }),
  };
}

function analysisFromFeedback(
  feedback: ParsedModelFeedback,
): OnDeviceAnalysis {
  if (feedback.features) {
    if (
      feedback.features.rows !== feedback.initialInference.timestamps.length ||
      feedback.features.timestamps.some(
        (timestamp, index) =>
          timestamp !== feedback.initialInference.timestamps[index],
      )
    ) {
      throw new ModelFeedbackValidationError(
        "Feature timestamps must align with the imported inference timeline",
      );
    }
  }
  const productionComponents = feedback.initialInference.productionComponents;
  const componentServeOutputs = feedback.initialInference.componentServeOutputs;
  const suppression = feedback.initialInference.suppression;
  return {
    modelId: feedback.initialInference.modelId,
    featurePath: "local-source",
    intervals: feedback.initialInference.ranges.map((range, index) =>
      interval(range, `bundle.initialInference.ranges[${index}].agreement`),
    ),
    times: new Float64Array(feedback.initialInference.timestamps),
    ...(feedback.features
      ? {
          featureNames: [...feedback.features.names],
          featureValues: new Float32Array(feedback.features.values),
        }
      : {}),
    rallyProbabilities: new Float32Array(
      feedback.initialInference.rallyProbabilities,
    ),
    serveProbabilities: new Float32Array(
      feedback.initialInference.serveProbabilities,
    ),
    deadStateProbabilities: new Float32Array(
      feedback.initialInference.deadStateProbabilities,
    ),
    ...(productionComponents
      ? {
          productionComponents: {
            allLabelsV2: productionComponents.allLabelsV2.map((range, index) =>
              interval(
                range,
                `bundle.initialInference.productionComponents.allLabelsV2[${index}].agreement`,
              ),
            ),
            previousProduction: productionComponents.previousProduction.map(
              (range, index) =>
                interval(
                  range,
                  `bundle.initialInference.productionComponents.previousProduction[${index}].agreement`,
                ),
            ),
          },
        }
      : {}),
    ...(componentServeOutputs
      ? {
          productionServeOutputs: {
            allLabelsV2: {
              probabilities: new Float32Array(
                componentServeOutputs.allLabelsV2.probabilities,
              ),
              detections: componentServeOutputs.allLabelsV2.detections.map(
                (detection) => ({ ...detection }),
              ),
            },
            previousProduction: {
              probabilities: new Float32Array(
                componentServeOutputs.previousProduction.probabilities,
              ),
              detections:
                componentServeOutputs.previousProduction.detections.map(
                  (detection) => ({ ...detection }),
                ),
            },
          },
        }
      : {}),
    ...(feedback.initialInference.servingSide
      ? { servingSide: servingSideOutput(feedback) }
      : {}),
    ...(suppression
      ? {
          suppression: {
            modelId: suppression.modelId,
            artifactSha256: suppression.artifactSha256,
            weightsSha256: suppression.weightsSha256,
            decoderVersion: suppression.decoderVersion,
            policyContractVersion: suppression.policyContractVersion,
            probabilities: new Float32Array(suppression.probabilities),
            decodedIntervals: suppression.decodedIntervals.map((range, index) =>
              interval(
                range,
                `bundle.initialInference.suppression.decodedIntervals[${index}].agreement`,
              ),
            ),
            suggestions: suppression.suggestions.map((suggestion) => ({
              ...suggestion,
              sourceProductionIds: [...suggestion.sourceProductionIds],
              eligiblePolicyIds: [...suggestion.eligiblePolicyIds],
            })),
            identicalPolicyResults: suppression.identicalPolicyResults,
          },
        }
      : {}),
  };
}

function sourceIgnoredIntervals(feedback: ParsedModelFeedback) {
  const intervals: Array<{ start: number; end: number; reason: string }> = [];
  if (feedback.source.gameWindow.start > 0) {
    intervals.push({
      start: 0,
      end: feedback.source.gameWindow.start,
      reason: "outside-game-window",
    });
  }
  if (feedback.source.gameWindow.end < feedback.source.media.duration) {
    intervals.push({
      start: feedback.source.gameWindow.end,
      end: feedback.source.media.duration,
      reason: "outside-game-window",
    });
  }
  return intervals;
}

function draftFromFeedback(
  feedback: ParsedModelFeedback,
  project: VolleyCutProject,
): CutDraft {
  const analysisId = projectAnalysisId(project);
  if (!analysisId || !project.analysis) {
    throw new ModelFeedbackValidationError(
      "The imported feedback could not create an editable analysis",
    );
  }
  const seed: CutDraftSeed = {
    analysisId,
    recordingId: project.id,
    duration: project.info.duration,
    analysisStart: project.analysisWindow.start,
    analysisEnd: project.analysisWindow.end,
    rallies: project.analysis.intervals,
    ignoredIntervals: sourceIgnoredIntervals(feedback),
    suppressionContractVersion:
      feedback.initialInference.suppression?.policyContractVersion ??
      SUPPRESSION_POLICY_CONTRACT_VERSION,
  };
  const base = createCutDraft(seed);
  const scoreTracking = feedback.corrections.scoreTracking
    ? migrateScoreTracking(
        feedback.corrections.scoreTracking.state,
        project.info.duration,
      )
    : base.scoreTracking;
  if (!scoreTracking) {
    throw new ModelFeedbackValidationError(
      "bundle.corrections.scoreTracking.state is not supported",
    );
  }
  const suppression = feedback.corrections.suppression;
  const candidate: CutDraft = {
    ...base,
    updatedAt: feedback.corrections.updatedAt,
    beforePaddingSeconds: feedback.corrections.beforePaddingSeconds,
    afterPaddingSeconds: feedback.corrections.afterPaddingSeconds,
    joinGapSeconds: feedback.corrections.joinGapSeconds,
    selectedSuppressionPolicy: suppression?.selectedPolicy ?? "none",
    suppressionDecisionOverrides: suppression?.decisionOverrides ?? {},
    suppressionScopeOverrides: suppression?.suppressionScopeOverrides ?? {},
    userTouchedCutIds: suppression?.userTouchedCutIds ?? [],
    scoreTracking,
    cuts: feedback.corrections.correctedRanges.map((cut, index) => {
      const { agreement: rawAgreement, ...cutWithoutAgreement } = cut;
      const normalizedAgreement = agreement(
        rawAgreement,
        `bundle.corrections.correctedRanges[${index}].agreement`,
      );
      return {
        ...cutWithoutAgreement,
        ...(normalizedAgreement ? { agreement: normalizedAgreement } : {}),
      };
    }),
    ignoredIntervals: feedback.corrections.ignoredIntervals.map((range) => ({
      ...range,
    })),
  };
  const validated = parseCutDraft(JSON.stringify(candidate), seed);
  if (!validated) {
    throw new ModelFeedbackValidationError(
      "The feedback corrections are not compatible with this editor version",
    );
  }
  return validated;
}

export function importModelFeedbackProject(
  text: string,
  options: ImportOptions = {},
): ImportedModelFeedbackProject {
  const feedback = parseModelFeedbackText(text);
  const importedAt = options.importedAt ?? new Date().toISOString();
  if (!Number.isFinite(Date.parse(importedAt))) {
    throw new Error("importedAt must be an ISO date");
  }
  const id = nextProjectId(
    feedback,
    options.occupiedProjectIds ?? new Set<string>(),
  );
  const analysis = analysisFromFeedback(feedback);
  const source = {
    name: feedback.source.file.name,
    size: feedback.source.file.sizeBytes,
    lastModified: feedback.source.file.lastModifiedMs,
    type: feedback.source.file.mimeType,
    ...(feedback.source.file.sampledFingerprint
      ? { fingerprint: feedback.source.file.sampledFingerprint }
      : {}),
  };
  const projectWithoutFeedback: VolleyCutProject = {
    schemaVersion: 1,
    id,
    source,
    info: mediaInfo(feedback),
    analysisWindow: { ...feedback.source.gameWindow },
    roi: { ...feedback.source.featureRoi },
    status: "ready",
    analysis,
    error: null,
    createdAt: importedAt,
    updatedAt: importedAt,
  };
  const initialDraft = draftFromFeedback(feedback, projectWithoutFeedback);
  const project: VolleyCutProject = {
    ...projectWithoutFeedback,
    importedFeedback: {
      schemaVersion: feedback.schemaVersion,
      generatedAt: feedback.generatedAt,
      importedAt,
      originalProjectId: feedback.source.projectId,
      originalAnalysisId: feedback.source.analysisId,
      runtimeVariant: feedback.source.runtimeVariant,
      warnings: [...feedback.warnings],
      initialDraft,
    },
  };
  return { project, warnings: [...feedback.warnings] };
}
