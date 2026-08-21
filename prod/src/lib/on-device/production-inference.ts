import { runtimeAssetUrl } from "../runtime-assets.ts";
import type { AnalysisWindow } from "./analysis-window.ts";
import {
  ALL_LABELS_V2_MODEL_ID,
  mergeProductionModelIntervals,
  PREVIOUS_PRODUCTION_MODEL_ID,
  PRODUCTION_ENSEMBLE_MODEL_ID,
} from "./ensemble.ts";
import { contextualizeFeatures } from "./feature-math.ts";
import type { OnDeviceModelBundle } from "./model.ts";
import { loadOnDeviceModelBundle, runOnDeviceModel } from "./model.ts";
import {
  loadSuppressionModelBundle,
  runSuppressionModel,
  SUPPRESSION_ARTIFACT_SHA256,
  SUPPRESSION_DECODER_VERSION,
  SUPPRESSION_MODEL_ID,
  SUPPRESSION_WEIGHTS_SHA256,
  type SuppressionModelBundle,
} from "./suppression-model.ts";
import {
  buildSuppressionSuggestions,
  SUPPRESSION_POLICY_CONTRACT_VERSION,
} from "./suppression-policy.ts";
import type {
  BaseFeatureSequence,
  OnDeviceAnalysis,
  OnDeviceInterval,
} from "./types.ts";

export type ProductionRuntimeModels = {
  allLabels: OnDeviceModelBundle;
  previousProduction: OnDeviceModelBundle;
  suppression: SuppressionModelBundle;
};

let runtimeModelsPromise: Promise<ProductionRuntimeModels> | null = null;

async function fetchJson(
  asset: Parameters<typeof runtimeAssetUrl>[0],
): Promise<unknown> {
  const response = await fetch(runtimeAssetUrl(asset));
  if (!response.ok) {
    throw new Error(`Could not load ${asset} (${response.status}).`);
  }
  return response.json();
}

export async function loadProductionRuntimeModels(): Promise<ProductionRuntimeModels> {
  runtimeModelsPromise ??= Promise.all([
    fetchJson("model-1ca43e38eefc.json"),
    fetchJson("model-9c92b8e9333f.json"),
    fetchJson("suppression-39eddf581639.json"),
  ])
    .then(([allLabels, previousProduction, suppression]) => ({
      allLabels: loadOnDeviceModelBundle(allLabels),
      previousProduction: loadOnDeviceModelBundle(previousProduction),
      suppression: loadSuppressionModelBundle(suppression),
    }))
    .catch((cause) => {
      runtimeModelsPromise = null;
      throw cause;
    });
  return runtimeModelsPromise;
}

function validateFeatureSignature(
  names: readonly string[],
  models: ProductionRuntimeModels,
): void {
  for (const [modelId, featureNames] of [
    [ALL_LABELS_V2_MODEL_ID, models.allLabels.featureNames],
    [PREVIOUS_PRODUCTION_MODEL_ID, models.previousProduction.featureNames],
    [SUPPRESSION_MODEL_ID, models.suppression.featureNames],
  ] as const) {
    if (
      names.length !== featureNames.length ||
      names.some((name, index) => name !== featureNames[index])
    ) {
      throw new Error(`Extracted feature signature does not match ${modelId}.`);
    }
  }
}

function taggedIntervals(
  prefix: string,
  source: readonly OnDeviceInterval[],
  analysisWindow: AnalysisWindow,
): OnDeviceInterval[] {
  return source
    .map((interval, index) => ({
      ...interval,
      id: `${prefix}${String(index + 1).padStart(3, "0")}`,
      start: Math.max(analysisWindow.start, interval.start),
      end: Math.min(analysisWindow.end, interval.end),
    }))
    .filter((interval) => interval.end > interval.start);
}

export async function runProductionInferenceFromFeatures(
  sequence: BaseFeatureSequence,
  analysisWindow: AnalysisWindow,
): Promise<OnDeviceAnalysis> {
  const models = await loadProductionRuntimeModels();
  return runProductionInferenceWithLoadedModels(
    sequence,
    analysisWindow,
    models,
  );
}

export function runProductionInferenceWithLoadedModels(
  sequence: BaseFeatureSequence,
  analysisWindow: AnalysisWindow,
  models: ProductionRuntimeModels,
): OnDeviceAnalysis {
  const contextual = contextualizeFeatures(
    sequence.times,
    sequence.values,
    sequence.names,
  );
  validateFeatureSignature(contextual.names, models);
  const allLabelsInference = runOnDeviceModel(
    models.allLabels,
    sequence.times,
    contextual.values,
    analysisWindow.end,
  );
  const previousProductionInference = runOnDeviceModel(
    models.previousProduction,
    sequence.times,
    contextual.values,
    analysisWindow.end,
  );
  const allLabelsV2 = taggedIntervals(
    "AV2-",
    allLabelsInference.rallies,
    analysisWindow,
  );
  const previousProduction = taggedIntervals(
    "PP-",
    previousProductionInference.rallies,
    analysisWindow,
  );
  const intervals = mergeProductionModelIntervals(
    allLabelsV2,
    previousProduction,
  )
    .map((interval) => ({
      ...interval,
      start: Math.max(analysisWindow.start, interval.start),
      end: Math.min(analysisWindow.end, interval.end),
    }))
    .filter((interval) => interval.end > interval.start);
  const suppressionInference = runSuppressionModel(
    models.suppression,
    sequence.times,
    contextual.values,
    analysisWindow.end,
  );
  const decodedIntervals = taggedIntervals(
    "S",
    suppressionInference.intervals.map((interval, index) => ({
      ...interval,
      id: `S${String(index + 1).padStart(3, "0")}`,
      included: true,
    })),
    analysisWindow,
  ).map((interval, index) => ({
    ...interval,
    id: `S${String(index + 1).padStart(3, "0")}`,
  }));
  const policy = buildSuppressionSuggestions(
    { allLabelsV2, previousProduction },
    decodedIntervals,
    analysisWindow.end,
  );
  return {
    modelId: PRODUCTION_ENSEMBLE_MODEL_ID,
    featurePath: "local-source",
    intervals,
    times: new Float64Array(sequence.times),
    featureNames: [...sequence.names],
    featureValues: new Float32Array(sequence.values),
    rallyProbabilities: allLabelsInference.probabilities.rally,
    serveProbabilities: allLabelsInference.probabilities.serve,
    deadStateProbabilities: allLabelsInference.probabilities.deadState,
    productionComponents: { allLabelsV2, previousProduction },
    productionServeOutputs: {
      allLabelsV2: {
        probabilities: allLabelsInference.probabilities.serve,
        detections: allLabelsInference.serves.map((detection) => ({
          ...detection,
        })),
      },
      previousProduction: {
        probabilities: previousProductionInference.probabilities.serve,
        detections: previousProductionInference.serves.map((detection) => ({
          ...detection,
        })),
      },
    },
    suppression: {
      modelId: SUPPRESSION_MODEL_ID,
      artifactSha256: SUPPRESSION_ARTIFACT_SHA256,
      weightsSha256: SUPPRESSION_WEIGHTS_SHA256,
      decoderVersion: SUPPRESSION_DECODER_VERSION,
      policyContractVersion: SUPPRESSION_POLICY_CONTRACT_VERSION,
      probabilities: suppressionInference.probabilities,
      decodedIntervals,
      suggestions: policy.suggestions,
      identicalPolicyResults: policy.identicalPolicyResults,
    },
  };
}

export async function augmentStoredAnalysisWithSuppression(
  analysis: OnDeviceAnalysis,
  analysisWindow: AnalysisWindow,
): Promise<OnDeviceAnalysis | null> {
  if (
    analysis.productionComponents &&
    analysis.productionServeOutputs &&
    analysis.suppression
  ) {
    return analysis;
  }
  if (!analysis.featureNames || !analysis.featureValues) return null;
  return runProductionInferenceFromFeatures(
    {
      times: analysis.times,
      values: analysis.featureValues,
      rows: analysis.times.length,
      columns: analysis.featureNames.length,
      names: analysis.featureNames,
    },
    analysisWindow,
  );
}
