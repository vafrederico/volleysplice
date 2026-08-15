export const PRODUCTION_MODEL_ID = "model-1ca43e38eefc";
export const PRODUCTION_MODEL_LABEL = "Production · All labels v2";
export const PRODUCTION_MODEL_VERSION = "environment-specialists-v2-all-labels";
export const PRODUCTION_MODEL_DESCRIPTION =
  "Production three-head architecture refit on all six grass recordings and five indoor recordings, including the latest walking and ball-retrieval hard negatives.";

export const PREVIOUS_PRODUCTION_MODEL_ID = "model-9c92b8e9333f";
export const PREVIOUS_PRODUCTION_MODEL_LABEL = "Previous production";
export const PRODUCTION_ENSEMBLE_ALGORITHM_VERSION =
  "overlap-union-disagreement-v1";
export const PRODUCTION_ENSEMBLE_MODEL_ID =
  "model-ensemble-1ca43e38eefc-9c92b8e9333f";
export const PRODUCTION_ENSEMBLE_MODEL_LABEL = "Production ensemble · New + previous";
export const PRODUCTION_ENSEMBLE_MODEL_VERSION =
  `${PRODUCTION_ENSEMBLE_ALGORITHM_VERSION}-1ca43e38eefc-9c92b8e9333f`;
export const PRODUCTION_ENSEMBLE_MODEL_DESCRIPTION =
  "The production-app recall-safety ensemble. It unions overlapping ranges from all-labels v2 and the previous production model; one-model detections remain included and are marked as disagreements for review.";
