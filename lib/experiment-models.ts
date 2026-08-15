import { PRODUCTION_MODEL_ID } from "./production-model.ts";

export type EnvironmentExperimentModel = {
  id: string;
  label: string;
  description: string;
};

export const ENVIRONMENT_EXPERIMENT_MODELS = [
  {
    id: "model-942b67f0d3ab",
    label: "All labels + latest grass",
    description:
      "Production three-head architecture refit with the established corpus plus the latest grass labels and walking / ball-retrieval hard negatives.",
  },
  {
    id: "model-04dc7d97e693",
    label: "Grass specialist",
    description:
      "Production three-head architecture trained only on grass labels, including the latest walking / ball-retrieval hard negatives.",
  },
  {
    id: "model-69a90313927f",
    label: "Indoor specialist",
    description:
      "Production three-head architecture trained only on the pre-existing indoor training labels; the newest indoor labels remain evaluation-only.",
  },
] as const satisfies readonly EnvironmentExperimentModel[];

export const PREFERRED_ENVIRONMENT_EXPERIMENT_MODEL_ID =
  ENVIRONMENT_EXPERIMENT_MODELS[0].id;

export const DEFAULT_VISIBLE_MODEL_IDS: ReadonlySet<string> = new Set([
  PRODUCTION_MODEL_ID,
  ...ENVIRONMENT_EXPERIMENT_MODELS.map((model) => model.id),
]);
