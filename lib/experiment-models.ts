import { PRODUCTION_MODEL_ID } from "./production-model.ts";

export type EnvironmentExperimentModel = {
  id: string;
  label: string;
  description: string;
};

export const ENVIRONMENT_EXPERIMENT_MODELS = [
  {
    id: "model-1ca43e38eefc",
    label: "All labels + latest grass v2",
    description:
      "Production three-head architecture refit on all six grass recordings and five indoor recordings, including Turkey Tourney, Forest Ridge, and YMCA KOB Set 4.",
  },
  {
    id: "model-04dc7d97e693",
    label: "Grass specialist",
    description:
      "Production three-head architecture trained only on grass labels, including the latest walking / ball-retrieval hard negatives.",
  },
  {
    id: "model-18d5e86f8923",
    label: "Indoor specialist v2",
    description:
      "Production three-head architecture trained only on five indoor recordings across four source groups; SPU Match 1 Set 2 remains evaluation-only.",
  },
] as const satisfies readonly EnvironmentExperimentModel[];

export const PREFERRED_ENVIRONMENT_EXPERIMENT_MODEL_ID =
  ENVIRONMENT_EXPERIMENT_MODELS[0].id;

export const DEFAULT_VISIBLE_MODEL_IDS: ReadonlySet<string> = new Set([
  PRODUCTION_MODEL_ID,
  ...ENVIRONMENT_EXPERIMENT_MODELS.map((model) => model.id),
]);
