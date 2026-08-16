import path from "node:path";

import type { TrainingCorpus } from "./analysis-types.ts";

const DEFAULT_NO_BEACH_WORKSPACE =
  "/mnt/freenas/volleycut/labeling-v1-2026-08-09-no-beach-2026-08-12";
const DEFAULT_INTAKE_WORKSPACE = "/mnt/freenas/volleycut/intake-2026-08-13";

export function getDataRoot(): string {
  const configured = process.env.VOLLEYCUT_DATA_ROOT?.trim();
  return configured ? path.resolve(configured) : path.join(process.cwd(), "data");
}

export function getAnalysesRoot(
  corpus: Extract<TrainingCorpus, "original" | "without-beach"> = "original",
): string {
  if (corpus === "without-beach") {
    const configured = process.env.VOLLEYCUT_NO_BEACH_ANALYSES_ROOT?.trim();
    return configured
      ? path.resolve(configured)
      : path.join(DEFAULT_NO_BEACH_WORKSPACE, "analyses");
  }
  return path.join(getDataRoot(), "analyses");
}

export function getIntakeWorkspace(): string {
  const configured = process.env.VOLLEYCUT_INTAKE_WORKSPACE?.trim();
  return configured ? path.resolve(configured) : DEFAULT_INTAKE_WORKSPACE;
}

export function getIntakeAnalysesRoot(): string {
  return path.join(getIntakeWorkspace(), "analyses");
}

export function getModelFeedbackRoot(): string {
  const configured = process.env.VOLLEYCUT_MODEL_FEEDBACK_ROOT?.trim();
  return configured
    ? path.resolve(configured)
    : path.join(getDataRoot(), "model-feedback");
}
