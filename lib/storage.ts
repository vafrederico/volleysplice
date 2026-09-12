import path from "node:path";

import type { TrainingCorpus } from "./analysis-types.ts";

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
      : path.join(getDataRoot(), "no-beach", "analyses");
  }
  return path.join(getDataRoot(), "analyses");
}

export function getIntakeWorkspace(): string {
  return getIntakeWorkspaces()[0];
}

export function getIntakeWorkspaces(): string[] {
  const configured = [
    ...(process.env.VOLLEYCUT_INTAKE_WORKSPACE?.trim()
      ? [process.env.VOLLEYCUT_INTAKE_WORKSPACE.trim()]
      : []),
    ...(process.env.VOLLEYCUT_INTAKE_WORKSPACES?.split(path.delimiter)
      .map((value) => value.trim())
      .filter(Boolean) ?? []),
  ];
  const roots = configured.length > 0 ? configured : [path.join(getDataRoot(), "intake")];
  return [...new Set(roots.map((root) => path.resolve(root)))];
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
