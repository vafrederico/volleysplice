import path from "node:path";

import type { TrainingCorpus } from "./analysis-types.ts";

const DEFAULT_NO_BEACH_WORKSPACE =
  "/mnt/freenas/volleycut/labeling-v1-2026-08-09-no-beach-2026-08-12";

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
