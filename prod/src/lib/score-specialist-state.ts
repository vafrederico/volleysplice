import { isReusableServingSideOutput } from "./on-device/serving-side-cache.ts";
import { isReusableSideSwitchOutput } from "./on-device/side-switch-model.ts";
import type { OnDeviceAnalysis } from "./on-device/types.ts";

export type ScoreSpecialistStage = "servingSide" | "sideSwitch";
export type ScoreSpecialistErrors = Partial<Record<ScoreSpecialistStage, string>>;
export type ScoreStageReadiness = {
  status: "complete" | "missing" | "failed" | "unavailable" | "disabled";
  message: string;
};
export type ScoreSpecialistReadiness = Record<ScoreSpecialistStage, ScoreStageReadiness>;

/** Completion is evidence, never the absence of markers or an old success flag.
 * undefined/null mean absent. A validated output with zero candidates is complete.
 * Imported historical outputs remain reviewable, but need current validation to
 * claim completion or skip regeneration.
 */
export function scoreSpecialistReadiness(project: {
  analysis: OnDeviceAnalysis | null;
  sideSwitchEnabled?: boolean;
  scoreSpecialistErrors?: ScoreSpecialistErrors;
}): ScoreSpecialistReadiness {
  const analysis = project.analysis;
  const features = Boolean(analysis?.featureNames?.length && analysis.featureValues);
  const stage = (key: ScoreSpecialistStage, complete: boolean, available: boolean): ScoreStageReadiness => {
    if (complete) return { status: "complete", message: "Predictions ready" };
    const message = project.scoreSpecialistErrors?.[key];
    if (message) return { status: "failed", message };
    return available
      ? { status: "missing", message: "Predictions have not been generated or need refreshing." }
      : { status: "unavailable", message: "This saved analysis has no score evidence. Add markers manually." };
  };
  return {
    servingSide: stage("servingSide", Boolean(analysis && isReusableServingSideOutput(
      analysis.servingSide, analysis.intervals, analysis.modelId,
    )), Boolean(analysis?.productionServeOutputs || features)),
    sideSwitch: project.sideSwitchEnabled === false
      ? { status: "disabled", message: "Automatic side switches are off." }
      : stage("sideSwitch", isReusableSideSwitchOutput(analysis?.sideSwitch),
          Boolean((analysis?.productionComponents && analysis.productionStateOutputs) || features)),
  };
}

export function missingScoreStages(project: Parameters<typeof scoreSpecialistReadiness>[0]): ScoreSpecialistStage[] {
  const readiness = scoreSpecialistReadiness(project);
  return (["servingSide", "sideSwitch"] as const).filter((key) =>
    readiness[key].status === "missing" || readiness[key].status === "failed",
  );
}
