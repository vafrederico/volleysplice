import type { AnalysisProgress } from "./on-device/types.ts";

export type InferenceProgressStepId =
  | "video"
  | "audio"
  | "rally"
  | "serving-side"
  | "side-switch";

export type InferenceProgressStepStatus =
  | "pending"
  | "running"
  | "complete"
  | "skipped"
  | "error";

export type SpecialistProgress = {
  stage: "loading" | "frames" | "features" | "complete";
  completed: number;
  total: number;
  detail: string;
};

export type InferenceProgressStep = {
  id: InferenceProgressStepId;
  label: string;
  status: InferenceProgressStepStatus;
  fraction: number;
  detail: string;
  startedAtMs: number | null;
  updatedAtMs: number | null;
  finishedAtMs: number | null;
  phase: string | null;
  phaseStartedAtMs: number | null;
  rate: number | null;
  rateUnit: "realtime" | "frames/s" | "rallies/s" | "candidates/s";
};

const STEP_LABELS: Record<InferenceProgressStepId, string> = {
  video: "Video features",
  audio: "Audio features",
  rally: "Rally inference",
  "serving-side": "Serving side",
  "side-switch": "Side switches",
};

export const CORE_INFERENCE_STEP_IDS = ["video", "audio", "rally"] as const;
export const SCORE_INFERENCE_STEP_IDS = [
  "serving-side",
  "side-switch",
] as const;

function clampFraction(value: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}

export function createInferenceProgressSteps(
  ids: readonly InferenceProgressStepId[],
): InferenceProgressStep[] {
  return ids.map((id) => ({
    id,
    label: STEP_LABELS[id],
    status: "pending",
    fraction: 0,
    detail: "Waiting for the previous step",
    startedAtMs: null,
    updatedAtMs: null,
    finishedAtMs: null,
    phase: null,
    phaseStartedAtMs: null,
    rate: null,
    rateUnit:
      id === "video" || id === "audio" || id === "rally"
        ? "realtime"
        : id === "serving-side"
          ? "rallies/s"
          : id === "side-switch"
            ? "candidates/s"
            : "frames/s",
  }));
}

function replaceStep(
  steps: readonly InferenceProgressStep[],
  id: InferenceProgressStepId,
  update: (step: InferenceProgressStep) => InferenceProgressStep,
): InferenceProgressStep[] {
  return steps.map((step) => (step.id === id ? update(step) : step));
}

function completeEarlierSteps(
  steps: readonly InferenceProgressStep[],
  id: InferenceProgressStepId,
  nowMs: number,
): InferenceProgressStep[] {
  const target = steps.findIndex((step) => step.id === id);
  if (target <= 0) return [...steps];
  return steps.map((step, index) => {
    if (
      index >= target ||
      step.status === "complete" ||
      step.status === "skipped" ||
      step.status === "error"
    )
      return step;
    return {
      ...step,
      status: "complete",
      fraction: 1,
      detail: `${step.label} complete`,
      startedAtMs: step.startedAtMs ?? nowMs,
      updatedAtMs: nowMs,
      finishedAtMs: nowMs,
    };
  });
}

function updateRunningStep(
  steps: readonly InferenceProgressStep[],
  id: InferenceProgressStepId,
  options: {
    fraction: number;
    detail: string;
    phase: string;
    completed: number;
    nowMs: number;
    rate?: number | null;
    rateUnit?: InferenceProgressStep["rateUnit"];
  },
): InferenceProgressStep[] {
  const prepared = completeEarlierSteps(steps, id, options.nowMs);
  return replaceStep(prepared, id, (step) => {
    const phaseChanged = step.phase !== options.phase;
    const phaseStartedAtMs = phaseChanged
      ? options.nowMs
      : (step.phaseStartedAtMs ?? options.nowMs);
    const phaseElapsedSeconds = Math.max(
      0,
      (options.nowMs - phaseStartedAtMs) / 1000,
    );
    const measuredRate =
      options.rate !== undefined
        ? options.rate
        : phaseElapsedSeconds >= 0.25 && options.completed > 0
          ? options.completed / phaseElapsedSeconds
          : phaseChanged
            ? null
            : step.rate;
    const fraction = clampFraction(options.fraction);
    const complete = fraction >= 1;
    return {
      ...step,
      status: complete ? "complete" : "running",
      fraction,
      detail: options.detail,
      startedAtMs: step.startedAtMs ?? options.nowMs,
      updatedAtMs: options.nowMs,
      finishedAtMs: complete ? options.nowMs : null,
      phase: options.phase,
      phaseStartedAtMs,
      rate: measuredRate ?? null,
      rateUnit: options.rateUnit ?? step.rateUnit,
    };
  });
}

export function updatePipelineInferenceSteps(
  steps: readonly InferenceProgressStep[],
  progress: AnalysisProgress,
  nowMs: number,
): InferenceProgressStep[] {
  const fraction = progress.total > 0 ? progress.completed / progress.total : 0;
  if (progress.stage === "video") {
    const performance = progress.performance;
    const rate =
      performance && performance.videoElapsedMs > 0
        ? performance.generatedVideoSeconds /
          (performance.videoElapsedMs / 1000)
        : undefined;
    return updateRunningStep(steps, "video", {
      fraction,
      detail: progress.detail,
      phase: "features",
      completed: progress.completed,
      nowMs,
      rate,
      rateUnit: "realtime",
    });
  }
  if (progress.stage === "audio") {
    const elapsedSeconds = progress.audioPerformance
      ? progress.audioPerformance.elapsedMs / 1000
      : 0;
    return updateRunningStep(steps, "audio", {
      fraction,
      detail: progress.detail,
      phase: progress.audioPerformance?.phase ?? "features",
      completed: progress.completed,
      nowMs,
      rate:
        elapsedSeconds > 0 ? progress.completed / elapsedSeconds : undefined,
      rateUnit: "realtime",
    });
  }
  if (progress.stage === "normalizing") {
    return updateRunningStep(steps, "rally", {
      fraction: 0.12,
      detail: progress.detail,
      phase: "normalizing",
      completed: 0,
      nowMs,
      rateUnit: "realtime",
    });
  }
  if (progress.stage === "inference") {
    return updateRunningStep(steps, "rally", {
      fraction: 0.42,
      detail: progress.detail,
      phase: "models",
      completed: 0,
      nowMs,
      rateUnit: "realtime",
    });
  }
  if (progress.stage === "complete") {
    const step = steps.find((candidate) => candidate.id === "rally");
    const elapsedSeconds =
      step?.startedAtMs === null || step?.startedAtMs === undefined
        ? 0
        : Math.max(0, (nowMs - step.startedAtMs) / 1000);
    return updateRunningStep(steps, "rally", {
      fraction: 1,
      detail: progress.detail,
      phase: "complete",
      completed: progress.total,
      nowMs,
      rate: elapsedSeconds > 0 ? progress.total / elapsedSeconds : undefined,
      rateUnit: "realtime",
    });
  }
  return [...steps];
}

export function updateSpecialistInferenceStep(
  steps: readonly InferenceProgressStep[],
  id: "serving-side" | "side-switch",
  progress: SpecialistProgress,
  nowMs: number,
): InferenceProgressStep[] {
  const phaseFraction =
    progress.total > 0 ? clampFraction(progress.completed / progress.total) : 0;
  const fraction =
    progress.stage === "loading"
      ? 0.02
      : progress.stage === "frames"
        ? 0.05 + phaseFraction * 0.63
        : progress.stage === "features"
          ? 0.68 + phaseFraction * 0.3
          : 1;
  const rateUnit =
    progress.stage === "frames"
      ? "frames/s"
      : id === "serving-side"
        ? "rallies/s"
        : "candidates/s";
  return updateRunningStep(steps, id, {
    fraction,
    detail: progress.detail,
    phase: progress.stage,
    completed: progress.completed,
    nowMs,
    rateUnit,
  });
}

export function finishInferenceStep(
  steps: readonly InferenceProgressStep[],
  id: InferenceProgressStepId,
  detail: string,
  nowMs: number,
  status: "complete" | "skipped" | "error" = "complete",
): InferenceProgressStep[] {
  const prepared = completeEarlierSteps(steps, id, nowMs);
  return replaceStep(prepared, id, (step) => ({
    ...step,
    status,
    fraction: status === "error" ? step.fraction : 1,
    detail,
    startedAtMs: step.startedAtMs ?? nowMs,
    updatedAtMs: nowMs,
    finishedAtMs: nowMs,
  }));
}

export function completedInferenceStepCount(
  steps: readonly InferenceProgressStep[],
): number {
  return steps.filter(
    (step) => step.status === "complete" || step.status === "skipped",
  ).length;
}

export function overallInferenceProgress(
  steps: readonly InferenceProgressStep[],
): number {
  if (steps.length === 0) return 0;
  return steps.reduce((sum, step) => sum + step.fraction, 0) / steps.length;
}

export function inferenceStepElapsedSeconds(
  step: InferenceProgressStep,
  nowMs: number,
): number {
  if (step.startedAtMs === null) return 0;
  return Math.max(0, ((step.finishedAtMs ?? nowMs) - step.startedAtMs) / 1000);
}

export function inferenceStepEtaSeconds(
  step: InferenceProgressStep,
  nowMs: number,
): number | null {
  if (step.status === "complete" || step.status === "skipped") return 0;
  if (step.status !== "running" || step.fraction < 0.03) return null;
  const elapsed = inferenceStepElapsedSeconds(step, nowMs);
  if (elapsed < 0.5) return null;
  return Math.max(0, (elapsed * (1 - step.fraction)) / step.fraction);
}
