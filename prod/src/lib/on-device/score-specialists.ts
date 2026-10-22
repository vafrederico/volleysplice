import {
  evaluateServingSideFrames,
  servingSideFramePlan,
  type ServingSideAnalysisInput,
  type ServingSideProgress,
} from "./serving-side.ts";
import {
  evaluateSideSwitchFrames,
  sideSwitchFramePlan,
  type SideSwitchProgress,
} from "./side-switch.ts";
import type { SideSwitchAnalysisInput } from "./side-switch-model.ts";
import { sampleSpecialistFramesSequentially } from "./specialist-frame-sampling.ts";
import { neuralServingSelection } from "./neural-serving-model.ts";
import type { OpenedMedia } from "./media.ts";
import type {
  NormalizedRoi,
  OnDeviceServingSideOutput,
  OnDeviceSideSwitchOutput,
} from "./types.ts";

export type ScoreSpecialistInferenceRequest = {
  servingSide?: {
    analysis: ServingSideAnalysisInput;
    onProgress?: (progress: ServingSideProgress) => void;
  };
  sideSwitch?: {
    analysis: SideSwitchAnalysisInput;
    onProgress?: (progress: SideSwitchProgress) => void;
  };
};

export type ScoreSpecialistInferenceOutput = {
  servingSide?: OnDeviceServingSideOutput;
  sideSwitch?: OnDeviceSideSwitchOutput;
};

/**
 * Runs one shared sequential video decode for every missing score specialist,
 * then evaluates their frozen feature and model contracts independently.
 */
export async function inferScoreSpecialists(
  media: OpenedMedia,
  roi: NormalizedRoi,
  request: ScoreSpecialistInferenceRequest,
  options: {
    signal?: AbortSignal;
    /** Save each completed stage before another stage can fail. */
    onOutput?: (output: ScoreSpecialistInferenceOutput) => void;
  } = {},
): Promise<ScoreSpecialistInferenceOutput> {
  const check = () => options.signal?.throwIfAborted();
  check();
  if (!request.servingSide && !request.sideSwitch) return {};
  const loadingOwner = request.servingSide ?? request.sideSwitch;
  loadingOwner?.onProgress?.({
    stage: "loading",
    completed: 0,
    total: 1,
    detail:
      request.servingSide && request.sideSwitch
        ? "Getting serve and side-switch markers ready"
        : request.servingSide
          ? "Getting serve markers ready"
          : "Getting side-switch markers ready",
  });
  // Planning can itself load a model or reject invalid candidate identity.
  // Capture synchronous and asynchronous failures per stage before building
  // the shared decode schedule, so one failed plan cannot discard the other.
  const [servingResult, switchResult] = await Promise.allSettled([
    Promise.resolve().then(() =>
      request.servingSide
        ? servingSideFramePlan(
            request.servingSide.analysis,
            media.info.duration,
            null,
          )
        : null,
    ),
    Promise.resolve().then(() =>
      request.sideSwitch
        ? sideSwitchFramePlan(request.sideSwitch.analysis, media.info.duration)
        : null,
    ),
  ]);
  check();
  const errors: string[] = [];
  const planned = <T>(result: PromiseSettledResult<T>): T | null => {
    if (result.status === "fulfilled") return result.value;
    errors.push(result.reason instanceof Error ? result.reason.message : String(result.reason));
    return null;
  };
  const servingPlan = planned(servingResult);
  const switchPlan = planned(switchResult);
  if (!servingPlan && !switchPlan) throw new Error(errors.join(" "));
  const frames = await sampleSpecialistFramesSequentially(
    media,
    roi,
    {
      servingSideTimes: servingPlan?.requestedTimes,
      sideSwitchTimes: switchPlan?.requestedTimes,
    },
    (completed, total) => {
      check();
      if (servingPlan && request.servingSide) {
        request.servingSide.onProgress?.({
          stage: "frames",
          completed,
          total,
          detail: switchPlan
            ? `Checking serves and side switches · ${completed}/${total}`
            : `Checking serve court sides · ${completed}/${total}`,
        });
      } else if (switchPlan) {
        request.sideSwitch?.onProgress?.({
          stage: "frames",
          completed,
          total,
          detail: `Looking for team side switches · ${completed}/${total}`,
        });
      }
    },
    options.signal,
  );
  const output: ScoreSpecialistInferenceOutput = {};
  if (servingPlan && request.servingSide) {
    try {
      check();
      const artifacts = await evaluateServingSideFrames(
        media.info.duration,
        request.servingSide.analysis,
        null,
        frames.servingSide,
        (completed, total) =>
          request.servingSide?.onProgress?.({
            stage: "features",
            completed,
            total,
            detail: `Checking serve court sides · ${completed}/${total} rallies`,
          }),
      );
      output.servingSide = neuralServingSelection(request.servingSide.analysis.modelId)
        ? await (await import("./neural-serving")).inferNeuralServingSideOutput(
            media, roi, request.servingSide.analysis, artifacts.output, request.servingSide.onProgress, options.signal,
          )
        : artifacts.output;
      check();
      options.onOutput?.({ servingSide: output.servingSide });
      request.servingSide.onProgress?.({
        stage: "complete",
        completed: artifacts.output.candidates.length,
        total: artifacts.output.candidates.length,
        detail: `Serve markers ready · ${artifacts.output.candidates.length} rallies checked`,
      });
    } catch (cause) {
      check();
      errors.push(cause instanceof Error ? cause.message : String(cause));
    }
  }
  if (switchPlan && request.sideSwitch) {
    try {
      check();
      const artifacts = await evaluateSideSwitchFrames(
        media.info.duration,
        request.sideSwitch.analysis,
        null,
        frames.sideSwitch,
        (completed, total) =>
          request.sideSwitch?.onProgress?.({
            stage: "features",
            completed,
            total,
            detail: `Checking possible side switches · ${completed}/${total}`,
          }),
      );
      output.sideSwitch = artifacts.output;
      check();
      options.onOutput?.({ sideSwitch: output.sideSwitch });
      request.sideSwitch.onProgress?.({
        stage: "complete",
        completed: artifacts.output.candidates.length,
        total: artifacts.output.candidates.length,
        detail: `Side switches ready · ${artifacts.output.candidates.length} found`,
      });
    } catch (cause) {
      check();
      errors.push(cause instanceof Error ? cause.message : String(cause));
    }
  }
  check();
  if (errors.length) throw new Error(errors.join(" "));
  return output;
}
