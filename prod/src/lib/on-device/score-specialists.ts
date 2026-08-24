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
): Promise<ScoreSpecialistInferenceOutput> {
  if (!request.servingSide && !request.sideSwitch) return {};
  const loadingOwner = request.servingSide ?? request.sideSwitch;
  loadingOwner?.onProgress?.({
    stage: "loading",
    completed: 0,
    total: 1,
    detail:
      request.servingSide && request.sideSwitch
        ? "Loading serving-side and team-switch models"
        : request.servingSide
          ? "Loading serving-side model"
          : "Loading team-side switch model",
  });
  const [servingPlan, switchPlan] = await Promise.all([
    request.servingSide
      ? Promise.resolve(
          servingSideFramePlan(
            request.servingSide.analysis,
            media.info.duration,
            null,
          ),
        )
      : Promise.resolve(null),
    request.sideSwitch
      ? sideSwitchFramePlan(request.sideSwitch.analysis, media.info.duration)
      : Promise.resolve(null),
  ]);
  const frames = await sampleSpecialistFramesSequentially(
    media,
    roi,
    {
      servingSideTimes: servingPlan?.requestedTimes,
      sideSwitchTimes: switchPlan?.requestedTimes,
    },
    (completed, total) => {
      if (request.servingSide) {
        request.servingSide.onProgress?.({
          stage: "frames",
          completed,
          total,
          detail: request.sideSwitch
            ? `Shared serving + side-switch decode · ${completed}/${total} frames`
            : `Sampling serving-side frames · ${completed}/${total}`,
        });
      } else {
        request.sideSwitch?.onProgress?.({
          stage: "frames",
          completed,
          total,
          detail: `Sampling team-side switch frames · ${completed}/${total}`,
        });
      }
    },
  );
  const output: ScoreSpecialistInferenceOutput = {};
  if (request.servingSide) {
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
          detail: `Measuring serving side · ${completed}/${total} rallies`,
        }),
    );
    output.servingSide = artifacts.output;
    request.servingSide.onProgress?.({
      stage: "complete",
      completed: artifacts.output.candidates.length,
      total: artifacts.output.candidates.length,
      detail: `Serving-side verdicts ready · ${artifacts.output.candidates.length} rallies`,
    });
  }
  if (request.sideSwitch) {
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
          detail: `Comparing team sides · ${completed}/${total} candidates`,
        }),
    );
    output.sideSwitch = artifacts.output;
    request.sideSwitch.onProgress?.({
      stage: "complete",
      completed: artifacts.output.candidates.length,
      total: artifacts.output.candidates.length,
      detail: `Team-side switch markers ready · ${artifacts.output.candidates.length} predicted`,
    });
  }
  return output;
}
