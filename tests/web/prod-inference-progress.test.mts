import assert from "node:assert/strict";
import test from "node:test";

import {
  completedInferenceStepCount,
  CORE_INFERENCE_STEP_IDS,
  createInferenceProgressSteps,
  finishInferenceStep,
  inferenceStepEtaSeconds,
  overallInferenceProgress,
  SCORE_INFERENCE_STEP_IDS,
  updatePipelineInferenceSteps,
  updateSpecialistInferenceStep,
} from "../../prod/src/lib/inference-progress.ts";

test("core and score inference use explicit countable steps", () => {
  const core = createInferenceProgressSteps(CORE_INFERENCE_STEP_IDS);
  assert.deepEqual(
    core.map((step) => step.label),
    ["Video features", "Audio features", "Rally inference"],
  );
  assert.equal(completedInferenceStepCount(core), 0);

  const withScore = createInferenceProgressSteps([
    ...CORE_INFERENCE_STEP_IDS,
    ...SCORE_INFERENCE_STEP_IDS,
  ]);
  assert.equal(withScore.length, 5);
  assert.deepEqual(
    withScore.slice(3).map((step) => step.label),
    ["Serving side", "Side switches"],
  );
});

test("pipeline transitions retain completed bars and stage-local speed", () => {
  let steps = createInferenceProgressSteps(CORE_INFERENCE_STEP_IDS);
  steps = updatePipelineInferenceSteps(
    steps,
    {
      stage: "video",
      completed: 0,
      total: 100,
      detail: "Starting video",
    },
    1_000,
  );
  steps = updatePipelineInferenceSteps(
    steps,
    {
      stage: "video",
      completed: 20,
      total: 100,
      detail: "Measuring motion",
    },
    3_000,
  );
  assert.equal(steps[0].status, "running");
  assert.equal(steps[0].fraction, 0.2);
  assert.equal(steps[0].rate, 10);

  steps = updatePipelineInferenceSteps(
    steps,
    {
      stage: "audio",
      completed: 10,
      total: 100,
      detail: "Decoding audio",
    },
    4_000,
  );
  assert.equal(steps[0].status, "complete");
  assert.equal(steps[0].fraction, 1);
  assert.equal(steps[1].status, "running");
  assert.equal(completedInferenceStepCount(steps), 1);
  assert.equal(overallInferenceProgress(steps), (1 + 0.1) / 3);
});

test("specialist frame and feature phases advance monotonically", () => {
  let steps = createInferenceProgressSteps(SCORE_INFERENCE_STEP_IDS);
  steps = updateSpecialistInferenceStep(
    steps,
    "serving-side",
    {
      stage: "loading",
      completed: 0,
      total: 10,
      detail: "Loading model",
    },
    1_000,
  );
  assert.equal(steps[0].fraction, 0.02);

  steps = updateSpecialistInferenceStep(
    steps,
    "serving-side",
    {
      stage: "frames",
      completed: 5,
      total: 10,
      detail: "Sampling frames",
    },
    2_000,
  );
  assert.equal(steps[0].fraction, 0.365);
  assert.equal(steps[0].rateUnit, "frames/s");

  steps = updateSpecialistInferenceStep(
    steps,
    "serving-side",
    {
      stage: "features",
      completed: 1,
      total: 4,
      detail: "Measuring rallies",
    },
    3_000,
  );
  assert.equal(steps[0].fraction, 0.755);
  assert.equal(steps[0].rateUnit, "rallies/s");

  steps = finishInferenceStep(steps, "serving-side", "Ready", 4_000);
  assert.equal(steps[0].status, "complete");
  assert.equal(steps[0].fraction, 1);
  assert.equal(completedInferenceStepCount(steps), 1);
});

test("each running step receives an independent ETA", () => {
  let steps = createInferenceProgressSteps(["video"]);
  steps = updatePipelineInferenceSteps(
    steps,
    {
      stage: "video",
      completed: 0,
      total: 100,
      detail: "Starting",
    },
    1_000,
  );
  steps = updatePipelineInferenceSteps(
    steps,
    {
      stage: "video",
      completed: 50,
      total: 100,
      detail: "Halfway",
    },
    5_000,
  );
  assert.equal(inferenceStepEtaSeconds(steps[0], 5_000), 4);

  steps = finishInferenceStep(steps, "video", "Done", 6_000);
  assert.equal(inferenceStepEtaSeconds(steps[0], 10_000), 0);
});
