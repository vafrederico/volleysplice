import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_VISIBLE_MODEL_IDS,
  ENVIRONMENT_EXPERIMENT_MODELS,
  PREFERRED_ENVIRONMENT_EXPERIMENT_MODEL_ID,
} from "../../lib/experiment-models.ts";
import { PRODUCTION_MODEL_ID } from "../../lib/production-model.ts";

test("environment experiment model identities and default visibility stay explicit", () => {
  assert.deepEqual(
    ENVIRONMENT_EXPERIMENT_MODELS.map((model) => model.id),
    ["model-942b67f0d3ab", "model-04dc7d97e693", "model-69a90313927f"],
  );
  assert.equal(PREFERRED_ENVIRONMENT_EXPERIMENT_MODEL_ID, "model-942b67f0d3ab");
  assert.deepEqual(
    [...DEFAULT_VISIBLE_MODEL_IDS],
    [
      PRODUCTION_MODEL_ID,
      "model-942b67f0d3ab",
      "model-04dc7d97e693",
      "model-69a90313927f",
    ],
  );
  assert.equal(
    [...DEFAULT_VISIBLE_MODEL_IDS].some((id) =>
      id.includes("browser-on-device"),
    ),
    false,
  );
});
