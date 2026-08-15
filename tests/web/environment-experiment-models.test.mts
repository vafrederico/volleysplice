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
    ["model-1ca43e38eefc", "model-04dc7d97e693", "model-18d5e86f8923"],
  );
  assert.equal(PREFERRED_ENVIRONMENT_EXPERIMENT_MODEL_ID, "model-1ca43e38eefc");
  assert.deepEqual(
    [...DEFAULT_VISIBLE_MODEL_IDS],
    [
      PRODUCTION_MODEL_ID,
      "model-1ca43e38eefc",
      "model-04dc7d97e693",
      "model-18d5e86f8923",
    ],
  );
  assert.equal(
    [...DEFAULT_VISIBLE_MODEL_IDS].some((id) =>
      id.includes("browser-on-device"),
    ),
    false,
  );
});
