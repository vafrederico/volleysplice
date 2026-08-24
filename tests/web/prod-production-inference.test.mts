import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { contextualizeFeatures } from "../../prod/src/lib/on-device/feature-math.ts";
import { BASE_FEATURE_NAMES } from "../../prod/src/lib/on-device/feature-schema.ts";
import {
  loadOnDeviceModelBundle,
  runOnDeviceModel,
} from "../../prod/src/lib/on-device/model.ts";
import { runProductionInferenceWithLoadedModels } from "../../prod/src/lib/on-device/production-inference.ts";
import { loadSuppressionModelBundle } from "../../prod/src/lib/on-device/suppression-model.ts";

async function runtimeJson(fileName: string): Promise<unknown> {
  return JSON.parse(
    await readFile(
      new URL(`../../prod/public/runtime/${fileName}`, import.meta.url),
      "utf8",
    ),
  );
}

test("production inference retains both component serve and rally-state outputs", async () => {
  const rows = 96;
  const columns = BASE_FEATURE_NAMES.length;
  const times = Float64Array.from({ length: rows }, (_, index) => index * 0.25);
  const values = Float32Array.from(
    { length: rows * columns },
    (_, index) => ((index * 17 + Math.floor(index / columns) * 13) % 101) / 100,
  );
  const sequence = {
    times,
    values,
    rows,
    columns,
    names: BASE_FEATURE_NAMES,
  };
  const allLabels = loadOnDeviceModelBundle(
    await runtimeJson("model-1ca43e38eefc.json"),
  );
  const previousProduction = loadOnDeviceModelBundle(
    await runtimeJson("model-9c92b8e9333f.json"),
  );
  const contextual = contextualizeFeatures(times, values, BASE_FEATURE_NAMES);
  const expectedAllLabels = runOnDeviceModel(
    allLabels,
    times,
    contextual.values,
    24,
  );
  const expectedPreviousProduction = runOnDeviceModel(
    previousProduction,
    times,
    contextual.values,
    24,
  );

  const analysis = runProductionInferenceWithLoadedModels(
    sequence,
    { start: 0, end: 24 },
    {
      allLabels,
      previousProduction,
      suppression: loadSuppressionModelBundle(
        await runtimeJson("suppression-39eddf581639.json"),
      ),
    },
  );

  assert.ok(analysis.productionServeOutputs);
  assert.deepEqual(
    analysis.productionServeOutputs.allLabelsV2.probabilities,
    expectedAllLabels.probabilities.serve,
  );
  assert.deepEqual(
    analysis.productionServeOutputs.allLabelsV2.detections,
    expectedAllLabels.serves,
  );
  assert.deepEqual(
    analysis.productionServeOutputs.previousProduction.probabilities,
    expectedPreviousProduction.probabilities.serve,
  );
  assert.deepEqual(
    analysis.productionServeOutputs.previousProduction.detections,
    expectedPreviousProduction.serves,
  );
  assert.strictEqual(
    analysis.serveProbabilities,
    analysis.productionServeOutputs.allLabelsV2.probabilities,
  );
  assert.ok(analysis.productionStateOutputs);
  assert.strictEqual(
    analysis.productionStateOutputs.allLabelsV2.rallyProbabilities,
    analysis.rallyProbabilities,
  );
  assert.deepEqual(
    analysis.productionStateOutputs.previousProduction.rallyProbabilities,
    expectedPreviousProduction.probabilities.rally,
  );
  assert.deepEqual(
    analysis.productionStateOutputs.previousProduction.deadStateProbabilities,
    expectedPreviousProduction.probabilities.deadState,
  );
});
