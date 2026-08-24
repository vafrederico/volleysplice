import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const prodSource = new URL("../../prod/src/", import.meta.url);

async function source(relativePath: string): Promise<string> {
  return await readFile(new URL(relativePath, prodSource), "utf8");
}

test("production score specialists share one sequential video decode", async () => {
  const [sampler, coordinator, app, editor] = await Promise.all([
    source("lib/on-device/specialist-frame-sampling.ts"),
    source("lib/on-device/score-specialists.ts"),
    source("App.tsx"),
    source("components/CutEditor.tsx"),
  ]);

  assert.equal((sampler.match(/new VideoSampleSink/g) ?? []).length, 1);
  assert.match(sampler, /samplesAtTimestampsFromSequentialPass/);
  assert.match(sampler, /servingSideTimes/);
  assert.match(sampler, /sideSwitchTimes/);
  assert.equal(
    (coordinator.match(/sampleSpecialistFramesSequentially\(/g) ?? []).length,
    1,
  );
  assert.match(coordinator, /frames\.servingSide/);
  assert.match(coordinator, /frames\.sideSwitch/);
  assert.match(app, /inferScoreSpecialists/);
  assert.match(editor, /inferScoreSpecialists/);
  assert.doesNotMatch(app, /inferServingSides|inferSideSwitches/);
  assert.doesNotMatch(editor, /inferServingSides|inferSideSwitches/);
  assert.doesNotMatch(app, /Generate serving-side score tracking/);
  assert.doesNotMatch(app, /setServingSideEnabled/);
  assert.match(app, /Generate team side-switch markers/);
  assert.match(app, /setSideSwitchEnabled/);
  assert.match(
    app,
    /const \[sideSwitchEnabled, setSideSwitchEnabled\] = useState\(false\)/,
  );
  assert.match(app, /const shouldInferServing = Boolean\(result\.productionServeOutputs\)/);
});

test("shared frames still use the frozen production feature and inference paths", async () => {
  const [serving, sideSwitch] = await Promise.all([
    source("lib/on-device/serving-side.ts"),
    source("lib/on-device/side-switch.ts"),
  ]);

  assert.match(serving, /composeServingSideVerdict/);
  assert.match(serving, /tiedPercentileRanks/);
  assert.match(serving, /SERVING_SIDE_FEATURE_VERSION/);
  assert.match(sideSwitch, /predictSideSwitchProbabilities/);
  assert.match(sideSwitch, /decodeSideSwitchCandidates/);
  assert.match(sideSwitch, /SIDE_SWITCH_FEATURE_VERSION/);
  assert.doesNotMatch(serving, /Benchmark|benchmark/);
  assert.doesNotMatch(sideSwitch, /Benchmark|benchmark/);
});
