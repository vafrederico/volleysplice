import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const prodSource = new URL("../../prod/src/", import.meta.url);

async function source(relativePath: string): Promise<string> {
  return await readFile(new URL(relativePath, prodSource), "utf8");
}

test("production score specialists share one sequential video decode", async () => {
  const [sampler, coordinator, app, editor, scorePanel] = await Promise.all([
    source("lib/on-device/specialist-frame-sampling.ts"),
    source("lib/on-device/score-specialists.ts"),
    source("App.tsx"),
    source("components/CutEditor.tsx"),
    source("components/ScoreTrackingPanel.tsx"),
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
  assert.match(app, /Teams change court sides during this video/);
  assert.match(app, /needed for accurate scorekeeping/);
  assert.match(app, /setSideSwitchEnabled/);
  assert.match(
    app,
    /const \[sideSwitchEnabled, setSideSwitchEnabled\] = useState\(false\)/,
  );
  assert.match(app, /servingSideEnabled: false/);
  assert.match(app, /const shouldInferServing = Boolean\(result\.productionServeOutputs\)/);
  assert.match(editor, /VolleyCut builds the score from serve markers/);
  assert.match(editor, />\s*Save project\s*</);
  assert.doesNotMatch(editor, /downloadEditList|Save a review copy/);
  assert.match(scorePanel, /HOW THE SCORE IS BUILT/);
  assert.match(scorePanel, /Add a side-switch marker whenever teams change court sides/);
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

test("the task-focused tutorial remains available in setup and review", async () => {
  const [app, editor, tutorial] = await Promise.all([
    source("App.tsx"),
    source("components/CutEditor.tsx"),
    source("components/GuidedTour.tsx"),
  ]);

  assert.match(app, /<GuidedTour stage="source"/);
  assert.match(editor, /<GuidedTour\s+stage="editor"/);
  assert.match(tutorial, /Restart tutorial/);
  assert.match(tutorial, /Choose your video/);
  assert.match(tutorial, /Review the suggested clips/);
  assert.match(tutorial, /Add anything VolleyCut missed/);
  assert.match(tutorial, /The first serve sets who starts serving/);
  assert.match(tutorial, /Save the finished video/);
  assert.doesNotMatch(tutorial, /Choose a suppression policy/);
  assert.doesNotMatch(tutorial, /queue inference/);
});

test("the Classic experience is the only production workflow", async () => {
  const [main, app, editor] = await Promise.all([
    source("main.tsx"),
    source("App.tsx"),
    source("components/CutEditor.tsx"),
  ]);

  assert.match(main, /<App \/>/);
  assert.doesNotMatch(main, /designVariantFromPath/);
  assert.match(main, /window\.history\.replaceState/);
  assert.match(app, /data-design="classic"/);
  assert.match(editor, /data-design="classic"/);
  assert.doesNotMatch(app, /DesignSwitcher|designVariant/);
  assert.doesNotMatch(editor, /designVariant/);
});

test("the analysis screen shows progress for every step", async () => {
  const [app, progressPanel] = await Promise.all([
    source("App.tsx"),
    source("components/InferenceProgressPanel.tsx"),
  ]);

  assert.match(app, /<InferenceProgressPanel/);
  assert.match(progressPanel, /VIDEO ANALYSIS/);
  assert.match(progressPanel, /Time left/);
  assert.match(progressPanel, /Time spent/);
  assert.doesNotMatch(progressPanel, /LOCAL INFERENCE/);
});
