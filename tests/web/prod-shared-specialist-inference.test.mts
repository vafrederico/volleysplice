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
    source("designs/taste/index.tsx"),
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
  assert.match(app, /scoreRecovery/);
  assert.match(editor, /onClick=\{recovery\.retry\}/);
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
  // New projects always generate serve markers; the legacy preference is retained only for storage compatibility.
  assert.match(app, /servingSideEnabled:\s*true/);
  assert.match(app, /missingScoreStages\(project\)/);
  assert.match(app, /runScoreStages\(checkpoint, opened, abort\)/);
  assert.match(editor, /Scores are derived from the ordered serve markers/);
  assert.match(editor, /Save project/);
  assert.doesNotMatch(editor, /downloadEditList|Save a review copy/);
  assert.match(editor, /state\.addSideSwitchAtPlayhead/);
  assert.match(editor, /Team-to-court mapping changes from the next serve/);
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
    source("designs/taste/index.tsx"),
    source("components/GuidedTour.tsx"),
  ]);

  assert.match(app, /<GuidedTour stage="source"/);
  assert.match(editor, /<RallyDeskGuidedTour/);
  assert.match(tutorial, /Restart tutorial/);
  assert.match(tutorial, /Choose your video/);
  assert.match(tutorial, /Start with what needs attention/);
  assert.match(tutorial, /Add a rally VolleySplice missed/);
  assert.match(tutorial, /Check serves and side switches/);
  assert.match(tutorial, /Export when the review is ready/);
  assert.doesNotMatch(tutorial, /Choose a suppression policy/);
  assert.doesNotMatch(tutorial, /queue inference/);
});

test("Rally Desk is the production review workspace", async () => {
  const [main, app, editor] = await Promise.all([
    source("main.tsx"),
    source("App.tsx"),
    source("designs/taste/index.tsx"),
  ]);

  assert.match(main, /<App \/>/);
  assert.doesNotMatch(main, /designVariantFromPath/);
  assert.match(main, /window\.history\.replaceState/);
  assert.match(app, /data-design="classic"/);
  assert.match(app, /<RallyDesk review=\{review\}/);
  assert.match(editor, /export function RallyDesk/);
  assert.match(editor, /taste-rally-desk/);
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
