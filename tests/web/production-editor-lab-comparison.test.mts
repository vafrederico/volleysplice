import assert from "node:assert/strict";
import test from "node:test";
import { compareLabDraft, labPaddingSensitivity } from "../../lib/production-editor-lab-comparison.ts";
import { initialLabDraft } from "../../lib/production-editor-lab-draft.ts";
import type { LabConfiguration, ProductionEditorLabTask } from "../../lib/production-editor-lab.ts";

function setup() {
  const human: LabConfiguration = { id: "human-export", label: "Human export", description: "", sourcePolicy: "human",
    events: [{ id: "human:1", parentId: "human:1", start: 10, end: 20 }, { id: "human:2", parentId: "human:2", start: 24, end: 30 }],
    ignoredIntervals: [{ start: 16, end: 18 }], humanReference: { revision: "human", source: "draft", scoreTracking: { serveMarkers: [], sideSwitchMarkers: [] } },
    proposals: [], splits: [], removals: [] };
  const task = { id: "fixture", durationSeconds: 60, sourceRevision: "model", ignoredIntervals: [], configurations: [human] } as unknown as ProductionEditorLabTask;
  return { task, human, draft: initialLabDraft(task, human) };
}

test("human comparison scores actual export and never treats user exclusions as unscored gold time", () => {
  const { task, human, draft } = setup();
  assert.equal(compareLabDraft(task, human, draft)!.F1_padP_coreR, 1);
  const edited = { ...draft, ignoredIntervals: [...draft.ignoredIntervals, { id: "user-remove", start: 11, end: 15, reason: "manual" }] };
  const comparison = compareLabDraft(task, human, edited)!;
  assert.equal(comparison.missedSeconds, 4);
  assert.equal(comparison.recall, 10 / 14);
  assert.equal(comparison.precision, 1);
  assert.equal(task.configurations[0].events[0].start, 10);
});

test("comparison uses strict joining, identical padding and all four sensitivity cases without mutating edits", () => {
  const { task, human } = setup();
  human.events = [{ id: "human:1", parentId: "human:1", start: 10, end: 21 }];
  human.ignoredIntervals = [];
  const model = { ...human, id: "model", humanReference: undefined,
    events: [{ id: "a", parentId: "a", start: 10, end: 12 }, { id: "b", parentId: "b", start: 19, end: 21 }] };
  const draft = initialLabDraft(task, model);
  const before = JSON.stringify(draft);
  const result = compareLabDraft(task, model, draft)!;
  assert.equal(result.model.length, 2, "Exactly three seconds is a cut");
  assert.equal(result.exportSeconds, 12);
  assert.equal(result.humanExportSeconds, 15);
  assert.equal(result.missedSeconds, 3);
  assert.equal(result.recall, 8 / 11);
  const sweep = labPaddingSensitivity(task, model, draft);
  assert.deepEqual(sweep.map(row => row.padding), [0, 1, 2, 3]);
  assert.deepEqual(sweep.map(row => row.comparison!.joinGapSeconds), [3, 3, 3, 3]);
  assert.equal(sweep[0].comparison!.exportSeconds, 4);
  assert.equal(sweep[3].comparison!.model.length, 1);
  assert.equal(JSON.stringify(draft), before);
});
