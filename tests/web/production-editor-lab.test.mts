import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { applyPaddingToCachedCuts, createCutDraft, materializeFinalCutIntervals } from "../../components/production-lab/editor/lib/cut-draft.ts";
import {
  applyLabProposal, deriveLabRemovals, deriveLabSplits, parseProductionEditorLabManifest,
  restoreLabRemoval, undoLabSplit, type LabEvent,
} from "../../lib/production-editor-lab.ts";

const event = (id: string, start: number, end: number, parentId = "P"): LabEvent => ({ id, start, end, parentId });
const parent = { ...event("P", 10, 30), serve: { time: 10, side: "near" as const, reason: "Original anchor" } };
const children = [event("a", 12, 18), event("b", 20, 28)];

test("raw boundary removal queue keeps prefix, internal gap and suffix separate regardless of export padding", () => {
  const removals = deriveLabRemovals([parent], children);
  assert.deepEqual(removals.map(({ start, end, kind }) => [start, end, kind]), [[10, 12, "prefix"], [18, 20, "gap"], [28, 30, "suffix"]]);
  assert.deepEqual(deriveLabSplits(children).map(({ start, end }) => [start, end]), [[18, 20]]);
  assert.deepEqual(deriveLabRemovals([parent], children, [{ start: 0, end: 11 }, { start: 29, end: 40 }])
    .map(({ start, end }) => [start, end]), [[11, 12], [18, 20], [28, 29]]);
});

test("individual removal restoration preserves unrelated user edits and can be performed in any order", () => {
  const removals = deriveLabRemovals([parent], children);
  const unrelated = event("other", 40, 50, "other");
  let current = [...children, unrelated];
  for (const index of [2, 0, 1]) {
    const result = restoreLabRemoval(current, removals[index]);
    assert.equal(result.ok, true);
    current = result.events;
  }
  assert.deepEqual(current.map(({ id, start, end }) => [id, start, end]), [["a", 10, 30], ["other", 40, 50]]);
  assert.equal(current[0].serve?.side, "near", "restored original anchor may reuse its own known side");
  assert.strictEqual(current[1], unrelated);
  const edited = [{ ...children[0], end: 17 }, children[1]];
  const prefixOnly = restoreLabRemoval(edited, removals[0]);
  assert.equal(prefixOnly.ok, true);
  assert.equal(prefixOnly.events[0].end, 17, "restoring prefix must preserve manually edited end");
});

test("restoration refuses stale adjacent boundaries and collisions instead of overwriting manual changes", () => {
  const [prefix, gap] = deriveLabRemovals([parent], children);
  const edited = [{ ...children[0], start: 13 }, children[1]];
  const stale = restoreLabRemoval(edited, prefix);
  assert.equal(stale.ok, false);
  assert.strictEqual(stale.events, edited);
  assert.equal(restoreLabRemoval([children[0], { ...children[1], start: 21 }], gap).ok, false);
  assert.equal(restoreLabRemoval([...children, event("manual", 10, 11, "manual")], prefix).ok, false);
});

test("split undo only merges the two adjacent identities from its parent, including a zero export gap", () => {
  const split = deriveLabSplits(children)[0];
  const other = event("adjacent-other-parent", 30, 40, "Q");
  const result = undoLabSplit([...children, other], split);
  assert.equal(result.ok, true);
  assert.deepEqual(result.events.map(({ id, start, end }) => [id, start, end]), [["a", 12, 28], ["adjacent-other-parent", 30, 40]]);
  const touching = [event("a", 12, 18), event("b", 18, 28)];
  assert.equal(undoLabSplit(touching, deriveLabSplits(touching)[0]).events.length, 1);
});

test("restoration cannot bridge an ignored span, including a wholly suppressed parent", () => {
  const removals = deriveLabRemovals([parent], children, [{ start: 19, end: 19.5 }]);
  let current = children;
  for (const removal of removals) {
    const result = restoreLabRemoval(current, removal);
    assert.equal(result.ok, true);
    current = result.events;
  }
  assert.deepEqual(current.map(({ start, end }) => [start, end]), [[10, 19], [19.5, 30]]);
  let empty: LabEvent[] = [];
  for (const removal of deriveLabRemovals([parent], [], [{ start: 18, end: 20 }])) {
    const result = restoreLabRemoval(empty, removal);
    assert.equal(result.ok, true);
    empty = result.events;
  }
  assert.deepEqual(empty.map(({ start, end }) => [start, end]), [[10, 18], [20, 30]]);
  assert.equal(empty[1].serve?.side, "review");
});

test("guidance applies only to its untouched parent and never changes another rally", () => {
  const other = event("other", 40, 50, "Q");
  const proposal = { id: "proposal:P", parentId: "P", before: [parent], after: children, reason: "Split" };
  assert.deepEqual(applyLabProposal([parent, other], proposal).events, [...children, other]);
  assert.equal(applyLabProposal([{ ...parent, end: 29 }, other], proposal).ok, false);
});

function fixture() {
  const recording = { id: "recording", videoFilename: "recording.mp4", durationSeconds: 60, contentSha256: "a".repeat(64) };
  return { schemaVersion: 1, kind: "volleycut-production-editor-lab", recording, ignoredIntervals: [],
    productionEvents: [{ ...parent, id: "P", confidence: .9, agreement: "both-models" }, event("Q", 40, 50, "Q")],
    policies: { none: { core: [parent, event("Q", 40, 50, "Q")] }, conservative: { core: [parent] }, balanced: { core: [parent] }, aggressive: { core: [parent] } },
    suppression: { identicalPolicyResults: true, suggestions: [{ id: "suppression", logicalId: "logical", suppressionEventId: "S", start: 40, end: 45, score: .9,
      sourceProductionIds: ["v2-1"], eligiblePolicyIds: ["conservative", "balanced", "aggressive"] }] },
    boundaryEvents: [event("a", 10, 18), event("b", 20, 28)], compactEvents: [event("compact1", 11, 29, "compact1")],
    serving: { candidates: [{ id: "P", anchor: 10, nearProbability: .9, side: "near", verdict: "near", serveDecisionSource: "serve-head", reviewReasons: [] },
      { id: "Q", anchor: 40, nearProbability: .9, side: "near", verdict: "not-serve", serveDecisionSource: "serve-head", reviewReasons: [] }] },
    signals: { times: [1, 2], live: [.2, .3], serve: [.4, .5], end: [.6, .7], keep: [.8, .9] },
    provenance: { labelBlind: true, sourceHashes: { "production.json": "a".repeat(64) } },
    humanOracleAnswers: [{ start: 1, end: 60 }], privatePath: "/private-do-not-expose" };
}

test("sandbox identity and native signals are validated, and public DTO cannot include unknown labels or private paths", () => {
  const source = fixture();
  const task = parseProductionEditorLabManifest(source, source.recording, "revision")!;
  assert.equal(task.configurations.length, 6);
  assert.equal(task.configurations.some(row => row.id === "suppression-balanced"), false);
  assert.equal(task.sourceRevision, "revision");
  assert.equal(JSON.stringify(task).includes("humanOracleAnswers"), false);
  assert.equal(JSON.stringify(task).includes("private-do-not-expose"), false);
  assert.equal(task.configurations.find(row => row.id === "production")!.events[1].serve, undefined, "rejected serve gates create no anchor");
  const boundary = task.configurations.find(row => row.id === "boundary-undo")!;
  assert.equal(boundary.events[0].serve?.side, "near");
  assert.equal(boundary.events[1].serve?.side, "review", "new start cannot inherit parent's serving-side answer");
  assert.equal(task.configurations.find(row => row.id === "compact-standalone")!.events[0].confidence, undefined, "do not invent compact confidence");
  assert.equal(task.configurations.find(row => row.id === "compact-standalone")!.events[0].serve?.side, "review");
  assert.equal(parseProductionEditorLabManifest(source, { ...source.recording, id: "another" }, "revision"), null);
  assert.throws(() => parseProductionEditorLabManifest(source, { ...source.recording, contentSha256: "b".repeat(64) }, "revision"), /identity/);
  source.signals.live.pop();
  assert.throws(() => parseProductionEditorLabManifest(source, source.recording, "revision"), /Unaligned/);
});

const nativeManifest = process.env.VOLLEYCUT_EDITOR_LAB_MANIFEST_PATH;
test("neural serve predictions bind to their own starts, preserve rejection, and reject stale anchors", () => {
  const source = fixture();
  const prediction = (id: string, anchor: number, verdict: string) => ({ id, anchor, nearProbability: .9, side: "near", verdict, reviewReasons: [], serveDecisionSource: "serve-head" });
  const neuralServing = { boundary: { candidates: [prediction("a", 10, "near"), prediction("b", 20, "not-serve")] },
    compact: { candidates: [prediction("compact1", 11, "far")] } };
  neuralServing.compact.candidates[0].side = "far";
  const task = parseProductionEditorLabManifest({ ...source, neuralServing }, source.recording, "neural")!;
  assert.equal(task.configurations.find(row => row.id === "boundary-undo")!.events[0].serve?.side, "near");
  assert.equal(task.configurations.find(row => row.id === "boundary-undo")!.events[1].serve, undefined);
  assert.equal(task.configurations.find(row => row.id === "compact-standalone")!.events[0].serve?.side, "far");
  neuralServing.compact.candidates[0].anchor = 12;
  assert.throws(() => parseProductionEditorLabManifest({ ...source, neuralServing }, source.recording, "bad"), /geometry/);
});
test("frozen PXL model input exposes exactly the expected real configurations and label-blind 60-removal queue", { skip: !nativeManifest }, async () => {
  const source = JSON.parse(await readFile(nativeManifest!, "utf8"));
  const task = parseProductionEditorLabManifest(source, source.recording, "frozen-test")!;
  assert.deepEqual(task.configurations.map(row => [row.id, row.events.length, row.removals.length, row.splits.length]), [
    ["production", 59, 0, 0], ["suppression-conservative", 50, 6, 0], ["suppression-aggressive", 44, 12, 0],
    ["compact-guidance", 44, 0, 0], ["boundary-undo", 46, 60, 2], ["compact-standalone", 34, 0, 0],
  ]);
  assert.equal(task.signals.times.length, 4245);
  assert.equal(task.serving.length, 59);
  assert.equal(task.configurations.find(row => row.id === "compact-guidance")!.proposals.length, 27);
  assert.equal(task.configurations.find(row => row.id === "compact-guidance")!.proposals.filter(row => row.recommended).length, 9);
  const boundary = task.configurations.find(row => row.id === "boundary-undo")!;
  let restored = boundary.events;
  for (const removal of boundary.removals.toReversed()) {
    const result = restoreLabRemoval(restored, removal);
    assert.equal(result.ok, true, `could not restore ${removal.id}`);
    restored = result.events;
  }
  assert.deepEqual(restored.map(({ start, end }) => [start, end]), task.configurations.find(row => row.id === "compact-guidance")!.events.map(({ start, end }) => [start, end]));
});

test("copied production materializer reproduces frozen suppression exports at every required padding", { skip: !nativeManifest }, async () => {
  const source = JSON.parse(await readFile(nativeManifest!, "utf8"));
  const replay = JSON.parse(await readFile(path.resolve(path.dirname(nativeManifest!), "../production-replay.json"), "utf8"));
  const task = parseProductionEditorLabManifest(source, source.recording, "frozen-test")!;
  for (const config of task.configurations.filter(row => row.id === "production" || row.suppression)) {
    const seed = createCutDraft({ analysisId: "test", recordingId: task.id, duration: task.durationSeconds,
      rallies: (config.draftEvents ?? config.events).map(event => ({ ...event, confidence: event.confidence ?? 0, included: true })),
      ignoredIntervals: task.ignoredIntervals.map(interval => ({ ...interval, reason: "non-game-content" })),
    });
    seed.selectedSuppressionPolicy = config.suppressionPolicy ?? "none";
    for (const padding of [0, 1, 2, 3]) {
      const draft = applyPaddingToCachedCuts(seed, padding, padding, task.durationSeconds);
      const actual = materializeFinalCutIntervals(draft, config.suppression).intervals.map(({ start, end }) => ({ start, end }));
      const expected = replay.variants[config.suppressionPolicy ?? "none"].exactExportsByPadding[String(padding)];
      assert.deepEqual(actual, expected, `${config.id}: padding ${padding}`);
    }
  }
});
