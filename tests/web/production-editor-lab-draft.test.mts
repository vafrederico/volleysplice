import assert from "node:assert/strict";
import test from "node:test";
import { humanExportConfiguration } from "../../lib/server/production-editor-lab.ts";
import type { LabelDocument } from "../../lib/annotations.ts";
import {
  currentLabDecision, draftLabEvents, draftWithLabEvents, initialLabDraft, labSeed, labStorageKey, recordLabDecision,
} from "../../lib/production-editor-lab-draft.ts";
import { deriveLabRemovals, restoreLabRemoval, type LabConfiguration, type LabEvent, type ProductionEditorLabTask } from "../../lib/production-editor-lab.ts";
import { materializeFinalCutIntervals, parseCutDraft, rallySuppressionDecisionKey, type CutDraft } from "../../components/production-lab/editor/lib/cut-draft.ts";
import { moveHistory, readEditHistory, recordEdit } from "../../components/production-lab/editor/designs/taste/edit-history.ts";

function fixture() {
  const events: LabEvent[] = [
    { id: "ignored", parentId: "ignored", start: 1, end: 3 },
    { id: "P", parentId: "P", start: 12, end: 18, confidence: .9, serve: { time: 12, side: "near", reason: "Original prediction" } },
    { id: "Q", parentId: "Q", start: 30, end: 40, confidence: .8 },
  ];
  const configuration: LabConfiguration = { id: "boundary-undo", label: "Boundary review", description: "Review", sourcePolicy: "head_refined",
    events, proposals: [], removals: [], splits: [] };
  const task: ProductionEditorLabTask = { schemaVersion: 1, id: "recording-A", name: "A.mp4", durationSeconds: 60,
    mediaUrl: "/video", sourceRevision: "a".repeat(64), ignoredIntervals: [{ start: 0, end: 5 }], configurations: [configuration],
    signals: { times: [0], live: [0], serve: [0], end: [0], keep: [0] }, serving: [], provenance: { labelBlind: true, sourceHashes: {} } };
  return { task, configuration, events };
}

test("human reference preserves exact markers, replay decisions and ignored spans without resetting model drafts", () => {
  const { task, configuration } = fixture();
  const modelKey = labStorageKey(task, configuration);
  const document = { rallies: [{ id: "R008", start: 12, end: 25, tags: [] }],
    ignoredIntervals: [{ start: 0, end: 10, reason: "ignored" }],
    serveMarkers: [{ time: 11.8, rallyId: "R008", side: "far", origin: "manual", ignorePreviousPoint: true }],
    sideSwitches: [{ time: 30, origin: "manual" }] } as unknown as LabelDocument;
  const human = humanExportConfiguration(document, "draft");
  const draft = initialLabDraft(task, human);
  assert.equal(draft.cuts[0].coreStart, 12);
  assert.equal(draft.ignoredIntervals[0].end, 10);
  assert.equal(draft.scoreTracking.serveMarkers[0].timestamp, 11.8);
  assert.equal(draft.scoreTracking.serveMarkers[0].origin, "manual");
  assert.equal(draft.scoreTracking.serveMarkers[0].ignorePreviousPoint, true);
  assert.equal(draft.scoreTracking.serveMarkers[0].rallyId, "human:R008");
  assert.equal(draft.scoreTracking.sideSwitchMarkers[0].timestamp, 30);
  assert.deepEqual(draft.reviewedCutIds, ["human:R008"]);
  const humanKey = labStorageKey(task, human);
  document.rallies[0].end = 26;
  assert.notEqual(humanKey, labStorageKey(task, humanExportConfiguration(document, "draft")));
  assert.equal(modelKey, labStorageKey(task, configuration));
  assert.equal(draft.cuts[0].coreEnd, 25);
});

test("lab persistence isolates recording, model mode and frozen source revision", () => {
  const { task, configuration } = fixture();
  const keys = [labStorageKey(task, configuration), labStorageKey({ ...task, id: "recording-B" }, configuration),
    labStorageKey({ ...task, sourceRevision: "b".repeat(64) }, configuration), labStorageKey(task, { ...configuration, id: "compact-standalone" })];
  assert.equal(new Set(keys).size, 4);
  assert.ok(keys.every(key => key.startsWith("volleycut:production-lab:")));
  const draft = initialLabDraft(task, configuration);
  assert.equal(parseCutDraft(JSON.stringify(draft), labSeed(task, { ...configuration, id: "production" })), null);
});

test("every model starts with pending human decisions, excludes ignored-only rallies and uses only supplied serve predictions", () => {
  const { task, configuration } = fixture();
  const draft = initialLabDraft(task, configuration);
  assert.deepEqual(draft.labDecisions, {});
  assert.deepEqual(draft.reviewedCutIds, []);
  assert.deepEqual(draft.suppressionDecisionOverrides, {});
  assert.deepEqual(draft.cuts.map(cut => cut.id), ["P", "Q"]);
  assert.deepEqual(draft.scoreTracking.serveMarkers.map(marker => [marker.rallyId, marker.side]), [["P", "near"]]);
  assert.equal(currentLabDecision(draft, "removed:P", { start: 10, end: 12 }), undefined);
  assert.equal(draft.cuts.find(cut => cut.id === "Q")!.included, true, "a rejected or missing serve is not permission to discard the rally");
});

test("restoring a trimmed edge preserves the other user-adjusted keep edge, manual footage and excluded rally", () => {
  const { task, configuration, events } = fixture();
  let draft = initialLabDraft(task, configuration);
  draft = { ...draft, cuts: [...draft.cuts.map(cut => cut.id === "P" ? { ...cut, keepEnd: 23.5 } : { ...cut, included: false }),
    { id: "manual", coreStart: 50, coreEnd: 55, keepStart: 49, keepEnd: 56, confidence: 1, included: true, origin: "manual" }],
    reviewedCutIds: ["Q"], scoreTracking: { ...draft.scoreTracking, team1Name: "User team" } };
  const original = { ...events[1], start: 10, serve: { time: 10, side: "far" as const, reason: "Original anchor" } };
  const removal = deriveLabRemovals([original], [events[1]])[0];
  const result = restoreLabRemoval(draftLabEvents(draft, configuration), removal);
  assert.equal(result.ok, true);
  const next = draftWithLabEvents(draft, result.events);
  const changed = next.cuts.find(cut => cut.id === "P")!;
  assert.deepEqual([changed.coreStart, changed.keepStart, changed.coreEnd, changed.keepEnd], [10, 8, 18, 23.5]);
  assert.deepEqual(next.cuts.find(cut => cut.id === "manual"), draft.cuts.find(cut => cut.id === "manual"));
  assert.deepEqual(next.cuts.find(cut => cut.id === "Q"), draft.cuts.find(cut => cut.id === "Q"));
  assert.equal(next.scoreTracking.team1Name, "User team");
  assert.ok(parseCutDraft(JSON.stringify(next), labSeed(task, configuration)), "the edit remains a valid persistent production draft");

  const splitParent: LabEvent = { id: "parent", parentId: "parent", start: 10, end: 30 };
  const splitChildren: LabEvent[] = [{ id: "left", parentId: "parent", start: 12, end: 18 }, { id: "right", parentId: "parent", start: 20, end: 28 }];
  const splitConfig = { ...configuration, events: splitChildren, proposals: [{ id: "split", parentId: "parent", before: [splitParent], after: splitChildren, reason: "Split" }],
    removals: deriveLabRemovals([splitParent], splitChildren) };
  const splitDraft = initialLabDraft(task, splitConfig);
  splitDraft.cuts.find(cut => cut.id === "right")!.included = false;
  splitDraft.cuts.push({ id: "manual-replacement", coreStart: 20, coreEnd: 25, keepStart: 20, keepEnd: 25, confidence: 1, included: true, origin: "manual" });
  const gap = splitConfig.removals.find(removal => removal.kind === "gap")!;
  const restore = restoreLabRemoval(draftLabEvents(splitDraft, splitConfig), gap);
  assert.equal(restore.ok, false, "an unrelated manual rally at an old model endpoint must not be mistaken for the linked model identity and merged away");
});

test("new and moved serve anchors require review and previously deleted model markers stay deleted", () => {
  const { task, configuration } = fixture();
  const draft = initialLabDraft(task, configuration);
  draft.scoreTracking.removedModelMarkerIds = ["lab-serve:Q"];
  const events = draftLabEvents(draft, configuration).map(event => event.id === "P"
    ? { ...event, start: 13, serve: { time: 13, side: "review" as const, reason: "New anchor" } }
    : { ...event, serve: { time: event.start, side: "review" as const, reason: "New anchor" } });
  events.push({ id: "new", parentId: "new", start: 45, end: 48, confidence: 0,
    serve: { time: 45, side: "review", reason: "New model anchor" } });
  const next = draftWithLabEvents(draft, events);
  assert.deepEqual(next.scoreTracking.serveMarkers.map(marker => [marker.id, marker.timestamp, marker.side]), [["lab-serve:P", 13, "review"], ["lab-serve:new", 45, "review"]]);
  assert.deepEqual(next.scoreTracking.removedModelMarkerIds, ["lab-serve:Q"]);
});

test("review decisions and geometry survive persistence, undo and redo together, and edited geometry reopens stale decisions", () => {
  const { task, configuration } = fixture();
  const initial = initialLabDraft(task, configuration);
  const range = { start: 10, end: 12 };
  const events = draftLabEvents(initial, configuration).map(event => event.id === "P" ? { ...event, start: 10,
    serve: { time: 10, side: "review" as const, reason: "Restored" } } : event);
  const changed = recordLabDecision(draftWithLabEvents(initial, events), "removal:P", range, "Footage restored");
  const history = recordEdit({ past: [], present: initial, future: [] }, changed);
  const persisted = parseCutDraft(JSON.stringify(changed), labSeed(task, configuration));
  assert.ok(persisted);
  assert.equal(currentLabDecision(persisted, "removal:P", range), "Footage restored");
  const reloaded = readEditHistory(JSON.stringify({ version: 1, ...history }), persisted, labSeed(task, configuration));
  const undone = moveHistory(reloaded, "undo");
  assert.equal(undone.present.cuts[0].coreStart, 12);
  assert.equal(currentLabDecision(undone.present, "removal:P", range), undefined);
  const redone = moveHistory(undone, "redo");
  assert.equal(redone.present.cuts[0].coreStart, 10);
  assert.equal(currentLabDecision(redone.present, "removal:P", range), "Footage restored");
  const moved: CutDraft = { ...redone.present, cuts: redone.present.cuts.map(cut => cut.id === "P" ? { ...cut, coreStart: 11 } : cut) };
  assert.equal(currentLabDecision(moved, "removal:P", range), undefined);
});

test("native suppression retains its joining barrier until a human keeps the suppressed rally", () => {
  const { task, configuration } = fixture();
  const production: LabEvent[] = [{ id: "A", parentId: "A", start: 10, end: 14 }, { id: "B", parentId: "B", start: 16, end: 17 },
    { id: "C", parentId: "C", start: 19, end: 22 }];
  const suppression: NonNullable<LabConfiguration["suppression"]> = { identicalPolicyResults: true,
    suggestions: [{ id: "s", logicalId: "s", suppressionEventId: "s", start: 16, end: 17, score: .99, sourceProductionIds: ["B"], eligiblePolicyIds: ["aggressive"] }] };
  const config: LabConfiguration = { ...configuration, id: "suppression-aggressive", events: [production[0], production[2]], draftEvents: production,
    suppressionPolicy: "aggressive", suppression };
  const draft = initialLabDraft(task, config);
  assert.equal(draft.cuts.length, 3, "native suppression needs the original retained candidate identities");
  const intervals = (value: CutDraft) => materializeFinalCutIntervals(value, suppression).intervals.map(({ start, end }) => [start, end]);
  assert.deepEqual(intervals(draft), [[8, 16], [17, 24]], "the suppressed middle rally prevents the short export gap from being joined");
  const kept = { ...draft, suppressionDecisionOverrides: { [rallySuppressionDecisionKey("B")]: "keep" as const } };
  assert.deepEqual(intervals(kept), [[8, 24]], "the human keep restores the rally and removes its suppression barrier");
  const history = recordEdit({ past: [], present: draft, future: [] }, kept);
  assert.deepEqual(intervals(moveHistory(history, "undo").present), [[8, 16], [17, 24]]);
});
