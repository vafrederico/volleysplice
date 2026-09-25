import assert from "node:assert/strict";
import test from "node:test";
import { withLabSuppression } from "../../lib/production-editor-lab-suppression.ts";
import { initialLabDraft, labStorageKey } from "../../lib/production-editor-lab-draft.ts";
import type { LabConfiguration, ProductionEditorLabTask } from "../../lib/production-editor-lab.ts";
import { materializeFinalCutIntervals, rallySuppressionDecisionKey } from "../../components/production-lab/editor/lib/cut-draft.ts";
import { parseEditorSuppression } from "../../lib/server/editor-lab-suppression.ts";
import type { LabelDocument } from "../../lib/annotations.ts";

function fixture() {
  const base: LabConfiguration = { id: "neural-mobile-tcn", label: "Mobile", description: "Frozen", sourcePolicy: "frozen",
    events: [{ id: "A", parentId: "A", start: 10, end: 20, serve: { time: 10, side: "near", reason: "Prediction" } },
      { id: "B", parentId: "B", start: 30, end: 40 }, { id: "C", parentId: "C", start: 50, end: 60 }],
    proposals: [], removals: [], splits: [] };
  const task: ProductionEditorLabTask = { schemaVersion: 1, id: "recording-A", name: "Synthetic", durationSeconds: 80,
    mediaUrl: "/video", sourceRevision: "a".repeat(64), ignoredIntervals: [{ start: 10, end: 12 }], configurations: [base],
    signals: { times: [], live: [], serve: [], end: [], keep: [] }, serving: [], provenance: { labelBlind: true, sourceHashes: {} },
    suppressionSource: { revision: "b".repeat(64), decoded: [{ start: 10, end: 11, score: .9 }, { start: 32, end: 35, score: .8 }, { start: 52, end: 55, score: .9 }],
      gated: { identicalPolicyResults: false, suggestions: [{ start: 32, end: 35, score: .8, id: "S", logicalId: "S", suppressionEventId: "S",
        sourceProductionIds: ["PP1"], eligiblePolicyIds: ["aggressive"] }] } } };
  return { task, base };
}

test("model combination preserves original boundaries, isolates drafts and obeys gates and ignored evidence", () => {
  const { task, base } = fixture();
  assert.equal(withLabSuppression(task, base, "none"), base);
  const conservative = withLabSuppression(task, base, "conservative");
  const aggressive = withLabSuppression(task, base, "aggressive");
  const direct = withLabSuppression(task, base, "direct");
  assert.deepEqual(conservative.events, base.events);
  assert.deepEqual(aggressive.events.map(e => e.id), ["A", "C"]);
  assert.deepEqual(direct.events.map(e => e.id), ["A"]);
  assert.deepEqual(direct.draftEvents, base.events);
  assert.deepEqual(direct.removals.map(r => [r.start, r.end]), [[30, 40], [50, 60]]);
  assert.equal(new Set([base, conservative, aggressive, direct].map(c => labStorageKey(task, c))).size, 4);
  assert.equal(base.events.length, 3);
  const draft = initialLabDraft(task, direct);
  assert.deepEqual(draft.scoreTracking.serveMarkers.map(m => m.rallyId), ["A"]);
  assert.deepEqual([...new Set(materializeFinalCutIntervals(draft, direct.suppression).intervals.flatMap(r => r.cutIds))], ["A"]);
  const restored = { ...draft, suppressionDecisionOverrides: { [rallySuppressionDecisionKey("B")]: "keep" as const } };
  assert.deepEqual([...new Set(materializeFinalCutIntervals(restored, direct.suppression).intervals.flatMap(r => r.cutIds))], ["A", "B"]);
  assert.deepEqual([...new Set(materializeFinalCutIntervals(draft, direct.suppression).intervals.flatMap(r => r.cutIds))], ["A"], "undo restores the removal");
});

test("missing evidence, human reference, existing suppression and production head-only fail closed", () => {
  const { task, base } = fixture();
  for (const config of [{ ...base, id: "production" }, { ...base, suppressionPolicy: "aggressive" as const },
    { ...base, humanReference: { revision: "human", source: "draft" as const, scoreTracking: { serveMarkers: [], sideSwitchMarkers: [] } } }]) {
    assert.equal(withLabSuppression(task, config, "direct"), config);
  }
  assert.equal(withLabSuppression({ ...task, suppressionSource: undefined }, base, "aggressive"), base);
});

test("historical production uses its own paired gate while comparison models use the study replay", () => {
  const { task, base } = fixture();
  const old = { ...base, id: "suppression-aggressive", suppression: { identicalPolicyResults: true, suggestions: [] } };
  task.configurations.push(old);
  const production = { ...base, id: "production" };
  assert.equal(withLabSuppression(task, production, "aggressive").removals.length, 0);
  assert.equal(withLabSuppression(task, { ...production, id: "production-replay" }, "aggressive").removals.length, 1);
  assert.equal(withLabSuppression(task, base, "aggressive").removals.length, 1);
});

test("server evidence binds recording identity and rejects malformed values; private fields never reach client", () => {
  const { task, base } = fixture();
  const recording = { id: task.id, contentSha256: "c".repeat(64), durationSeconds: 80 } as LabelDocument["recording"];
  const row = { recordingId: task.id, contentSha256: recording.contentSha256, durationSeconds: 80,
    ...task.suppressionSource, productionEvents: base.events.map(e => ({ ...e, confidence: .8 })), privateSource: "not-public" };
  const doc = { schemaVersion: 1, kind: "volleycut-editor-suppression-index", labelsUsed: false, records: [row] };
  const parsed = parseEditorSuppression(doc, recording)!;
  assert.equal(JSON.stringify(parsed).includes("not-public"), false);
  assert.equal(parsed.production.events.length, 3);
  assert.equal(parseEditorSuppression(doc, { ...recording, id: "missing" }), null);
  assert.throws(() => parseEditorSuppression(doc, { ...recording, contentSha256: "wrong" }));
  assert.throws(() => parseEditorSuppression({ ...doc, labelsUsed: true }, recording));
  assert.throws(() => parseEditorSuppression({ ...doc, records: [row, row] }, recording));
  assert.throws(() => parseEditorSuppression({ ...doc, records: [{ ...row, decoded: [{ start: 1, end: 3, score: NaN }] }] }, recording));
});
