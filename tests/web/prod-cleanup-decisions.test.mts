import assert from "node:assert/strict";
import test from "node:test";
import { createCutDraft, materializeFinalCutIntervals } from "../../prod/src/lib/cut-draft.ts";
import { cleanupDecisionsForRally, normalizeCleanupDecisions } from "../../prod/src/designs/taste/cleanup-decisions.ts";
import { readEditHistory, recordEdit, moveHistory } from "../../prod/src/designs/taste/edit-history.ts";
import type { SuppressionSuggestion } from "../../prod/src/lib/on-device/suppression-policy.ts";

const seed = { analysisId: "cleanup-regression", recordingId: "recording", duration: 60,
  rallies: [{ id: "R001", start: 5, end: 15, confidence: 0.9, included: true }], ignoredIntervals: [] };
const suggestion: SuppressionSuggestion = { id: "fragment-5-15", logicalId: "shared-event", suppressionEventId: "event",
  start: 5, end: 15, score: 0.9, sourceProductionIds: ["R001"], eligiblePolicyIds: ["aggressive"] };

test("Remove then Keep restores a suppressed rally when fragment and logical IDs differ", () => {
  const initial = createCutDraft(seed);
  const suppression = { suggestions: [suggestion] };
  assert.equal(materializeFinalCutIntervals(initial, suppression).intervals.length, 0);
  const removed = { ...initial, cuts: initial.cuts.map((cut) => ({ ...cut, included: false })),
    suppressionDecisionOverrides: { [suggestion.logicalId]: "suppress" as const } };
  assert.equal(materializeFinalCutIntervals(removed, suppression).intervals.length, 0);
  const restored = normalizeCleanupDecisions({ ...removed, cuts: initial.cuts,
    suppressionDecisionOverrides: { ...removed.suppressionDecisionOverrides, [suggestion.id]: "keep" } }, [suggestion]);
  assert.deepEqual(restored.suppressionDecisionOverrides, { [suggestion.logicalId]: "keep" });
  assert.deepEqual(materializeFinalCutIntervals(restored, suppression).intervals.flatMap((i) => i.cutIds), ["R001"]);
});

test("shared cleanup decisions cannot be overwritten by another fragment's pending state", () => {
  const suggestions = [{ ...suggestion, cutId: "R001" }, { ...suggestion, id: "fragment-two", cutId: "R002" },
    { ...suggestion, id: "other", logicalId: "unrelated", cutId: "R003" }];
  const kept = cleanupDecisionsForRally({ R001: "excluded", R002: "pending", R003: "pending" }, suggestions, "R001", "kept");
  assert.deepEqual(kept, { R001: "kept", R002: "kept", R003: "pending" });
});

test("legacy saved history migrates decisions without losing undo or redo", () => {
  const initial = createCutDraft(seed);
  let history = readEditHistory(null, initial, seed);
  history = recordEdit(history, { ...initial, suppressionDecisionOverrides: { [suggestion.id]: "suppress" } });
  history = recordEdit(history, { ...initial, suppressionDecisionOverrides: { [suggestion.id]: "keep" } });
  const normalize = (draft: typeof initial) => normalizeCleanupDecisions(draft, [suggestion]);
  const reloaded = readEditHistory(JSON.stringify({ version: 1, ...history }), normalize(history.present), seed, normalize);
  assert.equal(reloaded.past.length, 2);
  const undone = moveHistory(reloaded, "undo");
  assert.deepEqual(undone.present.suppressionDecisionOverrides, { [suggestion.logicalId]: "suppress" });
  assert.deepEqual(moveHistory(undone, "redo").present.suppressionDecisionOverrides, { [suggestion.logicalId]: "keep" });
});
