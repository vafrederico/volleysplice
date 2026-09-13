import assert from "node:assert/strict";
import test from "node:test";
import { createCutDraft, materializeFinalCutIntervals, rallySuppressionDecisionKey } from "../../prod/src/lib/cut-draft.ts";
import { cleanupDecisionsForRally, cleanupReviewDecisions, normalizeCleanupDecisions } from "../../prod/src/designs/taste/cleanup-decisions.ts";
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

test("Keep and Remove update only the selected rally's review decision", () => {
  const decisions = { R001: "excluded" as const, R002: "pending" as const, R003: "pending" as const };
  assert.deepEqual(cleanupDecisionsForRally(decisions, "R001", "kept"), { R001: "kept", R002: "pending", R003: "pending" });
});

test("rallies sharing one suppression event can be removed and restored independently", () => {
  const draft = createCutDraft({ ...seed, rallies: [
    ...seed.rallies, { id: "R002", start: 25, end: 35, confidence: 0.9, included: true },
  ] });
  const suppression = { suggestions: [{ ...suggestion, end: 35 }] };
  const keptIds = (value: typeof draft) => [...new Set(materializeFinalCutIntervals(value, suppression).intervals.flatMap((i) => i.cutIds))];
  assert.deepEqual(keptIds(draft), []);
  const removed = { ...draft, cuts: draft.cuts.map((cut) => cut.id === "R001" ? { ...cut, included: false } : cut),
    userTouchedCutIds: ["R001"], suppressionDecisionOverrides: { [rallySuppressionDecisionKey("R001")]: "suppress" as const } };
  assert.deepEqual(keptIds(removed), []);
  assert.equal(removed.cuts[1].included, true);
  const restored = { ...removed, cuts: draft.cuts,
    suppressionDecisionOverrides: { [suggestion.logicalId]: "suppress" as const, [rallySuppressionDecisionKey("R001")]: "keep" as const } };
  assert.deepEqual(keptIds(restored), ["R001"], "Keep overrides an old shared suppression without keeping the other rally");
  const both = { ...restored, suppressionDecisionOverrides: { ...restored.suppressionDecisionOverrides, [rallySuppressionDecisionKey("R002")]: "keep" as const } };
  assert.deepEqual(keptIds(both), ["R001", "R002"]);
  assert.deepEqual(keptIds({ ...both, cuts: removed.cuts, suppressionDecisionOverrides: {
    ...both.suppressionDecisionOverrides, [rallySuppressionDecisionKey("R001")]: "suppress",
  } }), ["R002"], "Removing one kept rally leaves the other kept");
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

test("reset recomputes pending cleanup from model policy instead of imported review decisions", () => {
  const baseline = createCutDraft(seed);
  const imported = [{ ...suggestion, decision: "suppress" as const }];
  assert.deepEqual(cleanupReviewDecisions(baseline, imported), { R001: "pending" });
  assert.deepEqual(cleanupReviewDecisions({ ...baseline, selectedSuppressionPolicy: "none" }, imported), { R001: "kept" });
});
