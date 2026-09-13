import assert from "node:assert/strict";
import test from "node:test";
import { createCutDraft, type CutDraftSeed } from "../../prod/src/lib/cut-draft.ts";
import { editFingerprint, editHistoryKey, moveHistory, readEditHistory, recordEdit, writeEditHistory } from "../../prod/src/designs/taste/edit-history.ts";

const seed: CutDraftSeed = {
  analysisId: "history-analysis", recordingId: "history-recording", duration: 60,
  rallies: [{ id: "R001", start: 5, end: 15, confidence: 0.8, included: true }],
  ignoredIntervals: [],
};
const initial = () => readEditHistory(null, createCutDraft(seed), seed);

test("history retains exactly 30 edits and supports undo and redo through the boundary", () => {
  let history = initial();
  for (let i = 1; i <= 40; i++) history = recordEdit(history, {
    ...history.present, scoreTracking: { ...history.present.scoreTracking, team1Name: `Team ${i}` },
  });
  assert.equal(history.past.length, 30);
  for (let i = 0; i < 30; i++) history = moveHistory(history, "undo");
  assert.equal(history.present.scoreTracking.team1Name, "Team 10");
  assert.equal(moveHistory(history, "undo"), history);
  assert.equal(history.future.length, 30);
  for (let i = 0; i < 30; i++) history = moveHistory(history, "redo");
  assert.equal(history.present.scoreTracking.team1Name, "Team 40");
  assert.equal(moveHistory(history, "redo"), history);
});

test("playback and save timestamps do not create edits or clear redo, but a new edit does", () => {
  const start = initial();
  const edited = recordEdit(start, { ...start.present, beforePaddingSeconds: 1 });
  const undone = moveHistory(edited, "undo");
  assert.equal(recordEdit(undone, { ...undone.present, playbackRate: 4, cutPreviewEnabled: true, updatedAt: "later" }), undone);
  const branch = recordEdit(undone, { ...undone.present, afterPaddingSeconds: 1 });
  assert.equal(branch.future.length, 0);
  assert.equal(branch.past.length, 1);
});

test("an edit restores rally decisions, ranges, marker deletions, and pending boundaries atomically", () => {
  const start = initial();
  const next = {
    ...start.present,
    cuts: start.present.cuts.map((cut) => ({ ...cut, included: false })),
    reviewedCutIds: ["R001"], userTouchedCutIds: ["R001"],
    pendingManualStart: 20,
    ignoredIntervals: [{ id: "I001", start: 30, end: 40, reason: "non-game-content" }],
    suppressionDecisionOverrides: { cleanup: "suppress" as const },
    scoreTracking: { ...start.present.scoreTracking, removedModelMarkerIds: ["model-serve"] },
  };
  const history = recordEdit(start, next);
  assert.equal(history.past.length, 1);
  const undone = moveHistory(history, "undo");
  assert.deepEqual(undone.present, start.present);
  assert.deepEqual(moveHistory(undone, "redo").present, next);
});

test("local storage round trip preserves both stacks across reloads", () => {
  let history = initial();
  history = recordEdit(history, { ...history.present, pendingManualStart: 20 });
  history = recordEdit(history, { ...history.present, pendingManualStart: null,
    cuts: [...history.present.cuts, { id: "M01", coreStart: 20, coreEnd: 25, keepStart: 17, keepEnd: 28, included: true, confidence: 1, origin: "manual" }],
  });
  history = moveHistory(history, "undo");
  const stored = new Map<string, string>();
  const storage = { setItem: (key: string, value: string) => { stored.set(key, value); } };
  const key = editHistoryKey("project-one");
  assert.equal(writeEditHistory(storage, key, history), true);
  const reloaded = readEditHistory(stored.get(key)!, { ...history.present, updatedAt: "saved later", playbackRate: 8 }, seed);
  assert.equal(reloaded.past.length, 1);
  assert.equal(reloaded.future.length, 1);
  assert.equal(moveHistory(reloaded, "redo").present.cuts.at(-1)?.id, "M01");
  assert.notEqual(editHistoryKey("project-two"), key);
  assert.equal(readEditHistory(stored.get(editHistoryKey("project-two")) ?? null, history.present, seed).past.length, 0);
});

test("malformed, stale, mismatched, or oversized saved histories are safely discarded", () => {
  const start = initial();
  const history = recordEdit(start, { ...start.present, pendingIgnoreStart: 20 });
  for (const value of ["broken", "null", "[]", JSON.stringify({ version: 99 }),
    JSON.stringify({ version: 1, ...history, past: [null] }),
    JSON.stringify({ version: 1, ...history, past: Array(31).fill(start.present) }),
    JSON.stringify({ version: 1, ...history, present: { ...history.present, sourceRevision: "stale" } }),
    JSON.stringify({ version: 1, ...history, future: [{ ...start.present, cuts: [{ id: "invalid" }] }] }),
  ]) assert.deepEqual(readEditHistory(value, history.present, seed), { past: [], present: history.present, future: [] });
  assert.equal(readEditHistory(JSON.stringify({ version: 1, ...history }), start.present, seed).past.length, 0);
  assert.equal(editFingerprint({ ...start.present, recordingId: "other" }) === editFingerprint(start.present), false);
});

test("unavailable storage preserves usable in-memory history", () => {
  const history = initial();
  assert.equal(writeEditHistory({ setItem: () => { throw new Error("Quota exceeded"); } }, "history", history), false);
  assert.equal(recordEdit(history, { ...history.present, pendingManualStart: 0 }).past.length, 1);
});
