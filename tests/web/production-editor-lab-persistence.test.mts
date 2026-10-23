import assert from "node:assert/strict";
import test from "node:test";
import { createCutDraft } from "../../components/production-lab/editor/lib/cut-draft.ts";
import { editHistoryKey, recordEdit, writeEditHistory, readEditHistory } from "../../components/production-lab/editor/designs/taste/edit-history.ts";
import { restoreLabDraft, startNewLabReview } from "../../lib/production-editor-lab-persistence.ts";

const key = "lab:synthetic-mode";
const seed = { analysisId: key, recordingId: "synthetic", duration: 60, rallies: [
  { id: "R1", start: 10, end: 20, confidence: 0.9, included: true },
], ignoredIntervals: [] };
const model = createCutDraft(seed);
function storage() {
  const values = new Map<string, string>();
  return { values, getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => { values.set(key, value); },
    removeItem: (key: string) => { values.delete(key); } };
}

test("unreadable, empty, future-version and mismatched lab drafts block restore without writes", () => {
  for (const raw of ["{broken", "", JSON.stringify({ ...model, version: 999 }), JSON.stringify({ ...model, analysisId: "other-mode" })]) {
    const store = storage(); store.setItem(key, raw); store.setItem(editHistoryKey(key), "history bytes");
    const before = [...store.values];
    const result = restoreLabDraft(store, key, seed, model);
    assert.equal(result.status, "blocked");
    if (result.status === "blocked") assert.equal(result.raw, raw);
    assert.deepEqual([...store.values], before);
  }
});

test("unavailable storage blocks restoration rather than initializing an autosaving editor", () => {
  const store = storage(); store.getItem = () => { throw new Error("denied"); };
  assert.equal(restoreLabDraft(store, key, seed, model).status, "blocked");
  assert.equal(store.values.size, 0);
});

test("absent and valid edited drafts remain usable; missing model markers migrate undo history", () => {
  const store = storage();
  assert.deepEqual(restoreLabDraft(store, key, seed, model), { status: "ready", draft: model });
  const edited = { ...model, cuts: model.cuts.map(cut => ({ ...cut, included: false })) };
  store.setItem(key, JSON.stringify(edited));
  const history = recordEdit({ past: [], present: model, future: [] }, edited);
  writeEditHistory(store, editHistoryKey(key), history);
  const withServe = { ...model, scoreTracking: { ...model.scoreTracking, serveMarkers: [
    { id: "serve:R1", rallyId: "R1", timestamp: 10, side: "near" as const, modelSide: "near" as const, origin: "model" as const, ignorePreviousPoint: false },
  ] } };
  const result = restoreLabDraft(store, key, seed, withServe);
  assert.equal(result.status, "ready");
  if (result.status !== "ready") return;
  assert.equal(result.draft.cuts[0].included, false);
  assert.equal(result.draft.scoreTracking.serveMarkers[0].id, "serve:R1");
  assert.equal(readEditHistory(store.getItem(editHistoryKey(key)), result.draft, seed).past.length, 1);
  assert.equal(store.getItem(key), JSON.stringify(edited), "restoration does not write the review draft");
});

test("explicit new review backs up exact draft and history and leaves other modes alone", () => {
  const store = storage(); store.setItem(key, "broken draft"); store.setItem(editHistoryKey(key), "old history"); store.setItem("other-mode", "other review");
  assert.equal(startNewLabReview(store, key, model).status, "ready");
  const backups = [...store.values].filter(([name]) => name.startsWith(`${key}:unrestored:`));
  assert.equal(backups.length, 1);
  assert.deepEqual(JSON.parse(backups[0][1]), { draft: "broken draft", history: "old history" });
  assert.equal(store.getItem(key), JSON.stringify(model));
  assert.equal(store.getItem(editHistoryKey(key)), null);
  assert.equal(store.getItem("other-mode"), "other review");
});

test("failed recovery backup leaves original draft and history untouched", () => {
  const store = storage(); store.setItem(key, "broken"); store.setItem(editHistoryKey(key), "history");
  const before = [...store.values]; store.setItem = () => { throw new Error("full"); };
  assert.equal(startNewLabReview(store, key, model).status, "blocked");
  assert.deepEqual([...store.values], before);
});
