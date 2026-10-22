import assert from "node:assert/strict";
import test from "node:test";
import * as production from "../../prod/src/lib/cut-draft.ts";
import * as lab from "../../components/production-lab/editor/lib/cut-draft.ts";

for (const [name, editor] of [["production", production], ["lab", lab]] as const) {
  const seed = { analysisId: "synthetic-split-limits", recordingId: "synthetic-split-limits", duration: 40,
    rallies: [{ id: "R001", start: 10, end: 20, confidence: .95, included: true }], ignoredIntervals: [] };
  const base = editor.createCutDraft(seed);
  const makeDraft = (assigned: number, targetAssigned: boolean, legacy = 0, id = "R001") => ({
    ...base,
    cuts: [
      { ...base.cuts[0], id, ...(targetAssigned ? { editableGroupId: "split:existing" } : {}) },
      ...Array.from({ length: assigned - Number(targetAssigned) }, (_, index) => ({
        ...base.cuts[0], id: `S${index}`, included: index % 2 === 0, editableGroupId: "split:siblings",
      })),
      ...Array.from({ length: legacy }, (_, index) => ({ ...base.cuts[0], id: `L${index}` })),
    ],
  });

  test(`${name} split refuses 2049 assigned members without changing its input`, () => {
    for (const targetAssigned of [true, false]) {
      const draft = makeDraft(targetAssigned ? 2048 : 2047, targetAssigned);
      const before = JSON.stringify(draft);
      assert.ok(editor.parseCutDraft(before, seed), "the input at the limit must be valid");
      assert.match(editor.splitCutValidationMessage(draft, "R001")!, /2,048/);
      assert.equal(editor.splitCutAt(draft, "R001", 15), null);
      assert.equal(JSON.stringify(draft), before);
    }
  });

  test(`${name} split counts assigned siblings, including excluded members, and allows exactly 2048`, () => {
    for (const targetAssigned of [true, false]) {
      const draft = makeDraft(targetAssigned ? 2047 : 2046, targetAssigned, 3);
      const before = structuredClone(draft);
      const split = editor.splitCutAt(draft, "R001", 15)!;
      assert.ok(split);
      assert.equal(split.draft.cuts.filter((cut) => cut.editableGroupId !== undefined).length, 2048);
      assert.equal(split.draft.cuts[0].id, "R001");
      assert.equal(split.newCut.id, "R002");
      assert.notEqual(split.draft.cuts[0].editableGroupId, split.newCut.editableGroupId);
      assert.deepEqual(split.draft.cuts.slice(2), before.cuts.slice(1));
      assert.ok(editor.parseCutDraft(JSON.stringify(split.draft), seed));
      assert.deepEqual(draft, before);
    }
  });

  test(`${name} split permits more than 2048 old unassigned cuts`, () => {
    const draft = makeDraft(0, false, 2048);
    const split = editor.splitCutAt(draft, "R001", 15)!;
    assert.ok(split);
    assert.equal(split.draft.cuts.length, 2050);
    assert.equal(split.draft.cuts.filter((cut) => cut.editableGroupId !== undefined).length, 2);
    assert.ok(editor.parseCutDraft(JSON.stringify(split.draft), seed));
  });

  test(`${name} split validates participating IDs without renaming legacy cuts`, () => {
    for (const id of ["L".repeat(257), " "]) {
      const draft = makeDraft(0, false, 0, id);
      const before = JSON.stringify(draft);
      assert.ok(editor.parseCutDraft(before, seed), "unassigned legacy IDs retain their defaults");
      assert.match(editor.splitCutValidationMessage(draft, id)!, /256/);
      assert.equal(editor.splitCutAt(draft, id, 15), null);
      assert.equal(JSON.stringify(draft), before);
    }
    const draft = makeDraft(2, true, 1, "L".repeat(256));
    draft.cuts.at(-1)!.id = "L".repeat(257);
    const split = editor.splitCutAt(draft, draft.cuts[0].id, 15)!;
    assert.ok(split);
    assert.equal(split.draft.cuts[0].id, draft.cuts[0].id);
    assert.equal(split.draft.cuts.at(-1)!.id, "L".repeat(257));
    assert.equal(split.draft.cuts.at(-1)!.editableGroupId, undefined);
    assert.ok(editor.parseCutDraft(JSON.stringify(split.draft), seed));
  });

  test(`${name} parser still rejects invalid present metadata`, () => {
    const overLimit = makeDraft(2049, true);
    assert.equal(editor.parseCutDraft(JSON.stringify(overLimit), seed), null);
    const overlong = makeDraft(1, true, 0, "L".repeat(257));
    assert.equal(editor.parseCutDraft(JSON.stringify(overlong), seed), null);
  });

  test(`${name} split allocates across cached/manual cuts and ignored interval IDs`, () => {
    const draft = makeDraft(0, false);
    draft.cuts.push({ ...base.cuts[0], id: "M001", origin: "manual" });
    draft.ignoredIntervals = ["R002", "R003", "M002", "M003"].map((id) => ({ id, start: 1, end: 2, reason: "other" }));
    const before = structuredClone(draft);
    assert.ok(editor.parseCutDraft(JSON.stringify(draft), seed));
    for (const [target, expected] of [["R001", "R004"], ["M001", "M004"]]) {
      assert.equal(editor.splitCutValidationMessage(draft, target), null);
      const split = editor.splitCutAt(draft, target, 15)!;
      assert.ok(split);
      assert.equal(split.newCut.id, expected);
      assert.deepEqual(split.draft.ignoredIntervals, before.ignoredIntervals);
      assert.equal(split.draft.cuts.find((cut) => cut.id === target)!.id, target);
      assert.ok(editor.parseCutDraft(JSON.stringify(split.draft), seed));
    }
    assert.deepEqual(draft, before);
  });

  test(`${name} split allocation remains fresh after the old 9999-ID search window`, () => {
    const draft = { ...base, cuts: Array.from({ length: 9999 }, (_, index) => ({
      ...base.cuts[0], id: `R${String(index + 1).padStart(3, "0")}`,
    })), ignoredIntervals: [{ id: "R10000", start: 1, end: 2, reason: "other" }] };
    assert.ok(editor.parseCutDraft(JSON.stringify(draft), seed));
    const split = editor.splitCutAt(draft, "R001", 15)!;
    assert.ok(split);
    assert.equal(split.newCut.id, "R10001");
    assert.ok(editor.parseCutDraft(JSON.stringify(split.draft), seed));
  });

  test(`${name} split preflight checks the complete range namespace before mutation`, () => {
    for (const conflict of ["cut-ignored", "ignored-ignored", "unassigned-cuts"]) {
      const draft = makeDraft(0, false);
      if (conflict === "cut-ignored") draft.ignoredIntervals = [{ id: "R001", start: 1, end: 2, reason: "other" }];
      if (conflict === "ignored-ignored") draft.ignoredIntervals = Array.from({ length: 2 }, () => ({ id: "I001", start: 1, end: 2, reason: "other" }));
      if (conflict === "unassigned-cuts") draft.cuts.push(...Array.from({ length: 2 }, () => ({ ...base.cuts[0], id: "legacy-sibling" })));
      const before = JSON.stringify(draft);
      assert.equal(editor.parseCutDraft(before, seed), null);
      assert.match(editor.splitCutValidationMessage(draft, "R001")!, /unique/);
      assert.equal(editor.splitCutAt(draft, "R001", 15), null);
      assert.equal(JSON.stringify(draft), before);
    }
  });

  test(`${name} range allocation respects the separate score-marker and group-token namespaces`, () => {
    const draft = makeDraft(1, true);
    draft.cuts[0].editableGroupId = "R002";
    draft.scoreTracking = { ...draft.scoreTracking,
      serveMarkers: [{ id: "R002", timestamp: 10, side: "near", origin: "manual", ignorePreviousPoint: false }],
      sideSwitchMarkers: [{ id: "R003", timestamp: 25, origin: "manual" }], removedModelMarkerIds: ["R004"] };
    assert.ok(editor.parseCutDraft(JSON.stringify(draft), seed));
    const split = editor.splitCutAt(draft, "R001", 15)!;
    assert.ok(split);
    assert.equal(split.newCut.id, "R002", "score identities and group tokens do not reserve range IDs");
    assert.deepEqual(split.draft.scoreTracking, draft.scoreTracking);
    assert.ok(editor.parseCutDraft(JSON.stringify(split.draft), seed));
  });
}
