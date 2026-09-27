import assert from "node:assert/strict";
import { mkdtemp, mkdir, writeFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { loadNeuralComparison, neuralComparisonCatalog } from "../../lib/server/neural-comparison.ts";
import { loadProductionEditorLab } from "../../lib/server/production-editor-lab.ts";
import type { LabelDocument } from "../../lib/annotations.ts";

test("five per-video model modes retain their own boundaries, signals and revision; mismatched sources fail closed", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "neural-comparison-"));
  const previous = process.env.VOLLEYCUT_NEURAL_COMPARISON_INDEX_PATH;
  const legacy = process.env.VOLLEYCUT_EDITOR_LAB_MANIFEST_PATH;
  const recording = { id: "example", videoFilename: "example.mp4", durationSeconds: 60, contentSha256: "a".repeat(64) } as LabelDocument["recording"];
  const index = { schemaVersion: 1, kind: "volleycut-neural-comparison-index", models: Array.from({ length: 5 }, (_, i) => ({ modelId: `neural-${i}` })),
    recordings: [{ id: "example", name: "example.mp4", file: "recordings/example.json", tier: "completed-exact" }] };
  const references = Array.from({ length: 5 }, (_, i) => ({ modelId: `neural-${i}`, modelLabel: `Model ${i} · recall target 99%`,
    description: "Frozen calibrated model", exportPolicy: "model-predictions", rallies: [{ start: i + 5, end: i + 10 }],
    exportRallies: [{ start: i + 5, end: i + 10 }], research: { recommendation: "Inspect model timing", signals: {
      times: [0, 10, 20], live: [i / 5, 1, 0], serve: [0, 1, 0], end: [0, 0, 1], keep: [0, 1, 0] },
      boundaryFlags: [], reviewRegions: [], queue: { budgetFraction: 0, reviewSeconds: 0, selectedParentCount: 0 } } }));
  try {
    await mkdir(path.join(root, "recordings"));
    await writeFile(path.join(root, "index.json"), JSON.stringify(index));
    await writeFile(path.join(root, "recordings/example.json"), JSON.stringify({ schemaVersion: 1,
      kind: "volleycut-labeling-research-references", recordings: [{ ...recording, recordingId: recording.id, references }] }));
    process.env.VOLLEYCUT_NEURAL_COMPARISON_INDEX_PATH = path.join(root, "index.json");
    delete process.env.VOLLEYCUT_EDITOR_LAB_MANIFEST_PATH;
    assert.equal((await neuralComparisonCatalog()).length, 1);
    const task = (await loadProductionEditorLab(recording))!;
    assert.equal(task.configurations.length, 5);
    assert.deepEqual(task.configurations.map(c => c.events[0].start), [5, 6, 7, 8, 9]);
    assert.deepEqual(task.configurations.map(c => c.signals!.live[0]), [0, .2, .4, .6, .8]);
    assert.ok(task.configurations.every(c => c.sourceRevision === task.sourceRevision));
    assert.ok(task.configurations.every(c => c.servingPredictions?.length === 0), "Do not reuse serving sides from different rally anchors");
    await assert.rejects(loadNeuralComparison({ ...recording, contentSha256: "b".repeat(64) }), /does not match/);
    assert.equal(await loadNeuralComparison({ ...recording, id: "other" }), null);
    const changed = JSON.parse(JSON.stringify(references));
    changed[0].modelLabel = "Renamed F1 option";
    changed[0].research.provenance = { uiDraftRevision: "c".repeat(64) };
    await writeFile(path.join(root, "recordings/example.json"), JSON.stringify({ schemaVersion: 1,
      kind: "volleycut-labeling-research-references", recordings: [{ ...recording, recordingId: recording.id, references: changed }] }));
    assert.equal((await loadProductionEditorLab(recording))!.configurations[0].sourceRevision, "c".repeat(64),
      "Adding options or relabeling an unchanged model preserves its existing draft revision");
    index.recordings[0].modelIds = ["neural-0", "neural-1"];
    await writeFile(path.join(root, "index.json"), JSON.stringify(index));
    await writeFile(path.join(root, "recordings/example.json"), JSON.stringify({ schemaVersion: 1,
      kind: "volleycut-labeling-research-references", recordings: [{ ...recording, recordingId: recording.id, references: changed.slice(0, 2) }] }));
    assert.equal((await loadProductionEditorLab(recording))!.configurations.length, 2,
      "A recording may publish a validated subset of the global comparison model catalog");
    index.recordings[0].modelIds = undefined;
    index.models.push({ modelId: "recall-pick" });
    await writeFile(path.join(root, "index.json"), JSON.stringify(index));
    await assert.rejects(loadNeuralComparison(recording), /match the model catalog/);
    index.recordings[0].file = "../outside.json";
    await writeFile(path.join(root, "index.json"), JSON.stringify(index));
    await assert.rejects(neuralComparisonCatalog(), /Invalid neural comparison recording/);
  } finally {
    if (previous === undefined) delete process.env.VOLLEYCUT_NEURAL_COMPARISON_INDEX_PATH;
    else process.env.VOLLEYCUT_NEURAL_COMPARISON_INDEX_PATH = previous;
    if (legacy === undefined) delete process.env.VOLLEYCUT_EDITOR_LAB_MANIFEST_PATH;
    else process.env.VOLLEYCUT_EDITOR_LAB_MANIFEST_PATH = legacy;
    await rm(root, { recursive: true, force: true });
  }
});
