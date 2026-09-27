import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { loadBindings, transform } from "next/dist/build/swc/index.js";
import { nearestSignalIndex, parseResearchReferences, referenceExportCore } from "../../lib/labeling-research.ts";
import { loadLabelingResearchReferences } from "../../lib/server/labeling-research.ts";
import type { LabelDocument } from "../../lib/annotations.ts";
import { padAndMergeRallies } from "../../lib/timeline-comparison.ts";

const identity = { id: "recording-a", videoFilename: "recording-a.mp4", durationSeconds: 60 };
function fixture() {
  return { schemaVersion: 1, kind: "volleycut-labeling-research-references", recordings: [{
    recordingId: identity.id, videoFilename: identity.videoFilename, durationSeconds: 60,
    references: [{ modelId: "compact-review", modelLabel: "Production + compact review", description: "Provisional full-parent review", exportPolicy: "fixed-production",
      rallies: [{ start: 10, end: 18 }, { start: 23, end: 29 }], exportRallies: [{ start: 8, end: 32 }],
      research: { recommendation: "Review full regions in evidence order; preserve production export.",
        signals: { times: [0, 10.1, 10.4, 59.5], live: [.1, .2, .3, .4], serve: [.2, .3, .4, .5], end: [.3, .4, .5, .6], keep: [.4, .5, .6, .7] },
        boundaryFlags: [{ id: "b1", parentId: "p1", kind: "additional_start", time: 23, priority: .8 }],
        reviewRegions: [{ id: "q1", parentId: "p1", start: 8, end: 32, recommended: true, priority: .8, reasons: ["additional_start"] },
          { id: "q2", parentId: "p2", start: 40, end: 48, recommended: false, priority: .2, reasons: ["cleanup"] }],
        queue: { budgetFraction: .1, reviewSeconds: 28, selectedParentCount: 1 },
      },
    }],
  }] };
}

async function detailedFixture() {
  const manifest = fixture();
  const research = manifest.recordings[0].references[0].research;
  const candidates = [{ id: "c1", parentId: "p1", componentId: "p1::valid:0", componentStart: 8, componentEnd: 32,
    start: 10, end: 18, originalNeuralStart: 9.5, originalNeuralEnd: 18.5 },
  { id: "c2", parentId: "p1", componentId: "p1::valid:0", componentStart: 8, componentEnd: 32,
    start: 23, end: 29, originalNeuralStart: 23, originalNeuralEnd: 29.5 }];
  const proposals = [
    { id: "first-start", candidateId: "c1", parentId: "p1", kind: "initial_start", time: 10, source: "head-serve", observed: true },
    { id: "separate-end", candidateId: "c1", parentId: "p1", kind: "end", time: 18, source: "head-end", observed: true },
    { id: "extra-start", candidateId: "c2", parentId: "p1", kind: "additional_start", time: 23, source: "compact-start", observed: true },
    { id: "last-end", candidateId: "c2", parentId: "p1", kind: "end", time: 29, source: "head-end", observed: true },
  ];
  research.boundaryFlags = proposals.map(proposal => ({ id: proposal.id, parentId: proposal.parentId, kind: proposal.kind, time: proposal.time, priority: .8 }));
  research.signals = { times: [9.9, 10.1, 18, 23, 29], live: [.1, .2, .3, .4, .5], serve: [.2, .3, .4, .5, .6], end: [.3, .4, .5, .6, .7], keep: [.4, .5, .6, .7, .8] };
  const adviser = { record: { id: identity.id, durationSeconds: 60, productionEvents: [{ id: "p1", start: 8, end: 32 }, { id: "p2", start: 40, end: 48 }] },
    plan: { eventCandidates: candidates, proposals },
    jobs: [{ parentId: "p1", splitIds: proposals.map(proposal => proposal.id) }, { parentId: "p2", splitIds: [] }],
    queue: { selectedParentIds: ["p1"] } };
  const temporary = await mkdtemp(path.join(os.tmpdir(), "volleycut-boundary-details-"));
  const source = path.join(temporary, "references.json");
  const plan = path.join(temporary, "adviser.json");
  const output = path.join(temporary, "details.json");
  try {
    const original = JSON.stringify(manifest);
    await writeFile(source, original);
    await writeFile(plan, JSON.stringify(adviser));
    const python = process.env.VOLLEYCUT_PYTHON ?? (process.platform === "win32" ? "python" : "python3");
    const script = fileURLToPath(new URL("../../scripts/enrich-labeling-boundary-details.py", import.meta.url));
    const generated = spawnSync(python, [script, "--references", source, "--adviser", plan, "--output", output], { encoding: "utf8" });
    assert.equal(generated.status, 0, generated.stderr || generated.error?.message);
    assert.equal(await readFile(source, "utf8"), original, "generator must preserve source manifest bytes");
    const result = JSON.parse(await readFile(output, "utf8"));
    const withoutDetails = structuredClone(result);
    delete withoutDetails.boundaryDetailsProvenance;
    for (const flag of withoutDetails.recordings[0].references[0].research.boundaryFlags) delete flag.details;
    assert.deepEqual(withoutDetails, manifest, "only explanation fields may change");
    const repeat = spawnSync(python, [script, "--references", source, "--adviser", plan, "--output", output], { encoding: "utf8" });
    assert.notEqual(repeat.status, 0, "derived artifacts must not be overwritten");
    return result;
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}

test("research lookup is recording-specific and preserves preview versus fixed production export", () => {
  const manifest = fixture();
  const before = JSON.stringify(manifest);
  const [reference] = parseResearchReferences(manifest, identity);
  assert.equal(reference.rallies.length, 2);
  assert.deepEqual(referenceExportCore(reference).map(({ start, end }) => ({ start, end })), [{ start: 8, end: 32 }]);
  assert.equal(referenceExportCore({ rallies: reference.rallies }), reference.rallies);
  assert.equal(reference.research?.signals.times[1], 10.1);
  assert.equal(JSON.stringify(manifest), before, "parsing must not mutate model or human input");
  assert.deepEqual(parseResearchReferences(manifest, { ...identity, id: "absent" }), []);
  assert.throws(() => parseResearchReferences(manifest, { ...identity, videoFilename: "other.mp4" }), /match/);
  assert.throws(() => parseResearchReferences(manifest, { ...identity, durationSeconds: 61 }), /match/);
});

test("research parser rejects malformed signal alignment, timestamps, queue and lineage", () => {
  const mutations = [
    (value: ReturnType<typeof fixture>) => { value.recordings[0].references[0].research.signals.live.pop(); },
    (value: ReturnType<typeof fixture>) => { value.recordings[0].references[0].research.signals.times[2] = 10.1; },
    (value: ReturnType<typeof fixture>) => { value.recordings[0].references[0].research.signals.serve[0] = 1.1; },
    (value: ReturnType<typeof fixture>) => { value.recordings[0].references[0].research.queue.selectedParentCount = 2; },
    (value: ReturnType<typeof fixture>) => { value.recordings[0].references[0].research.reviewRegions[1].parentId = "p1"; },
    (value: ReturnType<typeof fixture>) => { value.recordings[0].references[0].research.boundaryFlags[0].time = 33; },
    (value: ReturnType<typeof fixture>) => { value.recordings[0].references[0].rallies[1].start = 17; },
  ];
  for (const mutate of mutations) {
    const manifest = fixture(); mutate(manifest);
    assert.throws(() => parseResearchReferences(manifest, identity));
  }
});

test("nearest signal sample follows irregular actual timestamps with stable earlier ties", () => {
  assert.equal(nearestSignalIndex([], 1), -1);
  const times = [0, 10, 10.25, 59.5];
  assert.equal(nearestSignalIndex(times, -1), 0);
  assert.equal(nearestSignalIndex(times, 10.125), 1);
  assert.equal(nearestSignalIndex(times, 10.2), 2);
  assert.equal(nearestSignalIndex(times, 90), 3);
});

test("detail generator distinguishes moved parent boundaries from newly separated rally boundaries", async () => {
  const manifest = await detailedFixture();
  const [reference] = parseResearchReferences(manifest, identity);
  const flags = reference.research!.boundaryFlags;
  assert.equal(flags.length, 4);
  assert.equal(reference.research!.queue.selectedParentCount, 1);
  assert.deepEqual(flags.map(flag => [flag.details!.comparison, flag.details!.previousTime, flag.details!.shiftSeconds]), [
    ["original_parent_start", 8, 2], ["new_boundary", null, null], ["new_boundary", null, null], ["original_parent_end", 32, -3],
  ]);
  assert.equal(flags[0].details!.sample.index, 0, "nearest native sample uses earlier tie");
  assert.equal(flags[1].details!.nextProposedStart, 23);
  assert.equal(flags[2].details!.previousProposedEnd, 18);
  assert.equal(flags[2].details!.source, "compact-start");
  assert.equal(flags[2].details!.headShiftSeconds, 0);
  const original = structuredClone(manifest);
  manifest.recordings[0].references[0].research.boundaryFlags[0].details.sample.serve = .99;
  assert.throws(() => parseResearchReferences(manifest, identity), /scores do not match/);
  original.recordings[0].references[0].research.boundaryFlags[2].details.previousTime = 8;
  assert.throws(() => parseResearchReferences(original, identity), /shift does not match/);
});

test("server loader uses only configured manifest and matches its recording", async () => {
  const previous = process.env.VOLLEYCUT_LABELING_RESEARCH_REFERENCES_PATH;
  const temporary = await mkdtemp(path.join(os.tmpdir(), "volleycut-research-"));
  try {
    delete process.env.VOLLEYCUT_LABELING_RESEARCH_REFERENCES_PATH;
    const document = { recording: identity } as LabelDocument;
    assert.deepEqual(await loadLabelingResearchReferences(document), []);
    const filename = path.join(temporary, "references.json");
    await writeFile(filename, JSON.stringify(fixture()));
    process.env.VOLLEYCUT_LABELING_RESEARCH_REFERENCES_PATH = filename;
    assert.equal((await loadLabelingResearchReferences(document))[0].modelId, "compact-review");
  } finally {
    if (previous === undefined) delete process.env.VOLLEYCUT_LABELING_RESEARCH_REFERENCES_PATH;
    else process.env.VOLLEYCUT_LABELING_RESEARCH_REFERENCES_PATH = previous;
    await rm(temporary, { recursive: true, force: true });
  }
});

test("standalone reference uses its own cores for every export padding case", () => {
  const manifest = fixture();
  const entry = manifest.recordings[0].references[0];
  entry.exportPolicy = "model-predictions";
  entry.exportRallies = structuredClone(entry.rallies);
  const [reference] = parseResearchReferences(manifest, identity);
  const cores = referenceExportCore(reference).map((r, index) => ({ ...r, id: String(index), confidence: 1, included: true }));
  const boundaries = reference.rallies.map(({ start, end }) => [start, end]);
  const expected = [[[10, 18], [23, 29]], [[9, 19], [22, 30]], [[8, 31]], [[7, 32]]];
  for (const pad of [0, 1, 2, 3]) {
    assert.deepEqual(padAndMergeRallies(cores, pad, pad, 60, 3).map(({ start, end }) => [start, end]), expected[pad]);
    assert.deepEqual(reference.rallies.map(({ start, end }) => [start, end]), boundaries, "padding/joining must not mutate displayed rally boundaries");
  }
  entry.exportRallies = [{ start: 8, end: 32 }];
  assert.throws(() => parseResearchReferences(manifest, identity), /own model predictions/, "standalone cannot accidentally retain production export coverage");
});

test("research panel renders all four actual-time signals, recommended queue and typed proposals", async () => {
  // Exercise the actual component without requiring a browser or a Next server.
  // Only CSS module names are stubbed; JSX and React render normally.
  const require = createRequire(import.meta.url);
  await loadBindings();
  const filename = new URL("../../components/labeling-research-panel.tsx", import.meta.url);
  const source = (await readFile(filename, "utf8"))
    .replace(/import styles from "\.\/labeling-research-panel.module.css";/, "const styles = new Proxy({}, { get: (_target, key) => String(key) });")
    .replaceAll('"@/lib/annotations"', JSON.stringify(new URL("../../lib/annotations.ts", import.meta.url).href))
    .replaceAll('"@/lib/labeling-research"', JSON.stringify(new URL("../../lib/labeling-research.ts", import.meta.url).href));
  const compiled = (await transform(source, { filename: "labeling-research-panel.tsx", jsc: { parser: { syntax: "typescript", tsx: true }, target: "es2022", transform: { react: { runtime: "automatic" } } }, module: { type: "es6" } })).code
    .replaceAll('"react/jsx-runtime"', JSON.stringify(pathToFileURL(require.resolve("react/jsx-runtime")).href))
    .replaceAll('"react"', JSON.stringify(pathToFileURL(require.resolve("react")).href));
  const { LabelingResearchPanel } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`);
  const [reference] = parseResearchReferences(await detailedFixture(), identity);
  const markup = renderToStaticMarkup(createElement(LabelingResearchPanel, {
    modelLabel: reference.modelLabel, research: reference.research, duration: 60, currentTime: 10.3, onSeek: () => {},
  }));
  for (const label of ["Live play", "Serve start", "Rally end", "Keep", "0.300", "0.400", "0.500", "0.600", "Show all 2 flagged regions", "1 recommended regions", "Additional rally", "Production export stays unchanged", "they do not edit human labels", "4 boundary flags across 2 regions", "Initial start correction", "Separate rally end", "Final end correction", "+2.000 s", "-3.000 s", "Native sample", "not human verified", "Original production parent p1", "no head refinement", "Regions outside it can still need correction"]) {
    assert.ok(markup.includes(label), `missing ${label}`);
  }
  assert.equal((markup.match(/<path /g) ?? []).length, 4);
  assert.ok(markup.includes('role="slider"'));
  assert.ok(!markup.includes("cleanup</small>"), "unselected review region stays hidden by default");
  assert.ok(markup.includes("L336.67"), "trace X coordinate uses actual timestamp within its viewport");
  assert.ok(markup.includes("Proposed 0:29.000"), "region details include end proposals outside the current playhead");
  const standalone = renderToStaticMarkup(createElement(LabelingResearchPanel, {
    modelLabel: "Compact standalone", exportPolicy: "model-predictions", duration: 60, currentTime: 10.3, onSeek: () => {},
    research: { ...reference.research, recommendation: "Decoded compact rallies and their own export.", boundaryFlags: [], reviewRegions: [],
      queue: { budgetFraction: 0, reviewSeconds: 0, selectedParentCount: 0 } },
  }));
  assert.equal((standalone.match(/<path /g) ?? []).length, 4);
  assert.ok(standalone.includes("Standalone model predictions"));
  assert.ok(!standalone.includes("Production export stays unchanged"));
  assert.ok(!standalone.includes("Review queue"));
  assert.ok(!standalone.includes("boundary flags across"));
});
