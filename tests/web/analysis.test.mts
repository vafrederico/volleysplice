import assert from "node:assert/strict";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { loadAnalyses, parseAnalysis } from "../../lib/analysis.ts";

function fixture(id: string, title: string) {
  return {
    schemaVersion: 1,
    id,
    title,
    proxy: { duration: 120, width: 960, height: 540 },
    source: { filename: `${id}.mkv` },
    analysis: {
      cameraStability: 0.95,
      warnings: [],
      court: { confidence: 0.8, source: "detected-lines", lines: [] },
    },
    rallies: [{ id: "R01", start: 10, end: 20, confidence: 0.75 }],
  };
}

test("loadAnalyses returns every valid analysis newest first", async () => {
  const previous = process.env.VOLLEYCUT_DATA_ROOT;
  const dataRoot = await fs.mkdtemp(path.join(os.tmpdir(), "volleycut-analyses-"));
  process.env.VOLLEYCUT_DATA_ROOT = dataRoot;

  try {
    const analysesRoot = path.join(dataRoot, "analyses");
    const olderDirectory = path.join(analysesRoot, "grass-older");
    const newerDirectory = path.join(analysesRoot, "beach-newer");
    const invalidDirectory = path.join(analysesRoot, "indoor-invalid");
    await Promise.all([
      fs.mkdir(olderDirectory, { recursive: true }),
      fs.mkdir(newerDirectory, { recursive: true }),
      fs.mkdir(invalidDirectory, { recursive: true }),
    ]);
    const olderFile = path.join(olderDirectory, "analysis.json");
    const newerFile = path.join(newerDirectory, "analysis.json");
    await Promise.all([
      fs.writeFile(olderFile, JSON.stringify(fixture("grass-older", "Older grass"))),
      fs.writeFile(newerFile, JSON.stringify(fixture("beach-newer", "Newer beach"))),
      fs.writeFile(path.join(invalidDirectory, "analysis.json"), "not json"),
    ]);
    await fs.utimes(olderFile, new Date(1_000), new Date(1_000));
    await fs.utimes(newerFile, new Date(2_000), new Date(2_000));

    const analyses = await loadAnalyses();
    assert.deepEqual(analyses.map((analysis) => analysis.id), ["beach-newer", "grass-older"]);
    assert.equal(analyses[0].rallies.length, 1);
  } finally {
    if (previous === undefined) delete process.env.VOLLEYCUT_DATA_ROOT;
    else process.env.VOLLEYCUT_DATA_ROOT = previous;
    await fs.rm(dataRoot, { recursive: true, force: true });
  }
});

test("loadAnalyses returns an empty catalog when the data root is absent", async () => {
  const previous = process.env.VOLLEYCUT_DATA_ROOT;
  const missingRoot = path.join(os.tmpdir(), `volleycut-missing-${process.pid}-${Date.now()}`);
  process.env.VOLLEYCUT_DATA_ROOT = missingRoot;

  try {
    assert.deepEqual(await loadAnalyses(), []);
  } finally {
    if (previous === undefined) delete process.env.VOLLEYCUT_DATA_ROOT;
    else process.env.VOLLEYCUT_DATA_ROOT = previous;
  }
});

test("parseAnalysis accepts trained-model output without a heuristic proxy block", () => {
  const parsed = parseAnalysis({
    schemaVersion: 1,
    id: "model-full-percentile-v1--indoor-test-full",
    title: "indoor-test-full",
    source: {
      filename: "indoor-test-full.mp4",
      duration: 90,
      width: 960,
      height: 540,
    },
    analysis: {
      method: "court-motion-temporal-logistic-v0",
      modelVersion: "full-percentile-v1",
      court: { source: "manual-roi", lines: [] },
      warnings: [],
    },
    rallies: [{ id: "R001", start: 2, end: 8, confidence: 0.7 }],
  });
  assert.ok(parsed);
  assert.equal(parsed.kind, "model");
  assert.equal(parsed.modelVersion, "full-percentile-v1");
  assert.equal(parsed.recordingId, "indoor-test-full");
  assert.equal(parsed.courtConfidence, 1);
  assert.equal(parsed.rallies.length, 1);
});
