import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  loadSideSwitchReviewProposalBundle,
  type ProposalArtifactSource,
} from "../../lib/server/side-switch-review-proposals.ts";

function json(value: unknown): string {
  return `${JSON.stringify(value, null, 2)}\n`;
}

async function writeLayer(
  directory: string,
  modelId: "v5" | "v6",
  predictions: Array<{
    eventId: string;
    gapOrder: number;
    score: number;
    selectedPrediction: boolean;
  }>,
  rows: Array<{ eventId: string; sourceEventIds: string[] }>,
): Promise<ProposalArtifactSource> {
  const version = Number(modelId.slice(1));
  const evaluationPath = path.join(directory, `${modelId}-evaluation.json`);
  const featurePath = path.join(directory, `${modelId}-features.json`);
  const featureContent = json({
    schemaVersion: version,
    kind: `volleycut-side-switch-features-${modelId}`,
    rows,
  });
  const featureSha256 = createHash("sha256")
    .update(featureContent)
    .digest("hex");
  await Promise.all([
    writeFile(
      evaluationPath,
      json({
        schemaVersion: version,
        kind: `volleycut-side-switch-specialist-evaluation-${modelId}`,
        createdAt: "2026-08-20T00:00:00Z",
        selectedDecoder: {
          threshold: modelId === "v5" ? 0.36 : 0.43,
          candidateMargin: 1,
        },
        predictions,
        sources: { features: { sha256: featureSha256 } },
      }),
    ),
    writeFile(featurePath, featureContent),
  ]);
  return {
    modelId,
    label: `${modelId.toUpperCase()} fixture`,
    detail: "fixture layer",
    evaluationPath,
    featurePath,
  };
}

test("V5 and V6 predictions attach to stable review event IDs", async (context) => {
  const directory = await mkdtemp(
    path.join(os.tmpdir(), "volleycut-switch-proposals-"),
  );
  context.after(async () => rm(directory, { recursive: true, force: true }));
  const [v5, v6] = await Promise.all([
    writeLayer(
      directory,
      "v5",
      [
        {
          eventId: "game:gap:1",
          gapOrder: 1,
          score: 0.8,
          selectedPrediction: true,
        },
        {
          eventId: "game:gap:2",
          gapOrder: 2,
          score: 0.2,
          selectedPrediction: false,
        },
      ],
      [
        { eventId: "game:gap:1", sourceEventIds: ["game:candidate-gap:1"] },
        { eventId: "game:gap:2", sourceEventIds: ["game:candidate-gap:2"] },
      ],
    ),
    writeLayer(
      directory,
      "v6",
      [
        {
          eventId: "game:gap:1",
          gapOrder: 1,
          score: 0.3,
          selectedPrediction: false,
        },
        {
          eventId: "game:gap:2",
          gapOrder: 2,
          score: 0.9,
          selectedPrediction: true,
        },
        {
          eventId: "game:gap:3",
          gapOrder: 3,
          score: 0.7,
          selectedPrediction: true,
        },
      ],
      [
        { eventId: "game:gap:1", sourceEventIds: ["game:candidate-gap:1"] },
        { eventId: "game:gap:2", sourceEventIds: ["game:candidate-gap:2"] },
        { eventId: "game:gap:3", sourceEventIds: ["outside-report"] },
      ],
    ),
  ]);

  const bundle = await loadSideSwitchReviewProposalBundle(
    new Set(["game:candidate-gap:1", "game:candidate-gap:2"]),
    [v5, v6],
  );

  assert.equal(bundle.errors.length, 0);
  assert.deepEqual(
    bundle.layers.map((layer) => ({
      modelId: layer.modelId,
      evaluated: layer.evaluatedEvents,
      selected: layer.selectedEvents,
      attached: layer.attachedEvents,
      selectedAttached: layer.selectedAttachedEvents,
    })),
    [
      {
        modelId: "v5",
        evaluated: 2,
        selected: 1,
        attached: 2,
        selectedAttached: 1,
      },
      {
        modelId: "v6",
        evaluated: 3,
        selected: 2,
        attached: 2,
        selectedAttached: 1,
      },
    ],
  );
  assert.equal(bundle.byEventId["game:candidate-gap:1"]?.v5?.selected, true);
  assert.equal(bundle.byEventId["game:candidate-gap:1"]?.v6?.selected, false);
  assert.equal(bundle.byEventId["game:candidate-gap:2"]?.v5?.score, 0.2);
  assert.equal(bundle.byEventId["game:candidate-gap:2"]?.v6?.selected, true);
  assert.equal(bundle.byEventId["outside-report"], undefined);
});

test("one unavailable proposal artifact does not hide the other model", async (context) => {
  const directory = await mkdtemp(
    path.join(os.tmpdir(), "volleycut-switch-proposals-"),
  );
  context.after(async () => rm(directory, { recursive: true, force: true }));
  const v5 = await writeLayer(
    directory,
    "v5",
    [
      {
        eventId: "game:gap:1",
        gapOrder: 1,
        score: 0.8,
        selectedPrediction: true,
      },
    ],
    [{ eventId: "game:gap:1", sourceEventIds: ["game:candidate-gap:1"] }],
  );
  const missingV6: ProposalArtifactSource = {
    modelId: "v6",
    label: "V6 fixture",
    detail: "missing fixture layer",
    evaluationPath: path.join(directory, "missing-evaluation.json"),
    featurePath: path.join(directory, "missing-features.json"),
  };

  const bundle = await loadSideSwitchReviewProposalBundle(
    new Set(["game:candidate-gap:1"]),
    [v5, missingV6],
  );

  assert.deepEqual(
    bundle.layers.map((layer) => layer.modelId),
    ["v5"],
  );
  assert.equal(bundle.byEventId["game:candidate-gap:1"]?.v5?.selected, true);
  assert.equal(bundle.errors.length, 1);
  assert.equal(bundle.errors[0]?.modelId, "v6");
  assert.match(bundle.errors[0]?.message ?? "", /could not be read/);
});

test("conflicting mappings fail only the malformed layer", async (context) => {
  const directory = await mkdtemp(
    path.join(os.tmpdir(), "volleycut-switch-proposals-"),
  );
  context.after(async () => rm(directory, { recursive: true, force: true }));
  const v5 = await writeLayer(
    directory,
    "v5",
    [
      {
        eventId: "game:gap:1",
        gapOrder: 1,
        score: 0.8,
        selectedPrediction: true,
      },
      {
        eventId: "game:gap:2",
        gapOrder: 2,
        score: 0.7,
        selectedPrediction: true,
      },
    ],
    [
      { eventId: "game:gap:1", sourceEventIds: ["game:candidate-gap:1"] },
      { eventId: "game:gap:2", sourceEventIds: ["game:candidate-gap:1"] },
    ],
  );

  const bundle = await loadSideSwitchReviewProposalBundle(
    new Set(["game:candidate-gap:1"]),
    [v5],
  );

  assert.equal(bundle.layers.length, 0);
  assert.deepEqual(bundle.byEventId, {});
  assert.equal(bundle.errors[0]?.modelId, "v5");
  assert.match(bundle.errors[0]?.message ?? "", /maps multiple predictions/);
});

test("an evaluation cannot attach a feature artifact with a changed digest", async (context) => {
  const directory = await mkdtemp(
    path.join(os.tmpdir(), "volleycut-switch-proposals-"),
  );
  context.after(async () => rm(directory, { recursive: true, force: true }));
  const v6 = await writeLayer(
    directory,
    "v6",
    [
      {
        eventId: "game:gap:1",
        gapOrder: 1,
        score: 0.8,
        selectedPrediction: true,
      },
    ],
    [{ eventId: "game:gap:1", sourceEventIds: ["game:candidate-gap:1"] }],
  );
  const changed = JSON.parse(await readFile(v6.featurePath, "utf8")) as Record<
    string,
    unknown
  >;
  changed.tamperedAfterEvaluation = true;
  await writeFile(v6.featurePath, json(changed));

  const bundle = await loadSideSwitchReviewProposalBundle(
    new Set(["game:candidate-gap:1"]),
    [v6],
  );

  assert.equal(bundle.layers.length, 0);
  assert.match(bundle.errors[0]?.message ?? "", /feature SHA-256/);
});
