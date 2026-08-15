import assert from "node:assert/strict";
import test from "node:test";

import { parseLabelDocument } from "../../lib/annotations.ts";
import { buildProductionLabelSeed } from "../../lib/production-label-seed.ts";

const modelId = "model-production-fixture";

function baseDocument() {
  return parseLabelDocument({
    schemaVersion: 1,
    kind: "volleycut-rally-labels",
    createdAt: "2026-08-14T00:00:00Z",
    recording: {
      id: "indoor-fixture-full",
      video: "../../proxies/indoor/indoor-fixture-full.mp4",
      videoFilename: "indoor-fixture-full.mp4",
      contentSha256: "a".repeat(64),
      durationSeconds: 60,
      sourceGroup: "fixture-source",
      split: "challenge",
      environment: "indoor",
      game: { playersPerTeam: 4, targetPoints: null, format: "reverse 4s" },
      capture: { stationary: true },
      roi: { x: 0, y: 0, width: 1, height: 1 },
    },
    annotationPolicy: {
      id: "serve-contact-to-dead-ball-v1",
      rallyStart: "serve contact",
      rallyEnd: "dead ball",
      intervalConvention: "half-open",
    },
    annotation: {
      status: "not-started",
      annotator: "",
      continuousVideoReviewed: false,
      reviewedAt: null,
      notes: "",
    },
    rallies: [],
    ignoredIntervals: [],
    hardNegatives: [],
    sideSwitches: [],
  });
}

function analysis() {
  return {
    schemaVersion: 1,
    id: `${modelId}--indoor-fixture-full`,
    recordingId: "indoor-fixture-full",
    createdAt: "2026-08-14T01:02:03Z",
    source: {
      filename: "indoor-fixture-full.mp4",
      contentSha256: "a".repeat(64),
      duration: 60.00001,
    },
    analysis: {
      method: "production-three-head-model",
      variantLabel: "Production model fixture",
    },
    rallies: [
      { id: "R001", start: 4.1234, end: 9.9876, confidence: 0.87654, included: true },
      { id: "R002", start: 20, end: 25, confidence: 0.7, included: false },
      { id: "R003", start: 31, end: 35, confidence: 0.95, included: true },
    ],
  };
}

test("production predictions become the editable label seed", () => {
  const seed = buildProductionLabelSeed(baseDocument(), analysis(), modelId);

  assert.equal(seed.modelId, modelId);
  assert.equal(seed.modelLabel, "Production model fixture");
  assert.equal(seed.document.annotation.status, "in-progress");
  assert.equal(seed.document.annotation.annotator, "");
  assert.equal(seed.document.prelabel?.candidateFile, `${modelId}--indoor-fixture-full`);
  assert.deepEqual(seed.document.rallies, [
    {
      start: 4.123,
      end: 9.988,
      tags: ["ai-prelabel", "model-confidence:0.877"],
    },
    {
      start: 31,
      end: 35,
      tags: ["ai-prelabel", "model-confidence:0.950"],
    },
  ]);
});

test("production seed rejects an artifact for different media", () => {
  const mismatched = analysis();
  mismatched.source.contentSha256 = "b".repeat(64);

  assert.throws(
    () => buildProductionLabelSeed(baseDocument(), mismatched, modelId),
    /does not match the labeling task/,
  );
});
