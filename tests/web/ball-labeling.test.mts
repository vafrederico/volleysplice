import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { copyBallFrameLabelPreservingExposure } from "../../lib/ball-annotations.ts";
import {
  BallLabelingDraftValidationError,
  BallLabelingImageValidationError,
  BallComparisonAccessError,
  getBallComparisonLayers,
  getBallFrameImage,
  getBallLabelingCatalog,
  getBallReviewDocument,
  getBallSourceVideo,
  saveBallReviewDocument,
} from "../../lib/server/ball-labeling-tasks.ts";

test("copying a prior frame label preserves the target exposure audit", () => {
  const source = {
    status: "reviewed" as const,
    primaryBallState: "localizable" as const,
    objects: [
      {
        id: "track-1",
        category: "volleyball" as const,
        role: "primary-court" as const,
        bbox: { x: 0.2, y: 0.3, width: 0.04, height: 0.05 },
        visibility: "clear" as const,
        truncated: false,
      },
    ],
    notes: "copy semantics only",
    proposalExposure: "not_shown" as const,
    proposalSources: [] as Array<"sol" | "detector">,
  };
  const target = {
    ...source,
    objects: [],
    primaryBallState: "out_of_frame" as const,
    proposalExposure: "shown_before_label_finalized" as const,
    proposalSources: ["sol"] as Array<"sol" | "detector">,
  };

  const copied = copyBallFrameLabelPreservingExposure(source, target);

  assert.equal(copied.proposalExposure, "shown_before_label_finalized");
  assert.deepEqual(copied.proposalSources, ["sol"]);
  assert.equal(copied.primaryBallState, "localizable");
  assert.notEqual(copied.objects, source.objects);
  assert.notEqual(copied.objects[0].bbox, source.objects[0].bbox);
});

function sha256(value: Uint8Array | string): string {
  return createHash("sha256").update(value).digest("hex");
}

function canonicalJson(value: unknown): string {
  if (value === null || ["boolean", "number", "string"].includes(typeof value)) {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  const object = value as Record<string, unknown>;
  return `{${Object.keys(object)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${canonicalJson(object[key])}`)
    .join(",")}}`;
}

function pythonFloatJson(value: unknown): string {
  return `${JSON.stringify(value, null, 2)}\n`.replace('"fps": 30,', '"fps": 30.0,');
}

async function createWorkspace(options: { escapedImage?: boolean } = {}) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "volleycut-ball-labeling-"));
  const recordingId = "indoor-fixture-full";
  const taskDirectory = path.join(root, "tasks");
  const imageDirectory = path.join(root, "images", recordingId, "01-window");
  const labelingWorkspace = path.join(root, "labeling-workspace");
  const proxyDirectory = path.join(labelingWorkspace, "proxies", "indoor");
  const videoPath = path.join(proxyDirectory, `${recordingId}.mp4`);
  const videoBytes = Buffer.from("fixture-mp4-source-video");
  await Promise.all([
    fs.mkdir(taskDirectory, { recursive: true }),
    fs.mkdir(imageDirectory, { recursive: true }),
    fs.mkdir(proxyDirectory, { recursive: true }),
    fs.mkdir(path.join(root, "suggestions"), { recursive: true }),
  ]);
  await fs.writeFile(videoPath, videoBytes);
  const imageBytes = Buffer.from("exact-png-fixture");
  const frameIds = ["f000000000", "f000000002", "f000000004"];
  for (const frameId of frameIds) {
    await fs.writeFile(path.join(imageDirectory, `${frameId}.png`), imageBytes);
  }
  const immutable: Record<string, unknown> = {
    manifest: {
      name: "fixture",
      filename: "manifest.json",
      pathHint: "/private/manifest.json",
      sha256: "1".repeat(64),
    },
    recording: {
      id: recordingId,
      split: "train",
      sourceGroup: "source-a",
      environment: "indoor",
    },
    source: {
      proxy: {
        filename: `${recordingId}.mp4`,
        pathHint: videoPath,
        sizeBytes: videoBytes.byteLength,
        sha256: sha256(videoBytes),
        width: 960,
        height: 540,
        fps: 30,
        frameCount: 6,
        durationSeconds: 6 / 30,
      },
      normalizationProvenance: null,
    },
    sampling: { policyId: "fixture", round: 1 },
    annotationPolicy: { id: "fixture" },
    windows: [
      {
        id: "01-window",
        requestedStratum: "fixture",
        actualSource: "fixture",
        startSampleIndex: 0,
        startSeconds: 0,
        endSeconds: 1 / 5,
        centerSeconds: 1 / 15,
        reference: {},
      },
    ],
    frames: frameIds.map((frameId, index) => ({
      id: frameId,
      windowId: "01-window",
      sampleOffset: index,
      sourceFrameIndex: index * 2,
      sourceTimestampSeconds: (index * 2) / 30,
      image: {
        path: options.escapedImage
          ? "../../escape.png"
          : `../images/${recordingId}/01-window/${frameId}.png`,
        sha256: sha256(imageBytes),
        width: 960,
        height: 540,
        format: "png",
      },
    })),
  };
  const digest = sha256(canonicalJson(immutable).replace('"fps":30', '"fps":30.0'));
  immutable.taskId = `ball-presence-${digest.slice(0, 24)}`;
  immutable.digestSha256 = digest;
  const taskDocument = {
    schemaVersion: 1,
    taskType: "volleycut-ball-presence-frame-annotation",
    immutable,
    suggestions: { status: "empty", model: null, frames: {} },
    annotations: {
      review: { status: "unreviewed", annotator: null, reviewedAt: null, notes: "" },
      frames: Object.fromEntries(
        frameIds.map((frameId) => [
          frameId,
          { status: "unreviewed", primaryBallState: null, objects: [], notes: "" },
        ]),
      ),
    },
  };
  const taskFilename = `${recordingId}.ball-presence.json`;
  const encodedTask = pythonFloatJson(taskDocument);
  await fs.writeFile(path.join(taskDirectory, taskFilename), encodedTask);
  await fs.writeFile(
    path.join(root, "suggestions", taskFilename),
    "this detector artifact must never be read by the labeling server",
  );
  await fs.writeFile(
    path.join(root, "index.json"),
    JSON.stringify({
      schemaVersion: 1,
      artifactType: "volleycut-ball-presence-pilot-index",
      manifest: { sha256: "1".repeat(64) },
      samplingPolicyId: "fixture",
      round: 1,
      developmentOnly: true,
      recordingCount: 1,
      windowCount: 1,
      frameCount: 3,
      tasks: [
        {
          recordingId,
          split: "train",
          task: `tasks/${taskFilename}`,
          taskId: immutable.taskId,
          initialTaskSha256: sha256(encodedTask),
          frameCount: 3,
        },
      ],
    }),
  );
  return {
    root,
    recordingId,
    frameIds,
    imageDirectory,
    taskDocument,
    taskPath: path.join(taskDirectory, taskFilename),
    taskFilename,
    encodedTask,
    labelingWorkspace,
    videoBytes,
    videoPath,
  };
}

async function writeComparisonArtifacts(fixture: Awaited<ReturnType<typeof createWorkspace>>) {
  type MutableTask = Record<string, unknown> & {
    immutable: Record<string, unknown>;
    suggestions: unknown;
    annotations: { review: unknown; frames: Record<string, unknown> };
    solReviewProvenance?: unknown;
  };
  const detector = structuredClone(fixture.taskDocument) as unknown as MutableTask;
  detector.suggestions = {
    status: "complete",
    model: {
      modelId: "fixture-detector",
      modelSha256: "4".repeat(64),
      sourceTask: {
        pathHint: fixture.taskPath,
        sha256: sha256(fixture.encodedTask),
      },
    },
    frames: Object.fromEntries(
      fixture.frameIds.map((frameId) => [
        frameId,
        {
          ballPresenceProbability: 0.8,
          detections: [
            {
              confidence: 0.7,
              bbox: { x: 0.4, y: 0.3, width: 0.02, height: 0.03 },
            },
          ],
        },
      ]),
    ),
  };
  await fs.writeFile(
    path.join(fixture.root, "suggestions", fixture.taskFilename),
    pythonFloatJson(detector),
  );

  const detectorTiled = structuredClone(fixture.taskDocument) as unknown as MutableTask;
  detectorTiled.suggestions = {
    status: "complete",
    model: {
      modelId: "fixture-detector-tiled",
      modelSha256: "6".repeat(64),
      sourceTask: {
        pathHint: fixture.taskPath,
        sha256: sha256(fixture.encodedTask),
      },
      settings: {
        viewStrategy: {
          id: "full-plus-overlap-2x2-v1",
          viewsPerFrame: 5,
        },
      },
    },
    frames: Object.fromEntries(
      fixture.frameIds.map((frameId) => [
        frameId,
        {
          ballPresenceProbability: 0.65,
          detections: [
            {
              confidence: 0.6,
              bbox: { x: 0.1, y: 0.2, width: 0.04, height: 0.05 },
            },
          ],
        },
      ]),
    ),
  };
  const tiledDirectory = path.join(
    fixture.root,
    "suggestions-yolox-s-full-plus-2x2-v1",
  );
  await fs.mkdir(tiledDirectory, { recursive: true });
  await fs.writeFile(
    path.join(tiledDirectory, fixture.taskFilename),
    pythonFloatJson(detectorTiled),
  );

  const sol = structuredClone(fixture.taskDocument) as unknown as MutableTask;
  const solDirectory = path.join(fixture.root, "sol-labels");
  const solPath = path.join(solDirectory, fixture.taskFilename);
  const receiptPath = `${solPath}.preparation-receipt.json`;
  const preparedAt = "2026-08-11T19:00:00+00:00";
  const sourceTask = {
    pathHint: fixture.taskPath,
    sha256: sha256(fixture.encodedTask),
    requiredState: "unreviewed annotations with empty suggestions",
  };
  const reviewer = {
    kind: "detector-blind-sol-agent",
    agentId: "sol-agent",
    modelId: "gpt-fixture",
    runId: "fixture-run",
  };
  const implementationSha256 = "5".repeat(64);
  sol.annotations.review = {
    status: "complete",
    annotator: "sol-agent",
    reviewedAt: "2026-08-11T20:00:00+00:00",
    notes: "",
  };
  for (const frameId of fixture.frameIds) {
    sol.annotations.frames[frameId] = {
      status: "reviewed",
      primaryBallState: "out_of_frame",
      objects: [],
      notes: "",
    };
  }
  const indexPath = path.join(fixture.root, "index.json");
  const receipt = {
    schemaVersion: 1,
    kind: "volleycut-sol-ball-review-preparation-receipt",
    preparedAt,
    outputTask: {
      pathHint: solPath,
      requiredState: "unreviewed annotations with empty suggestions",
    },
    sourceTask,
    pilotIndex: {
      pathHint: indexPath,
      sha256: sha256(await fs.readFile(indexPath)),
    },
    reviewer,
    detectorSuggestionsAbsent: true,
    immutableDigestSha256: fixture.taskDocument.immutable.digestSha256,
    implementationSha256,
  };
  const receiptText = `${JSON.stringify(receipt, null, 2)}\n`;
  sol.solReviewProvenance = {
    schemaVersion: 1,
    kind: "volleycut-detector-blind-sol-ball-review",
    preparedAt,
    sourceTask,
    preparationReceipt: {
      pathHint: receiptPath,
      sha256: sha256(receiptText),
    },
    reviewer,
    detectorSuggestionsAbsent: true,
    immutableDigestSha256: fixture.taskDocument.immutable.digestSha256,
    implementationSha256,
  };
  await fs.mkdir(solDirectory, { recursive: true });
  await fs.writeFile(receiptPath, receiptText);
  await fs.writeFile(solPath, pythonFloatJson(sol));
}

test("blind ball review catalog, saves, provenance, and image binding", async (context) => {
  const previousRoot = process.env.VOLLEYCUT_BALL_PILOT_ROOT;
  const previousLabelingWorkspace = process.env.VOLLEYCUT_LABELING_WORKSPACE;
  const fixture = await createWorkspace();
  process.env.VOLLEYCUT_BALL_PILOT_ROOT = fixture.root;
  process.env.VOLLEYCUT_LABELING_WORKSPACE = fixture.labelingWorkspace;

  try {
    await context.test("loads pristine tasks without reading detector suggestions", async () => {
      const catalog = await getBallLabelingCatalog();
      assert.equal(catalog.length, 1);
      assert.equal(catalog[0].reviewedFrameCount, 0);
      const loaded = await getBallReviewDocument(fixture.recordingId);
      assert.equal("suggestions" in loaded.document, false);
      assert.equal(loaded.savedAt, null);
    });

    await context.test("writes a native empty-suggestion review atomically", async () => {
      const loaded = await getBallReviewDocument(fixture.recordingId);
      const firstFrame = fixture.frameIds[0];
      loaded.document.annotations.frames[firstFrame] = {
        status: "reviewed",
        primaryBallState: "out_of_frame",
        objects: [],
        notes: "",
        proposalExposure: "not_shown",
      };
      loaded.document.annotations.review = {
        status: "in_progress",
        annotator: null,
        reviewedAt: null,
        notes: "",
      };
      const saved = await saveBallReviewDocument(fixture.recordingId, loaded.document);
      assert.ok(saved.savedAt);
      const disk = JSON.parse(
        await fs.readFile(
          path.join(fixture.root, "reviews", `${fixture.recordingId}.ball-presence.json`),
          "utf8",
        ),
      );
      assert.deepEqual(disk.suggestions, { status: "empty", model: null, frames: {} });
      const diskSource = await fs.readFile(
        path.join(fixture.root, "reviews", `${fixture.recordingId}.ball-presence.json`),
        "utf8",
      );
      assert.match(diskSource, /"fps": 30\.0,/);
      assert.equal(disk.annotations.frames[firstFrame].status, "reviewed");
      assert.equal(disk.annotations.review.annotator, null);
      assert.deepEqual(
        (await fs.readdir(path.join(fixture.root, "reviews"))).filter((name) => name.endsWith(".tmp")),
        [],
      );
    });

    await context.test("gates comparison until a decision and keeps proposal sources separate", async () => {
      await writeComparisonArtifacts(fixture);
      await assert.rejects(
        getBallComparisonLayers(
          fixture.recordingId,
          fixture.frameIds[1],
          "post-decision",
          ["sol", "detector"],
        ),
        BallComparisonAccessError,
      );
      const comparison = await getBallComparisonLayers(
        fixture.recordingId,
        fixture.frameIds[0],
        "post-decision",
        ["sol", "detector"],
      );
      assert.equal(comparison.layers.sol?.provenance.annotator, "sol-agent");
      assert.equal(comparison.layers.detector?.provenance.modelId, "fixture-detector");
      assert.equal(comparison.layers.detector?.detections.length, 1);
      assert.equal(comparison.layers.detector?.provenance.variantId, "full-frame-v1");
      assert.equal(
        comparison.layers.detectorTiled?.provenance.variantId,
        "full-plus-overlap-2x2-v1",
      );
      assert.equal(
        comparison.layers.detectorTiled?.provenance.modelId,
        "fixture-detector-tiled",
      );
      assert.equal(comparison.layers.detectorTiled?.ballPresenceProbability, 0.65);
      assert.deepEqual(comparison.layers.detectorTiled?.detections[0].bbox, {
        x: 0.1,
        y: 0.2,
        width: 0.04,
        height: 0.05,
      });
      const rawSol = JSON.parse(
        await fs.readFile(
          path.join(fixture.root, "sol-labels", fixture.taskFilename),
          "utf8",
        ),
      );
      assert.equal(
        "proposalExposure" in rawSol.annotations.frames[fixture.frameIds[0]],
        false,
      );
      const saved = (await getBallReviewDocument(fixture.recordingId)).document;
      assert.equal(
        saved.annotations.frames[fixture.frameIds[0]].proposalExposure,
        "not_shown",
      );
      const audit = JSON.parse(
        await fs.readFile(
          path.join(
            fixture.root,
            "reviews",
            `${fixture.recordingId}.proposal-exposure.json`,
          ),
          "utf8",
        ),
      );
      assert.deepEqual(audit.frames[fixture.frameIds[0]].postDecisionSources, [
        "sol",
        "detector",
      ]);
      assert.equal(audit.frames[fixture.frameIds[0]].proposalExposure, "blind");
      assert.match(
        audit.frames[fixture.frameIds[0]].postDecisionHumanAnnotationSha256,
        /^[a-f0-9]{64}$/,
      );
    });

    await context.test("supports either detector artifact without double-counting exposure", async () => {
      const fullPath = path.join(fixture.root, "suggestions", fixture.taskFilename);
      const tiledPath = path.join(
        fixture.root,
        "suggestions-yolox-s-full-plus-2x2-v1",
        fixture.taskFilename,
      );
      await fs.rm(tiledPath);
      const fullOnly = await getBallComparisonLayers(
        fixture.recordingId,
        fixture.frameIds[0],
        "post-decision",
        ["detector"],
      );
      assert.ok(fullOnly.layers.detector);
      assert.equal(fullOnly.layers.detectorTiled, null);

      await writeComparisonArtifacts(fixture);
      await fs.rm(fullPath);
      const tiledOnly = await getBallComparisonLayers(
        fixture.recordingId,
        fixture.frameIds[0],
        "post-decision",
        ["detector"],
      );
      assert.equal(tiledOnly.layers.detector, null);
      assert.ok(tiledOnly.layers.detectorTiled);
      const audit = JSON.parse(
        await fs.readFile(
          path.join(
            fixture.root,
            "reviews",
            `${fixture.recordingId}.proposal-exposure.json`,
          ),
          "utf8",
        ),
      );
      assert.deepEqual(audit.frames[fixture.frameIds[0]].postDecisionSources, [
        "sol",
        "detector",
      ]);
      await writeComparisonArtifacts(fixture);
    });

    await context.test("rejects detector artifacts in the wrong mode directory", async () => {
      const fullPath = path.join(fixture.root, "suggestions", fixture.taskFilename);
      const tiledPath = path.join(
        fixture.root,
        "suggestions-yolox-s-full-plus-2x2-v1",
        fixture.taskFilename,
      );
      await fs.copyFile(tiledPath, fullPath);
      await assert.rejects(
        getBallComparisonLayers(
          fixture.recordingId,
          fixture.frameIds[0],
          "post-decision",
          ["detector"],
        ),
        /wrong view strategy for full-frame-v1/,
      );

      await writeComparisonArtifacts(fixture);
      const tiled = JSON.parse(await fs.readFile(tiledPath, "utf8"));
      tiled.suggestions.model.settings.viewStrategy.viewsPerFrame = 4;
      await fs.writeFile(tiledPath, pythonFloatJson(tiled));
      await assert.rejects(
        getBallComparisonLayers(
          fixture.recordingId,
          fixture.frameIds[0],
          "post-decision",
          ["detector"],
        ),
        /wrong view strategy for full-plus-overlap-2x2-v1/,
      );
      await writeComparisonArtifacts(fixture);
    });

    await context.test("rejects detector artifact symlinks", async () => {
      const fullPath = path.join(fixture.root, "suggestions", fixture.taskFilename);
      const outsidePath = path.join(fixture.root, "outside-detector.json");
      await fs.rename(fullPath, outsidePath);
      await fs.symlink(outsidePath, fullPath);
      await assert.rejects(
        getBallComparisonLayers(
          fixture.recordingId,
          fixture.frameIds[0],
          "post-decision",
          ["detector"],
        ),
        /regular file, not a symbolic link/,
      );
      await fs.rm(fullPath);
      await fs.rename(outsidePath, fullPath);
    });

    await context.test("validates the Sol preparation receipt and reviewer chronology", async () => {
      const solPath = path.join(fixture.root, "sol-labels", fixture.taskFilename);
      const receiptPath = `${solPath}.preparation-receipt.json`;
      await fs.appendFile(receiptPath, " ");
      await assert.rejects(
        getBallComparisonLayers(
          fixture.recordingId,
          fixture.frameIds[0],
          "post-decision",
          ["sol"],
        ),
        /preparation receipt SHA-256 differs/,
      );

      await writeComparisonArtifacts(fixture);
      const sol = JSON.parse(await fs.readFile(solPath, "utf8"));
      sol.annotations.review.annotator = "different-agent";
      await fs.writeFile(solPath, pythonFloatJson(sol));
      await assert.rejects(
        getBallComparisonLayers(
          fixture.recordingId,
          fixture.frameIds[0],
          "post-decision",
          ["sol"],
        ),
        /incomplete/,
      );

      await writeComparisonArtifacts(fixture);
      const early = JSON.parse(await fs.readFile(solPath, "utf8"));
      early.annotations.review.reviewedAt = "2000-01-01T00:00:00+00:00";
      await fs.writeFile(solPath, pythonFloatJson(early));
      await assert.rejects(
        getBallComparisonLayers(
          fixture.recordingId,
          fixture.frameIds[0],
          "post-decision",
          ["sol"],
        ),
        /incomplete/,
      );
      await writeComparisonArtifacts(fixture);
    });

    await context.test("marks a human revision after post-decision reveal as assisted", async () => {
      const loaded = await getBallReviewDocument(fixture.recordingId);
      loaded.document.annotations.frames[fixture.frameIds[0]] = {
        ...loaded.document.annotations.frames[fixture.frameIds[0]],
        primaryBallState: "fully_occluded",
        notes: "Revised after comparison.",
      };
      const saved = await saveBallReviewDocument(fixture.recordingId, loaded.document);
      assert.equal(
        saved.document.annotations.frames[fixture.frameIds[0]].proposalExposure,
        "shown_before_label_finalized",
      );
      assert.deepEqual(
        saved.document.annotations.frames[fixture.frameIds[0]].proposalSources,
        ["sol", "detector"],
      );
      const audit = JSON.parse(
        await fs.readFile(
          path.join(
            fixture.root,
            "reviews",
            `${fixture.recordingId}.proposal-exposure.json`,
          ),
          "utf8",
        ),
      );
      assert.deepEqual(audit.frames[fixture.frameIds[0]].assistedSources, [
        "sol",
        "detector",
      ]);
      assert.equal(audit.frames[fixture.frameIds[0]].proposalExposure, "both");
    });

    await context.test("persists assisted exposure irreversibly in frame and audit", async () => {
      const comparison = await getBallComparisonLayers(
        fixture.recordingId,
        fixture.frameIds[1],
        "assisted",
        ["sol", "detector"],
      );
      assert.equal(comparison.proposalExposure, "both");
      const loaded = await getBallReviewDocument(fixture.recordingId);
      assert.equal(
        loaded.document.annotations.frames[fixture.frameIds[1]].proposalExposure,
        "shown_before_label_finalized",
      );
      assert.deepEqual(
        loaded.document.annotations.frames[fixture.frameIds[1]].proposalSources,
        ["sol", "detector"],
      );
      loaded.document.annotations.frames[fixture.frameIds[1]].proposalExposure = "not_shown";
      loaded.document.annotations.frames[fixture.frameIds[1]].proposalSources = [];
      await assert.rejects(
        saveBallReviewDocument(fixture.recordingId, loaded.document),
        /proposal exposure is irreversible/,
      );
      const catalog = await getBallLabelingCatalog();
      assert.equal(catalog[0].assistedFrameCount, 2);
    });

    await context.test("serializes a stale save racing an assisted reveal", async () => {
      const stale = (await getBallReviewDocument(fixture.recordingId)).document;
      stale.annotations.frames[fixture.frameIds[2]] = {
        status: "reviewed",
        primaryBallState: "out_of_frame",
        objects: [],
        notes: "Concurrent autosave.",
        proposalExposure: "not_shown",
      };
      const results = await Promise.allSettled([
        saveBallReviewDocument(fixture.recordingId, stale),
        getBallComparisonLayers(
          fixture.recordingId,
          fixture.frameIds[2],
          "assisted",
          ["sol", "detector"],
        ),
      ]);
      assert.equal(results[1].status, "fulfilled");
      const final = await getBallReviewDocument(fixture.recordingId);
      assert.equal(
        final.document.annotations.frames[fixture.frameIds[2]].proposalExposure,
        "shown_before_label_finalized",
      );
      const audit = JSON.parse(
        await fs.readFile(
          path.join(
            fixture.root,
            "reviews",
            `${fixture.recordingId}.proposal-exposure.json`,
          ),
          "utf8",
        ),
      );
      assert.deepEqual(audit.frames[fixture.frameIds[2]].assistedSources, [
        "sol",
        "detector",
      ]);
    });

    await context.test("rejects Sol artifacts whose blind provenance is altered", async () => {
      const solPath = path.join(fixture.root, "sol-labels", fixture.taskFilename);
      const sol = JSON.parse(await fs.readFile(solPath, "utf8"));
      sol.solReviewProvenance.detectorSuggestionsAbsent = false;
      await fs.writeFile(solPath, `${JSON.stringify(sol, null, 2)}\n`);
      await assert.rejects(
        getBallComparisonLayers(
          fixture.recordingId,
          fixture.frameIds[0],
          "post-decision",
          ["sol"],
        ),
        /provenance is invalid/,
      );
    });

    await context.test("rejects detector fields and immutable mutations", async () => {
      const loaded = await getBallReviewDocument(fixture.recordingId);
      await assert.rejects(
        saveBallReviewDocument(fixture.recordingId, {
          ...loaded.document,
          suggestions: { status: "complete", model: {}, frames: {} },
        }),
        BallLabelingDraftValidationError,
      );
      const changed = structuredClone(loaded.document);
      changed.immutable.frames[0].sourceFrameIndex += 2;
      await assert.rejects(
        saveBallReviewDocument(fixture.recordingId, changed),
        /immutable source and image provenance cannot change/,
      );
    });

    await context.test("serves only SHA-pinned images bound to the source task", async () => {
      const image = await getBallFrameImage(fixture.recordingId, fixture.frameIds[0]);
      assert.equal(Buffer.from(image.bytes).toString("utf8"), "exact-png-fixture");
      await fs.writeFile(
        path.join(fixture.imageDirectory, `${fixture.frameIds[0]}.png`),
        "tampered",
      );
      await assert.rejects(
        getBallFrameImage(fixture.recordingId, fixture.frameIds[0]),
        BallLabelingImageValidationError,
      );
    });

    await context.test("serves only the immutable SHA-pinned source video", async () => {
      const video = await getBallSourceVideo(fixture.recordingId);
      assert.equal(video.filePath, fixture.videoPath);
      assert.equal(video.filename, `${fixture.recordingId}.mp4`);
      assert.equal(video.size, fixture.videoBytes.byteLength);
      assert.equal(video.sha256, sha256(fixture.videoBytes));
      await fs.appendFile(fixture.videoPath, "tampered");
      await assert.rejects(
        getBallSourceVideo(fixture.recordingId),
        /metadata does not match/,
      );
    });
  } finally {
    if (previousRoot === undefined) delete process.env.VOLLEYCUT_BALL_PILOT_ROOT;
    else process.env.VOLLEYCUT_BALL_PILOT_ROOT = previousRoot;
    if (previousLabelingWorkspace === undefined) {
      delete process.env.VOLLEYCUT_LABELING_WORKSPACE;
    } else {
      process.env.VOLLEYCUT_LABELING_WORKSPACE = previousLabelingWorkspace;
    }
    await fs.rm(fixture.root, { recursive: true, force: true });
  }

  await context.test("rejects an image path that escapes its recording directory", async () => {
    const escaped = await createWorkspace({ escapedImage: true });
    process.env.VOLLEYCUT_BALL_PILOT_ROOT = escaped.root;
    try {
      await assert.rejects(
        getBallFrameImage(escaped.recordingId, escaped.frameIds[0]),
        BallLabelingImageValidationError,
      );
    } finally {
      if (previousRoot === undefined) delete process.env.VOLLEYCUT_BALL_PILOT_ROOT;
      else process.env.VOLLEYCUT_BALL_PILOT_ROOT = previousRoot;
      await fs.rm(escaped.root, { recursive: true, force: true });
    }
  });
});
