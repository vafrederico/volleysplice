import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

const server = await import("../../lib/server/ball-review-benchmark.ts");
const publicTypes = await import("../../lib/ball-review-benchmark.ts");

const realBenchmarkRoot =
  "/mnt/freenas/volleycut/ball-presence-v1/reports/ball-review-effort-screen12-v1";
const realReportSha256 =
  "818641939aa3f8e50d901933c69ad4229ea4146aefc1bce480740c93b3062638";

const { benchmarkConfigurationIds } = publicTypes;
const {
  BallReviewBenchmarkNotFoundError,
  BallReviewBenchmarkUnavailableError,
  BallReviewBenchmarkValidationError,
  getBallReviewBenchmarkBundle,
  getBallReviewBenchmarkImage,
} = server;

function sha256(value: Uint8Array | string): string {
  return createHash("sha256").update(value).digest("hex");
}

function json(value: unknown): string {
  return `${JSON.stringify(value, null, 2)}\n`;
}

function samplePng(width = 960, height = 540): Buffer {
  const bytes = Buffer.alloc(33);
  Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]).copy(bytes, 0);
  bytes.writeUInt32BE(13, 8);
  bytes.write("IHDR", 12, "ascii");
  bytes.writeUInt32BE(width, 16);
  bytes.writeUInt32BE(height, 20);
  bytes[24] = 8;
  bytes[25] = 2;
  return bytes;
}

function annotation(frameId?: string) {
  const result: Record<string, unknown> = {
    status: "reviewed",
    primaryBallState: "localizable",
    objects: [
      {
        id: "ball-1",
        category: "volleyball",
        role: "primary-court",
        bbox: { x: 0.25, y: 0.25, width: 0.05, height: 0.05 },
        visibility: "clear",
        truncated: false,
      },
    ],
    notes: "fixture",
  };
  return frameId === undefined ? result : { frameId, ...result };
}

function metrics() {
  return {
    frameCount: 12,
    stateAccuracy: 1,
    stateMacroF1: 1,
    primaryPresenceF1: 1,
    boxF1Iou25: 1,
    boxF1Iou50: 1,
    matchedPrimaryCount: 12,
    matchedPrimaryMeanIou: 1,
    matchedPrimaryMedianCenterErrorPixels: 0,
    matchedPrimaryVisibilityAccuracyIou25: 1,
    objectCountAccuracy: 1,
  };
}

const specs = {
  "sol-low": ["gpt-5.6-sol", "low"],
  "sol-medium": ["gpt-5.6-sol", "medium"],
  "sol-high": ["gpt-5.6-sol", "high"],
  "sol-xhigh": ["gpt-5.6-sol", "xhigh"],
  "sol-max": ["gpt-5.6-sol", "max"],
  "terra-xhigh": ["gpt-5.6-terra", "xhigh"],
  "terra-max": ["gpt-5.6-terra", "max"],
  "luna-xhigh": ["gpt-5.6-luna", "xhigh"],
  "luna-max": ["gpt-5.6-luna", "max"],
} as const;

type FixtureOptions = {
  imageName?: string;
  imageSymlink?: boolean;
  imageWidth?: number;
  tamperResultAfterReport?: boolean;
  unknownConfiguration?: boolean;
};

async function createFixture(options: FixtureOptions = {}) {
  const root = await fs.mkdtemp(
    path.join(os.tmpdir(), "volleycut-ball-benchmark-"),
  );
  const imageDirectory = path.join(root, "inputs");
  await fs.mkdir(imageDirectory, { recursive: true });
  const frames = Array.from({ length: 12 }, (_, index) => ({
    attachmentIndex: index + 1,
    frameId: `f${String(index + 1).padStart(9, "0")}`,
    image:
      index === 0
        ? (options.imageName ?? "image-01.png")
        : `image-${String(index + 1).padStart(2, "0")}.png`,
    sha256: "",
  }));
  for (const frame of frames) {
    const bytes = samplePng(
      frame.attachmentIndex === 1 ? options.imageWidth : undefined,
    );
    frame.sha256 = sha256(bytes);
    if (frame.attachmentIndex === 1 && options.imageSymlink) {
      const escaped = path.join(root, "escaped.png");
      await fs.writeFile(escaped, bytes);
      await fs.symlink(escaped, path.join(imageDirectory, "image-01.png"));
    } else if (
      !frame.image.includes("/") &&
      !frame.image.includes("\\") &&
      frame.image !== ".."
    ) {
      await fs.writeFile(path.join(imageDirectory, frame.image), bytes);
    }
  }

  const blindPackBytes = json({
    schemaVersion: 1,
    policy: "volleyball-ball-presence-v1",
    width: 960,
    height: 540,
    frames,
  });
  await fs.writeFile(path.join(root, "blind-pack.json"), blindPackBytes);

  const resultSchemaBytes = json({
    $schema: "https://json-schema.org/draft/2020-12/schema",
    type: "object",
    additionalProperties: false,
    required: ["schemaVersion", "frames"],
    properties: {
      schemaVersion: { const: 1 },
      frames: { type: "array" },
    },
  });
  await fs.writeFile(path.join(root, "result.schema.json"), resultSchemaBytes);

  const referenceBytes = json({
    schemaVersion: 1,
    source: "prior blind Sol xhigh pseudo-reference",
    sourceTaskSha256: Object.fromEntries(
      Array.from({ length: 8 }, (_, index) => [
        `source-${index + 1}`,
        String(index + 1).repeat(64),
      ]),
    ),
    frames: Object.fromEntries(
      frames.map((frame) => [frame.frameId, annotation()]),
    ),
  });
  await fs.writeFile(path.join(root, "sealed-reference.json"), referenceBytes);

  const artifactRuns: Record<string, Record<string, string>> = {};
  const configurations: Record<string, unknown> = {};
  const baselineSeconds = 40;
  for (const [index, configId] of benchmarkConfigurationIds.entries()) {
    const runDirectory = path.join(root, "runs", configId);
    await fs.mkdir(runDirectory, { recursive: true });
    const [model, effort] = specs[configId];
    const elapsedSeconds = index === 3 ? baselineSeconds : 10 + index;
    const usage = {
      cache_write_input_tokens: 0,
      cached_input_tokens: 0,
      input_tokens: 100,
      output_tokens: 20 + index,
      reasoning_output_tokens: 10 + index,
    };
    const resultBytes = json({
      schemaVersion: 1,
      frames: frames.map((frame) => annotation(frame.frameId)),
    });
    const resultSha = sha256(resultBytes);
    const receiptBytes = json({
      schemaVersion: 1,
      configId,
      model,
      effort,
      serviceTier: "default",
      startedEpochSeconds: 1,
      elapsedSeconds,
      exitCode: 0,
      valid: true,
      usage,
      resultSha256: resultSha,
    });
    const eventsBytes = `${json({ type: "turn.completed", usage })}`;
    const stderrBytes = "fixture complete\n";
    await Promise.all([
      fs.writeFile(path.join(runDirectory, "result.json"), resultBytes),
      fs.writeFile(path.join(runDirectory, "receipt.json"), receiptBytes),
      fs.writeFile(path.join(runDirectory, "events.jsonl"), eventsBytes),
      fs.writeFile(path.join(runDirectory, "stderr.log"), stderrBytes),
    ]);
    artifactRuns[configId] = {
      resultSha256: resultSha,
      receiptSha256: sha256(receiptBytes),
      eventsSha256: sha256(eventsBytes),
      stderrSha256: sha256(stderrBytes),
    };
    configurations[configId] = {
      model,
      effort,
      elapsedSeconds,
      speedupVsFreshSolXhigh: baselineSeconds / elapsedSeconds,
      usage,
      similarityToPriorSolXhigh: metrics(),
    };
  }
  if (options.unknownConfiguration) {
    configurations["sol-ultra"] = configurations["sol-low"];
  }

  const pairwise = Object.fromEntries(
    benchmarkConfigurationIds.map((first) => [
      first,
      Object.fromEntries(
        benchmarkConfigurationIds.map((second) => [second, metrics()]),
      ),
    ]),
  );
  const referenceSha = sha256(referenceBytes);
  const reportBytes = json({
    schemaVersion: 1,
    status: "complete",
    sample: {
      frameCount: 12,
      recordingCount: 8,
      environments: ["beach", "grass", "indoor"],
      reference: "prior blind Sol-xhigh pseudo-reference, not human truth",
      sealedReferenceSha256: referenceSha,
    },
    configurations,
    pairwise,
    metricSemantics: {
      similarityToPriorSolXhigh: "fixture directional semantics",
      pairwise: "fixture symmetric semantics",
    },
    artifactIntegrity: {
      blindPackSha256: sha256(blindPackBytes),
      schemaSha256: sha256(resultSchemaBytes),
      sealedReferenceSha256: referenceSha,
      runs: artifactRuns,
    },
    limitations: ["Fixture limitation."],
  });
  await fs.writeFile(path.join(root, "report.json"), reportBytes);

  if (options.tamperResultAfterReport) {
    await fs.appendFile(
      path.join(root, "runs", "sol-low", "result.json"),
      " ",
    );
  }
  return { root, reportSha256: sha256(reportBytes) };
}

async function withFixture<T>(
  options: FixtureOptions,
  callback: (fixture: Awaited<ReturnType<typeof createFixture>>) => Promise<T>,
): Promise<T> {
  const fixture = await createFixture(options);
  const priorRoot = process.env.VOLLEYCUT_BALL_BENCHMARK_ROOT;
  const priorSha = process.env.VOLLEYCUT_BALL_BENCHMARK_REPORT_SHA256;
  process.env.VOLLEYCUT_BALL_BENCHMARK_ROOT = fixture.root;
  process.env.VOLLEYCUT_BALL_BENCHMARK_REPORT_SHA256 = fixture.reportSha256;
  try {
    return await callback(fixture);
  } finally {
    if (priorRoot === undefined) {
      delete process.env.VOLLEYCUT_BALL_BENCHMARK_ROOT;
    } else {
      process.env.VOLLEYCUT_BALL_BENCHMARK_ROOT = priorRoot;
    }
    if (priorSha === undefined) {
      delete process.env.VOLLEYCUT_BALL_BENCHMARK_REPORT_SHA256;
    } else {
      process.env.VOLLEYCUT_BALL_BENCHMARK_REPORT_SHA256 = priorSha;
    }
    await fs.rm(fixture.root, { recursive: true, force: true });
  }
}

test("the complete fixture is sanitized into twelve frames and nine exact configurations", async () => {
  await withFixture({}, async () => {
    const bundle = await getBallReviewBenchmarkBundle();

    assert.equal(bundle.benchmarkId, "ball-review-effort-screen12-v1");
    assert.equal(bundle.frames.length, 12);
    assert.deepEqual(Object.keys(bundle.configurations).sort(), [
      ...benchmarkConfigurationIds,
    ].sort());
    assert.deepEqual(
      Object.keys(bundle.frames[0].outputs).sort(),
      [...benchmarkConfigurationIds].sort(),
    );
    assert.equal(
      bundle.frames[0].imageUrl,
      "/api/ball-review-benchmark/images/1",
    );
    const serialized = JSON.stringify(bundle);
    for (const forbidden of [
      "sourceTaskSha256",
      "artifactIntegrity",
      "startedEpochSeconds",
      "events.jsonl",
      "receipt.json",
      "result.json",
      "pathHint",
    ]) {
      assert.equal(serialized.includes(forbidden), false, forbidden);
    }

    const image = await getBallReviewBenchmarkImage(1);
    assert.equal(image.filename, "image-01.png");
    assert.equal(sha256(image.bytes), image.sha256);
  });
});

test("the compiled frozen benchmark artifacts load successfully when available", async (t) => {
  try {
    await fs.access(path.join(realBenchmarkRoot, "report.json"));
  } catch {
    t.skip("The frozen benchmark share is unavailable");
    return;
  }
  const priorRoot = process.env.VOLLEYCUT_BALL_BENCHMARK_ROOT;
  const priorSha = process.env.VOLLEYCUT_BALL_BENCHMARK_REPORT_SHA256;
  process.env.VOLLEYCUT_BALL_BENCHMARK_ROOT = realBenchmarkRoot;
  process.env.VOLLEYCUT_BALL_BENCHMARK_REPORT_SHA256 = realReportSha256;
  try {
    const bundle = await getBallReviewBenchmarkBundle();
    assert.equal(bundle.reportSha256, realReportSha256);
    assert.equal(bundle.frames.length, 12);
    assert.equal(Object.keys(bundle.configurations).length, 9);
  } finally {
    if (priorRoot === undefined) {
      delete process.env.VOLLEYCUT_BALL_BENCHMARK_ROOT;
    } else {
      process.env.VOLLEYCUT_BALL_BENCHMARK_ROOT = priorRoot;
    }
    if (priorSha === undefined) {
      delete process.env.VOLLEYCUT_BALL_BENCHMARK_REPORT_SHA256;
    } else {
      process.env.VOLLEYCUT_BALL_BENCHMARK_REPORT_SHA256 = priorSha;
    }
  }
});

test("the report must match the trusted SHA-256", async () => {
  await withFixture({}, async (fixture) => {
    process.env.VOLLEYCUT_BALL_BENCHMARK_REPORT_SHA256 = "f".repeat(64);
    await assert.rejects(
      getBallReviewBenchmarkBundle(),
      (error: unknown) =>
        error instanceof BallReviewBenchmarkValidationError &&
        /compiled trusted SHA-256/.test(error.message),
    );
    process.env.VOLLEYCUT_BALL_BENCHMARK_REPORT_SHA256 = fixture.reportSha256;
  });
});

test("a missing benchmark root is classified as unavailable", async () => {
  const priorRoot = process.env.VOLLEYCUT_BALL_BENCHMARK_ROOT;
  const priorSha = process.env.VOLLEYCUT_BALL_BENCHMARK_REPORT_SHA256;
  const missingRoot = path.join(
    os.tmpdir(),
    `volleycut-missing-ball-benchmark-${process.pid}-${Date.now()}`,
  );
  process.env.VOLLEYCUT_BALL_BENCHMARK_ROOT = missingRoot;
  process.env.VOLLEYCUT_BALL_BENCHMARK_REPORT_SHA256 = "f".repeat(64);
  try {
    await assert.rejects(
      getBallReviewBenchmarkBundle(),
      BallReviewBenchmarkUnavailableError,
    );
  } finally {
    if (priorRoot === undefined) {
      delete process.env.VOLLEYCUT_BALL_BENCHMARK_ROOT;
    } else {
      process.env.VOLLEYCUT_BALL_BENCHMARK_ROOT = priorRoot;
    }
    if (priorSha === undefined) {
      delete process.env.VOLLEYCUT_BALL_BENCHMARK_REPORT_SHA256;
    } else {
      process.env.VOLLEYCUT_BALL_BENCHMARK_REPORT_SHA256 = priorSha;
    }
  }
});

test("a result changed after report generation is rejected", async () => {
  await withFixture({ tamperResultAfterReport: true }, async () => {
    await assert.rejects(
      getBallReviewBenchmarkBundle(),
      (error: unknown) =>
        error instanceof BallReviewBenchmarkValidationError &&
        /trusted SHA-256/.test(error.message),
    );
  });
});

test("an unknown benchmark configuration is rejected", async () => {
  await withFixture({ unknownConfiguration: true }, async () => {
    await assert.rejects(
      getBallReviewBenchmarkBundle(),
      (error: unknown) =>
        error instanceof BallReviewBenchmarkValidationError &&
        /unexpected shape/.test(error.message),
    );
  });
});

test("manifest path traversal is rejected before image access", async () => {
  await withFixture({ imageName: "../escape.png" }, async () => {
    await assert.rejects(
      getBallReviewBenchmarkImage(1),
      (error: unknown) =>
        error instanceof BallReviewBenchmarkValidationError &&
        /frozen benchmark/.test(error.message),
    );
  });
});

test("symlinked image artifacts are rejected", async () => {
  await withFixture({ imageSymlink: true }, async () => {
    await assert.rejects(
      getBallReviewBenchmarkImage(1),
      (error: unknown) =>
        error instanceof BallReviewBenchmarkValidationError &&
        /non-symlink file/.test(error.message),
    );
  });
});

test("PNG IHDR dimensions are validated", async () => {
  await withFixture({ imageWidth: 959 }, async () => {
    await assert.rejects(
      getBallReviewBenchmarkImage(1),
      (error: unknown) =>
        error instanceof BallReviewBenchmarkValidationError &&
        /dimensions/.test(error.message),
    );
  });
});

test("unknown and non-integer attachment indexes are rejected", async () => {
  await assert.rejects(
    getBallReviewBenchmarkImage(0),
    BallReviewBenchmarkNotFoundError,
  );
  await assert.rejects(
    getBallReviewBenchmarkImage(13),
    BallReviewBenchmarkNotFoundError,
  );
  await assert.rejects(
    getBallReviewBenchmarkImage(1.5),
    BallReviewBenchmarkNotFoundError,
  );
});
