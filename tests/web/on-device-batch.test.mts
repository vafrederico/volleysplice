import assert from "node:assert/strict";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  GET as getBatchRoute,
  POST as postBatchRoute,
} from "../../app/api/on-device-batch/route.ts";
import { GET as getBatchMediaRoute } from "../../app/api/on-device-batch/media/[id]/route.ts";
import { parseAnalysis } from "../../lib/analysis.ts";
import {
  DEFAULT_ON_DEVICE_RUNTIME_VARIANT,
  type OnDeviceRuntimeVariant,
} from "../../lib/on-device/runtime-variants.ts";
import {
  assertOnDeviceBatchAuthorized,
  buildOnDeviceBatchCatalog,
  configuredOnDeviceBatchOutputRoot,
  ON_DEVICE_BATCH_BUNDLE_SHA256,
  ON_DEVICE_BATCH_FEATURE_PATH,
  ON_DEVICE_BATCH_MODEL_ID,
  ON_DEVICE_BATCH_MODEL_VERSION,
  ON_DEVICE_BATCH_RECORDING_IDS,
  OnDeviceBatchAuthorizationError,
  OnDeviceBatchConfigurationError,
  OnDeviceBatchConflictError,
  OnDeviceBatchValidationError,
  onDeviceBatchAnalysisId,
  onDeviceRuntimeVariantFromRequest,
  persistOnDeviceBatchAnalysis,
  readPersistedOnDeviceBatchAnalysis,
  type OnDeviceBatchSubmission,
  type OnDeviceBatchTask,
  validateOnDeviceBatchSubmission,
} from "../../lib/server/on-device-batch.ts";

const TOKEN_ENV = "VOLLEYCUT_ON_DEVICE_BATCH_TOKEN";
const OUTPUT_ROOT_ENV = "VOLLEYCUT_ON_DEVICE_BATCH_OUTPUT_ROOT";
const PRODUCTION_MODEL_SLUG = ON_DEVICE_BATCH_MODEL_ID.replace("model-", "");
const LINEAR_ANALYSIS_PREFIX = `model-browser-on-device-${PRODUCTION_MODEL_SLUG}--`;
const RESAMPLED_ANALYSIS_PREFIX =
  `model-browser-on-device-libswresample-wasm-${PRODUCTION_MODEL_SLUG}--`;

type SavedArtifact = {
  schemaVersion: number;
  id: string;
  recordingId: string;
  createdAt: string;
  analysis: {
    method: string;
    modelVersion: string;
    variantLabel: string;
    modelBundleSha256: string;
    provenance: {
      inferenceLocation: string;
      runtimeVariant: string;
      audioResampler: string;
    };
    warnings: string[];
  };
  rallies: unknown[];
};

function taskFor(id: string, duration = 100): OnDeviceBatchTask {
  const environment = id.startsWith("beach-")
    ? "beach"
    : id.startsWith("grass-")
      ? "grass"
      : "indoor";
  return {
    id,
    batch: "full",
    originalFilename: `Original ${id}.mkv`,
    proxySize: 123_456,
    document: {
      recording: {
        id,
        videoFilename: `${id}.mp4`,
        contentSha256: "a".repeat(64),
        durationSeconds: duration,
        environment,
        roi: { x: 0.04, y: 0.14, width: 0.92, height: 0.84 },
        capture: { stationary: true },
      },
    },
  } as OnDeviceBatchTask;
}

function validBody(
  recordingId: string,
  duration = 100,
  runtimeVariant: OnDeviceRuntimeVariant = DEFAULT_ON_DEVICE_RUNTIME_VARIANT,
): unknown {
  return {
    recordingId,
    modelId: ON_DEVICE_BATCH_MODEL_ID,
    featurePath: ON_DEVICE_BATCH_FEATURE_PATH,
    runtimeVariant,
    media: {
      duration: duration + 0.02,
      mimeType: "video/mp4",
      width: 960,
      height: 540,
      rotation: 0,
      videoCodec: "avc",
      videoCodecString: null,
      canDecodeVideo: true,
      hasAudio: true,
      audioCodec: "aac",
      sampleRate: 48_000,
      channels: 2,
      canDecodeAudio: true,
    },
    intervals: [
      { id: "R001", start: 1.25, end: 5.5, confidence: 0.8, included: true },
      { id: "R002", start: 5.5, end: 12, confidence: 0.95, included: true },
    ],
    provenance: {
      secureContext: true,
      userAgent: "Browser batch test",
      completedAt: "2026-08-13T20:00:00.000Z",
    },
  };
}

test("batch authorization requires the configured bearer token exactly", () => {
  const previous = process.env[TOKEN_ENV];
  try {
    delete process.env[TOKEN_ENV];
    assert.throws(
      () => assertOnDeviceBatchAuthorized(new Request("https://example.test")),
      OnDeviceBatchConfigurationError,
    );

    process.env[TOKEN_ENV] = "exact-token";
    assert.doesNotThrow(() =>
      assertOnDeviceBatchAuthorized(
        new Request("https://example.test", {
          headers: { Authorization: "Bearer exact-token" },
        }),
      ),
    );
    for (const authorization of [
      "exact-token",
      "bearer exact-token",
      "Bearer wrong-token",
    ]) {
      assert.throws(
        () =>
          assertOnDeviceBatchAuthorized(
            new Request("https://example.test", { headers: { authorization } }),
          ),
        OnDeviceBatchAuthorizationError,
      );
    }
  } finally {
    if (previous === undefined) delete process.env[TOKEN_ENV];
    else process.env[TOKEN_ENV] = previous;
  }
});

test("runtime variant query defaults to linear and rejects unknown or repeated values", () => {
  assert.equal(
    onDeviceRuntimeVariantFromRequest(
      new Request("https://example.test/api/on-device-batch"),
    ),
    "linear-v1",
  );
  assert.equal(
    onDeviceRuntimeVariantFromRequest(
      new Request(
        "https://example.test/api/on-device-batch?variant=libswresample-wasm-v1",
      ),
    ),
    "libswresample-wasm-v1",
  );
  for (const query of [
    "?variant=unknown",
    "?variant=linear-v1&variant=libswresample-wasm-v1",
    "?variant=",
  ]) {
    assert.throws(
      () =>
        onDeviceRuntimeVariantFromRequest(
          new Request(`https://example.test/api/on-device-batch${query}`),
        ),
      OnDeviceBatchValidationError,
    );
  }
});

test("route rejects unauthorized, non-JSON, malformed, and oversized requests", async () => {
  const previous = process.env[TOKEN_ENV];
  process.env[TOKEN_ENV] = "route-token";
  const authorization = {
    Authorization: "Bearer route-token",
    Origin: "https://example.test",
  };
  try {
    const unauthorized = await getBatchRoute(new Request("https://example.test/api/on-device-batch"));
    assert.equal(unauthorized.status, 401);
    assert.deepEqual(await unauthorized.json(), { error: "Unauthorized" });
    assert.equal(unauthorized.headers.get("www-authenticate"), "Bearer");
    assert.equal(unauthorized.headers.get("cache-control"), "private, no-store");

    const unknownGetVariant = await getBatchRoute(
      new Request("https://example.test/api/on-device-batch?variant=unknown", {
        headers: { Authorization: "Bearer route-token" },
      }),
    );
    assert.equal(unknownGetVariant.status, 400);
    assert.deepEqual(await unknownGetVariant.json(), {
      error: "variant must be linear-v1 or libswresample-wasm-v1",
    });

    const unknownPostVariant = await postBatchRoute(
      new Request("https://example.test/api/on-device-batch?variant=unknown", {
        method: "POST",
        headers: { Authorization: "Bearer route-token" },
      }),
    );
    assert.equal(unknownPostVariant.status, 400);

    for (const origin of [undefined, "https://attacker.test"] as const) {
      const headers: Record<string, string> = {
        Authorization: "Bearer route-token",
        "Content-Type": "application/json",
      };
      if (origin) headers.Origin = origin;
      const forbidden = await postBatchRoute(
        new Request("https://example.test/api/on-device-batch", {
          method: "POST",
          headers,
          body: "{}",
        }),
      );
      assert.equal(forbidden.status, 403);
      assert.deepEqual(await forbidden.json(), {
        error: "Origin does not match the application",
      });
    }

    const forwardedWrongType = await postBatchRoute(
      new Request("http://127.0.0.1:3001/api/on-device-batch", {
        method: "POST",
        headers: {
          ...authorization,
          Origin: "https://volleysplice.example",
          Host: "volleysplice.example",
          "X-Forwarded-Proto": "https",
          "Content-Type": "text/plain",
        },
        body: "{}",
      }),
    );
    assert.equal(forwardedWrongType.status, 415);

    const wrongType = await postBatchRoute(
      new Request("https://example.test/api/on-device-batch", {
        method: "POST",
        headers: { ...authorization, "Content-Type": "text/plain" },
        body: "{}",
      }),
    );
    assert.equal(wrongType.status, 415);

    const malformed = await postBatchRoute(
      new Request("https://example.test/api/on-device-batch", {
        method: "POST",
        headers: { ...authorization, "Content-Type": "application/json" },
        body: "{",
      }),
    );
    assert.equal(malformed.status, 400);
    assert.deepEqual(await malformed.json(), { error: "Request body must be valid JSON" });

    const oversized = await postBatchRoute(
      new Request("https://example.test/api/on-device-batch", {
        method: "POST",
        headers: {
          ...authorization,
          "Content-Type": "application/json",
          "Content-Length": String(1024 * 1024 + 1),
        },
        body: "{}",
      }),
    );
    assert.equal(oversized.status, 413);
  } finally {
    if (previous === undefined) delete process.env[TOKEN_ENV];
    else process.env[TOKEN_ENV] = previous;
  }
});

test("batch media route requires the bearer and restricts IDs to the fixed batch", async () => {
  const previous = process.env[TOKEN_ENV];
  process.env[TOKEN_ENV] = "media-route-token";
  const context = { params: Promise.resolve({ id: "not-in-the-fixed-batch" }) };
  try {
    const unauthorized = await getBatchMediaRoute(
      new Request("https://example.test/api/on-device-batch/media/not-in-the-fixed-batch"),
      context,
    );
    assert.equal(unauthorized.status, 401);
    assert.equal(unauthorized.headers.get("www-authenticate"), "Bearer");

    const unknown = await getBatchMediaRoute(
      new Request("https://example.test/api/on-device-batch/media/not-in-the-fixed-batch", {
        headers: { Authorization: "Bearer media-route-token" },
      }),
      context,
    );
    assert.equal(unknown.status, 404);
    assert.deepEqual(await unknown.json(), { error: "Video is not in the fixed batch" });
  } finally {
    if (previous === undefined) delete process.env[TOKEN_ENV];
    else process.env[TOKEN_ENV] = previous;
  }
});

test("batch output root is mandatory, absolute, existing, and not a filesystem root", async () => {
  const previous = process.env[OUTPUT_ROOT_ENV];
  const temporary = await fs.mkdtemp(path.join(os.tmpdir(), "volleycut-browser-output-"));
  try {
    delete process.env[OUTPUT_ROOT_ENV];
    await assert.rejects(configuredOnDeviceBatchOutputRoot(), OnDeviceBatchConfigurationError);
    process.env[OUTPUT_ROOT_ENV] = "relative/analyses";
    await assert.rejects(configuredOnDeviceBatchOutputRoot(), OnDeviceBatchConfigurationError);
    process.env[OUTPUT_ROOT_ENV] = path.parse(temporary).root;
    await assert.rejects(configuredOnDeviceBatchOutputRoot(), OnDeviceBatchConfigurationError);
    process.env[OUTPUT_ROOT_ENV] = temporary;
    assert.equal(await configuredOnDeviceBatchOutputRoot(), temporary);
  } finally {
    if (previous === undefined) delete process.env[OUTPUT_ROOT_ENV];
    else process.env[OUTPUT_ROOT_ENV] = previous;
    await fs.rm(temporary, { recursive: true, force: true });
  }
});

test("catalog exposes only fixed media metadata and completion state", async () => {
  const outputRoot = await fs.mkdtemp(path.join(os.tmpdir(), "volleycut-browser-catalog-"));
  const completedId = ON_DEVICE_BATCH_RECORDING_IDS[2];
  await fs.mkdir(
    path.join(outputRoot, `${LINEAR_ANALYSIS_PREFIX}${completedId}`),
  );
  await fs.writeFile(
    path.join(
      outputRoot,
      `${LINEAR_ANALYSIS_PREFIX}${completedId}`,
      "analysis.json",
    ),
    "{}",
  );
  try {
    const catalog = await buildOnDeviceBatchCatalog(
      ON_DEVICE_BATCH_RECORDING_IDS.map((id) => taskFor(id)),
      outputRoot,
    );
    assert.equal(catalog.schemaVersion, 1);
    assert.equal(catalog.modelId, ON_DEVICE_BATCH_MODEL_ID);
    assert.equal(catalog.featurePath, "training-proxy");
    assert.equal(catalog.runtimeVariant, "linear-v1");
    assert.deepEqual(catalog.videos.map(({ id }) => id), [...ON_DEVICE_BATCH_RECORDING_IDS]);
    assert.equal(catalog.videos.find(({ id }) => id === completedId)?.completed, true);
    assert.equal(catalog.videos.filter(({ completed }) => completed).length, 1);
    assert.deepEqual(Object.keys(catalog.videos[0]).sort(), [
      "completed",
      "duration",
      "environment",
      "filename",
      "id",
      "mediaUrl",
      "originalFilename",
      "roi",
      "size",
    ]);
    const serialized = JSON.stringify(catalog);
    assert.doesNotMatch(serialized, /rall(?:y|ies)|labels|document|contentSha256/);
    assert.equal(
      catalog.videos[0].mediaUrl,
      `/api/on-device-batch/media/${encodeURIComponent(ON_DEVICE_BATCH_RECORDING_IDS[0])}`,
    );
    assert.equal(
      onDeviceBatchAnalysisId(completedId, "libswresample-wasm-v1"),
      `${RESAMPLED_ANALYSIS_PREFIX}${completedId}`,
    );
    const resampledCatalog = await buildOnDeviceBatchCatalog(
      ON_DEVICE_BATCH_RECORDING_IDS.map((id) => taskFor(id)),
      outputRoot,
      "libswresample-wasm-v1",
    );
    assert.equal(resampledCatalog.runtimeVariant, "libswresample-wasm-v1");
    assert.equal(resampledCatalog.videos.filter(({ completed }) => completed).length, 0);
    await assert.rejects(
      buildOnDeviceBatchCatalog(
        ON_DEVICE_BATCH_RECORDING_IDS.slice(1).map((id) => taskFor(id)),
        outputRoot,
      ),
      OnDeviceBatchConfigurationError,
    );
  } finally {
    await fs.rm(outputRoot, { recursive: true, force: true });
  }
});

test("submission validation accepts only the fixed browser model and proxy media contract", () => {
  const task = taskFor("indoor-source-07");
  const submission = validateOnDeviceBatchSubmission(validBody(task.id), task);
  assert.equal(submission.recordingId, task.id);
  assert.equal(submission.modelId, ON_DEVICE_BATCH_MODEL_ID);
  assert.equal(submission.featurePath, "training-proxy");
  assert.equal(submission.runtimeVariant, "linear-v1");
  assert.equal(submission.media.duration, 100.02);
  assert.deepEqual(submission.intervals.map(({ id }) => id), ["R001", "R002"]);

  const urlSourceBody = validBody(task.id) as Record<string, unknown>;
  urlSourceBody.media = {
    ...(urlSourceBody.media as Record<string, unknown>),
    mimeType: 'video/mp4; codecs="avc1.42c01f, mp4a.40.2"',
    videoCodecString: "avc1.42c01f",
  };
  assert.equal(
    validateOnDeviceBatchSubmission(urlSourceBody, task).media.mimeType,
    'video/mp4; codecs="avc1.42c01f, mp4a.40.2"',
  );

  const invalidBodies = [
    { ...validBody(task.id) as Record<string, unknown>, modelId: "another-model" },
    { ...validBody(task.id) as Record<string, unknown>, featurePath: "raw-virtual-proxy" },
    { ...validBody(task.id) as Record<string, unknown>, runtimeVariant: "unknown-v1" },
    {
      ...validBody(task.id) as Record<string, unknown>,
      media: {
        ...(validBody(task.id) as { media: Record<string, unknown> }).media,
        width: 1920,
      },
    },
    {
      ...validBody(task.id) as Record<string, unknown>,
      media: {
        ...(validBody(task.id) as { media: Record<string, unknown> }).media,
        mimeType: "text/html",
      },
    },
    {
      ...validBody(task.id) as Record<string, unknown>,
      media: {
        ...(validBody(task.id) as { media: Record<string, unknown> }).media,
        duration: 101,
      },
    },
    {
      ...validBody(task.id) as Record<string, unknown>,
      intervals: [
        { id: "R001", start: 1, end: 8, confidence: 0.8, included: true },
        { id: "R002", start: 7, end: 9, confidence: 0.8, included: true },
      ],
    },
    {
      ...validBody(task.id) as Record<string, unknown>,
      intervals: [{ id: "R001", start: Number.NaN, end: 8, confidence: 0.8, included: true }],
    },
    {
      ...validBody(task.id) as Record<string, unknown>,
      provenance: {
        secureContext: false,
        userAgent: "Browser batch test",
        completedAt: "2026-08-13T20:00:00.000Z",
      },
    },
  ];
  for (const invalid of invalidBodies) {
    assert.throws(
      () => validateOnDeviceBatchSubmission(invalid, task),
      OnDeviceBatchValidationError,
    );
  }

  assert.throws(
    () =>
      validateOnDeviceBatchSubmission(
        validBody(task.id),
        task,
        "libswresample-wasm-v1",
      ),
    OnDeviceBatchValidationError,
  );
  assert.equal(
    validateOnDeviceBatchSubmission(
      validBody(task.id, 100, "libswresample-wasm-v1"),
      task,
      "libswresample-wasm-v1",
    ).runtimeVariant,
    "libswresample-wasm-v1",
  );
});

test("persistence creates one ordinary no-beach analysis atomically and append-only", async () => {
  const outputRoot = await fs.mkdtemp(path.join(os.tmpdir(), "volleycut-browser-save-"));
  const task = taskFor("indoor-source-07");
  const submission = validateOnDeviceBatchSubmission(
    validBody(task.id),
    task,
  ) as OnDeviceBatchSubmission;
  const previewDirectory = path.join(outputRoot, `${ON_DEVICE_BATCH_MODEL_ID}--${task.id}`);
  await fs.mkdir(previewDirectory);
  await fs.writeFile(path.join(previewDirectory, "court-preview.jpg"), "preview-fixture");
  try {
    const result = await persistOnDeviceBatchAnalysis(
      task,
      submission,
      outputRoot,
      "2026-08-13T20:01:00.000Z",
    );
    assert.deepEqual(result, {
      analysisId: `${LINEAR_ANALYSIS_PREFIX}${task.id}`,
      recordingId: task.id,
      rallyCount: 2,
    });
    const artifact = await readPersistedOnDeviceBatchAnalysis(
      outputRoot,
      task.id,
    ) as SavedArtifact;
    assert.equal(artifact.schemaVersion, 1);
    assert.equal(artifact.id, result.analysisId);
    assert.equal(artifact.recordingId, task.id);
    assert.equal(artifact.analysis.method, "browser-on-device-webcodecs-opencv-wasm-v1");
    assert.equal(artifact.analysis.modelVersion, ON_DEVICE_BATCH_MODEL_VERSION);
    assert.equal(artifact.analysis.variantLabel, `Browser on-device · ${ON_DEVICE_BATCH_MODEL_ID}`);
    assert.equal(artifact.analysis.modelBundleSha256, ON_DEVICE_BATCH_BUNDLE_SHA256);
    assert.equal(artifact.analysis.provenance.inferenceLocation, "browser");
    assert.equal(artifact.analysis.provenance.runtimeVariant, "linear-v1");
    assert.equal(
      artifact.analysis.provenance.audioResampler,
      "deterministic-linear-48khz-to-16khz",
    );
    assert.match(artifact.analysis.warnings.join(" "), /resampler/);
    assert.equal(artifact.rallies.length, 2);
    assert.equal(
      await fs.readFile(
        path.join(outputRoot, result.analysisId, "court-preview.jpg"),
        "utf8",
      ),
      "preview-fixture",
    );

    const parsed = parseAnalysis(artifact, { trainingCorpus: "without-beach" });
    assert.ok(parsed);
    assert.equal(parsed.kind, "model");
    assert.equal(parsed.trainingCorpus, "without-beach");
    assert.equal(parsed.modelVersion, ON_DEVICE_BATCH_MODEL_VERSION);
    assert.equal(parsed.variantLabel, `Browser on-device · ${ON_DEVICE_BATCH_MODEL_ID}`);
    assert.equal(parsed.rallies.length, 2);

    await assert.rejects(
      persistOnDeviceBatchAnalysis(task, submission, outputRoot),
      OnDeviceBatchConflictError,
    );
    const unchanged = await readPersistedOnDeviceBatchAnalysis(
      outputRoot,
      task.id,
    ) as SavedArtifact;
    assert.equal(unchanged.createdAt, "2026-08-13T20:01:00.000Z");
    assert.deepEqual(
      (await fs.readdir(outputRoot)).filter((name) => name.includes(".staging-")),
      [],
    );
  } finally {
    await fs.rm(outputRoot, { recursive: true, force: true });
  }
});

test("libswresample persistence uses a distinct append-only artifact identity", async () => {
  const outputRoot = await fs.mkdtemp(
    path.join(os.tmpdir(), "volleycut-browser-libswresample-save-"),
  );
  const task = taskFor("indoor-source-07");
  const runtimeVariant = "libswresample-wasm-v1";
  const submission = validateOnDeviceBatchSubmission(
    validBody(task.id, 100, runtimeVariant),
    task,
    runtimeVariant,
  );
  try {
    const result = await persistOnDeviceBatchAnalysis(
      task,
      submission,
      outputRoot,
      "2026-08-13T20:02:00.000Z",
    );
    assert.equal(
      result.analysisId,
      `${RESAMPLED_ANALYSIS_PREFIX}${task.id}`,
    );
    const artifact = await readPersistedOnDeviceBatchAnalysis(
      outputRoot,
      task.id,
      runtimeVariant,
    ) as SavedArtifact;
    assert.equal(
      artifact.analysis.method,
      "browser-on-device-webcodecs-opencv-libswresample-wasm-v1",
    );
    assert.equal(
      artifact.analysis.variantLabel,
      `Browser on-device · libswresample WASM · ${ON_DEVICE_BATCH_MODEL_ID}`,
    );
    assert.equal(artifact.analysis.provenance.runtimeVariant, runtimeVariant);
    assert.equal(
      artifact.analysis.provenance.audioResampler,
      "ffmpeg-libswresample-wasm",
    );
    assert.match(artifact.analysis.warnings.join(" "), /libswresample WASM/);
    await assert.rejects(
      readPersistedOnDeviceBatchAnalysis(outputRoot, task.id),
      { code: "ENOENT" },
    );
  } finally {
    await fs.rm(outputRoot, { recursive: true, force: true });
  }
});

test("an incomplete final directory conflicts without leaking a staging directory", async () => {
  const outputRoot = await fs.mkdtemp(path.join(os.tmpdir(), "volleycut-browser-conflict-"));
  const task = taskFor("indoor-source-07");
  const submission = validateOnDeviceBatchSubmission(validBody(task.id), task);
  const destination = path.join(
    outputRoot,
    `${LINEAR_ANALYSIS_PREFIX}${task.id}`,
  );
  await fs.mkdir(destination);
  try {
    await assert.rejects(
      persistOnDeviceBatchAnalysis(task, submission, outputRoot),
      OnDeviceBatchConflictError,
    );
    assert.deepEqual(await fs.readdir(destination), []);
    assert.deepEqual(
      (await fs.readdir(outputRoot)).filter((name) => name.includes(".staging-")),
      [],
    );
  } finally {
    await fs.rm(outputRoot, { recursive: true, force: true });
  }
});

test("beach artifacts carry an explicit out-of-distribution warning", async () => {
  const outputRoot = await fs.mkdtemp(path.join(os.tmpdir(), "volleycut-browser-beach-"));
  const task = taskFor("beach-source-02");
  const submission = validateOnDeviceBatchSubmission(validBody(task.id), task);
  try {
    await persistOnDeviceBatchAnalysis(task, submission, outputRoot);
    const artifact = await readPersistedOnDeviceBatchAnalysis(
      outputRoot,
      task.id,
    ) as SavedArtifact;
    assert.match(artifact.analysis.warnings.join(" "), /out of distribution/);
  } finally {
    await fs.rm(outputRoot, { recursive: true, force: true });
  }
});
