import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { GET as loadImport } from "../../app/api/model-feedback/[id]/route.ts";
import { GET as loadImportSource } from "../../app/api/model-feedback/[id]/source/route.ts";
import {
  POST as createImport,
  GET as listImports,
} from "../../app/api/model-feedback/route.ts";
import {
  ModelFeedbackValidationError,
  parseModelFeedback,
} from "../../lib/model-feedback.ts";
import {
  listModelFeedbackImports,
  ModelFeedbackSourceMismatchError,
  readModelFeedbackBundle,
  sampledSourceFingerprint,
  saveModelFeedbackImport,
} from "../../lib/server/model-feedback-store.ts";

function encoded(
  values: readonly number[],
  dataType: "float32" | "float64",
  shape: number[],
) {
  const bytes = Buffer.alloc(values.length * (dataType === "float32" ? 4 : 8));
  values.forEach((value, index) => {
    if (dataType === "float32") bytes.writeFloatLE(value, index * 4);
    else bytes.writeDoubleLE(value, index * 8);
  });
  return {
    encoding: "base64",
    byteOrder: "little-endian",
    dataType,
    shape,
    data: bytes.toString("base64"),
  };
}

function fixture(
  runtimeVariant = "libswresample-wasm-v1",
  sourceSize = 6,
  sampledFingerprint: string | null = null,
) {
  return {
    schema: "volleycut-model-feedback",
    schemaVersion: 1,
    generatedAt: "2026-08-15T20:00:00.000Z",
    source: {
      projectId: "match-one",
      analysisId: "analysis-one",
      timelineCoordinates: "seconds-from-start-of-source",
      file: {
        name: "match.mp4",
        sizeBytes: sourceSize,
        lastModifiedMs: 1_786_800_000_000,
        mimeType: "video/mp4",
        sampledFingerprint,
      },
      media: {
        duration: 30,
        mimeType: "video/mp4",
        width: 1920,
        height: 1080,
        rotation: 0,
        videoCodec: "avc",
        videoCodecString: "avc1.640028",
        canDecodeVideo: true,
        hasAudio: true,
        audioCodec: "aac",
        sampleRate: 48_000,
        channels: 2,
        canDecodeAudio: true,
      },
      gameWindow: { start: 2, end: 28 },
      featureRoi: { x: 0.03, y: 0.12, width: 0.94, height: 0.86 },
      runtimeVariant,
      videoBytesIncluded: false,
    },
    features: {
      analysisFps: 4,
      rows: 2,
      columns: 2,
      names: ["motion", "audio"],
      timestamps: encoded([2, 2.25], "float64", [2]),
      values: encoded([1.25, -2.5, 3.75, 4.5], "float32", [2, 2]),
    },
    initialInference: {
      modelId: "model-production-ensemble-v1",
      components: [{ modelId: "model-a", bundleSha256: "abc" }],
      ensembleAlgorithmVersion: "production-ensemble-v1",
      ranges: [
        { id: "R001", start: 5, end: 8, confidence: 0.8, included: true },
      ],
      probabilityModelId: "model-a",
      timestamps: encoded([2, 2.25], "float64", [2]),
      probabilities: {
        rally: encoded([0.1, 0.9], "float32", [2]),
        serve: encoded([0.2, 0.8], "float32", [2]),
        deadState: encoded([0.3, 0.7], "float32", [2]),
      },
    },
    corrections: {
      updatedAt: "2026-08-15T19:00:00.000Z",
      beforePaddingSeconds: 2,
      afterPaddingSeconds: 2,
      joinGapSeconds: 3,
      correctedRanges: [
        {
          id: "R001",
          coreStart: 5,
          coreEnd: 8,
          keepStart: 3,
          keepEnd: 10,
          confidence: 0.8,
          included: false,
          origin: "cached-label",
        },
        {
          id: "M001",
          coreStart: 12,
          coreEnd: 15,
          keepStart: 10,
          keepEnd: 17,
          confidence: 1,
          included: true,
          origin: "manual",
        },
      ],
      ignoredIntervals: [
        { id: "I001", start: 20, end: 22, reason: "camera-gap" },
      ],
      labels: {
        falsePositives: [{ id: "R001", start: 5, end: 8, confidence: 0.8 }],
        falseNegatives: [{ id: "M001", start: 12, end: 15, confidence: 1 }],
        confirmedModelRanges: [],
        discardedManualRanges: [],
      },
    },
    finalExportIntervals: [{ start: 10, end: 17, cutIds: ["M001"] }],
    warnings: [],
  };
}

test("feedback importer decodes production web features, traces, and labels", () => {
  const imported = parseModelFeedback(fixture());
  assert.equal(imported.producer, "production-web");
  assert.deepEqual(Array.from(imported.features!.timestamps), [2, 2.25]);
  assert.deepEqual(
    Array.from(imported.features!.values),
    [1.25, -2.5, 3.75, 4.5],
  );
  assert.ok(
    Math.abs(imported.initialInference.rallyProbabilities[1] - 0.9) < 1e-6,
  );
  assert.deepEqual(
    imported.corrections.labels.falsePositives.map(({ id }) => id),
    ["R001"],
  );
  assert.deepEqual(
    imported.corrections.labels.falseNegatives.map(({ id }) => id),
    ["M001"],
  );
});

test("feedback importer recognizes the Android producer using the shared contract", () => {
  const imported = parseModelFeedback(fixture("native-android-dsp-v1"));
  assert.equal(imported.producer, "android");
  assert.equal(imported.features?.columns, 2);
  assert.equal(imported.corrections.ignoredIntervals[0].reason, "camera-gap");
});

test("feedback importer rejects malformed numeric array shapes", () => {
  const malformed = fixture();
  malformed.features.values.shape = [4, 1];
  assert.throws(
    () => parseModelFeedback(malformed),
    (error) =>
      error instanceof ModelFeedbackValidationError &&
      /shape/.test(error.message),
  );
});

test("permanent imports verify the source and can be reopened", async () => {
  const temporary = await mkdtemp(path.join(tmpdir(), "volleycut-feedback-"));
  try {
    const sourcePath = path.join(temporary, "match.mp4");
    const storeRoot = path.join(temporary, "store");
    await writeFile(sourcePath, Buffer.from("source"));
    const fingerprint = await sampledSourceFingerprint(sourcePath);
    const bundleText = JSON.stringify(
      fixture("native-android-dsp-v1", 6, fingerprint),
    );
    const saved = await saveModelFeedbackImport(bundleText, sourcePath, {
      root: storeRoot,
      importedAt: "2026-08-15T21:00:00.000Z",
    });

    assert.equal(saved.producer, "android");
    assert.equal(saved.sourceLinked, true);
    assert.ok(saved.mediaUrl);
    assert.match(
      saved.mediaUrl,
      new RegExp(`/api/model-feedback/${saved.id}/source$`),
    );
    const catalog = await listModelFeedbackImports({ root: storeRoot });
    assert.deepEqual(
      catalog.map(({ id }) => id),
      [saved.id],
    );
    const reopened = await readModelFeedbackBundle(saved.id, {
      root: storeRoot,
    });
    assert.equal(reopened.metadata.sourcePath, sourcePath);
    assert.equal(
      JSON.parse(reopened.text).source.file.sampledFingerprint,
      fingerprint,
    );
    assert.equal(
      JSON.parse(
        await readFile(path.join(storeRoot, saved.id, "import.json"), "utf8"),
      ).sourcePath,
      sourcePath,
    );
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
});

test("feedback JSON can be saved and reopened without a linked source", async () => {
  const temporary = await mkdtemp(
    path.join(tmpdir(), "volleycut-feedback-json-only-"),
  );
  try {
    const storeRoot = path.join(temporary, "store");
    const bundleText = JSON.stringify(fixture());
    const saved = await saveModelFeedbackImport(bundleText, null, {
      root: storeRoot,
      importedAt: "2026-08-15T22:00:00.000Z",
    });

    assert.equal(saved.sourceLinked, false);
    assert.equal(saved.mediaUrl, null);
    const catalog = await listModelFeedbackImports({ root: storeRoot });
    assert.equal(catalog[0].id, saved.id);
    assert.equal(catalog[0].sourceLinked, false);
    const reopened = await readModelFeedbackBundle(saved.id, {
      root: storeRoot,
    });
    assert.equal(reopened.metadata.sourcePath, null);
    assert.equal(JSON.parse(reopened.text).source.projectId, "match-one");
    const metadata = JSON.parse(
      await readFile(path.join(storeRoot, saved.id, "import.json"), "utf8"),
    );
    assert.equal(metadata.sourcePath, null);
    assert.equal(metadata.sourceSize, null);
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
});

test("permanent imports reject a different source with the same byte length", async () => {
  const temporary = await mkdtemp(
    path.join(tmpdir(), "volleycut-feedback-mismatch-"),
  );
  try {
    const expectedPath = path.join(temporary, "expected.mp4");
    const selectedPath = path.join(temporary, "selected.mp4");
    await writeFile(expectedPath, Buffer.from("source"));
    await writeFile(selectedPath, Buffer.from("xxxxxx"));
    const fingerprint = await sampledSourceFingerprint(expectedPath);
    await assert.rejects(
      saveModelFeedbackImport(
        JSON.stringify(fixture("libswresample-wasm-v1", 6, fingerprint)),
        selectedPath,
        { root: path.join(temporary, "store") },
      ),
      ModelFeedbackSourceMismatchError,
    );
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
});

test("model-feedback routes save, reopen, and byte-range stream a permanent link", async () => {
  const temporary = await mkdtemp(
    path.join(tmpdir(), "volleycut-feedback-route-"),
  );
  const previousRoot = process.env.VOLLEYCUT_MODEL_FEEDBACK_ROOT;
  try {
    const sourcePath = path.join(temporary, "match.mp4");
    await writeFile(sourcePath, Buffer.from("source"));
    const fingerprint = await sampledSourceFingerprint(sourcePath);
    const bundleText = JSON.stringify(
      fixture("libswresample-wasm-v1", 6, fingerprint),
    );
    process.env.VOLLEYCUT_MODEL_FEEDBACK_ROOT = path.join(temporary, "store");
    const form = new FormData();
    form.set(
      "bundle",
      new File([bundleText], "match.model-feedback.json", {
        type: "application/json",
      }),
    );
    form.set("sourcePath", sourcePath);
    const created = await createImport(
      new Request("http://dev.test/api/model-feedback", {
        method: "POST",
        headers: { origin: "http://dev.test", host: "dev.test" },
        body: form,
      }),
    );
    assert.equal(created.status, 201);
    const createdPayload = (await created.json()) as {
      import: { id: string; mediaUrl: string };
    };

    const catalog = await listImports();
    assert.deepEqual(
      (
        (await catalog.json()) as { imports: Array<{ id: string }> }
      ).imports.map(({ id }) => id),
      [createdPayload.import.id],
    );
    const reopened = await loadImport(new Request("http://dev.test"), {
      params: Promise.resolve({ id: createdPayload.import.id }),
    });
    assert.equal(reopened.status, 200);
    assert.equal(
      decodeURIComponent(reopened.headers.get("X-VolleyCut-Source-Path") ?? ""),
      sourcePath,
    );
    assert.equal(
      JSON.parse(await reopened.text()).source.projectId,
      "match-one",
    );

    const streamed = await loadImportSource(
      new Request("http://dev.test", { headers: { range: "bytes=1-3" } }),
      { params: Promise.resolve({ id: createdPayload.import.id }) },
    );
    assert.equal(streamed.status, 206);
    assert.equal(streamed.headers.get("Content-Range"), "bytes 1-3/6");
    assert.equal(
      Buffer.from(await streamed.arrayBuffer()).toString("utf8"),
      "our",
    );
  } finally {
    if (previousRoot === undefined)
      delete process.env.VOLLEYCUT_MODEL_FEEDBACK_ROOT;
    else process.env.VOLLEYCUT_MODEL_FEEDBACK_ROOT = previousRoot;
    await rm(temporary, { recursive: true, force: true });
  }
});

test("model-feedback routes save JSON-only feedback and expose no source URL", async () => {
  const temporary = await mkdtemp(
    path.join(tmpdir(), "volleycut-feedback-route-json-only-"),
  );
  const previousRoot = process.env.VOLLEYCUT_MODEL_FEEDBACK_ROOT;
  try {
    process.env.VOLLEYCUT_MODEL_FEEDBACK_ROOT = path.join(temporary, "store");
    const form = new FormData();
    form.set(
      "bundle",
      new File([JSON.stringify(fixture())], "match.model-feedback.json", {
        type: "application/json",
      }),
    );
    const created = await createImport(
      new Request("http://dev.test/api/model-feedback", {
        method: "POST",
        headers: { origin: "http://dev.test", host: "dev.test" },
        body: form,
      }),
    );
    assert.equal(created.status, 201);
    const createdPayload = (await created.json()) as {
      import: { id: string; mediaUrl: null; sourceLinked: boolean };
    };
    assert.equal(createdPayload.import.mediaUrl, null);
    assert.equal(createdPayload.import.sourceLinked, false);

    const reopened = await loadImport(new Request("http://dev.test"), {
      params: Promise.resolve({ id: createdPayload.import.id }),
    });
    assert.equal(reopened.status, 200);
    assert.equal(reopened.headers.get("X-VolleyCut-Source-Path"), null);
    assert.equal(reopened.headers.get("X-VolleyCut-Source-Url"), null);

    const source = await loadImportSource(new Request("http://dev.test"), {
      params: Promise.resolve({ id: createdPayload.import.id }),
    });
    assert.equal(source.status, 404);
    assert.deepEqual(await source.json(), {
      error: "This feedback import has no linked source",
    });
  } finally {
    if (previousRoot === undefined)
      delete process.env.VOLLEYCUT_MODEL_FEEDBACK_ROOT;
    else process.env.VOLLEYCUT_MODEL_FEEDBACK_ROOT = previousRoot;
    await rm(temporary, { recursive: true, force: true });
  }
});
