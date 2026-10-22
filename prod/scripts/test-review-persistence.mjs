import assert from "node:assert/strict";
import test, { beforeEach } from "node:test";
import { IDBFactory } from "fake-indexeddb";
import { createCutDraft, cutDraftStorageKey } from "../src/lib/cut-draft.ts";
import {
  deleteProject,
  listProjects,
  markProjectExported,
  projectAnalysisId,
  putProject,
  putProjectReviewDraft,
  sourceFileFingerprint,
  projectSource,
} from "../src/lib/project-store.ts";
import {
  restoreProjectReviewDraft,
  saveProjectReviewDraft,
} from "../src/lib/review-draft-persistence.ts";

beforeEach(() => {
  globalThis.indexedDB = new IDBFactory();
});

export function fixture() {
  const project = {
    schemaVersion: 1,
    id: "review-fixture",
    source: {
      name: "match.mp4",
      size: 1000,
      lastModified: 100,
      type: "video/mp4",
    },
    info: {
      duration: 90,
      width: 1920,
      height: 1080,
      mimeType: "video/mp4",
      videoCodec: "avc",
      canDecodeVideo: true,
      hasAudio: true,
      canDecodeAudio: true,
    },
    analysisWindow: { start: 0, end: 90 },
    roi: { x: 0, y: 0, width: 1, height: 1 },
    status: "ready",
    error: null,
    analysis: {
      modelId: "distilled-large-recall-v1",
      featurePath: "local-source",
      intervals: [10, 30].map((start, index) => ({
        id: `R00${index + 1}`,
        start,
        end: start + 5,
        confidence: 0.5,
        included: true,
        agreement: "neural",
      })),
      times: new Float64Array([0]),
      rallyProbabilities: new Float32Array([0.5]),
      serveProbabilities: new Float32Array([0.5]),
      deadStateProbabilities: new Float32Array([0.5]),
    },
    createdAt: "2026-01-01T00:00:00.000Z",
    updatedAt: "2026-01-01T00:00:00.000Z",
  };
  const seed = {
    analysisId: projectAnalysisId(project),
    recordingId: project.id,
    duration: 90,
    rallies: project.analysis.intervals,
    ignoredIntervals: [],
  };
  const edited = createCutDraft(seed);
  edited.cuts[1].included = false;
  edited.reviewedCutIds = ["R001", "R002"];
  edited.scoreTracking.serveMarkers = [
    {
      id: "serve-R001",
      timestamp: 10,
      side: "far",
      modelSide: "review",
      origin: "model",
      rallyId: "R001",
      ignorePreviousPoint: true,
    },
  ];
  edited.ignoredIntervals = [
    { id: "I001", start: 50, end: 60, reason: "camera-break" },
  ];
  const values = new Map();
  const storage = {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => {
      values.set(key, value);
    },
  };
  return { project, seed, edited, storage, values };
}

test("a legacy source can acquire a fingerprint without a later stale write downgrading it", async () => {
  const { project } = fixture();
  const file = new File(["original"], "synthetic-source.mp4", { type: "video/mp4", lastModified: 100 });
  project.source = projectSource(file);
  await putProject(project);
  const stale = structuredClone((await listProjects())[0]);
  const verified = { ...project.source, fingerprint: await sourceFileFingerprint(file) };
  await putProject({ ...project, source: verified });
  await putProject({ ...stale, status: "error", error: "Synthetic read failure" });
  const restored = (await listProjects())[0];
  assert.deepEqual(restored.source, verified);
  assert.equal(restored.status, "error");
  assert.equal(restored.error, "Synthetic read failure");
});

test("status and metadata updates cannot remove or replace a recorded source identity", async () => {
  const { project } = fixture();
  const original = new File(["original"], "synthetic-source.mp4", { type: "video/mp4", lastModified: 100 });
  const replacement = new File(["replaced"], original.name, { type: original.type, lastModified: original.lastModified });
  const verified = { ...projectSource(original), fingerprint: await sourceFileFingerprint(original) };
  project.source = verified;
  await putProject(project);
  const attempts = [
    { status: "queued", source: projectSource(original), normalizedStatus: "waiting" },
    { status: "analyzing", source: { ...verified, fingerprint: undefined }, normalizedStatus: "waiting" },
    { status: "error", source: { ...projectSource(replacement), fingerprint: await sourceFileFingerprint(replacement) }, normalizedStatus: "error" },
    { status: "ready", source: { ...verified, name: "synthetic-transfer.mp4", lastModified: 999, type: "application/octet-stream" }, normalizedStatus: "ready" },
  ];
  for (const { status, source, normalizedStatus } of attempts) {
    await putProject({ ...project, status, source });
    const restored = (await listProjects())[0];
    assert.deepEqual(restored.source, verified, `The ${status} update must retain the recorded source identity`);
    assert.equal(restored.id, project.id);
    assert.equal(restored.status, normalizedStatus);
  }
});

test("rally, review, serve and exclusion edits survive a complete project reload", async () => {
  const { project, seed, edited, storage } = fixture();
  await putProject(project);
  assert.equal(
    (await saveProjectReviewDraft(project.id, seed, edited, storage)).saved,
    true,
  );
  const [reloaded] = await listProjects();
  const restored = restoreProjectReviewDraft(reloaded, seed, storage);
  for (const key of [
    "cuts",
    "reviewedCutIds",
    "scoreTracking",
    "ignoredIntervals",
  ]) {
    assert.deepEqual(restored[key], edited[key]);
  }
});

test("a full localStorage falls back to the project record", async () => {
  const { project, seed, edited } = fixture();
  await putProject(project);
  const full = {
    setItem() {
      throw new DOMException("Full", "QuotaExceededError");
    },
  };
  assert.equal(
    (await saveProjectReviewDraft(project.id, seed, edited, full)).saved,
    true,
  );
  const [reloaded] = await listProjects();
  assert.deepEqual(
    restoreProjectReviewDraft(reloaded, seed, null).cuts,
    edited.cuts,
  );
});

test("a failed database write falls back to the synchronous draft", async () => {
  const { project, seed, edited, storage } = fixture();
  const database = globalThis.indexedDB;
  globalThis.indexedDB = {
    open() {
      throw new Error("Unavailable");
    },
  };
  try {
    assert.equal(
      (await saveProjectReviewDraft(project.id, seed, edited, storage)).saved,
      true,
    );
    assert.deepEqual(
      restoreProjectReviewDraft(project, seed, storage).scoreTracking,
      edited.scoreTracking,
    );
  } finally {
    globalThis.indexedDB = database;
  }
});

test("failure of both storage writes is reported and preserves the last successful save", async () => {
  const { project, seed, edited, storage, values } = fixture();
  await putProject(project);
  await saveProjectReviewDraft(project.id, seed, edited, storage);
  const before = [...values];
  const database = globalThis.indexedDB;
  globalThis.indexedDB = {
    open() {
      throw new Error("Unavailable");
    },
  };
  try {
    const result = await saveProjectReviewDraft(
      project.id,
      seed,
      createCutDraft(seed),
      {
        setItem() {
          throw new Error("Full");
        },
      },
    );
    assert.equal(result.saved, false);
    assert.match(result.message, /not saved.*Export a project backup/);
  } finally {
    globalThis.indexedDB = database;
  }
  assert.deepEqual([...values], before);
  assert.deepEqual((await listProjects())[0].reviewDraft.cuts, edited.cuts);
});

test("invalid edits cannot overwrite either previously saved copy", async () => {
  const { project, seed, edited, storage, values } = fixture();
  await putProject(project);
  await saveProjectReviewDraft(project.id, seed, edited, storage);
  const before = [...values];
  const invalid = {
    ...edited,
    ignoredIntervals: [...edited.ignoredIntervals, ...edited.ignoredIntervals],
  };
  assert.equal(
    (await saveProjectReviewDraft(project.id, seed, invalid, storage)).saved,
    false,
  );
  assert.deepEqual([...values], before);
  assert.deepEqual(
    (await listProjects())[0].reviewDraft.ignoredIntervals,
    edited.ignoredIntervals,
  );
});

test("incompatible saved drafts are preserved instead of replaced by initial model results", () => {
  const { project, seed, edited, storage, values } = fixture();
  const invalid = { ...edited, sourceRevision: "incompatible-revision" };
  project.reviewDraft = invalid;
  storage.setItem(cutDraftStorageKey(seed.analysisId), JSON.stringify(invalid));
  const before = [...values];
  assert.throws(
    () => restoreProjectReviewDraft(project, seed, storage),
    /preserved/,
  );
  assert.deepEqual([...values], before);
  assert.deepEqual(project.reviewDraft, invalid);
});

test("the newest valid copy wins when one backend has an older draft", () => {
  const { project, seed, edited, storage } = fixture();
  project.reviewDraft = { ...edited, updatedAt: "2026-01-02T00:00:00.000Z" };
  const old = {
    ...createCutDraft(seed),
    updatedAt: "2026-01-01T00:00:00.000Z",
  };
  storage.setItem(cutDraftStorageKey(seed.analysisId), JSON.stringify(old));
  assert.deepEqual(
    restoreProjectReviewDraft(project, seed, storage).cuts,
    edited.cuts,
  );
  const recent = { ...old, updatedAt: "2026-01-03T00:00:00.000Z" };
  storage.setItem(cutDraftStorageKey(seed.analysisId), JSON.stringify(recent));
  assert.deepEqual(
    restoreProjectReviewDraft(project, seed, storage).cuts,
    recent.cuts,
  );
});

test("an older valid draft cannot overwrite a newer incompatible review", () => {
  const { project, seed, edited, storage, values } = fixture();
  project.reviewDraft = { ...edited, updatedAt: "2026-01-01T00:00:00.000Z" };
  const raw = JSON.stringify({
    ...edited,
    sourceRevision: "incompatible-revision",
    updatedAt: "2026-01-02T00:00:00.000Z",
  });
  storage.setItem(cutDraftStorageKey(seed.analysisId), raw);
  assert.throws(
    () => restoreProjectReviewDraft(project, seed, storage),
    /preserved/,
  );
  assert.equal(values.get(cutDraftStorageKey(seed.analysisId)), raw);
});

test("background project updates cannot overwrite newer reviews or export metadata", async () => {
  const { project, seed, edited } = fixture();
  await putProject(project);
  edited.updatedAt = "2026-01-03T00:00:00.000Z";
  await putProjectReviewDraft(project.id, edited);
  await markProjectExported(project.id, "2026-01-03T01:00:00.000Z");
  for (const stale of [
    project,
    { ...project, reviewDraft: createCutDraft(seed) },
  ]) {
    await putProject({ ...stale, updatedAt: "2026-01-04T00:00:00.000Z" });
    const [reloaded] = await listProjects();
    assert.deepEqual(reloaded.reviewDraft, edited);
    assert.equal(reloaded.lastExportedAt, "2026-01-03T01:00:00.000Z");
  }
});

test("late review saves cannot replace newer edits or attach to another analysis", async () => {
  const { project, seed, edited } = fixture();
  await putProject(project);
  edited.updatedAt = "2026-01-03T00:00:00.000Z";
  await putProjectReviewDraft(project.id, edited);
  await putProjectReviewDraft(project.id, createCutDraft(seed));
  assert.deepEqual((await listProjects())[0].reviewDraft, edited);
  await assert.rejects(
    putProjectReviewDraft(project.id, {
      ...edited,
      analysisId: "other-analysis",
    }),
    /does not belong/,
  );
  assert.deepEqual((await listProjects())[0].reviewDraft, edited);
  await deleteProject(project.id);
  await assert.rejects(
    putProjectReviewDraft(project.id, edited),
    /unavailable/,
  );
  assert.deepEqual(await listProjects(), []);
});
