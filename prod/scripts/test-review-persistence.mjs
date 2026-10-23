import assert from "node:assert/strict";
import test, { beforeEach } from "node:test";
import { IDBFactory } from "fake-indexeddb";
import { createCutDraft, cutDraftStorageKey } from "../src/lib/cut-draft.ts";
import {
  deleteProject,
  confirmProjectSource,
  getProject,
  listProjects,
  markProjectExported,
  projectAnalysisId,
  putProject,
  putProjectReviewDraft,
  sourceFileFingerprint,
  projectSource,
  ProjectSourceConflictError,
} from "../src/lib/project-store.ts";
import {
  restoreProjectReviewDraft,
  saveProjectReviewDraft,
  projectDraftStorageKey,
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
  Object.assign(project, await putProject(project));
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
  Object.assign(project, await putProject(project));
  const attempts = [
    { status: "queued", source: projectSource(original), normalizedStatus: "ready" },
    { status: "analyzing", source: { ...verified, fingerprint: undefined }, normalizedStatus: "ready" },
    { status: "error", source: { ...projectSource(replacement), fingerprint: await sourceFileFingerprint(replacement) }, conflict: true },
    { status: "ready", source: { ...verified, name: "synthetic-transfer.mp4", lastModified: 999, type: "application/octet-stream" }, normalizedStatus: "ready" },
  ];
  for (const { status, source, normalizedStatus, conflict } of attempts) {
    const previous = (await listProjects())[0];
    if (conflict) await assert.rejects(putProject({ ...project, status, source }), ProjectSourceConflictError);
    else await putProject({ ...project, status, source });
    const restored = (await listProjects())[0];
    assert.deepEqual(restored.source, verified, `The ${status} update must retain the recorded source identity`);
    assert.equal(restored.id, project.id);
    assert.equal(restored.status, conflict ? previous.status : normalizedStatus);
    assert.deepEqual(restored.analysis, project.analysis, "Checkpointed rally evidence remains ready under PR13 recovery");
  }
});

test("source confirmations and bound writes reject an upgrade after validation and return current identity", async () => {
  const { project } = fixture();
  Object.assign(project, await putProject(project));
  const checked = await getProject(project.id);
  const source = { ...project.source, fingerprint: `sampled-sha256-v1:${"1".repeat(64)}` };
  await putProject({ ...project, source });
  for (const operation of [
    () => confirmProjectSource(project.id, checked.source, checked.instanceId),
    () => putProject({ ...checked, status: "queued" }, { expectedSource: checked.source }),
  ]) {
    await assert.rejects(operation(), cause => cause instanceof ProjectSourceConflictError &&
      cause.project.source.fingerprint === source.fingerprint);
    assert.equal((await getProject(project.id)).status, "ready");
  }
  const saved = await putProject({ ...checked, updatedAt: "2026-01-04T00:00:00.000Z" });
  assert.deepEqual(saved.source, source, "Ordinary background writes return the authoritative persisted source");
  assert.deepEqual((await confirmProjectSource(project.id, source, project.instanceId)).source, source);
});

test("source CAS permits unrelated concurrent review and export writes", async () => {
  const { project, edited } = fixture();
  Object.assign(project, await putProject(project));
  const checked = await getProject(project.id);
  await putProjectReviewDraft(project.id, edited, project.instanceId);
  await markProjectExported(project.id, "2026-01-05T00:00:00.000Z", project.instanceId);
  const queued = await putProject({ ...checked, status: "queued" }, { expectedSource: checked.source });
  assert.deepEqual(queued.reviewDraft, edited);
  assert.equal(queued.lastExportedAt, "2026-01-05T00:00:00.000Z");
  assert.equal(queued.status, "queued");
});

test("source CAS prevents conflicting creation and resurrection after deletion", async () => {
  const { project } = fixture();
  Object.assign(project, await putProject(project, { expectedSource: null }));
  await assert.rejects(putProject(project, { expectedSource: null }), ProjectSourceConflictError);
  await deleteProject(project.id, project.instanceId);
  await assert.rejects(putProject(project, { expectedSource: project.source }), ProjectSourceConflictError);
  await assert.rejects(confirmProjectSource(project.id, project.source, project.instanceId), ProjectSourceConflictError);
  assert.equal(await getProject(project.id), null);
});

test("rally, review, serve and exclusion edits survive a complete project reload", async () => {
  const { project, seed, edited, storage } = fixture();
  Object.assign(project, await putProject(project));
  assert.equal(
    (await saveProjectReviewDraft(project.id, seed, edited, storage, project.instanceId)).saved,
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

test("same-source recreation rejects old source-bound completions and neutral writes without touching the new instance", async () => {
  const { project, edited } = fixture();
  project.source.fingerprint = `sampled-sha256-v1:${"a".repeat(64)}`;
  Object.assign(project, await putProject(project));
  const old = structuredClone(project);
  await deleteProject(old.id, old.instanceId);
  const fresh = await putProject({ ...project, reviewDraft: undefined,
    createdAt: "2026-02-01T00:00:00.000Z", updatedAt: "2026-02-01T00:00:00.000Z" }, { expectedSource: null });
  assert.notEqual(fresh.instanceId, old.instanceId);
  for (const operation of [
    ...["analyzing", "ready", "error"].map(status => () => putProject({ ...old, status,
      analysis: status === "error" ? null : old.analysis, reviewDraft: edited }, { expectedSource: old.source })),
    () => putProject({ ...old, reviewDraft: edited }),
    () => confirmProjectSource(old.id, old.source, old.instanceId),
    () => putProjectReviewDraft(old.id, edited, old.instanceId),
    () => markProjectExported(old.id, "2026-03-01T00:00:00.000Z", old.instanceId),
    () => deleteProject(old.id, old.instanceId),
  ]) {
    await assert.rejects(operation(), ProjectSourceConflictError);
    assert.deepEqual(await getProject(fresh.id), fresh);
  }
  await putProject({ ...fresh, sideSwitchEnabled: false });
  await putProjectReviewDraft(fresh.id, edited, fresh.instanceId);
  await markProjectExported(fresh.id, "2026-03-01T00:00:00.000Z", fresh.instanceId);
  const updated = await getProject(fresh.id);
  assert.equal(updated.instanceId, fresh.instanceId);
  assert.equal(updated.createdAt, fresh.createdAt);
  assert.equal(updated.sideSwitchEnabled, false);
  assert.deepEqual(updated.reviewDraft, edited);
  assert.equal(updated.lastExportedAt, "2026-03-01T00:00:00.000Z");
});

test("legacy migration assigns one atomic instance and recreated projects never restore deleted local drafts", async () => {
  const { project, seed, edited, storage, values } = fixture();
  const request = indexedDB.open("volleycut-projects", 1);
  request.onupgradeneeded = () => request.result.createObjectStore("projects", { keyPath: "id" });
  const db = await new Promise((resolve, reject) => { request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error); });
  const transaction = db.transaction("projects", "readwrite");
  transaction.objectStore("projects").put(project);
  await new Promise((resolve, reject) => { transaction.oncomplete = resolve; transaction.onerror = () => reject(transaction.error); });
  db.close();
  storage.setItem(cutDraftStorageKey(seed.analysisId), JSON.stringify(edited));
  const [first, listed, second] = await Promise.all([getProject(project.id), listProjects(), getProject(project.id)]);
  assert.ok(first.instanceId);
  assert.equal(first.instanceId, listed[0].instanceId); assert.equal(first.instanceId, second.instanceId);
  assert.equal(first.legacyDraftStorage, true);
  assert.deepEqual(restoreProjectReviewDraft(first, seed, storage).cuts, edited.cuts);
  const same = await putProject({ ...first, sideSwitchEnabled: false });
  assert.equal(same.instanceId, first.instanceId);
  await deleteProject(first.id, first.instanceId);
  const recreated = await putProject({ ...project }, { expectedSource: null });
  assert.notEqual(recreated.instanceId, first.instanceId);
  assert.equal(recreated.legacyDraftStorage, false);
  assert.ok(restoreProjectReviewDraft(recreated, seed, storage).cuts.every(cut => cut.included));
  const late = await saveProjectReviewDraft(first.id, seed, edited, storage, first.instanceId);
  assert.equal(late.saved, false);
  assert.match(late.message, /instance.*changed/);
  assert.ok(values.has(projectDraftStorageKey(seed.analysisId, first.instanceId)));
  assert.equal(values.has(projectDraftStorageKey(seed.analysisId, recreated.instanceId)), false);
  assert.deepEqual(await getProject(recreated.id), recreated);
  assert.ok(restoreProjectReviewDraft(recreated, seed, storage).cuts.every(cut => cut.included));
});

test("a full localStorage falls back to the project record", async () => {
  const { project, seed, edited } = fixture();
  Object.assign(project, await putProject(project));
  const full = {
    setItem() {
      throw new DOMException("Full", "QuotaExceededError");
    },
  };
  assert.equal(
    (await saveProjectReviewDraft(project.id, seed, edited, full, project.instanceId)).saved,
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
      (await saveProjectReviewDraft(project.id, seed, edited, storage, project.instanceId)).saved,
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
  Object.assign(project, await putProject(project));
  await saveProjectReviewDraft(project.id, seed, edited, storage, project.instanceId);
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
  Object.assign(project, await putProject(project));
  await saveProjectReviewDraft(project.id, seed, edited, storage, project.instanceId);
  const before = [...values];
  const invalid = {
    ...edited,
    ignoredIntervals: [...edited.ignoredIntervals, ...edited.ignoredIntervals],
  };
  assert.equal(
    (await saveProjectReviewDraft(project.id, seed, invalid, storage, project.instanceId)).saved,
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
  Object.assign(project, await putProject(project));
  edited.updatedAt = "2026-01-03T00:00:00.000Z";
  await putProjectReviewDraft(project.id, edited, project.instanceId);
  await markProjectExported(project.id, "2026-01-03T01:00:00.000Z", project.instanceId);
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
  Object.assign(project, await putProject(project));
  edited.updatedAt = "2026-01-03T00:00:00.000Z";
  await putProjectReviewDraft(project.id, edited, project.instanceId);
  await putProjectReviewDraft(project.id, createCutDraft(seed), project.instanceId);
  assert.deepEqual((await listProjects())[0].reviewDraft, edited);
  await assert.rejects(
    putProjectReviewDraft(project.id, {
      ...edited,
      analysisId: "other-analysis",
    }, project.instanceId),
    /does not belong/,
  );
  assert.deepEqual((await listProjects())[0].reviewDraft, edited);
  await deleteProject(project.id, project.instanceId);
  await assert.rejects(
    putProjectReviewDraft(project.id, edited, project.instanceId),
    /unavailable/,
  );
  assert.deepEqual(await listProjects(), []);
});
