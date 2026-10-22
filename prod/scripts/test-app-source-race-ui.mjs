import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { IDBFactory } from "fake-indexeddb";
import { JSDOM } from "jsdom";
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { createServer } from "vite";

// Actual App and IndexedDB transactions. A second writer uses the real store
// API while App retains its first-tab legacy snapshot. No app module is replaced.
const server = await createServer({ root: fileURLToPath(new URL("..", import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false }, appType: "custom" });
const dom = new JSDOM('<div id="root"></div>', { url: "https://example.test", pretendToBeVisual: true });
Object.assign(globalThis, { window: dom.window, document: dom.window.document,
  Element: dom.window.Element, HTMLElement: dom.window.HTMLElement,
  IS_REACT_ACT_ENVIRONMENT: true, indexedDB: new IDBFactory(),
  ResizeObserver: class { observe() {} disconnect() {} },
  requestAnimationFrame: callback => setTimeout(callback, 0), cancelAnimationFrame: clearTimeout });
Object.defineProperty(globalThis, "navigator", { value: dom.window.navigator, configurable: true });
let codecSupported = true;
class CodecCapabilities { static async isConfigSupported(config) { return { supported: codecSupported, config }; } }
globalThis.VideoDecoder = window.VideoDecoder = CodecCapabilities;
globalThis.AudioDecoder = window.AudioDecoder = CodecCapabilities;
window.VideoFrame = class {};
window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
window.scrollTo = () => {}; window.isSecureContext = true;
dom.window.HTMLElement.prototype.scrollIntoView = () => {};
dom.window.HTMLCanvasElement.prototype.getContext = () => null;
dom.window.HTMLMediaElement.prototype.pause = () => {};
dom.window.HTMLDialogElement.prototype.showModal = function () { this.open = true; };
dom.window.HTMLDialogElement.prototype.close = function () { this.open = false; };
const originalUrls = { create: URL.createObjectURL, revoke: URL.revokeObjectURL };
const linked = [];
URL.createObjectURL = file => { linked.push(file); return `blob:synthetic-race-${linked.length}`; };
URL.revokeObjectURL = () => {};
const root = createRoot(document.getElementById("root"));
const waitFor = async condition => {
  for (let i = 0; i < 180; i++) {
    let ready;
    await act(async () => { ready = await condition(); if (!ready) await new Promise(resolve => setTimeout(resolve, 5)); });
    if (ready) return;
  }
  assert.fail(`App did not settle: ${document.body.textContent.slice(-1500)}`);
};
const choose = async file => {
  const input = document.querySelector('input[type="file"][accept^="video/"]'); assert.ok(input);
  Object.defineProperty(input, "files", { value: [file], configurable: true });
  await act(async () => input.dispatchEvent(new window.Event("change", { bubbles: true })));
};
const selectNew = async () => {
  const input = document.querySelector('select[aria-label="Current review project"],select[aria-label="Selected project"]'); assert.ok(input);
  await act(async () => { input.value = "__new__"; input.dispatchEvent(new window.Event("change", { bubbles: true })); });
};
try {
  const { App } = await server.ssrLoadModule("/src/App.tsx");
  const store = await server.ssrLoadModule("/src/lib/project-store.ts");
  const savedProject = async id => {
    let saved;
    await act(async () => { saved = (await store.listProjects()).find(p => p.id === id); });
    return saved;
  };
  const { openLocalMedia } = await server.ssrLoadModule("/src/lib/on-device/media.ts");
  const { createCutDraft } = await server.ssrLoadModule("/src/lib/cut-draft.ts");
  const bytes = await readFile(new URL("../../tests/fixtures/source-fingerprint-synthetic.mp4", import.meta.url));
  const changed = Buffer.from(bytes); changed[changed.indexOf("mdat") + 12] ^= 1;
  const file = data => new File([data], "synthetic-race.mp4", { type: "video/mp4", lastModified: 100 });
  const original = file(bytes);
  const media = await openLocalMedia(original); const info = media.info; media.input.dispose();
  const source = store.projectSource(original);
  const fingerprint = await store.sourceFileFingerprint(original);
  const base = { schemaVersion: 1, id: store.projectId(source, info, { start: 0, end: info.duration }, "high-recall"), source, info,
    analysisWindow: { start: 0, end: info.duration }, roi: { x: 0, y: 0, width: 1, height: 1 },
    rallyModel: "high-recall", sideSwitchEnabled: false, status: "ready", error: null,
    analysis: { modelId: "distilled-large-recall-v1", featurePath: "local-source",
      intervals: [{ id: "R001", start: .1, end: .9, confidence: .9, included: true, agreement: "neural" }],
      times: new Float64Array([0]), featureNames: ["synthetic"], featureValues: new Float32Array([0]),
      rallyProbabilities: new Float32Array([.9]), serveProbabilities: new Float32Array([.1]), deadStateProbabilities: new Float32Array([.1]) },
    createdAt: "2026-01-01T00:00:00.000Z", updatedAt: "2026-01-01T00:00:00.000Z" };
  base.reviewDraft = createCutDraft({ analysisId: store.projectAnalysisId(base), recordingId: base.id,
    duration: info.duration, rallies: base.analysis.intervals, ignoredIntervals: [] });
  base.reviewDraft.reviewedCutIds = ["R001"];
  for (const status of ["ready", "waiting"]) for (const path of ["attach", "setup"]) for (const timing of ["before", "during"]) {
    await act(async () => root.render(null));
    indexedDB = new IDBFactory(); window.localStorage.clear(); codecSupported = true;
    const legacy = { ...structuredClone(base), status, ...(status === "waiting" ? { analysis: null } : {}) };
    await store.putProject(legacy);
    window.localStorage.setItem(store.SELECTED_PROJECT_STORAGE_KEY, legacy.id);
    window.localStorage.setItem("volleycut:rally-desk-tour:v1", "dismissed");
    await act(async () => root.render(React.createElement(App)));
    await waitFor(() => status === "ready" ? document.querySelector(".td-reconnect") : document.body.textContent.includes("Choose original video"));
    const upgraded = { ...legacy, source: { ...source, fingerprint } };
    let raceWrite;
    const replacement = file(changed);
    if (path === "setup") {
      await selectNew(); await choose(replacement);
      await waitFor(() => document.querySelector('[data-tour="source-create"]'));
    }
    if (timing === "before") {
      await act(async () => { await store.putProject(upgraded); });
    } else {
      // Reading the last metadata field completes legacy validation. Start a
      // real second-writer transaction before App can confirm/write that source.
      let metadataReads = 0;
      Object.defineProperty(replacement, "lastModified", { get() {
        if (++metadataReads === (path === "setup" ? 2 : 1)) raceWrite = store.putProject(upgraded);
        return 100;
      } });
    }
    const before = linked.length;
    codecSupported = false; // Stop any incorrectly queued media before inference.
    if (path === "setup") await act(async () => document.querySelector('[data-tour="source-create"]').click());
    else await choose(replacement);
    await waitFor(() => /does not match|identity changed|source changed/i.test(document.body.textContent));
    if (timing === "during") { assert.ok(raceWrite, "The upgrade must occur during legacy validation"); await act(async () => raceWrite); }
    assert.equal(linked.length, before, `${status}/${path}/${timing}: stale legacy snapshot must not link wrong bytes`);
    const persisted = await savedProject(base.id);
    assert.equal(persisted.source.fingerprint, fingerprint);
    assert.equal(persisted.status, status);
    assert.deepEqual(persisted.analysis, legacy.analysis);
    assert.deepEqual(persisted.reviewDraft.reviewedCutIds, ["R001"]);
    if (status === "ready" && path === "attach") {
      assert.ok(document.querySelector(".td-reconnect"));
      await act(async () => document.querySelector(".td-stages button:last-child").click());
      assert.equal([...document.querySelectorAll("button")].find(b => b.textContent === "Reconnect source for MP4").disabled, true);
    }
    // Correct original bytes retry against the reconciled authoritative record.
    if (path === "setup") {
      codecSupported = true; await choose(original);
      await waitFor(() => document.querySelector('[data-tour="source-create"]'));
      codecSupported = false;
      await act(async () => document.querySelector('[data-tour="source-create"]').click());
    } else await choose(original);
    await waitFor(() => status === "ready"
      ? document.querySelector(".taste-rally-desk") && !document.querySelector(".td-reconnect")
      : [...document.querySelectorAll("button")].some(b => b.textContent === "Try again"));
    if (status === "ready") {
      await act(async () => document.querySelector(".td-stages button:last-child").click());
      assert.equal([...document.querySelectorAll("button")].find(b => b.textContent === "Save MP4").disabled, false);
    }
    assert.ok(linked.slice(before).every(value => value === original));
    assert.equal((await savedProject(base.id)).source.fingerprint, fingerprint);
  }
  for (const status of ["ready", "waiting"]) for (const path of ["attach", "setup"]) {
    await act(async () => root.render(null));
    indexedDB = new IDBFactory(); window.localStorage.clear(); codecSupported = true;
    const legacy = { ...structuredClone(base), status, ...(status === "waiting" ? { analysis: null } : {}) };
    await store.putProject(legacy);
    window.localStorage.setItem(store.SELECTED_PROJECT_STORAGE_KEY, legacy.id);
    window.localStorage.setItem("volleycut:rally-desk-tour:v1", "dismissed");
    await act(async () => root.render(React.createElement(App)));
    await waitFor(() => status === "ready" ? document.querySelector(".td-reconnect") : document.body.textContent.includes("Choose original video"));
    const candidate = file(bytes);
    if (path === "setup") {
      await selectNew(); await choose(candidate);
      await waitFor(() => document.querySelector('[data-tour="source-create"]'));
    }
    let raceWrite; let metadataReads = 0;
    const updatedAt = new Date(Date.now() + 100000).toISOString();
    const reviewDraft = structuredClone(legacy.reviewDraft);
    reviewDraft.updatedAt = updatedAt;
    reviewDraft.scoreTracking.team1Name = "Concurrent team";
    Object.defineProperty(candidate, "lastModified", { get() {
      if (++metadataReads === (path === "setup" ? 2 : 1)) {
        raceWrite = store.putProject({ ...legacy, reviewDraft, updatedAt, lastExportedAt: updatedAt });
      }
      return 100;
    } });
    const before = linked.length;
    codecSupported = false;
    if (path === "setup") await act(async () => document.querySelector('[data-tour="source-create"]').click());
    else await choose(candidate);
    await waitFor(() => status === "ready" ? document.querySelector(".taste-rally-desk") && !document.querySelector(".td-reconnect")
      : [...document.querySelectorAll("button")].some(b => b.textContent === "Try again"));
    assert.ok(raceWrite, "An unrelated write must actually occur during validation");
    await act(async () => raceWrite);
    assert.ok(linked.length > before);
    assert.ok(linked.slice(before).every(value => value === candidate));
    const persisted = await savedProject(base.id);
    assert.equal(persisted.source.fingerprint, status === "ready" ? undefined : fingerprint);
    assert.equal(persisted.reviewDraft.scoreTracking.team1Name, "Concurrent team");
    assert.equal(persisted.lastExportedAt, updatedAt);
  }
  console.log("App cross-tab source races: 8 ready/pending attach/setup identity conflicts reject wrong bytes and allow original retry; 4 unrelated concurrent saves preserve reviews/export metadata and permit attachment.");
} finally {
  await act(async () => root.unmount()); await server.close(); dom.window.close();
  URL.createObjectURL = originalUrls.create; URL.revokeObjectURL = originalUrls.revoke;
}
