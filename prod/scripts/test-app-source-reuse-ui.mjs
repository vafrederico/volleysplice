import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { IDBFactory } from "fake-indexeddb";
import { JSDOM } from "jsdom";
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { createServer } from "vite";

// Real App setup/reuse handlers, MP4 probing, fingerprinting and persistence.
// Only WebCodecs capability queries are supplied: no decode/inference is run and
// no application module is replaced. The tiny checked-in media is synthetic.
const server = await createServer({ root: fileURLToPath(new URL("..", import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false }, appType: "custom" });
const dom = new JSDOM('<div id="root"></div>', { url: "https://example.test", pretendToBeVisual: true });
Object.assign(globalThis, { window: dom.window, document: dom.window.document,
  Element: dom.window.Element, HTMLElement: dom.window.HTMLElement,
  IS_REACT_ACT_ENVIRONMENT: true, indexedDB: new IDBFactory(),
  ResizeObserver: class { observe() {} disconnect() {} },
  requestAnimationFrame: callback => setTimeout(callback, 0), cancelAnimationFrame: clearTimeout });
Object.defineProperty(globalThis, "navigator", { value: dom.window.navigator, configurable: true });
class CodecCapabilities { static async isConfigSupported(config) { return { supported: true, config }; } }
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
const urls = [];
URL.createObjectURL = file => { urls.push(file); return `blob:synthetic-source-${urls.length}`; };
URL.revokeObjectURL = () => {};
const root = createRoot(document.getElementById("root"));
const waitFor = async condition => {
  for (let i = 0; i < 150; i++) {
    let ready;
    await act(async () => { ready = await condition(); if (!ready) await new Promise(resolve => setTimeout(resolve, 5)); });
    if (ready) return;
  }
  assert.fail(`App did not settle: ${document.body.textContent.slice(-1500)}`);
};
const select = async id => {
  const input = document.querySelector('select[aria-label="Current review project"],select[aria-label="Selected project"]');
  assert.ok(input);
  await act(async () => { input.value = id; input.dispatchEvent(new window.Event("change", { bubbles: true })); });
};
const choose = async file => {
  const input = document.querySelector('input[type="file"][accept^="video/"]'); assert.ok(input);
  Object.defineProperty(input, "files", { value: [file], configurable: true });
  await act(async () => input.dispatchEvent(new window.Event("change", { bubbles: true })));
  await waitFor(() => document.querySelector('[aria-label="Rally detection model"]'));
};
const find = async () => { const button = document.querySelector('[data-tour="source-create"]'); assert.ok(button); await act(async () => button.click()); };
try {
  const { App } = await server.ssrLoadModule("/src/App.tsx");
  const store = await server.ssrLoadModule("/src/lib/project-store.ts");
  const { openLocalMedia } = await server.ssrLoadModule("/src/lib/on-device/media.ts");
  const { createCutDraft } = await server.ssrLoadModule("/src/lib/cut-draft.ts");
  const savedProject = async id => {
    let saved;
    await act(async () => { saved = (await store.listProjects()).find(p => p.id === id); });
    return saved;
  };
  const bytes = await readFile(new URL("../../tests/fixtures/source-fingerprint-synthetic.mp4", import.meta.url));
  const changed = Buffer.from(bytes);
  const mediaPayload = changed.indexOf("mdat"); assert.ok(mediaPayload >= 0);
  changed[mediaPayload + 12] ^= 1; // Change sample content while retaining the same container/timeline metadata.
  const original = new File([bytes], "synthetic-source-reuse.mp4", { type: "video/mp4", lastModified: 100 });
  const replacement = new File([changed], original.name, { type: original.type, lastModified: original.lastModified });
  const opened = await openLocalMedia(original); const info = opened.info; opened.input.dispose();
  const other = await openLocalMedia(replacement); assert.deepEqual(other.info, info); other.input.dispose();
  const source = { ...store.projectSource(original), fingerprint: await store.sourceFileFingerprint(original) };
  assert.notEqual(await store.sourceFileFingerprint(replacement), source.fingerprint);
  const project = { schemaVersion: 1, id: store.projectId(source, info, { start: 0, end: info.duration }, "high-recall"), source, info,
    analysisWindow: { start: 0, end: info.duration }, roi: { x: 0, y: 0, width: 1, height: 1 },
    rallyModel: "high-recall", sideSwitchEnabled: false, status: "ready", error: null,
    analysis: { modelId: "distilled-large-recall-v1", featurePath: "local-source",
      intervals: [{ id: "R001", start: .1, end: .9, confidence: .9, included: true, agreement: "neural" }],
      times: new Float64Array([0]), featureNames: ["synthetic"], featureValues: new Float32Array([0]),
      rallyProbabilities: new Float32Array([.9]), serveProbabilities: new Float32Array([.1]), deadStateProbabilities: new Float32Array([.1]) },
    createdAt: "2026-01-01T00:00:00.000Z", updatedAt: "2026-01-01T00:00:00.000Z" };
  project.reviewDraft = createCutDraft({ analysisId: store.projectAnalysisId(project), recordingId: project.id,
    duration: info.duration, rallies: project.analysis.intervals, ignoredIntervals: [] });
  project.reviewDraft.reviewedCutIds = ["R001"];
  project.reviewDraft.scoreTracking.team1Name = "Retained team";
  await store.putProject(project);
  window.localStorage.setItem(store.SELECTED_PROJECT_STORAGE_KEY, project.id);
  window.localStorage.setItem("volleycut:rally-desk-tour:v1", "dismissed");
  await act(async () => root.render(React.createElement(App)));
  await waitFor(() => document.querySelector(".td-reconnect"));
  await select("__new__"); await waitFor(() => document.querySelector('input[type="file"][accept^="video/"]'));
  const immutableBefore = await savedProject(project.id);
  await choose(replacement);
  const beforeRejectedReuse = urls.length;
  await find();
  await waitFor(() => document.body.textContent.includes("does not match"));
  assert.equal(document.querySelector('select[aria-label="Selected project"]').value, "__new__");
  assert.equal(urls.length, beforeRejectedReuse, "Rejected setup reuse must not link the candidate to the saved project");
  assert.deepEqual(await savedProject(project.id), immutableBefore, "Rejected setup reuse must not overwrite saved identity, analysis or review");
  await select(project.id); await waitFor(() => document.querySelector(".td-reconnect"));
  await act(async () => document.querySelector(".td-stages button:last-child").click());
  assert.equal([...document.querySelectorAll("button")].find(b => b.textContent === "Reconnect source for MP4").disabled, true);

  // The same setup with verified bytes reopens the immutable project, without inference.
  await select("__new__"); await waitFor(() => document.querySelector('input[type="file"][accept^="video/"]'));
  await choose(original); await find();
  await waitFor(() => document.querySelector('select[aria-label="Current review project"]')?.value === project.id && !document.querySelector(".td-reconnect"));
  const verifiedSaved = await savedProject(project.id);
  assert.deepEqual(verifiedSaved.analysis, project.analysis);
  assert.deepEqual(verifiedSaved.reviewDraft.reviewedCutIds, ["R001"]);
  assert.equal(verifiedSaved.reviewDraft.scoreTracking.team1Name, "Retained team");
  await act(async () => document.querySelector(".td-stages button:last-child").click());
  assert.equal([...document.querySelectorAll("button")].find(b => b.textContent === "Save MP4").disabled, false);

  // Legacy records retain their prior metadata-only contract through the setup path.
  await act(async () => root.render(null));
  const legacy = structuredClone(project); delete legacy.source.fingerprint;
  await store.putProject(legacy);
  await act(async () => root.render(React.createElement(App)));
  await waitFor(() => document.querySelector(".td-reconnect"));
  await select("__new__"); await waitFor(() => document.querySelector('input[type="file"][accept^="video/"]'));
  await choose(original); await find();
  await waitFor(() => document.querySelector('select[aria-label="Current review project"]')?.value === project.id && !document.querySelector(".td-reconnect"));
  assert.equal((await savedProject(project.id)).source.fingerprint, undefined);
  console.log("App setup reuse: same-metadata wrong bytes rejected, verified/legacy sources retain immutable IDs, analysis, review and MP4 gates.");
} finally {
  await act(async () => root.unmount()); await server.close(); dom.window.close();
  URL.createObjectURL = originalUrls.create; URL.revokeObjectURL = originalUrls.revoke;
}
