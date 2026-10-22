import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { IDBFactory } from "fake-indexeddb";
import { JSDOM } from "jsdom";
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { createServer } from "vite";

// Exercise actual App/RallyDesk reconnect handlers and Web Crypto with synthetic
// File bytes. No application module, fingerprint computation or asset is mocked.
// JSDOM does not decode/play the attached video; the URL observer verifies whether
// the app links a rejected source, while the real export UI verifies its gate.
const server = await createServer({ root: fileURLToPath(new URL("..", import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false }, appType: "custom" });
const dom = new JSDOM('<div id="root"></div>', { url: "https://example.test", pretendToBeVisual: true });
Object.assign(globalThis, { window: dom.window, document: dom.window.document,
  Element: dom.window.Element, HTMLElement: dom.window.HTMLElement,
  IS_REACT_ACT_ENVIRONMENT: true, indexedDB: new IDBFactory(),
  ResizeObserver: class { observe() {} disconnect() {} },
  requestAnimationFrame: callback => setTimeout(callback, 0), cancelAnimationFrame: clearTimeout });
Object.defineProperty(globalThis, "navigator", { value: dom.window.navigator, configurable: true });
window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
window.scrollTo = () => {}; window.isSecureContext = true;
window.VideoDecoder = window.AudioDecoder = window.VideoFrame = class {};
dom.window.HTMLElement.prototype.scrollIntoView = () => {};
dom.window.HTMLCanvasElement.prototype.getContext = () => null;
dom.window.HTMLMediaElement.prototype.pause = () => {};
dom.window.HTMLDialogElement.prototype.showModal = function () { this.open = true; };
dom.window.HTMLDialogElement.prototype.close = function () { this.open = false; };
const originalUrls = { create: URL.createObjectURL, revoke: URL.revokeObjectURL };
const linkedFiles = [];
URL.createObjectURL = file => { linkedFiles.push(file); return `blob:synthetic-source-${linkedFiles.length}`; };
URL.revokeObjectURL = () => {};
const root = createRoot(document.getElementById("root"));
const waitFor = async condition => {
  for (let i = 0; i < 150; i++) {
    let ready;
    await act(async () => { ready = await condition(); if (!ready) await new Promise(resolve => setTimeout(resolve, 5)); });
    if (ready) return;
  }
  assert.fail(`App did not settle: ${document.body.textContent.slice(-1000)}`);
};
const click = async selector => { const target = document.querySelector(selector); assert.ok(target, selector); await act(async () => target.click()); };
const attach = async file => {
  const input = document.querySelector('.td-reconnect input[type="file"]'); assert.ok(input);
  Object.defineProperty(input, "files", { value: [file], configurable: true });
  await act(async () => input.dispatchEvent(new window.Event("change", { bubbles: true })));
};
const select = async id => {
  const input = document.querySelector('select[aria-label="Current review project"]'); assert.ok(input);
  await act(async () => { input.value = id; input.dispatchEvent(new window.Event("change", { bubbles: true })); });
  await waitFor(() => document.querySelector('select[aria-label="Current review project"]')?.value === id);
};
try {
  const { App } = await server.ssrLoadModule("/src/App.tsx");
  const store = await server.ssrLoadModule("/src/lib/project-store.ts");
  const original = new File(["original"], "synthetic-reconnect.mp4", { type: "video/mp4", lastModified: 100 });
  const replacement = new File(["replaced"], original.name, { type: original.type, lastModified: original.lastModified });
  const fingerprint = await store.sourceFileFingerprint(original);
  const first = { schemaVersion: 1, id: "synthetic-app-fingerprinted",
    source: { ...store.projectSource(original), fingerprint },
    info: { duration: 2, width: 320, height: 240, mimeType: "video/mp4", videoCodec: "avc", canDecodeVideo: true, hasAudio: true, canDecodeAudio: true },
    analysisWindow: { start: 0, end: 2 }, roi: { x: 0, y: 0, width: 1, height: 1 },
    rallyModel: "high-recall", sideSwitchEnabled: false, status: "ready", error: null,
    analysis: { modelId: "distilled-large-recall-v1", featurePath: "local-source",
      intervals: [{ id: "R001", start: .25, end: 1.75, confidence: .9, included: true, agreement: "neural" }],
      times: new Float64Array([0]), featureNames: ["synthetic"], featureValues: new Float32Array([0]),
      rallyProbabilities: new Float32Array([.9]), serveProbabilities: new Float32Array([.1]), deadStateProbabilities: new Float32Array([.1]) },
    createdAt: "2026-01-01T00:00:00.000Z", updatedAt: "2026-01-01T00:00:00.000Z" };
  const legacy = structuredClone(first); legacy.id = "synthetic-app-legacy";
  legacy.source.name = "synthetic-legacy.mp4"; delete legacy.source.fingerprint;
  Object.assign(first, await store.putProject(first)); Object.assign(legacy, await store.putProject(legacy));
  window.localStorage.setItem(store.SELECTED_PROJECT_STORAGE_KEY, first.id);
  window.localStorage.setItem("volleycut:rally-desk-tour:v1", "dismissed");
  await act(async () => root.render(React.createElement(App)));
  await waitFor(() => document.querySelector(".td-reconnect"));
  await click(".td-stages button:last-child");
  assert.equal([...document.querySelectorAll("button")].find(b => b.textContent === "Reconnect source for MP4").disabled, true);
  await waitFor(async () => (await store.listProjects()).find(p => p.id === first.id).reviewDraft);
  const saved = JSON.stringify((await store.listProjects()).find(p => p.id === first.id).reviewDraft);
  assert.equal(store.sourceMatchesFile(first.source, replacement), true);
  await attach(replacement);
  await waitFor(() => document.querySelector(".td-reconnect")?.textContent.includes("does not match"));
  assert.equal(linkedFiles.length, 0, "Metadata collision must never link the wrong bytes");
  assert.equal(JSON.stringify((await store.listProjects()).find(p => p.id === first.id).reviewDraft), saved);
  const unreadable = new File(["original"], original.name, { type: original.type, lastModified: original.lastModified });
  Object.defineProperty(unreadable, "slice", { value: () => { throw new Error("Synthetic read failure"); } });
  await attach(unreadable);
  await waitFor(() => document.querySelector(".td-reconnect")?.textContent.includes("Could not reconnect"));
  assert.equal(linkedFiles.length, 0);
  assert.equal([...document.querySelectorAll("button")].find(b => b.textContent === "Reconnect source for MP4").disabled, true);
  assert.equal(JSON.stringify((await store.listProjects()).find(p => p.id === first.id).reviewDraft), saved);
  // A second read used to reach a separately rejecting App check whose void
  // return was ignored by the hook. One authoritative validation removes that
  // fallible duplicate gate; it must never be asked to hash this File twice.
  const oneRead = new File(["original"], original.name, { type: original.type, lastModified: original.lastModified });
  let reads = 0;
  Object.defineProperty(oneRead, "slice", { value: (...args) => {
    if (++reads > 1) throw new Error("Synthetic second read failure");
    return original.slice(...args);
  } });
  await attach(oneRead);
  await waitFor(() => document.querySelector(".taste-rally-desk") && !document.querySelector(".td-reconnect"));
  assert.equal(reads, 1, "App must authoritatively validate once before the hook links a source");
  await click(".td-stages button:last-child");
  assert.equal([...document.querySelectorAll("button")].find(b => b.textContent === "Save MP4").disabled, false);
  assert.ok(linkedFiles.every(file => file === oneRead));

  // Reopening clears the tab's source link. A transfer with different metadata
  // still connects by its persisted fingerprint and leaves analysis/review intact.
  await act(async () => root.render(null));
  await act(async () => root.render(React.createElement(App)));
  await waitFor(() => document.querySelector(".td-reconnect"));
  const transferred = new File(["original"], "synthetic-transferred.mp4", { type: "video/mp4", lastModified: 999 });
  await attach(transferred);
  await waitFor(() => document.querySelector(".taste-rally-desk") && !document.querySelector(".td-reconnect"));
  assert.equal((await store.listProjects()).find(p => p.id === first.id).source.fingerprint, fingerprint);
  assert.deepEqual((await store.listProjects()).find(p => p.id === first.id).analysis, first.analysis);

  await select(legacy.id); await waitFor(() => document.querySelector(".td-reconnect"));
  const beforeLegacy = linkedFiles.length;
  await attach(new File(["original"], legacy.source.name, { type: "video/mp4", lastModified: 999 }));
  assert.ok(document.querySelector(".td-reconnect")); assert.equal(linkedFiles.length, beforeLegacy);
  await attach(new File(["original"], legacy.source.name, { type: "video/mp4", lastModified: 100 }));
  await waitFor(() => document.querySelector(".taste-rally-desk") && !document.querySelector(".td-reconnect"));
  console.log("App reconnect: fingerprint beats metadata, wrong/unreadable bytes remain disconnected, exact/transfer/legacy matches reconnect and enable MP4.");
} finally {
  await act(async () => root.unmount()); await server.close(); dom.window.close();
  URL.createObjectURL = originalUrls.create; URL.revokeObjectURL = originalUrls.revoke;
}
