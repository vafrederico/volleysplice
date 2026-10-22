import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { IDBFactory } from "fake-indexeddb";
import { JSDOM } from "jsdom";
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { createServer } from "vite";

// Actual App/RallyDesk export and file-input import, with real serialization,
// validation and persistence. Seed a completed synthetic analysis; do not run
// inference, encode video or replace any application module.
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
window.scrollTo = () => {};
window.isSecureContext = true;
window.VideoDecoder = window.AudioDecoder = window.VideoFrame = class {};
dom.window.HTMLElement.prototype.scrollIntoView = () => {};
dom.window.HTMLCanvasElement.prototype.getContext = () => null;
dom.window.HTMLMediaElement.prototype.pause = () => {};
dom.window.HTMLDialogElement.prototype.showModal = function () { this.open = true; };
dom.window.HTMLDialogElement.prototype.close = function () { this.open = false; };
dom.window.HTMLAnchorElement.prototype.click = () => {};
const originalUrls = { create: URL.createObjectURL, revoke: URL.revokeObjectURL };
let downloaded;
URL.createObjectURL = blob => { downloaded = blob; return "blob:synthetic-project"; };
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
const click = async selector => {
  const target = document.querySelector(selector); assert.ok(target, selector);
  await act(async () => target.click());
};
const clickText = async text => {
  const target = [...document.querySelectorAll("button")].find(node => node.textContent === text);
  assert.ok(target, text); await act(async () => target.click());
};
try {
  const { App } = await server.ssrLoadModule("/src/App.tsx");
  const store = await server.ssrLoadModule("/src/lib/project-store.ts");
  const { PRODUCTION_ENSEMBLE_MODEL_ID } = await server.ssrLoadModule("/src/lib/on-device/ensemble.ts");
  const duration = 1.0026666666666666;
  const project = { schemaVersion: 1, id: "synthetic-app-roundtrip",
    source: { name: "synthetic-fractional.mp4", size: 1000, lastModified: 100, type: "video/mp4" },
    info: { duration, width: 320, height: 240, rotation: 0, mimeType: "video/mp4", videoCodec: "avc", videoCodecString: null,
      canDecodeVideo: true, hasAudio: false, audioCodec: null, sampleRate: null, channels: null, canDecodeAudio: false },
    analysisWindow: { start: 0, end: duration }, roi: { x: 0, y: 0, width: 1, height: 1 },
    rallyModel: "ensemble", sideSwitchEnabled: false, status: "ready", error: null,
    analysis: { modelId: PRODUCTION_ENSEMBLE_MODEL_ID, featurePath: "local-source",
      intervals: [{ id: "R001", start: .1, end: .9, confidence: .9, included: true, agreement: "both-models" }],
      times: new Float64Array([0]), rallyProbabilities: new Float32Array([.9]),
      serveProbabilities: new Float32Array([.1]), deadStateProbabilities: new Float32Array([.1]) },
    createdAt: "2026-01-01T00:00:00.000Z", updatedAt: "2026-01-01T00:00:00.000Z" };
  await store.putProject(project);
  window.localStorage.setItem(store.SELECTED_PROJECT_STORAGE_KEY, project.id);
  window.localStorage.setItem("volleycut:rally-desk-tour:v1", "dismissed");
  await act(async () => root.render(React.createElement(App)));
  await waitFor(() => document.querySelector(".taste-rally-desk"));
  const download = async () => {
    await click(".td-stages button:last-child"); await click('input[value="feedback"]');
    downloaded = undefined; await clickText("Save project"); assert.ok(downloaded);
    return JSON.parse(await downloaded.text());
  };
  const exported = await download();
  assert.deepEqual(exported.finalExportIntervals, [{ start: 0, end: duration, cutIds: ["R001"] }]);
  assert.equal(exported.corrections.correctedRanges[0].keepEnd, duration);
  const select = document.querySelector('select[aria-label="Current review project"]');
  await act(async () => { select.value = "__new__"; select.dispatchEvent(new window.Event("change", { bubbles: true })); });
  await waitFor(() => document.querySelector('input[accept="application/json,.json"]'));
  const input = document.querySelector('input[accept="application/json,.json"]');
  Object.defineProperty(input, "files", { value: [new File([JSON.stringify(exported)], "synthetic-project.json", { type: "application/json" })], configurable: true });
  await act(async () => input.dispatchEvent(new window.Event("change", { bubbles: true })));
  await waitFor(async () => (await store.listProjects()).some(p => p.importedFeedback));
  await waitFor(() => document.querySelector(".taste-rally-desk"));
  const imported = (await store.listProjects()).find(p => p.importedFeedback);
  assert.equal(imported.reviewDraft.cuts[0].keepEnd, duration);
  const reexported = await download();
  assert.deepEqual(reexported.finalExportIntervals, exported.finalExportIntervals, "Save/open/save must preserve exact fractional bounds");
  assert.equal(document.body.textContent.includes("Could not open that saved project"), false);
  console.log("App project roundtrip: unmodified Save project output imports and re-exports within fractional source bounds.");
} finally {
  await act(async () => root.unmount()); await server.close(); dom.window.close();
  URL.createObjectURL = originalUrls.create; URL.revokeObjectURL = originalUrls.revoke;
}
