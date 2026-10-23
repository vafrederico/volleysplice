import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import React, { act } from "react";
import { createRoot } from "react-dom/client";

// Use the existing production headless-test dependencies, without serving Next.
const require = createRequire(new URL("../prod/package.json", import.meta.url));
const { JSDOM } = require("jsdom");
const { createServer } = await import(pathToFileURL(require.resolve("vite")).href);
const repository = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({ configFile: false, root: repository,
  resolve: { alias: { "@": repository } }, server: { middlewareMode: true, hmr: false }, appType: "custom" });
const dom = new JSDOM('<div id="root"></div>', { url: "https://example.test", pretendToBeVisual: true });
Object.assign(globalThis, { window: dom.window, document: dom.window.document, localStorage: dom.window.localStorage,
  Element: dom.window.Element, HTMLElement: dom.window.HTMLElement, IS_REACT_ACT_ENVIRONMENT: true,
  ResizeObserver: class { observe() {} disconnect() {} },
  requestAnimationFrame: callback => setTimeout(callback, 0), cancelAnimationFrame: clearTimeout });
Object.defineProperty(globalThis, "navigator", { value: dom.window.navigator, configurable: true });
window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
dom.window.HTMLElement.prototype.scrollIntoView = () => {};
dom.window.HTMLMediaElement.prototype.pause = () => {};
dom.window.HTMLCanvasElement.prototype.getContext = () => null;
dom.window.HTMLDialogElement.prototype.showModal = function () { this.open = true; };
const root = createRoot(document.getElementById("root"));
try {
  const { Trial } = await server.ssrLoadModule("/components/production-lab/workspace.tsx");
  const { labStorageKey, initialLabDraft } = await server.ssrLoadModule("/lib/production-editor-lab-draft.ts");
  const configuration = { id: "mode", label: "Synthetic", description: "Synthetic", sourcePolicy: "head_refined",
    events: [{ id: "R1", start: 10, end: 20, confidence: 0.9 }], proposals: [], removals: [], splits: [] };
  const task = { schemaVersion: 1, id: "synthetic", name: "fixture.mp4", durationSeconds: 60, mediaUrl: "/video",
    sourceRevision: "a".repeat(64), ignoredIntervals: [], configurations: [configuration],
    signals: { times: [0], live: [0], serve: [0], end: [0], keep: [0] }, serving: [], provenance: { labelBlind: true, sourceHashes: {} } };
  const key = labStorageKey(task, configuration);
  const mount = async name => act(async () => { root.render(React.createElement(Trial,
    { key: name, task, configuration, reportSave: () => {} })); });
  for (const raw of ["{broken", "", JSON.stringify({ ...initialLabDraft(task, configuration), version: 999 })]) {
    localStorage.setItem(key, raw);
    await mount(raw);
    assert.match(document.querySelector('[role="alert"]').textContent, /preserved/);
    assert.equal(document.querySelector(".taste-root"), null, "autosaving editor must not mount");
    assert.equal(localStorage.getItem(key), raw, "mount preserves exact bytes");
  }
  window.confirm = () => false;
  const reset = [...document.querySelectorAll("button")].find(button => button.textContent === "Start new review");
  await act(async () => reset.click());
  assert.equal(document.querySelector(".taste-root"), null, "cancel recovery keeps blocked state");
  const old = localStorage.getItem(key);
  window.confirm = () => true;
  await act(async () => reset.click());
  assert.ok(document.querySelector(".taste-root"), "explicit reset opens editor");
  const backupKey = Object.keys(localStorage).find(name => name.startsWith(`${key}:unrestored:`));
  assert.equal(JSON.parse(localStorage.getItem(backupKey)).draft, old);
  assert.ok(JSON.parse(localStorage.getItem(key)).cuts.length);
  const edited = { ...initialLabDraft(task, configuration), cuts: initialLabDraft(task, configuration).cuts.map(cut => ({ ...cut, included: false })) };
  localStorage.setItem(key, JSON.stringify(edited));
  await mount("valid-edited");
  assert.ok(document.querySelector(".taste-root"));
  assert.equal(JSON.parse(localStorage.getItem(key)).cuts[0].included, false);
  const storageDescriptor = Object.getOwnPropertyDescriptor(window, "localStorage");
  Object.defineProperty(window, "localStorage", { configurable: true, get() { throw new Error("denied"); } });
  await mount("storage-denied");
  assert.equal(document.querySelector(".taste-root"), null);
  assert.match(document.querySelector('[role="alert"]').textContent, /storage could not be read/);
  Object.defineProperty(window, "localStorage", storageDescriptor);
  const retry = [...document.querySelectorAll("button")].find(button => button.textContent === "Try restoring again");
  await act(async () => retry.click());
  assert.ok(document.querySelector(".taste-root"));
  assert.equal(JSON.parse(localStorage.getItem(key)).cuts[0].included, false);
  console.log("Lab UI: corrupt/empty/future drafts preserved; reset cancellation and backed-up recovery; valid edits restored.");
} finally { await act(async () => root.unmount()); await server.close(); dom.window.close(); }
