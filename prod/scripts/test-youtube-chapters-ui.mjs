import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { JSDOM } from "jsdom";
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { createServer } from "vite";

// Exercise chapter export through the production Rally Desk without serving the app.
const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  resolve: { dedupe: ["react", "react-dom"] },
  server: { middlewareMode: true, hmr: false, ws: false },
  appType: "custom",
});
const dom = new JSDOM('<div id="root"></div>', { url: "https://example.test", pretendToBeVisual: true });
Object.assign(globalThis, {
  window: dom.window,
  document: dom.window.document,
  Element: dom.window.Element,
  HTMLElement: dom.window.HTMLElement,
  IS_REACT_ACT_ENVIRONMENT: true,
  ResizeObserver: class { observe() {} disconnect() {} },
  requestAnimationFrame: callback => setTimeout(callback, 0),
  cancelAnimationFrame: clearTimeout,
});
Object.defineProperty(globalThis, "navigator", { value: dom.window.navigator, configurable: true });
dom.window.HTMLDialogElement.prototype.showModal = function () { this.open = true; };
dom.window.HTMLDialogElement.prototype.close = function () { this.open = false; };
dom.window.HTMLElement.prototype.scrollIntoView = () => {};
dom.window.HTMLCanvasElement.prototype.getContext = () => null;
dom.window.HTMLMediaElement.prototype.pause = () => {};
window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
window.scrollTo = () => {};
window.localStorage.setItem("volleycut:rally-desk-tour:v1", "dismissed");
let copied = "";
let download;
let downloadedBlob;
Object.defineProperty(navigator, "clipboard", { value: { writeText: async (text) => { copied = text; } } });
const originalCreateURL = URL.createObjectURL;
const originalRevokeURL = URL.revokeObjectURL;
URL.createObjectURL = (blob) => { downloadedBlob = blob; return "blob:chapters"; };
URL.revokeObjectURL = () => {};
dom.window.HTMLAnchorElement.prototype.click = function () { download = this.download; };
const root = createRoot(document.getElementById("root"));

try {
  const { RallyDesk } = await server.ssrLoadModule("/src/designs/taste/index.tsx");
  const { createCutDraft, applyPaddingToCachedCuts } = await server.ssrLoadModule("/src/lib/cut-draft.ts");
  const { createScoreTracking, addServeMarker } = await server.ssrLoadModule("/src/lib/score-tracking.ts");
  const seed = {
    analysisId: "synthetic-chapters", recordingId: "synthetic-chapters", duration: 60,
    rallies: [10, 40].map((start, index) => ({ id: `R${index + 1}`, start, end: start + 10, confidence: 1, included: true })),
    ignoredIntervals: [],
  };
  const fixture = (scoreTracking) => {
    const draft = { ...applyPaddingToCachedCuts(createCutDraft(seed), 0, 0, 60), scoreTracking };
    return {
      state: "ready", projectId: seed.recordingId, projectName: "Match", sourceName: "Match.mp4",
      sourceSize: 1000, sourceNeedsReconnect: true, sourceFile: null, videoUrl: null, productAnalysis: null,
      duration: 60, width: 320, height: 240, gameStart: 0, gameEnd: 60,
      cropCourt: false, sideSwitchEnabled: false, projectStatus: "ready",
      draft, modelDraft: draft, draftSeed: seed, suppression: undefined, cleanupSuggestions: [],
      projects: [], workActivity: [], exportJob: null, selectProject() {},
      attachSource: async () => ({ ok: false, message: "No source in this fixture", videoUrl: null }),
      queueVideoExport() {}, saveDraft() {},
    };
  };
  const click = async (element) => { assert.ok(element); await act(async () => element.click()); };
  const button = (text) => [...document.querySelectorAll("button")].find((node) => node.textContent === text);
  const credit = () => [...document.querySelectorAll("label")].find((node) => node.textContent.includes("Include VolleySplice credit")).querySelector("input");
  const plain = "0:10 0–0 - Team 1 serving";
  const credited = `Edited with https://volleysplice.com\n\n${plain}`;
  const openChapters = async (scoreTracking, key) => {
    await act(async () => root.render(React.createElement(RallyDesk, { review: fixture(scoreTracking), key })));
    await click(button("Export"));
    await click(document.querySelector('input[value="chapters"]'));
  };
  await openChapters(addServeMarker(createScoreTracking(), 42, "near", { id: "S2", rallyId: "R2" }), "with-serve");
  assert.equal(credit().checked, true);
  assert.equal(document.querySelector("pre").textContent, credited);
  await click(button("Copy to clipboard"));
  assert.equal(copied, credited);
  await click(button("Download text file"));
  assert.equal(download, "Match-youtube-chapters.txt");
  assert.equal(await downloadedBlob.text(), `${credited}\n`);
  await click(credit());
  assert.equal(credit().checked, false);
  assert.equal(document.querySelector("pre").textContent, plain);
  await click(button("Download text file"));
  assert.equal(await downloadedBlob.text(), `${plain}\n`);
  await click(button("Copy to clipboard"));
  assert.equal(copied, plain);
  await openChapters(createScoreTracking(), "without-serves");
  assert.equal(button("Copy to clipboard").disabled, true);
  assert.equal(document.querySelector("pre").textContent, "No retained rally is available for chapters.");
  downloadedBlob = undefined;
  await click(button("Download text file"));
  assert.equal(downloadedBlob, undefined, "empty chapter export must not download a file");
  console.log("Rally Desk chapters: default credit, toggle, preview, clipboard, download, serve filtering, and empty export passed.");
} finally {
  await act(async () => root.unmount());
  await server.close();
  URL.createObjectURL = originalCreateURL;
  URL.revokeObjectURL = originalRevokeURL;
  dom.window.close();
}
