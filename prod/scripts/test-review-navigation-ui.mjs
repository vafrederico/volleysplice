import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { JSDOM } from "jsdom";
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { createServer } from "vite";

// Exercise the real keyboard handler without serving the production app.
const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  server: { middlewareMode: true, hmr: false },
  appType: "custom",
});
const dom = new JSDOM('<div id="root"></div>', {
  url: "https://example.test", pretendToBeVisual: true,
});
Object.assign(globalThis, {
  window: dom.window, document: dom.window.document,
  Element: dom.window.Element, HTMLElement: dom.window.HTMLElement,
  IS_REACT_ACT_ENVIRONMENT: true,
  ResizeObserver: class { observe() {} disconnect() {} },
  requestAnimationFrame: (callback) => setTimeout(callback, 0),
  cancelAnimationFrame: clearTimeout,
});
Object.defineProperty(globalThis, "navigator", { value: dom.window.navigator, configurable: true });
window.localStorage.setItem("volleysplice:review-workspace", "editor");
window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
dom.window.HTMLElement.prototype.scrollIntoView = () => {};
dom.window.HTMLCanvasElement.prototype.getContext = () => null;
dom.window.HTMLDialogElement.prototype.showModal = function () { this.open = true; };
dom.window.HTMLDialogElement.prototype.close = function () { this.open = false; };
window.localStorage.setItem("volleycut:rally-desk-tour:v1", "dismissed");
const root = createRoot(document.getElementById("root"));
const key = async (key) => act(async () => {
  document.body.dispatchEvent(new window.KeyboardEvent("keydown", { key, bubbles: true }));
});
const time = () => document.querySelector(".rd-live-score i")?.textContent;
const reviewNextAt = async (expected) => {
  await key("r");
  assert.equal(time(), expected);
};

try {
  const { RallyDesk } = await server.ssrLoadModule("/src/designs/taste/index.tsx");
  const { createCutDraft } = await server.ssrLoadModule("/src/lib/cut-draft.ts");
  const seed = {
    analysisId: "synthetic-timeline-review", recordingId: "synthetic-timeline-review",
    duration: 60, ignoredIntervals: [],
    rallies: [
      { id: "R001", start: 10, end: 18, confidence: 0.5, included: true, agreement: "neural" },
      { id: "R002", start: 30, end: 38, confidence: 0.5, included: true, agreement: "neural" },
      { id: "R003", start: 45, end: 48, confidence: 0.5, included: false, agreement: "neural" },
      { id: "R004", start: 50, end: 55, confidence: 0.5, included: true, agreement: "neural" },
    ],
  };
  const draft = createCutDraft(seed);
  draft.reviewedCutIds = ["R004"];
  draft.ignoredIntervals = [{ id: "ignored", start: 24, end: 26, reason: "other" }];
  draft.scoreTracking.serveMarkers = [
    { id: "serve-1", timestamp: 10, rallyId: "R001" },
    { id: "serve-between", timestamp: 20 },
    { id: "serve-ignored", timestamp: 25 },
    { id: "serve-2", timestamp: 30, rallyId: "R002" },
    { id: "serve-known", timestamp: 35, side: "near" },
    { id: "serve-excluded", timestamp: 45, rallyId: "R003" },
  ].map((marker) => ({ side: "review", origin: "manual", ignorePreviousPoint: false, ...marker }));
  let savedDraft = draft;
  const review = {
    state: "ready", projectId: seed.recordingId, projectName: "Timeline review",
    sourceName: "synthetic.mp4", sourceSize: 0, sourceNeedsReconnect: true,
    sourceFile: null, videoUrl: null, productAnalysis: null,
    duration: 60, width: 320, height: 240, gameStart: 0, gameEnd: 60,
    cropCourt: false, sideSwitchEnabled: false, projectStatus: "ready",
    draft, modelDraft: draft, draftSeed: seed, suppression: undefined,
    cleanupSuggestions: [], projects: [], workActivity: [], exportJob: null,
    selectProject() {}, attachSource() {}, queueVideoExport() {},
    saveDraft(value) { savedDraft = value; },
  };
  await act(async () => root.render(React.createElement(RallyDesk, { review })));
  await key("ArrowLeft");
  await key("ArrowLeft");
  assert.equal(time(), "0:00.0");

  // Each rally opens at its padded start, followed by its tied serve.
  for (const expected of ["0:08.0", "0:10.0", "0:20.0", "0:28.0", "0:30.0", "0:08.0"]) {
    await reviewNextAt(expected);
  }

  // Resolving a clip must still advance to its same-time serve.
  await key("k");
  assert.ok(savedDraft.reviewedCutIds.includes("R001"));
  await reviewNextAt("0:10.0");
  await key("n");
  assert.equal(savedDraft.scoreTracking.serveMarkers.find((marker) => marker.id === "serve-1").side, "near");
  await reviewNextAt("0:20.0");

  // Seeking abandons the old cursor. It must revisit this unresolved serve.
  await key("ArrowLeft");
  await reviewNextAt("0:20.0");
  await key("ArrowRight");
  await reviewNextAt("0:28.0");

  // Category buttons also seek; R then uses that position in the combined queue.
  const reviewClips = [...document.querySelectorAll("button")]
    .find((button) => button.querySelector("span")?.textContent === "Review clips");
  assert.ok(reviewClips);
  await act(async () => reviewClips.click());
  assert.equal(time(), "0:28.0");
  await reviewNextAt("0:28.0"); // Opens the clip for queue navigation from its padding.
  await reviewNextAt("0:30.0");

  console.log("Review navigation: timeline order, tied serves, padding, wrap, resolved items, seeking, category buttons, and ignored/excluded items passed.");
} finally {
  await act(async () => root.unmount());
  dom.window.close();
  await server.close();
}
