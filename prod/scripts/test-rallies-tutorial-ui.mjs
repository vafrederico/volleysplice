import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { JSDOM } from "jsdom";
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { createServer } from "vite";

const server = await createServer({ root: fileURLToPath(new URL("..", import.meta.url)),
  resolve: { dedupe: ["react", "react-dom"] }, server: { middlewareMode: true, hmr: false, ws: false }, appType: "custom" });
const dom = new JSDOM('<div id="root"></div>', { url: "https://example.test", pretendToBeVisual: true });
Object.assign(globalThis, { window: dom.window, document: dom.window.document, Element: dom.window.Element,
  HTMLElement: dom.window.HTMLElement, Node: dom.window.Node, IS_REACT_ACT_ENVIRONMENT: true,
  ResizeObserver: class { observe() {} disconnect() {} } });
Object.defineProperty(globalThis, "navigator", { value: dom.window.navigator, configurable: true });
window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
dom.window.HTMLElement.prototype.scrollIntoView = () => {};
dom.window.HTMLMediaElement.prototype.pause = () => {};
dom.window.HTMLMediaElement.prototype.play = async () => {};
dom.window.HTMLCanvasElement.prototype.getContext = () => null;
const root = createRoot(document.getElementById("root"));
const tour = () => document.querySelector("[data-rallies-tour-step]");
const step = () => tour()?.dataset.ralliesTourStep;
const click = async (name, scope = document) => {
  const node = [...scope.querySelectorAll("button")].find(node => node.getAttribute("aria-label") === name || node.textContent.trim() === name);
  assert.ok(node, `Missing ${name}`);
  await act(async () => node.click());
};
const key = async (key, options = {}) => act(async () => window.dispatchEvent(new window.KeyboardEvent("keydown", { key, bubbles: true, ...options })));
let saved;
try {
  const { RallyDesk } = await server.ssrLoadModule("/src/designs/taste/index.tsx");
  const { createCutDraft } = await server.ssrLoadModule("/src/lib/cut-draft.ts");
  const seed = { analysisId: "synthetic-rallies-tour", recordingId: "synthetic-rallies-tour", duration: 60,
    rallies: [{ id: "R001", start: 10, end: 20, confidence: .5, included: true }, { id: "R002", start: 30, end: 40, confidence: .5, included: false }], ignoredIntervals: [] };
  const draft = createCutDraft(seed);
  draft.scoreTracking.enabled = true;
  draft.scoreTracking.serveMarkers = [{ id: "S1", timestamp: 10, rallyId: "R001", side: "review", origin: "model", ignorePreviousPoint: false }];
  draft.pendingManualStart = 24;
  const review = { state: "ready", projectId: seed.recordingId, projectName: "Tutorial practice", sourceName: "Practice.mp4",
    sourceNeedsReconnect: true, sourceFile: null, videoUrl: null, productAnalysis: null,
    duration: 60, width: 320, height: 240, gameStart: 0, gameEnd: 60, cropCourt: false, sideSwitchEnabled: false,
    draft, modelDraft: draft, draftSeed: seed, cleanupSuggestions: [], projects: [], workActivity: [], exportJob: null,
    selectProject() {}, attachSource() {}, queueVideoExport() {}, saveDraft(value) { saved = value; return { ok: true }; } };
  await act(async () => root.render(React.createElement(RallyDesk, { review })));
  assert.equal(step(), "reconnect", "new users enter the Rallies walkthrough, including source reconnection");
  const baseline = structuredClone(saved);
  await key("r", { shiftKey: true });
  await key("Backspace");
  await key("Escape");
  assert.equal(tour(), null, "Escape closes the tutorial without cancelling an unfinished range");
  assert.deepEqual(saved, baseline, "keyboard edits are suspended during the walkthrough");
  await click("App menu");
  await click("Restart tutorial");
  assert.equal(step(), "reconnect", "the app menu restarts from the beginning");
  const targets = {
    reconnect: "rw-reconnect", review: "rw-review", decisions: "rw-decisions", trim: "rw-trim", serve: "rw-serve",
    navigation: "rw-navigation", "all-rallies": "rw-all-rallies", list: "rw-list", source: "rw-source", missing: "rw-missing",
    add: "rw-add", editor: "rd-video", export: "rw-export-tab", settings: "rw-settings",
  };
  const visited = [];
  while (tour()) {
    const current = step();
    visited.push(current);
    assert.ok(document.querySelector(`[data-tour="${targets[current]}"]`), `${current} navigates to a mounted target`);
    if (current === "all-rallies") assert.match(document.querySelector('[data-tour="rw-all-rallies"]').textContent, /All rallies/);
    if (current === "source") {
      assert.match(tour().textContent, /single filled circle.*outlined diamond/s);
      await click("Back", tour());
      assert.equal(step(), "list");
      await click("Next", tour());
      assert.equal(step(), "source");
    }
    await click(current === "settings" ? "Finish tutorial" : "Next", tour());
    assert.deepEqual(saved, baseline, `${current} changes views without changing any saved review state`);
  }
  assert.deepEqual(visited, Object.keys(targets));
  assert.equal(window.localStorage.getItem("volleycut:rallies-tour:v1"), "done");
  await click("Close final cut settings");
  await act(async () => root.render(React.createElement(RallyDesk, { key: "finished", review })));
  assert.equal(tour(), null, "completion persists after remounting");
  assert.equal(saved.pendingManualStart, 24, "an existing missing-rally draft survives the whole tour");

  window.localStorage.removeItem("volleycut:rallies-tour:v1");
  window.localStorage.setItem("volleycut:rally-desk-tour:v1", "dismissed");
  const noScore = structuredClone(draft);
  noScore.scoreTracking.enabled = false;
  noScore.pendingManualStart = null;
  await act(async () => root.render(React.createElement(RallyDesk, { key: "legacy", review: { ...review, draft: noScore, sourceNeedsReconnect: false, videoUrl: "blob:synthetic-tutorial" } })));
  assert.equal(tour(), null, "the previous tutorial’s dismissal is respected");
  await click("App menu"); await click("Restart tutorial");
  const scoreOffSteps = [];
  while (tour()) {
    scoreOffSteps.push(step());
    await click(step() === "settings" ? "Finish tutorial" : "Next", tour());
  }
  assert.ok(!scoreOffSteps.includes("serve"), "score-disabled projects skip unavailable serve controls");
  assert.ok(!scoreOffSteps.includes("reconnect"), "connected projects skip reconnection");

  window.localStorage.removeItem("volleycut:rallies-tour:v1");
  window.localStorage.removeItem("volleycut:rally-desk-tour:v1");
  const emptySeed = { ...seed, rallies: [] };
  const emptyDraft = createCutDraft(emptySeed);
  await act(async () => root.render(React.createElement(RallyDesk, { key: "empty", review: { ...review, draftSeed: emptySeed, draft: emptyDraft, modelDraft: emptyDraft } })));
  while (tour()) {
    const current = step();
    assert.ok(document.querySelector(`[data-tour="${targets[current]}"]`), `empty project has a target for ${current}`);
    await click(current === "settings" ? "Finish tutorial" : "Next", tour());
  }
  assert.equal(saved.cuts.length, 0, "the empty-project walkthrough never creates a rally");
  assert.equal(saved.pendingManualStart, null);
  console.log("Rallies tutorial: all native review steps, real navigation/targets, Back, Escape, menu restart, persisted progress, score-off/empty projects and no draft edits passed.");
} finally { await act(async () => root.unmount()); dom.window.close(); await server.close(); }
