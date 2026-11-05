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
  ResizeObserver: class { observe() {} disconnect() {} }, requestAnimationFrame: callback => setTimeout(callback, 0), cancelAnimationFrame: clearTimeout });
Object.defineProperty(globalThis, "navigator", { value: dom.window.navigator, configurable: true });
window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
window.localStorage.setItem("volleycut:rally-desk-tour:v1", "dismissed");
dom.window.HTMLElement.prototype.scrollIntoView = () => {};
dom.window.HTMLMediaElement.prototype.pause = () => {};
dom.window.HTMLCanvasElement.prototype.getContext = () => null;
const root = createRoot(document.getElementById("root"));
const button = (name, scope = document) => [...scope.querySelectorAll("button")].find(node =>
  node.getAttribute("aria-label") === name || node.textContent.trim() === name || node.childNodes[0]?.textContent?.trim() === name);
const click = async (name, scope) => { const node = button(name, scope); assert.ok(node, `Missing ${name}`); await act(async () => node.click()); };
const key = async (key, options = {}) => act(async () => window.dispatchEvent(new window.KeyboardEvent("keydown", { key, bubbles: true, ...options })));
let saved;
try {
  const { RallyDesk } = await server.ssrLoadModule("/src/designs/taste/index.tsx");
  const { createCutDraft } = await server.ssrLoadModule("/src/lib/cut-draft.ts");
  const seed = { analysisId: "synthetic-rally-workspace", recordingId: "synthetic-rally-workspace", duration: 60,
    rallies: [{ id: "R001", start: 10, end: 20, confidence: .4, included: true }, { id: "R002", start: 30, end: 40, confidence: .4, included: true }], ignoredIntervals: [] };
  const draft = createCutDraft(seed);
  draft.beforePaddingSeconds = 1; draft.afterPaddingSeconds = 1;
  draft.scoreTracking.enabled = true;
  draft.scoreTracking.serveMarkers = [{ id: "S1", timestamp: 10, rallyId: "R001", side: "review", origin: "model", ignorePreviousPoint: false }];
  const review = { state: "ready", projectId: seed.recordingId, projectName: "Practice", sourceName: "Practice.mp4",
    sourceNeedsReconnect: false, sourceFile: null, videoUrl: null, productAnalysis: null,
    duration: 60, width: 320, height: 240, gameStart: 0, gameEnd: 60, cropCourt: false, sideSwitchEnabled: false,
    draft, modelDraft: draft, draftSeed: seed, cleanupSuggestions: [], projects: [], workActivity: [], exportJob: null,
    selectProject() {}, attachSource() {}, queueVideoExport() {}, saveDraft(value) { saved = value; return { ok: true }; } };
  await act(async () => root.render(React.createElement(RallyDesk, { review })));
  assert.match(document.querySelector(".rw-heading").textContent, /Reviewing 2 rallies/);
  assert.match(document.querySelector(".td-video-reconnect").textContent, /Reconnect your video.*Practice.mp4/s);
  assert.equal(button("Play").disabled, true, "Play is disabled without a video URL, even before a reconnect flag is set");
  assert.equal(document.querySelector(".td-production-score-overlay"), null, "disconnected playback does not show a floating scoreboard");
  const pausedPlayhead = document.querySelector('[aria-label="Source playhead"]').value;
  await key(" ");
  await act(async () => { await new Promise(resolve => setTimeout(resolve, 1100)); });
  assert.equal(document.querySelector('[aria-label="Source playhead"]').value, pausedPlayhead, "Space cannot advance a disconnected video");
  assert.equal(button("Pause"), undefined);
  await click("Keep");
  assert.ok(saved.reviewedCutIds.includes("R001"));
  assert.equal(button("Next review").disabled, true, "serve decision remains pending after Keep");
  await key("r");
  assert.match(document.querySelector(".rw-bound-head h2").textContent, /Rally 1/, "R respects the disabled Next review button");
  await key("f");
  assert.equal(saved.scoreTracking.serveMarkers[0].side, "far", "F keeps the web binding in Rallies");
  assert.equal(button("Next review").disabled, false);
  let reviewClicks = 0;
  button("Next review").addEventListener("click", () => reviewClicks++);
  await key("r");
  assert.equal(reviewClicks, 1, "R clicks the actual Next review button");
  assert.match(document.querySelector(".rw-bound-head h2").textContent, /Rally 2/);
  await key("Backspace");
  assert.equal(saved.cuts[1].included, false);
  await key("z", { ctrlKey: true });
  assert.equal(saved.cuts[1].included, true, "same undo history in focused view");
  await click("Move rally end 1 second later");
  assert.equal(saved.cuts[1].coreEnd, 41);
  await click("Editor");
  assert.equal(button("Play").disabled, true, "Editor shares the disconnected playback guard");
  assert.equal(window.localStorage.getItem("volleysplice:review-workspace"), "editor");
  assert.match(document.querySelector(".td-clip-inspector").textContent, /R002/);
  assert.match(document.querySelector(".td-range-editor").textContent, /0:41.0/);
  await click("Rallies");
  await click("All rallies (2) →");
  await key("r");
  assert.ok(document.querySelector('[aria-label="All rallies browser"]'), "R does nothing when Next review is absent");
  await click("Open rally 2");
  await click("Edit rally 2");
  await click("Not a serve");
  await key("n");
  assert.equal(saved.scoreTracking.serveMarkers.find(marker => marker.rallyId === "R002").side, "near", "N can add a missing rally serve");
  await click("Not a serve");
  assert.ok(!saved.scoreTracking.serveMarkers.some(marker => marker.rallyId === "R002"));
  await click("← Review rallies");
  await click("All rallies (2) →");
  await click("+ Add missing rally");
  await key("r", { shiftKey: true });
  assert.notEqual(saved.pendingManualStart, null);
  await key("ArrowRight");
  await key("r", { shiftKey: true });
  assert.equal(saved.cuts.length, 3, "original missed-rally shortcut adds to the shared draft");
  await key("z", { ctrlKey: true });
  assert.equal(saved.cuts.length, 2);
  await key("Escape");
  assert.equal(saved.pendingManualStart, null);
  const standaloneDraft = structuredClone(draft);
  standaloneDraft.reviewedCutIds = ["R001", "R002"];
  standaloneDraft.scoreTracking.serveMarkers = [{ id: "outside", timestamp: 50, side: "review", origin: "model", ignorePreviousPoint: false }];
  await act(async () => root.render(React.createElement(RallyDesk, { key: "standalone", review: {
    ...review, projectId: "synthetic-outside-serve", draft: standaloneDraft, modelDraft: standaloneDraft,
  } })));
  assert.match(document.querySelector(".rw-heading").textContent, /Reviewing 1 rally/);
  assert.ok(document.querySelector('[aria-label="Serve outside a rally"]'));
  await key("r");
  assert.ok(document.querySelector('[aria-label="Serve outside a rally"]'), "R cannot skip an unresolved standalone serve");
  await key("f");
  assert.equal(saved.scoreTracking.serveMarkers[0].side, "far");
  await key("r");
  assert.match(document.querySelector(".rw-complete").textContent, /All checked/);

  const markerDraft = structuredClone(draft);
  markerDraft.beforePaddingSeconds = 0; markerDraft.afterPaddingSeconds = 0;
  markerDraft.cuts = markerDraft.cuts.map(cut => ({ ...cut, keepStart: cut.coreStart, keepEnd: cut.coreEnd }));
  await act(async () => root.render(React.createElement(RallyDesk, { key: "timeline-markers", review: { ...review, draft: markerDraft } })));
  await click("All rallies (2) →");
  const playheadMarker = () => {
    const markers = document.querySelectorAll(".rw-source-map [data-playhead-marker]");
    assert.equal(markers.length, 1, "the source overview has exactly one playback circle");
    return markers[0];
  };
  assert.equal(playheadMarker().dataset.playheadMarker, "1");
  await click("Open rally 2");
  assert.equal(document.querySelector('[aria-label="Source playhead"]').value, "30");
  assert.equal(playheadMarker().dataset.playheadMarker, "2", "the exact midpoint belongs only to the second column");
  for (let i = 0; i < 7; i++) await key("ArrowRight");
  assert.equal(playheadMarker().dataset.playheadMarker, "2");
  assert.equal(playheadMarker().style.top, "100%", "recording end retains one circle at the end of column two");
  await click("Open rally 1");
  assert.equal(playheadMarker().dataset.playheadMarker, "1", "seeking backward moves the same indicator back to column one");
  const list = document.querySelector(".rw-rally-list");
  Object.defineProperty(list, "clientHeight", { value: 100, configurable: true });
  list.getBoundingClientRect = () => ({ top: 0, bottom: 100 });
  list.children[0].getBoundingClientRect = () => ({ top: -80, bottom: 0 });
  list.children[1].getBoundingClientRect = () => ({ top: 0, bottom: 80 });
  const beforeScroll = document.querySelector('[aria-label="Source playhead"]').value;
  const beforeScrollDraft = structuredClone(saved);
  await act(async () => {
    list.dispatchEvent(new window.WheelEvent("wheel", { bubbles: true, deltaY: 80 }));
    list.dispatchEvent(new window.Event("scroll", { bubbles: true }));
  });
  for (const selector of [".rw-rally-list", ".rw-source-map", ".rw-source-slider"]) {
    assert.equal(document.querySelector(`${selector} [data-rally-location]`).dataset.rallyLocation, "R002", "matching diamonds follow the centered list rally");
  }
  assert.equal(document.querySelector('[aria-label="Source playhead"]').value, beforeScroll, "list scrolling never seeks the video");
  assert.deepEqual(saved, beforeScrollDraft, "list scrolling never edits the draft");
  await click("Edit rally 1");
  assert.equal(document.querySelector(".rw-bounds [data-rally-location]").dataset.rallyLocation, "R001");
  await click("Editor");
  assert.equal(document.querySelectorAll(".rd-timeline [data-rally-location]").length, 1);
  assert.equal(document.querySelector(".rd-timeline [data-rally-location]").dataset.rallyLocation, "R001");

  let playRequests = 0;
  dom.window.HTMLMediaElement.prototype.play = async () => { playRequests++; };
  function ReconnectHarness({ available = true }) {
    const [connected, setConnected] = React.useState(false);
    return React.createElement(RallyDesk, { review: { ...review,
      projectId: "synthetic-reconnect-workspace", sourceNeedsReconnect: !available || !connected,
      videoUrl: available && connected ? "blob:synthetic-reconnected-video" : null,
      attachSource: async file => {
        if (file.name !== review.sourceName) return { ok: false, videoUrl: null, message: "That video does not match the saved project." };
        setConnected(true);
        return { ok: true, videoUrl: "blob:synthetic-reconnected-video", message: "Original video reconnected." };
      },
    } });
  }
  await act(async () => root.render(React.createElement(ReconnectHarness, { key: "reconnect" })));
  let pickerClicks = 0;
  const input = document.querySelector('.td-video-reconnect input[type="file"]');
  input.addEventListener("click", () => pickerClicks++);
  await click("Reconnect video");
  assert.equal(pickerClicks, 1, "the prominent action opens the original-video picker");
  async function pickSource(name) {
    Object.defineProperty(input, "files", { value: [new dom.window.File(["synthetic"], name, { type: "video/mp4" })], configurable: true });
    await act(async () => input.dispatchEvent(new window.Event("change", { bubbles: true })));
  }
  const beforeReconnect = structuredClone(saved.cuts);
  await pickSource("wrong.mp4");
  assert.match(document.querySelector(".td-video-reconnect-status").textContent, /does not match/);
  assert.equal(button("Play").disabled, true);
  assert.deepEqual(saved.cuts, beforeReconnect, "failed reconnection preserves edits");
  await pickSource("Practice.mp4");
  assert.equal(document.querySelector(".td-video-reconnect"), null);
  assert.equal(button("Play").disabled, false, "matching video restores playback");
  assert.deepEqual(saved.cuts, beforeReconnect);
  await key(" ");
  assert.ok(button("Pause"));
  assert.equal(playRequests, 1, "Space starts the connected media element");
  await act(async () => root.render(React.createElement(ReconnectHarness, { key: "reconnect", available: false })));
  assert.equal(button("Play").disabled, true, "losing the source stops active playback");
  assert.equal(button("Pause"), undefined);
  assert.ok(document.querySelector(".td-video-reconnect"));
  console.log("Rallies UI: combined decisions, web shortcuts, shared Editor state, trims/undo, serve reviews, missing rallies, disconnected playback guards and source reconnection passed.");
} finally { await act(async () => root.unmount()); dom.window.close(); await server.close(); }
