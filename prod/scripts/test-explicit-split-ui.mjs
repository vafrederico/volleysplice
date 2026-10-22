import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { IDBFactory } from "fake-indexeddb";
import { JSDOM } from "jsdom";
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { createServer } from "vite";

// Render the production component, use its real actions, and persist through its real hook.
const server = await createServer({ root: fileURLToPath(new URL("..", import.meta.url)),
  server: { middlewareMode: true, hmr: false }, appType: "custom" });
const dom = new JSDOM('<div id="root"></div>', { url: "https://example.test", pretendToBeVisual: true });
Object.assign(globalThis, {
  window: dom.window, document: dom.window.document, Element: dom.window.Element,
  HTMLElement: dom.window.HTMLElement, Node: dom.window.Node, indexedDB: new IDBFactory(),
  IS_REACT_ACT_ENVIRONMENT: true,
  ResizeObserver: class { observe() {} disconnect() {} },
  requestAnimationFrame: (callback) => setTimeout(callback, 0), cancelAnimationFrame: clearTimeout,
});
Object.defineProperty(globalThis, "navigator", { value: dom.window.navigator, configurable: true });
window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
dom.window.HTMLElement.prototype.scrollIntoView = () => {};
dom.window.HTMLCanvasElement.prototype.getContext = () => null;
dom.window.HTMLDialogElement.prototype.showModal = function () { this.open = true; };
dom.window.HTMLDialogElement.prototype.close = function () { this.open = false; this.dispatchEvent(new window.Event("close")); };
dom.window.HTMLAnchorElement.prototype.click = () => {};
const downloads = [];
const realCreateUrl = URL.createObjectURL;
const realRevokeUrl = URL.revokeObjectURL;
URL.createObjectURL = (blob) => { downloads.push(blob); return "blob:synthetic-project"; };
URL.revokeObjectURL = () => {};
const root = createRoot(document.getElementById("root"));
const waitFor = async (condition, message = "component persistence did not settle") => {
  for (let attempt = 0; attempt < 150; attempt++) {
    if (condition()) return;
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 5)); });
  }
  assert.ok(condition(), message);
};
const clickNode = async (node) => { assert.ok(node, "missing component control"); await act(async () => node.click()); };
const button = (text, scope = document) => [...scope.querySelectorAll("button")].find((node) => node.textContent === text);
const key = async (key, options = {}) => act(async () => {
  document.body.dispatchEvent(new window.KeyboardEvent("keydown", { key, bubbles: true, ...options }));
});

try {
  const { RallyDesk } = await server.ssrLoadModule("/src/designs/taste/index.tsx");
  const { useDesignReview } = await server.ssrLoadModule("/src/designs/useDesignReview.ts");
  const { createCutDraft, cutDraftStorageKey } = await server.ssrLoadModule("/src/lib/cut-draft.ts");
  const { putProject, listProjects, projectAnalysisId } = await server.ssrLoadModule("/src/lib/project-store.ts");
  const { importModelFeedbackProject } = await server.ssrLoadModule("/src/lib/model-feedback-import.ts");
  const { PRODUCTION_ENSEMBLE_MODEL_ID } = await server.ssrLoadModule("/src/lib/on-device/ensemble.ts");
  const source = { name: "synthetic-component.mp4", size: 1024, lastModified: 100, type: "video/mp4" };
  const info = { duration: 40, width: 320, height: 240, rotation: 0, mimeType: "video/mp4",
    videoCodec: "avc", videoCodecString: "avc1.640028", canDecodeVideo: true, hasAudio: false,
    audioCodec: null, sampleRate: null, channels: null, canDecodeAudio: false };
  const rallies = [
    { id: "R001", start: 10, end: 20, confidence: .95, included: true, agreement: "both-models" },
    { id: "R009", start: 30, end: 35, confidence: .95, included: true, agreement: "both-models" },
  ];
  const project = { schemaVersion: 1, id: "synthetic-component-split", source, info,
    analysisWindow: { start: 0, end: 40 }, roi: { x: 0, y: 0, width: 1, height: 1 },
    status: "ready", error: null,
    analysis: { modelId: PRODUCTION_ENSEMBLE_MODEL_ID, featurePath: "local-source", intervals: rallies,
      times: new Float64Array([0]), rallyProbabilities: new Float32Array([.5]),
      serveProbabilities: new Float32Array([.5]), deadStateProbabilities: new Float32Array([.5]) },
    createdAt: "2026-01-01T00:00:00.000Z", updatedAt: "2026-01-01T00:00:00.000Z" };
  const seed = { analysisId: projectAnalysisId(project), recordingId: project.id, duration: 40,
    rallies, ignoredIntervals: [], scoreTrackingEnabled: false };
  const futureUi = { schemaVersion: 2, preferences: { chapterOptions: { future: [null, { retained: true }] } } };
  project.reviewDraft = createCutDraft(seed);
  project.reviewDraft.cuts[0] = { ...project.reviewDraft.cuts[0], coreStart: 12, coreEnd: 18,
    keepStart: 10, keepEnd: 20, editableGroupId: "split:imported-part" };
  project.reviewDraft.cuts.push({ id: "M001", coreStart: 23, coreEnd: 25, keepStart: 23, keepEnd: 25,
    confidence: 1, included: true, origin: "manual", editableGroupId: "split:imported-manual" });
  project.reviewDraft.portableUi = futureUi;
  const analysis = { id: seed.analysisId, recordingId: project.id, kind: "model", modelId: PRODUCTION_ENSEMBLE_MODEL_ID,
    duration: 40, analysisWindow: project.analysisWindow, width: 320, height: 240,
    sourceFilename: source.name, source, mediaInfo: info, roi: project.roi, runtimeVariant: "synthetic-component",
    videoUrl: null, rallies, ignoredIntervals: [], features: null, inferenceTimes: new Float64Array([0]),
    probabilities: { rally: new Float32Array([.5]), serve: new Float32Array([.5]), deadState: new Float32Array([.5]) } };
  function Editor({ savedProject }) {
    const projects = React.useMemo(() => [savedProject], [savedProject]);
    const review = useDesignReview({ projects, projectId: savedProject.id, productAnalysis: analysis });
    return review.state === "ready" ? React.createElement(RallyDesk, { review }) : React.createElement("p", null, review.message);
  }
  const storageKey = cutDraftStorageKey(seed.analysisId);
  const current = () => JSON.parse(window.localStorage.getItem(storageKey));
  await putProject(project);
  await act(async () => root.render(React.createElement(Editor, { savedProject: project })));
  await waitFor(() => window.localStorage.getItem(storageKey));
  await key("ArrowRight"); // The real transport action seeks from trimmed start 12 to 17.
  await key("S", { shiftKey: true }); // Calls the real production splitSelectedAtPlayhead.
  await waitFor(() => current().cuts.length === 4, "production split action did not persist");
  const splitCuts = current().cuts;
  assert.equal(splitCuts[0].id, "R001", "splitting must keep the existing cut ID");
  assert.equal(splitCuts[1].id, "R002");
  assert.deepEqual(splitCuts.slice(0, 2).map((cut) => [cut.coreStart, cut.coreEnd]), [[12, 17], [17, 18]]);
  assert.ok(splitCuts[0].editableGroupId && splitCuts[1].editableGroupId);
  assert.notEqual(splitCuts[0].editableGroupId, splitCuts[1].editableGroupId);
  assert.deepEqual(splitCuts.slice(2), project.reviewDraft.cuts.slice(1));
  let reloaded;
  await act(async () => { reloaded = (await listProjects()).find((item) => item.id === project.id); });
  assert.deepEqual(reloaded.reviewDraft.cuts, splitCuts);
  await act(async () => root.render(React.createElement(Editor, { key: "reopen", savedProject: reloaded })));
  await waitFor(() => document.querySelector('button[title^="R002"]'));
  assert.deepEqual(current().cuts, splitCuts);

  await clickNode(document.querySelector('button[aria-label="App menu"]'));
  await clickNode(button("Reset project changes", document.querySelector(".rd-menu-panel")));
  await clickNode(button("Reset project changes", document.querySelector(".rd-reset-project-dialog")));
  await waitFor(() => current().cuts.length === 2);
  assert.deepEqual(current().cuts.map((cut) => [cut.id, cut.coreStart, cut.coreEnd, cut.included]),
    [["R001", 10, 20, true], ["R009", 30, 35, true]]);
  assert.ok(current().cuts.every((cut) => cut.editableGroupId === undefined && cut.origin === "cached-label"));
  assert.deepEqual(current().portableUi, futureUi, "full reset must keep future UI and chapter options opaque");
  await key("z", { ctrlKey: true });
  await waitFor(() => current().cuts.length === 4);
  assert.deepEqual(current().cuts, splitCuts, "undo must restore explicit split identity");
  await key("y", { ctrlKey: true });
  await waitFor(() => current().cuts.length === 2);
  await key("z", { ctrlKey: true });
  await waitFor(() => current().cuts.length === 4);

  await key("E", { shiftKey: true });
  await clickNode(document.querySelector('input[name="taste-export-kind"][value="feedback"]'));
  await clickNode(button("Save project"));
  assert.equal(downloads.length, 1, "real component project download must run");
  let exported;
  let imported;
  await act(async () => {
    exported = JSON.parse(await downloads[0].text());
    imported = importModelFeedbackProject(JSON.stringify(exported));
    await putProject(imported.project);
  });
  assert.deepEqual(exported.corrections.correctedRanges, splitCuts);
  assert.deepEqual(imported.project.importedFeedback.initialDraft.cuts, splitCuts);
  await act(async () => root.render(React.createElement(Editor, { key: "reimport", savedProject: imported.project })));
  const importedKey = cutDraftStorageKey(projectAnalysisId(imported.project));
  await waitFor(() => window.localStorage.getItem(importedKey));
  assert.deepEqual(JSON.parse(window.localStorage.getItem(importedKey)).cuts, splitCuts);
  console.log("Production RallyDesk: real split, stable IDs, metadata, IndexedDB/reopen, full confirmed reset, opaque UI, undo/redo, download and reimport passed.");
} finally {
  await act(async () => root.unmount());
  await server.close(); dom.window.close();
  URL.createObjectURL = realCreateUrl; URL.revokeObjectURL = realRevokeUrl;
}
