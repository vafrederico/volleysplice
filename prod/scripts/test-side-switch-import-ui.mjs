import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { IDBFactory } from "fake-indexeddb";
import { JSDOM } from "jsdom";
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { createServer } from "vite";

// Load the actual production editor without opening a port or running inference.
const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  server: { middlewareMode: true, hmr: false }, appType: "custom",
});
const dom = new JSDOM('<div id="root"></div>', {
  url: "https://example.test", pretendToBeVisual: true,
});
Object.assign(globalThis, {
  window: dom.window, document: dom.window.document, Node: dom.window.Node,
  Element: dom.window.Element, HTMLElement: dom.window.HTMLElement,
  IS_REACT_ACT_ENVIRONMENT: true, indexedDB: new IDBFactory(),
  ResizeObserver: class { observe() {} disconnect() {} },
  requestAnimationFrame: callback => setTimeout(callback, 0), cancelAnimationFrame: clearTimeout,
});
Object.defineProperty(globalThis, "navigator", { value: dom.window.navigator, configurable: true });
window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
dom.window.HTMLElement.prototype.scrollIntoView = () => {};
dom.window.HTMLCanvasElement.prototype.getContext = () => null;
dom.window.HTMLDialogElement.prototype.showModal = function () { this.open = true; };
dom.window.HTMLDialogElement.prototype.close = function () { this.open = false; };
const originalCreateURL = URL.createObjectURL;
const originalRevokeURL = URL.revokeObjectURL;
let downloaded;
URL.createObjectURL = blob => { downloaded = blob; return "blob:synthetic-project"; };
URL.revokeObjectURL = () => {};
dom.window.HTMLAnchorElement.prototype.click = () => {};
const root = createRoot(document.getElementById("root"));
const click = async selector => {
  const target = document.querySelector(selector);
  assert.ok(target, selector);
  await act(async () => target.click());
};
const clickText = async text => {
  const target = [...document.querySelectorAll("button")].find(node => node.textContent === text);
  assert.ok(target, text);
  await act(async () => target.click());
};
const waitFor = async condition => {
  for (let i = 0; i < 150; i += 1) {
    let settled;
    await act(async () => {
      settled = await condition();
      if (!settled) await new Promise(resolve => setTimeout(resolve, 5));
    });
    if (settled) return;
  }
  assert.ok(await condition(), "UI/persistence did not settle");
};

try {
  const { useDesignReview } = await server.ssrLoadModule("/src/designs/useDesignReview.ts");
  const { RallyDesk } = await server.ssrLoadModule("/src/designs/taste/index.tsx");
  const { importModelFeedbackProject } = await server.ssrLoadModule("/src/lib/model-feedback-import.ts");
  const { listProjects, putProject, projectAnalysisId } = await server.ssrLoadModule("/src/lib/project-store.ts");
  const { projectDraftStorageKey } = await server.ssrLoadModule("/src/lib/review-draft-persistence.ts");
  const { deriveScoreAt } = await server.ssrLoadModule("/src/lib/score-tracking.ts");
  const fixturePath = fileURLToPath(new URL("../../tests/fixtures/web-side-switch-project.ts", import.meta.url)).replaceAll("\\", "/");
  const { sideSwitchProjectFixture } = await server.ssrLoadModule(`/@fs/${fixturePath}`);
  let readyReview;
  function Editor({ project, originalAnalysis }) {
    const projects = React.useMemo(() => [project], [project]);
    const productAnalysis = React.useMemo(() => ({
      ...originalAnalysis, id: projectAnalysisId(project), recordingId: project.id,
      sideSwitch: project.analysis.sideSwitch, sideSwitchEnabled: project.sideSwitchEnabled,
    }), [project, originalAnalysis]);
    const review = useDesignReview({ projects, projectId: project.id, productAnalysis });
    if (review.state === "ready") readyReview = review;
    return review.state === "ready" ? React.createElement(RallyDesk, { review })
      : React.createElement("p", { role: "status" }, review.message ?? "Loading");
  }
  const occupiedProjectIds = new Set();
  for (const scenario of ["enabled", "disabled", "legacy-absent", "empty"]) {
    const { analysis, bundle } = sideSwitchProjectFixture(scenario !== "disabled");
    if (scenario === "legacy-absent") {
      delete bundle.source.generateSideSwitchMarkers;
      delete bundle.initialInference.sideSwitch;
    }
    if (scenario === "empty") bundle.initialInference.sideSwitch.candidates = [];
    const { project } = importModelFeedbackProject(JSON.stringify(bundle), { occupiedProjectIds });
    occupiedProjectIds.add(project.id);
    await act(async () => { Object.assign(project, await putProject(project, { expectedSource: null })); });
    const key = projectDraftStorageKey(projectAnalysisId(project), project.instanceId);
    const saved = () => JSON.parse(window.localStorage.getItem(key));
    await act(async () => root.render(React.createElement(Editor, { key: scenario, project, originalAnalysis: analysis })));
    await waitFor(() => window.localStorage.getItem(key));
    assert.equal(readyReview.sideSwitchEnabled, scenario !== "disabled");
    assert.deepEqual(saved().scoreTracking, bundle.corrections.scoreTracking.state);
    assert.match(document.querySelector('.rd-event-list [data-kind="switch"]').textContent, /0:35.*model/);
    assert.ok([...document.querySelectorAll('.rd-event-list [data-kind="switch"]')].some(node => /0:40.*manual/.test(node.textContent)));

    const downloadProject = async () => {
      await click('.td-stages button:last-child');
      await click('input[value="feedback"]');
      downloaded = undefined;
      await clickText("Save project");
      assert.ok(downloaded, "real RallyDesk project download");
      let text;
      await act(async () => { text = await downloaded.text(); });
      return JSON.parse(text);
    };
    const before = await downloadProject();
    assert.deepEqual(before.initialInference.sideSwitch, bundle.initialInference.sideSwitch ?? null);
    assert.equal(before.source.generateSideSwitchMarkers, scenario !== "disabled");
    assert.deepEqual(before.corrections.scoreTracking, bundle.corrections.scoreTracking);
    assert.ok(before.finalExportIntervals.length > 0);
    await click('.td-stages button:first-child');

    await click('button[aria-label="App menu"]');
    await click('.rd-menu-panel button:nth-of-type(3)');
    assert.equal(document.querySelector('.rd-reset-project-dialog').open, true);
    await click('.rd-reset-project-dialog footer button:last-child');
    const expected = scenario === "enabled" ? [["switch-boundary-1", 25, "model"], ["switch-internal-2", 45, "model"]] : [];
    const markerSummary = draft => draft.scoreTracking.sideSwitchMarkers.map(m => [m.id, m.timestamp, m.origin]);
    await waitFor(() => JSON.stringify(markerSummary(saved())) === JSON.stringify(expected));
    assert.deepEqual(saved().scoreTracking.removedModelMarkerIds, []);
    assert.deepEqual(saved().scoreTracking, readyReview.modelDraft.scoreTracking);
    // The switch before R003 changes the next point's team attribution.
    if (scenario === "enabled") {
      const score = deriveScoreAt(saved().scoreTracking, 60);
      assert.deepEqual([score.team1Score, score.team2Score], [0, 3]);
      assert.deepEqual([...document.querySelectorAll('.rd-event-list [data-kind="switch"] small')].slice(0, 2).map(node => node.textContent), ["0:25.0 · model", "0:45.0 · model"]);
    }
    await act(async () => {
      window.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "z", code: "KeyZ", ctrlKey: true, bubbles: true }));
    });
    await waitFor(() => saved().scoreTracking.sideSwitchMarkers[0]?.timestamp === 35);
    assert.deepEqual(saved().scoreTracking, bundle.corrections.scoreTracking.state, "reset undo restores corrections and tombstones");
    await act(async () => {
      window.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "y", code: "KeyY", ctrlKey: true, bubbles: true }));
    });
    await waitFor(() => JSON.stringify(markerSummary(saved())) === JSON.stringify(expected));
    const after = await downloadProject();
    assert.deepEqual(after.initialInference.sideSwitch, before.initialInference.sideSwitch);
    assert.equal(after.source.generateSideSwitchMarkers, before.source.generateSideSwitchMarkers);
    assert.deepEqual(after.corrections.scoreTracking.state, saved().scoreTracking);
    const reimported = importModelFeedbackProject(JSON.stringify(after)).project;
    assert.deepEqual(reimported.importedFeedback.initialDraft.scoreTracking, saved().scoreTracking);
    await waitFor(async () => {
      const persisted = (await listProjects()).find(candidate => candidate.id === project.id);
      return JSON.stringify(persisted.reviewDraft?.scoreTracking) === JSON.stringify(saved().scoreTracking);
    });
    let reloaded;
    await act(async () => { reloaded = (await listProjects()).find(candidate => candidate.id === project.id); });
    assert.deepEqual(reloaded.analysis.sideSwitch, project.analysis.sideSwitch);
    assert.equal(reloaded.sideSwitchEnabled, project.sideSwitchEnabled);
    window.localStorage.removeItem(key); // Verify the IndexedDB restore path by itself.
    await act(async () => root.render(React.createElement(Editor, { key: `${scenario}-reopen`, project: reloaded, originalAnalysis: analysis })));
    await waitFor(() => window.localStorage.getItem(key));
    assert.deepEqual(markerSummary(saved()), expected);
  }
  console.log("Side-switch import UI: raw evidence, corrected/manual markers, disabled/legacy/empty output, reset/undo/redo, scoring, project download/reimport and IndexedDB reopen passed.");
} finally {
  URL.createObjectURL = originalCreateURL;
  URL.revokeObjectURL = originalRevokeURL;
  await act(async () => root.unmount());
  await server.close();
  dom.window.close();
}
