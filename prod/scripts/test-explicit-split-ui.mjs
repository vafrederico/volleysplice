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
  const { createCutDraft, parseCutDraft } = await server.ssrLoadModule("/src/lib/cut-draft.ts");
  const { editHistoryKey } = await server.ssrLoadModule("/src/designs/taste/edit-history.ts");
  const { putProject, listProjects, projectAnalysisId } = await server.ssrLoadModule("/src/lib/project-store.ts");
  const { projectDraftStorageKey } = await server.ssrLoadModule("/src/lib/review-draft-persistence.ts");
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
  function Editor({ savedProject, productAnalysis = analysis }) {
    const projects = React.useMemo(() => [savedProject], [savedProject]);
    const review = useDesignReview({ projects, projectId: savedProject.id, productAnalysis });
    return review.state === "ready" ? React.createElement(RallyDesk, { review }) : React.createElement("p", null, review.message);
  }
  Object.assign(project, await putProject(project, { expectedSource: null }));
  const storageKey = projectDraftStorageKey(seed.analysisId, project.instanceId);
  const current = () => JSON.parse(window.localStorage.getItem(storageKey));
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
    Object.assign(imported.project, await putProject(imported.project, { expectedSource: null }));
  });
  assert.deepEqual(exported.corrections.correctedRanges, splitCuts);
  assert.deepEqual(imported.project.importedFeedback.initialDraft.cuts, splitCuts);
  await act(async () => root.render(React.createElement(Editor, { key: "reimport", savedProject: imported.project })));
  const importedKey = projectDraftStorageKey(projectAnalysisId(imported.project), imported.project.instanceId);
  await waitFor(() => window.localStorage.getItem(importedKey));
  assert.deepEqual(JSON.parse(window.localStorage.getItem(importedKey)).cuts, splitCuts);
  console.log("Production RallyDesk: real split, stable IDs, metadata, IndexedDB/reopen, full confirmed reset, opaque UI, undo/redo, download and reimport passed.");

  const limitCases = [
    { name: "at-limit", assigned: 2048, targetAssigned: true, blocked: /2,048/ },
    { name: "long-legacy-id", assigned: 2, targetAssigned: false, id: "L".repeat(257), blocked: /256/ },
    { name: "one-below-limit", assigned: 2047, targetAssigned: true },
    { name: "unassigned-adds-two", assigned: 2047, targetAssigned: false, blocked: /2,048/ },
    { name: "many-unassigned", assigned: 0, targetAssigned: false, legacy: 2048 },
    { name: "assigned-siblings", assigned: 2046, targetAssigned: false, legacy: 3, longSibling: true },
    { name: "ignored-id-collision", assigned: 0, targetAssigned: false, rightId: "R003",
      ignoredIntervals: [{ id: "R002", start: 1, end: 2, reason: "other" }] },
  ];
  const failures = [];
  for (const scenario of limitCases) {
    try {
      const id = scenario.id ?? "R001";
      const caseProject = structuredClone(project);
      caseProject.id = `synthetic-split-${scenario.name}`;
      const caseRallies = [{ ...rallies[0], id }, rallies[1]];
      caseProject.analysis.intervals = caseRallies;
      const caseSeed = { ...seed, analysisId: projectAnalysisId(caseProject), recordingId: caseProject.id, rallies: caseRallies };
      caseProject.reviewDraft = createCutDraft(caseSeed);
      const cut = caseProject.reviewDraft.cuts[0];
      caseProject.reviewDraft.cuts = [
        { ...cut, ...(scenario.targetAssigned ? { editableGroupId: "split:existing" } : {}) },
        ...Array.from({ length: scenario.assigned - Number(scenario.targetAssigned) }, (_, index) => ({
          ...cut, id: `S${index}`, included: index % 2 === 0, editableGroupId: "split:siblings",
        })),
        ...Array.from({ length: scenario.legacy ?? 0 }, (_, index) => ({ ...cut,
          id: scenario.longSibling && index === 0 ? "L".repeat(257) : `L${index}` })),
      ];
      caseProject.reviewDraft.portableUi = futureUi;
      caseProject.reviewDraft.ignoredIntervals = scenario.ignoredIntervals ?? [];
      assert.ok(parseCutDraft(JSON.stringify(caseProject.reviewDraft), caseSeed), `${scenario.name}: input must be valid`);
      await act(async () => { Object.assign(caseProject, await putProject(caseProject, { expectedSource: null })); });
      await act(async () => root.render(React.createElement(Editor, { key: scenario.name, savedProject: caseProject,
        productAnalysis: { ...analysis, id: caseSeed.analysisId, recordingId: caseProject.id, rallies: caseRallies } })));
      const caseKey = projectDraftStorageKey(caseSeed.analysisId, caseProject.instanceId);
      const historyKey = editHistoryKey(caseProject.id, caseProject.instanceId);
      await waitFor(() => window.localStorage.getItem(caseKey));
      await key("ArrowRight");
      const draftBefore = window.localStorage.getItem(caseKey);
      const historyBefore = window.localStorage.getItem(historyKey);
      let projectBefore;
      await act(async () => { projectBefore = (await listProjects()).find((item) => item.id === caseProject.id); });
      await key("S", { shiftKey: true });
      const expectedLength = caseProject.reviewDraft.cuts.length + Number(!scenario.blocked);
      assert.equal(document.querySelector(".rd-register-list").children.length, expectedLength,
        `${scenario.name}: rejected splits must not mutate the rendered working state`);
      if (scenario.blocked) {
        assert.match(document.querySelector(".rd-sidebar-tool-status").textContent, scenario.blocked,
          `${scenario.name}: show the actual split limit`);
        assert.equal(window.localStorage.getItem(caseKey), draftBefore, `${scenario.name}: cache must be unchanged`);
        assert.equal(window.localStorage.getItem(historyKey), historyBefore, `${scenario.name}: history must be unchanged`);
        let projectAfter;
        await act(async () => { projectAfter = (await listProjects()).find((item) => item.id === caseProject.id); });
        assert.deepEqual(projectAfter.reviewDraft, projectBefore.reviewDraft, `${scenario.name}: IndexedDB must be unchanged`);
      } else {
        if (scenario.rightId) {
          assert.equal(document.querySelector(".rd-register-list").children[1].querySelector(".rd-register-id").textContent,
            scenario.rightId, "the rendered split must allocate outside the ignored-range namespace");
        }
        await waitFor(() => JSON.parse(window.localStorage.getItem(caseKey)).cuts.length === expectedLength);
        const cuts = JSON.parse(window.localStorage.getItem(caseKey)).cuts;
        assert.equal(cuts[0].id, id, `${scenario.name}: existing IDs stay stable`);
        assert.equal(cuts[1].id, scenario.rightId ?? "R002");
        assert.ok(cuts[0].editableGroupId && cuts[1].editableGroupId);
        assert.notEqual(cuts[0].editableGroupId, cuts[1].editableGroupId);
        assert.deepEqual(cuts.slice(2), caseProject.reviewDraft.cuts.slice(1), `${scenario.name}: preserve assigned siblings and unassigned legacy cuts`);
        assert.equal(cuts.filter((item) => item.editableGroupId !== undefined).length, Math.max(2, scenario.assigned + (scenario.targetAssigned ? 1 : 2)));
        if (scenario.name === "ignored-id-collision") {
          assert.deepEqual(cuts.map((item) => item.id), ["R001", "R003"]);
          assert.deepEqual(JSON.parse(window.localStorage.getItem(caseKey)).ignoredIntervals, scenario.ignoredIntervals);
          await key("z", { ctrlKey: true });
          await waitFor(() => JSON.parse(window.localStorage.getItem(caseKey)).cuts.length === 1);
          assert.equal(document.querySelector(".rd-register-list").children.length, 1);
          assert.ok(parseCutDraft(window.localStorage.getItem(caseKey), caseSeed));
          await key("y", { ctrlKey: true });
          await waitFor(() => JSON.parse(window.localStorage.getItem(caseKey)).cuts.length === 2);
          assert.equal(document.querySelector(".rd-register-list").children.length, 2);
          assert.deepEqual(JSON.parse(window.localStorage.getItem(caseKey)).cuts, cuts, "redo keeps the allocated IDs and explicit metadata");
          let reloadedCase;
          await act(async () => { reloadedCase = (await listProjects()).find((item) => item.id === caseProject.id); });
          assert.deepEqual(reloadedCase.reviewDraft.cuts, cuts, "the fresh range IDs must autosave to IndexedDB");
          assert.deepEqual(reloadedCase.reviewDraft.ignoredIntervals, scenario.ignoredIntervals);
          const history = JSON.parse(window.localStorage.getItem(historyKey));
          for (const snapshot of [...history.past, history.present, ...history.future]) {
            assert.ok(parseCutDraft(JSON.stringify(snapshot), caseSeed), "undo/redo snapshots must remain parseable");
          }
        }
      }
      const saved = JSON.parse(window.localStorage.getItem(caseKey));
      assert.ok(parseCutDraft(JSON.stringify(saved), caseSeed), `${scenario.name}: draft save remains valid`);
      const downloadIndex = downloads.length;
      await key("E", { shiftKey: true });
      await clickNode(document.querySelector('input[name="taste-export-kind"][value="feedback"]'));
      await clickNode(button("Save project"));
      assert.equal(downloads.length, downloadIndex + 1);
      let backup;
      await act(async () => { backup = JSON.parse(await downloads[downloadIndex].text()); });
      assert.deepEqual(backup.corrections.correctedRanges, saved.cuts);
      const reimportedDraft = importModelFeedbackProject(JSON.stringify(backup)).project.importedFeedback.initialDraft;
      assert.deepEqual(reimportedDraft.cuts, saved.cuts,
        `${scenario.name}: the actual component backup must reimport`);
      assert.deepEqual(reimportedDraft.ignoredIntervals, saved.ignoredIntervals);
      // Import validation stays strict even though the action now prevents creating these states.
      if (scenario.name === "at-limit") {
        backup.corrections.correctedRanges.push({ ...backup.corrections.correctedRanges[0], id: "extra-part" });
        assert.throws(() => importModelFeedbackProject(JSON.stringify(backup)), /too many explicit rally members/);
      } else if (scenario.name === "long-legacy-id") {
        backup.corrections.correctedRanges[0].editableGroupId = "split:invalid-legacy";
        assert.throws(() => importModelFeedbackProject(JSON.stringify(backup)), /stable cut identity/);
      } else if (scenario.name === "ignored-id-collision") {
        backup.corrections.correctedRanges[1].id = "R002";
        assert.throws(() => importModelFeedbackProject(JSON.stringify(backup)), /not compatible/);
      }
      console.log(`Production RallyDesk split limit: ${scenario.name} passed.`);
    } catch (error) {
      failures.push(`${scenario.name}: ${error.message}`);
      console.error(failures.at(-1));
    }
  }
  assert.deepEqual(failures, [], "production split limit regressions");
} finally {
  await act(async () => root.unmount());
  await server.close(); dom.window.close();
  URL.createObjectURL = realCreateUrl; URL.revokeObjectURL = realRevokeUrl;
}
