import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { IDBFactory } from "fake-indexeddb";
import { JSDOM } from "jsdom";
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { createServer } from "vite";

// Exercise the real editor and its persistence hook without serving the app.
const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  server: { middlewareMode: true, hmr: false },
  appType: "custom",
});
const dom = new JSDOM('<div id="root"></div>', {
  url: "https://example.test",
  pretendToBeVisual: true,
});
Object.assign(globalThis, {
  window: dom.window,
  document: dom.window.document,
  Element: dom.window.Element,
  HTMLElement: dom.window.HTMLElement,
  IS_REACT_ACT_ENVIRONMENT: true,
  indexedDB: new IDBFactory(),
  ResizeObserver: class {
    observe() {}
    disconnect() {}
  },
  requestAnimationFrame: (callback) => setTimeout(callback, 0),
  cancelAnimationFrame: clearTimeout,
});
Object.defineProperty(globalThis, "navigator", {
  value: dom.window.navigator,
  configurable: true,
});
window.matchMedia = () => ({
  matches: false,
  addEventListener() {},
  removeEventListener() {},
});
dom.window.HTMLElement.prototype.scrollIntoView = () => {};
dom.window.HTMLCanvasElement.prototype.getContext = () => null;
dom.window.HTMLMediaElement.prototype.load = () => {};
dom.window.HTMLMediaElement.prototype.pause = () => {};
dom.window.HTMLDialogElement.prototype.showModal = function () {
  this.open = true;
};
const root = createRoot(document.getElementById("root"));
const realSetItem = dom.window.Storage.prototype.setItem;
const database = globalThis.indexedDB;
const button = (text) =>
  [...document.querySelectorAll("button")].find(
    (node) => node.textContent === text || node.querySelector("strong")?.textContent === text,
  );
const click = async (text) => {
  const target = button(text);
  assert.ok(target, text);
  await act(async () => target.click());
};
const waitFor = async (condition) => {
  for (let i = 0; i < 100; i += 1) {
    if (condition()) return;
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 5));
    });
  }
  assert.ok(condition(), "UI did not settle");
};

try {
  const { useDesignReview } = await server.ssrLoadModule(
    "/src/designs/useDesignReview.ts",
  );
  const { RallyDesk } = await server.ssrLoadModule(
    "/src/designs/taste/index.tsx",
  );
  const { putProject, listProjects, projectAnalysisId } =
    await server.ssrLoadModule("/src/lib/project-store.ts");
  const { buildFinalCutIntervals, createCutDraft, cutDraftStorageKey } = await server.ssrLoadModule(
    "/src/lib/cut-draft.ts",
  );
  const project = {
    schemaVersion: 1,
    id: "ui-review-fixture",
    source: {
      name: "match.mp4",
      size: 1000,
      lastModified: 100,
      type: "video/mp4",
    },
    info: {
      duration: 60,
      width: 1920,
      height: 1080,
      mimeType: "video/mp4",
      videoCodec: "avc",
      canDecodeVideo: true,
      hasAudio: true,
      canDecodeAudio: true,
    },
    analysisWindow: { start: 0, end: 60 },
    roi: { x: 0, y: 0, width: 1, height: 1 },
    status: "ready",
    error: null,
    analysis: {
      modelId: "distilled-large-recall-v1",
      featurePath: "local-source",
      intervals: [
        {
          id: "R001",
          start: 10,
          end: 20,
          confidence: 0.5,
          included: true,
          agreement: "neural",
        },
      ],
      times: new Float64Array([0]),
      rallyProbabilities: new Float32Array([0.5]),
      serveProbabilities: new Float32Array([0.5]),
      deadStateProbabilities: new Float32Array([0.5]),
    },
    createdAt: "2026-01-01T00:00:00.000Z",
    updatedAt: "2026-01-01T00:00:00.000Z",
  };
  const seed = {
    analysisId: projectAnalysisId(project),
    recordingId: project.id,
    duration: 60,
    rallies: project.analysis.intervals,
    ignoredIntervals: [],
  };
  const key = cutDraftStorageKey(seed.analysisId);
  // A v15 draft never persisted the final-video choice. Preview off must not
  // change its historical final-video default of on.
  project.reviewDraft = { ...createCutDraft(seed), version: 15, renderScoreOverlay: false };
  delete project.reviewDraft.renderScoreOnFinalVideo;
  const sourceFile = new File([], "match.mp4", { type: "video/mp4" });
  const exports = [];
  const queueVideoExport = (request) => exports.push(request);
  let readyReview;
  function Editor({ savedProject }) {
    const projects = React.useMemo(() => [savedProject], [savedProject]);
    const review = useDesignReview({
      projects,
      projectId: savedProject.id,
      sourceFile,
      videoUrl: "blob:synthetic-review-source",
      onQueueVideoExport: queueVideoExport,
    });
    if (review.state === "ready") readyReview = review;
    return review.state === "ready"
      ? React.createElement(RallyDesk, { review })
      : React.createElement(
          "p",
          { role: "status" },
          review.message ?? "Loading",
        );
  }
  await putProject(project);
  window.localStorage.setItem("volleycut:rally-desk-tour:v1", "dismissed");
  await act(async () =>
    root.render(React.createElement(Editor, { savedProject: project })),
  );
  await waitFor(() => window.localStorage.getItem(key));
  const scoreCheckbox = (text) => {
    const label = [...document.querySelectorAll("label")].find(
      (node) => node.textContent.includes(text),
    );
    assert.ok(label, text);
    return label.querySelector('input[type="checkbox"]');
  };
  await click("Export");
  assert.equal(scoreCheckbox("Render score on final video").checked, true);
  await click("Save MP4");
  assert.ok(exports.at(-1).scoreOverlay, "legacy preview off still exports score");
  await act(async () => scoreCheckbox("Render score on final video").click());
  await waitFor(() => JSON.parse(window.localStorage.getItem(key)).renderScoreOnFinalVideo === false);
  let savedScoreProject;
  await act(async () => { [savedScoreProject] = await listProjects(); });
  assert.equal(savedScoreProject.reviewDraft.renderScoreOnFinalVideo, false);
  assert.equal(savedScoreProject.reviewDraft.renderScoreOverlay, false);

  // Fully unmount and reopen from IndexedDB, with no local draft fallback.
  await act(async () => root.render(null));
  window.localStorage.clear();
  window.localStorage.setItem("volleycut:rally-desk-tour:v1", "dismissed");
  await act(async () => root.render(React.createElement(Editor, { savedProject: savedScoreProject })));
  await waitFor(() => button("Export"));
  assert.equal(scoreCheckbox("Preview score on video").checked, false);
  await act(async () => scoreCheckbox("Preview score on video").click());
  await click("Export");
  assert.equal(scoreCheckbox("Render score on final video").checked, false);
  await click("Save MP4");
  assert.equal(exports.at(-1).scoreOverlay, undefined, "preview on cannot re-enable the final overlay");
  await act(async () => scoreCheckbox("Render score on final video").click());
  await click("Save MP4");
  assert.ok(exports.at(-1).scoreOverlay, "checking the final-video option restores its export overlay");
  // Final-video preferences participate in the editor's normal undo/redo.
  await click("Review");
  await act(async () => window.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "z", ctrlKey: true, bubbles: true })));
  await click("Export");
  assert.equal(scoreCheckbox("Render score on final video").checked, false);
  await click("Review");
  await act(async () => window.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "y", ctrlKey: true, bubbles: true })));
  await click("Export");
  assert.equal(scoreCheckbox("Render score on final video").checked, true);
  await click("Review");
  await click("Remove rally");
  await waitFor(
    () =>
      JSON.parse(window.localStorage.getItem(key)).cuts[0].included === false,
  );
  let reloaded;
  await act(async () => {
    [reloaded] = await listProjects();
  });
  assert.equal(reloaded.reviewDraft.cuts[0].included, false);
  await act(async () =>
    root.render(
      React.createElement(Editor, { key: "reload", savedProject: reloaded }),
    ),
  );
  await waitFor(() =>
    document.querySelector('button[title^="R001 · Removed rally"]'),
  );

  dom.window.Storage.prototype.setItem = function (storageKey, value) {
    if (storageKey.startsWith("volleycut:"))
      throw new DOMException("Full", "QuotaExceededError");
    return realSetItem.call(this, storageKey, value);
  };
  globalThis.indexedDB = {
    open() {
      throw new Error("Unavailable");
    },
  };
  await click("Keep rally");
  await waitFor(() => document.querySelector(".rd-storage-alert"));
  assert.match(
    document.querySelector('[role="alert"]').textContent,
    /not saved.*Export a project backup/,
  );
  assert.equal(
    JSON.parse(window.localStorage.getItem(key)).cuts[0].included,
    false,
  );
  const closing = new dom.window.Event("beforeunload", { cancelable: true });
  window.dispatchEvent(closing);
  assert.equal(closing.defaultPrevented, true);

  dom.window.Storage.prototype.setItem = realSetItem;
  globalThis.indexedDB = database;
  await click("Remove rally");
  await waitFor(() => !document.querySelector(".rd-storage-alert"));
  const savedClosing = new dom.window.Event("beforeunload", {
    cancelable: true,
  });
  window.dispatchEvent(savedClosing);
  assert.equal(savedClosing.defaultPrevented, false);

  // A slow success from a previous edit must not hide the latest failure.
  const pending = [];
  const deferredReview = {
    ...readyReview,
    saveDraft: () => new Promise((resolve) => pending.push(resolve)),
  };
  await act(async () =>
    root.render(
      React.createElement(RallyDesk, {
        key: "deferred",
        review: deferredReview,
      }),
    ),
  );
  await waitFor(() => pending.length === 1);
  await click("Keep rally");
  assert.equal(pending.length, 2);
  await act(async () =>
    pending[1]({ saved: false, message: "Latest edit is not saved" }),
  );
  await act(async () =>
    pending[0]({ saved: true, message: "Earlier edit saved" }),
  );
  assert.equal(
    document.querySelector(".rd-storage-alert").textContent,
    "Latest edit is not saved",
  );

  // Imported highlight-only reviews must retain footage through the real editor's
  // immediate autosave. A deliberate active-level change opts into removal.
  const cleanupDraft = { ...createCutDraft(seed), suppressionInitialBehavior: "highlight-only",
    cuts: createCutDraft(seed).cuts.map(cut => ({ ...cut, agreement: "both-models" })) };
  const suggestion = { id: "cleanup-1", logicalId: "cleanup-1", suppressionEventId: "S1", start: 12, end: 13,
    score: .95, sourceProductionIds: ["R001"], eligiblePolicyIds: ["conservative", "balanced", "aggressive"] };
  const cleanupSuppression = { suggestions: [suggestion] };
  let savedCleanup;
  const cleanupReview = { ...readyReview, productAnalysis: null, draft: cleanupDraft, modelDraft: cleanupDraft,
    suppression: cleanupSuppression, cleanupSuggestions: [{ ...suggestion, cutId: null, decision: "pending" }],
    saveDraft: async draft => { savedCleanup = draft; return { saved: true, message: "Saved" }; } };
  await act(async () => root.render(React.createElement(RallyDesk, { key: "highlight-only", review: cleanupReview })));
  await waitFor(() => savedCleanup);
  assert.equal(savedCleanup.suppressionInitialBehavior, "highlight-only");
  assert.ok(buildFinalCutIntervals(savedCleanup, cleanupSuppression).length, "mount/autosave retains highlighted footage");
  const settings = document.querySelector('button[aria-label="Final cut settings"]');
  assert.ok(settings);
  await act(async () => settings.click());
  const cleanupSelect = [...document.querySelectorAll("label")].find(label => label.textContent.includes("Automatic cleanup"))?.querySelector("select");
  assert.ok(cleanupSelect);
  await act(async () => { cleanupSelect.value = "light"; cleanupSelect.dispatchEvent(new dom.window.Event("change", { bubbles: true })); });
  await waitFor(() => savedCleanup.suppressionInitialBehavior === "disable-initially");
  assert.deepEqual(buildFinalCutIntervals(savedCleanup, cleanupSuppression), []);

  const invalid = {
    ...createCutDraft(seed),
    sourceRevision: "incompatible-revision",
    updatedAt: "2099-01-01T00:00:00.000Z",
  };
  const raw = JSON.stringify(invalid);
  window.localStorage.setItem(key, raw);
  await putProject({ ...project, reviewDraft: invalid });
  await act(async () =>
    root.render(
      React.createElement(Editor, {
        key: "incompatible",
        savedProject: { ...project, reviewDraft: invalid },
      }),
    ),
  );
  await waitFor(() => document.body.textContent.includes("preserved"));
  assert.equal(window.localStorage.getItem(key), raw);
  assert.deepEqual((await listProjects())[0].reviewDraft, invalid);
  assert.equal(
    button("Keep rally"),
    undefined,
    "No fresh editor may autosave over incompatible edits",
  );
  console.log(
    "Review UI: legacy final-score default, final-score save/unmount/IndexedDB reopen, independent preview/export options, final-score undo/redo, rally reload, failed-save warning, close protection, recovery, stale acknowledgements and incompatible-draft preservation passed.",
  );
} finally {
  dom.window.Storage.prototype.setItem = realSetItem;
  globalThis.indexedDB = database;
  await act(async () => root.unmount());
  await server.close();
  dom.window.close();
}
