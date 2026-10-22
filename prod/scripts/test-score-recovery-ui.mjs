import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { IDBFactory } from "fake-indexeddb";
import { JSDOM } from "jsdom";
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { createServer } from "vite";
import { recoveryFixture, servingFixture, switchFixture } from "./score-recovery-fixtures.mjs";

// Real App -> ReadyProjectEditor -> useDesignReview -> RallyDesk, with only
// source decoding and GPU inference stubbed. No media, browser server or port.
const harness = { requests: [], rallyRuns: 0, evidenceRuns: 0, opened: 0, disposed: 0, behavior: null };
globalThis.__scoreRecoveryHarness = harness;
const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false }, appType: "custom",
  plugins: [{ name: "synthetic-inference", enforce: "pre", transform(_code, id) {
    if (id.endsWith("/src/lib/on-device/media.ts")) return `
      export async function openLocalMedia(file) {
        const h = globalThis.__scoreRecoveryHarness; h.opened++;
        let disposed = false;
        return { info: h.info, input: { dispose() { if (!disposed) { disposed = true; h.disposed++; } } } };
      }`;
    if (id.endsWith("/src/lib/on-device/pipeline.ts")) return `
      export const DEFAULT_FEATURE_REDUCTION_KERNEL = "synthetic";
      export const DEFAULT_VIDEO_DECODE_STRATEGY = "synthetic";
      export const VIDEO_DECODER_HARDWARE_ACCELERATION = "synthetic";
      export async function analyzeOpenedMedia() { const h = globalThis.__scoreRecoveryHarness; h.rallyRuns++; return structuredClone(h.analysis); }`;
    if (id.endsWith("/src/lib/on-device/score-specialists.ts")) return `
      export async function inferScoreSpecialists(media, roi, request, options = {}) {
        const h = globalThis.__scoreRecoveryHarness; h.requests.push(Object.keys(request).filter(key => request[key]));
        return h.behavior(request, options);
      }`;
    if (id.endsWith("/src/lib/on-device/production-inference.ts")) return `
      export async function augmentStoredAnalysisWithSuppression() { return null; }
      export async function runProductionInferenceFromFeatures() {
        const h = globalThis.__scoreRecoveryHarness; h.evidenceRuns++; return structuredClone(h.evidence);
      }`;
  } }],
});
const dom = new JSDOM('<div id="root"></div>', { url: "https://example.test", pretendToBeVisual: true });
Object.assign(globalThis, { window: dom.window, document: dom.window.document,
  Element: dom.window.Element, HTMLElement: dom.window.HTMLElement, IS_REACT_ACT_ENVIRONMENT: true,
  indexedDB: new IDBFactory(), ResizeObserver: class { observe() {} disconnect() {} },
  requestAnimationFrame: (callback) => setTimeout(callback, 0), cancelAnimationFrame: clearTimeout });
Object.defineProperty(globalThis, "navigator", { value: dom.window.navigator, configurable: true });
window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
window.scrollTo = () => {};
window.confirm = () => true;
window.isSecureContext = true;
window.VideoDecoder = window.AudioDecoder = window.VideoFrame = class {};
dom.window.HTMLElement.prototype.scrollIntoView = () => {};
dom.window.HTMLCanvasElement.prototype.getContext = () => null;
dom.window.HTMLMediaElement.prototype.pause = () => {};
dom.window.HTMLDialogElement.prototype.showModal = function () { this.open = true; };
dom.window.HTMLDialogElement.prototype.close = function () { this.open = false; };
const originalCreateUrl = URL.createObjectURL, originalRevokeUrl = URL.revokeObjectURL;
URL.createObjectURL = () => "blob:synthetic";
URL.revokeObjectURL = () => {};
let root;
const text = () => document.body.textContent;
const buttons = () => [...document.querySelectorAll("button")];
const button = (label) => buttons().find((node) => node.textContent === label || node.querySelector(":scope > strong")?.textContent === label);
const waitFor = async (condition, message = "UI did not settle") => {
  for (let i = 0; i < 150; i++) {
    let ready;
    await act(async () => { ready = await condition(); await new Promise((resolve) => setTimeout(resolve, 5)); });
    if (ready) return;
  }
  assert.ok(await condition(), message);
};
const click = async (label) => {
  const target = button(label); assert.ok(target, `Missing button: ${label}`); assert.equal(target.disabled, false, label);
  await act(async () => target.click());
};
const inputFile = async (input, file) => {
  assert.ok(input, "Missing source picker");
  Object.defineProperty(input, "files", { value: [file], configurable: true });
  await act(async () => input.dispatchEvent(new dom.window.Event("change", { bubbles: true })));
};
const sourceFile = (name = "match.mp4") => new File(["video"], name, { type: "video/mp4", lastModified: 100 });
const deferred = () => { let resolve; const promise = new Promise((r) => { resolve = r; }); return { promise, resolve }; };

try {
  const { App } = await server.ssrLoadModule("/src/App.tsx");
  const store = await server.ssrLoadModule("/src/lib/project-store.ts");
  const serving = await server.ssrLoadModule("/src/lib/on-device/serving-side-cache.ts");
  const neural = await server.ssrLoadModule("/src/lib/on-device/neural-serving-model.ts");
  const switches = await server.ssrLoadModule("/src/lib/on-device/side-switch-model.ts");
  const { createCutDraft, cutDraftStorageKey } = await server.ssrLoadModule("/src/lib/cut-draft.ts");
  const { addServeMarker, addSideSwitchMarker } = await server.ssrLoadModule("/src/lib/score-tracking.ts");
  const read = async () => (await store.listProjects())[0];
  const mount = async (project, reset = true, extras = []) => {
    if (root) await act(async () => { root.unmount(); await new Promise((resolve) => setTimeout(resolve, 30)); });
    if (reset) {
      globalThis.indexedDB = new IDBFactory(); window.localStorage.clear();
      await store.putProject(project);
      for (const extra of extras) await store.putProject(extra);
      window.localStorage.setItem(store.SELECTED_PROJECT_STORAGE_KEY, project.id);
    }
    harness.info = project.info; harness.analysis = project.analysis;
    root = createRoot(document.getElementById("root"));
    await act(async () => root.render(React.createElement(App)));
    await waitFor(() => project.status === "ready" ? document.querySelector(".taste-rally-desk") : text().includes("Choose original video"));
  };
  const attach = async (name = "match.mp4") => {
    await inputFile(document.querySelector('.td-reconnect input[type="file"]'), sourceFile(name));
    await waitFor(() => !button("Reconnect source for score markers"));
  };
  const generate = () => click("Generate missing score markers");
  const output = (request) => ({
    ...(request.servingSide ? { servingSide: servingFixture(request.servingSide.analysis, serving, neural) } : {}),
    ...(request.sideSwitch ? { sideSwitch: switchFixture(switches) } : {}),
  });

  // The first analysis also reaches the active ready editor when specialists
  // fail. Rally output is checkpointed and remains reusable by its retry.
  let initial = recoveryFixture(); const initialAnalysis = initial.analysis;
  initial = { ...initial, status: "waiting", analysis: null };
  await mount(initial); harness.analysis = initialAnalysis;
  harness.behavior = async () => { throw new Error("Synthetic initial specialist failure"); };
  await inputFile(document.querySelector('input[type="file"]'), sourceFile());
  await waitFor(() => document.querySelector(".taste-rally-desk") && text().includes("Synthetic initial specialist failure"));
  await waitFor(async () => (await read()).status === "ready" && (await read()).scoreSpecialistErrors?.servingSide);
  assert.equal(harness.rallyRuns, 1);
  harness.behavior = async (request, options) => { const result = output(request); options.onOutput(result); return result; };
  await generate(); await waitFor(() => !document.querySelector('[aria-label="Score prediction recovery"]'));
  assert.equal(harness.rallyRuns, 1, "Initial score failure retry must reuse rally output");
  harness.rallyRuns = 0;

  // Reopen a failed ready project. Source checks, persisted retry failure and
  // subsequent success all exercise App callbacks, not a standalone editor.
  let project = recoveryFixture();
  project.scoreSpecialistErrors = { servingSide: "Synthetic model unavailable" };
  await mount(project);
  assert.match(text(), /Synthetic model unavailable/);
  assert.equal(button("Reconnect source for score markers").disabled, true);
  await click("Export");
  assert.match(text(), /Serve predictions are incomplete/);
  assert.doesNotMatch(text(), /All serve sides checked/);
  await click("Review");
  await inputFile(document.querySelector('.td-reconnect input[type="file"]'), new File(["wrong"], "other.mp4", { lastModified: 100 }));
  assert.match(text(), /does not match/);
  assert.equal(button("Reconnect source for score markers").disabled, true);
  await attach();
  harness.behavior = async () => { throw new Error("Synthetic retry failed"); };
  await generate();
  await waitFor(() => text().includes("Synthetic retry failed") && button("Generate missing score markers"));
  await waitFor(async () => (await read()).scoreSpecialistErrors?.servingSide.includes("Synthetic retry failed"));
  await mount(await read(), false);
  assert.match(text(), /Synthetic retry failed/);
  await attach();
  const gate = deferred();
  harness.behavior = async (request, options) => {
    request.servingSide.onProgress({ stage: "frames", completed: 1, total: 2, detail: "Synthetic serve check halfway" });
    await gate.promise; options.signal.throwIfAborted();
    const result = output(request); options.onOutput(result); return result;
  };
  await generate();
  await waitFor(() => text().includes("Synthetic serve check halfway"));
  assert.ok(document.querySelector('[aria-label="Score prediction recovery"][aria-busy="true"]'));
  assert.equal(button("Generate missing score markers"), undefined, "Duplicate retries are unavailable while running");
  await click("Remove rally");
  await act(async () => gate.resolve());
  await waitFor(() => !document.querySelector('[aria-label="Score prediction recovery"]'));
  await waitFor(async () => (await read()).analysis.servingSide && (await read()).reviewDraft.cuts[0].included === false);
  assert.equal((await read()).scoreSpecialistErrors.servingSide, undefined);
  assert.equal(harness.rallyRuns, 0, "Ready recovery and reconnect never rerun rally inference");

  // A stale paired cache becomes missing on reopen. New evidence fills missing
  // markers without erasing manual, reviewed, removed, split or moved markers.
  project = recoveryFixture({ switches: true });
  project.analysis.intervals.push({ ...project.analysis.intervals[0], id: "R002", start: 35, end: 40 });
  project.analysis.servingSide = servingFixture(project.analysis, serving, neural);
  project.analysis.servingSide.modelFingerprint = "stale-pair";
  project.analysis.sideSwitch = switchFixture(switches, [{ id: "old", timestamp: 30, probability: .9, kind: "adjacent-rally-boundary", sourceRangeIds: ["R001"] }]);
  const seed = { analysisId: store.projectAnalysisId(project), recordingId: project.id, duration: 60, rallies: project.analysis.intervals, ignoredIntervals: [] };
  project.reviewDraft = createCutDraft(seed);
  project.reviewDraft.cuts[0].coreEnd = 19;
  project.reviewDraft.reviewedCutIds = ["R001"];
  project.reviewDraft.beforePaddingSeconds = 1;
  project.reviewDraft.cuts.push({ id: "manual-rally", coreStart: 26, coreEnd: 28, keepStart: 26, keepEnd: 28, confidence: 1, included: true, origin: "manual" });
  project.reviewDraft.userTouchedCutIds = ["R001"];
  project.reviewDraft.ignoredIntervals = [{ id: "synthetic-ignore", start: 50, end: 52, reason: "camera-break" }];
  project.reviewDraft.scoreTracking = addServeMarker(project.reviewDraft.scoreTracking, 10, "far", { id: "serve-R001", rallyId: "R001", origin: "model", modelSide: "near" });
  project.reviewDraft.scoreTracking = addServeMarker(project.reviewDraft.scoreTracking, 25, "near", { id: "manual-serve", origin: "manual" });
  project.reviewDraft.scoreTracking = addSideSwitchMarker(project.reviewDraft.scoreTracking, 33, { id: "switch-old", origin: "model", modelEventId: "old" });
  project.reviewDraft.scoreTracking = addSideSwitchMarker(project.reviewDraft.scoreTracking, 45, { id: "manual-switch", origin: "manual" });
  project.reviewDraft.scoreTracking.removedModelMarkerIds = ["serve-R002"];
  const edited = structuredClone(project.reviewDraft);
  await mount(project);
  assert.equal((await read()).analysis.servingSide, undefined);
  await attach();
  harness.behavior = async (request, options) => { const result = output(request); options.onOutput(result); return result; };
  await generate();
  await waitFor(() => !document.querySelector('[aria-label="Score prediction recovery"]'));
  assert.deepEqual(harness.requests.at(-1), ["servingSide"]);
  await waitFor(async () => (await read()).analysis.servingSide);
  const saved = (await read()).reviewDraft;
  assert.deepEqual(saved.cuts, edited.cuts);
  assert.deepEqual(saved.reviewedCutIds, edited.reviewedCutIds);
  assert.deepEqual(saved.ignoredIntervals, edited.ignoredIntervals);
  assert.deepEqual(saved.userTouchedCutIds, edited.userTouchedCutIds);
  assert.equal(saved.beforePaddingSeconds, 1);
  assert.deepEqual(saved.scoreTracking.serveMarkers, edited.scoreTracking.serveMarkers);
  assert.deepEqual(saved.scoreTracking.sideSwitchMarkers, edited.scoreTracking.sideSwitchMarkers);
  assert.deepEqual(saved.scoreTracking.removedModelMarkerIds, edited.scoreTracking.removedModelMarkerIds);

  // A serve stage can complete before a switch stage fails. Reopen and recover
  // only the failed stage; a real zero-candidate switch output is successful.
  project = recoveryFixture({ switches: true });
  await mount(project); await attach();
  harness.behavior = async (request, options) => { options.onOutput({ servingSide: output(request).servingSide }); throw new Error("Synthetic switch failure"); };
  await generate();
  await waitFor(() => text().includes("Synthetic switch failure") && button("Generate missing score markers"));
  await waitFor(async () => (await read()).analysis.servingSide && (await read()).scoreSpecialistErrors.sideSwitch);
  assert.equal((await read()).scoreSpecialistErrors.servingSide, undefined);
  await mount(await read(), false); await attach();
  harness.behavior = async (request, options) => { const result = output(request); options.onOutput(result); return result; };
  await generate();
  await waitFor(() => !document.querySelector('[aria-label="Score prediction recovery"]'));
  assert.deepEqual(harness.requests.at(-1), ["sideSwitch"]);

  // Cancellation rejects late results, keeps successful evidence, and permits a
  // later retry. Changing projects cannot send results into the selected one.
  project = recoveryFixture();
  await mount(project); await attach();
  const cancelled = deferred();
  harness.behavior = async (request, options) => { await cancelled.promise; const result = output(request); options.onOutput(result); return result; };
  await generate(); await waitFor(() => button("Cancel score markers"));
  await click("Cancel score markers");
  await act(async () => cancelled.resolve());
  await waitFor(() => text().includes("Score markers cancelled") && button("Generate missing score markers"));
  assert.equal((await read()).analysis.servingSide, undefined);
  harness.behavior = async (request, options) => { const result = output(request); options.onOutput(result); return result; };
  await generate(); await waitFor(() => !document.querySelector('[aria-label="Score prediction recovery"]'));

  // Two ready-project recoveries share the App queue. A queued cancellation
  // never starts a decode, and changing the selected project cannot redirect
  // an in-flight result into that project's analysis or review.
  project = recoveryFixture();
  const second = recoveryFixture(); second.id = "synthetic-second-project"; second.source.name = "second.mp4";
  await mount(project, true, [second]); await attach();
  const queueGate = deferred(); const requestCount = harness.requests.length;
  let entered = 0;
  harness.behavior = async (request, options) => {
    if (entered++ === 0) await queueGate.promise;
    const result = output(request); options.onOutput(result); return result;
  };
  await generate(); await waitFor(() => entered === 1);
  const selectProject = async (id) => {
    const select = document.querySelector('select[aria-label="Current review project"]');
    await act(async () => { select.value = id; select.dispatchEvent(new dom.window.Event("change", { bubbles: true })); });
    await waitFor(() => document.querySelector('select[aria-label="Current review project"]')?.value === id);
  };
  await selectProject(second.id); await attach("second.mp4"); await generate();
  await waitFor(() => text().includes("Score markers queued"));
  assert.equal(harness.requests.length, requestCount + 1);
  await click("Cancel score markers");
  await waitFor(() => button("Generate missing score markers"));
  assert.equal(harness.requests.length, requestCount + 1);
  await generate();
  await act(async () => queueGate.resolve());
  await waitFor(async () => (await store.listProjects()).every((p) => p.analysis.servingSide));
  assert.equal(harness.requests.length, requestCount + 2);
  assert.equal(harness.rallyRuns, 0);

  // Both neural pairings and the Legacy specialist use the same recovery API.
  const { PRODUCTION_ENSEMBLE_MODEL_ID } = await server.ssrLoadModule("/src/lib/on-device/ensemble.ts");
  for (const modelId of [PRODUCTION_ENSEMBLE_MODEL_ID, "distilled-large-f1-v1"]) {
    project = recoveryFixture({ modelId });
    await mount(project); await attach();
    harness.behavior = async (request, options) => { const result = output(request); options.onOutput(result); return result; };
    await generate(); await waitFor(() => !document.querySelector('[aria-label="Score prediction recovery"]'));
    await waitFor(async () => (await read()).analysis.servingSide?.modelId === servingFixture(project.analysis, serving, neural).modelId);
  }
  // Missing source evidence cannot masquerade as an empty successful check.
  project = recoveryFixture(); delete project.analysis.productionServeOutputs;
  await mount(project); await attach();
  assert.match(text(), /no score evidence/);
  assert.equal(button("Generate missing score markers"), undefined);
  await click("Export");
  assert.match(text(), /Serve predictions are incomplete/);
  assert.doesNotMatch(text(), /All serve sides checked/);

  // Cached features can rebuild missing serve/state evidence without replacing
  // the existing rally model, its intervals or its editable review.
  project = recoveryFixture({ switches: true });
  harness.evidence = structuredClone(project.analysis);
  harness.evidence.intervals = [{ ...project.analysis.intervals[0], id: "unused-rally", start: 1, end: 2 }];
  delete project.analysis.productionServeOutputs; delete project.analysis.productionStateOutputs; delete project.analysis.productionComponents;
  project.analysis.featureNames = ["synthetic-feature"]; project.analysis.featureValues = new Float32Array([0]);
  const originalRallies = structuredClone(project.analysis.intervals);
  await mount(project); await attach(); await generate();
  await waitFor(() => !document.querySelector('[aria-label="Score prediction recovery"]'));
  await waitFor(async () => (await read()).analysis.servingSide);
  assert.equal(harness.evidenceRuns, 1);
  assert.deepEqual((await read()).analysis.intervals, originalRallies);

  // Genuine empty output survives IndexedDB round trips; absent/null output
  // never claims completion. Disabled switches never enter a request.
  project = recoveryFixture({ empty: true });
  project.analysis.servingSide = null;
  await mount(project); await attach(); await generate();
  await waitFor(() => !document.querySelector('[aria-label="Score prediction recovery"]'));
  assert.deepEqual(harness.requests.at(-1), ["servingSide"]);
  await waitFor(async () => (await read()).analysis.servingSide?.candidates.length === 0);
  await mount(await read(), false);
  assert.equal(document.querySelector('[aria-label="Score prediction recovery"]'), null);
  await click("Export");
  assert.match(text(), /Serve check complete; no serve markers found/);
  // A deleted project cannot be recreated by late callbacks or queued writes.
  project = recoveryFixture(); await mount(project); await attach();
  const deleted = deferred(); let started = false;
  harness.behavior = async (request, options) => { started = true; await deleted.promise; const result = output(request); options.onOutput(result); return result; };
  await generate(); await waitFor(() => started);
  await act(async () => document.querySelector(".rd-delete-project").click());
  await act(async () => deleted.resolve());
  await waitFor(async () => (await store.listProjects()).length === 0);
  await act(async () => { await new Promise((resolve) => setTimeout(resolve, 20)); });
  assert.equal((await store.listProjects()).length, 0);
  assert.equal(harness.rallyRuns, 0);
  assert.equal(harness.opened, harness.disposed, "Recovery always disposes media");
  console.log("Score recovery UI: real App ready path, persisted failure/reopen, source rejection/reconnect, retry failure/success, edits during inference, paired cache invalidation, partial completion, missing stages only, manual-marker preservation, cancellation, shared queue serialization/queued cancellation, selection and deletion races, genuine empty output and disabled switches passed.");
} finally {
  if (root) await act(async () => root.unmount());
  await server.close(); dom.window.close();
  URL.createObjectURL = originalCreateUrl; URL.revokeObjectURL = originalRevokeUrl;
  delete globalThis.__scoreRecoveryHarness;
}
