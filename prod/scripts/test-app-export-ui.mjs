import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { IDBFactory } from "fake-indexeddb";
import { JSDOM } from "jsdom";
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { createServer } from "vite";

// Real App, RallyDesk, project persistence and export queue. Only media opening,
// worker readiness and encoding are controlled boundaries; no test prepares the
// download itself. A held readiness promise catches missing awaits in App.
const probe = { preparations: [], exports: [], nextPreparation: null, nextExport: null };
globalThis.__appExportProbe = probe;
const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  base: "/editor/",
  server: { middlewareMode: true, hmr: false, ws: false }, appType: "custom",
  plugins: [{ name: "export-boundaries", enforce: "pre", transform(_code, id) {
    if (id.endsWith("/src/lib/on-device/media.ts")) return `
      export async function openLocalMedia() {
        return { info: globalThis.__appExportProbe.info, input: { dispose() {} } };
      }`;
    if (id.endsWith("/src/lib/on-device/pipeline.ts")) return `
      export const DEFAULT_FEATURE_REDUCTION_KERNEL = "synthetic";
      export const DEFAULT_VIDEO_DECODE_STRATEGY = "synthetic";
      export const VIDEO_DECODER_HARDWARE_ACCELERATION = "synthetic";
      export async function analyzeOpenedMedia() { throw new Error("Unexpected rally inference"); }`;
    if (id.endsWith("/src/lib/on-device/stream-download.ts")) return `
      export async function prepareServiceWorkerStreamDownload(url) {
        const p = globalThis.__appExportProbe; p.preparations.push(url);
        return p.nextPreparation ? p.nextPreparation() : { ready: true, reason: null };
      }`;
    if (id.endsWith("/src/lib/on-device/export.ts")) return `
      export async function exportRawQualityReel(...args) {
        const p = globalThis.__appExportProbe; p.exports.push(args);
        await p.nextExport?.();
      }`;
  } }],
});
const dom = new JSDOM('<div id="root"></div>', { url: "https://example.test/editor/", pretendToBeVisual: true });
Object.assign(globalThis, {
  window: dom.window, document: dom.window.document, Element: dom.window.Element,
  HTMLElement: dom.window.HTMLElement, IS_REACT_ACT_ENVIRONMENT: true,
  indexedDB: new IDBFactory(), ResizeObserver: class { observe() {} disconnect() {} },
  requestAnimationFrame: callback => setTimeout(callback, 0), cancelAnimationFrame: clearTimeout,
});
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
const originalUrls = { create: URL.createObjectURL, revoke: URL.revokeObjectURL };
URL.createObjectURL = () => "blob:synthetic";
URL.revokeObjectURL = () => {};
const root = createRoot(document.getElementById("root"));
const waitFor = async (condition, message = "App did not settle") => {
  for (let i = 0; i < 200; i++) {
    let ready;
    await act(async () => { ready = await condition(); await new Promise(resolve => setTimeout(resolve, 5)); });
    if (ready) return;
  }
  assert.fail(`${message}: ${document.body.textContent.slice(0, 2000)}`);
};
const click = async (text) => {
  const target = [...document.querySelectorAll("button")].find(node => node.textContent === text || node.querySelector("strong")?.textContent === text);
  assert.ok(target, text); assert.equal(target.disabled, false, text);
  await act(async () => target.click());
};
const select = async (id) => {
  const target = document.querySelector('select[aria-label="Current review project"], select[aria-label="Selected project"]');
  assert.ok(target);
  await act(async () => { target.value = id; target.dispatchEvent(new window.Event("change", { bubbles: true })); });
  await waitFor(() => document.querySelector('select[aria-label="Current review project"]')?.value === id);
};
const attach = async (project, sourceFile = new File(["video"], project.source.name, { type: "video/mp4", lastModified: 100 })) => {
  const input = document.querySelector('.td-reconnect input[type="file"]'); assert.ok(input);
  Object.defineProperty(input, "files", { value: [sourceFile], configurable: true });
  await act(async () => input.dispatchEvent(new window.Event("change", { bubbles: true })));
  // Missing score predictions can retain a panel with the same visual style.
  await waitFor(() => document.querySelector(".taste-rally-desk") && !document.querySelector('.td-reconnect input[type="file"]') && document.querySelector(".td-stages button"));
};
const deferred = () => {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
};

try {
  const { App } = await server.ssrLoadModule("/src/App.tsx");
  const store = await server.ssrLoadModule("/src/lib/project-store.ts");
  const { createCutDraft } = await server.ssrLoadModule("/src/lib/cut-draft.ts");
  const fixture = (id) => ({
    schemaVersion: 1, id, source: { name: `${id}.mp4`, size: 5, lastModified: 100, type: "video/mp4" },
    info: { duration: 2, width: 320, height: 240, mimeType: "video/mp4", videoCodec: "avc", canDecodeVideo: true, hasAudio: true, canDecodeAudio: true },
    analysisWindow: { start: 0, end: 2 }, roi: { x: 0, y: 0, width: 1, height: 1 },
    rallyModel: "high-recall", sideSwitchEnabled: false, status: "ready", error: null,
    analysis: { modelId: "distilled-large-recall-v1", featurePath: "local-source",
      intervals: [{ id: "R001", start: .25, end: 1.75, confidence: .9, included: true, agreement: "neural" }],
      times: new Float64Array([0]), rallyProbabilities: new Float32Array([.9]),
      featureNames: ["synthetic"], featureValues: new Float32Array([0]),
      serveProbabilities: new Float32Array([.5]), deadStateProbabilities: new Float32Array([.1]) },
    createdAt: "2026-01-01T00:00:00.000Z", updatedAt: "2026-01-01T00:00:00.000Z",
  });
  const [first, second, third] = ["synthetic-export-first", "synthetic-export-second", "synthetic-export-third"].map(fixture);
  for (const project of [first, second, third]) Object.assign(project, await store.putProject(project, { expectedSource: null }));
  probe.info = first.info;
  window.localStorage.setItem(store.SELECTED_PROJECT_STORAGE_KEY, first.id);
  window.localStorage.setItem("volleycut:rally-desk-tour:v1", "dismissed");
  await act(async () => root.render(React.createElement(App)));
  await waitFor(() => document.querySelector(".taste-rally-desk"));
  await attach(first); await click("Export");
  assert.equal(window.showSaveFilePicker, undefined, "Exercise the API-absent fallback through App");

  const activation = deferred(); probe.nextPreparation = () => activation.promise;
  const encoding = deferred(); probe.nextExport = () => encoding.promise;
  await click("Save MP4"); await waitFor(() => probe.preparations.length === 1);
  assert.equal(probe.preparations[0], "https://example.test/editor/volleycut-export-sw.js", "Resolve the worker under the app's deployment base");
  assert.equal(probe.exports.length, 0, "Encoding must wait for worker activation");
  await select(second.id); await attach(second); await click("Export"); await click("Save MP4");
  assert.equal(probe.preparations.length, 1, "The second project stays queued during preparation");
  probe.nextPreparation = () => ({ ready: true, reason: null });
  await act(async () => activation.resolve({ ready: true, reason: null }));
  await waitFor(() => probe.exports.length === 1);
  assert.equal(probe.exports[0][0].name, first.source.name);
  assert.equal(probe.exports[0][5], "stream-download");
  assert.equal(probe.exports[0][6].target, undefined);
  probe.nextExport = null;
  await act(async () => encoding.resolve());
  await waitFor(() => probe.exports.length === 2);
  await waitFor(async () => (await store.listProjects()).filter(p => p.lastExportedAt).length === 2);
  assert.equal(probe.exports[1][0].name, second.source.name, "Queued exports retain their source project");

  probe.nextPreparation = () => ({ ready: false, reason: "Synthetic worker registration failure" });
  await waitFor(() => [...document.querySelectorAll("button")].some(b => b.textContent === "Save MP4" && !b.disabled));
  await click("Save MP4");
  await waitFor(() => document.body.textContent.includes("Could not create the video: Synthetic worker registration failure"));
  assert.equal(probe.exports.length, 2, "Failed readiness never invokes encoding");
  probe.nextPreparation = null;
  await click("Save MP4"); await waitFor(() => probe.exports.length === 3);
  await waitFor(() => document.body.textContent.includes("Your final MP4 was saved."));

  probe.nextExport = () => { throw new DOMException("Synthetic cancel", "AbortError"); };
  await click("Save MP4"); await waitFor(() => document.body.textContent.includes("Video export was canceled."));
  probe.nextExport = null;
  const preparationsBeforePicker = probe.preparations.length;
  const target = { createWritable: async () => new WritableStream() };
  window.showSaveFilePicker = async () => target;
  await click("Save MP4"); await waitFor(() => probe.exports.length === 5);
  assert.equal(probe.preparations.length, preparationsBeforePicker, "Native picker exports skip streaming preparation");
  assert.equal(probe.exports.at(-1)[5], "compatible");
  assert.equal(probe.exports.at(-1)[6].target, target);
  await waitFor(() => document.body.textContent.includes("Your final MP4 was saved."));

  window.showSaveFilePicker = undefined;
  await select(third.id); await attach(third); await click("Export");
  const deletedActivation = deferred(); probe.nextPreparation = () => deletedActivation.promise;
  await click("Save MP4"); await waitFor(() => probe.preparations.length === preparationsBeforePicker + 1);
  await click("Delete");
  await waitFor(async () => !(await store.listProjects()).some(p => p.id === third.id));
  await act(async () => deletedActivation.resolve({ ready: true, reason: null }));
  await select(second.id); await click("Export"); probe.nextPreparation = null;
  await click("Save MP4"); await waitFor(() => probe.exports.length === 6);
  assert.equal(probe.exports.at(-1)[0].name, second.source.name, "Deletion during activation skips encoding and drains the next export");
  assert.equal(probe.exports.some(args => args[0].name === third.source.name), false);
  await waitFor(() => document.body.textContent.includes("Your final MP4 was saved."));

  for (const race of ["fingerprint", "instance"]) {
    await act(async () => root.render(null));
    indexedDB = new IDBFactory(); window.localStorage.clear();
    const project = fixture(`synthetic-export-${race}-guard`);
    project.reviewDraft = createCutDraft({ analysisId: store.projectAnalysisId(project), recordingId: project.id,
      duration: project.info.duration, rallies: project.analysis.intervals, ignoredIntervals: [] });
    project.reviewDraft.updatedAt = new Date(Date.now() + 60000).toISOString();
    project.reviewDraft.reviewedCutIds = ["R001"];
    Object.assign(project, await store.putProject(project, { expectedSource: null }));
    window.localStorage.setItem(store.SELECTED_PROJECT_STORAGE_KEY, project.id);
    await act(async () => root.render(React.createElement(App)));
    await waitFor(() => document.querySelector(".taste-rally-desk"));
    const original = new File(["video"], project.source.name, { type: "video/mp4", lastModified: 100 });
    const cached = race === "fingerprint" ? new File(["wrong"], original.name, { type: original.type, lastModified: 100 }) : original;
    await attach(project, cached); await click("Export");
    const held = deferred(), before = probe.exports.length, preparations = probe.preparations.length;
    probe.nextPreparation = () => held.promise;
    try {
      await click("Save MP4"); await waitFor(() => probe.preparations.length === preparations + 1);
      const current = await store.getProject(project.id);
      if (race === "fingerprint") {
        await store.putProject({ ...current, source: { ...current.source, fingerprint: await store.sourceFileFingerprint(original) } }, { expectedSource: current.source });
      } else {
        const replacement = { ...current, reviewDraft: structuredClone(current.reviewDraft) };
        delete replacement.instanceId;
        replacement.reviewDraft.scoreTracking.team1Name = "Replacement-owned review";
        await store.deleteProject(project.id, current.instanceId);
        await store.putProject(replacement, { expectedSource: null });
      }
      const expected = await store.getProject(project.id);
      await act(async () => held.resolve({ ready: true, reason: null }));
      await waitFor(() => probe.exports.length > before || /does not match|instance changed/i.test(document.body.textContent));
      assert.equal(probe.exports.length, before, `${race}: a stale export must be rejected BEFORE the encoder after worker readiness`);
      assert.match(document.body.textContent, /does not match|instance changed/i, "Source conflict must be visible");
      assert.ok(document.querySelector('.td-reconnect input[type="file"]'), "Conflicting source must be unlinked");
      assert.deepEqual(await store.getProject(project.id), expected, "Rejected export must preserve frozen analysis and reviews");
      probe.nextPreparation = null;
      await attach(expected, original); await click("Export"); await click("Save MP4");
      await waitFor(() => probe.exports.length === before + 1 && document.body.textContent.includes("Your final MP4 was saved."));
      assert.equal(probe.exports.at(-1)[0], original, "The authoritative source remains exportable after reconnect");
      assert.deepEqual(probe.exports.at(-1)[1], [{ start: 0, end: 2, cutIds: ["R001"] }], "Source validation must preserve the frozen cut intervals");
    } finally { held.resolve({ ready: true, reason: null }); }
  }
  const staleStatusFailures = [];
  for (const failureMode of ["conflict", "storage"]) {
    await act(async () => root.render(null));
    indexedDB = new IDBFactory(); window.localStorage.clear();
    const project = fixture(`synthetic-export-status-${failureMode}`);
    Object.assign(project, await store.putProject(project, { expectedSource: null }));
    window.localStorage.setItem(store.SELECTED_PROJECT_STORAGE_KEY, project.id);
    window.localStorage.setItem("volleycut:rally-desk-tour:v1", "dismissed");
    await act(async () => root.render(React.createElement(App)));
    await waitFor(() => document.querySelector(".taste-rally-desk"));
    const original = new File(["video"], project.source.name, { type: "video/mp4", lastModified: 100 });
    await attach(project, original); await click("Export");

    // Delay only the old status write's DB-open completion. Its production
    // persistence/token checks remain real; the second case injects a read error.
    const open = indexedDB.open.bind(indexedDB);
    let interceptNext = false, releaseOpen = null, settled = false, rejectOpen;
    indexedDB.open = function (...args) {
      const request = open(...args);
      if (interceptNext) {
        interceptNext = false;
        const add = request.addEventListener.bind(request);
        request.addEventListener = function (type, listener, options) {
          if (type === "error") rejectOpen = listener;
          if (type === "success") add(type, event => {
            releaseOpen = () => {
              if (failureMode === "storage") {
                Object.defineProperty(request, "error", { value: new DOMException("Synthetic delayed storage failure", "UnknownError"), configurable: true });
                rejectOpen.call(request, event); settled = true;
              } else {
                const database = request.result;
                const transaction = database.transaction.bind(database);
                database.transaction = (...transactionArgs) => {
                  const result = transaction(...transactionArgs);
                  result.addEventListener("complete", () => { settled = true; });
                  result.addEventListener("abort", () => { settled = true; });
                  return result;
                };
                listener.call(request, event);
              }
            };
          }, options);
          else add(type, listener, options);
        };
      }
      return request;
    };
    probe.nextExport = async () => { interceptNext = true; };
    try {
      await click("Save MP4"); await waitFor(() => releaseOpen !== null);
      probe.nextExport = null;
      const old = await store.getProject(project.id), recreated = { ...old };
      delete recreated.instanceId;
      await store.deleteProject(project.id, old.instanceId);
      const fresh = await store.putProject(recreated, { expectedSource: null });
      await click("Save MP4");
      await waitFor(() => document.body.textContent.includes("instance changed"));
      await attach(fresh, original);
      const expected = await store.getProject(project.id);
      await act(async () => releaseOpen()); releaseOpen = null;
      await waitFor(() => settled);
      assert.deepEqual(await store.getProject(project.id), expected, "Old status callbacks must preserve replacement storage");
      assert.equal(Boolean(document.querySelector('.td-reconnect input[type="file"]')), false,
        `${failureMode}: old status callbacks must preserve the replacement's verified source link`);
      assert.doesNotMatch(document.body.textContent, /Reopen it before saving export status|exported status could not be saved/,
        `${failureMode}: obsolete status errors must not reach the replacement review`);
    } catch (error) {
      staleStatusFailures.push(`${failureMode}: ${error.message}`);
    } finally {
      if (releaseOpen) await act(async () => releaseOpen());
      indexedDB.open = open; probe.nextExport = null;
    }
  }
  assert.deepEqual(staleStatusFailures, [], "Late export-status conflict/storage failures must be scoped to their original project instance");
  const staleDeleteFailures = [];
  for (const outcome of ["conflict", "storage", "success"]) {
    await act(async () => root.render(null));
    indexedDB = new IDBFactory(); window.localStorage.clear();
    const project = fixture(`synthetic-delete-${outcome}`);
    project.reviewDraft = createCutDraft({ analysisId: store.projectAnalysisId(project), recordingId: project.id,
      duration: project.info.duration, rallies: project.analysis.intervals, ignoredIntervals: [] });
    project.reviewDraft.updatedAt = new Date(Date.now() + 60000).toISOString();
    project.reviewDraft.scoreTracking.team1Name = "Replacement-owned review";
    Object.assign(project, await store.putProject(project, { expectedSource: null }));
    window.localStorage.setItem(store.SELECTED_PROJECT_STORAGE_KEY, project.id);
    window.localStorage.setItem("volleycut:rally-desk-tour:v1", "dismissed");
    await act(async () => root.render(React.createElement(App)));
    await waitFor(() => document.querySelector(".taste-rally-desk"));
    const original = new File(["video"], project.source.name, { type: "video/mp4", lastModified: 100 });
    await attach(project, original); await click("Export");

    // Hold the old delete's open result for failures, or its transaction-complete
    // delivery for success. Production storage and instance checks remain real.
    const open = indexedDB.open.bind(indexedDB);
    let interceptNext = true, release = null, settled = false, rejectOpen;
    indexedDB.open = function (...args) {
      const request = open(...args);
      if (!interceptNext) return request;
      interceptNext = false;
      const add = request.addEventListener.bind(request);
      request.addEventListener = function (type, listener, options) {
        if (type === "error") rejectOpen = listener;
        if (type !== "success") { add(type, listener, options); return; }
        add(type, event => {
          const deliver = () => {
            if (outcome === "storage") {
              request.result.close();
              Object.defineProperty(request, "error", { value: new DOMException("Synthetic delayed delete storage failure", "UnknownError"), configurable: true });
              rejectOpen.call(request, event); settled = true;
              return;
            }
            const database = request.result, transaction = database.transaction.bind(database);
            database.transaction = (...transactionArgs) => {
              const result = transaction(...transactionArgs);
              if (outcome === "success") {
                const addTransaction = result.addEventListener.bind(result);
                let heldCompletion = false;
                result.addEventListener = function (eventType, callback, eventOptions) {
                  if (eventType === "complete" && !heldCompletion) {
                    heldCompletion = true;
                    addTransaction(eventType, completeEvent => {
                      release = () => { callback.call(result, completeEvent); settled = true; };
                    }, eventOptions);
                  } else addTransaction(eventType, callback, eventOptions);
                };
              } else {
                result.addEventListener("complete", () => { settled = true; });
                result.addEventListener("abort", () => { settled = true; });
              }
              return result;
            };
            listener.call(request, event);
          };
          if (outcome === "success") deliver();
          else release = deliver;
        }, options);
      };
      return request;
    };
    try {
      await click("Delete"); await waitFor(() => release !== null);
      const old = await store.getProject(project.id);
      if (outcome === "success") assert.equal(old, null, "The old delete committed before completion delivery");
      else await store.deleteProject(project.id, old.instanceId);
      const recreated = { ...project }; delete recreated.instanceId;
      const fresh = await store.putProject(recreated, { expectedSource: null });
      await click("Save MP4");
      await waitFor(() => document.body.textContent.includes("instance changed"));
      await attach(fresh, original);
      const expected = await store.getProject(project.id);
      await act(async () => release()); release = null;
      await waitFor(() => settled);
      assert.deepEqual(await store.getProject(project.id), expected, "Late delete callbacks must preserve replacement storage");
      assert.ok(document.querySelector(".taste-rally-desk"), `${outcome}: replacement editor must remain open`);
      assert.equal(document.querySelector('select[aria-label="Current review project"]')?.value, project.id,
        `${outcome}: replacement project must remain selected`);
      assert.equal(Boolean(document.querySelector('.td-reconnect input[type="file"]')), false,
        `${outcome}: replacement must retain its verified source link`);
      assert.doesNotMatch(document.body.textContent, /Reopen it before deleting|Synthetic delayed delete storage failure/,
        `${outcome}: obsolete delete errors must not reach the replacement review`);
    } catch (error) {
      staleDeleteFailures.push(`${outcome}: ${error.message}`);
    } finally {
      if (release) await act(async () => release());
      indexedDB.open = open;
    }
  }
  assert.deepEqual(staleDeleteFailures, [], "Late delete conflict, storage and success callbacks must belong to their captured instance");
  console.log("App export: awaited readiness, project queue, failure/retry, cancellation, native target, deletion and source/instance races and delayed status/delete callbacks passed.");
} finally {
  await act(async () => root.unmount());
  await server.close(); dom.window.close();
  URL.createObjectURL = originalUrls.create; URL.revokeObjectURL = originalUrls.revoke;
  delete globalThis.__appExportProbe;
}
