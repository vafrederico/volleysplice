import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { IDBFactory } from "fake-indexeddb";
import { JSDOM } from "jsdom";
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { createServer } from "vite";
import { recoveryFixture, servingFixture, switchFixture } from "./score-recovery-fixtures.mjs";

// Actual App, queue, coordinator, planners and cached model loaders. Only media
// opening/rally inference are stubbed; empty synthetic candidates need no GPU.
// The held fetch remains unresolved until AFTER cancellation and the next job.
async function withApp(test) {
  const harness = { opened: 0, disposed: 0, rallyRuns: 0 };
  globalThis.__scoreCancellationHarness = harness;
  const server = await createServer({
    root: fileURLToPath(new URL("..", import.meta.url)),
    server: { middlewareMode: true, hmr: false, ws: false }, appType: "custom",
    plugins: [{ name: "synthetic-source-only", enforce: "pre", transform(_code, id) {
      if (id.endsWith("/src/lib/on-device/media.ts")) return `
        export async function openLocalMedia() {
          const h = globalThis.__scoreCancellationHarness; h.opened++;
          let disposed = false;
          return { info: h.info, input: { dispose() { if (!disposed) { disposed = true; h.disposed++; } } } };
        }`;
      if (id.endsWith("/src/lib/on-device/pipeline.ts")) return `
        export const DEFAULT_FEATURE_REDUCTION_KERNEL = "synthetic";
        export const DEFAULT_VIDEO_DECODE_STRATEGY = "synthetic";
        export const VIDEO_DECODER_HARDWARE_ACCELERATION = "synthetic";
        export async function analyzeOpenedMedia() { globalThis.__scoreCancellationHarness.rallyRuns++; throw new Error("Rally rerun was not requested"); }`;
    } }],
  });
  const dom = new JSDOM('<script data-volleycut-opencv="true"></script><div id="root"></div>', {
    url: "https://example.test", pretendToBeVisual: true,
  });
  const globals = {
    window: dom.window, document: dom.window.document, navigator: dom.window.navigator,
    Element: dom.window.Element, HTMLElement: dom.window.HTMLElement, IS_REACT_ACT_ENVIRONMENT: true,
    indexedDB: new IDBFactory(), ResizeObserver: class { observe() {} disconnect() {} },
    requestAnimationFrame: (callback) => setTimeout(callback, 0), cancelAnimationFrame: clearTimeout,
  };
  const previous = Object.fromEntries([...Object.keys(globals), "fetch"].map((key) => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  for (const [key, value] of Object.entries(globals)) Object.defineProperty(globalThis, key, { value, writable: true, configurable: true });
  window.cv = {};
  window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
  window.scrollTo = () => {};
  window.isSecureContext = true;
  window.VideoDecoder = window.AudioDecoder = window.VideoFrame = class {};
  dom.window.HTMLElement.prototype.scrollIntoView = () => {};
  dom.window.HTMLCanvasElement.prototype.getContext = () => null;
  dom.window.HTMLMediaElement.prototype.pause = () => {};
  dom.window.HTMLDialogElement.prototype.showModal = function () { this.open = true; };
  dom.window.HTMLDialogElement.prototype.close = function () { this.open = false; };
  const originalUrls = { create: URL.createObjectURL, revoke: URL.revokeObjectURL };
  URL.createObjectURL = () => "blob:synthetic"; URL.revokeObjectURL = () => {};
  const root = createRoot(document.getElementById("root"));
  const unhandled = [], onUnhandled = (error) => unhandled.push(error);
  process.on("unhandledRejection", onUnhandled);
  const waitFor = async (condition, message = "UI did not settle") => {
    for (let i = 0; i < 150; i++) {
      let ready;
      await act(async () => { ready = await condition(); await new Promise((resolve) => setTimeout(resolve, 5)); });
      if (ready) return;
    }
    assert.fail(message);
  };
  const button = (label) => [...document.querySelectorAll("button")].find((node) => node.textContent === label);
  const click = async (label) => {
    const target = button(label); assert.ok(target, label); assert.equal(target.disabled, false);
    await act(async () => target.click());
  };
  try {
    const { App } = await server.ssrLoadModule("/src/App.tsx");
    const store = await server.ssrLoadModule("/src/lib/project-store.ts");
    const serving = await server.ssrLoadModule("/src/lib/on-device/serving-side-cache.ts");
    const neural = await server.ssrLoadModule("/src/lib/on-device/neural-serving-model.ts");
    const switches = await server.ssrLoadModule("/src/lib/on-device/side-switch-model.ts");
    const { PRODUCTION_ENSEMBLE_MODEL_ID } = await server.ssrLoadModule("/src/lib/on-device/ensemble.ts");
    const fixture = (switchesEnabled = false) => recoveryFixture({ empty: true, switches: switchesEnabled, modelId: PRODUCTION_ENSEMBLE_MODEL_ID });
    const read = async (id) => (await store.listProjects()).find((project) => project.id === id);
    const mount = async (projects) => {
      for (const project of projects) Object.assign(project, await store.putProject(project, { expectedSource: null }));
      harness.info = projects[0].info;
      window.localStorage.setItem(store.SELECTED_PROJECT_STORAGE_KEY, projects[0].id);
      await act(async () => root.render(React.createElement(App)));
      await waitFor(() => document.querySelector(".taste-rally-desk"));
    };
    const select = async (id) => {
      const target = document.querySelector('select[aria-label="Current review project"]');
      await act(async () => { target.value = id; target.dispatchEvent(new window.Event("change", { bubbles: true })); });
      await waitFor(() => document.querySelector('select[aria-label="Current review project"]')?.value === id);
    };
    const attach = async (name) => {
      const input = document.querySelector('.td-reconnect input[type="file"]'); assert.ok(input);
      Object.defineProperty(input, "files", { value: [new File(["video"], name, { type: "video/mp4", lastModified: 100 })], configurable: true });
      await act(async () => input.dispatchEvent(new window.Event("change", { bubbles: true })));
      await waitFor(() => button("Generate missing score markers"));
    };
    await test({ root, server, fixture, store, serving, neural, switches, harness, read, mount, select, attach, waitFor, click, button });
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 20)); });
    assert.deepEqual(unhandled, [], "Late model rejection must stay handled");
  } finally {
    await act(async () => root.unmount());
    await server.close(); dom.window.close(); process.off("unhandledRejection", onUnhandled);
    URL.createObjectURL = originalUrls.create; URL.revokeObjectURL = originalUrls.revoke;
    for (const [key, descriptor] of Object.entries(previous)) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
    }
    delete globalThis.__scoreCancellationHarness;
  }
}

for (const completion of ["never", "resolve", "reject"]) await withApp(async ({ fixture, serving, neural, switches, harness, read, mount, select, attach, waitFor, click, button }) => {
  let resolveTransport, rejectTransport, settled = false, switchFetches = 0, transportAborts = 0;
  const transport = new Promise((resolve, reject) => { resolveTransport = resolve; rejectTransport = reject; });
  globalThis.fetch = async (url, options) => {
    const asset = new URL(url).pathname.split("/").at(-1);
    if (asset === switches.SIDE_SWITCH_ASSET && ++switchFetches === 1) {
      options?.signal?.addEventListener("abort", () => transportAborts++);
      return transport;
    }
    return new Response(await readFile(new URL(`../public/runtime/${asset}`, import.meta.url)));
  };
  const first = fixture(true), second = fixture();
  first.analysis.servingSide = servingFixture(first.analysis, serving, neural);
  second.id = "synthetic-next-project"; second.source.name = "second.mp4";
  await mount([first, second]); await attach(first.source.name);
  await click("Generate missing score markers"); await waitFor(() => switchFetches === 1);
  const sharedRuntime = switches.loadSideSwitchRuntime();
  await click("Cancel score markers");
  await waitFor(() => button("Generate missing score markers"), "Cancellation waited for the stalled model transport");
  await waitFor(async () => (await read(first.id)).scoreSpecialistErrors?.sideSwitch.includes("cancelled"));
  assert.equal(settled, false); assert.equal(transportAborts, 0);
  assert.strictEqual(switches.loadSideSwitchRuntime(), sharedRuntime, "One consumer's cancellation cannot evict the shared pending runtime");
  await select(second.id); await attach(second.source.name); await click("Generate missing score markers");
  await waitFor(async () => (await read(second.id)).analysis.servingSide, "The next project did not finish while the first fetch was still pending");
  await waitFor(() => !document.querySelector('[aria-label="Score prediction recovery"]'));
  assert.equal(settled, false, "The queue must drain without releasing the first fetch");
  assert.equal(harness.opened, 2); assert.equal(harness.disposed, 2); assert.equal(harness.rallyRuns, 0);
  const snapshot = async () => JSON.stringify([await read(first.id), await read(second.id)]);
  const beforeLate = await snapshot();
  if (completion !== "never") {
    const bytes = await readFile(new URL(`../public/runtime/${switches.SIDE_SWITCH_ASSET}`, import.meta.url));
    await act(async () => {
      settled = true;
      if (completion === "resolve") resolveTransport(new Response(bytes));
      else rejectTransport(new TypeError("Synthetic late model rejection"));
      await new Promise((resolve) => setTimeout(resolve, 20));
    });
    assert.equal(await snapshot(), beforeLate, "An abandoned planner must not checkpoint into either project");
    assert.equal(transportAborts, 0);
    await select(first.id);
    assert.match(document.body.textContent, /Score markers cancelled/);
    await click("Generate missing score markers");
    await waitFor(async () => (await read(first.id)).analysis.sideSwitch);
    assert.equal(switchFetches, completion === "resolve" ? 1 : 2, "Retry reuses a fulfilled shared load or replaces its rejected promise");
    assert.deepEqual((await read(first.id)).analysis.servingSide, first.analysis.servingSide);
    assert.equal(JSON.stringify((await read(second.id)).analysis), JSON.stringify(JSON.parse(beforeLate)[1].analysis));
  }
});

// Keep the new live model-draft refresh compatible with the switch preference
// contract from PR17, without importing its other changes or source snapshots.
await withApp(async ({ root, server, fixture, switches, waitFor }) => {
  const { useDesignReview } = await server.ssrLoadModule("/src/designs/useDesignReview.ts");
  let review;
  function Review({ project }) {
    const projects = React.useMemo(() => [project], [project]);
    review = useDesignReview({ projects, projectId: project.id });
    return React.createElement("p", null, review.state);
  }
  let project = fixture(false);
  project.analysis.sideSwitch = switchFixture(switches, [{ id: "cached-switch", timestamp: 30, probability: .9, kind: "adjacent-rally-boundary", sourceRangeIds: [] }]);
  await act(async () => root.render(React.createElement(Review, { project })));
  await waitFor(() => review.state === "ready");
  assert.deepEqual(review.modelDraft.scoreTracking.sideSwitchMarkers, []);
  project = { ...project, analysis: { ...project.analysis } };
  await act(async () => root.render(React.createElement(Review, { project })));
  assert.deepEqual(review.modelDraft.scoreTracking.sideSwitchMarkers, [], "A scoreRecovery refresh must forward disabled detection even with cached raw evidence");
  project = { ...project, sideSwitchEnabled: true };
  await act(async () => root.render(React.createElement(Review, { project })));
  assert.equal(review.modelDraft.scoreTracking.sideSwitchMarkers.length, 1, "The refresh must forward the actual preference, not hard-code false");
});

console.log("App + real coordinator: stalled-fetch cancellation drains the queue before transport settles; never/late resolve/late reject, cache isolation, retry, checkpoint ordering and disabled-switch model-draft refresh passed.");
