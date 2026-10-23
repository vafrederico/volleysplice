import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { IDBFactory } from "fake-indexeddb";
import { JSDOM } from "jsdom";
import React, { act } from "react";
import { createServer } from "vite";

// Mount the real App and IndexedDB store without listening on a port. Only media,
// inference and editor presentation are replaced; setup, queue and reuse run normally.
const probe = globalThis.__projectSetupProbe = { runs: [], specialists: [], editor: null };
const stubs = {
  "lib/on-device/media": `export async function openLocalMedia() {
    return { info: globalThis.__projectSetupProbe.info, input: { dispose() {} } };
  }`,
  "lib/on-device/pipeline": `
    export const DEFAULT_FEATURE_REDUCTION_KERNEL = 'javascript';
    export const DEFAULT_VIDEO_DECODE_STRATEGY = 'sparse';
    export const VIDEO_DECODER_HARDWARE_ACCELERATION = 'prefer-hardware';
    export async function analyzeOpenedMedia(media, roi, path, runtime, progress, source, options) {
      globalThis.__projectSetupProbe.runs.push({ roi, source, options });
      return structuredClone(globalThis.__projectSetupProbe.analysis);
    }`,
  "lib/on-device/score-specialists": `export async function inferScoreSpecialists(media, roi, options) {
    globalThis.__projectSetupProbe.specialists.push({ roi, sideSwitch: Boolean(options.sideSwitch) });
    return {};
  }`,
  "designs/useDesignReview": `export function useDesignReview(options) {
    globalThis.__projectSetupProbe.editor = options;
    return { ...options, state: 'ready' };
  }`,
  "designs/taste": `import React from 'react';
    export function RallyDesk({ review }) {
      return React.createElement('button', { onClick: () => review.onSelectProject(null) }, 'Start a new video');
    }`,
  "components/GuidedTour": "export function GuidedTour() { return null; }",
};
const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false },
  appType: "custom",
  plugins: [{
    name: "project-setup-boundaries", enforce: "pre",
    resolveId(id) {
      const path = id.replaceAll("\\", "/").replace(/\.[jt]sx?$/, "");
      const key = Object.keys(stubs).find(key => path.endsWith(`/src/${key}`) || path === `@/${key}`);
      if (key) return `\0project-setup:${key}`;
    },
    load(id) {
      if (id.startsWith("\0project-setup:")) return stubs[id.slice("\0project-setup:".length)];
    },
  }],
});
const dom = new JSDOM('<div id="root"></div>', { url: "https://example.test/?new=1" });
Object.assign(globalThis, {
  window: dom.window, document: dom.window.document,
  IS_REACT_ACT_ENVIRONMENT: true, indexedDB: new IDBFactory(),
});
Object.defineProperty(globalThis, "navigator", { value: dom.window.navigator, configurable: true });
Object.assign(window, { VideoDecoder: class {}, AudioDecoder: class {}, VideoFrame: class {},
  isSecureContext: true, scrollTo() {} });
window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
window.HTMLMediaElement.prototype.pause = () => {};
const { createRoot } = await import("react-dom/client");
let root = createRoot(document.getElementById("root"));
const button = text => [...document.querySelectorAll("button")].find(node => node.textContent === text);
const click = async text => {
  assert.ok(button(text), `Missing button: ${text}`);
  await act(async () => button(text).click());
};
const settle = async condition => {
  for (let i = 0; i < 100; i++) {
    if (await condition()) return;
    await act(async () => { await new Promise(resolve => setTimeout(resolve, 10)); });
  }
  assert.fail("Setup did not settle");
};
const change = async (selector, value) => {
  const input = document.querySelector(selector);
  assert.ok(input, selector);
  const prototype = input instanceof window.HTMLSelectElement
    ? window.HTMLSelectElement.prototype : window.HTMLInputElement.prototype;
  await act(async () => {
    Object.getOwnPropertyDescriptor(prototype, "value").set.call(input, value);
    input.dispatchEvent(new window.Event("input", { bubbles: true }));
    input.dispatchEvent(new window.Event("change", { bubbles: true }));
  });
};

try {
  const store = await server.ssrLoadModule("/src/lib/project-store.ts");
  const { createCutDraft } = await server.ssrLoadModule("/src/lib/cut-draft.ts");
  const { restoreProjectReviewDraft } = await server.ssrLoadModule("/src/lib/review-draft-persistence.ts");
  const { PRODUCTION_ENSEMBLE_MODEL_ID } = await server.ssrLoadModule("/src/lib/on-device/ensemble.ts");
  const file = new File([new Uint8Array(123_456)], "match.mp4", {
    type: "video/mp4", lastModified: 1_786_752_000_000,
  });
  probe.info = { duration: 90, width: 1920, height: 1080, rotation: 0,
    mimeType: "video/mp4", videoCodec: "avc", videoCodecString: "avc1.640028",
    canDecodeVideo: true, hasAudio: true, audioCodec: "aac", sampleRate: 48_000,
    channels: 2, canDecodeAudio: true };
  probe.analysis = { modelId: PRODUCTION_ENSEMBLE_MODEL_ID, featurePath: "local-source",
    intervals: [{ id: "R001", start: 10, end: 20, confidence: 0.9, included: true, agreement: "both-models" }],
    times: new Float64Array([0]), rallyProbabilities: new Float32Array([0]),
    serveProbabilities: new Float32Array([0]), deadStateProbabilities: new Float32Array([0]),
    productionComponents: { allLabelsV2: [], previousProduction: [] },
    productionServeOutputs: Object.fromEntries(["allLabelsV2", "previousProduction"].map(key =>
      [key, { probabilities: new Float32Array([0]), detections: [] }])),
    productionStateOutputs: Object.fromEntries(["allLabelsV2", "previousProduction"].map(key =>
      [key, { rallyProbabilities: new Float32Array([0]), deadStateProbabilities: new Float32Array([0]) }])),
  };
  const saved = { schemaVersion: 1, id: "project-1v8p28b", source: store.projectSource(file),
    info: probe.info, analysisWindow: { start: 0, end: 90 },
    roi: { x: 0, y: 0, width: 1, height: 1 }, sideSwitchEnabled: false,
    status: "ready", analysis: probe.analysis, error: null,
    createdAt: "2026-01-01T00:00:00.000Z", updatedAt: "2026-01-01T00:00:00.000Z" };
  const seed = { analysisId: store.projectAnalysisId(saved), recordingId: saved.id,
    duration: 90, rallies: saved.analysis.intervals, ignoredIntervals: [] };
  saved.reviewDraft = createCutDraft(seed);
  saved.reviewDraft.cuts[0].included = false;
  saved.reviewDraft.reviewedCutIds = ["R001"];
  Object.assign(saved, await store.putProject(saved, { expectedSource: null }));
  const original = (await store.listProjects())[0];
  const { App } = await server.ssrLoadModule("/src/App.tsx");
  await act(async () => root.render(React.createElement(App)));
  await settle(() => document.querySelector('input[accept^="video/"]')?.disabled === false);

  const setup = async ({ crop = "1", switches = false,
    cropChanges = [["width", crop]], gameWindow = null } = {}) => {
    probe.editor = null;
    if (button("Start a new video")) await click("Start a new video");
    const input = document.querySelector('input[accept^="video/"]');
    assert.ok(input, "The real source picker is shown");
    Object.defineProperty(input, "files", { value: [file], configurable: true });
    await act(async () => input.dispatchEvent(new window.Event("change", { bubbles: true })));
    await change('[aria-label="Rally detection model"]', "ensemble");
    for (const [field, value] of cropChanges) {
      await change(`[aria-label="Feature crop ${field}"]`, value);
    }
    if (gameWindow) {
      await change('[aria-label="Game start in seconds"]', gameWindow.start);
      await change('[aria-label="Game end in seconds"]', gameWindow.end);
    }
    if (switches) await act(async () => document.querySelector('input[type="checkbox"]').click());
    await click("Find the rallies");
    await settle(() => probe.editor !== null);
    await settle(async () => (await store.listProjects()).some(p => p.id === probe.editor.projectId && p.status === "ready"));
    return probe.editor.projectId;
  };

  assert.equal(await setup(), saved.id, "Matching legacy setup reopens without inference");
  assert.equal(probe.runs.length, 0);
  assert.equal(probe.editor.sourceFile, file, "Legacy source link uses the original project ID");
  const croppedId = await setup({ crop: "0.8" });
  assert.match(croppedId, /^project-v2-/);
  assert.notEqual(croppedId, saved.id);
  assert.equal(probe.runs.length, 1);
  assert.equal(probe.runs[0].roi.width, 0.8, "Requested crop reaches the real queue worker");
  assert.equal(probe.runs[0].options.rallyModel, "ensemble");
  assert.deepEqual(probe.runs[0].options.analysisWindow, { start: 0, end: 90 });
  assert.equal(await setup({ crop: "0.8" }), croppedId);
  assert.equal(probe.runs.length, 1, "Identical current setup reuses saved inference");
  const switchId = await setup({ switches: true });
  assert.notEqual(switchId, saved.id);
  assert.notEqual(switchId, croppedId);
  assert.equal(probe.runs.length, 2);
  assert.deepEqual(probe.specialists.map(call => call.sideSwitch), [false, true],
    "Requested team switches reach specialist execution");
  assert.equal(await setup({ switches: true }), switchId);
  assert.equal(probe.runs.length, 2);

  const clampedId = await setup({ cropChanges: [["x", "0.7"]] });
  assert.equal(probe.runs.length, 3);
  assert.equal(probe.runs[2].roi.width, 0.30000000000000004,
    "X first exercises the real clamp's floating-point representation");
  assert.equal(await setup({ cropChanges: [["width", "0.3"], ["x", "0.7"]] }), clampedId,
    "Width first selects the equivalent saved crop despite raw float differences");
  assert.equal(probe.runs.length, 3, "Equivalent crop control order must not rerun inference");
  const narrowerId = await setup({ cropChanges: [["width", "0.29"], ["x", "0.7"]] });
  assert.notEqual(narrowerId, clampedId, "Genuinely different crops retain separate projects");
  assert.equal(probe.runs.length, 4);
  const windowedId = await setup({ gameWindow: { start: "10", end: "80" } });
  assert.equal(probe.runs.length, 5);
  assert.equal(await setup(), saved.id, "Returning to the original configuration keeps the legacy project");
  const projects = await store.listProjects();
  assert.equal(projects.length, 6);
  assert.deepEqual(projects.find(p => p.id === saved.id), original,
    "New analyses never mutate the legacy project or its review");
  assert.equal(projects.find(p => p.id === switchId).sideSwitchEnabled, true);
  assert.equal(projects.find(p => p.id === croppedId).roi.width, 0.8);

  await click("Start a new video");
  const options = [...document.querySelectorAll('[aria-label="Selected project"] option')]
    .filter(option => option.value !== "__new__");
  assert.equal(options.length, projects.length);
  assert.equal(new Set(options.map(option => option.textContent)).size, projects.length,
    "The real project picker distinguishes ready projects for the same source/model");
  const label = id => options.find(option => option.value === id).textContent;
  assert.match(label(saved.id), /^match\.mp4 - Legacy model.*Full frame.*Full video.*Court switches off/);
  assert.match(label(croppedId), /Crop x 0%, y 0%, 80% × 100%/);
  assert.match(label(clampedId), /Crop x 70%, y 0%, 30% × 100%/);
  assert.match(label(narrowerId), /Crop x 70%, y 0%, 29% × 100%/);
  assert.match(label(switchId), /Court switches on/);
  assert.match(label(windowedId), /Game 10–80 s/);
  assert.deepEqual(await store.listProjects(), projects, "Displaying labels never rewrites project IDs or sources");
  await change('[aria-label="Selected project"]', clampedId);
  assert.equal(probe.editor.projectId, clampedId, "A labeled crop variant opens its own saved analysis");
  await click("Start a new video");
  await change('[aria-label="Selected project"]', saved.id);
  assert.equal(probe.editor.projectId, saved.id, "The legacy label still opens the original saved review");

  await act(async () => root.unmount());
  root = createRoot(document.getElementById("root"));
  await act(async () => root.render(React.createElement(App)));
  await settle(() => probe.editor?.sourceFile === null);
  assert.equal(probe.editor.projectId, saved.id, "Selected legacy ID survives a fresh App mount");
  assert.deepEqual(restoreProjectReviewDraft(projects.find(p => p.id === saved.id), seed, null), saved.reviewDraft);
  await act(async () => { await probe.editor.onAttachSource(file); });
  await settle(() => probe.editor?.sourceFile === file);
  await settle(async () => (await store.listProjects()).find(p => p.id === saved.id)?.status === "ready");
  assert.equal(probe.editor.projectId, saved.id, "Relinking retains the legacy identity");
  assert.deepEqual((await store.listProjects()).find(p => p.id === saved.id).reviewDraft, saved.reviewDraft);
  console.log("Project setup: changed crop, side-switch, reuse and legacy restore passed");
} finally {
  await act(async () => root.unmount());
  await server.close();
  dom.window.close();
  delete globalThis.__projectSetupProbe;
}
