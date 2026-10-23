import assert from "node:assert/strict";
import test from "node:test";
import { setImmediate as nextTurn } from "node:timers/promises";
import { fileURLToPath } from "node:url";
import { IDBFactory } from "fake-indexeddb";
import { JSDOM } from "jsdom";
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { createServer } from "vite";
import { recoveryFixture, servingFixture, switchFixture } from "./score-recovery-fixtures.mjs";

// Integration contract for PR5/13/17/27. Actual App, queue, editor and IDB store;
// only media decoding and inference boundaries are controlled. All inputs are
// synthetic. Middleware mode opens no listener and every case closes its server.
// The instanceId arguments below use PR27's public API. Before that API lands,
// JS ignores the extra arguments, allowing the old behavior to reproduce.
async function withApp(body) {
  const h = { opened: [], disposed: [], requests: [], rallyRuns: 0, expectedRallyRuns: 0, running: 0, evidenceRuns: 0,
    cleanups: [], behavior: null, evidenceBehavior: null };
  const dom = new JSDOM('<div id="root"></div>', { url: "https://example.test", pretendToBeVisual: true });
  const globals = { window: dom.window, document: dom.window.document,
    Node: dom.window.Node, Element: dom.window.Element, HTMLElement: dom.window.HTMLElement,
    navigator: dom.window.navigator, indexedDB: new IDBFactory(), IS_REACT_ACT_ENVIRONMENT: true,
    ResizeObserver: class { observe() {} disconnect() {} },
    requestAnimationFrame: callback => setTimeout(callback, 0), cancelAnimationFrame: clearTimeout,
    __scoreSourceIntegration: h };
  const previous = Object.fromEntries(Object.keys(globals).map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  for (const [key, value] of Object.entries(globals)) Object.defineProperty(globalThis, key, { value, writable: true, configurable: true });
  window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
  window.scrollTo = () => {}; window.confirm = () => true; window.isSecureContext = true;
  window.VideoDecoder = window.AudioDecoder = window.VideoFrame = class {};
  dom.window.HTMLElement.prototype.scrollIntoView = () => {};
  dom.window.HTMLCanvasElement.prototype.getContext = () => null;
  dom.window.HTMLMediaElement.prototype.pause = () => {};
  dom.window.HTMLDialogElement.prototype.showModal = function () { this.open = true; };
  dom.window.HTMLDialogElement.prototype.close = function () { this.open = false; };
  const urls = { create: URL.createObjectURL, revoke: URL.revokeObjectURL };
  URL.createObjectURL = () => "blob:synthetic-score-source"; URL.revokeObjectURL = () => {};
  const unhandled = [], onUnhandled = cause => unhandled.push(cause);
  process.on("unhandledRejection", onUnhandled);
  let server, root;
  const tick = () => act(async () => { await nextTurn(); });
  const until = async (condition, message = "App/persistence did not settle") => {
    for (let i = 0; i < 250; i++) {
      let ready;
      await act(async () => { ready = await condition(); await new Promise(resolve => setTimeout(resolve, 5)); });
      if (ready) return;
    }
    assert.fail(message);
  };
  const button = label => [...document.querySelectorAll("button")].find(node => node.getAttribute("aria-label") === label || node.textContent === label || node.querySelector(":scope > strong")?.textContent === label);
  const click = async label => {
    const target = button(label); assert.ok(target, `Missing button: ${label}`); assert.equal(target.disabled, false, label);
    await act(async () => target.click());
  };
  const gate = () => {
    let resolve; const promise = new Promise(r => { resolve = r; });
    h.cleanups.push(resolve);
    return { promise, release: () => act(async () => resolve()) };
  };
  try {
    server = await createServer({ root: fileURLToPath(new URL("..", import.meta.url)),
      server: { middlewareMode: true, hmr: false, ws: false }, appType: "custom",
      plugins: [{ name: "controlled-score-source-boundaries", enforce: "pre", transform(_code, id) {
        if (id.endsWith("/src/lib/on-device/media.ts")) return `
          export async function openLocalMedia(file) {
            const h = globalThis.__scoreSourceIntegration; h.opened.push(file);
            let disposed = false;
            return { info: h.info, sourceFile: file, input: { dispose() {
              if (!disposed) { disposed = true; h.disposed.push(file); }
            } } };
          }`;
        if (id.endsWith("/src/lib/on-device/pipeline.ts")) return `
          export const DEFAULT_FEATURE_REDUCTION_KERNEL = "synthetic";
          export const DEFAULT_VIDEO_DECODE_STRATEGY = "synthetic";
          export const VIDEO_DECODER_HARDWARE_ACCELERATION = "synthetic";
          export async function analyzeOpenedMedia() {
            const h = globalThis.__scoreSourceIntegration; h.rallyRuns++; h.running++;
            try {
              if (!h.rallyBehavior) throw new Error("Ready score recovery must never rerun rally inference");
              return await h.rallyBehavior();
            } finally { h.running--; }
          }`;
        if (id.endsWith("/src/lib/on-device/score-specialists.ts")) return `
          export async function inferScoreSpecialists(media, roi, request, options = {}) {
            const h = globalThis.__scoreSourceIntegration;
            h.requests.push({ file: media.sourceFile, stages: Object.keys(request).filter(key => request[key]) });
            h.running++;
            try { return await h.behavior(request, options, media.sourceFile); }
            finally { h.running--; }
          }`;
        if (id.endsWith("/src/lib/on-device/production-inference.ts")) return `
          export async function augmentStoredAnalysisWithSuppression(...args) {
            const h = globalThis.__scoreSourceIntegration;
            if (!h.augmentBehavior) return null;
            h.running++;
            try { return await h.augmentBehavior(...args); }
            finally { h.running--; }
          }
          export async function runProductionInferenceFromFeatures(...args) {
            const h = globalThis.__scoreSourceIntegration; h.running++; h.evidenceRuns++;
            try { return await h.evidenceBehavior(...args); }
            finally { h.running--; }
          }`;
      } }],
    });
    assert.equal(server.httpServer, null, "The harness must not open a listener");
    const { App } = await server.ssrLoadModule("/src/App.tsx");
    const store = await server.ssrLoadModule("/src/lib/project-store.ts");
    const serving = await server.ssrLoadModule("/src/lib/on-device/serving-side-cache.ts");
    const neural = await server.ssrLoadModule("/src/lib/on-device/neural-serving-model.ts");
    const switches = await server.ssrLoadModule("/src/lib/on-device/side-switch-model.ts");
    const { createCutDraft } = await server.ssrLoadModule("/src/lib/cut-draft.ts");
    const { addSideSwitchMarker } = await server.ssrLoadModule("/src/lib/score-tracking.ts");
    const { PRODUCTION_ENSEMBLE_MODEL_ID } = await server.ssrLoadModule("/src/lib/on-device/ensemble.ts");
    const suppression = await server.ssrLoadModule("/src/lib/on-device/suppression-model.ts");
    const { SUPPRESSION_POLICY_CONTRACT_VERSION } = await server.ssrLoadModule("/src/lib/on-device/suppression-policy.ts");
    const emptySuppression = () => ({ modelId: suppression.SUPPRESSION_MODEL_ID,
      artifactSha256: suppression.SUPPRESSION_ARTIFACT_SHA256, weightsSha256: suppression.SUPPRESSION_WEIGHTS_SHA256,
      decoderVersion: suppression.SUPPRESSION_DECODER_VERSION, policyContractVersion: SUPPRESSION_POLICY_CONTRACT_VERSION,
      probabilities: new Float32Array([0]), decodedIntervals: [], suggestions: [], identicalPolicyResults: true });
    const read = async id => (await store.listProjects()).find(project => project.id === id);
    const file = (name = "synthetic-score-source.mp4", bytes = "video") => new File([bytes], name, { type: "video/mp4", lastModified: 100 });
    const fixture = (id = "synthetic-source-project", sourceFile = file()) => ({ ...recoveryFixture(), id, source: store.projectSource(sourceFile) });
    const draftFor = project => createCutDraft({ analysisId: store.projectAnalysisId(project), recordingId: project.id,
      duration: project.info.duration, rallies: project.analysis.intervals, ignoredIntervals: [] });
    const mount = async projects => {
      for (const project of projects) Object.assign(project, await store.putProject(project, { expectedSource: null }));
      h.info = projects[0].info;
      window.localStorage.setItem(store.SELECTED_PROJECT_STORAGE_KEY, projects[0].id);
      root = createRoot(document.getElementById("root"));
      await act(async () => root.render(React.createElement(App)));
      await until(() => projects[0].status === "ready" ? document.querySelector(".taste-rally-desk") : document.body.textContent.includes("Choose original video"));
    };
    const chooseSource = async sourceFile => {
      const input = document.querySelector('.td-reconnect input[type="file"],input[accept^="video/"]'); assert.ok(input, "Source reconnect picker");
      Object.defineProperty(input, "files", { value: [sourceFile], configurable: true });
      await act(async () => input.dispatchEvent(new window.Event("change", { bubbles: true })));
    };
    const attach = async sourceFile => {
      await chooseSource(sourceFile);
      await until(() => button("Generate missing score markers"), "Original source must remain reconnectable");
    };
    const select = async id => {
      const input = document.querySelector('select[aria-label="Current review project"]'); assert.ok(input);
      await act(async () => { input.value = id; input.dispatchEvent(new window.Event("change", { bubbles: true })); });
      await until(() => document.querySelector('select[aria-label="Current review project"]')?.value === id);
    };
    const idle = () => until(() => h.running === 0 && !button("Cancel score markers"), "The selected score job must drain");
    const result = request => ({
      ...(request.servingSide ? { servingSide: servingFixture(request.servingSide.analysis, serving, neural) } : {}),
      ...(request.sideSwitch ? { sideSwitch: switchFixture(switches, [{ id: "synthetic-auto-switch", timestamp: 30,
        probability: .9, kind: "adjacent-rally-boundary", sourceRangeIds: [] }]) } : {}),
    });
    h.behavior = async (request, options) => { const output = result(request); await options.onOutput?.(output); return output; };
    const generate = () => click("Generate missing score markers");
    await body({ h, store, read, file, fixture, draftFor, mount, attach, select, idle, result,
      generate, gate, until, tick, click, button, addSideSwitchMarker, chooseSource, PRODUCTION_ENSEMBLE_MODEL_ID, emptySuppression, server });
  } finally {
    try {
      if (root) await act(async () => root.unmount());
      for (const release of h.cleanups) release();
      await until(() => h.running === 0, "Controlled inference did not stop during cleanup");
      for (let i = 0; i < 3; i++) await tick();
      assert.equal(h.disposed.length, h.opened.length, "Every opened media input must be disposed");
      assert.equal(h.rallyRuns, h.expectedRallyRuns, "Only explicitly requested full analysis may run rally inference");
      assert.deepEqual(unhandled, [], "Late callbacks must not leave an unhandled rejection");
    } finally {
      if (server) await server.close();
      dom.window.close(); process.off("unhandledRejection", onUnhandled);
      URL.createObjectURL = urls.create; URL.revokeObjectURL = urls.revoke;
      for (const [key, descriptor] of Object.entries(previous)) {
        if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
      }
    }
  }
}

for (const timing of ["before retry", "while queued"]) await test(`regression: cached legacy wrong bytes after fingerprint upgrade ${timing}`, { timeout: 20000 }, () => withApp(async a => {
  const { h, store, read, file, fixture, mount, attach, select, generate, gate, until, idle } = a;
  const original = file(), wrong = file(original.name, "wrong");
  const target = fixture(), blockerFile = file("synthetic-blocker.mp4"), blocker = fixture("synthetic-blocker", blockerFile);
  const held = gate(); let blockerEntered = false;
  const normal = h.behavior;
  h.behavior = async (request, options, sourceFile) => {
    if (sourceFile === blockerFile) { blockerEntered = true; await held.promise; }
    return normal(request, options, sourceFile);
  };
  await mount(timing === "while queued" ? [blocker, target] : [target]);
  if (timing === "while queued") {
    await attach(blockerFile); await generate(); await until(() => blockerEntered);
    await select(target.id);
  }
  await attach(wrong); // Legacy metadata permits this link, before any fingerprint exists.
  assert.equal(h.opened.includes(wrong), false, "Linking a ready source must not start rally/media work");
  const upgrade = async () => {
    const current = await read(target.id);
    await store.putProject({ ...current, source: { ...current.source, fingerprint: await store.sourceFileFingerprint(original) } },
      { expectedSource: current.source });
  };
  if (timing === "before retry") await upgrade();
  await generate();
  if (timing === "while queued") {
    await until(() => document.body.textContent.includes("Score markers queued"));
    await upgrade(); await held.release();
  }
  await idle();
  assert.equal(h.opened.filter(value => value === wrong).length, 0, "Fingerprint-upgraded recovery must reject cached wrong bytes BEFORE opening media");
  assert.equal(h.requests.some(request => request.file === wrong), false, "Wrong source must not reach specialists");
  assert.equal(h.rallyRuns, 0);
  assert.equal((await read(target.id)).analysis.servingSide, undefined, "Rejected retry must preserve incomplete saved evidence");
  await attach(original); await generate(); await idle();
  await until(async () => (await read(target.id)).analysis.servingSide);
  assert.equal(h.opened.filter(value => value === original).length, 1, "Original-source retry must still succeed");
  assert.equal((await read(target.id)).source.fingerprint, await store.sourceFileFingerprint(original));
  assert.equal(h.rallyRuns, 0);
}));

await test("control: fingerprinted original source recovers without rally inference", { timeout: 20000 }, () => withApp(async a => {
  const original = a.file(), project = a.fixture();
  project.source.fingerprint = await a.store.sourceFileFingerprint(original);
  await a.mount([project]); await a.attach(original); await a.generate(); await a.idle();
  await a.until(async () => (await a.read(project.id)).analysis.servingSide);
  assert.deepEqual(a.h.opened, [original]); assert.equal(a.h.rallyRuns, 0);
}));

await test("regression: rejected cached source leaves imported raw analysis and corrected review untouched", { timeout: 20000 }, () => withApp(async a => {
  const fixturePath = fileURLToPath(new URL("../../tests/fixtures/web-side-switch-project.ts", import.meta.url)).replaceAll("\\", "/");
  const { sideSwitchProjectFixture } = await a.server.ssrLoadModule(`/@fs/${fixturePath}`);
  const { importModelFeedbackProject } = await a.server.ssrLoadModule("/src/lib/model-feedback-import.ts");
  const { project } = importModelFeedbackProject(JSON.stringify(sideSwitchProjectFixture(false).bundle));
  const original = a.file(), wrong = a.file(original.name, "wrong");
  project.source = a.store.projectSource(original);
  project.analysis.servingSide = undefined;
  project.analysis.productionServeOutputs = Object.fromEntries(["allLabelsV2", "previousProduction"].map(key =>
    [key, { probabilities: new Float32Array(project.analysis.times.length), detections: [] }]));
  project.reviewDraft = structuredClone(project.importedFeedback.initialDraft);
  project.reviewDraft.updatedAt = new Date(Date.now() + 60000).toISOString();
  await a.mount([project]); await a.attach(wrong);
  const current = await a.read(project.id);
  await a.store.putProject({ ...current, source: { ...current.source, fingerprint: await a.store.sourceFileFingerprint(original) } }, { expectedSource: current.source });
  const expected = await a.read(project.id);
  await a.generate(); await a.idle();
  assert.deepEqual(a.h.opened, [], "Imported projects must reject stale cached files before decoding");
  assert.deepEqual(await a.read(project.id), expected, "Source rejection must preserve the full imported analysis, raw evidence and corrected review");
}));

for (const boundary of ["evidence", "output", "failure"]) await test(`regression: same-ID same-source replacement during held ${boundary}`, { timeout: 20000 }, () => withApp(async a => {
  const { h, store, read, file, fixture, draftFor, mount, attach, generate, gate, until, idle, result } = a;
  const sourceFile = file(), project = fixture(), evidence = structuredClone(project.analysis), held = gate();
  let entered = false;
  if (boundary === "evidence") {
    delete project.analysis.productionServeOutputs;
    project.analysis.featureNames = ["synthetic-feature"]; project.analysis.featureValues = new Float32Array([0]);
    h.evidenceBehavior = async () => { entered = true; await held.promise; return evidence; };
  } else {
    h.behavior = async (request, options) => {
      entered = true; await held.promise;
      if (boundary === "failure") throw new Error("Synthetic old-instance specialist failure");
      const output = result(request); await options.onOutput?.(output); return output;
    };
  }
  await mount([project]); await attach(sourceFile); await generate(); await until(() => entered);
  const old = await read(project.id), replacement = structuredClone(old);
  delete replacement.instanceId; // Explicit creation, using the real store's fresh-generation API.
  replacement.analysis.intervals[0].id = "replacement-owned-rally";
  replacement.analysis.intervals[0].end = 22;
  replacement.reviewDraft = draftFor(replacement);
  replacement.reviewDraft.cuts[0].included = false;
  replacement.reviewDraft.reviewedCutIds = ["replacement-owned-rally"];
  replacement.reviewDraft.updatedAt = new Date(Date.now() + 60000).toISOString();
  replacement.scoreSpecialistErrors = { sideSwitch: "Replacement-owned marker note" };
  await store.deleteProject(old.id, old.instanceId);
  await store.putProject(replacement, { expectedSource: null });
  const expected = await read(project.id);
  if (old.instanceId !== undefined) assert.notEqual(expected.instanceId, old.instanceId, "Explicit recreation must mint a fresh instance");
  assert.deepEqual(expected.source, old.source, "This race must change only the generation, not the source");
  await held.release(); await idle();
  const saved = await read(project.id);
  assert.deepEqual(saved.reviewDraft, expected.reviewDraft, `A late ${boundary} checkpoint must preserve the replacement review`);
  assert.deepEqual(saved.analysis.intervals, expected.analysis.intervals, `A late ${boundary} checkpoint must preserve replacement rallies`);
  assert.deepEqual(saved.scoreSpecialistErrors, expected.scoreSpecialistErrors, `A late ${boundary} failure must preserve replacement notes`);
  assert.deepEqual(saved, expected, `A late ${boundary} checkpoint must not change the replacement record`);
  assert.equal(h.rallyRuns, 0);
}));

async function recreate(a, id, analysis) {
  const old = await a.read(id), replacement = { ...old, analysis: structuredClone(analysis ?? old.analysis) };
  delete replacement.instanceId;
  replacement.analysis.intervals[0].id = "replacement-owned-rally";
  replacement.analysis.intervals[0].end = 22;
  replacement.reviewDraft = a.draftFor(replacement);
  replacement.reviewDraft.cuts[0].included = false;
  replacement.reviewDraft.reviewedCutIds = ["replacement-owned-rally"];
  replacement.reviewDraft.updatedAt = new Date(Date.now() + 60000).toISOString();
  await a.store.deleteProject(id, old.instanceId);
  const fresh = await a.store.putProject(replacement, { expectedSource: null });
  assert.notEqual(fresh.instanceId, old.instanceId);
  return a.read(id);
}

await test("regression: queued score work cannot adopt a recreated same-source project", { timeout: 20000 }, () => withApp(async a => {
  const firstFile = a.file("synthetic-queue-blocker.mp4"), targetFile = a.file();
  const first = a.fixture("synthetic-queue-blocker", firstFile), target = a.fixture(), held = a.gate();
  let entered = false;
  const normal = a.h.behavior;
  a.h.behavior = async (request, options, file) => {
    if (file === firstFile) { entered = true; await held.promise; }
    return normal(request, options, file);
  };
  await a.mount([first, target]); await a.attach(firstFile); await a.generate(); await a.until(() => entered);
  await a.select(target.id); await a.attach(targetFile); await a.generate();
  await a.until(() => document.body.textContent.includes("Score markers queued"));
  const expected = await recreate(a, target.id);
  await held.release(); await a.idle();
  assert.equal(a.h.opened.includes(targetFile), false, "An old queued instance must not open media for the replacement");
  assert.deepEqual(await a.read(target.id), expected);
  await a.attach(targetFile); await a.generate(); await a.idle();
  await a.until(async () => (await a.read(target.id)).analysis.servingSide);
  assert.equal((await a.read(target.id)).instanceId, expected.instanceId, "Fresh-instance retry remains usable");
}));

await test("regression: old queue cleanup preserves a newly queued same-ID lifetime", { timeout: 20000 }, () => withApp(async a => {
  const sourceFile = a.file(), project = a.fixture(), held = a.gate();
  project.rallyModel = "high-recall";
  project.id = a.store.projectId(project.source, project.info, project.analysisWindow, project.rallyModel,
    { roi: project.roi, sideSwitchEnabled: false });
  const analysis = structuredClone(project.analysis), normal = a.h.behavior;
  let entered = false;
  a.h.behavior = async (...args) => {
    if (!entered) { entered = true; await held.promise; }
    return normal(...args);
  };
  a.h.expectedRallyRuns = 1; a.h.rallyBehavior = async () => structuredClone(analysis);
  await a.mount([project]); await a.attach(sourceFile); await a.generate(); await a.until(() => entered);
  const old = await a.read(project.id);
  await a.click("Delete"); await a.until(async () => !(await a.read(project.id)));
  await a.chooseSource(sourceFile);
  await a.until(() => a.button("Find the rallies") && !a.button("Find the rallies").disabled);
  await a.click("Find the rallies");
  await a.until(async () => (await a.read(project.id))?.instanceId !== old.instanceId && Boolean(await a.read(project.id)));
  const fresh = await a.read(project.id);
  await held.release();
  await a.until(async () => (await a.read(project.id))?.analysis?.servingSide,
    "The old job's finally must not remove the new lifetime from the queue");
  assert.equal((await a.read(project.id)).instanceId, fresh.instanceId);
  await a.idle(); assert.equal(a.h.rallyRuns, 1);
}));

for (const boundary of ["rally", "score output"]) await test(`regression: full analysis cannot write to a recreated project after held ${boundary}`, { timeout: 20000 }, () => withApp(async a => {
  const sourceFile = a.file(), project = a.fixture(), analysis = structuredClone(project.analysis), held = a.gate();
  project.status = "waiting"; project.analysis = null; a.h.expectedRallyRuns = 1;
  let entered = false;
  a.h.rallyBehavior = async () => {
    if (boundary === "rally") { entered = true; await held.promise; }
    return structuredClone(analysis);
  };
  if (boundary === "score output") {
    const normal = a.h.behavior;
    a.h.behavior = async (...args) => { entered = true; await held.promise; return normal(...args); };
  }
  await a.mount([project]); await a.chooseSource(sourceFile); await a.until(() => entered);
  const expected = await recreate(a, project.id, analysis);
  await held.release(); await a.until(() => a.h.running === 0);
  // Wait for the full job's finally/reconciliation, not only the held boundary.
  await a.until(() => document.body.textContent.includes("instance changed"));
  assert.deepEqual(await a.read(project.id), expected, "Full rally/score checkpoint, failure and ready finalization must retain the original instance fence");
}));

await test("regression: suppression augmentation rejects a same-instance source upgrade", { timeout: 20000 }, () => withApp(async a => {
  const project = a.fixture(), held = a.gate(), refreshed = a.gate();
  project.analysis = recoveryFixture({ modelId: a.PRODUCTION_ENSEMBLE_MODEL_ID }).analysis;
  const evidence = structuredClone(project.analysis);
  evidence.suppression = a.emptySuppression();
  delete project.analysis.productionServeOutputs;
  project.analysis.featureNames = ["synthetic-feature"]; project.analysis.featureValues = new Float32Array([0]);
  let entered = 0;
  a.h.augmentBehavior = async () => { const turn = ++entered; await (turn === 1 ? held : refreshed).promise; return evidence; };
  await a.mount([project]); await a.until(() => entered === 1);
  const current = await a.read(project.id);
  await a.store.putProject({ ...current, source: { ...current.source, fingerprint: await a.store.sourceFileFingerprint(a.file()) } }, { expectedSource: current.source });
  const expected = await a.read(project.id);
  await held.release(); await a.until(() => /identity changed/.test(document.body.textContent));
  assert.deepEqual(await a.read(project.id), expected, "An augmentation cannot adopt the upgraded source on completion");
  assert.deepEqual(a.h.opened, []);
}));

await test("regression: old augmentation completion cannot erase a replacement's pending evidence", { timeout: 20000 }, () => withApp(async a => {
  const sourceFile = a.file(), project = a.fixture(), oldGate = a.gate(), newGate = a.gate();
  project.analysis = recoveryFixture({ modelId: a.PRODUCTION_ENSEMBLE_MODEL_ID }).analysis;
  delete project.analysis.productionServeOutputs;
  project.analysis.featureNames = ["synthetic-feature"]; project.analysis.featureValues = new Float32Array([0]);
  let entered = 0;
  a.h.augmentBehavior = async analysis => {
    const turn = ++entered;
    await (turn === 1 ? oldGate : newGate).promise;
    const evidence = structuredClone(analysis);
    evidence.suppression = a.emptySuppression();
    evidence.productionServeOutputs = recoveryFixture({ modelId: a.PRODUCTION_ENSEMBLE_MODEL_ID }).analysis.productionServeOutputs;
    return evidence;
  };
  a.h.evidenceBehavior = () => { throw new Error("The replacement's shared augmentation should supply score evidence"); };
  await a.mount([project]); await a.until(() => entered === 1);
  const replacement = await recreate(a, project.id);
  await oldGate.release(); await a.until(() => entered === 2);
  assert.deepEqual(await a.read(project.id), replacement, "Old completion must not persist into the replacement");
  await a.attach(sourceFile); await a.generate();
  await a.until(() => document.querySelector('[aria-label="Score prediction recovery"][aria-busy="true"]'));
  await newGate.release(); await a.idle();
  await a.until(async () => (await a.read(project.id)).analysis.servingSide);
  const saved = await a.read(project.id);
  assert.equal(saved.instanceId, replacement.instanceId);
  assert.deepEqual(saved.analysis.intervals, replacement.analysis.intervals);
  assert.deepEqual(saved.reviewDraft, replacement.reviewDraft);
  assert.equal(a.h.evidenceRuns, 0, "Old Map cleanup must not evict the new instance's shared augmentation");
  assert.equal(entered, 2);
}));

await test("control: active and queued same-instance jobs retain concurrent review/export metadata", { timeout: 20000 }, () => withApp(async a => {
  const { h, store, read, file, fixture, draftFor, mount, attach, select, generate, gate, until, idle, result } = a;
  const firstFile = file(), secondFile = file("synthetic-queued.mp4");
  const first = fixture(), second = fixture("synthetic-queued-project", secondFile);
  first.sideSwitchEnabled = true;
  const held = gate(); let partial = false;
  h.behavior = async (request, options, sourceFile) => {
    const output = result(request);
    await options.onOutput?.({ servingSide: output.servingSide });
    if (sourceFile === firstFile) { partial = true; await held.promise; }
    if (output.sideSwitch) await options.onOutput?.({ sideSwitch: output.sideSwitch });
    return output;
  };
  await mount([first, second]); await attach(firstFile); await generate(); await until(() => partial);
  await until(async () => (await read(first.id)).analysis.servingSide);
  await select(second.id); await attach(secondFile); await generate();
  await until(() => document.body.textContent.includes("Score markers queued"));
  const expected = new Map();
  for (const project of [first, second]) {
    const current = await read(project.id), draft = draftFor(current);
    const stamp = new Date(Date.now() + 60000).toISOString();
    draft.updatedAt = stamp; draft.cuts[0].included = false; draft.reviewedCutIds = [draft.cuts[0].id];
    draft.beforePaddingSeconds = 2; draft.scoreTracking.team1Name = `Concurrent ${project.id}`;
    await store.putProjectReviewDraft(project.id, draft, current.instanceId);
    await store.markProjectExported(project.id, stamp, current.instanceId);
    expected.set(project.id, { draft, stamp, instanceId: current.instanceId });
  }
  await held.release(); await idle();
  await until(async () => (await read(second.id)).analysis.servingSide && (await read(first.id)).analysis.sideSwitch);
  for (const project of [first, second]) {
    const saved = await read(project.id), wanted = expected.get(project.id);
    assert.deepEqual(saved.reviewDraft, wanted.draft, `${project.id}: score saves must preserve the latest full review`);
    assert.equal(saved.lastExportedAt, wanted.stamp); assert.equal(saved.instanceId, wanted.instanceId);
    assert.equal(saved.status, "ready");
  }
  assert.equal(h.rallyRuns, 0); assert.deepEqual(h.opened, [firstFile, secondFile]);
}));

// Hold a real fake-IDB readwrite transaction; do not mock putProject or the queue.
// Keeping requests pending delays subsequent checkpoints until release().
async function holdProjectWrites(h) {
  const [{ name }] = await indexedDB.databases();
  const database = await new Promise((resolve, reject) => {
    const request = indexedDB.open(name); request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
  });
  const transaction = database.transaction([...database.objectStoreNames], "readwrite");
  let released = false;
  const release = () => { released = true; }; h.cleanups.push(release);
  const complete = new Promise((resolve, reject) => {
    transaction.oncomplete = () => { database.close(); resolve(); };
    transaction.onerror = transaction.onabort = () => { database.close(); reject(transaction.error); };
  });
  const pump = () => {
    const request = transaction.objectStore(database.objectStoreNames[0]).get("synthetic-lock-probe");
    request.onsuccess = () => { if (!released) pump(); };
  };
  pump();
  return { release: async () => { release(); await complete; } };
}

await test("regression: next queued recovery waits for prior score checkpoint persistence", { timeout: 20000 }, () => withApp(async a => {
  const firstFile = a.file(), secondFile = a.file("synthetic-after-checkpoint.mp4");
  const first = a.fixture(), second = a.fixture("synthetic-after-checkpoint", secondFile), held = a.gate();
  let entered = false, outputAttempted = false;
  const normal = a.h.behavior;
  a.h.behavior = async (request, options, sourceFile) => {
    if (sourceFile === firstFile) { entered = true; await held.promise; outputAttempted = true; }
    return normal(request, options, sourceFile);
  };
  await a.mount([first, second]); await a.attach(firstFile); await a.generate(); await a.until(() => entered);
  await a.select(second.id); await a.attach(secondFile); await a.generate();
  await a.until(() => document.body.textContent.includes("Score markers queued"));
  const lock = await holdProjectWrites(a.h);
  try {
    await held.release(); await a.until(() => outputAttempted);
    // Let App effects and ready microtasks run while storage is still blocked.
    for (let i = 0; i < 20; i++) await a.tick();
    assert.equal(a.h.opened.includes(secondFile), false, "Queue must not start the next source while the prior score checkpoint is uncommitted");
  } finally { await lock.release(); }
  await a.idle();
  await a.until(async () => (await a.read(first.id)).analysis.servingSide && (await a.read(second.id)).analysis.servingSide);
  assert.deepEqual(a.h.opened, [firstFile, secondFile]); assert.equal(a.h.rallyRuns, 0);
}));

for (const preference of [false, true, undefined]) await test(`control: switch preference ${String(preference)} survives score recovery and model-draft refresh`, { timeout: 20000 }, () => withApp(async a => {
  const sourceFile = a.file(), project = a.fixture();
  if (preference === undefined) delete project.sideSwitchEnabled; else project.sideSwitchEnabled = preference;
  const cachedSwitch = a.result({ sideSwitch: { analysis: project.analysis } }).sideSwitch;
  if (preference === false) project.analysis.sideSwitch = cachedSwitch;
  project.reviewDraft = a.draftFor(project);
  project.reviewDraft.scoreTracking = a.addSideSwitchMarker(project.reviewDraft.scoreTracking, 45, { id: "synthetic-manual-switch", origin: "manual" });
  await a.mount([project]); await a.attach(sourceFile); await a.generate(); await a.idle();
  await a.until(async () => (await a.read(project.id)).analysis.servingSide);
  assert.deepEqual(a.h.requests[0].stages, preference === false ? ["servingSide"] : ["servingSide", "sideSwitch"]);
  await a.until(async () => {
    const markers = (await a.read(project.id)).reviewDraft?.scoreTracking.sideSwitchMarkers ?? [];
    return markers.some(marker => marker.id === "synthetic-manual-switch") && markers.length === (preference === false ? 1 : 2);
  }, "Recovery must retain manual markers and respect automatic-switch preference");
  assert.deepEqual((await a.read(project.id)).analysis.sideSwitch, cachedSwitch, "Raw switch evidence remains available even when detection is disabled");
  await a.click("App menu");
  const reset = document.querySelector('.rd-menu-panel button:nth-of-type(3)'); assert.ok(reset);
  await act(async () => reset.click());
  const confirm = document.querySelector('.rd-reset-project-dialog footer button:last-child'); assert.ok(confirm);
  await act(async () => confirm.click());
  await a.until(async () => {
    const markers = (await a.read(project.id)).reviewDraft?.scoreTracking.sideSwitchMarkers ?? [];
    return JSON.stringify(markers.map(marker => marker.id)) === JSON.stringify(preference === false ? [] : ["switch-synthetic-auto-switch"]);
  }, "Reset must use the refreshed model draft with the saved switch preference");
  assert.equal(a.h.rallyRuns, 0);
}));
