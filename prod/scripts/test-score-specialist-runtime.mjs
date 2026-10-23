import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import { JSDOM } from "jsdom";
import { recoveryFixture } from "./score-recovery-fixtures.mjs";

// Real coordinator, planners, evaluators, validators and model loaders. Empty
// synthetic candidates avoid decoding frames or running OpenCV/GPU operations.
// Only asset transport and OpenCV's browser script events are simulated.
async function withRuntime(test, { cvReady = true } = {}) {
  const server = await createServer({
    root: fileURLToPath(new URL("..", import.meta.url)), configFile: false,
    optimizeDeps: { noDiscovery: true, entries: [] },
    server: { middlewareMode: true, hmr: false, ws: false }, appType: "custom",
  });
  const dom = new JSDOM(cvReady ? '<script data-volleycut-opencv="true"></script>' : "", { url: "https://example.test/" });
  const previous = { fetch: globalThis.fetch, window: globalThis.window, document: globalThis.document };
  globalThis.window = dom.window; globalThis.document = dom.window.document;
  if (cvReady) window.cv = {};
  const calls = [];
  let failure = null;
  const unhandled = [], onUnhandled = (cause) => unhandled.push(cause);
  process.on("unhandledRejection", onUnhandled);
  globalThis.fetch = async (url) => {
    const name = new URL(url).pathname.split("/").at(-1);
    calls.push(name);
    if (failure?.asset === name) {
      if (failure.kind === "stalled") return failure.pending;
      if (failure.kind === "network") throw new TypeError("Synthetic network unavailable");
      if (failure.kind === "http") return new Response("Unavailable", { status: 503 });
      if (failure.kind === "json") return new Response("invalid JSON");
      if (failure.kind === "contract") return Response.json({ schemaVersion: -1 });
    }
    return new Response(await readFile(new URL(`../public/runtime/${name}`, import.meta.url)));
  };
  try {
    const { inferScoreSpecialists } = await server.ssrLoadModule("/src/lib/on-device/score-specialists.ts");
    const serving = await server.ssrLoadModule("/src/lib/on-device/serving-side.ts");
    const servingModel = await server.ssrLoadModule("/src/lib/on-device/serving-side-model.ts");
    const switchModel = await server.ssrLoadModule("/src/lib/on-device/side-switch-model.ts");
    const { scoreSpecialistReadiness, missingScoreStages } = await server.ssrLoadModule("/src/lib/score-specialist-state.ts");
    const { PRODUCTION_ENSEMBLE_MODEL_ID } = await server.ssrLoadModule("/src/lib/on-device/ensemble.ts");
    const project = recoveryFixture({ empty: true, switches: true, modelId: PRODUCTION_ENSEMBLE_MODEL_ID });
    const request = { servingSide: { analysis: project.analysis }, sideSwitch: { analysis: project.analysis } };
    const saved = [];
    const capture = (output) => {
      saved.push(...Object.keys(output));
      project.analysis = { ...project.analysis, ...output };
    };
    const run = (stages = request, options = {}) => inferScoreSpecialists({ info: project.info }, project.roi, stages, { onOutput: capture, ...options });
    await test({ run, request, project, saved, capture, serving, servingModel, switchModel,
      scoreSpecialistReadiness, missingScoreStages,
      fail: (asset, kind) => { failure = { asset, kind }; }, restore: () => { failure = null; },
      stall: (asset) => {
        let resolve, reject;
        const pending = new Promise((yes, no) => { resolve = yes; reject = no; });
        failure = { asset, kind: "stalled", pending };
        return { resolve, reject };
      },
      count: (asset) => calls.filter((name) => name === asset).length });
    await new Promise((resolve) => setImmediate(resolve));
    assert.deepEqual(unhandled, [], "An abandoned load's late rejection must remain handled");
  } finally {
    process.off("unhandledRejection", onUnhandled);
    Object.assign(globalThis, previous);
    dom.window.close(); await server.close();
  }
}

async function promptly(pending) {
  let timeout;
  try {
    return await Promise.race([pending, new Promise((_, reject) => {
      timeout = setTimeout(() => reject(new Error("Cancellation waited for a stalled model transport")), 1000);
    })]);
  } finally { clearTimeout(timeout); }
}

async function waitFor(condition) {
  for (let i = 0; i < 100; i++) {
    if (condition()) return;
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  assert.fail("The requested model load did not start");
}

// A rejected request, HTTP response, JSON decoder or contract parser must not
// poison a tab's promise cache. Concurrent callers still share each attempt.
for (const kind of ["network", "http", "json", "contract"]) {
  await withRuntime(async ({ serving, servingModel, fail, restore, count }) => {
    const asset = servingModel.SERVING_SIDE_ASSET;
    fail(asset, kind);
    const first = serving.loadServingSideRuntime();
    assert.strictEqual(serving.loadServingSideRuntime(), first);
    const results = await Promise.allSettled([first, serving.loadServingSideRuntime()]);
    assert.ok(results.every((result) => result.status === "rejected"), kind);
    assert.equal(count(asset), 1);
    restore();
    const retry = serving.loadServingSideRuntime();
    assert.notStrictEqual(retry, first, `${kind}: a retry must create a fresh request`);
    assert.strictEqual(serving.loadServingSideRuntime(), retry);
    assert.equal((await retry).modelId, servingModel.SERVING_SIDE_MODEL_ID);
    assert.strictEqual(serving.loadServingSideRuntime(), retry, "Successful runtimes remain cached");
    assert.equal(count(asset), 2);
  });
}

// Reproduce the original failure through the real coordinator, then recover in
// that same module instance without reloading the tab or clearing test modules.
await withRuntime(async ({ run, request, servingModel, fail, restore, saved, count, project, scoreSpecialistReadiness }) => {
  const asset = servingModel.SERVING_SIDE_ASSET;
  fail(asset, "network");
  await assert.rejects(run({ servingSide: request.servingSide }), /Synthetic network/);
  assert.deepEqual(saved, []);
  restore();
  await run({ servingSide: request.servingSide });
  assert.deepEqual(saved, ["servingSide"]);
  assert.equal(scoreSpecialistReadiness(project).servingSide.status, "complete");
  assert.equal(count(asset), 2, "Retry must fetch again after the network returns");
});

await withRuntime(async ({ run, request, project, saved, switchModel, servingModel, fail, restore, count, missingScoreStages }) => {
  fail(switchModel.SIDE_SWITCH_ASSET, "http");
  await assert.rejects(run(), /side-switch model \(503\)/);
  assert.deepEqual(saved, ["servingSide"], "A failed switch plan cannot block the healthy serving checkpoint");
  assert.deepEqual(missingScoreStages(project), ["sideSwitch"]);
  restore();
  await run({ sideSwitch: request.sideSwitch });
  assert.deepEqual(missingScoreStages(project), []);
  assert.equal(count(switchModel.SIDE_SWITCH_ASSET), 2, "The switch loader also supports failure then retry");
  assert.equal(count(servingModel.SERVING_SIDE_ASSET), 1, "The completed serving stage is not regenerated");
});

// Serving planning is synchronous: duplicate candidate identity throws before
// returning a promise. The converse stage must still plan, run and checkpoint.
await withRuntime(async ({ run, request, saved, project, servingModel, switchModel, count, missingScoreStages }) => {
  const interval = recoveryFixture().analysis.intervals[0];
  const invalidServing = { analysis: { ...project.analysis, intervals: [interval, interval] } };
  await assert.rejects(run({ ...request, servingSide: invalidServing }), /stable unique IDs/);
  assert.deepEqual(saved, ["sideSwitch"]);
  assert.deepEqual(missingScoreStages(project), ["servingSide"]);
  assert.equal(count(servingModel.SERVING_SIDE_ASSET), 0, "A failed serving plan must not evaluate");
  await run({ servingSide: request.servingSide });
  assert.deepEqual(missingScoreStages(project), []);
  assert.equal(count(switchModel.SIDE_SWITCH_ASSET), 1);
});

// OpenCV is a shared direct specialist dependency. A failed browser script must
// be removed as well as evicted from the promise cache before another attempt.
for (const event of ["error", "load"]) await withRuntime(async ({ run, request, saved, servingModel, count }) => {
  const first = run({ servingSide: request.servingSide });
  const rejected = assert.rejects(first, /OpenCV.*(?:browser asset|browser runtime)/);
  const waitForScript = async () => {
    for (let i = 0; i < 100; i++) {
      const script = document.querySelector('script[data-volleycut-opencv="true"]');
      if (script) return script;
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
    assert.fail("OpenCV script was not requested");
  };
  const failed = await waitForScript();
  failed.dispatchEvent(new window.Event(event));
  await rejected;
  assert.equal(failed.isConnected, false);
  assert.deepEqual(saved, []);
  const retry = run({ servingSide: request.servingSide });
  const replacement = await waitForScript();
  assert.notStrictEqual(replacement, failed);
  window.cv = {};
  replacement.dispatchEvent(new window.Event("load"));
  await retry;
  assert.deepEqual(saved, ["servingSide"]);
  assert.equal(count(servingModel.SERVING_SIDE_ASSET), 1, "A healthy model cache survives an OpenCV retry");
}, { cvReady: false });

// Planning (switches) and evaluation (serves) both await shared model promises.
// Cancelling one consumer must settle before transport and leave another caller
// sharing that exact pending runtime able to finish or observe its own failure.
for (const stage of ["servingSide", "sideSwitch"]) for (const completion of ["never", "resolve", "reject"]) {
  await withRuntime(async ({ run, request, serving, servingModel, switchModel, stall, restore, count }) => {
    const asset = stage === "servingSide" ? servingModel.SERVING_SIDE_ASSET : switchModel.SIDE_SWITCH_ASSET;
    const load = stage === "servingSide" ? serving.loadServingSideRuntime : switchModel.loadSideSwitchRuntime;
    const transport = stall(asset), abort = new AbortController(), saved = [];
    const first = run({ [stage]: request[stage] }, { signal: abort.signal, onOutput: (output) => saved.push(output) });
    const cancelled = assert.rejects(promptly(first), { name: "AbortError" });
    await waitFor(() => count(asset) === 1);
    const shared = load(); abort.abort(); await cancelled;
    assert.strictEqual(load(), shared, "A consumer abort cannot invalidate or abort the shared loader cache");
    assert.deepEqual(saved, []);
    const otherStage = stage === "servingSide" ? "sideSwitch" : "servingSide";
    await promptly(run({ [otherStage]: request[otherStage] }));
    assert.equal(count(asset), 1, "Unrelated healthy work can finish while the first fetch never resolves");
    if (completion === "never") return;

    const secondSaved = [];
    const second = run({ [stage]: request[stage] }, { onOutput: (output) => secondSaved.push(output) });
    const finished = completion === "reject" ? assert.rejects(second, /Synthetic late model rejection/) : second;
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(count(asset), 1); restore();
    if (completion === "resolve") transport.resolve(new Response(await readFile(new URL(`../public/runtime/${asset}`, import.meta.url))));
    else transport.reject(new TypeError("Synthetic late model rejection"));
    await promptly(finished);
    assert.deepEqual(saved, [], "Late load completion cannot checkpoint for the cancelled caller");
    assert.equal(secondSaved.length, completion === "resolve" ? 1 : 0);
    await promptly(run({ [stage]: request[stage] }));
    assert.equal(count(asset), completion === "resolve" ? 1 : 2, "Successful shared loads stay cached; rejected loads can retry");
  });
}

console.log("Specialist runtime: loader retry, concurrent cache reuse, independent planning failures, OpenCV recovery, prompt planning/evaluation cancellation, never/late resolve/late reject and shared-consumer isolation passed.");
