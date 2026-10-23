import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

// Exercise the real shared coordinator and partial-output/cancellation contract,
// substituting only frame sampling and the individual specialist evaluations.
const harness = { samples: 0, stages: [], fail: null, planFail: [], planGate: null, schedules: [] };
globalThis.__specialistCoordinatorHarness = harness;
const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)), configFile: false, optimizeDeps: { noDiscovery: true, entries: [] },
  server: { middlewareMode: true, hmr: false, ws: false }, appType: "custom",
  plugins: [{ name: "synthetic-specialist-evaluation", transform(_code, id) {
    if (id.endsWith("/on-device/specialist-frame-sampling.ts")) return `
      export async function sampleSpecialistFramesSequentially(media, roi, plan, progress, signal) {
        signal?.throwIfAborted(); const h = globalThis.__specialistCoordinatorHarness;
        h.samples++; h.schedules.push(plan);
        progress?.(0, 0); return { servingSide: new Map(), sideSwitch: new Map() };
      }`;
    if (id.endsWith("/on-device/serving-side.ts") || id.endsWith("/on-device/side-switch.ts")) {
      const serve = id.endsWith("/serving-side.ts"), stage = serve ? "servingSide" : "sideSwitch";
      return `export ${serve ? "" : "async "}function ${serve ? "servingSideFramePlan" : "sideSwitchFramePlan"}() {
          const h = globalThis.__specialistCoordinatorHarness;
          ${serve ? "" : "await h.planGate;"}
          if (h.planFail.includes("${stage}")) throw new Error("Synthetic ${stage} planning failure");
          return { requestedTimes: [${serve ? 1 : 2}] };
        }
        export async function ${serve ? "evaluateServingSideFrames" : "evaluateSideSwitchFrames"}() {
          const h = globalThis.__specialistCoordinatorHarness; h.stages.push("${stage}");
          if (h.fail === "${stage}") throw new Error("Synthetic ${stage} failure");
          return { output: { candidates: [] } };
        }`;
    }
  } }],
});
try {
  const { inferScoreSpecialists } = await server.ssrLoadModule("/src/lib/on-device/score-specialists.ts");
  const media = { info: { duration: 60 } }, roi = { x: 0, y: 0, width: 1, height: 1 };
  const request = { servingSide: { analysis: {} }, sideSwitch: { analysis: {} } };
  const saved = [];
  const capture = (output) => saved.push(...Object.keys(output));
  await inferScoreSpecialists(media, roi, request, { onOutput: capture });
  assert.equal(harness.samples, 1);
  assert.deepEqual(saved, ["servingSide", "sideSwitch"], "Genuine empty outputs are still checkpointed");
  // The next specialist must wait for the previous stage's durable checkpoint.
  let releaseCheckpoint, checkpointStarted;
  const checkpointGate = new Promise(resolve => { releaseCheckpoint = resolve; });
  const checkpointEntry = new Promise(resolve => { checkpointStarted = resolve; });
  harness.stages = [];
  const ordered = inferScoreSpecialists(media, roi, request, { onOutput: async output => {
    if (output.servingSide) { checkpointStarted(); await checkpointGate; }
  } });
  try {
    await checkpointEntry;
    assert.deepEqual(harness.stages, ["servingSide"], "A pending checkpoint must fence the next stage");
  } finally { releaseCheckpoint(); }
  await ordered;
  assert.deepEqual(harness.stages, ["servingSide", "sideSwitch"]);
  const conflict = new Error("Synthetic checkpoint conflict");
  harness.stages = [];
  await assert.rejects(inferScoreSpecialists(media, roi, request, { onOutput: async () => { throw conflict; } }),
    cause => cause === conflict, "Checkpoint rejection must retain its identity for App conflict handling");
  assert.deepEqual(harness.stages, ["servingSide"], "A rejected checkpoint must stop the next stage");
  for (const fail of ["servingSide", "sideSwitch"]) {
    saved.length = 0; harness.fail = fail; harness.stages = [];
    await assert.rejects(inferScoreSpecialists(media, roi, request, { onOutput: capture }), /Synthetic/);
    assert.deepEqual(harness.stages, ["servingSide", "sideSwitch"], "A failed specialist must not prevent the other stage");
    assert.deepEqual(saved, [fail === "servingSide" ? "sideSwitch" : "servingSide"]);
  }
  harness.fail = null; harness.stages = [];
  for (const failed of ["servingSide", "sideSwitch"]) {
    harness.planFail = [failed]; harness.stages = []; saved.length = 0;
    const progress = { servingSide: [], sideSwitch: [] };
    const trackedRequest = Object.fromEntries(Object.entries(request).map(([key, value]) => [key,
      { ...value, onProgress: (value) => progress[key].push(value.stage) }]));
    await assert.rejects(inferScoreSpecialists(media, roi, trackedRequest, { onOutput: capture }), /planning failure/);
    const healthy = failed === "servingSide" ? "sideSwitch" : "servingSide";
    assert.deepEqual(saved, [healthy]);
    assert.deepEqual(harness.stages, [healthy], "A failed planner must never evaluate or checkpoint");
    assert.equal(progress[failed].includes("frames"), false, "Failed plans must not receive healthy frame progress");
    const schedule = harness.schedules.at(-1);
    assert.equal(schedule[failed === "servingSide" ? "servingSideTimes" : "sideSwitchTimes"], undefined);
    assert.deepEqual(schedule[healthy === "servingSide" ? "servingSideTimes" : "sideSwitchTimes"], [healthy === "servingSide" ? 1 : 2]);
  }
  harness.planFail = ["servingSide", "sideSwitch"]; harness.stages = []; saved.length = 0;
  const beforePlanningFailure = harness.samples;
  await assert.rejects(inferScoreSpecialists(media, roi, request, { onOutput: capture }), (error) => {
    assert.match(error.message, /servingSide planning failure/);
    assert.match(error.message, /sideSwitch planning failure/);
    return true;
  });
  assert.equal(harness.samples, beforePlanningFailure, "No healthy plans means no shared decode");
  assert.deepEqual(saved, []); assert.deepEqual(harness.stages, []);

  // Cancellation wins over a late planning rejection and prevents every output.
  let releasePlan;
  harness.planGate = new Promise((resolve) => { releasePlan = resolve; });
  harness.planFail = ["sideSwitch"];
  const duringPlanning = new AbortController();
  const pending = inferScoreSpecialists(media, roi, request, { signal: duringPlanning.signal, onOutput: capture });
  const cancelledPlanning = assert.rejects(pending, { name: "AbortError" });
  await new Promise((resolve) => setImmediate(resolve));
  duringPlanning.abort(); releasePlan(); await cancelledPlanning;
  assert.equal(harness.samples, beforePlanningFailure);
  assert.deepEqual(saved, []); assert.deepEqual(harness.stages, []);
  harness.planGate = null; harness.planFail = [];
  await inferScoreSpecialists(media, roi, request, { onOutput: capture });
  assert.deepEqual(saved, ["servingSide", "sideSwitch"], "A cancelled plan leaves a subsequent retry usable");

  harness.stages = [];
  const abort = new AbortController();
  await assert.rejects(inferScoreSpecialists(media, roi, request, { signal: abort.signal, onOutput: () => abort.abort() }), { name: "AbortError" });
  assert.deepEqual(harness.stages, ["servingSide"], "Cancellation must prevent the next specialist");
  const samples = harness.samples;
  await assert.rejects(inferScoreSpecialists(media, roi, request, { signal: abort.signal }), { name: "AbortError" });
  assert.equal(harness.samples, samples, "Already cancelled work must not decode frames");
  console.log("Score coordinator: shared decode, empty outputs, independent planning/evaluation failures, healthy schedules/checkpoints, both-plan failure, planning cancellation/retry and AbortSignal passed.");
} finally {
  await server.close(); delete globalThis.__specialistCoordinatorHarness;
}
