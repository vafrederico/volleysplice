import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

// Exercise the real shared coordinator and partial-output/cancellation contract,
// substituting only frame sampling and the individual specialist evaluations.
const harness = { samples: 0, stages: [], fail: null };
globalThis.__specialistCoordinatorHarness = harness;
const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)), configFile: false, optimizeDeps: { noDiscovery: true, entries: [] },
  server: { middlewareMode: true, hmr: false, ws: false }, appType: "custom",
  plugins: [{ name: "synthetic-specialist-evaluation", transform(_code, id) {
    if (id.endsWith("/on-device/specialist-frame-sampling.ts")) return `
      export async function sampleSpecialistFramesSequentially(media, roi, plan, progress, signal) {
        signal?.throwIfAborted(); globalThis.__specialistCoordinatorHarness.samples++;
        progress?.(0, 0); return { servingSide: new Map(), sideSwitch: new Map() };
      }`;
    if (id.endsWith("/on-device/serving-side.ts") || id.endsWith("/on-device/side-switch.ts")) {
      const serve = id.endsWith("/serving-side.ts"), stage = serve ? "servingSide" : "sideSwitch";
      return `export function ${serve ? "servingSideFramePlan" : "sideSwitchFramePlan"}() { return { requestedTimes: [] }; }
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
  for (const fail of ["servingSide", "sideSwitch"]) {
    saved.length = 0; harness.fail = fail; harness.stages = [];
    await assert.rejects(inferScoreSpecialists(media, roi, request, { onOutput: capture }), /Synthetic/);
    assert.deepEqual(harness.stages, ["servingSide", "sideSwitch"], "A failed specialist must not prevent the other stage");
    assert.deepEqual(saved, [fail === "servingSide" ? "sideSwitch" : "servingSide"]);
  }
  harness.fail = null; harness.stages = [];
  const abort = new AbortController();
  await assert.rejects(inferScoreSpecialists(media, roi, request, { signal: abort.signal, onOutput: () => abort.abort() }), { name: "AbortError" });
  assert.deepEqual(harness.stages, ["servingSide"], "Cancellation must prevent the next specialist");
  const samples = harness.samples;
  await assert.rejects(inferScoreSpecialists(media, roi, request, { signal: abort.signal }), { name: "AbortError" });
  assert.equal(harness.samples, samples, "Already cancelled work must not decode frames");
  console.log("Score coordinator: shared decode, empty outputs, independent partial completion and AbortSignal passed.");
} finally {
  await server.close(); delete globalThis.__specialistCoordinatorHarness;
}
