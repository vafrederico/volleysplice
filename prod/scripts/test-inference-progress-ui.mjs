import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { JSDOM } from "jsdom";
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { createServer } from "vite";

// Load the real component without listening on a port or serving the app.
const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  server: { middlewareMode: true, hmr: false }, appType: "custom",
});
const dom = new JSDOM('<div id="root"></div>');
Object.assign(globalThis, { window: dom.window, document: dom.window.document, IS_REACT_ACT_ENVIRONMENT: true });
const root = createRoot(document.getElementById("root"));
try {
  const { InferenceProgressPanel } = await server.ssrLoadModule("/src/components/InferenceProgressPanel.tsx");
  const { createInferenceProgressSteps, updatePipelineInferenceSteps, CORE_INFERENCE_STEP_IDS } =
    await server.ssrLoadModule("/src/lib/inference-progress.ts");
  let steps = createInferenceProgressSteps(CORE_INFERENCE_STEP_IDS);
  const update = async (stage, completed, total, detail) => {
    steps = updatePipelineInferenceSteps(steps, { stage, completed, total, detail }, performance.now());
    await act(async () => root.render(React.createElement(InferenceProgressPanel, { steps })));
  };
  await update("video", 0, 100, "Loading the selected image model before reading video frames");
  assert.match(document.body.textContent, /Loading the selected image model/);
  await update("video", 25, 100, "Reading game images - 50/100 - GPU");
  assert.match(document.body.textContent, /Reading game images - 50\/100/);
  assert.equal(document.querySelector('[aria-label="Checking the video progress"]').getAttribute("aria-valuenow"), "25");
  await update("video", 50, 100, "Reusing saved image features for the selected model");
  assert.match(document.body.textContent, /Reusing saved image features/);
  await update("inference", 128, 512, "Finding rallies - 25%");
  const bar = () => document.querySelector('[aria-label="Finding rallies progress"]');
  const first = Number(bar().getAttribute("aria-valuenow"));
  await update("inference", 384, 512, "Finding rallies - 75%");
  assert.ok(Number(bar().getAttribute("aria-valuenow")) > first);
  assert.match(document.body.textContent, /Finding rallies - 75%/);
  await update("inference", 512, 512, "Preparing serve and court-state evidence for score tracking");
  assert.equal(bar().getAttribute("aria-valuenow"), "95");
  assert.match(document.body.textContent, /Preparing serve and court-state evidence/);
  await update("complete", 100, 100, "Ready");
  assert.equal(bar().getAttribute("aria-valuenow"), "100");
  assert.match(document.body.textContent, /Your video is ready/);
  console.log("Inference progress UI: loading, image counts, cache reuse, temporal chunks and completion passed.");
} finally {
  await act(async () => root.unmount());
  await server.close();
  dom.window.close();
}
