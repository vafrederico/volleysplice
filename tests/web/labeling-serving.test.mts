import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { loadBindings, transform } from "next/dist/build/swc/index.js";
import { parseServingPredictions, servingPredictionsToMarkers } from "../../lib/labeling-serving.ts";

function fixture() {
  const head = { peakProbability: .9, peakTime: 10, threshold: .85, crossesThreshold: true };
  return { candidates: [
    { id: "R1", anchor: 10, side: "near", verdict: "near", nearProbability: .8, serveDecisionSource: "serve-head", reviewReasons: [], serveEvidence: { allLabelsV2: head } },
    { id: "R2", anchor: 20, side: "far", verdict: "far", nearProbability: .1, serveDecisionSource: "serve-head", reviewReasons: [] },
    { id: "R3", anchor: 30, side: "far", verdict: "review", nearProbability: .4, serveDecisionSource: "production-rally-recovery", reviewReasons: ["side-score", "production-rally-recovery"] },
    { id: "R4", anchor: 40, side: "near", verdict: "not-serve", nearProbability: .99, serveDecisionSource: "none", reviewReasons: [] },
  ] };
}

test("serve gate rejections never become serves; review retains guessed side and its probability", () => {
  const input = fixture();
  const unchanged = JSON.stringify(input);
  const rows = parseServingPredictions(input, 60);
  const markers = servingPredictionsToMarkers(rows, "serving-side-fixed-flight-v3");
  assert.deepEqual(markers.map(r => r.rallyId), ["R1", "R2", "R3"]);
  assert.deepEqual(markers.map(r => [r.side, r.modelSide, r.modelConfidence]), [["near", "near", .8], ["far", "far", .9], ["review", "far", .6]]);
  const nearReview = servingPredictionsToMarkers([{ ...rows[0], verdict: "review", nearProbability: .48 }], "model");
  assert.equal(nearReview[0].modelSide, "near", "Use the model's side threshold, not a new 0.5 cutoff");
  assert.equal(nearReview[0].modelConfidence, .48, "Use the predicted side's probability, not max(near, far)");
  assert.match(markers[2].notes ?? "", /neither serve head/);
  assert.deepEqual(servingPredictionsToMarkers(rows.filter(r => r.verdict === "not-serve"), "model"), []);
  assert.equal(JSON.stringify(input), unchanged);
});

test("serve candidate parsing rejects invalid timestamps, probabilities, decisions and duplicate IDs", () => {
  for (const invalid of [
    { anchor: NaN }, { anchor: 61 }, { nearProbability: 1.2 }, { verdict: "accepted" }, { side: "review" },
  ]) {
    const input = fixture(); Object.assign(input.candidates[0], invalid);
    assert.throws(() => parseServingPredictions(input, 60));
  }
  const input = fixture();input.candidates[1].id = "R1";
  assert.throws(() => parseServingPredictions(input, 60), /Duplicate/);
});

test("serve panel renders review and rejected candidates distinctly with source evidence", async () => {
  const require = createRequire(import.meta.url);
  await loadBindings();
  const source = (await readFile(new URL("../../components/labeling-serving-panel.tsx", import.meta.url), "utf8"))
    .replace(/import styles from "\.\/labeling-serving-panel.module.css";/, "const styles = new Proxy({}, { get: (_target, key) => String(key) });")
    .replaceAll('"@/lib/annotations"', JSON.stringify(new URL("../../lib/annotations.ts", import.meta.url).href))
    .replaceAll('"@/lib/labeling-serving"', JSON.stringify(new URL("../../lib/labeling-serving.ts", import.meta.url).href));
  const compiled = (await transform(source, { filename: "labeling-serving-panel.tsx", jsc: { parser: { syntax: "typescript", tsx: true }, target: "es2022", transform: { react: { runtime: "automatic" } } }, module: { type: "es6" } })).code
    .replaceAll('"react/jsx-runtime"', JSON.stringify(pathToFileURL(require.resolve("react/jsx-runtime")).href))
    .replaceAll('"react"', JSON.stringify(pathToFileURL(require.resolve("react")).href));
  const { LabelingServingPanel } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`);
  const markup = renderToStaticMarkup(createElement(LabelingServingPanel, {
    predictions: parseServingPredictions(fixture(), 60), modelLabel: "Serving model", ignoredIntervals: [], onSeek: () => {},
  }));
  for (const expected of ["Production serve predictions", "Needs review", "Side suggestion: far", "Rejected by serve gate", "Near 0.990", "Far 0.010", "threshold 0.850", "not confirmed serve-contact times"]) {
    assert.ok(markup.includes(expected), `Missing ${expected}`);
  }
  const filtered = renderToStaticMarkup(createElement(LabelingServingPanel, {
    predictions: parseServingPredictions(fixture(), 60), modelLabel: "Serving model", ignoredIntervals: [{ start: 0, end: 15, reason: "opening" }], onSeek: () => {},
  }));
  assert.ok(filtered.includes("Include 1 candidates in ignored footage"));
  assert.ok(!filtered.includes("R1 ·"));
});
