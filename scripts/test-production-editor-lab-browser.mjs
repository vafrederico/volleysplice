import { privateValue } from "../lib/server/private-ledger.mjs";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve, join } from "node:path";

const args = new Map(process.argv.slice(2).reduce((pairs, value, i, values) => i % 2 ? pairs : [...pairs, [value, values[i + 1]]], []));
if (!args.get("--harness")) throw new Error("Pass an existing Playwright dependency directory via --harness");
const require = createRequire(resolve(args.get("--harness"), "package.json"));
const { chromium } = require("playwright-core");
const output = resolve(args.get("--output") ?? "artifacts/private-media/editor-lab");
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ executablePath: args.get("--browser") ?? "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe", headless: true });
const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, reducedMotion: "reduce" });
const page = await context.newPage();
const errors = [];
const checks = [];
page.on("pageerror", error => errors.push(error.message));
const url = (args.get("--url") ?? "http://localhost:3000") + privateValue("private-reference-0115");
const draft = () => page.evaluate(() => {
  const mode = new URL(location.href).searchParams.get("mode") ?? "production";
  const key = Object.keys(localStorage).find(k => k.includes(":trial:v1:") && k.endsWith(`:${mode}`));
  return key ? JSON.parse(localStorage.getItem(key)) : null;
});
const waitForDraft = async predicate => {
  for (let i = 0; i < 60; i++) { const value = await draft(); if (value && predicate(value)) return value; await page.waitForTimeout(100); }
  throw new Error("Draft did not reach expected state");
};
const mode = async label => { await page.getByRole("button", { name: label, exact: true }).click(); await page.waitForTimeout(250); };
try {
  await page.goto(url, { waitUntil: "networkidle", timeout: 60000 });
  await page.getByRole("button", { name: "Compact standalone", exact: true }).waitFor();
  await page.waitForFunction(() => document.querySelector("video")?.readyState >= 2);
  assert.equal((await draft()).cuts.length, 52);
  assert.notEqual(await page.locator(".taste-root").evaluate(element => getComputedStyle(element).backgroundColor), "rgba(0, 0, 0, 0)");
  assert.equal(await page.getByRole("dialog").count(), 0);
  checks.push("Production video plays from NAS; 52 evaluable candidates; production styles loaded; no irrelevant onboarding");

  await mode("Boundary edits + removal review");
  await page.getByRole("button", { name: "Removed footage (60)", exact: true }).waitFor();
  let before = await draft(); assert.equal(before.cuts.length, 42); assert.deepEqual(before.labDecisions, {});
  await page.getByRole("group", { name: "Human comparison rail", exact: true }).waitFor();
  const metricsBefore = await page.getByTestId("lab-comparison-metrics").innerText();
  assert.equal(before.scoreTracking.serveMarkers.filter(marker => marker.side === "review").length, 4);
  await page.getByRole("button", { name: "Restore footage", exact: true }).click();
  let restored = await waitForDraft(d => Object.keys(d.labDecisions ?? {}).length === 1);
  assert.ok(restored.cuts[0].coreStart < before.cuts[0].coreStart);
  assert.deepEqual(restored.cuts.slice(1), before.cuts.slice(1));
  assert.notEqual(await page.getByTestId("lab-comparison-metrics").innerText(), metricsBefore);
  await page.keyboard.press("Control+z");
  const undone = await waitForDraft(d => Object.keys(d.labDecisions ?? {}).length === 0);
  assert.deepEqual(undone.cuts, before.cuts);
  assert.equal(await page.getByTestId("lab-comparison-metrics").innerText(), metricsBefore);
  const comparisonSegment = page.getByRole("group", { name: "Human comparison rail", exact: true }).getByTitle(/^Matches padded human export/).first();
  const comparisonTime = Number(await comparisonSegment.getAttribute("data-timeline-seek-time"));
  await comparisonSegment.click();
  await page.waitForFunction(time => Math.abs(document.querySelector("video").currentTime - time) < .1, comparisonTime);
  checks.push("Human comparison metrics update on restore and Undo; rail seeks the source video");
  checks.push("Removal restores just its original portion; keyboard Undo reverses geometry and decision together");

  await page.getByText("Model signals for this region", { exact: true }).click();
  const signalChart = page.getByRole("img", { name: "Four compact model signals; click to seek", exact: true });
  const signalBounds = await signalChart.boundingBox();
  await signalChart.click({ position: { x: signalBounds.width / 2, y: 20 } });
  await page.waitForFunction(() => Math.abs(Number(document.querySelector('[data-testid="signal-playhead"]')?.getAttribute("x1")) - 300) < 2);
  await page.getByRole("button", { name: "Play with context", exact: true }).click();
  await page.waitForFunction(() => { const v = document.querySelector("video"); return v && !v.paused && v.currentTime >= 157; });
  await page.waitForFunction(() => {
    const signal = Number(document.querySelector('[data-testid="signal-playhead"]')?.getAttribute("x1"));
    const region = parseFloat(document.querySelector('[data-testid="region-playhead"]')?.style.left);
    return signal > 80 && signal < 590 && Math.abs(signal / 6 - region) < .1;
  });
  await page.waitForFunction(() => { const v = document.querySelector("video"); return v && v.paused && Math.abs(v.currentTime - 162.241722) < .05; }, null, { timeout: 15000 });
  assert.equal((await draft()).cutPreviewEnabled, false);
  checks.push("Source context playback includes removed footage and pauses at its end");
  checks.push("Region and signal playheads follow seeking and move together during source playback");

  await page.getByRole("button", { name: "Rally splits (2)", exact: true }).click();
  await page.getByRole("button", { name: "Undo split", exact: true }).click();
  let merged = await waitForDraft(d => d.cuts.length === 41);
  assert.ok(merged.scoreTracking.serveMarkers.every(marker => !marker.rallyId || merged.cuts.some(cut => cut.id === marker.rallyId)));
  await page.keyboard.press("Control+z"); await waitForDraft(d => d.cuts.length === 42);
  checks.push("Explicit split undo merges linked events and removes obsolete rally-linked markers");

  await mode("Compact standalone");
  assert.equal((await draft()).cuts.length, 34);
  assert.equal((await draft()).scoreTracking.serveMarkers.length, 32);
  assert.equal((await draft()).scoreTracking.serveMarkers.filter(marker => marker.side === "review").length, 0);
  await page.getByRole("button", { name: "Remove rally", exact: true }).click();
  const editedStandalone = await waitForDraft(d => d.cuts.some(cut => !cut.included));
  await mode("Production ensemble"); assert.equal((await draft()).cuts.filter(cut => cut.included).length, 52);
  await mode("Compact standalone"); assert.deepEqual((await draft()).cuts, editedStandalone.cuts);
  await page.reload({ waitUntil: "networkidle" });
  await page.getByRole("button", { name: "Compact standalone", exact: true }).waitFor();
  await waitForDraft(d => d.cuts.some(cut => !cut.included));
  checks.push("Manual editing persists across mode switches and reload; production draft stays independent");

  await mode("Human export");
  const humanBefore = await draft();
  assert.equal(humanBefore.cuts.length, 37);
  assert.equal(humanBefore.reviewedCutIds.length, 37);
  assert.equal(humanBefore.scoreTracking.serveMarkers.length, 36);
  assert.equal(await page.locator(".td-confidence").innerText(), "Human export");
  assert.equal(await page.getByTestId("lab-comparison-metrics").locator("strong").nth(2).innerText(), "100.0%");
  await page.getByRole("button", { name: "Remove rally", exact: true }).click();
  const humanEdited = await waitForDraft(d => d.cuts.some(cut => !cut.included));
  await mode("Production ensemble"); assert.equal((await draft()).cuts.filter(cut => cut.included).length, 52);
  await mode("Human export"); assert.deepEqual((await draft()).cuts, humanEdited.cuts);
  await page.keyboard.press("Control+z"); await waitForDraft(d => d.cuts.every(cut => cut.included));
  checks.push("Human export loads 37 reviewed regions and 36 saved serves; edits/undo stay isolated; neural modes use fresh start-specific serve predictions");

  await mode("Compact review");
  await page.getByRole("button", { name: "Boundary proposals (9)", exact: true }).waitFor();
  await page.getByRole("button", { name: "All 27 flagged rallies", exact: true }).click();
  await page.getByRole("button", { name: "Boundary proposals (27)", exact: true }).waitFor();
  const guidanceBefore = await draft();
  await page.getByRole("button", { name: "Apply proposal", exact: true }).click();
  const guidanceAfter = await waitForDraft(d => Object.values(d.labDecisions ?? {}).some(decision => decision.action === "Proposal applied"));
  assert.notDeepEqual(guidanceAfter.cuts, guidanceBefore.cuts);
  await page.keyboard.press("Control+z"); await waitForDraft(d => Object.keys(d.labDecisions ?? {}).length === 0);
  checks.push("Guidance exposes nine recommended/all 27 flagged parents; Apply edits events and is undoable");

  await mode("Suppression · aggressive");
  assert.equal((await draft()).selectedSuppressionPolicy, "aggressive");
  await page.getByRole("button", { name: "Restore footage", exact: true }).click();
  const suppressionDraft = await waitForDraft(d => Object.values(d.suppressionDecisionOverrides).includes("keep"));
  assert.ok(suppressionDraft.cuts.length === 52);
  checks.push("Suppression restoration uses real production override/barrier semantics");

  await mode("Boundary edits + removal review");
  await page.screenshot({ path: join(output, "desktop.png"), fullPage: true });
  const desktop = await page.locator("video").boundingBox();
  assert.ok(desktop.width < 1440 && desktop.height < 900, `Desktop video must fit workspace: ${JSON.stringify(desktop)}`);
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), "Desktop page overflows horizontally");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(300);
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), "Mobile page overflows horizontally");
  assert.ok(await page.evaluate(() => {
    const media = document.querySelector(".rd-media-stack");
    const serves = document.querySelector(".rd-mobile-events");
    return media && serves && media.getBoundingClientRect().bottom <= serves.getBoundingClientRect().top;
  }), "Mobile serves must follow the player so source review stays adjacent");
  await page.getByRole("button", { name: "Play with context", exact: true }).scrollIntoViewIfNeeded();
  await page.screenshot({ path: join(output, "mobile.png"), fullPage: true });
  checks.push("Desktop/mobile layouts fit their viewports with functional review controls");
  assert.deepEqual(errors, []);
  await writeFile(join(output, "verification.json"), JSON.stringify({ passed: true, url, checks, pageErrors: errors, screenshots: ["desktop.png", "mobile.png"] }, null, 2));
  console.log(JSON.stringify({ passed: true, checks, output }));
} catch (error) {
  await page.screenshot({ path: join(output, "failure.png"), fullPage: true }).catch(() => {});
  await writeFile(join(output, "failure.json"), JSON.stringify({ error: String(error), errors, checks, text: await page.locator("body").innerText() }, null, 2));
  throw error;
} finally { await browser.close(); }
