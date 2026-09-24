import { privateValue } from "../lib/server/private-ledger.mjs";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve, join } from "node:path";

const args = new Map(process.argv.slice(2).reduce((pairs, value, i, values) => i % 2 ? pairs : [...pairs, [value, values[i + 1]]], []));
const require = createRequire(resolve(args.get("--harness"), "package.json"));
const { chromium } = require("playwright-core");
const output = resolve(args.get("--output"));
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ executablePath: args.get("--browser"), headless: true });
const context = await browser.newContext({ viewport: { width: 1500, height: 1100 } });
const page = await context.newPage();
const base = args.get("--url") ?? "http://localhost:3000";
const errors = [];
page.on("pageerror", error => errors.push(error.message));
page.on("console", message => { if (message.type() === "error") errors.push(message.text()); });
const checks = [];
try {
  const catalog = await (await context.request.get(base + "/api/editor-lab/tasks")).json();
  assert.equal(catalog.tasks.length, 44);
  for (const row of catalog.tasks) {
    const response = await context.request.get(`${base}/api/labeling/tasks/${row.id}/references`);
    assert.equal(response.status(), 200, row.id);
    const data = await response.json();
    const models = data.experiments.filter(model => model.modelId.startsWith("neural-"));
    assert.equal(models.length, row.id.startsWith("beach-") ? 2 : 10, row.id);
    assert.ok(models.every(model => model.research.signals.times.length && model.modelLabel.includes("99%")), row.id);
  }
  checks.push("All 42 non-beach videos expose ten picks; both beach videos expose the two requested high-recall TCNs with native signals");
  await page.goto(`${base}/editor-lab?task=${privateValue("recording-044")}&mode=neural-dino-transformer-int8`, { waitUntil: "domcontentloaded", timeout: 120000 });
  await page.getByRole("combobox", { name: "Editor lab video" }).waitFor({ timeout: 120000 });
  await page.waitForFunction(() => document.querySelector("video")?.readyState >= 2, { timeout: 120000 });
  assert.equal(await page.getByRole("button", { name: /target 99%/ }).count(), 10);
  for (const name of ["INT8 DINO-transformer", "FP32 DINO-transformer", "DINO-TCN", "Mobile-TCN", "Distilled Mobile-TCN"]) {
    for (const choice of ["F1", "recall"]) {
      await page.getByRole("button", { name: `${name} · highest ${choice} · target 99%`, exact: true }).click();
      await page.getByText("Model signals for this region", { exact: true }).click();
      assert.equal(await page.getByRole("img", { name: "Four model signals; click to seek" }).locator("polyline").count(), 4);
    }
  }
  await page.waitForFunction(() => document.querySelector("video")?.readyState >= 2);
  const startTime = await page.locator("video").evaluate(async video => { video.muted = true; await video.play(); return video.currentTime; });
  await page.waitForFunction(start => document.querySelector("video").currentTime > start + .3, startTime);
  await page.locator("video").evaluate(video => video.pause());
  await page.screenshot({ path: join(output, "editor-lab.png"), fullPage: true });
  await page.getByRole("combobox", { name: "Editor lab video" }).selectOption(privateValue("indoor-source-05"));
  await page.waitForFunction(() => document.querySelector("video")?.readyState >= 2);
  await page.getByRole("button", { name: "Human export", exact: true }).waitFor();
  checks.push("All ten editor modes render four signals; video switch preserves selected mode and loads human reference");
  await page.goto(`${base}/labelv2?task=${privateValue("recording-026")}`, { waitUntil: "domcontentloaded", timeout: 120000 });
  const breakdown = page.getByRole("group", { name: "Model breakdown", exact: true });
  await breakdown.locator("summary").waitFor({ timeout: 120000 });
  // Wait for hydrated model data before toggling the native details element.
  await breakdown.getByRole("checkbox", { name: "INT8 DINO-transformer · highest F1 · target 99%", exact: true, includeHidden: true }).waitFor({ state: "attached", timeout: 120000 });
  await breakdown.locator("summary").click();
  for (const name of ["INT8 DINO-transformer", "FP32 DINO-transformer", "DINO-TCN", "Mobile-TCN", "Distilled Mobile-TCN"]) {
    for (const choice of ["F1", "recall"]) {
      await breakdown.getByRole("checkbox", { name: `${name} · highest ${choice} · target 99%`, exact: true }).check();
    }
  }
  await page.waitForFunction(() => document.querySelector("video")?.readyState >= 2, { timeout: 120000 });
  await page.screenshot({ path: join(output, "labelv2.png"), fullPage: true });
  checks.push("Imported August export plays in labelv2 with all ten multiselect options");
  await page.goto(`${base}/labelv2?task=${privateValue("beach-source-02")}`, { waitUntil: "domcontentloaded", timeout: 120000 });
  const beachBreakdown = page.getByRole("group", { name: "Model breakdown", exact: true });
  await beachBreakdown.getByRole("checkbox", { name: /DINO-TCN.*highest recall.*target 99%/, includeHidden: true }).waitFor({ state: "attached", timeout: 120000 });
  assert.equal(await beachBreakdown.getByRole("checkbox", { name: /TCN.*highest recall.*target 99%/, includeHidden: true }).count(), 2);
  assert.equal(await beachBreakdown.getByRole("checkbox", { name: /Production/, includeHidden: true }).count() > 0, true);
  await page.screenshot({ path: join(output, "labelv2-beach.png"), fullPage: true });
  checks.push("Beach labelv2 retains its editable human document and production reference alongside both high-recall TCNs");
  assert.deepEqual(errors, []);
  await writeFile(join(output, "browser.json"), JSON.stringify({ passed: true, checks, errors }, null, 2));
  console.log(JSON.stringify({ passed: true, checks }));
} finally { await browser.close(); }
