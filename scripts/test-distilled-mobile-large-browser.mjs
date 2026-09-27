import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { privateValue } from "../lib/server/private-ledger.mjs";

const args = new Map(process.argv.slice(2).reduce((rows, v, i, all) => i % 2 ? rows : [...rows, [v, all[i + 1]]], []));
if (!args.get("--output") || !args.get("--browser")) throw Error("Specify external --output and --browser");
const require = createRequire(path.resolve(args.get("--harness") ?? privateValue("private-reference-0109"), "package.json"));
const { chromium } = require("playwright-core");
const base = args.get("--url") ?? "http://localhost:3000";
const ids = ["neural-distilled-mobile-large-tcn-fp32", "neural-distilled-mobile-large-tcn-fp32-high-recall"];
const request = async route => {
  const response = await fetch(new URL(route, base));
  assert.equal(response.status, 200);
  return response.json();
};
const catalog = await request("/api/editor-lab/tasks");
assert.equal(catalog.tasks.length, 44);
let next = 0, verified = 0;
await Promise.all(Array.from({ length: 3 }, async () => {
  while (next < catalog.tasks.length) {
    const row = catalog.tasks[next++];
    const task = await request(`/api/editor-lab/tasks/${encodeURIComponent(row.id)}`);
    for (const id of ids) {
      const config = task.configurations.find(c => c.id === id);
      assert.ok(config && config.signals.times.length > 0);
      for (const head of ["live", "serve", "end", "keep"]) assert.equal(config.signals[head].length, config.signals.times.length);
    }
    verified++;
  }
}));
const browser = await chromium.launch({ executablePath: args.get("--browser"), headless: true });
const context = await browser.newContext({ viewport: { width: 1500, height: 1100 }, reducedMotion: "reduce" });
const page = await context.newPage();
const errors = [];
page.on("pageerror", e => errors.push(e.message));
try {
  const url = new URL(privateValue("private-reference-0115"), base);
  url.searchParams.set("mode", ids[0]);
  url.searchParams.set("suppression", "none");
  await page.goto(url.href, { waitUntil: "networkidle", timeout: 90000 });
  const group = page.getByRole("group", { name: "Model configuration", exact: true });
  const waitForSelectedModel = async (id, label) => {
    // Mode switches remount Trial asynchronously; a visible metrics element can
    // still belong to the previous model. Wait for the new rail's own label.
    await page.waitForFunction(({ id, label }) => {
      const selected = document.querySelector('[aria-label="Model configuration"] button[aria-pressed="true"]');
      const rail = document.querySelector('[aria-label="Human comparison rail"]');
      return new URL(location.href).searchParams.get("mode") === id
        && selected?.textContent?.trim() === label
        && Array.from(rail?.querySelectorAll("small") ?? []).some(detail => detail.textContent?.startsWith(`${label} · export `));
    }, { id, label });
    await page.getByTestId("lab-comparison-metrics").waitFor();
  };
  let finalLabel;
  for (const [index, mode] of ["F1", "recall"].entries()) {
    const button = group.getByRole("button", { name: new RegExp(`^Distilled MobileNetV3 Large-TCN .*highest ${mode}`) });
    const label = (await button.innerText()).trim();
    await button.click();
    await waitForSelectedModel(ids[index], label);
    assert.equal(await button.getAttribute("aria-pressed"), "true");
    await page.getByLabel("Suppression combination", { exact: true }).selectOption("aggressive");
    await page.getByTestId("suppression-baseline-comparison").waitFor();
    await page.getByLabel("Suppression combination", { exact: true }).selectOption("none");
    await waitForSelectedModel(ids[index], label);
    finalLabel = label;
  }
  await page.reload({ waitUntil: "networkidle" });
  await waitForSelectedModel(ids[1], finalLabel);
  await mkdir(args.get("--output"), { recursive: true });
  await page.screenshot({ path: path.join(args.get("--output"), "distilled-large-editor.png"), fullPage: true });
  assert.deepEqual(errors, []);
  const result = { passed: true, verifiedRecordings: verified, models: ids,
    checks: ["All catalog videos expose both models and four-head signals", "Both editor choices render and retain URL selection on reload", "Both models support suppression combinations and human comparison"], errors };
  await writeFile(path.join(args.get("--output"), "browser-checks.json"), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result));
} finally { await context.close(); await browser.close(); }
