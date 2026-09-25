import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import { privateValue } from "../lib/server/private-ledger.mjs";

const args = new Map(process.argv.slice(2).reduce((rows, value, index, all) =>
  index % 2 ? rows : [...rows, [value, all[index + 1]]], []));
assert.ok(args.get("--output") && args.get("--browser"), "Specify external --output and --browser");
assert.ok(path.isAbsolute(args.get("--output")), "QA output must be an absolute external directory");
const require = createRequire(path.resolve(args.get("--harness") ?? privateValue("private-reference-0109"), "package.json"));
const { chromium } = require("playwright-core");
const base = args.get("--url") ?? "http://localhost:3000";
const modelIds = ["neural-distilled-mobile-large-tcn-fp32", "neural-distilled-mobile-large-tcn-fp32-high-recall"];
const heads = ["live", "serve", "end", "keep"];
const hash = value => createHash("sha256").update(JSON.stringify(value)).digest("hex");
const request = async route => {
  const response = await fetch(new URL(route, base), { signal: AbortSignal.timeout(120_000) });
  assert.equal(response.status, 200, "Expected a successful read-only labeling API response");
  return response.json();
};

// Discover IDs only through runtime catalogs; never commit private recording IDs.
const catalog = await request("/api/editor-lab/tasks");
assert.equal(catalog.tasks.length, Number(args.get("--expected-recordings") ?? 44));
assert.equal(new Set(catalog.tasks.map(task => task.id)).size, catalog.tasks.length);
const labels = await request("/api/labeling/tasks");
const taskIds = new Set(catalog.tasks.map(task => task.id));
assert.ok(catalog.tasks.every(task => labels.tasks.some(row => row.id === task.id)), "Every comparison video must be available to labelv2");
const labelTasks = labels.tasks.filter(task => taskIds.has(task.id));
const labelById = new Map(labelTasks.map(task => [task.id, task]));
const exact = labelTasks.find(task => task.environment !== "beach" && task.documentSource === "completed"
  && task.rallyCount > 0 && !task.modelSeeded)
  ?? labelTasks.find(task => task.environment !== "beach" && task.documentSource === "completed" && task.rallyCount > 0);
const beach = labelTasks.find(task => task.environment === "beach" && task.rallyCount > 0);
assert.ok(exact, "An exact human-labeled non-beach recording is required for UI validation");
assert.ok(beach, "A beach recording with labels is required for UI validation");
const sampleReferences = new Map();
let cursor = 0;
let verified = 0;
let minimumSignalSamples = Infinity;
let maximumSignalSamples = 0;
await Promise.all(Array.from({ length: 2 }, async () => {
  while (cursor < catalog.tasks.length) {
    const task = catalog.tasks[cursor++];
    const payload = await request(`/api/labeling/tasks/${encodeURIComponent(task.id)}/references`);
    const references = payload.experiments;
    assert.ok(Array.isArray(references), "Label references must include the experiment array");
    const selected = modelIds.map(modelId => {
      const matches = references.filter(reference => reference.modelId === modelId);
      assert.equal(matches.length, 1, "Each new model must appear exactly once");
      const reference = matches[0];
      const signals = reference.research?.signals;
      assert.ok(signals && signals.times.length > 0, "Model signals must not be empty");
      assert.ok(signals.times.every((time, index) => Number.isFinite(time) && time >= 0
        && time <= labelById.get(task.id).durationSeconds && (!index || time > signals.times[index - 1])), "Signals must use an ordered source timeline");
      for (const head of heads) {
        assert.equal(signals[head]?.length, signals.times.length, "All four signals must align");
        assert.ok(signals[head].every(value => Number.isFinite(value) && value >= 0 && value <= 1), "Signals must be finite probabilities");
      }
      minimumSignalSamples = Math.min(minimumSignalSamples, signals.times.length);
      maximumSignalSamples = Math.max(maximumSignalSamples, signals.times.length);
      assert.ok(Array.isArray(reference.rallies), "Read-only model rallies are required");
      return reference;
    });
    if (task.id === exact.id || task.id === beach.id) sampleReferences.set(task.id, selected);
    verified++;
  }
}));

await mkdir(args.get("--output"), { recursive: true });
const browser = await chromium.launch({ executablePath: args.get("--browser"), headless: true });
const context = await browser.newContext({ viewport: { width: 1600, height: 1200 }, reducedMotion: "reduce" });
const pageErrors = [];
const blockedWrites = [];
// This isolated browser context cannot save/modify any label document.
await context.route("**/api/**", async route => {
  if (!["GET", "HEAD", "OPTIONS"].includes(route.request().method())) {
    blockedWrites.push(route.request().method());
    return route.abort();
  }
  return route.continue();
});
const examples = [];
try {
  for (const [kind, task] of [["exact", exact], ["beach", beach]]) {
    const before = await request(`/api/labeling/tasks/${encodeURIComponent(task.id)}`);
    const page = await context.newPage();
    page.on("pageerror", error => pageErrors.push(error.message));
    const url = new URL("/labelv2", base);
    url.searchParams.set("task", task.id);
    await page.goto(url.href, { waitUntil: "domcontentloaded", timeout: 90_000 });
    const group = page.getByRole("group", { name: "Model breakdown", exact: true });
    await group.waitFor({ state: "visible", timeout: 120_000 });
    const dropdown = group.locator("details");
    if (await dropdown.getAttribute("open") === null) await dropdown.locator("summary").click();
    const references = sampleReferences.get(task.id);
    const boxes = references.map(reference => group.getByRole("checkbox", { name: reference.modelLabel, exact: true }));
    for (const box of boxes) await box.waitFor({ state: "visible", timeout: 120_000 });
    const original = await group.getByRole("checkbox").evaluateAll(elements => elements.map(element => ({
      label: element.closest("label").textContent.trim(), checked: element.checked,
    })));
    if (original.some(value => value.checked)) await group.getByRole("button", { name: "Clear selected model rails", exact: true }).click();
    const timeline = page.getByRole("group", { name: "Editable human labels and read-only model references", exact: true });
    for (const [index, reference] of references.entries()) {
      await boxes[index].check();
      await timeline.getByText(reference.modelLabel, { exact: true }).waitFor({ state: "visible" });
      const signals = page.getByRole("region", { name: "Model signals", exact: true });
      await signals.getByText(reference.modelLabel, { exact: true }).waitFor({ state: "visible" });
      const paths = await signals.locator("svg path").evaluateAll(elements => elements.map(element => element.getAttribute("d")));
      assert.equal(paths.length, 4);
      assert.ok(paths.every(value => typeof value === "string" && value.startsWith("M")), "All four signal charts must render");
      await boxes[index].uncheck();
    }
    for (const box of boxes) await box.check();
    for (const reference of references) await timeline.getByText(reference.modelLabel, { exact: true }).waitFor({ state: "visible" });
    assert.equal(await group.getByRole("checkbox", { checked: true }).count(), 2, "Both model rails must stay selected together");
    await dropdown.locator("summary").click();
    await timeline.scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(args.get("--output"), `${kind}-labelv2.png`), fullPage: true });
    await dropdown.locator("summary").click();
    for (const value of original) await group.getByRole("checkbox", { name: value.label, exact: true }).setChecked(value.checked);
    const after = await request(`/api/labeling/tasks/${encodeURIComponent(task.id)}`);
    assert.equal(hash(after), hash(before), "Human labels must remain unchanged");
    examples.push({ kind, environment: task.environment, labelSource: task.documentSource,
      models: references.map(reference => reference.modelId), humanLabelsUnchanged: true,
      individualSignalsRendered: true, simultaneousRailsRendered: true });
    await page.close();
  }
  assert.deepEqual(pageErrors, []);
  assert.deepEqual(blockedWrites, [], "Multiselect interaction must not attempt label writes");
  const result = { passed: true, verifiedReferenceRecordings: verified, models: modelIds,
    minimumSignalSamples, maximumSignalSamples, examples,
    checks: ["All comparison videos are available in the labeling catalog", "Both models expose four aligned signals on every references endpoint",
      "Each model renders its signal panel; both rails can be selected together on exact and beach videos", "Human labels unchanged; no write requests"],
    errors: pageErrors };
  await writeFile(path.join(args.get("--output"), "labelv2-browser-checks.json"), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result));
} finally {
  await context.close();
  await browser.close();
}
