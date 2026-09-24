import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { Children, createElement, isValidElement, type ReactElement, type ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { loadBindings, transform } from "next/dist/build/swc/index.js";
import { initialModelSelection, selectedModelReferences, toggleModelSelection } from "../../lib/labeling-model-selection.ts";

const production = { modelId: "production-id" };
const experiments = [{ modelId: "compact-review", research: {} }, { modelId: "compact-preview" }, { modelId: "av-component" }];
const options = [{ value: "production", label: "Production ensemble" },
  { value: "model:compact-review", label: "Production + compact review" },
  { value: "model:compact-preview", label: "Proposed boundary preview" },
  { value: "model:av-component", label: "AV component" }, { value: "sol", label: "Sol reference" }];

async function loadComponent() {
  const require = createRequire(import.meta.url);
  await loadBindings();
  const source = (await readFile(new URL("../../components/model-breakdown-select.tsx", import.meta.url), "utf8"))
    .replace(/import styles from "\.\/model-breakdown-select.module.css";/, "const styles = new Proxy({}, { get: (_target, key) => String(key) });")
    .replaceAll('"@/lib/labeling-model-selection"', JSON.stringify(new URL("../../lib/labeling-model-selection.ts", import.meta.url).href));
  const compiled = (await transform(source, { filename: "model-breakdown-select.tsx", jsc: { parser: { syntax: "typescript", tsx: true }, target: "es2022", transform: { react: { runtime: "automatic" } } }, module: { type: "es6" } })).code
    .replaceAll('"react/jsx-runtime"', JSON.stringify(pathToFileURL(require.resolve("react/jsx-runtime")).href));
  return (await import(`data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`)).ModelBreakdownSelect;
}

function elements(node: ReactNode): ReactElement<Record<string, unknown>>[] {
  return Children.toArray(node).flatMap(child => {
    if (!isValidElement<Record<string, unknown>>(child)) return [];
    return [child, ...elements(child.props.children as ReactNode)];
  });
}

test("new recording defaults show production and compact together without selecting every component", () => {
  assert.deepEqual(initialModelSelection(production, experiments), ["production", "model:compact-review"]);
  assert.deepEqual(initialModelSelection(null, experiments), ["model:compact-review"]);
  assert.deepEqual(initialModelSelection(production, []), ["production"]);
  assert.deepEqual(initialModelSelection(null, []), []);
  assert.deepEqual(toggleModelSelection(["production"], "production", true), ["production"], "duplicate checkbox changes cannot duplicate rails");
});

test("checkbox changes retain arbitrary subsets and All/Clear affect every available reference", async () => {
  const Component = await loadComponent();
  let selected = initialModelSelection(production, experiments);
  const render = () => Component({ options, selected, onChange: (values: string[]) => { selected = values; } });
  function check(index: number, checked: boolean) {
    const input = elements(render()).filter(element => element.type === "input")[index];
    (input.props.onChange as (event: { target: { checked: boolean } }) => void)({ target: { checked } });
  }
  check(2, true);
  assert.deepEqual(selectedModelReferences(production, experiments, selected).map(reference => reference.modelId), ["production-id", "compact-review", "compact-preview"]);
  check(1, false);
  check(4, true);
  assert.deepEqual(selected, ["production", "model:compact-preview", "sol"]);
  assert.deepEqual(selectedModelReferences(production, experiments, selected).map(reference => reference.modelId), ["production-id", "compact-preview"]);
  const all = elements(render()).find(element => element.props["aria-label"] === "Select all model rails")!;
  (all.props.onClick as () => void)();
  assert.deepEqual(selected, options.map(option => option.value));
  const clear = elements(render()).find(element => element.props["aria-label"] === "Clear selected model rails")!;
  (clear.props.onClick as () => void)();
  assert.deepEqual(selected, []);
  assert.deepEqual(selectedModelReferences(production, experiments, selected), []);
});

test("multiselect renders native labeled checkboxes with a selected count and human-only empty state", async () => {
  const Component = await loadComponent();
  const markup = renderToStaticMarkup(createElement(Component, { options, selected: ["production", "model:compact-preview", "sol"], onChange: () => {} }));
  assert.ok(markup.includes("Model breakdown"));
  assert.ok(markup.includes("3 model rails selected"));
  assert.equal((markup.match(/type="checkbox"/g) ?? []).length, 5);
  assert.equal((markup.match(/checked=""/g) ?? []).length, 3);
  for (const option of options) assert.ok(markup.includes(option.label));
  assert.ok(!markup.includes("<select"), "multi-selection does not require Ctrl or platform-specific native list gestures");
  const empty = renderToStaticMarkup(createElement(Component, { options, selected: [], onChange: () => {} }));
  assert.ok(empty.includes("Human only"));
});
