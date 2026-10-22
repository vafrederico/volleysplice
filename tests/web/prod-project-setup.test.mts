import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import test from "node:test";

test("real production setup queues changed inputs and preserves saved reviews", () => {
  const output = execFileSync(process.execPath, [
    fileURLToPath(new URL("../../prod/scripts/test-project-setup.mjs", import.meta.url)),
  ], { encoding: "utf8", timeout: 60_000 });
  assert.match(output, /Project setup: changed crop, side-switch, reuse and legacy restore passed/);
});
