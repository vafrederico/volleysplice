import assert from "node:assert/strict";
import path from "node:path";
import test from "node:test";
import { getAnalysesRoot, getDataRoot } from "../../lib/storage.ts";

test("storage defaults to the repository data directory", () => {
  const previous = process.env.VOLLEYCUT_DATA_ROOT;
  delete process.env.VOLLEYCUT_DATA_ROOT;
  assert.equal(getDataRoot(), path.join(process.cwd(), "data"));
  assert.equal(getAnalysesRoot(), path.join(process.cwd(), "data", "analyses"));
  if (previous === undefined) delete process.env.VOLLEYCUT_DATA_ROOT;
  else process.env.VOLLEYCUT_DATA_ROOT = previous;
});

test("storage honors the server-side configured media root", () => {
  const previous = process.env.VOLLEYCUT_DATA_ROOT;
  process.env.VOLLEYCUT_DATA_ROOT = "/mnt/example/volleycut";
  assert.equal(getDataRoot(), "/mnt/example/volleycut");
  assert.equal(getAnalysesRoot(), "/mnt/example/volleycut/analyses");
  if (previous === undefined) delete process.env.VOLLEYCUT_DATA_ROOT;
  else process.env.VOLLEYCUT_DATA_ROOT = previous;
});
