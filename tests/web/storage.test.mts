import assert from "node:assert/strict";
import path from "node:path";
import test from "node:test";
import {
  getAnalysesRoot,
  getDataRoot,
  getIntakeAnalysesRoot,
  getIntakeWorkspace,
  getIntakeWorkspaces,
  getModelFeedbackRoot,
} from "../../lib/storage.ts";

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

test("storage supports a separate no-beach analysis root", () => {
  const previous = process.env.VOLLEYCUT_NO_BEACH_ANALYSES_ROOT;
  process.env.VOLLEYCUT_NO_BEACH_ANALYSES_ROOT = "/mnt/example/no-beach/analyses";
  assert.equal(
    getAnalysesRoot("without-beach"),
    "/mnt/example/no-beach/analyses",
  );
  if (previous === undefined) delete process.env.VOLLEYCUT_NO_BEACH_ANALYSES_ROOT;
  else process.env.VOLLEYCUT_NO_BEACH_ANALYSES_ROOT = previous;
});

test("storage supports a supplemental intake workspace", () => {
  const previous = process.env.VOLLEYCUT_INTAKE_WORKSPACE;
  process.env.VOLLEYCUT_INTAKE_WORKSPACE = "/mnt/example/intake";
  assert.equal(getIntakeWorkspace(), "/mnt/example/intake");
  assert.equal(getIntakeAnalysesRoot(), "/mnt/example/intake/analyses");
  if (previous === undefined) delete process.env.VOLLEYCUT_INTAKE_WORKSPACE;
  else process.env.VOLLEYCUT_INTAKE_WORKSPACE = previous;
});

test("storage uses a neutral intake directory by default", () => {
  const previousWorkspace = process.env.VOLLEYCUT_INTAKE_WORKSPACE;
  const previousWorkspaces = process.env.VOLLEYCUT_INTAKE_WORKSPACES;
  delete process.env.VOLLEYCUT_INTAKE_WORKSPACE;
  delete process.env.VOLLEYCUT_INTAKE_WORKSPACES;
  assert.deepEqual(getIntakeWorkspaces(), [path.join(getDataRoot(), "intake")]);
  if (previousWorkspace === undefined)
    delete process.env.VOLLEYCUT_INTAKE_WORKSPACE;
  else process.env.VOLLEYCUT_INTAKE_WORKSPACE = previousWorkspace;
  if (previousWorkspaces === undefined)
    delete process.env.VOLLEYCUT_INTAKE_WORKSPACES;
  else process.env.VOLLEYCUT_INTAKE_WORKSPACES = previousWorkspaces;
});

test("model-feedback imports default under data and support a durable override", () => {
  const previousData = process.env.VOLLEYCUT_DATA_ROOT;
  const previousFeedback = process.env.VOLLEYCUT_MODEL_FEEDBACK_ROOT;
  process.env.VOLLEYCUT_DATA_ROOT = "/mnt/example/volleycut";
  delete process.env.VOLLEYCUT_MODEL_FEEDBACK_ROOT;
  assert.equal(getModelFeedbackRoot(), "/mnt/example/volleycut/model-feedback");
  process.env.VOLLEYCUT_MODEL_FEEDBACK_ROOT = "/mnt/example/feedback";
  assert.equal(getModelFeedbackRoot(), "/mnt/example/feedback");
  if (previousData === undefined) delete process.env.VOLLEYCUT_DATA_ROOT;
  else process.env.VOLLEYCUT_DATA_ROOT = previousData;
  if (previousFeedback === undefined)
    delete process.env.VOLLEYCUT_MODEL_FEEDBACK_ROOT;
  else process.env.VOLLEYCUT_MODEL_FEEDBACK_ROOT = previousFeedback;
});
