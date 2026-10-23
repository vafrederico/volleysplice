import assert from "node:assert/strict";
import test from "node:test";
import { scoreSpecialistReadiness, missingScoreStages } from "../../prod/src/lib/score-specialist-state.ts";
import { normalizeStoredProject, type VolleySpliceProject } from "../../prod/src/lib/project-store.ts";
import * as serving from "../../prod/src/lib/on-device/serving-side-cache.ts";
import * as neural from "../../prod/src/lib/on-device/neural-serving-model.ts";
import * as switches from "../../prod/src/lib/on-device/side-switch-model.ts";
import { PRODUCTION_ENSEMBLE_MODEL_ID } from "../../prod/src/lib/on-device/ensemble.ts";
import { recoveryFixture, servingFixture, switchFixture } from "../../prod/scripts/score-recovery-fixtures.mjs";

test("nullable and absent outputs need recovery; validated zero candidates complete both stages", () => {
  const project = recoveryFixture({ empty: true, switches: true }) as VolleySpliceProject;
  project.analysis!.servingSide = null;
  project.analysis!.sideSwitch = null;
  assert.deepEqual(missingScoreStages(project), ["servingSide", "sideSwitch"]);
  const normalized = normalizeStoredProject(project);
  assert.equal(normalized.status, "ready");
  project.analysis!.servingSide = servingFixture(project.analysis, serving, neural);
  project.analysis!.sideSwitch = switchFixture(switches);
  assert.deepEqual(missingScoreStages(project), []);
  assert.equal(scoreSpecialistReadiness(project).servingSide.status, "complete");
  delete (project.analysis!.sideSwitch as unknown as { candidates?: unknown }).candidates;
  assert.equal(scoreSpecialistReadiness(project).sideSwitch.status, "missing", "Malformed candidates cannot masquerade as an empty successful output");
});

test("paired neural cache invalidation and errors survive rally readiness; valid legacy caches skip", () => {
  for (const modelId of [PRODUCTION_ENSEMBLE_MODEL_ID, "distilled-large-recall-v1", "distilled-large-f1-v1"]) {
    const project = recoveryFixture({ modelId, switches: true }) as VolleySpliceProject;
    project.analysis!.servingSide = servingFixture(project.analysis, serving, neural);
    project.analysis!.sideSwitch = switchFixture(switches);
    assert.deepEqual(missingScoreStages(normalizeStoredProject(project)), []);
    project.analysis!.servingSide.modelFingerprint = "incompatible";
    project.scoreSpecialistErrors = { servingSide: "Synthetic specialist failed" };
    const restored = normalizeStoredProject(project);
    assert.equal(restored.status, "ready");
    assert.equal(restored.analysis!.servingSide, undefined);
    assert.deepEqual(missingScoreStages(restored), ["servingSide"]);
    assert.deepEqual(scoreSpecialistReadiness(restored).servingSide, { status: "failed", message: "Synthetic specialist failed" });
  }
});

test("disabled switches and missing evidence remain explicit; checkpointed rallies resume as ready", () => {
  const project = recoveryFixture() as VolleySpliceProject;
  assert.equal(scoreSpecialistReadiness(project).sideSwitch.status, "disabled");
  delete project.analysis!.productionServeOutputs;
  assert.equal(scoreSpecialistReadiness(project).servingSide.status, "unavailable");
  project.analysis!.featureNames = ["synthetic-feature"];
  project.analysis!.featureValues = new Float32Array([0]);
  assert.equal(scoreSpecialistReadiness(project).servingSide.status, "missing");
  project.status = "analyzing";
  assert.equal(normalizeStoredProject(project).status, "ready");
  assert.strictEqual(normalizeStoredProject(project).analysis, project.analysis);
});
