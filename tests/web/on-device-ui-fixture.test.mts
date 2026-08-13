import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { buildEditList, type Rally } from "../../lib/edit-list.ts";

type Fixture = {
  schemaVersion: number;
  provenance: { kind: string; modelId: string; note: string };
  source: { info: { duration: number } };
  analysis: { modelId: string; intervals: Rally[] };
  exportDefaults: { preRoll: number; postRoll: number };
};

const fixture = JSON.parse(
  readFileSync(
    new URL("../../public/on-device/browser-prediction-y9.json", import.meta.url),
    "utf8",
  ),
) as Fixture;

test("UI fixture is explicitly sourced from a browser on-device prediction", () => {
  assert.equal(fixture.schemaVersion, 1);
  assert.equal(fixture.provenance.kind, "browser-on-device-prediction");
  assert.equal(fixture.provenance.modelId, "model-9c92b8e9333f");
  assert.equal(fixture.analysis.modelId, fixture.provenance.modelId);
  assert.match(fixture.provenance.note, /not Python or server predictions/);
  assert.equal(fixture.analysis.intervals.length, 40);
  assert.ok(fixture.analysis.intervals.every((interval) => interval.end > interval.start));
});

test("cached predictions produce merged padded export sections", () => {
  const sections = buildEditList(
    fixture.analysis.intervals,
    fixture.exportDefaults.preRoll,
    fixture.exportDefaults.postRoll,
    fixture.source.info.duration,
  );
  assert.equal(sections.length, 38);
  const splitRallySection = sections.find((section) => section.rallyIds.includes("R008"));
  assert.deepEqual(splitRallySection?.rallyIds, ["R008", "R009"]);

  const widePadding = buildEditList(
    fixture.analysis.intervals,
    8,
    8,
    fixture.source.info.duration,
  );
  assert.equal(widePadding.length, 12);
});
