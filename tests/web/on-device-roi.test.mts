import assert from "node:assert/strict";
import test from "node:test";

import { inferRoiProfile, roiProfiles } from "../../lib/on-device/roi.ts";
import type { NormalizedRoi } from "../../lib/on-device/types.ts";

type ExpectedProfile = {
  filename: string;
  id: string;
  label: string;
  roi: NormalizedRoi;
};

const CANONICAL_PROFILES: ExpectedProfile[] = [
  {
    filename: "beach-source-02.mp4",
    id: "beach-source-02",
    label: "Known beach camera · beach-source-02",
    roi: { x: 0.02, y: 0.12, width: 0.96, height: 0.86 },
  },
  {
    filename: "beach-source-01.mp4",
    id: "beach-source-01",
    label: "Known beach camera · beach-source-01",
    roi: { x: 0.02, y: 0.12, width: 0.96, height: 0.86 },
  },
  {
    filename: "grass-source-04.mp4",
    id: "grass-source-04",
    label: "Known grass camera · grass-source-04",
    roi: { x: 0.02, y: 0.22, width: 0.96, height: 0.76 },
  },
  {
    filename: "grass-source-01.mp4",
    id: "grass-source-01",
    label: "Known grass camera · grass-source-01",
    roi: { x: 0.02, y: 0.18, width: 0.96, height: 0.8 },
  },
  {
    filename: "grass-source-09.mp4",
    id: "grass-source-09",
    label: "Known grass camera · grass-source-09",
    roi: { x: 0.02, y: 0.18, width: 0.96, height: 0.8 },
  },
  {
    filename: "grass-source-10.mp4",
    id: "grass-source-10",
    label: "Known grass camera · grass-source-10",
    roi: { x: 0.02, y: 0.22, width: 0.96, height: 0.76 },
  },
  {
    filename: "indoor-source-01.mp4",
    id: "indoor-source-01",
    label: "Known indoor camera · indoor-source-01",
    roi: { x: 0.04, y: 0.14, width: 0.92, height: 0.84 },
  },
  {
    filename: "indoor-source-07.mp4",
    id: "indoor-source-07",
    label: "Known indoor camera · indoor-source-07",
    roi: { x: 0.04, y: 0.14, width: 0.92, height: 0.84 },
  },
  {
    filename: "indoor-source-05.mp4",
    id: "indoor-source-05",
    label: "Known indoor camera · indoor-source-05",
    roi: { x: 0.03, y: 0.12, width: 0.94, height: 0.86 },
  },
];

test("all nine canonical proxy filenames resolve to their known ROI profiles", () => {
  assert.equal(roiProfiles.length, CANONICAL_PROFILES.length);

  for (const expected of CANONICAL_PROFILES) {
    const profile = {
      id: expected.id,
      label: expected.label,
      roi: expected.roi,
      source: "known-recording" as const,
    };
    assert.deepEqual(inferRoiProfile(expected.filename), profile);
    assert.deepEqual(
      roiProfiles.find(({ id }) => id === expected.id),
      profile,
    );
  }
});
