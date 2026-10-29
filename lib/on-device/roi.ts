import type { NormalizedRoi, RoiProfile } from "./types";

const INDOOR_DEFAULT: NormalizedRoi = {
  x: 0.03,
  y: 0.12,
  width: 0.94,
  height: 0.86,
};

const KNOWN_PROFILES: ReadonlyArray<{
  videoId: string;
  label: string;
  roi: NormalizedRoi;
}> = [
  {
    videoId: "beach-source-02",
    label: "Known beach camera · beach-source-02",
    roi: { x: 0.02, y: 0.12, width: 0.96, height: 0.86 },
  },
  {
    videoId: "beach-source-01",
    label: "Known beach camera · beach-source-01",
    roi: { x: 0.02, y: 0.12, width: 0.96, height: 0.86 },
  },
  {
    videoId: "grass-source-04",
    label: "Known grass camera · grass-source-04",
    roi: { x: 0.02, y: 0.22, width: 0.96, height: 0.76 },
  },
  {
    videoId: "grass-source-01",
    label: "Known grass camera · grass-source-01",
    roi: { x: 0.02, y: 0.18, width: 0.96, height: 0.8 },
  },
  {
    videoId: "grass-source-09",
    label: "Known grass camera · grass-source-09",
    roi: { x: 0.02, y: 0.18, width: 0.96, height: 0.8 },
  },
  {
    videoId: "grass-source-10",
    label: "Known grass camera · grass-source-10",
    roi: { x: 0.02, y: 0.22, width: 0.96, height: 0.76 },
  },
  {
    videoId: "indoor-source-01",
    label: "Known indoor camera · indoor-source-01",
    roi: { x: 0.04, y: 0.14, width: 0.92, height: 0.84 },
  },
  {
    videoId: "indoor-source-07",
    label: "Known indoor camera · indoor-source-07",
    roi: { x: 0.04, y: 0.14, width: 0.92, height: 0.84 },
  },
  {
    videoId: "indoor-source-05",
    label: "Known indoor camera · indoor-source-05",
    roi: { x: 0.03, y: 0.12, width: 0.94, height: 0.86 },
  },
];

function cloneRoi(roi: NormalizedRoi): NormalizedRoi {
  return { ...roi };
}

export function inferRoiProfile(filename: string): RoiProfile {
  const known = KNOWN_PROFILES.find(({ videoId }) => filename.includes(videoId));
  if (known) {
    return {
      id: known.videoId,
      label: known.label,
      roi: cloneRoi(known.roi),
      source: "known-recording",
    };
  }
  return {
    id: "indoor-camera-default",
    label: "Indoor camera default (adjust if needed)",
    roi: cloneRoi(INDOOR_DEFAULT),
    source: "camera-default",
  };
}

export const roiProfiles = KNOWN_PROFILES.map(({ videoId, label, roi }) => ({
  id: videoId,
  label,
  roi: cloneRoi(roi),
  source: "known-recording" as const,
}));
