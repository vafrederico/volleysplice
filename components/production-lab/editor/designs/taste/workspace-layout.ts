export type WorkspaceLayout = { left?: number; right?: number; video?: number };

export function readLayout(raw: string | null): WorkspaceLayout {
  try {
    const data: unknown = JSON.parse(raw ?? "{}");
    if (!data || typeof data !== "object" || Array.isArray(data)) return {};
    const result: WorkspaceLayout = {};
    for (const key of ["left", "right", "video"] as const) {
      const value = (data as Record<string, unknown>)[key];
      if (typeof value === "number" && Number.isFinite(value) && value > 0)
        result[key] = value;
    }
    return result;
  } catch {
    return {};
  }
}

// All dimensions are CSS pixels, so OS scaling and browser zoom are already included.
// Clamp the displayed layout without overwriting preferences saved on a larger screen.
export function clampLayout(
  saved: WorkspaceLayout,
  width: number,
  height: number,
) {
  const threeColumns = width > 1320;
  const leftDefault = width > 1600 ? 310 : width > 1320 ? 285 : 275;
  const rightDefault = width > 1600 ? 340 : 310;
  const clamp = (value: number, min: number, max: number) =>
    Math.max(min, Math.min(max, value));
  const leftMax = Math.max(
    240,
    Math.min(480, width - 480 - (threeColumns ? 260 : 0)),
  );
  const left = clamp(saved.left ?? leftDefault, 240, leftMax);
  const rightMax = Math.max(260, Math.min(480, width - left - 480));
  const right = clamp(saved.right ?? rightDefault, 260, rightMax);
  // Manual sizing is independent of the viewport; the timeline may scroll below it.
  const videoMax = 4096;
  const videoWidth = Math.max(
    0,
    width -
      (width > 960 ? left : 0) -
      (threeColumns ? right : 0) -
      (width > 1600 ? 44 : width > 767 ? 36 : 28),
  );
  const video =
    saved.video !== undefined
      ? clamp(saved.video, 180, videoMax)
      : clamp((videoWidth * 9) / 16, 180, Math.max(180, height * 0.7));
  return {
    left,
    right,
    video,
    leftMax: Math.max(
      240,
      Math.min(leftMax, width - 480 - (threeColumns ? right : 0)),
    ),
    rightMax,
    videoMax,
  };
}

// Reserve the measured controls/timeline height plus breathing room below the legend.
export function fittedVideoHeight(
  width: number,
  viewportHeight: number,
  overhead: number,
) {
  return Math.max(
    180,
    Math.min(
      (width * 9) / 16,
      viewportHeight * 0.7,
      viewportHeight - overhead - 16,
    ),
  );
}
