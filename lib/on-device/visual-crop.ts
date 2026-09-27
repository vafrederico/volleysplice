import type { NormalizedRoi } from "./types";

function roundEven(value: number): number {
  const lower = Math.floor(value);
  const fraction = value - lower;
  return lower + Number(fraction > 0.5 || (fraction === 0.5 && lower % 2 !== 0));
}

/** Match the Python crop before resizing; include at least one valid pixel. */
export function visualCrop(roi: NormalizedRoi, width: number, height: number, areaContract = true) {
  if (!areaContract) {
    const left = Math.round(roi.x * width), top = Math.round(roi.y * height);
    return { left, top, width: Math.max(1, Math.round((roi.x + roi.width) * width) - left),
      height: Math.max(1, Math.round((roi.y + roi.height) * height) - top) };
  }
  const left = Math.min(width - 1, Math.max(0, roundEven(roi.x * width)));
  const top = Math.min(height - 1, Math.max(0, roundEven(roi.y * height)));
  const right = Math.min(width, Math.max(left + 1, roundEven((roi.x + roi.width) * width)));
  const bottom = Math.min(height, Math.max(top + 1, roundEven((roi.y + roi.height) * height)));
  return { left, top, width: right - left, height: bottom - top };
}
