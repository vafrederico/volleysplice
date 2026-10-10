/** Display-light compositing for BT.2020 HLG/PQ, with 203-nit graphics white. */
export type HdrTransfer = "hlg" | "pq";
export type Rgb = readonly [number, number, number];
const kr = 0.2627;
const kb = 0.0593;
const kg = 1 - kr - kb;
const clamp = (v: number, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));
const luma = (v: Rgb) => kr * v[0] + kg * v[1] + kb * v[2];
const m1 = 2610 / 16384, m2 = 2523 / 32;
const c1 = 3424 / 4096, c2 = 2413 / 128, c3 = 2392 / 128;

export function hdrToNits(rgb: Rgb, transfer: HdrTransfer): Rgb {
  if (transfer === "pq") return rgb.map(value => {
    const p = clamp(value) ** (1 / m2);
    return 10000 * (Math.max(p - c1, 0) / (c2 - c3 * p)) ** (1 / m1);
  }) as unknown as Rgb;
  const scene = rgb.map(value => {
    const v = clamp(value);
    return v <= 0.5 ? v * v / 3 : (Math.exp((v - 0.55991073) / 0.17883277) + 0.28466892) / 12;
  }) as unknown as Rgb;
  const gain = 1000 * luma(scene) ** 0.2;
  return [scene[0] * gain, scene[1] * gain, scene[2] * gain];
}

export function nitsToHdr(rgb: Rgb, transfer: HdrTransfer): Rgb {
  if (transfer === "pq") return rgb.map(value => {
    const p = clamp(value / 10000) ** m1;
    return ((c1 + c2 * p) / (1 + c3 * p)) ** m2;
  }) as unknown as Rgb;
  const gain = 1000 * Math.max(0, luma(rgb) / 1000) ** (1 / 6);
  if (!gain) return [0, 0, 0];
  return rgb.map(value => {
    const v = clamp(value / gain);
    return v <= 1 / 12 ? Math.sqrt(3 * v) : 0.17883277 * Math.log(12 * v - 0.28466892) + 0.55991073;
  }) as unknown as Rgb;
}

export function srgbToNits(rgb: Rgb): Rgb {
  const [r, g, b] = rgb.map(v => v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);
  return [
    203 * (0.627403896 * r + 0.329283038 * g + 0.043313066 * b),
    203 * (0.069097289 * r + 0.919540395 * g + 0.011362316 * b),
    203 * (0.016391439 * r + 0.088013308 * g + 0.895595253 * b),
  ];
}

export function rgbToYuv(rgb: Rgb): Rgb {
  const y = luma(rgb);
  return [y, (rgb[2] - y) / (2 * (1 - kb)), (rgb[0] - y) / (2 * (1 - kr))];
}

export function yuvToRgb(y: number, u: number, v: number): Rgb {
  const r = y + 2 * (1 - kr) * v;
  const b = y + 2 * (1 - kb) * u;
  return [r, (y - kr * r - kb * b) / kg, b];
}

export type HdrMask = {
  width: number;
  height: number;
  data: Uint8ClampedArray;
};

/** Only covered 2x2 blocks are changed; untouched video remains bit-exact. */
export function compositeHdrScore(
  pixels: Uint16Array, width: number, height: number,
  mask: HdrMask, rotation: number, displayWidth: number, displayHeight: number,
  transfer: HdrTransfer, fullRange: boolean,
): void {
  if (![0, 90, 180, 270].includes(rotation)) throw new Error("Unsupported HDR rotation.");
  const cw = Math.ceil(width / 2), ch = Math.ceil(height / 2);
  const uOffset = width * height, vOffset = uOffset + cw * ch;
  if (pixels.length !== vOffset + cw * ch || mask.data.length !== mask.width * mask.height * 4) {
    throw new Error("Invalid HDR frame or score mask size.");
  }
  const yOffset = fullRange ? 0 : 64, yScale = fullRange ? 1023 : 876;
  const cScale = fullRange ? 1023 : 896;
  const palette = new Map<number, { nits: Rgb; yuv: Rgb }>();
  // Map display-space score pixels back into the unrotated encoded frame.
  const rotatedWidth = rotation % 180 ? height : width;
  const rotatedHeight = rotation % 180 ? width : height;
  const strip = Math.min(rotatedHeight, Math.ceil(mask.height * rotatedHeight / displayHeight));
  const xStart = rotation === 270 ? Math.max(0, width - strip) & ~1 : 0;
  const xEnd = rotation === 90 ? Math.min(width, strip + 1) : width;
  const yStart = rotation === 180 ? Math.max(0, height - strip) & ~1 : 0;
  const yEnd = rotation === 0 ? Math.min(height, strip + 1) : height;
  for (let by = yStart; by < yEnd; by += 2) {
    for (let bx = xStart; bx < xEnd; bx += 2) {
      const ci = (by >> 1) * cw + (bx >> 1);
      const originalU = (pixels[uOffset + ci] - 512) / cScale;
      const originalV = (pixels[vOffset + ci] - 512) / cScale;
      let deltaU = 0, deltaV = 0, covered = false, count = 0;
      for (let dy = 0; dy < 2 && by + dy < height; dy++) {
        for (let dx = 0; dx < 2 && bx + dx < width; dx++) {
          count++;
          const x = bx + dx, y = by + dy;
          const rx = rotation === 90 ? height - 1 - y : rotation === 180 ? width - 1 - x : rotation === 270 ? y : x;
          const ry = rotation === 90 ? x : rotation === 180 ? height - 1 - y : rotation === 270 ? width - 1 - x : y;
          const mx = Math.floor((rx + 0.5) * displayWidth / rotatedWidth);
          const my = Math.floor((ry + 0.5) * displayHeight / rotatedHeight);
          if (mx >= mask.width || my >= mask.height) continue;
          const mi = (my * mask.width + mx) * 4;
          const alpha = mask.data[mi + 3] / 255;
          if (!alpha) continue;
          const key = (mask.data[mi] << 16) | (mask.data[mi + 1] << 8) | mask.data[mi + 2];
          let color = palette.get(key);
          if (!color) {
            const nits = srgbToNits([mask.data[mi] / 255, mask.data[mi + 1] / 255, mask.data[mi + 2] / 255]);
            color = { nits, yuv: rgbToYuv(nitsToHdr(nits, transfer)) };
            palette.set(key, color);
          }
          const yi = y * width + x;
          let result = color.yuv;
          if (alpha < 1) {
            const background = hdrToNits(yuvToRgb((pixels[yi] - yOffset) / yScale, originalU, originalV), transfer);
            const blended = color.nits.map((v, i) => v * alpha + background[i] * (1 - alpha)) as unknown as Rgb;
            result = rgbToYuv(nitsToHdr(blended, transfer));
          }
          pixels[yi] = Math.round(clamp(yOffset + yScale * result[0], 0, 1023));
          deltaU += result[1] - originalU;
          deltaV += result[2] - originalV;
          covered = true;
        }
      }
      if (covered) {
        pixels[uOffset + ci] = Math.round(clamp(512 + cScale * (originalU + deltaU / count), 0, 1023));
        pixels[vOffset + ci] = Math.round(clamp(512 + cScale * (originalV + deltaV / count), 0, 1023));
      }
    }
  }
}
