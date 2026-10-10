import assert from "node:assert/strict";
import { test } from "node:test";
import { compositeHdrScore, hdrToNits, nitsToHdr, srgbToNits, type Rgb } from "../../prod/src/lib/on-device/hdr-color.ts";
import { validateDolbyVisionRecord, validateHdrContainer, validateHdrMovieMetadata } from "../../prod/src/lib/on-device/hdr-container.ts";

const close = (actual: number, expected: number, tolerance = 1e-4) => assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);

test("HLG and PQ match reference white and luminance anchors", () => {
  close(hdrToNits([0.75, 0.75, 0.75], "hlg")[0], 203.152, 0.01);
  close(hdrToNits([1, 1, 1], "hlg")[0], 1000, 0.01);
  close(nitsToHdr([100, 100, 100], "pq")[0], 0.5080784215);
  close(nitsToHdr([1000, 1000, 1000], "pq")[0], 0.7518270962);
  for (const transfer of ["hlg", "pq"] as const) {
    for (const rgb of [[0, 0, 0], [0.15, 0.5, 0.8], [0.9, 0.3, 0.4], [1, 1, 1]] as Rgb[]) {
      hdrToNits(nitsToHdr(hdrToNits(rgb, transfer), transfer), transfer).forEach((v, i) => close(v, hdrToNits(rgb, transfer)[i], 0.001));
    }
  }
});

test("sRGB graphics white is 203 nits, red is converted into BT.2020", () => {
  srgbToNits([1, 1, 1]).forEach(v => close(v, 203));
  const red = srgbToNits([1, 0, 0]);
  close(red[0], 127.362991, 0.001);
  close(red[1], 14.02675, 0.001);
  close(red[2], 3.32746, 0.001);
});

function frame(width = 4, height = 4) {
  const result = new Uint16Array(width * height + 2 * Math.ceil(width / 2) * Math.ceil(height / 2));
  result.fill(64, 0, width * height); result.fill(512, width * height);
  return result;
}
test("transparent masks preserve every video bit, including superwhite codes", () => {
  const pixels = frame(); pixels[4] = 1023; pixels[17] = 12;
  const original = pixels.slice();
  compositeHdrScore(pixels, 4, 4, { width: 4, height: 2, data: new Uint8ClampedArray(32) }, 0, 4, 4, "hlg", false);
  assert.deepEqual(pixels, original);
});

test("opaque/half-alpha white composites in light, with full and limited range", () => {
  for (const transfer of ["hlg", "pq"] as const) for (const full of [false, true]) {
    for (const alpha of [255, 128]) {
      const pixels = frame(); pixels.fill(full ? 0 : 64, 0, 16);
      const data = new Uint8ClampedArray(32);
      for (let i = 0; i < 8; i++) data.set([255, 255, 255, alpha], i * 4);
      compositeHdrScore(pixels, 4, 4, { width: 4, height: 2, data }, 0, 4, 4, transfer, full);
      const encoded = (pixels[0] - (full ? 0 : 64)) / (full ? 1023 : 876);
      close(hdrToNits([encoded, encoded, encoded], transfer)[0], 203 * alpha / 255, 1.5);
      assert.deepEqual([...pixels.slice(8, 16)], Array(8).fill(full ? 0 : 64));
      assert.deepEqual([...pixels.slice(16)], Array(8).fill(512));
    }
  }
});

test("score coordinates follow all four rotations while retaining coded video orientation", () => {
  const data = new Uint8ClampedArray(32); data.set([255, 255, 255, 255], 0);
  for (const [rotation, index] of [[0, 0], [90, 12], [180, 15], [270, 3]]) {
    const pixels = frame();
    compositeHdrScore(pixels, 4, 4, { width: 4, height: 2, data }, rotation, 4, 4, "hlg", false);
    const changed = [...pixels.slice(0, 16)].flatMap((v, i) => v === 64 ? [] : [i]);
    assert.deepEqual(changed, [index]);
  }
});

test("partial coverage updates only the affected shared chroma sample", () => {
  const pixels = frame();
  const data = new Uint8ClampedArray(32); data.set([255, 0, 0, 255], 0);
  compositeHdrScore(pixels, 4, 4, { width: 4, height: 2, data }, 0, 4, 4, "hlg", false);
  assert.notEqual(pixels[16], 512); assert.notEqual(pixels[20], 512);
  assert.deepEqual([...pixels.slice(17, 20)], [512, 512, 512]);
  assert.deepEqual([...pixels.slice(21)], [512, 512, 512]);
  assert.deepEqual([...pixels.slice(1, 16)], Array(15).fill(64));
});

const record = (profile: number, compatibility: number) => Uint8Array.from([1, 0, profile << 1, 5, compatibility << 4]);
test("only compatible Dolby Vision base layers can be exported as HLG/PQ", () => {
  validateDolbyVisionRecord(record(8, 4), "hlg");
  validateDolbyVisionRecord(record(8, 1), "pq");
  for (const [profile, compatibility, transfer] of [[5, 0, "pq"], [7, 1, "pq"], [8, 4, "pq"], [8, 2, "hlg"]] as const) {
    assert.throws(() => validateDolbyVisionRecord(record(profile, compatibility), transfer), /profile/);
  }
  assert.throws(() => validateDolbyVisionRecord(new Uint8Array(3), "hlg"), /Truncated/);
});

function box(type: string, data: Uint8Array): Uint8Array {
  const bytes = new Uint8Array(8 + data.length);
  new DataView(bytes.buffer).setUint32(0, bytes.length);
  bytes.set(new TextEncoder().encode(type), 4); bytes.set(data, 8); return bytes;
}
test("movie scanner checks nested Dolby metadata and skips media without reading it", async () => {
  const entry = new Uint8Array(78 + 13); entry.set(box("dvcC", record(8, 4)), 78);
  const stsd = new Uint8Array(8 + entry.length + 8); stsd.set(box("hvc1", entry), 8);
  const moov = box("moov", box("trak", box("mdia", box("minf", box("stbl", box("stsd", stsd))))));
  validateHdrMovieMetadata(moov, "hlg");
  assert.throws(() => validateHdrMovieMetadata(moov, "pq"), /profile/);
  await validateHdrContainer(new Blob([box("ftyp", new Uint8Array(8)), box("mdat", new Uint8Array(1024)), moov]), "hlg");
  await assert.rejects(validateHdrContainer(new Blob([box("mdat", new Uint8Array(8))]), "hlg"), /missing/);
  assert.throws(() => validateHdrMovieMetadata(moov.slice(0, -1), "hlg"), /size/);
});
