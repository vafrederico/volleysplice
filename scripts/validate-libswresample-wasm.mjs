#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const repository = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const modulePath = resolve(
  repository,
  "vendor/libswresample-wasm/dist/libswresample.mjs",
);
const wasmPath = resolve(
  repository,
  "vendor/libswresample-wasm/dist/libswresample.wasm",
);

const { default: createModule } = await import(pathToFileURL(modulePath));
const wasmBinary = await readFile(wasmPath);

function deterministicStereo(frameCount) {
  const planes = [new Float32Array(frameCount), new Float32Array(frameCount)];
  let random = 0x9e3779b9;
  for (let frame = 0; frame < frameCount; frame += 1) {
    random = (Math.imul(random, 1664525) + 1013904223) >>> 0;
    const noise = (random / 0xffffffff - 0.5) * 0.02;
    const seconds = frame / 48000;
    planes[0][frame] =
      0.45 * Math.sin(2 * Math.PI * 997 * seconds) +
      0.16 * Math.sin(2 * Math.PI * 12000 * seconds) +
      noise;
    planes[1][frame] =
      0.35 * Math.sin(2 * Math.PI * 2237 * seconds) -
      0.14 * Math.sin(2 * Math.PI * 15000 * seconds) -
      noise;
  }
  planes[0][0] = 0.9;
  planes[1][0] = -0.7;
  planes[0][frameCount - 1] = -0.8;
  planes[1][frameCount - 1] = 0.6;
  return planes;
}

function interleave(planes) {
  const output = new Float32Array(planes[0].length * planes.length);
  for (let frame = 0; frame < planes[0].length; frame += 1) {
    for (let channel = 0; channel < planes.length; channel += 1) {
      output[frame * planes.length + channel] = planes[channel][frame];
    }
  }
  return output;
}

function nativeReference(planes) {
  const input = interleave(planes);
  const result = spawnSync(
    "ffmpeg",
    [
      "-hide_banner",
      "-loglevel",
      "error",
      "-f",
      "f32le",
      "-ar",
      "48000",
      "-ac",
      String(planes.length),
      "-i",
      "pipe:0",
      "-map",
      "0:a:0",
      "-vn",
      "-ac",
      "1",
      "-ar",
      "16000",
      "-f",
      "s16le",
      "pipe:1",
    ],
    {
      input: Buffer.from(input.buffer, input.byteOffset, input.byteLength),
      maxBuffer: 64 * 1024 * 1024,
    },
  );
  if (result.status !== 0) {
    throw new Error(`Native FFmpeg failed: ${result.stderr.toString("utf8")}`);
  }
  return new Int16Array(
    result.stdout.buffer,
    result.stdout.byteOffset,
    result.stdout.byteLength / Int16Array.BYTES_PER_ELEMENT,
  ).slice();
}

async function wasmOutput(planes, chunks) {
  const wasmModule = await createModule({ wasmBinary, noInitialRun: true });
  const result = wasmModule._vc_swr_init(48000, planes.length);
  if (result < 0) throw new Error(`WASM initialization failed: ${result}`);
  const output = [];
  let offset = 0;
  for (const requested of chunks) {
    if (offset >= planes[0].length) break;
    const frames = Math.min(requested, planes[0].length - offset);
    const pointers = planes.map((plane) => {
      const pointer = wasmModule._malloc(frames * Float32Array.BYTES_PER_ELEMENT);
      wasmModule.HEAPF32.set(plane.subarray(offset, offset + frames), pointer / 4);
      return pointer;
    });
    const capacity = Math.max(1, wasmModule._vc_swr_get_out_samples(frames));
    const outputPointer = wasmModule._malloc(capacity * Int16Array.BYTES_PER_ELEMENT);
    const converted = wasmModule._vc_swr_convert(
      pointers[0],
      pointers[1] ?? 0,
      frames,
      outputPointer,
      capacity,
    );
    if (converted < 0) throw new Error(`WASM conversion failed: ${converted}`);
    output.push(...wasmModule.HEAP16.subarray(outputPointer / 2, outputPointer / 2 + converted));
    for (const pointer of pointers) wasmModule._free(pointer);
    wasmModule._free(outputPointer);
    offset += frames;
  }
  if (offset !== planes[0].length) {
    throw new Error(`Chunk plan consumed ${offset}/${planes[0].length} input frames.`);
  }
  for (;;) {
    const capacity = Math.max(32, wasmModule._vc_swr_get_out_samples(0));
    const outputPointer = wasmModule._malloc(capacity * Int16Array.BYTES_PER_ELEMENT);
    const converted = wasmModule._vc_swr_convert(0, 0, 0, outputPointer, capacity);
    if (converted < 0) throw new Error(`WASM flush failed: ${converted}`);
    output.push(...wasmModule.HEAP16.subarray(outputPointer / 2, outputPointer / 2 + converted));
    wasmModule._free(outputPointer);
    if (converted === 0) break;
  }
  wasmModule._vc_swr_close();
  return Int16Array.from(output);
}

function compare(reference, actual) {
  const length = Math.min(reference.length, actual.length);
  let equal = 0;
  let absoluteTotal = 0;
  let squaredTotal = 0;
  let maximum = 0;
  for (let index = 0; index < length; index += 1) {
    const difference = Math.abs(reference[index] - actual[index]);
    if (difference === 0) equal += 1;
    absoluteTotal += difference;
    squaredTotal += difference * difference;
    maximum = Math.max(maximum, difference);
  }
  return {
    referenceSamples: reference.length,
    actualSamples: actual.length,
    equalSamples: equal,
    equalPercent: (100 * equal) / Math.max(1, length),
    meanAbsoluteLsb: absoluteTotal / Math.max(1, length),
    rootMeanSquareLsb: Math.sqrt(squaredTotal / Math.max(1, length)),
    maximumAbsoluteLsb: maximum,
  };
}

function pcmSha256(samples) {
  return createHash("sha256")
    .update(Buffer.from(samples.buffer, samples.byteOffset, samples.byteLength))
    .digest("hex");
}

const planes = deterministicStereo(48000 + 137);
const native = nativeReference(planes);
const single = await wasmOutput(planes, [planes[0].length]);
const irregular = await wasmOutput(
  planes,
  Array.from({ length: 1000 }, (_, index) => [1, 7, 31, 257, 1024, 4093][index % 6]),
);
const chunkInvariant =
  single.length === irregular.length && single.every((value, index) => value === irregular[index]);
const comparison = compare(native, irregular);
console.log(
  JSON.stringify(
    {
      chunkInvariant,
      nativeSha256: pcmSha256(native),
      wasmSha256: pcmSha256(irregular),
      ...comparison,
    },
    null,
    2,
  ),
);
if (!chunkInvariant || native.length !== irregular.length || comparison.maximumAbsoluteLsb > 2) {
  process.exitCode = 1;
}
