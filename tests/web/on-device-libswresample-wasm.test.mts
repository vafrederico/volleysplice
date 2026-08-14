import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";

import {
  LIBSWRESAMPLE_WASM_SHA256,
} from "../../lib/on-device/libswresample-wasm.ts";

const MODULE_PATH = resolve("vendor/libswresample-wasm/dist/libswresample.mjs");
const WASM_PATH = resolve("vendor/libswresample-wasm/dist/libswresample.wasm");
const GLUE_SHA256 = "022782d1e08e483d8c67de30f68177eff5904998c410df028f26e342c793ae48";

type WasmModule = {
  HEAPF32: Float32Array;
  HEAP16: Int16Array;
  _malloc(bytes: number): number;
  _free(pointer: number): void;
  _vc_swr_init(rate: number, channels: number): number;
  _vc_swr_get_out_samples(frames: number): number;
  _vc_swr_convert(
    plane0: number,
    plane1: number,
    inputFrames: number,
    output: number,
    outputCapacity: number,
  ): number;
  _vc_swr_close(): void;
};

type WasmFactory = (options: { wasmBinary: Uint8Array }) => Promise<WasmModule>;

function sha256(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function stereoFixture(): readonly Float32Array[] {
  return [997, 2237].map((frequency) =>
    Float32Array.from(
      { length: 4800 },
      (_, frame) => 0.5 * Math.sin((2 * Math.PI * frequency * frame) / 48_000),
    ),
  );
}

async function resample(chunkSizes: readonly number[]): Promise<Int16Array> {
  const imported = (await import(pathToFileURL(MODULE_PATH).href)) as {
    default: WasmFactory;
  };
  const wasmBinary = await readFile(WASM_PATH);
  const wasmModule = await imported.default({ wasmBinary });
  assert.equal(wasmModule._vc_swr_init(48_000, 2), 0);
  const planes = stereoFixture();
  const output: number[] = [];
  let offset = 0;
  let chunkIndex = 0;
  while (offset < planes[0].length) {
    const frames = Math.min(
      chunkSizes[chunkIndex % chunkSizes.length],
      planes[0].length - offset,
    );
    const pointers = planes.map((plane) => {
      const pointer = wasmModule._malloc(frames * Float32Array.BYTES_PER_ELEMENT);
      wasmModule.HEAPF32.set(plane.subarray(offset, offset + frames), pointer / 4);
      return pointer;
    });
    const capacity = Math.max(1, wasmModule._vc_swr_get_out_samples(frames));
    const outputPointer = wasmModule._malloc(capacity * Int16Array.BYTES_PER_ELEMENT);
    const converted = wasmModule._vc_swr_convert(
      pointers[0],
      pointers[1],
      frames,
      outputPointer,
      capacity,
    );
    assert.ok(converted >= 0);
    output.push(...wasmModule.HEAP16.subarray(outputPointer / 2, outputPointer / 2 + converted));
    pointers.forEach((pointer) => wasmModule._free(pointer));
    wasmModule._free(outputPointer);
    offset += frames;
    chunkIndex += 1;
  }
  for (;;) {
    const capacity = Math.max(32, wasmModule._vc_swr_get_out_samples(0));
    const outputPointer = wasmModule._malloc(capacity * Int16Array.BYTES_PER_ELEMENT);
    const converted = wasmModule._vc_swr_convert(0, 0, 0, outputPointer, capacity);
    assert.ok(converted >= 0);
    output.push(...wasmModule.HEAP16.subarray(outputPointer / 2, outputPointer / 2 + converted));
    wasmModule._free(outputPointer);
    if (converted === 0) break;
  }
  wasmModule._vc_swr_close();
  return Int16Array.from(output);
}

test("checked-in libswresample browser assets match their pinned digests", async () => {
  assert.equal(sha256(await readFile(WASM_PATH)), LIBSWRESAMPLE_WASM_SHA256);
  assert.equal(sha256(await readFile(MODULE_PATH)), GLUE_SHA256);
});

test("libswresample WASM output is invariant to streaming chunk boundaries", async () => {
  const single = await resample([4800]);
  const irregular = await resample([1, 7, 31, 257, 1024]);
  assert.equal(single.length, 1600);
  assert.deepEqual(irregular, single);
  assert.ok(Math.max(...single.map(Math.abs)) > 1000);
});
