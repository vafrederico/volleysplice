import { runtimeAssetUrl } from "../runtime-assets.ts";

// Updated by scripts/build-libswresample-wasm.sh together with the checked-in binary.
export const LIBSWRESAMPLE_WASM_SHA256 =
  "c7ed95ed8b6f5e11ea9e86262214b978bd145a6ec1f648dd56616af298449f90";
export const LIBSWRESAMPLE_VERSION = "FFmpeg 7.1.5 / libswresample 5.3.100";

type LibswresampleModule = {
  HEAPF32: Float32Array;
  HEAP16: Int16Array;
  _malloc(byteLength: number): number;
  _free(pointer: number): void;
  _vc_swr_init(inputSampleRate: number, inputChannels: number): number;
  _vc_swr_get_out_samples(inputFrames: number): number;
  _vc_swr_convert(
    inputPlane0: number,
    inputPlane1: number,
    inputFrames: number,
    output: number,
    outputCapacity: number,
  ): number;
  _vc_swr_close(): void;
};

type LibswresampleFactory = (options: {
  wasmBinary: ArrayBuffer;
  noInitialRun?: boolean;
}) => Promise<LibswresampleModule>;

function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join("");
}

async function fetchVerifiedWasm(): Promise<ArrayBuffer> {
  const response = await fetch(runtimeAssetUrl("libswresample.wasm"), {
    cache: "force-cache",
  });
  if (!response.ok) {
    throw new Error(`Could not load libswresample WASM (${response.status}).`);
  }
  const binary = await response.arrayBuffer();
  const digest = bytesToHex(new Uint8Array(await crypto.subtle.digest("SHA-256", binary)));
  if (digest !== LIBSWRESAMPLE_WASM_SHA256) {
    throw new Error(
      `libswresample WASM integrity check failed (expected ${LIBSWRESAMPLE_WASM_SHA256}, got ${digest}).`,
    );
  }
  return binary;
}

let assetsPromise: Promise<{
  binary: ArrayBuffer;
  factory: LibswresampleFactory;
}> | null = null;

async function loadModule(): Promise<LibswresampleModule> {
  if (!assetsPromise) {
    assetsPromise = (async () => {
      const binary = await fetchVerifiedWasm();
      const imported = (await import(
        /* @vite-ignore */ runtimeAssetUrl("libswresample.mjs")
      )) as {
        default?: LibswresampleFactory;
      };
      if (typeof imported.default !== "function") {
        throw new Error("libswresample WASM loader did not export a module factory.");
      }
      return { binary, factory: imported.default };
    })().catch((error) => {
      assetsPromise = null;
      throw error;
    });
  }
  const { binary, factory } = await assetsPromise;
  return factory({ wasmBinary: binary, noInitialRun: true });
}

function checkResult(result: number, operation: string): number {
  if (result < 0) throw new Error(`libswresample ${operation} failed with code ${result}.`);
  return result;
}

/** A single-stream, fail-closed FLTP source -> mono 16 kHz S16 resampler. */
export class LibswresampleWasmResampler {
  private readonly module: LibswresampleModule;
  private inputPointers: [number, number] = [0, 0];
  private inputCapacityFrames = 0;
  private outputPointer = 0;
  private outputCapacityFrames = 0;
  private configured = false;
  private channels = 0;
  private finished = false;

  private constructor(module: LibswresampleModule) {
    this.module = module;
  }

  static async create(): Promise<LibswresampleWasmResampler> {
    return new LibswresampleWasmResampler(await loadModule());
  }

  configure(inputSampleRate: number, inputChannels: number): void {
    if (this.finished) throw new Error("Cannot configure a finished libswresample stream.");
    if (this.configured) {
      if (this.channels !== inputChannels) {
        throw new Error("Audio channel count changed within the source track.");
      }
      return;
    }
    checkResult(this.module._vc_swr_init(inputSampleRate, inputChannels), "initialization");
    this.channels = inputChannels;
    this.configured = true;
  }

  push(planes: readonly Float32Array[]): Int16Array {
    if (!this.configured) throw new Error("libswresample stream is not configured.");
    if (this.finished) throw new Error("Cannot push to a finished libswresample stream.");
    if (planes.length !== this.channels || planes.length < 1 || planes.length > 2) {
      throw new Error("libswresample expects the configured mono or stereo planar input.");
    }
    const inputFrames = planes[0].length;
    if (planes.some((plane) => plane.length !== inputFrames)) {
      throw new Error("libswresample input planes have different frame counts.");
    }
    if (inputFrames === 0) return new Int16Array(0);
    this.ensureInputCapacity(inputFrames);
    for (let channel = 0; channel < planes.length; channel += 1) {
      this.module.HEAPF32.set(planes[channel], this.inputPointers[channel] / 4);
    }
    const outputCapacity = checkResult(
      this.module._vc_swr_get_out_samples(inputFrames),
      "output sizing",
    );
    this.ensureOutputCapacity(Math.max(1, outputCapacity));
    const outputFrames = checkResult(
      this.module._vc_swr_convert(
        this.inputPointers[0],
        planes.length === 2 ? this.inputPointers[1] : 0,
        inputFrames,
        this.outputPointer,
        this.outputCapacityFrames,
      ),
      "conversion",
    );
    return this.copyOutput(outputFrames);
  }

  flush(): Int16Array {
    if (this.finished) return new Int16Array(0);
    this.finished = true;
    if (!this.configured) return new Int16Array(0);
    const chunks: Int16Array[] = [];
    let total = 0;
    for (;;) {
      const estimated = checkResult(this.module._vc_swr_get_out_samples(0), "flush sizing");
      this.ensureOutputCapacity(Math.max(32, estimated));
      const frames = checkResult(
        this.module._vc_swr_convert(0, 0, 0, this.outputPointer, this.outputCapacityFrames),
        "flush",
      );
      if (frames === 0) break;
      const chunk = this.copyOutput(frames);
      chunks.push(chunk);
      total += chunk.length;
    }
    const output = new Int16Array(total);
    let offset = 0;
    for (const chunk of chunks) {
      output.set(chunk, offset);
      offset += chunk.length;
    }
    return output;
  }

  close(): void {
    if (this.configured) this.module._vc_swr_close();
    for (const pointer of this.inputPointers) {
      if (pointer) this.module._free(pointer);
    }
    if (this.outputPointer) this.module._free(this.outputPointer);
    this.inputPointers = [0, 0];
    this.outputPointer = 0;
    this.inputCapacityFrames = 0;
    this.outputCapacityFrames = 0;
    this.configured = false;
    this.finished = true;
  }

  private ensureInputCapacity(frames: number): void {
    if (frames <= this.inputCapacityFrames) return;
    for (let channel = 0; channel < this.channels; channel += 1) {
      if (this.inputPointers[channel]) this.module._free(this.inputPointers[channel]);
      this.inputPointers[channel] = this.module._malloc(frames * Float32Array.BYTES_PER_ELEMENT);
      if (!this.inputPointers[channel]) throw new Error("libswresample input allocation failed.");
    }
    this.inputCapacityFrames = frames;
  }

  private ensureOutputCapacity(frames: number): void {
    if (frames <= this.outputCapacityFrames) return;
    if (this.outputPointer) this.module._free(this.outputPointer);
    this.outputPointer = this.module._malloc(frames * Int16Array.BYTES_PER_ELEMENT);
    if (!this.outputPointer) throw new Error("libswresample output allocation failed.");
    this.outputCapacityFrames = frames;
  }

  private copyOutput(frames: number): Int16Array {
    return new Int16Array(
      this.module.HEAP16.subarray(
        this.outputPointer / Int16Array.BYTES_PER_ELEMENT,
        this.outputPointer / Int16Array.BYTES_PER_ELEMENT + frames,
      ),
    );
  }
}
