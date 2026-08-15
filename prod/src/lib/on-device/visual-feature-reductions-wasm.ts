import { ANALYSIS_HEIGHT, ANALYSIS_WIDTH, FRAME_FEATURE_NAMES } from "./feature-schema.ts";
import { runtimeAssetUrl } from "../runtime-assets";
import type { VisualReductionInput } from "./visual-feature-reductions.ts";

const PIXELS = ANALYSIS_WIDTH * ANALYSIS_HEIGHT;

type ReductionExports = {
  memory: WebAssembly.Memory;
  grayPointer(): number;
  hsvPointer(): number;
  edgesPointer(): number;
  laplacianPointer(): number;
  gradientXPointer(): number;
  gradientYPointer(): number;
  previousGrayPointer(): number;
  flowPointer(): number;
  outputPointer(): number;
  compute(hasPrevious: number, shiftX: number, shiftY: number, shiftResponse: number): void;
};

export class WasmVisualFeatureReducer {
  private readonly exports: ReductionExports;

  private constructor(exports: ReductionExports) {
    this.exports = exports;
  }

  static async load(
    url = runtimeAssetUrl("feature-reductions.wasm"),
  ): Promise<WasmVisualFeatureReducer> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Could not load the WASM feature reductions (${response.status}).`);
    }
    const bytes = await response.arrayBuffer();
    return WasmVisualFeatureReducer.instantiate(bytes);
  }

  static async instantiate(bytes: BufferSource): Promise<WasmVisualFeatureReducer> {
    const instance = await WebAssembly.instantiate(bytes, {
      env: {
        abort: () => {
          throw new Error("The WASM feature reduction kernel aborted.");
        },
      },
    });
    return new WasmVisualFeatureReducer(instance.instance.exports as ReductionExports);
  }

  reduce(input: VisualReductionInput): Float32Array {
    if (
      input.gray.length !== PIXELS ||
      input.hsv.length !== PIXELS * 3 ||
      input.edges.length !== PIXELS ||
      input.laplacian.length !== PIXELS ||
      input.gradientX.length !== PIXELS ||
      input.gradientY.length !== PIXELS ||
      (input.previousGray !== null && input.previousGray.length !== PIXELS) ||
      input.flow.length !== PIXELS * 2
    ) {
      throw new Error("WASM feature reductions received an unexpected analysis buffer size.");
    }
    const { memory } = this.exports;
    new Uint8Array(memory.buffer, this.exports.grayPointer(), PIXELS).set(input.gray);
    new Uint8Array(memory.buffer, this.exports.hsvPointer(), PIXELS * 3).set(input.hsv);
    new Uint8Array(memory.buffer, this.exports.edgesPointer(), PIXELS).set(input.edges);
    new Float32Array(memory.buffer, this.exports.laplacianPointer(), PIXELS).set(
      input.laplacian,
    );
    new Float32Array(memory.buffer, this.exports.gradientXPointer(), PIXELS).set(
      input.gradientX,
    );
    new Float32Array(memory.buffer, this.exports.gradientYPointer(), PIXELS).set(
      input.gradientY,
    );
    if (input.previousGray) {
      new Uint8Array(memory.buffer, this.exports.previousGrayPointer(), PIXELS).set(
        input.previousGray,
      );
    }
    new Float32Array(memory.buffer, this.exports.flowPointer(), PIXELS * 2).set(input.flow);
    this.exports.compute(
      input.previousGray ? 1 : 0,
      input.shiftX,
      input.shiftY,
      input.shiftResponse,
    );
    return new Float32Array(
      memory.buffer,
      this.exports.outputPointer(),
      FRAME_FEATURE_NAMES.length,
    ).slice();
  }
}
