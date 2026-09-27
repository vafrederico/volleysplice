import { VideoSample } from "mediabunny";
import type * as Ort from "onnxruntime-web";
import { prepareImage } from "./neural-image";
import { decodeRallies, fuseFeatures, roundTokensToFloat16, FUSED_DIMENSION, type NeuralPipelineConfig } from "./neural-contract";
import { verifiedAsset, type NeuralBundle } from "./rally-model";
import type { NeuralWorkerRequest, NeuralWorkerResponse } from "./neural-worker-protocol";

type CvRuntime = typeof import("@techstark/opencv-js");
const scope = self as unknown as DedicatedWorkerGlobalScope;
let ort: typeof Ort;
let cv: CvRuntime;
let bundle: NeuralBundle;
let baseUrl: string;
let config: NeuralPipelineConfig;
let provider: "webgpu" | "wasm" = "wasm";
let encoder: Ort.InferenceSession | undefined;
let canvas: OffscreenCanvas | undefined;

function post(value: NeuralWorkerResponse, transfer: Transferable[] = []) { scope.postMessage(value, transfer); }
function finite(values: Float32Array): Float32Array {
  if (!values.every(Number.isFinite)) throw new Error("The neural runtime produced nonfinite values.");
  return values;
}
async function loadCv(url: string): Promise<CvRuntime> {
  await import(/* @vite-ignore */ url);
  const candidate = (globalThis as unknown as { cv?: CvRuntime & { then?: (callback: () => void) => void } }).cv;
  if (!candidate) throw new Error("Image preprocessing could not load.");
  return new Promise(resolve => {
    const ready = () => resolve(new Proxy(candidate, {
      get(target, key, receiver) { return key === "then" ? undefined : Reflect.get(target, key, receiver); },
    }));
    if (typeof candidate.then === "function") candidate.then(ready); else ready();
  });
}
async function createSession(asset: NeuralBundle["files"]["encoder"]): Promise<Ort.InferenceSession> {
  const bytes = await verifiedAsset(new URL(asset.name, baseUrl).href, asset);
  return ort.InferenceSession.create(bytes, { executionProviders: [provider], graphOptimizationLevel: "all" });
}
async function initialize(request: Extract<NeuralWorkerRequest, { type: "initialize" }>) {
  baseUrl = request.baseUrl; bundle = request.bundle; config = request.config;
  ort = await import(/* @vite-ignore */ new URL("ort.all.min.mjs", request.ortBaseUrl).href) as typeof Ort;
  if (ort.env.versions.web !== "1.22.0") throw new Error("The neural runtime version does not match its models.");
  // The deployed static site is deliberately usable without cross-origin isolation.
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = request.ortBaseUrl;
  const gpu = (navigator as unknown as { gpu?: { requestAdapter: (options: { powerPreference: string }) => Promise<unknown> } }).gpu;
  const adapter = await gpu?.requestAdapter({ powerPreference: "high-performance" }).catch(() => null);
  if (adapter && !(adapter as { isFallbackAdapter?: boolean }).isFallbackAdapter) {
    ort.env.webgpu.adapter = adapter as typeof ort.env.webgpu.adapter;
    provider = "webgpu";
  }
  if (request.needEncoder) {
    cv = await loadCv(request.openCvUrl);
    // A GPU graph failure is an explicit error; never substitute another detector.
    encoder = await createSession(bundle.files.encoder);
  }
  post({ id: request.id, type: "ready", provider });
}
async function encode(request: Extract<NeuralWorkerRequest, { type: "frame" }>) {
  let sample: VideoSample | undefined;
  try {
    if (!encoder) throw new Error("The image encoder is not initialized.");
    sample = new VideoSample(request.frame, { timestamp: request.timestamp, duration: request.duration, rotation: request.rotation });
    if (!canvas || canvas.width !== request.width || canvas.height !== request.height) canvas = new OffscreenCanvas(request.width, request.height);
    const context = canvas.getContext("2d", { alpha: false, willReadFrequently: true });
    if (!context) throw new Error("The browser cannot prepare image features.");
    sample.draw(context, 0, 0, canvas.width, canvas.height);
    const rgba = context.getImageData(0, 0, canvas.width, canvas.height);
    const prepared = prepareImage(cv, rgba, [request.roi.x, request.roi.y, request.roi.width, request.roi.height], request.offset);
    const image = new ort.Tensor("float32", prepared.image, [1, 3, 224, 224]);
    const pool = new ort.Tensor("float32", prepared.poolWeights, [1, 4, 7, 7]);
    try {
      const outputs = await encoder.run({ image, pool_weights: pool });
      try {
        if (outputs.tokens?.dims.join(",") !== "1,4,960") throw new Error("The encoder output shape changed.");
        const tokens = roundTokensToFloat16(finite(outputs.tokens.data as Float32Array));
        post({ id: request.id, type: "frame", tokens, quality: prepared.quality }, [tokens.buffer, prepared.quality.buffer]);
      } finally { for (const output of Object.values(outputs)) output.dispose(); }
    } finally { image.dispose(); pool.dispose(); }
  } finally { if (sample) sample.close(); else request.frame.close(); }
}
async function temporal(request: Extract<NeuralWorkerRequest, { type: "temporal" }>) {
  await encoder?.release(); encoder = undefined; canvas = undefined;
  const head = await createSession(bundle.files.temporal);
  try {
    const rows = request.times.length, probabilities = new Float32Array(rows * 4);
    for (let core = 0; core < rows; core += 128) {
      const end = Math.min(rows, core + 128), left = Math.max(0, core - 62), right = Math.min(rows, end + 62);
      // Bounded fusion: never materialize a whole-video 3,952-column matrix.
      const fused = fuseFeatures(request.times.slice(left, right), request.rankedAv.slice(left * 104, right * 104),
        request.embedding.times, request.embedding.tokens, request.embedding.quality, config);
      const input = new ort.Tensor("float32", fused, [1, right - left, FUSED_DIMENSION]);
      try {
        const outputs = await head.run({ features: input });
        try {
          if (outputs.logits?.dims.join(",") !== `1,${right - left},4`) throw new Error("The temporal output shape changed.");
          const logits = finite(outputs.logits.data as Float32Array);
          for (let row = core; row < end; row++) for (let column = 0; column < 4; column++)
            probabilities[row * 4 + column] = 1 / (1 + Math.exp(-logits[(row - left) * 4 + column]));
        } finally { for (const output of Object.values(outputs)) output.dispose(); }
      } finally { input.dispose(); }
      post({ id: request.id, type: "progress", completed: end, total: rows });
    }
    const rallies = decodeRallies(request.times, probabilities, request.duration, config.decoder);
    post({ id: request.id, type: "temporal", probabilities, rallies }, [probabilities.buffer]);
  } finally { await head.release(); }
}
let queue = Promise.resolve();
scope.addEventListener("message", (event: MessageEvent<NeuralWorkerRequest>) => {
  const request = event.data;
  queue = queue.then(async () => {
    try {
      if (request.type === "initialize") await initialize(request);
      else if (request.type === "frame") await encode(request);
      else await temporal(request);
    } catch (error) {
      post({ id: request.id, type: "error", message: error instanceof Error ? error.message : String(error) });
    }
  });
});
