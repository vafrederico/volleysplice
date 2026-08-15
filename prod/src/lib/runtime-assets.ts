export type RuntimeAsset =
  | "feature-reductions.wasm"
  | "libswresample.mjs"
  | "libswresample.wasm"
  | "model-9c92b8e9333f.json"
  | "opencv.js"
  | "opencv-worker.js"
  | "volleycut-logo.png";

export function runtimeAssetUrl(asset: RuntimeAsset): string {
  return new URL(`${import.meta.env.BASE_URL}runtime/${asset}`, window.location.href).href;
}
