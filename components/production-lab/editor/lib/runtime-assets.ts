export type RuntimeAsset =
  | "feature-reductions.wasm"
  | "libswresample.mjs"
  | "libswresample.wasm"
  | "model-1ca43e38eefc.json"
  | "model-9c92b8e9333f.json"
  | "serving-side-85bc3325fbd4.json"
  | "side-switch-c2570481c30d.json"
  | "suppression-39eddf581639.json"
  | "opencv.js"
  | "opencv-worker.js"
  | "volleysplice-icon-transparent.png"
  | "volleysplice-logo.png";

export function runtimeAssetUrl(asset: RuntimeAsset): string {
  return `/production-lab/${asset}`;
}
