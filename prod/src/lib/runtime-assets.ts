export type RuntimeAsset =
  | "feature-reductions.wasm"
  | "libswresample.mjs"
  | "libswresample.wasm"
  | "model-1ca43e38eefc.json"
  | "model-9c92b8e9333f.json"
  | "suppression-39eddf581639.json"
  | "opencv.js"
  | "opencv-worker.js"
  | "volleycut-logo.png";

export const ANDROID_APK_FILENAME =
  "VolleyCut-v0.10.1-arm64-release-signed.apk";

function publicAssetUrl(path: string): string {
  return new URL(`${import.meta.env.BASE_URL}${path}`, window.location.href).href;
}

export function runtimeAssetUrl(asset: RuntimeAsset): string {
  return publicAssetUrl(`runtime/${asset}`);
}

export function androidApkUrl(): string {
  return publicAssetUrl(`downloads/${ANDROID_APK_FILENAME}`);
}
