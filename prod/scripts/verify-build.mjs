import { createHash } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const appRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

const expectedHashes = new Map([
  [
    "android/VolleyCut-v0.10.8-arm64-release-signed.apk",
    "50bb001ca9e918707c8fa2e682ffdccd7756d49c02ab6c446fbb3d729e28b1af",
  ],
  [
    "runtime/model-1ca43e38eefc.json",
    "d2c2c11e8fed8b6c6ad77d244b613e81d5bab101939a8f57be5166b45ebca78f",
  ],
  [
    "runtime/model-9c92b8e9333f.json",
    "d8cc42f70bc10576a5e03251b05981ceeee1a61a15c61cc5dfb68dd631e6f90d",
  ],
  [
    "runtime/serving-side-85bc3325fbd4.json",
    "14f18bf0b0f326ccd7ef4b3d614a96a53dd9675df61813fd375677489d0e5a7c",
  ],
  [
    "runtime/side-switch-c2570481c30d.json",
    "ab4197545fb916a37ee6ac1d69e74ddfc0123c09039cdfa88c4ef378dd3e27fc",
  ],
  [
    "runtime/suppression-39eddf581639.json",
    "ef0ad4eb93fa61ce1d403f083d91f7578cf9ff0f31fac797fde9ab8b73f42794",
  ],
  [
    "runtime/feature-reductions.wasm",
    "2b11060145a38e598df5ffdc15e8a8ef6783b05217445d8bc579552788d45446",
  ],
  [
    "runtime/libswresample.mjs",
    "022782d1e08e483d8c67de30f68177eff5904998c410df028f26e342c793ae48",
  ],
  [
    "runtime/libswresample.wasm",
    "c7ed95ed8b6f5e11ea9e86262214b978bd145a6ec1f648dd56616af298449f90",
  ],
]);

const canonicalLfAssets = new Set([
  "runtime/side-switch-c2570481c30d.json",
]);

for (const [asset, expected] of expectedHashes) {
  const bytes = await readFile(resolve(appRoot, "dist", asset));
  const hashInput = canonicalLfAssets.has(asset)
    ? Buffer.from(bytes.toString("utf8").replaceAll("\r\n", "\n"), "utf8")
    : bytes;
  const actual = createHash("sha256").update(hashInput).digest("hex");
  if (actual !== expected) {
    throw new Error(`${asset} integrity mismatch: expected ${expected}, got ${actual}`);
  }
}

for (const asset of [
  "runtime/suppression-39eddf581639.manifest.json",
  "runtime/opencv.js",
  "runtime/opencv-worker.js",
  "runtime/volleycut-logo.png",
  "licenses/FFmpeg-LGPL-2.1.txt",
  "licenses/libswresample-wrapper-source.c",
  "licenses/opencv-js.txt",
]) {
  const metadata = await stat(resolve(appRoot, "dist", asset));
  if (!metadata.isFile() || metadata.size === 0) throw new Error(`${asset} is missing.`);
}

const index = await readFile(resolve(appRoot, "dist/index.html"), "utf8");
if (!index.includes('src="./assets/') || !index.includes('href="./assets/')) {
  throw new Error("The static build does not use deployment-portable relative assets.");
}

console.log("Verified portable build assets, integrity hashes, and relative paths.");
