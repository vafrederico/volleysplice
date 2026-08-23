import { createHash } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const appRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

const expectedHashes = new Map([
  [
    "android/VolleyCut-v0.10.5-arm64-release-signed.apk",
    "334cb90fc6fc90940abe8028639b30e90a57c6ca7df2d0906773d39f224f247e",
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

for (const [asset, expected] of expectedHashes) {
  const bytes = await readFile(resolve(appRoot, "dist", asset));
  const actual = createHash("sha256").update(bytes).digest("hex");
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
