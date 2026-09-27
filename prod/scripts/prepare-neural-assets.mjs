import { createHash } from "node:crypto";
import { copyFile, mkdir, readFile, cp } from "node:fs/promises";
import { resolve } from "node:path";

// These are the exact package assets used by the qualified ORT Web 1.22.0 runtime.
export const ortAssets = new Map([
  ["ort.all.min.mjs", "abd87226e2b3d80dbe1d4286781b5e949a8e81a0f361425f7b4937a2c599dc2b"],
  ["ort-wasm-simd-threaded.jsep.mjs", "1cbcba8f2c769c1eecbab66a1b1e55ef11704515bf4306373e3db3c37cf6dcd8"],
  ["ort-wasm-simd-threaded.jsep.wasm", "b45970d0632383a057c27ca5b660b216f8e00c17cf8db9f6207b5e4abc839368"],
  ["ort-wasm-simd-threaded.mjs", "30dd851d9c00622940500f71ddd2ff8820c5cb65270816080175b958705385a8"],
  ["ort-wasm-simd-threaded.wasm", "71aef04959c5c1b6de461b6538e2058e306610034a85aad2742d0c7fd4533fe4"],
]);
function sha(bytes) { return createHash("sha256").update(bytes).digest("hex"); }
export async function verifyNeuralAssets(appRoot, targetRoot) {
  const pinnedBytes = await readFile(resolve(appRoot, "../models/distilled-large/web-manifest.json"));
  const manifest = JSON.parse(pinnedBytes);
  const shipped = await readFile(resolve(targetRoot, "rally-models/manifest.json"));
  if (sha(pinnedBytes) !== sha(shipped)) throw new Error("The neural model manifest does not match the pinned release.");
  for (const variant of Object.values(manifest.variants)) for (const asset of Object.values(variant.files)) {
    const bytes = await readFile(resolve(targetRoot, "rally-models", variant.directory, asset.name));
    if (bytes.byteLength !== asset.sizeBytes || sha(bytes) !== asset.sha256)
      throw new Error(`The neural bundle failed its integrity check: ${variant.id}/${asset.name}.`);
  }
  for (const [name, hash] of ortAssets) {
    if (sha(await readFile(resolve(targetRoot, "ort", name))) !== hash) throw new Error(`ORT Web asset integrity failed: ${name}.`);
  }
}
export async function prepareNeuralAssets(appRoot, publicDirectory = resolve(appRoot, "public")) {
  const pinned = await readFile(resolve(appRoot, "../models/distilled-large/web-manifest.json"));
  // Public release inputs travel with the checkout, including Cloudflare builds.
  // The private bundle environment remains an Android/research build input only.
  const source = resolve(appRoot, "public/runtime/rally-models");
  const supplied = await readFile(resolve(source, "manifest.json"));
  if (sha(supplied) !== sha(pinned)) throw new Error("The supplied model bundle manifest does not match the pinned release.");
  const target = resolve(publicDirectory, "runtime");
  const manifest = JSON.parse(pinned);
  // Validate every source before copying, including encoder/head pairing and scalers.
  for (const variant of Object.values(manifest.variants)) for (const asset of Object.values(variant.files)) {
    const bytes = await readFile(resolve(source, variant.directory, asset.name));
    if (bytes.byteLength !== asset.sizeBytes || sha(bytes) !== asset.sha256) throw new Error(`Invalid source model asset: ${variant.id}/${asset.name}.`);
  }
  if (source !== resolve(target, "rally-models")) {
    await mkdir(resolve(target, "rally-models"), { recursive: true });
    for (const variant of Object.values(manifest.variants)) {
      await mkdir(resolve(target, "rally-models", variant.directory), { recursive: true });
      for (const asset of Object.values(variant.files)) await copyFile(resolve(source, variant.directory, asset.name), resolve(target, "rally-models", variant.directory, asset.name));
    }
    await copyFile(resolve(source, "manifest.json"), resolve(target, "rally-models/manifest.json"));
  }
  await mkdir(resolve(target, "ort"), { recursive: true });
  for (const [name, hash] of ortAssets) {
    const original = resolve(appRoot, "node_modules/onnxruntime-web/dist", name);
    if (sha(await readFile(original)) !== hash) throw new Error(`Expected pinned ONNX Runtime Web 1.22.0 asset: ${name}.`);
    await copyFile(original, resolve(target, "ort", name));
  }
  await cp(resolve(appRoot, "../models/distilled-large/licenses"), resolve(publicDirectory, "licenses/distilled-large"), { recursive: true });
  await verifyNeuralAssets(appRoot, target);
}
