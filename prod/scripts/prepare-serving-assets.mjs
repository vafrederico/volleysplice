import { createHash } from "node:crypto";
import { copyFile, mkdir, readFile } from "node:fs/promises";
import { resolve } from "node:path";

const hash = bytes => createHash("sha256").update(bytes).digest("hex");
export async function verifyServingAssets(appRoot, targetRoot) {
  const pinned = await readFile(resolve(appRoot, "../models/serving-side/web-manifest.json"));
  const shipped = await readFile(resolve(targetRoot, "serving-models/manifest.json"));
  if (hash(pinned) !== hash(shipped)) throw new Error("The serving-side manifest does not match the pinned testing models.");
  const manifest = JSON.parse(pinned), rally = JSON.parse(await readFile(resolve(targetRoot, "rally-models/manifest.json")));
  if (manifest.primaryTarget !== .98 || manifest.policy !== "all-three-seeds-agree"
      || manifest.inputContract !== "centered-13x3843-plus-249-v1") throw new Error("Invalid serving-side input or acceptance contract.");
  for (const [selection, bundle] of Object.entries(manifest.variants)) {
    const paired = rally.variants[selection];
    if (!paired || bundle.rallyModelId !== paired.id || bundle.encoderSha256 !== paired.files.encoder.sha256
        || bundle.encoderWeightsSha256 !== paired.encoderWeightsSha256) throw new Error("Serving-side and rally encoders do not match.");
    const bytes = await readFile(resolve(targetRoot, "serving-models", bundle.directory, bundle.file.name));
    if (bytes.byteLength !== bundle.file.sizeBytes || hash(bytes) !== bundle.file.sha256) throw new Error(`Serving-side model integrity failed: ${selection}.`);
  }
}
export async function prepareServingAssets(appRoot, targetRoot) {
  const source = resolve(appRoot, "public/runtime");
  await verifyServingAssets(appRoot, source);
  if (source !== resolve(targetRoot)) {
    const manifest = JSON.parse(await readFile(resolve(appRoot, "../models/serving-side/web-manifest.json")));
    for (const bundle of Object.values(manifest.variants)) {
      await mkdir(resolve(targetRoot, "serving-models", bundle.directory), { recursive: true });
      await copyFile(resolve(source, "serving-models", bundle.directory, bundle.file.name),
        resolve(targetRoot, "serving-models", bundle.directory, bundle.file.name));
    }
    await copyFile(resolve(source, "serving-models/manifest.json"), resolve(targetRoot, "serving-models/manifest.json"));
  }
  await verifyServingAssets(appRoot, targetRoot);
}
