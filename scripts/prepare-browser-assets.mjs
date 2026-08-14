import { copyFile, mkdir, stat } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repository = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const assets = [
  {
    source: resolve(repository, "node_modules/@techstark/opencv-js/dist/opencv.js"),
    destination: resolve(repository, "public/on-device/opencv.js"),
    missing:
      "Pinned OpenCV browser asset is missing. Run npm install before preparing assets.",
  },
  ...["libswresample.mjs", "libswresample.wasm"].map((filename) => ({
    source: resolve(repository, "vendor/libswresample-wasm/dist", filename),
    destination: resolve(repository, "public/on-device/libswresample", filename),
    missing:
      "Pinned libswresample WASM asset is missing. Run scripts/build-libswresample-wasm.sh.",
  })),
];

for (const asset of assets) {
  await stat(asset.source).catch(() => {
    throw new Error(asset.missing);
  });
  await mkdir(dirname(asset.destination), { recursive: true });
  await copyFile(asset.source, asset.destination);
}
