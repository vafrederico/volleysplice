import { copyFile, mkdir, stat } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repository = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const source = resolve(
  repository,
  "node_modules/@techstark/opencv-js/dist/opencv.js",
);
const destination = resolve(repository, "public/on-device/opencv.js");

await stat(source).catch(() => {
  throw new Error(
    "Pinned OpenCV browser asset is missing. Run npm install before preparing assets.",
  );
});
await mkdir(dirname(destination), { recursive: true });
await copyFile(source, destination);
