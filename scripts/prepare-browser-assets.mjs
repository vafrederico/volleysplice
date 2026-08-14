import { copyFile, mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repository = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const source = resolve(
  repository,
  "node_modules/@techstark/opencv-js/dist/opencv.js",
);
const destination = resolve(repository, "public/on-device/opencv.js");
const workerDestination = resolve(repository, "public/on-device/opencv-worker.js");

await stat(source).catch(() => {
  throw new Error(
    "Pinned OpenCV browser asset is missing. Run npm install before preparing assets.",
  );
});
await mkdir(dirname(destination), { recursive: true });
await copyFile(source, destination);
const sourceText = await readFile(source, "utf8");
const workerText = sourceText.replace(
  "}(this, function () {",
  "}(globalThis, function () {",
);
if (workerText === sourceText) {
  throw new Error("Could not adapt the pinned OpenCV asset for module workers.");
}
await writeFile(workerDestination, workerText);
