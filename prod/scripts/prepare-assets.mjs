import { copyFile, mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const appRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const source = resolve(
  appRoot,
  "node_modules/@techstark/opencv-js/dist/opencv.js",
);
const runtimeDirectory = resolve(appRoot, "public/runtime");
const licenseDirectory = resolve(appRoot, "public/licenses");

await stat(source).catch(() => {
  throw new Error("OpenCV.js is unavailable. Run npm install in the prod directory.");
});
await mkdir(runtimeDirectory, { recursive: true });
await copyFile(source, resolve(runtimeDirectory, "opencv.js"));

const sourceText = await readFile(source, "utf8");
const workerText = sourceText.replace(
  "}(this, function () {",
  "}(globalThis, function () {",
);
if (workerText === sourceText) {
  throw new Error("Could not adapt the pinned OpenCV asset for module workers.");
}
await writeFile(resolve(runtimeDirectory, "opencv-worker.js"), workerText);

const packageRoot = resolve(appRoot, "node_modules/@techstark/opencv-js");
const licenseCandidates = ["LICENSE", "LICENSE.md", "LICENSE.txt"];
for (const filename of licenseCandidates) {
  const candidate = resolve(packageRoot, filename);
  try {
    await stat(candidate);
    await mkdir(licenseDirectory, { recursive: true });
    await copyFile(candidate, resolve(licenseDirectory, "opencv-js.txt"));
    break;
  } catch {
    // The package currently publishes one of the candidates above.
  }
}

await mkdir(licenseDirectory, { recursive: true });
await Promise.all([
  copyFile(
    resolve(appRoot, "licenses/FFmpeg-LGPL-2.1.txt"),
    resolve(licenseDirectory, "FFmpeg-LGPL-2.1.txt"),
  ),
  copyFile(
    resolve(appRoot, "licenses/libswresample-wrapper-source.c"),
    resolve(licenseDirectory, "libswresample-wrapper-source.c"),
  ),
]);
