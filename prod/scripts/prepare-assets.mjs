import { prepareNeuralAssets } from "./prepare-neural-assets.mjs";
import { publicSuppressionManifest } from "./public-suppression-manifest.mjs";
import { copyFile, mkdir, readFile, stat, writeFile, cp } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const appRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const source = resolve(
  appRoot,
  "node_modules/@techstark/opencv-js/dist/opencv.js",
);
const publicDirectory = process.env.VOLLEYCUT_PROD_PUBLIC_DIR
  ? resolve(process.env.VOLLEYCUT_PROD_PUBLIC_DIR) : resolve(appRoot, "public");
const runtimeDirectory = resolve(publicDirectory, "runtime");
const licenseDirectory = resolve(publicDirectory, "licenses");
if (publicDirectory !== resolve(appRoot, "public")) {
  await mkdir(publicDirectory, { recursive: true });
  const generated = ["runtime/opencv.js", "runtime/opencv-worker.js", "runtime/rally-models", "runtime/ort", "licenses"]
    .map(path => resolve(appRoot, "public", path));
  await cp(resolve(appRoot, "public"), publicDirectory, { recursive: true,
    filter: path => !generated.some(ignored => path === ignored || path.startsWith(`${ignored}/`) || path.startsWith(`${ignored}\\`)),
  });
}

await stat(source).catch(() => {
  throw new Error("OpenCV.js is unavailable. Run npm install in the prod directory.");
});
await mkdir(runtimeDirectory, { recursive: true });
const suppressionManifestPath = resolve(runtimeDirectory, "suppression-39eddf581639.manifest.json");
const suppressionManifest = JSON.parse(await readFile(suppressionManifestPath, "utf8"));
await writeFile(suppressionManifestPath, `${JSON.stringify(publicSuppressionManifest(suppressionManifest), null, 2)}\n`);
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
    resolve(appRoot, "../LICENSE"),
    resolve(licenseDirectory, "VolleySplice-MIT.txt"),
  ),
  copyFile(
    resolve(appRoot, "../THIRD_PARTY_NOTICES.md"),
    resolve(licenseDirectory, "THIRD_PARTY_NOTICES.md"),
  ),
  copyFile(
    resolve(appRoot, "node_modules/mediabunny/LICENSE"),
    resolve(licenseDirectory, "mediabunny-MPL-2.0.txt"),
  ),
  copyFile(
    resolve(appRoot, "node_modules/react/LICENSE"),
    resolve(licenseDirectory, "react-MIT.txt"),
  ),
  copyFile(
    resolve(appRoot, "node_modules/react-dom/LICENSE"),
    resolve(licenseDirectory, "react-dom-MIT.txt"),
  ),
  copyFile(
    resolve(appRoot, "licenses/fft.js-MIT.txt"),
    resolve(licenseDirectory, "fft.js-MIT.txt"),
  ),
  copyFile(
    resolve(appRoot, "licenses/FFmpeg-LGPL-2.1.txt"),
    resolve(licenseDirectory, "FFmpeg-LGPL-2.1.txt"),
  ),
  copyFile(
    resolve(appRoot, "licenses/libswresample-wrapper-source.c"),
    resolve(licenseDirectory, "libswresample-wrapper-source.c"),
  ),
]);

await prepareNeuralAssets(appRoot, publicDirectory);
