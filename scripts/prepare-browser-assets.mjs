import { execFile } from "node:child_process";
import { copyFile, mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const repository = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const openCvSource = resolve(
  repository,
  "node_modules/@techstark/opencv-js/dist/opencv.js",
);
const openCvDestination = resolve(repository, "public/on-device/opencv.js");
const workerDestination = resolve(repository, "public/on-device/opencv-worker.js");
const assemblyScriptCompiler = resolve(repository, "node_modules/assemblyscript/bin/asc.js");
const reductionsSource = resolve(repository, "assembly/feature-reductions.ts");
const reductionsDestination = resolve(repository, "public/on-device/feature-reductions.wasm");

const assets = [
  {
    source: openCvSource,
    destination: openCvDestination,
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

const sourceText = await readFile(openCvSource, "utf8");
const workerText = sourceText.replace(
  "}(this, function () {",
  "}(globalThis, function () {",
);
if (workerText === sourceText) {
  throw new Error("Could not adapt the pinned OpenCV asset for module workers.");
}
await writeFile(workerDestination, workerText);

await promisify(execFile)(
  process.execPath,
  [
    assemblyScriptCompiler,
    reductionsSource,
    "--outFile",
    reductionsDestination,
    "--optimize",
    "--runtime",
    "stub",
    "--noAssert",
  ],
  { cwd: repository },
);
