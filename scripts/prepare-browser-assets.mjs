import { execFile } from "node:child_process";
import { copyFile, mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const repository = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const source = resolve(
  repository,
  "node_modules/@techstark/opencv-js/dist/opencv.js",
);
const destination = resolve(repository, "public/on-device/opencv.js");
const workerDestination = resolve(repository, "public/on-device/opencv-worker.js");
const assemblyScriptCompiler = resolve(repository, "node_modules/assemblyscript/bin/asc.js");
const reductionsSource = resolve(repository, "assembly/feature-reductions.ts");
const reductionsDestination = resolve(repository, "public/on-device/feature-reductions.wasm");

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
