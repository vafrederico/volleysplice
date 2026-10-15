// Run the exact pinned browser WASM engine on private, temporary float32 inputs.
// The descriptor and buffers stay outside Git; stdout contains aggregate parity.
import { readFile, open } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "../prod/node_modules/onnxruntime-web/dist/ort.all.min.mjs";
import { neuralServingDecision } from "../prod/src/lib/on-device/neural-serving-model.ts";

const root = fileURLToPath(new URL("..", import.meta.url));
const descriptor = JSON.parse(await readFile(process.argv[2], "utf8"));
const manifest = JSON.parse(await readFile(resolve(root, "models/serving-side/web-manifest.json"), "utf8"));
const bundle = manifest.variants[descriptor.selection];
if (!bundle || !Number.isSafeInteger(descriptor.rows) || descriptor.rows < 1) throw new Error("Invalid parity input descriptor.");
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
const session = await ort.InferenceSession.create(await readFile(resolve(root,
  "prod/public/runtime/serving-models", bundle.directory, bundle.file.name)), { executionProviders: ["wasm"] });
const windowsFile = await open(descriptor.windows), featuresFile = await open(descriptor.features);
const expectedBytes = await readFile(descriptor.logits);
const expected = new Float32Array(expectedBytes.buffer, expectedBytes.byteOffset, expectedBytes.byteLength / 4);
if (expected.length !== descriptor.rows * 6) throw new Error("Invalid reference logit dimensions.");
const sigmoid = value => 1 / (1 + Math.exp(-Math.max(-35, Math.min(35, value))));
function seedDecision(logits, seed) {
  const far = sigmoid(logits[0]), near = sigmoid(logits[1]), mass = far + near;
  const route = near / mass, side = route >= .5 ? 1 : 0;
  const confidence = Math.min(mass, 1) * (side ? route : 1 - route);
  return confidence > 0 && confidence >= (side ? seed.nearCutoff : seed.farCutoff) ? side : -1;
}
let maximumLogitError = 0, maximumRouteError = 0, maximumEvidenceError = 0;
async function readRows(file, start, rows, width) {
  const bytes = Buffer.alloc(rows * width * 4);
  const { bytesRead } = await file.read(bytes, 0, bytes.length, start * width * 4);
  if (bytesRead !== bytes.length) throw new Error("Truncated parity input buffer.");
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}
try {
  for (let start = 0; start < descriptor.rows; start += 8) {
    const rows = Math.min(8, descriptor.rows - start);
    const windows = new ort.Tensor("float32", await readRows(windowsFile, start, rows, 13 * 3843), [rows, 13, 3843]);
    const features = new ort.Tensor("float32", await readRows(featuresFile, start, rows, 249), [rows, 249]);
    try {
      const output = await session.run({ windows, features });
      try {
        if (output.logits.dims.join(",") !== `${rows},3,2`) throw new Error("Browser output dimensions changed.");
        const logits = output.logits.data;
        for (let row = 0; row < rows; row++) {
          const actual = logits.subarray(row * 6, row * 6 + 6), reference = expected.subarray((start + row) * 6, (start + row + 1) * 6);
          for (let column = 0; column < 6; column++) {
            maximumLogitError = Math.max(maximumLogitError, Math.abs(actual[column] - reference[column]));
            maximumEvidenceError = Math.max(maximumEvidenceError, Math.abs(sigmoid(actual[column]) - sigmoid(reference[column])));
          }
          for (let seed = 0; seed < 3; seed++) if (seedDecision(actual.subarray(seed * 2, seed * 2 + 2), bundle.seeds[seed])
              !== seedDecision(reference.subarray(seed * 2, seed * 2 + 2), bundle.seeds[seed])) throw new Error("Browser changed a frozen seed acceptance decision.");
          const a = neuralServingDecision(actual, bundle), b = neuralServingDecision(reference, bundle);
          maximumRouteError = Math.max(maximumRouteError, Math.abs(a.nearProbability - b.nearProbability));
          if (a.accepted !== b.accepted || a.side !== b.side) throw new Error("Browser changed the three-seed serving decision.");
        }
      } finally { for (const tensor of Object.values(output)) tensor.dispose(); }
    } finally { windows.dispose(); features.dispose(); }
  }
  if (maximumLogitError > 1e-4 || maximumRouteError > 2e-5 || maximumEvidenceError > 2e-5) throw new Error("Browser numerical parity exceeded float32 tolerances.");
  console.log(JSON.stringify({ runtime: ort.env.versions.web, validationRows: descriptor.rows, maximumLogitError,
    maximumRouteError, maximumEvidenceError, seedAcceptanceDecisionsMatch: true, consensusDecisionsMatch: true }));
} finally { await windowsFile.close(); await featuresFile.close(); await session.release(); }
