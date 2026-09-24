import { readFile, writeFile, mkdir } from "node:fs/promises";
import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import { resolve, dirname } from "node:path";

const args = new Map(process.argv.slice(2).reduce((pairs, value, i, values) => i % 2 ? pairs : [...pairs, [value, values[i + 1]]], []));
const root = args.get("--study-root");
if (!root) throw Error("--study-root is required");
const work = resolve("artifacts/private-media/editor-lab-neural-serves");
await mkdir(work, { recursive: true });
const manifestPath = resolve(root, "production-editor-lab-v1/manifest-guidance-v3.json");
const manifestRaw = await readFile(manifestPath, "utf8");
const manifest = JSON.parse(manifestRaw);
const sha = value => createHash("sha256").update(value).digest("hex");
const evidenceRaw = await readFile(resolve(root, "serving-side-v2/serve-head-evidence.json"), "utf8");
const evidence = JSON.parse(evidenceRaw);
const feedbackRegistration = JSON.parse(await readFile(resolve(root, "inference-registration.json"), "utf8"));
const feedbackRaw = await readFile(feedbackRegistration.feedback.path, "utf8");
if (sha(feedbackRaw) !== feedbackRegistration.feedback.sha256) throw Error("Immutable feedback changed");
const feedback = JSON.parse(feedbackRaw);
// Only immutable pre-edit capture geometry is used. Never corrections or human labels.
const roi = feedback.source.featureRoi;
if (!roi || ![roi.x, roi.y, roi.width, roi.height].every(Number.isFinite)) throw Error("Source feature ROI is unavailable");
const parentById = new Map(manifest.productionEvents.map(event => [event.id, event]));
const boundary = manifest.boundaryEvents.map(event => ({ ...event, included: true,
  agreement: parentById.get(event.parentId)?.agreement }));
const compact = manifest.compactEvents.map(event => ({ ...event, included: true }));
const input = { recordingId: manifest.recording.id, duration: manifest.recording.durationSeconds,
  videoUrl: `/lab/api/labeling/tasks/${manifest.recording.id}/video`, roi,
  populations: { boundary, compact }, evidence,
  manifestSha256: sha(manifestRaw), serveEvidenceSha256: sha(evidenceRaw), labelsUsed: false };
if (args.get("--publish") === "true") {
  const resultRaw = await readFile(resolve(work, "results.json"), "utf8");
  const result = JSON.parse(resultRaw);
  if (result.manifestSha256 !== input.manifestSha256 || result.labelsUsed !== false) throw Error("Inference source mismatch");
  for (const [key, events] of Object.entries(input.populations)) {
    const predictions = result.outputs[key].candidates;
    if (predictions.length !== events.length || events.some(event => !predictions.some(row => row.id === event.id && row.anchor === event.start))) throw Error("Serve anchor mismatch");
  }
  const target = resolve(root, "production-editor-lab-v1/manifest-neural-serves-v4.json");
  const output = { ...manifest, neuralServing: result.outputs, modelGeometryRevision: input.manifestSha256,
    provenance: { ...manifest.provenance, sourceHashes: { ...manifest.provenance.sourceHashes,
      "neural-serving-results.json": sha(resultRaw) } } };
  await writeFile(resolve(dirname(target), "neural-serving-results.json"), resultRaw, { flag: "wx" });
  await writeFile(target, JSON.stringify(output, null, 2) + "\n", { flag: "wx" });
  console.log(JSON.stringify({ target, counts: result.counts, modelId: result.modelId }));
} else {
  await writeFile(resolve(work, "input.json"), JSON.stringify(input));
  const require = createRequire(resolve("prod/package.json"));
  const { build } = require("esbuild");
  await build({ entryPoints: [resolve("scripts/editor-lab-neural-serves-browser.ts")], bundle: true, format: "esm", platform: "browser",
    define: { "import.meta.env.BASE_URL": JSON.stringify("/") }, outfile: resolve(work, "runner.js") });
  console.log(JSON.stringify({ work, roi, candidates: Object.fromEntries(Object.entries(input.populations).map(([key, value]) => [key, value.length])) }));
}
