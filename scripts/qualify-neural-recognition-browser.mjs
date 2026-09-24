#!/usr/bin/env node
import { privateValue } from "../lib/server/private-ledger.mjs";
// Fresh headless desktop Chrome; loopback fixture server; no production app.
import { createServer } from "node:http";
import { createHash } from "node:crypto";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { createRequire } from "node:module";
import { resolve, join, relative, sep, extname, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";
import os from "node:os";
import { spawnSync } from "node:child_process";

const cli = new Map();
for (let i = 2; i < process.argv.length; i += 2) cli.set(process.argv[i], process.argv[i + 1]);
for (const name of ["--harness", "--source"]) if (!cli.has(name)) throw new Error(`${name} is required`);
if (cli.has("--output") === cli.has("--wsl-output")) throw new Error("Supply exactly one of --output or --wsl-output");
const harness = resolve(cli.get("--harness"));
const source = resolve(cli.get("--source"));
const output = cli.has("--wsl-output") ? cli.get("--wsl-output") : resolve(cli.get("--output"));
const repeats = Number(cli.get("--repeats") ?? 10);
if (!Number.isInteger(repeats) || repeats < 1) throw new Error("repeats must be positive");
const require = createRequire(join(harness, "package.json"));
const { chromium } = require("playwright-core");
const runtimeRoot = join(harness, "node_modules", "onnxruntime-web", "dist");
const runtimePackage = JSON.parse(await readFile(join(runtimeRoot, "..", "package.json"), "utf8"));
if (runtimePackage.version !== "1.22.0") throw new Error("Expected pinned onnxruntime-web1.22.0");
if (!cli.has("--wsl-output")) await mkdir(output, { recursive: false });
const hash = data => createHash("sha256").update(data).digest("hex");
const fixturePath = join(source, "fixture-manifest.json");
const manifestBytes = await readFile(fixturePath);
const manifest = JSON.parse(manifestBytes);
if (manifest.schemaVersion !== 1 || manifest.kind !== "recognition-portability-fixtures-v1" || typeof manifest.temporalModelsTrained !== "boolean") throw new Error("Unexpected fixture contract");
if (manifest.temporalModelsTrained && manifest.models.some(model => model.state !== "trained completed checkpoint with fitted scaler" || !model.checkpoint?.scalerIncludedInGraph)) throw new Error("Trained fixture checkpoint/scaler provenance missing");
const verifiedFiles = new Map();
async function verify(entry) {
  if (basename(entry.file) !== entry.file) throw new Error("Fixture filename must be local");
  const data = await readFile(join(source, entry.file));
  if (data.length !== entry.sizeBytes || hash(data) !== entry.sha256) throw new Error(`Fixture identity mismatch: ${entry.file}`);
  verifiedFiles.set(entry.file, entry);
  return data;
}
const prerequisite = JSON.parse(await verify(manifest.cpuReport));
if (prerequisite.status !== "pass" || prerequisite.gpuUsed !== false || prerequisite.physicalPhoneMeasured !== false) throw new Error("CPU qualification prerequisite did not pass");
for (const model of manifest.models) {
  await verify(model.graph);
  for (const test of model.cases) for (const entry of [...test.inputs, test.expected]) await verify(entry);
}
const servedRuntime = new Set();
const server = createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, "http://localhost").pathname);
    response.setHeader("Cache-Control", "no-store");
    response.setHeader("Cross-Origin-Opener-Policy", "same-origin");
    response.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
    if (pathname === "/") {
      response.setHeader("Content-Type", "text/html");
      response.end('<!doctype html><title>Recognition runtime engineering</title><script src="/runtime/ort.wasm.min.js"></script>');
      return;
    }
    const isRuntime = pathname.startsWith("/runtime/");
    const isFixture = pathname.startsWith("/fixtures/");
    if (!isRuntime && !isFixture) throw new Error("unknown route");
    const root = isRuntime ? runtimeRoot : source;
    const requested = resolve(root, pathname.split("/").slice(2).join("/"));
    const rel = relative(root, requested);
    if (!rel || rel === ".." || rel.startsWith(`..${sep}`)) throw new Error("outside root");
    if (isFixture && !verifiedFiles.has(rel)) throw new Error("unregistered fixture");
    response.setHeader("Content-Type", ({ ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm", ".json": "application/json" })[extname(requested)] ?? "application/octet-stream");
    const data = await readFile(requested);
    if (isFixture && hash(data) !== verifiedFiles.get(rel).sha256) throw new Error("fixture changed after verification");
    if (isRuntime) servedRuntime.add(requested);
    response.end(data);
  } catch {
    response.statusCode = 404;
    response.end("Not found");
  }
});
await new Promise(done => server.listen(0, "127.0.0.1", done));
let browser, result;
const errors = [];
try {
  browser = await chromium.launch({ executablePath: cli.get("--browser") ?? "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    headless: true, args: ["--disable-gpu", "--no-first-run", "--no-default-browser-check"] });
  const page = await browser.newPage();
  page.on("pageerror", error => errors.push(String(error)));
  page.on("requestfailed", request => errors.push(`${request.url()}: ${request.failure()?.errorText}`));
  await page.goto(`http://127.0.0.1:${server.address().port}/`, { waitUntil: "load", timeout: 30000 });
  result = await page.evaluate(async ({ manifest, repeats }) => {
    const ort = window.ort;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = `${location.origin}/runtime/`;
    const load = async entry => {
      const response = await fetch(`/fixtures/${entry.file}`);
      if (!response.ok) throw new Error(`Failed fetch: ${entry.file}`);
      const bytes = await response.arrayBuffer();
      const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), value => value.toString(16).padStart(2, "0")).join("");
      if (digest !== entry.sha256 || bytes.byteLength !== entry.sizeBytes) throw new Error(`Browser fixture hash mismatch: ${entry.file}`);
      return bytes;
    };
    const compare = (actual, expected, tolerance) => {
      if (actual.length !== expected.length) throw new Error("output length mismatch");
      let maximum = 0, sum = 0;
      for (let i = 0; i < actual.length; i++) {
        const delta = Math.abs(actual[i] - expected[i]);
        if (!Number.isFinite(actual[i]) || !Number.isFinite(expected[i]) || delta > tolerance.atol + tolerance.rtol * Math.abs(expected[i])) throw new Error(`parity failed at ${i}: ${actual[i]} vs ${expected[i]}`);
        maximum = Math.max(maximum, delta); sum += delta;
      }
      return { passed: true, maximumAbsoluteError: maximum, meanAbsoluteError: sum / actual.length };
    };
    const models = [];
    for (const model of manifest.models) {
      const downloadStart = performance.now();
      const graphBytes = await load(model.graph);
      const loopbackGraphDownloadAndVerificationMilliseconds = performance.now() - downloadStart;
      const creationStart = performance.now();
      const session = await ort.InferenceSession.create(graphBytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
      const sessionCreationMilliseconds = performance.now() - creationStart;
      const cases = [], outputsByCase = new Map();
      let benchmarkInputs;
      try {
        for (const test of model.cases) {
          const inputs = {};
          for (const entry of test.inputs) {
            const bytes = await load(entry);
            inputs[entry.name] = new ort.Tensor(entry.dtype, entry.dtype === "bool" ? new Uint8Array(bytes) : new Float32Array(bytes), entry.shape);
          }
          const expected = new Float32Array(await load(test.expected));
          const predictions = (await session.run(inputs))[model.graph.outputName];
          if (JSON.stringify(predictions.dims) !== JSON.stringify(test.expected.shape)) throw new Error("output dimensions differ");
          outputsByCase.set(test.name, { data: new Float32Array(predictions.data), shape: predictions.dims });
          cases.push({ name: test.name, ...compare(predictions.data, expected, test.tolerance), description: test.description });
          if (!benchmarkInputs) benchmarkInputs = inputs;
        }
        const chunkEquivalence = [];
        for (const rule of model.chunkEquivalence ?? []) {
          const chunk = outputsByCase.get(rule.chunkCase), reference = outputsByCase.get(rule.referenceCase);
          if (chunk.shape[0] !== 1 || reference.shape[0] !== 1 || chunk.shape[2] !== 4 || reference.shape[2] !== 4) throw new Error("expected single-record four-head chunks");
          const actual = chunk.data.slice(rule.chunkStart * 4, (rule.chunkStart + rule.length) * 4);
          const expected = reference.data.slice(rule.referenceStart * 4, (rule.referenceStart + rule.length) * 4);
          if (actual.length !== rule.length * 4 || expected.length !== actual.length) throw new Error("chunk reference slice exceeds output");
          chunkEquivalence.push({ ...rule, ...compare(actual, expected, { atol: 1e-5, rtol: 1e-4 }) });
        }
        for (let i = 0; i < 2; i++) await session.run(benchmarkInputs);
        const durations = [];
        for (let i = 0; i < repeats; i++) {
          const start = performance.now();
          await session.run(benchmarkInputs);
          durations.push(performance.now() - start);
        }
        const sorted = [...durations].sort((a, b) => a - b);
        const quantile = q => {
          const position = (sorted.length - 1) * q, low = Math.floor(position), high = Math.ceil(position);
          return sorted[low] + (sorted[high] - sorted[low]) * (position - low);
        };
        models.push({ name: model.name, state: model.state, checkpoint: model.checkpoint, graph: model.graph, cases, chunkEquivalence,
          loopbackGraphDownloadAndVerificationMilliseconds, sessionCreationMilliseconds,
          preparedInputLatency: { warmupCalls: 2, repeats, medianMilliseconds: quantile(.5), p95Milliseconds: quantile(.95), samplesMilliseconds: durations } });
      } finally { await session.release(); }
    }
    return { status: "pass", models, runtimeVersion: ort.env.versions.web, userAgent: navigator.userAgent };
  }, { manifest, repeats });
  result.browserVersion = browser.version();
} catch (error) {
  result = { status: "fail", failure: { type: error.name, message: error.message } };
} finally {
  if (browser) await browser.close();
  await new Promise(done => server.close(done));
}
const runtimeArtifacts = [];
for (const path of [...servedRuntime].sort()) {
  const data = await readFile(path);
  runtimeArtifacts.push({ file: basename(path), sha256: hash(data), sizeBytes: data.length, gzipBytes: gzipSync(data).length });
}
const report = { schemaVersion: 1, createdAt: new Date().toISOString(), ...result,
  scope: `Isolated desktop Chrome WASM CPU engineering parity; ${manifest.temporalModelsTrained ? "identified trained temporal checkpoints and scalers" : "seeded untrained temporal heads"}`,
  gpuUsed: false, wasmThreads: 1, executionProviders: ["wasm"], physicalPhoneMeasured: false, productionAppStarted: false,
  cpu: os.cpus()[0]?.model, operatingSystem: `${os.platform()} ${os.release()}`, browserErrors: errors,
  fixtureManifest: { path: fixturePath, sha256: hash(manifestBytes), sizeBytes: manifestBytes.length },
  cpuPrerequisite: manifest.cpuReport, runtimeArtifacts,
  runtimeTotalBytes: runtimeArtifacts.reduce((total, entry) => total + entry.sizeBytes, 0),
  runtimeTotalGzipBytes: runtimeArtifacts.reduce((total, entry) => total + entry.gzipBytes, 0),
  packageLockSha256: hash(await readFile(join(harness, "package-lock.json"))),
  scriptSha256: hash(await readFile(fileURLToPath(import.meta.url))),
  limitations: ["No physical phone or Android delegate measurement; browser is desktop Chrome.",
    "Only explicitly identified trained checkpoint/scaler bytes qualify; seeded graphs prove architecture compatibility only.",
    "The DINO graph includes only feature fusion and temporal prediction, not DINO image extraction.",
    "Prepared-input latency excludes video decode, feature extraction/preprocessing, IO and UI.",
    "Graph downloads are loopback NAS-backed diagnostics, not network-download forecasts; byte counts are uncompressed and gzip estimates.",
    ...(manifest.models.some(model => model.graph.dynamicAxes?.features)
      ? ["Dynamic mobile TCN uses true-length segments and real halos; separate calls reset ignored gaps. No synthetic padding is qualified."]
      : ["Static252 TCN graphs qualify interior chunks only, not short true-segment boundaries."]),
    "Other experiments share this desktop, so latency samples are engineering diagnostics."] };
if (errors.length) report.status = "fail";
const reportText = JSON.stringify(report, null, 2) + "\n";
if (cli.has("--wsl-output")) {
  // Only report JSON crosses stdin; no shell interpolation or fixture staging.
  const saved = spawnSync("wsl.exe", ["--", privateValue("private-reference-0110"), "-c",
    "import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(parents=True,exist_ok=False); f=(p/'browser-report.json').open('x',encoding='utf-8'); f.write(sys.stdin.read()); f.close()", output],
  { input: reportText, windowsHide: true, encoding: "utf8", timeout: 30000 });
  if (saved.status !== 0) throw new Error(`NAS report publication failed: ${saved.stderr || saved.stdout}`);
} else {
  await writeFile(join(output, "browser-report.json"), reportText, { flag: "wx" });
}
console.log(JSON.stringify({ status: report.status, output, failure: report.failure,
  runtimeBytes: report.runtimeTotalBytes, models: report.models?.map(model => ({ name: model.name, bytes: model.graph.sizeBytes,
    medianMs: model.preparedInputLatency.medianMilliseconds, maxError: Math.max(...model.cases.map(test => test.maximumAbsoluteError)) })) }));
if (report.status !== "pass") process.exitCode = 1;
