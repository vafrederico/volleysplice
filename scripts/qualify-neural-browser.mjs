#!/usr/bin/env node
import { privateValue } from "../lib/server/private-ledger.mjs";
// Isolated desktop-Chrome WASM qualification; never serves or modifies the app.
import { createServer } from "node:http";
import { createHash } from "node:crypto";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { createRequire } from "node:module";
import { resolve, join, relative, sep, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { gzipSync } from "node:zlib";
import os from "node:os";

const cli = new Map();
for (let index = 2; index < process.argv.length; index += 2) cli.set(process.argv[index], process.argv[index + 1]);
const harness = resolve(cli.get("--harness") ?? "");
const chrome = cli.get("--browser") ?? "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const source = cli.get("--source") ?? privateValue("private-reference-0111");
const output = cli.get("--output") ?? privateValue("private-reference-0112");
if (!cli.has("--harness")) throw new Error("--harness must name an isolated directory with pinned onnxruntime-web and playwright-core dependencies");
const require = createRequire(join(harness, "package.json"));
const { chromium } = require("playwright-core");
const runtimeRoot = join(harness, "node_modules", "onnxruntime-web", "dist");
const runtimePackage = JSON.parse(await readFile(join(runtimeRoot, "..", "package.json"), "utf8"));
if (runtimePackage.version !== "1.22.0") throw new Error("Expected pinned onnxruntime-web 1.22.0");
const fixtures = join(harness, "fixtures");
await mkdir(fixtures, { recursive: false });
const repository = resolve(fileURLToPath(new URL("..", import.meta.url)));
const wslPath = value => value.replaceAll("\\", "/").replace(/^([A-Za-z]):/, (_, drive) => `/mnt/${drive.toLowerCase()}`);
const python = privateValue("private-reference-0110");
const prepareCode = `
import hashlib, importlib.util, json, pathlib, shutil, sys
import numpy as np
import torch
repo, source, target = map(pathlib.Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location('qualification', repo / 'scripts/qualify-neural-runtime.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
report = json.loads((source / 'runtime-qualification.json').read_text())
if report['status'] != 'pass': raise ValueError('CPU prerequisite did not pass')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def identity(path): return {'file':path.name,'sha256':sha(path),'sizeBytes':path.stat().st_size}
manifest = {'seed':report['seed'],'cpuQualification':{'path':str(source / 'runtime-qualification.json'),'sha256':sha(source / 'runtime-qualification.json')},'cases':[],'models':[]}
case_inputs = module.inputs(1024, report['seed'] + 1)
for name, values in case_inputs.items():
    path = target / (name + '-input.f32')
    values.tofile(path)
    manifest['cases'].append({'name':name,'input':identity(path),'shape':list(values.shape)})
for model_row in report['models']:
    kind = model_row['kind']
    torch.manual_seed(report['seed'])
    model = module.CompactTemporalNetwork(module.CompactTemporalConfig(kind=kind)).cpu().eval()
    row = {'kind':kind,'haloTicks':model.config.halo_ticks,'expected':{},'models':{}}
    for key, artifact in model_row['artifacts'].items():
        original = source / pathlib.Path(artifact['path']).name
        if sha(original) != artifact['sha256']: raise ValueError('Model hash changed')
        destination = target / original.name
        shutil.copyfile(original, destination)
        row['models'][key] = identity(destination)
    for name, values in case_inputs.items():
        row['expected'][name] = {}
        for label, ticks in [('full',1024),('chunk',252)]:
            path = target / (kind + '-' + name + '-' + label + '.f32')
            module.tensor_output(model, values[:, :ticks]).tofile(path)
            row['expected'][name][label] = identity(path)
    manifest['models'].append(row)
(target / 'fixture-manifest.json').write_text(json.dumps(manifest,indent=2)+'\\n')
`;
const prepared = spawnSync("wsl.exe", ["-d", "Ubuntu", "--", python, "-c", prepareCode, wslPath(repository), source, wslPath(fixtures)], { windowsHide: true, encoding: "utf8", timeout: 60000 });
if (prepared.status !== 0) throw new Error(`Fixture preparation failed: ${prepared.stderr || prepared.stdout}`);
const fixtureManifest = JSON.parse(await readFile(join(fixtures, "fixture-manifest.json"), "utf8"));
const loadedRuntime = new Set();
const contentTypes = { ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm", ".json": "application/json" };
const server = createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, "http://localhost").pathname);
    response.setHeader("Cache-Control", "no-store");
    response.setHeader("Cross-Origin-Opener-Policy", "same-origin");
    response.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
    if (pathname === "/") {
      response.setHeader("Content-Type", "text/html");
      response.end('<!doctype html><title>Synthetic neural WASM qualification</title><script src="/runtime/ort.wasm.min.js"></script>');
      return;
    }
    const root = pathname.startsWith("/runtime/") ? runtimeRoot : pathname.startsWith("/fixtures/") ? fixtures : null;
    if (!root) throw new Error("Unknown route");
    const requested = resolve(root, pathname.split("/").slice(2).join("/"));
    const rel = relative(root, requested);
    if (!rel || rel.startsWith(`..${sep}`) || rel === "..") throw new Error("Path outside harness");
    const body = await readFile(requested);
    if (root === runtimeRoot) loadedRuntime.add(requested);
    const extension = requested.slice(requested.lastIndexOf("."));
    response.setHeader("Content-Type", contentTypes[extension] ?? "application/octet-stream");
    response.setHeader("Content-Length", body.length);
    response.end(body);
  } catch {
    response.statusCode = 404;
    response.end("Not found");
  }
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
const port = server.address().port;
let browser;
let report;
const errors = [];
try {
  browser = await chromium.launch({ executablePath: chrome, headless: true, args: ["--disable-gpu", "--no-first-run", "--no-default-browser-check"] });
  const context = await browser.newContext();
  const page = await context.newPage();
  page.on("pageerror", error => errors.push(String(error)));
  page.on("requestfailed", request => errors.push(`${request.url()}: ${request.failure()?.errorText}`));
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: "load", timeout: 30000 });
  const result = await page.evaluate(async manifest => {
    const ort = window.ort;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = `${location.origin}/runtime/`;
    const load = async filename => {
      const response = await fetch(`/fixtures/${filename}`);
      if (!response.ok) throw new Error(`Failed fixture: ${filename}`);
      return new Float32Array(await response.arrayBuffer());
    };
    const check = (actual, expected, description) => {
      if (actual.length !== expected.length) throw new Error(`Output length mismatch: ${description}`);
      let maximum = 0, sum = 0;
      for (let index = 0; index < actual.length; index++) {
        const difference = Math.abs(actual[index] - expected[index]);
        if (!Number.isFinite(actual[index]) || difference > 1e-5 + 1e-4 * Math.abs(expected[index])) throw new Error(`Numerical mismatch ${description} at ${index}: ${difference}`);
        maximum = Math.max(maximum, difference); sum += difference;
      }
      return { comparison: description, pass: true, maximumAbsoluteLogitError: maximum, meanAbsoluteLogitError: sum / actual.length };
    };
    const run = async (session, values, ticks) => (await session.run({ features: new ort.Tensor("float32", values, [1, ticks, 104]) })).logits.data;
    const stitch = async (session, values, halo) => {
      const output = new Float32Array(1024 * 3);
      for (let left = 0; left < 1024; left += 128) {
        const right = Math.min(1024, left + 128);
        const inputLeft = Math.min(Math.max(0, left - halo), 1024 - 252);
        const prediction = await run(session, values.subarray(inputLeft * 104, (inputLeft + 252) * 104), 252);
        output.set(prediction.subarray((left - inputLeft) * 3, (right - inputLeft) * 3), left * 3);
      }
      return output;
    };
    const time = async call => {
      for (let index = 0; index < 3; index++) await call();
      const timings = [];
      for (let repeat = 0; repeat < 20; repeat++) {
        const start = performance.now();
        for (let inner = 0; inner < 10; inner++) await call();
        timings.push((performance.now() - start) / 10);
      }
      timings.sort((a, b) => a - b);
      return { medianMilliseconds: (timings[9] + timings[10]) / 2, p95Milliseconds: timings[18], repeats: 20, callsPerRepeat: 10, warmupCalls: 3 };
    };
    const cases = [];
    for (const item of manifest.cases) cases.push({ ...item, values: await load(item.input.file) });
    const models = [];
    for (const item of manifest.models) {
      const start = performance.now();
      const chunk = await ort.InferenceSession.create(`/fixtures/${item.models.portableChunk.file}`, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
      const chunkSessionMilliseconds = performance.now() - start;
      const startFull = performance.now();
      const full = await ort.InferenceSession.create(`/fixtures/${item.models.fullSequenceDiagnostic.file}`, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
      const fullSessionMilliseconds = performance.now() - startFull;
      const checks = [];
      for (const input of cases) {
        const expectedFull = await load(item.expected[input.name].full.file);
        const expectedChunk = await load(item.expected[input.name].chunk.file);
        const fullOutput = await run(full, input.values, 1024);
        const chunkOutput = await run(chunk, input.values.subarray(0, 252 * 104), 252);
        const stitched = await stitch(chunk, input.values, item.haloTicks);
        checks.push(check(fullOutput, expectedFull, `${input.name}: WASM full vs PyTorch`));
        checks.push(check(chunkOutput, expectedChunk, `${input.name}: WASM chunk vs PyTorch`));
        checks.push(check(stitched, expectedFull, `${input.name}: WASM stitched vs PyTorch full`));
        checks.push(check(stitched, fullOutput, `${input.name}: WASM stitched vs WASM full`));
      }
      const values = cases[0].values;
      models.push({ kind: item.kind, status: "pass", checks, maximumAbsoluteLogitError: Math.max(...checks.map(row => row.maximumAbsoluteLogitError)), latency: {
        chunkSessionCreationMilliseconds: chunkSessionMilliseconds,
        fullSessionCreationMilliseconds: fullSessionMilliseconds,
        single252TickChunk: await time(() => run(chunk, values.subarray(0, 252 * 104), 252)),
        full1024Ticks: await time(() => run(full, values, 1024)),
        stitched1024Ticks: await time(() => stitch(chunk, values, item.haloTicks)),
      } });
      await chunk.release(); await full.release();
    }
    return { models, userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency, crossOriginIsolated, performanceEntries: performance.getEntriesByType("resource").map(row => ({ name: row.name, decodedBodySize: row.decodedBodySize, encodedBodySize: row.encodedBodySize, durationMilliseconds: row.duration })), runtimeVersion: ort.env.versions.web };
  }, fixtureManifest);
  report = { ...result, status: "pass", browserVersion: browser.version() };
} catch (error) {
  report = { status: "fail", failure: { name: error.name, message: error.message } };
} finally {
  if (browser) await browser.close();
  await new Promise(resolve => server.close(resolve));
}
const identity = async path => {
  const bytes = await readFile(path);
  return { file: basename(path), sizeBytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex") };
};
const runtimeAssets = [];
for (const path of [...loadedRuntime].sort()) {
  const bytes = await readFile(path);
  runtimeAssets.push({ ...await identity(path), gzipBytesEstimate: gzipSync(bytes).length });
}
Object.assign(report, {
  schemaVersion: 1, createdAt: new Date().toISOString(),
  scope: "synthetic untrained compact models in isolated headless desktop Chrome; WASM CPU only",
  executionProviders: ["wasm"], wasmThreads: 1, gpuUsed: false,
  physicalPhoneMeasured: false, productionAppStarted: false,
  cpu: os.cpus()[0]?.model, hostPlatform: `${os.platform()} ${os.release()}`,
  runtimePackage: { name: runtimePackage.name, version: runtimePackage.version },
  harnessSource: await identity(fileURLToPath(import.meta.url)),
  packageLock: await identity(join(harness, "package-lock.json")),
  fixtureManifest: await identity(join(fixtures, "fixture-manifest.json")),
  cpuPrerequisite: fixtureManifest.cpuQualification,
  modelFiles: fixtureManifest.models.map(row => ({ kind: row.kind, ...row.models })),
  runtimeAssets, runtimeUncompressedBytes: runtimeAssets.reduce((sum, row) => sum + row.sizeBytes, 0),
  runtimeGzipBytesEstimate: runtimeAssets.reduce((sum, row) => sum + row.gzipBytesEstimate, 0),
  browserErrors: errors,
  acceptance: { allFinite: true, atol: 1e-5, rtol: 1e-4, expectedOutputs: "seeded PyTorch CPU float32" },
  limitations: ["Desktop Chrome WASM results do not establish Android/iOS browser performance.", "No decode, feature extraction, UI, network-delivery, battery or thermal benchmark.", "Inference timing includes JS tensor construction/session.run; excludes fixture fetch and model initialization.", "Runtime assets served uncompressed locally; gzip figures are offline estimates, not measured network downloads.", "First session includes WASM initialization; later session creation benefits from an initialized runtime.", "Float32 synthetic weights only; trained-model/quantized and physical-device checks remain required."],
});
const localReport = join(harness, "browser-runtime-qualification.json");
await writeFile(localReport, `${JSON.stringify(report, null, 2)}\n`, { flag: "wx" });
const publishCode = `import pathlib,shutil,sys; source=pathlib.Path(sys.argv[1]); target=pathlib.Path(sys.argv[2]); target.mkdir(parents=True,exist_ok=False); shutil.copy2(source/'browser-runtime-qualification.json',target/'browser-runtime-qualification.json'); shutil.copy2(source/'package-lock.json',target/'package-lock.json'); shutil.copytree(source/'fixtures',target/'fixtures')`;
const published = spawnSync("wsl.exe", ["-d", "Ubuntu", "--", python, "-c", publishCode, wslPath(harness), output], { windowsHide: true, encoding: "utf8", timeout: 60000 });
if (published.status !== 0) throw new Error(`Artifact publication failed: ${published.stderr}`);
console.log(JSON.stringify({ status: report.status, output, report: await identity(localReport), models: report.models?.map(row => ({ kind: row.kind, maxLogitError: row.maximumAbsoluteLogitError, latency: row.latency })), runtimeUncompressedBytes: report.runtimeUncompressedBytes, runtimeGzipBytesEstimate: report.runtimeGzipBytesEstimate }));
if (report.status !== "pass") { console.error(report.failure); process.exitCode = 1; }
