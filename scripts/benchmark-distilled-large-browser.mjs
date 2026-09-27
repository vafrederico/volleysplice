#!/usr/bin/env node
// Isolated real-video experiment; all private source/model/output paths are
// supplied at runtime from the external ledger or ignored local environment.
import { createRequire } from 'node:module';
import { createReadStream } from 'node:fs';
import { mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, join, basename, extname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import os from 'node:os';

const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) {
  if (!process.argv[i].startsWith('--') || process.argv[i + 1] === undefined) throw new Error('Arguments require explicit values');
  args.set(process.argv[i], process.argv[i + 1]);
}
for (const key of ['--output', '--harness', '--runtime', '--browser', '--graphs', '--selection'])
  if (!args.get(key)) throw new Error(`${key} is required (resolve private ledger/local environment)`);
const validateOnly = args.get('--validate-only') === 'true';
const operatorOnly = args.get('--operator-only') === 'true';
if (operatorOnly && !validateOnly) throw new Error('--operator-only requires --validate-only true');
if (!validateOnly && !args.get('--source')) throw new Error('--source is required for real-video runs');
if (validateOnly && !args.get('--fixtures')) throw new Error('--fixtures is required for validation');
if (!validateOnly && args.has('--fixtures')) throw new Error('Run fixture qualification separately so it cannot warm the measured model/runtime');
const selection = args.get('--selection'), provider = args.get('--provider') ?? 'wasm';
const sourceTransport = args.get('--source-transport') ?? 'file';
if (!['file', 'url'].includes(sourceTransport)) throw new Error('--source-transport must be file or url');
if (!['recall', 'f1'].includes(selection) || !['wasm', 'webgpu'].includes(provider)) throw new Error('Unknown selection or provider');
const wasmThreads = Number(args.get('--threads') ?? 4);
if (!Number.isInteger(wasmThreads) || wasmThreads < 1 || wasmThreads > 8) throw new Error('Invalid WASM thread count');
const isolationText = args.get('--isolation') ?? 'true';
if (!['true', 'false'].includes(isolationText)) throw new Error('--isolation must be true or false');
const isolation = isolationText === 'true';
if (!isolation && wasmThreads !== 1) throw new Error('Nonisolated runs require --threads 1');
const isolationHeaders = isolation
  ? { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' }
  : {};
const gpuLayout = args.get('--gpu-layout') ?? 'default';
const graphOptimizationLevel = args.get('--graph-optimization') ?? 'all';
if (!['default', 'NCHW', 'NHWC'].includes(gpuLayout) || (provider !== 'webgpu' && gpuLayout !== 'default'))
  throw new Error('Explicit GPU layout requires webgpu and NCHW or NHWC');
if (!['all', 'disabled'].includes(graphOptimizationLevel)) throw new Error('Graph optimization must be all or disabled');
const visualCases = (args.get('--visual-cases') ?? 'legacy,corrected').split(',');
if (!visualCases.length || visualCases.some(id => !['legacy', 'corrected'].includes(id))) throw new Error('Invalid visual cases');
const seconds = args.has('--seconds') ? Number(args.get('--seconds')) : undefined;
if (seconds !== undefined && (!Number.isFinite(seconds) || seconds <= 0)) throw new Error('Invalid duration');
const repo = fileURLToPath(new URL('..', import.meta.url));
const output = resolve(args.get('--output')), graphRoot = resolve(args.get('--graphs'));
const source = args.has('--source') ? resolve(args.get('--source')) : null;
const fixtures = args.has('--fixtures') ? resolve(args.get('--fixtures')) : null;
if (output === resolve(repo) || output.startsWith(resolve(repo) + (process.platform === 'win32' ? '\\' : '/')))
  throw new Error('Experiment artifacts must remain outside the repository');
const require = createRequire(join(resolve(args.get('--harness')), 'package.json'));
const { chromium } = require('playwright-core');
const { createServer } = await import(pathToFileURL(require.resolve('vite')).href);
const runtimeRoot = resolve(args.get('--runtime'));
const runtimePackage = JSON.parse(await readFile(join(runtimeRoot, '..', 'package.json'), 'utf8'));
if (runtimePackage.version !== '1.22.0') throw new Error('Expected pinned onnxruntime-web 1.22.0');
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const contract = JSON.parse(await readFile(join(graphRoot, 'input-contract.json'), 'utf8'));
const graphFiles = ['mobile-large-encoder-fp32.onnx', 'mobile-large-tcn-dynamic-fp32.onnx',
  'mobile-large-pipeline.json', 'mobile-large-encoder-pool_weights.f32'];
const graphAssets = {};
for (const name of graphFiles) {
  const bytes = await readFile(join(graphRoot, name));
  const sha256 = digest(bytes);
  if (sha256 !== contract.hashes[name]) throw new Error(`Frozen graph contract changed: ${name}`);
  graphAssets[name] = { sizeBytes: bytes.byteLength, sha256 };
}
if (contract.modelIdentity !== 'dino-distilled-mobilenet-v3-large-tcn' || contract.selectionMode !== selection)
  throw new Error('Frozen selection contract differs');
const videoStat = source ? await stat(source) : null;
await mkdir(output, { recursive: true });
try { await stat(join(output, 'result.json')); throw new Error('Refusing to overwrite a completed experiment'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const root = join(output, 'site');
await mkdir(root, { recursive: true });
const moduleUrl = name => `/@fs/${join(repo, 'scripts', name).replaceAll('\\', '/')}`;
const scores = args.get('--scores') === 'true';
const runnerSourceNames = ['benchmark-distilled-large-browser.mjs', 'browser-distilled-large.mjs',
  'browser-distilled-large-contract.mjs', ...(scores ? ['browser-distilled-large-scores.mjs'] : [])];
const productionSourceNames = (await readdir(join(repo, 'prod/src/lib'), { recursive: true }))
  .filter(name => /\.(ts|tsx|js|mjs)$/.test(name)).map(name => name.replaceAll('\\', '/')).sort();
const guardedSourceNames = [...runnerSourceNames.map(name => `scripts/${name}`),
  ...productionSourceNames.map(name => `prod/src/lib/${name}`),
  'prod/node_modules/mediabunny/dist/bundles/mediabunny.mjs', 'prod/node_modules/fft.js/lib/fft.js'];
const snapshotBrowserSources = async () => Object.fromEntries(await Promise.all(guardedSourceNames
  .map(async name => [name, digest(await readFile(join(repo, name)))])));
let sourceStartHashes;
const verifyBrowserSources = async () => {
  const end = await snapshotBrowserSources();
  const changed = guardedSourceNames.filter(name => sourceStartHashes[name] !== end[name]);
  if (changed.length) throw new Error(`Browser source changed during measurement: ${changed.join(', ')}`);
  return { passed: true, fileCount: guardedSourceNames.length, hashes: sourceStartHashes };
};
await writeFile(join(root, 'index.html'), '<!doctype html><title>Distilled Large video experiment</title><input type="file" id="benchmark-source-file" hidden><script src="/ort/ort.all.min.js"></script><script type="module" src="/entry.mjs"></script>');
await writeFile(join(root, 'entry.mjs'), `
import {runBrowserExperiment,validateFixtures,validateOperatorFixtures} from ${JSON.stringify(moduleUrl('browser-distilled-large.mjs'))};
${scores ? `import {runScoreModels} from ${JSON.stringify(moduleUrl('browser-distilled-large-scores.mjs'))};` : ''}
window.run = options => runBrowserExperiment(options, ${scores ? 'runScoreModels' : 'undefined'});
window.validate = validateFixtures;
window.validateOperators = validateOperatorFixtures;
`);
const servedRuntime = new Set(), servedProductionRuntime = new Set(), artifactFiles = {};
const sourceIo = { requests: 0, requestedBytes: 0, clampedRanges: 0, rejectedRanges: 0,
  sourceBytesRead: sourceTransport === 'file' ? null : 0,
  sourceBytesReadMeaning: sourceTransport === 'file' ? 'Browser BlobSource filesystem reads are not measured by the HTTP host.' : 'ReadStream bytes read across all HTTP media responses, including canceled read-ahead.',
  requestedBytesMeaning: 'Sum of declared HTTP response ranges before read-ahead cancellation; not bytes transferred.',
  closedBeforeFinish: 0, streamErrors: [], failedBrowserRequests: [] };
const contentType = name => ({ '.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm',
  '.json': 'application/json' })[extname(name)] ?? 'application/octet-stream';
const server = await createServer({
  configFile: false, root, publicDir: false, cacheDir: join(output, 'vite-cache'),
  resolve: { alias: {
    mediabunny: join(repo, 'prod/node_modules/mediabunny/dist/bundles/mediabunny.mjs'),
    'fft.js': join(repo, 'prod/node_modules/fft.js/lib/fft.js'),
  } },
  server: { host: '127.0.0.1', port: 0, hmr: false, fs: { allow: [repo, root] },
    headers: isolationHeaders },
  plugins: [{ name: 'distilled-private-fixtures', configureServer(server) {
    server.middlewares.use(async (req, res, next) => {
      try {
        const pathname = new URL(req.url, 'http://localhost').pathname;
        for (const [name, value] of Object.entries(isolationHeaders)) res.setHeader(name, value);
        if (pathname === '/fixture.mp4' && source) {
          if (sourceTransport !== 'url') { sourceIo.requests++; throw new Error('HTTP media forbidden for File mode'); }
          // The source models a user-selected local file. Avoid duplicating it
          // in Chrome's disk cache (especially a NAS-backed research profile).
          res.setHeader('Cache-Control', 'no-store');
          const range = /^bytes=(\d+)-(\d*)$/.exec(req.headers.range ?? '');
          const start = range ? Number(range[1]) : 0;
          const requestedEnd = range?.[2] ? Number(range[2]) : videoStat.size - 1;
          const end = Math.min(requestedEnd, videoStat.size - 1);
          sourceIo.requests++;
          if (start > end || start >= videoStat.size) {
            sourceIo.rejectedRanges++; res.statusCode = 416;
            res.setHeader('Content-Range', `bytes */${videoStat.size}`); res.end(); return;
          }
          if (requestedEnd !== end) sourceIo.clampedRanges++;
          sourceIo.requestedBytes += end - start + 1;
          res.setHeader('Content-Type', 'video/mp4'); res.setHeader('Accept-Ranges', 'bytes');
          res.setHeader('Content-Length', end - start + 1);
          if (range) { res.statusCode = 206; res.setHeader('Content-Range', `bytes ${start}-${end}/${videoStat.size}`); }
          const stream = createReadStream(source, { start, end });
          stream.on('close', () => { sourceIo.sourceBytesRead += stream.bytesRead; });
          req.on('aborted', () => stream.destroy());
          res.on('close', () => {
            if (!res.writableFinished) sourceIo.closedBeforeFinish++;
            stream.destroy();
          });
          stream.on('error', error => {
            sourceIo.streamErrors.push({ start, end, code: error.code ?? error.name });
            res.destroy(error);
          });
          stream.pipe(res); return;
        }
        if (pathname.startsWith('/artifacts/')) {
          const name = pathname.slice('/artifacts/'.length);
          if (req.method !== 'PUT' || !/^[a-z0-9-]+\.(f32|f64|u8)$/.test(name) || artifactFiles[name])
            throw new Error('Invalid artifact publication');
          const chunks = []; let length = 0;
          for await (const chunk of req) {
            length += chunk.byteLength;
            if (length > 128 * 1024 * 1024) throw new Error('Artifact exceeds experiment limit');
            chunks.push(chunk);
          }
          const bytes = Buffer.concat(chunks);
          await writeFile(join(output, name), bytes, { flag: 'wx' });
          artifactFiles[name] = { sizeBytes: bytes.byteLength, sha256: digest(bytes) };
          res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify(artifactFiles[name])); return;
        }
        for (const [prefix, folder, allowed] of [
          ['/runtime/', join(repo, 'prod/public/runtime'), null], ['/ort/', runtimeRoot, null],
          ['/graphs/', graphRoot, graphFiles], ['/fixtures/', fixtures, null],
        ]) {
          if (!pathname.startsWith(prefix) || !folder) continue;
          const name = pathname.slice(prefix.length);
          if (basename(name) !== name || !name || (allowed && !allowed.includes(name))) throw new Error('Invalid asset');
          const bytes = await readFile(join(folder, name));
          if (prefix === '/graphs/' && digest(bytes) !== graphAssets[name].sha256) throw new Error('Graph changed after validation');
          if (prefix === '/ort/') servedRuntime.add(name);
          if (prefix === '/runtime/') servedProductionRuntime.add(name);
          res.setHeader('Content-Type', contentType(name)); res.end(bytes); return;
        }
        next();
      } catch { res.statusCode = 400; res.end('Fixture request rejected'); }
    });
  } }],
});
let context, page;
const errors = [], warnings = [], consoleDiagnostics = [], heapSamples = [];
try {
  await server.listen();
  context = await chromium.launchPersistentContext(join(output, 'profile'), {
    executablePath: args.get('--browser'), headless: args.get('--headless') !== 'false',
    args: ['--no-first-run', '--no-default-browser-check'],
  });
  page = await context.newPage();
  const cdp = await context.newCDPSession(page);
  await cdp.send('Performance.enable');
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'warning' && warnings.length < 100) warnings.push(message.text());
    if (['warning', 'error'].includes(message.type()) && consoleDiagnostics.length < 100)
      consoleDiagnostics.push({ type: message.type(), message: message.text() });
  });
  page.on('requestfailed', request => {
    if (new URL(request.url()).pathname === '/fixture.mp4') sourceIo.failedBrowserRequests.push({
      range: request.headers().range ?? null, error: request.failure()?.errorText ?? null });
  });
  sourceStartHashes = await snapshotBrowserSources();
  await page.goto(server.resolvedUrls.local[0]);
  await page.waitForFunction(() => typeof window.run === 'function');
  if (!validateOnly && sourceTransport === 'file') await page.setInputFiles('#benchmark-source-file', source);
  const options = { selection, provider, wasmThreads, isolation, gpuLayout, graphOptimizationLevel, seconds, visualCases, sourceTransport,
    validateEncoder: args.get('--validate-encoder') === 'true' };
  if (fixtures) {
    const validation = await page.evaluate(({ options, operatorOnly }) => operatorOnly
      ? window.validateOperators(options) : window.validate(options), { options, operatorOnly });
    Object.assign(validation, { selection, graphAssets, graphRewrite: contract.graphRewrite ?? null,
      browserVersion: context.browser()?.version(),
      codeSha256: sourceStartHashes['scripts/benchmark-distilled-large-browser.mjs'], sourceGuard: await verifyBrowserSources() });
    await writeFile(join(output, operatorOnly ? 'operator-validation.json' : 'fixture-validation.json'), JSON.stringify(validation, null, 2));
    console.log(JSON.stringify({ fixtureValidation: validation.passed, cases: validation.cases.length,
      ...(operatorOnly ? { operatorResults: validation.cases } : {}) }));
    if (!validation.passed) process.exitCode = 1;
  }
  if (!validateOnly) {
    const timer = setInterval(async () => {
      try {
        const progress = await page.evaluate(() => window.progress ?? {});
        const { metrics } = await cdp.send('Performance.getMetrics');
        const values = Object.fromEntries(metrics.map(row => [row.name, row.value]));
        heapSamples.push({ ...progress, jsHeapUsedBytes: values.JSHeapUsedSize, jsHeapTotalBytes: values.JSHeapTotalSize });
        console.log(JSON.stringify(progress));
      } catch {}
    }, 15000);
    let result;
    try { result = await page.evaluate(options => window.run(options), options); }
    finally { clearInterval(timer); }
    if (JSON.stringify(result.artifacts) !== JSON.stringify(artifactFiles)) throw new Error('Browser/host artifact receipt mismatch');
    if (source) {
      const after = await stat(source);
      if (after.size !== videoStat.size || after.mtimeMs !== videoStat.mtimeMs) throw new Error('Source changed while benchmarking');
    }
    const runtimeFiles = {};
    for (const name of [...servedRuntime].sort()) {
      const bytes = await readFile(join(runtimeRoot, name));
      runtimeFiles[name] = { sizeBytes: bytes.byteLength, sha256: digest(bytes) };
    }
    const productionRuntimeFiles = {};
    for (const name of [...servedProductionRuntime].sort()) {
      const bytes = await readFile(join(repo, 'prod/public/runtime', name));
      productionRuntimeFiles[name] = { sizeBytes: bytes.byteLength, sha256: digest(bytes) };
    }
    const sourceGuard = await verifyBrowserSources();
    Object.assign(result, { createdAt: new Date().toISOString(),
      host: { browserVersion: context.browser()?.version(), nodeVersion: process.version, platform: process.platform,
        architecture: process.arch, cpu: os.cpus()[0]?.model },
      source: { transport: sourceTransport, recordingIndex: args.get('--recording-index') ?? null, sizeBytes: videoStat.size,
        sha256: args.get('--source-sha256') ?? null, sha256Basis: args.has('--source-sha256') ? 'caller-provided prior verified receipt' : 'not independently hashed',
        sourceOffsetSeconds: Number(args.get('--source-offset') ?? 0) },
      graphAssets, graphRewrite: contract.graphRewrite ?? null,
      sourceGuard,
      runtimeFiles, productionRuntimeFiles, browserErrors: errors, browserWarnings: warnings,
      consoleDiagnostics, sourceIo,
      timingIntegrity: {
        sourceTransport,
        sourceCachePolicy: sourceTransport === 'file' ? 'BlobSource filesystem reads; no HTTP media transport' : 'no-store',
        clean: sourceIo.streamErrors.length === 0 && sourceIo.rejectedRanges === 0
          && (sourceTransport !== 'file' || sourceIo.requests === 0)
          && !consoleDiagnostics.some(row => row.message.includes('Retrying failed fetch'))
          && !sourceIo.failedBrowserRequests.some(row => row.error !== 'net::ERR_ABORTED'),
        note: sourceTransport === 'file'
          ? 'Actual browser File input with shipped openLocalMedia/BlobSource; unexpected HTTP media requests or fetch retries invalidate clean timing. Filesystem byte totals are unmeasured.'
          : 'Intentional read-ahead request cancellation is allowed; retries, range errors, and other fetch failures invalidate clean timing.' },
      memory: { metric: 'CDP JavaScript heap only; excludes WASM linear memory and GPU allocations', samples: heapSamples,
        maximumSampledJsHeapUsedBytes: heapSamples.length ? Math.max(...heapSamples.map(s => s.jsHeapUsedBytes)) : null },
      code: Object.fromEntries(runnerSourceNames.map(name => [name, sourceStartHashes[`scripts/${name}`]])),
      productionCode: Object.fromEntries(productionSourceNames.filter(name => name.startsWith('on-device/'))
        .map(name => [name.slice('on-device/'.length), sourceStartHashes[`prod/src/lib/${name}`]])) });
    await writeFile(join(output, 'result.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify({ completed: true, selection, provider, duration: result.analysisDurationSeconds,
      embeddingMs: result.embedding.timings.pipelineMs,
      cases: result.avCases.map(c => ({ id: c.id, rows: c.rows, rallies: c.rallies.length, ralliesReadyMs: c.timings.neuralRalliesReadyMs })) }));
    if (errors.length) throw new Error('Browser errors recorded; inspect private result');
  }
} catch (error) {
  const diagnostics = page ? await page.evaluate(() => window.fixtureDiagnostics ?? null).catch(() => null) : null;
  await writeFile(join(output, 'failure.json'), JSON.stringify({ type: error.name, message: error.message, errors, warnings, diagnostics }, null, 2));
  throw error;
} finally {
  await context?.close(); await server.close();
}
