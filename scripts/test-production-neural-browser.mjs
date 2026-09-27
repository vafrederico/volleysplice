#!/usr/bin/env node
// Isolated integration test host. Imports the actual production modules while
// serving only a private test page; it does not start the production application.
import { createRequire } from 'node:module';
import { createReadStream } from 'node:fs';
import { mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, join, dirname, extname, relative, isAbsolute } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) {
  if (!process.argv[i].startsWith('--') || process.argv[i + 1] === undefined)
    throw new Error('Arguments require explicit values.');
  args.set(process.argv[i], process.argv[i + 1]);
}
for (const key of ['--output', '--harness', '--browser', '--source']) {
  if (!args.get(key)) throw new Error(`${key} is required; resolve private inputs through the ledger/local environment.`);
}
const repo = fileURLToPath(new URL('..', import.meta.url));
const output = resolve(args.get('--output'));
const relativeOutput = relative(repo, output);
if (!relativeOutput || (!relativeOutput.startsWith('..') && !isAbsolute(relativeOutput)))
  throw new Error('Integration artifacts must remain outside the repository.');
const seconds = Number(args.get('--seconds') ?? 120);
if (!Number.isFinite(seconds) || seconds < 2) throw new Error('Use a test window of at least two seconds.');
const provider = args.get('--provider') ?? 'auto';
if (!['auto', 'wasm'].includes(provider)) throw new Error('--provider must be auto or wasm.');
const source = resolve(args.get('--source'));
const publicRoot = resolve(args.get('--public-dir') ?? process.env.VOLLEYCUT_PROD_PUBLIC_DIR ?? join(repo, 'prod/public'));
const csp = (await readFile(join(publicRoot, '_headers'), 'utf8')).split(/\r?\n/)
  .find(line => line.trim().startsWith('Content-Security-Policy:'))?.trim().slice('Content-Security-Policy:'.length).trim();
if (!csp) throw new Error('Production content security policy is missing.');
const sourceBefore = await stat(source);
const require = createRequire(join(resolve(args.get('--harness')), 'package.json'));
const { chromium } = require('playwright-core');
const productionRequire = createRequire(join(repo, 'prod/package.json'));
const { createServer } = await import(pathToFileURL(productionRequire.resolve('vite')).href);
await mkdir(output, { recursive: true });
try { await stat(join(output, 'result.json')); throw new Error('Refusing to replace a completed integration test.'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const site = join(output, 'site');
await mkdir(site, { recursive: true });
const importUrl = path => `/@fs/${join(repo, path).replaceAll('\\', '/')}`;
await writeFile(join(site, 'index.html'), '<!doctype html><title>Production neural integration test</title><input id="source" type="file"><script type="module" src="/entry.mjs"></script>');
await writeFile(join(site, 'entry.mjs'), `
import { analyzeOpenedMedia } from ${JSON.stringify(importUrl('prod/src/lib/on-device/pipeline.ts'))};
import { openLocalMedia } from ${JSON.stringify(importUrl('prod/src/lib/on-device/media.ts'))};
import { inferScoreSpecialists } from ${JSON.stringify(importUrl('prod/src/lib/on-device/score-specialists.ts'))};
import { augmentStoredAnalysisWithSuppression } from ${JSON.stringify(importUrl('prod/src/lib/on-device/production-inference.ts'))};
import { normalizeStoredProject, projectId } from ${JSON.stringify(importUrl('prod/src/lib/project-store.ts'))};
const digest = async values => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',
  values.buffer.slice(values.byteOffset, values.byteOffset + values.byteLength))), v => v.toString(16).padStart(2, '0')).join('');
window.runCase = async options => {
  const file = document.querySelector('#source').files[0];
  if (!file) throw new Error('Select the private test source.');
  const media = await openLocalMedia(file);
  const source = { name: file.name, size: file.size, lastModified: file.lastModified };
  const roi = { x: 0, y: 0, width: 1, height: 1 };
  const analysisWindow = { start: options.start ?? 0, end: Math.min(options.seconds, media.info.duration) };
  const abort = new AbortController();
  const progress = [];
  let cancelRequested = false;
  const began = performance.now();
  try {
    const analysis = await analyzeOpenedMedia(media, roi, 'local-source', undefined, value => {
      window.progress = { testCase: options.id, stage: value.stage, detail: value.detail,
        completed: value.completed, total: value.total };
      if (progress.at(-1)?.detail !== value.detail) progress.push(window.progress);
      if (options.cancel && value.stage === 'video' && value.detail?.startsWith('Reading game images') && value.completed > 0) {
        cancelRequested = true; abort.abort();
      }
    }, source, { rallyModel: options.selection, analysisWindow, signal: abort.signal,
      detailedProfiling: false, decodeStrategy: 'sequential' });
    if (options.cancel) throw new Error('Cancellation test unexpectedly completed.');
    const expectedModel = options.selection === 'high-f1' ? 'distilled-large-f1-v1' : 'distilled-large-recall-v1';
    if (analysis.modelId !== expectedModel || analysis.suppression !== undefined)
      throw new Error('Selected neural identity or suppression policy changed.');
    if (!analysis.productionServeOutputs || !analysis.productionStateOutputs || !analysis.productionComponents)
      throw new Error('Score-specialist source heads are missing.');
    if (!analysis.times.length || !analysis.rallyProbabilities.every(Number.isFinite))
      throw new Error('Neural output is empty or nonfinite.');
    if (analysis.intervals.some(v => v.start < analysisWindow.start || v.end > analysisWindow.end || v.end <= v.start))
      throw new Error('Neural intervals escape the requested game window.');
    if (await augmentStoredAnalysisWithSuppression(analysis, analysisWindow) !== null)
      throw new Error('Legacy augmentation can replace neural predictions.');
    const project = { id: projectId(source, media.info, analysisWindow, options.selection),
      source, info: media.info, analysisWindow, roi, rallyModel: options.selection, status: 'ready', analysis,
      createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(), error: null };
    const restored = normalizeStoredProject(project);
    if (restored.analysis !== analysis || restored.status !== 'ready') throw new Error('Restoring the selected neural project invalidated its analysis.');
    const identities = ['ensemble', 'high-recall', 'high-f1'].map(v => projectId(source, media.info, analysisWindow, v));
    if (new Set(identities).size !== 3) throw new Error('Detector selections collide in saved project IDs.');
    const ralliesReadyMs = performance.now() - began;
    let specialists;
    if (options.scores) {
      const scoreStart = performance.now();
      const output = await inferScoreSpecialists(media, roi, {
        servingSide: { analysis, onProgress: value => { window.progress = { testCase: options.id, ...value }; } },
        sideSwitch: { analysis: { intervals: analysis.intervals, times: analysis.times,
          deadStateProbabilities: analysis.deadStateProbabilities, productionComponents: analysis.productionComponents,
          productionStateOutputs: analysis.productionStateOutputs },
          onProgress: value => { window.progress = { testCase: options.id, ...value }; } },
      });
      if (!output.servingSide || !output.sideSwitch) throw new Error('Score-specialist output is incomplete.');
      specialists = { elapsedMs: performance.now() - scoreStart,
        servingCandidates: output.servingSide.candidates, sideSwitchCandidates: output.sideSwitch.candidates };
    }
    return { id: options.id, passed: true, modelId: analysis.modelId, analysisWindow,
      featureRows: analysis.times.length, featureColumns: analysis.featureNames.length,
      rallies: analysis.intervals, rallyProbabilities: Array.from(analysis.rallyProbabilities),
      probabilitiesSha256: await digest(analysis.rallyProbabilities),
      timesSha256: await digest(analysis.times), featureValuesSha256: await digest(analysis.featureValues),
      reusedEmbeddings: progress.some(v => v.detail?.includes('Reused image features')),
      preparedImageRows: progress.filter(v => v.detail?.startsWith('Reading game images')).length,
      observedProviders: [...new Set(progress.filter(v => v.detail?.startsWith('Reading game images'))
        .map(v => v.detail.endsWith(' - CPU') ? 'wasm' : v.detail.endsWith(' - GPU') ? 'webgpu' : 'unknown'))],
      ralliesReadyMs, allReadyMs: performance.now() - began, specialists };
  } catch (error) {
    if (options.cancel && cancelRequested && error?.name === 'AbortError')
      return { id: options.id, passed: true, cancellationObserved: true, elapsedMs: performance.now() - began };
    throw error;
  } finally { media.input.dispose(); }
};
`);

const contentType = filename => ({ '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.wasm': 'application/wasm', '.json': 'application/json' })[extname(filename)] ?? 'application/octet-stream';
const assetRequests = [];
const server = await createServer({ configFile: false, root: site, publicDir: false,
  cacheDir: join(output, 'vite-cache'),
  resolve: { alias: { mediabunny: join(repo, 'prod/node_modules/mediabunny/dist/bundles/mediabunny.mjs'),
    'fft.js': join(repo, 'prod/node_modules/fft.js/lib/fft.js') } },
  server: { host: '127.0.0.1', port: 0, hmr: false, fs: { allow: [repo, site] } },
  plugins: [{ name: 'production-neural-private-test-assets', configureServer(vite) {
    vite.middlewares.use(async (req, res, next) => {
      res.setHeader('Content-Security-Policy', csp);
      const pathname = new URL(req.url, 'http://localhost').pathname;
      if (!pathname.startsWith('/runtime/')) return next();
      try {
        const name = decodeURIComponent(pathname.slice('/runtime/'.length));
        if (!/^[a-zA-Z0-9_./-]+$/.test(name) || name.split('/').includes('..')) throw new Error('Invalid asset name.');
        const path = join(publicRoot, 'runtime', name);
        const info = await stat(path);
        if (!info.isFile()) throw new Error('Missing runtime asset.');
        assetRequests.push({ name, sizeBytes: info.size });
        res.setHeader('Content-Type', contentType(name));
        res.setHeader('Content-Length', info.size);
        const stream = createReadStream(path);
        stream.on('error', () => res.destroy());
        res.on('close', () => stream.destroy());
        stream.pipe(res);
      } catch { res.statusCode = 404; res.end('Runtime asset unavailable.'); }
    });
  } }],
});

const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const guardedFiles = (await readdir(join(repo, 'prod/src/lib'), { recursive: true }))
  .filter(name => /\.(ts|tsx|mjs|js)$/.test(name)).map(name => `prod/src/lib/${name.replaceAll('\\', '/')}`);
const snapshot = async () => Object.fromEntries(await Promise.all(guardedFiles.map(async name =>
  [name, digest(await readFile(join(repo, name)))])));
let context;
const results = [];
const errors = [];
try {
  await server.listen();
  context = await chromium.launchPersistentContext(join(output, 'profile'), {
    executablePath: args.get('--browser'), headless: args.get('--headless') !== 'false',
    // Disable the Chromium WebGPU service for the whole browser, including
    // dedicated workers. Video decoding keeps its ordinary acceleration policy.
    args: ['--no-first-run', '--no-default-browser-check',
      ...(provider === 'wasm' ? ['--disable-features=WebGPUService'] : [])],
  });
  const page = await context.newPage();
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(server.resolvedUrls.local[0]);
  await page.waitForFunction(() => typeof window.runCase === 'function');
  const capabilities = await page.evaluate(async () => ({
    crossOriginIsolated,
    webGpuAdapterAvailable: Boolean(await navigator.gpu?.requestAdapter().catch(() => null)),
  }));
  if (provider === 'wasm' && capabilities.webGpuAdapterAvailable)
    throw new Error('This Chromium version did not disable WebGPU; refusing an unverified WASM test.');
  await page.setInputFiles('#source', source);
  const hashes = await snapshot();
  const cases = [
    { id: 'recall-fresh', selection: 'high-recall', seconds, scores: args.get('--scores') !== 'false' },
    { id: 'recall-cached', selection: 'high-recall', seconds },
    { id: 'f1-fresh', selection: 'high-f1', seconds, scores: args.get('--scores') !== 'false' },
    { id: 'recall-window', selection: 'high-recall', seconds: Math.min(seconds, 8), start: .25 },
    { id: 'cancel', selection: 'high-recall', seconds: Math.min(seconds, 3), cancel: true },
  ];
  for (const options of cases) {
    const timer = setInterval(() => page.evaluate(() => window.progress ?? {})
      .then(value => console.log(JSON.stringify(value))).catch(() => {}), 15000);
    let result;
    try { result = await page.evaluate(value => window.runCase(value), options); }
    finally { clearInterval(timer); }
    results.push(result);
    if (provider === 'wasm' && result.preparedImageRows > 0
        && (result.observedProviders.length !== 1 || result.observedProviders[0] !== 'wasm'))
      throw new Error('The real neural worker did not report CPU execution.');
    await writeFile(join(output, `${options.id}.json`), JSON.stringify(result, null, 2));
    console.log(JSON.stringify({ id: result.id, passed: result.passed, modelId: result.modelId,
      rallies: result.rallies?.length, allReadyMs: result.allReadyMs, reusedEmbeddings: result.reusedEmbeddings }));
  }
  if (results[0].probabilitiesSha256 !== results[1].probabilitiesSha256
      || results[0].timesSha256 !== results[1].timesSha256
      || results[0].featureValuesSha256 !== results[1].featureValuesSha256
      || JSON.stringify(results[0].rallies) !== JSON.stringify(results[1].rallies)
      || results[0].reusedEmbeddings || !results[1].reusedEmbeddings || results[2].reusedEmbeddings)
    throw new Error('Fresh/cache/variant isolation did not preserve the selected pipeline.');
  const expected = args.get('--recall-reference');
  let parity;
  if (expected) {
    const reference = JSON.parse(await readFile(expected, 'utf8'));
    const corrected = reference.avCases?.find(value => value.id === 'corrected');
    if (!corrected || reference.analysisDurationSeconds !== seconds) throw new Error('Reference scope differs.');
    const expectedRallies = corrected.rallies.map(({ start, end }) => ({ start, end }));
    const actualRallies = results[0].rallies.map(({ start, end }) => ({ start, end }));
    if (JSON.stringify(expectedRallies) !== JSON.stringify(actualRallies)) throw new Error('Integrated recall rally boundaries differ from qualified browser reference.');
    if (![corrected.rawAvFile, corrected.timesFile, corrected.probabilitiesFile].every(name =>
      typeof name === 'string' && /^[a-z0-9-]+\.(f32|f64)$/.test(name))) throw new Error('Invalid reference tensor names.');
    const raw = await readFile(join(dirname(expected), corrected.rawAvFile));
    const times = await readFile(join(dirname(expected), corrected.timesFile));
    if (digest(raw) !== results[0].featureValuesSha256 || digest(times) !== results[0].timesSha256)
      throw new Error('Integrated AV feature/timestamp inputs differ from the qualified browser reference.');
    const probabilities = await readFile(join(dirname(expected), corrected.probabilitiesFile));
    if (probabilities.length !== results[0].rallyProbabilities.length * 16) throw new Error('Reference probability shape differs.');
    let maximumProbabilityError = 0;
    for (let row = 0; row < results[0].rallyProbabilities.length; row++) {
      const expectedProbability = probabilities.readFloatLE(row * 16);
      const error = Math.abs(results[0].rallyProbabilities[row] - expectedProbability);
      if (!Number.isFinite(error) || error > 1e-4 + 1e-4 * Math.abs(expectedProbability))
        throw new Error(`Integrated recall probability parity failed at tick ${row}.`);
      maximumProbabilityError = Math.max(maximumProbabilityError, error);
    }
    parity = { passed: true, referenceSha256: digest(await readFile(expected)), exactRallyBoundaries: true,
      exactAvFeatures: true, exactAvTimestamps: true, maximumProbabilityError };
  }
  const after = await snapshot();
  if (JSON.stringify(hashes) !== JSON.stringify(after)) throw new Error('Production modules changed during integration testing.');
  const sourceAfter = await stat(source);
  if (sourceAfter.size !== sourceBefore.size || sourceAfter.mtimeMs !== sourceBefore.mtimeMs)
    throw new Error('Test source changed during analysis.');
  if (errors.length) throw new Error('Browser emitted unhandled errors.');
  const summary = { schemaVersion: 1, kind: 'production-neural-browser-integration-v1', passed: true,
    browserVersion: context.browser()?.version(), requestedProvider: provider, capabilities, productionCspApplied: true, seconds, results, parity,
    sourceGuard: { passed: true, hashes }, assetRequests, errors,
    limitations: ['Desktop browser integration, not a physical-phone timing.',
      'The isolated page imports production modules and workers; it does not render the production React application.',
      'Repeated inference uses the browser feature caches intentionally.'] };
  await writeFile(join(output, 'result.json'), JSON.stringify(summary, null, 2));
  console.log(JSON.stringify({ passed: true, cases: results.length, referenceParity: Boolean(parity) }));
} catch (error) {
  await writeFile(join(output, 'failure.private.json'), JSON.stringify({ message: error.message, stack: error.stack, results, errors }, null, 2));
  throw error;
} finally { await context?.close(); await server.close(); }
