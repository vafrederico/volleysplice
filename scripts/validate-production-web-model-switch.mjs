// Real built-app test. All private paths are supplied from the external ledger.
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { resolve, join, extname, relative, isAbsolute } from 'node:path';
import { fileURLToPath } from 'node:url';

const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i], process.argv[i + 1]);
for (const name of ['source', 'output', 'build', 'harness', 'browser', 'recording-index'])
  if (!args.get(`--${name}`)) throw new Error(`--${name} is required`);
const repo = fileURLToPath(new URL('..', import.meta.url));
const output = resolve(args.get('--output')), build = resolve(args.get('--build'));
const outputRelative = relative(repo, output);
assert.ok(outputRelative.startsWith('..') || isAbsolute(outputRelative), 'Keep artifacts outside Git');
await mkdir(output, { recursive: true });
try { await stat(join(output, 'result.json')); throw new Error('Refusing to replace completed validation'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const require = createRequire(join(resolve(args.get('--harness')), 'package.json'));
const { chromium } = require('playwright-core');
const headers = await readFile(join(build, '_headers'), 'utf8');
const csp = headers.split(/\r?\n/).find(v => v.trim().startsWith('Content-Security-Policy:'))?.trim().slice(24).trim();
assert.ok(csp);
const requests = [], errors = [], results = [];
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json',
  '.wasm': 'application/wasm', '.css': 'text/css', '.svg': 'image/svg+xml', '.png': 'image/png', '.woff2': 'font/woff2' };
const server = createServer(async (req, res) => {
  try {
    const path = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    const file = resolve(build, `.${path === '/' ? '/index.html' : path}`);
    const rel = relative(build, file);
    if (rel.startsWith('..') || isAbsolute(rel)) throw new Error('Invalid asset');
    const info = await stat(file);
    if (!info.isFile()) throw new Error('Missing file');
    requests.push(path);
    res.writeHead(200, { 'Content-Type': mime[extname(file)] ?? 'application/octet-stream',
      'Content-Length': info.size, 'Content-Security-Policy': csp });
    const stream = createReadStream(file);
    stream.on('error', () => res.destroy()); res.on('close', () => stream.destroy()); stream.pipe(res);
  } catch { res.writeHead(404); res.end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let context;
try {
  context = await chromium.launchPersistentContext(join(output, 'profile'), {
    executablePath: args.get('--browser'), headless: true, viewport: { width: 1440, height: 1100 },
    args: ['--no-first-run', '--no-default-browser-check'],
  });
  const page = await context.newPage();
  page.on('pageerror', error => errors.push(error.message));
  await page.addInitScript(() => {
    localStorage.setItem('volleycut:guided-tour:v11', 'dismissed');
    localStorage.setItem('volleycut:rally-desk-tour:v1', 'dismissed');
    window.validationProgress = [];
    let previous = '';
    const record = () => {
      const steps = [...document.querySelectorAll('[role="progressbar"]')].map(bar => ({
        label: bar.getAttribute('aria-label'), percent: Number(bar.getAttribute('aria-valuenow')),
        status: bar.closest('article')?.getAttribute('data-status'),
        detail: bar.closest('article')?.querySelector('small')?.textContent,
      })).filter(v => v.label?.endsWith(' progress'));
      const key = JSON.stringify(steps);
      if (steps.length && key !== previous) {
        previous = key; window.validationProgress.push({ elapsedMs: performance.now(), steps });
      }
    };
    new MutationObserver(record).observe(document, { subtree: true, childList: true, attributes: true, characterData: true });
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const capabilities = await page.evaluate(async () => ({
    webGpuAvailable: Boolean(await navigator.gpu?.requestAdapter()), crossOriginIsolated,
  }));
  console.log(JSON.stringify({ event: 'started', capabilities }));
  const projects = async (full = false) => page.evaluate(async full => {
    const db = await new Promise((resolve, reject) => {
      const request = indexedDB.open('volleycut-projects');
      request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
    });
    try {
      if (!db.objectStoreNames.contains('projects')) return [];
      const values = await new Promise((resolve, reject) => {
        const request = db.transaction('projects').objectStore('projects').getAll();
        request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
      });
      return values.map(p => ({ selection: p.rallyModel, status: p.status, error: p.error,
        ...(full ? { duration: p.info.duration, window: p.analysisWindow, modelId: p.analysis?.modelId,
          intervals: p.analysis?.intervals, probabilities: Array.from(p.analysis?.rallyProbabilities ?? []),
          featureRows: p.analysis?.times.length, servingCandidates: p.analysis?.servingSide?.candidates,
          sideSwitchCandidates: p.analysis?.sideSwitch?.candidates,
          suppressionPresent: Boolean(p.analysis?.suppression),
        } : {}) }));
    } finally { db.close(); }
  }, full);
  const selectNew = async () => {
    await page.getByRole('combobox', { name: /^(Selected project|Current review project)$/ }).selectOption('__new__');
  };
  const run = async (selection, id) => {
    await page.locator('input[type="file"][accept^="video/"]').setInputFiles(args.get('--source'));
    await page.getByRole('combobox', { name: 'Rally detection model' }).waitFor({ timeout: 60000 });
    await page.getByRole('combobox', { name: 'Rally detection model' }).selectOption(selection);
    await page.getByRole('checkbox', { name: /Teams change court sides/ }).check();
    await page.evaluate(() => { window.validationProgress = []; });
    const requestStart = requests.length, began = Date.now();
    await page.getByRole('button', { name: 'Find the rallies', exact: true }).click();
    let project;
    for (let poll = 0; poll < 1440; poll++) {
      await page.waitForTimeout(2500);
      project = (await projects()).find(p => p.selection === selection);
      if (project?.status === 'error') throw new Error(`${id}: ${project.error}`);
      if (project?.status === 'ready') break;
      if (poll % 6 === 0) console.log(JSON.stringify({ id, seconds: (Date.now() - began) / 1000,
        progress: await page.evaluate(() => window.validationProgress.at(-1)) }));
    }
    assert.equal(project?.status, 'ready', `${id} must complete`);
    const allReadyMs = Date.now() - began;
    project = (await projects(true)).find(p => p.selection === selection);
    assert.equal(project.modelId, selection === 'high-f1' ? 'distilled-large-f1-v1' : 'distilled-large-recall-v1');
    assert.equal(project.window.start, 0); assert.equal(project.window.end, project.duration);
    assert.ok(project.duration > 1000, 'Expected the full benchmark recording');
    assert.ok(project.probabilities.length > 0 && project.probabilities.every(Number.isFinite));
    assert.ok(project.intervals.length > 0); assert.equal(project.suppressionPresent, false);
    assert.equal(project.servingCandidates.length, project.intervals.length, 'Every rally gets a serving-side result');
    assert.ok(Array.isArray(project.sideSwitchCandidates));
    const progress = await page.evaluate(() => window.validationProgress);
    const imageUpdates = progress.flatMap(p => p.steps).filter(p => p.detail?.startsWith('Reading game images'));
    const rallyUpdates = progress.flatMap(p => p.steps).filter(p => p.label === 'Finding rallies progress' && p.status === 'running');
    const fractions = [...new Set(rallyUpdates.map(p => p.percent))];
    assert.ok(fractions.length >= 3, 'Rally progress must advance across chunks');
    assert.ok(fractions.every((v, i) => i === 0 || v >= fractions[i - 1]), 'Rally bar must not go backwards');
    assert.ok(progress.some(p => p.steps.some(s => /Loading the rally classifier|Finding rallies -/.test(s.detail))), 'Classifier detail visible');
    if (id.endsWith('fresh')) assert.ok(imageUpdates.length > 10, 'Image processing must visibly advance');
    if (id.endsWith('cached')) {
      assert.equal(imageUpdates.length, 0, 'Cached model must not encode images again');
      assert.ok(progress.some(p => p.steps.some(s => /Reus(ed|ing).*image features/.test(s.detail))), 'Cache reuse visible');
    }
    await page.waitForSelector('.taste-root', { timeout: 30000 });
    await page.screenshot({ path: join(output, `${id}.private.png`), fullPage: true });
    const result = { id, ...project, allReadyMs, imageProgressUpdates: imageUpdates.length,
      rallyProgressPercents: fractions, requests: requests.slice(requestStart).filter(p => /rally-models/.test(p)), progress };
    results.push(result);
    await writeFile(join(output, `${id}.json`), JSON.stringify(result));
    console.log(JSON.stringify({ id, passed: true, allReadyMs, rallies: project.intervals.length,
      servingCandidates: project.servingCandidates.length, rallyProgressPercents: fractions }));
  };
  await run('high-f1', 'balanced-fresh');
  await selectNew();
  await run('high-recall', 'coverage-fresh');
  // Switching back through normal UI must open the saved result without inference.
  await selectNew();
  await page.locator('input[type="file"][accept^="video/"]').setInputFiles(args.get('--source'));
  await page.getByRole('combobox', { name: 'Rally detection model' }).selectOption('high-f1');
  const restoreStart = Date.now();
  await page.getByRole('button', { name: 'Find the rallies', exact: true }).click();
  await page.waitForSelector('.taste-root', { timeout: 30000 });
  const savedSwitchMs = Date.now() - restoreStart;
  assert.deepEqual((await projects(true)).find(p => p.selection === 'high-f1').intervals, results[0].intervals);
  // In this isolated test profile, evict only the saved F1 project to force a
  // rerun through the UI while retaining AV/embedding caches. This is test setup,
  // not a claim that the product exposes a force-rerun button for ready projects.
  await page.evaluate(async () => {
    const db = await new Promise((resolve, reject) => { const r = indexedDB.open('volleycut-projects'); r.onsuccess = () => resolve(r.result); r.onerror = () => reject(r.error); });
    await new Promise((resolve, reject) => {
      const tx = db.transaction('projects', 'readwrite'), store = tx.objectStore('projects'), cursor = store.openCursor();
      cursor.onsuccess = () => { const row = cursor.result; if (!row) return; if (row.value.rallyModel === 'high-f1') row.delete(); row.continue(); };
      tx.oncomplete = resolve; tx.onerror = () => reject(tx.error);
    });
    db.close(); localStorage.removeItem('volleycut:selected-project:v1');
  });
  await page.reload();
  await selectNew();
  await run('high-f1', 'balanced-cached');
  assert.deepEqual(results[2].intervals, results[0].intervals);
  assert.deepEqual(results[2].probabilities, results[0].probabilities);
  assert.equal(errors.length, 0, errors.join('\n'));
  await writeFile(join(output, 'result.json'), JSON.stringify({ passed: true, recordingIndex: args.get('--recording-index'),
    browserVersion: context.browser()?.version(), capabilities, savedSwitchMs, results, errors,
    limitations: ['Desktop Chrome, not a physical mobile-browser timing.', 'Cached rerun deliberately evicts only the saved project in an isolated test profile.'] }));
  console.log(JSON.stringify({ passed: true, cases: results.length, savedSwitchMs }));
} catch (error) {
  await writeFile(join(output, 'failure.private.json'), JSON.stringify({ message: error.message, stack: error.stack, errors, results }));
  throw error;
} finally {
  await context?.close();
  await new Promise(resolve => server.close(resolve));
}
