#!/usr/bin/env node
// Isolated loopback harness: real media through the shipped browser AV extractor
// and production ensemble. Resolve private inputs through local ledger/env first.
import { createRequire } from "node:module";
import { createReadStream } from "node:fs";
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { resolve, join, basename } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i], process.argv[i + 1]);
for (const key of ["--source", "--output", "--harness", "--browser"])
  if (!args.get(key)) throw new Error(`${key} is required (use private ledger/local environment)`);
const repo = fileURLToPath(new URL("..", import.meta.url));
const output = resolve(args.get("--output"));
const source = resolve(args.get("--source"));
const moduleRoot = args.has("--module-root") ? resolve(args.get("--module-root")) : join(repo, "prod/src/lib/on-device");
const require = createRequire(join(resolve(args.get("--harness")), "package.json"));
const { chromium } = require("playwright-core");
const { createServer } = await import(pathToFileURL(require.resolve("vite")).href);
await mkdir(output, { recursive: true });
const root = join(output, "site");
await mkdir(root, { recursive: true });
const moduleUrl = name => `/@fs/${join(moduleRoot, name).replaceAll("\\", "/")}`;
await writeFile(join(root, "index.html"), '<!doctype html><title>AV extraction validation</title><script type="module" src="/entry.ts"></script>');
await writeFile(join(root, "entry.ts"), `
import {openUrlMedia} from ${JSON.stringify(moduleUrl("media.ts"))};
import {extractBrowserFeatures} from ${JSON.stringify(moduleUrl("pipeline.ts"))};
import {runProductionInferenceFromFeatures} from ${JSON.stringify(moduleUrl("production-inference.ts"))};
import {VisualFeatureWorkerClient} from ${JSON.stringify(moduleUrl("visual-feature-worker-client.ts"))};
import {loadOpenCv, extractVisualFeaturesFromImageData} from ${JSON.stringify(moduleUrl("visual-features.ts"))};
import {WasmVisualFeatureReducer} from ${JSON.stringify(moduleUrl("visual-feature-reductions-wasm.ts"))};
window.validateVisual = async () => {
  const cv = await loadOpenCv();
  const reducer = await WasmVisualFeatureReducer.load('/runtime/feature-reductions.wasm');
  const source = new OffscreenCanvas(1920,1080), sc = source.getContext('2d');
  const pixels = sc.createImageData(source.width,source.height);
  for (let y=0;y<source.height;y++) for(let x=0;x<source.width;x++) {
    const i=(y*source.width+x)*4;
    pixels.data.set([(x%7)*41,(y%5)*59,((x+y)%3)*127,255],i);
  }
  sc.putImageData(pixels,0,0);
  const cases=[];
  for (const rotation of [0,90,180,270]) for (const fractional of [false,true]) {
    const rotated = new OffscreenCanvas(rotation%180 ? 1080:1920,rotation%180 ? 1920:1080);
    const rc=rotated.getContext('2d');
    rc.translate(rotated.width/2,rotated.height/2);rc.rotate(rotation*Math.PI/180);
    rc.drawImage(source,-source.width/2,-source.height/2);
    const crop={left:fractional?7:0,top:fractional?11:0,
      width:rotated.width-(fractional?23:0),height:rotated.height-(fractional?27:0)};
    const expected=extractVisualFeaturesFromImageData(cv,
      rc.getImageData(crop.left,crop.top,crop.width,crop.height),null,0,true,reducer);
    const worker=await VisualFeatureWorkerClient.create(true,'wasm',true);
    try {
      const result=await worker.extract(new VideoFrame(source,{timestamp:0}),
        {timestamp:0,duration:.25,rotation,crop});
      const actual=new Float32Array(result.values);
      const maxError=Math.max(...actual.map((v,i)=>Math.abs(v-expected.values[i])));
      if (!Number.isFinite(maxError) || maxError>2e-6) throw new Error('Area/rotation parity failed: '+maxError);
      cases.push({rotation,fractional,maxError});
    } finally { expected.gray.delete();worker.dispose(); }
  }
  return {passed:true,cases};
};
window.run = async (options) => {
  const media = await openUrlMedia('/fixture.mp4');
  const createWorker = VisualFeatureWorkerClient.create;
  if (options.forceFallback) VisualFeatureWorkerClient.create = async () => {throw new Error('Validation: worker unavailable');};
  try {
    const begin = performance.now();
    const sequence = await extractBrowserFeatures(media, {x:0,y:0,width:1,height:1}, undefined,
      p => { window.progress = {stage:p.stage, completed:p.completed}; }, undefined, options);
    const featuresMs = performance.now()-begin;
    const inferStart = performance.now();
    const result = await runProductionInferenceFromFeatures(sequence, {start:0,end:media.info.duration});
    return {featuresMs, inferenceMs:performance.now()-inferStart, performance:sequence.performance,
      rows:sequence.rows, columns:sequence.columns, names:sequence.names,
      times:Array.from(sequence.times), values:Array.from(sequence.values), result};
  } finally { VisualFeatureWorkerClient.create = createWorker; media.input.dispose(); }
};`);
const videoStat = await stat(source);
const server = await createServer({
  configFile: false, root, publicDir: false, cacheDir: join(output, "vite-cache"),
  resolve: { alias: {
    "mediabunny": join(repo, "prod/node_modules/mediabunny/dist/bundles/mediabunny.mjs"),
    "fft.js": join(repo, "prod/node_modules/fft.js/lib/fft.js"),
  } },
  server: { host: "127.0.0.1", port: 0, fs: { allow: [repo, root, resolve(moduleRoot, "..")] } },
  plugins: [{ name: "benchmark-fixtures", configureServer(server) {
    server.middlewares.use(async (req, res, next) => {
      const pathname = new URL(req.url, "http://localhost").pathname;
      if (pathname === "/fixture.mp4") {
        const range = /^bytes=(\d+)-(\d*)$/.exec(req.headers.range ?? "");
        const start = range ? Number(range[1]) : 0;
        const end = range?.[2] ? Number(range[2]) : videoStat.size - 1;
        if (start > end || end >= videoStat.size) { res.statusCode = 416; res.end(); return; }
        res.setHeader("Content-Type", "video/mp4"); res.setHeader("Accept-Ranges", "bytes");
        res.setHeader("Content-Length", end - start + 1);
        if (range) { res.statusCode = 206; res.setHeader("Content-Range", `bytes ${start}-${end}/${videoStat.size}`); }
        createReadStream(source, {start, end}).pipe(res); return;
      }
      if (pathname.startsWith("/runtime/") && basename(pathname) === pathname.slice(9)) {
        try {
          const data = await readFile(join(repo, "prod/public/runtime", basename(pathname)));
          res.setHeader("Content-Type", pathname.endsWith(".wasm") ? "application/wasm" : pathname.endsWith(".json") ? "application/json" : "text/javascript");
          res.end(data); return;
        } catch { res.statusCode = 404; res.end(); return; }
      }
      next();
    });
  } }],
});
let context;
try {
  await server.listen();
  context = await chromium.launchPersistentContext(join(output, "profile"), {
    executablePath: args.get("--browser"), headless: true,
    args: ["--no-first-run", "--no-default-browser-check"],
  });
  const page = await context.newPage();
  await writeFile(join(output, "environment.json"), JSON.stringify({
    browserVersion: context.browser()?.version(), nodeVersion: process.version,
    platform: process.platform, architecture: process.arch,
    sourceBytes: videoStat.size, featureCache: "bypassed",
    visualPreprocessing: args.get("--visual") ?? "opencv-area-nearest-grid-v1",
  }, null, 2));
  page.on("pageerror", error => console.error(error.message));
  await page.goto(server.resolvedUrls.local[0]);
  await page.waitForFunction(() => typeof window.run === "function");
  if (args.get("--validate") === "true") {
    const validation = await page.evaluate(() => window.validateVisual());
    await writeFile(join(output, "visual-validation.json"), JSON.stringify(validation, null, 2));
    console.log(JSON.stringify(validation));
  }
  for (let index = 0; index < Number(args.get("--repeats") ?? 2); index++) {
    const progress = setInterval(async () => {
      try { console.log(JSON.stringify(await page.evaluate(() => window.progress ?? {}))); } catch {}
    }, 15000);
    let result;
    try {
      result = await page.evaluate(options => window.run(options), {
        decodeStrategy: args.get("--strategy") ?? "sequential", detailedProfiling: true,
        visualPreprocessing: args.get("--visual") ?? "opencv-area-nearest-grid-v1",
        ...(args.get("--fallback") === "true" ? {forceFallback:true,reductionKernel:"javascript"} : {}),
      });
    } finally { clearInterval(progress); }
    await writeFile(join(output, `run-${index + 1}.json`), JSON.stringify(result));
    console.log(JSON.stringify({run:index + 1, rows:result.rows, featuresMs:result.featuresMs,
      videoMs:result.performance.videoElapsedMs, rallies:result.result.intervals.length}));
  }
} finally {
  await context?.close();
  await server.close();
}
