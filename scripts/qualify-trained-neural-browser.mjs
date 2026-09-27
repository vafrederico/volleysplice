#!/usr/bin/env node
import { privateValue } from "../lib/server/private-ledger.mjs";
// Isolated trained-checkpoint WASM check. Never starts the production app.
import { createServer } from "node:http";
import { createHash } from "node:crypto";
import { readFile, writeFile, mkdir, mkdtemp, copyFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { resolve, join, relative, sep, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import os from "node:os";

const cli = new Map();
for (let index = 2; index < process.argv.length; index += 2) cli.set(process.argv[index], process.argv[index + 1]);
if (!cli.has("--harness")) throw new Error("--harness must name the isolated pinned dependency directory");
const harness = resolve(cli.get("--harness"));
const source = cli.get("--source") ?? privateValue("private-reference-0113");
const output = cli.get("--output") ?? privateValue("private-reference-0114");
const require = createRequire(join(harness, "package.json"));
const { chromium } = require("playwright-core");
const runtimeRoot = join(harness, "node_modules", "onnxruntime-web", "dist");
const runtimePackage = JSON.parse(await readFile(join(runtimeRoot, "..", "package.json"), "utf8"));
if (runtimePackage.version !== "1.22.0") throw new Error("Expected isolated onnxruntime-web 1.22.0");
const working = await mkdtemp(join(os.tmpdir(), "volleycut-trained-neural-browser-"));
const fixtures = join(working, "fixtures");
await mkdir(fixtures);
await copyFile(join(harness, "package-lock.json"), join(working, "package-lock.json"));
const repository = resolve(fileURLToPath(new URL("..", import.meta.url)));
const wslPath = value => value.replaceAll("\\", "/").replace(/^([A-Za-z]):/, (_, drive) => `/mnt/${drive.toLowerCase()}`);
const python = privateValue("private-reference-0110");
const pythonRun = (code, args) => {
  const result = spawnSync("wsl.exe", ["-d", "Ubuntu", "--", python, "-c", code, ...args], {
    windowsHide: true, encoding: "utf8", timeout: 60000,
  });
  if (result.status !== 0) throw new Error(`Python qualification failed: ${result.stderr || result.stdout}`);
  return result.stdout;
};
pythonRun(`
import hashlib,importlib.util,json,pathlib,shutil,sys
import numpy as np
import torch
repo,source,target=map(pathlib.Path,sys.argv[1:])
spec=importlib.util.spec_from_file_location('exporter',repo/'scripts/export-neural-checkpoint.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
torch.set_num_threads(1)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify(item):
 path=pathlib.Path(item['path'])
 if sha(path)!=item['sha256']:raise ValueError('Changed prerequisite: '+str(path))
 return path
report=json.loads((source/'trained-runtime-qualification.json').read_text())
if report['status']!='pass' or (report['kind'],report['seed'],report['outerIndex'])!=('tcn',3407,0):raise ValueError('Predetermined trained TCN prerequisite changed')
metadata=json.loads(verify(report['metadata']).read_text())
graph=verify(report['graph']);verify(report['traces']);verify(metadata['manifest']);verify(metadata['completed'])
checkpoint=pathlib.Path(metadata['checkpoint']['path']).parent
prereg=json.loads(verify(metadata['preregistration']).read_text())
model,mean,scale,_=module.load_checkpoint(checkpoint,metadata['epoch'],'tcn',prereg['sha256'])
wrapper=module.ScaledProbabilityModel(model,mean,scale).eval()
examples={row.id:row for row in module.load_examples(pathlib.Path(metadata['manifest']['path']),False)}
shutil.copy2(graph,target/'trained-tcn.onnx')
manifest={'model':{'file':'trained-tcn.onnx','sha256':sha(graph)},'cpuReport':{'path':str(source/'trained-runtime-qualification.json'),'sha256':sha(source/'trained-runtime-qualification.json')},'records':[],'shortCases':[],'decoder':metadata['decoder']}
with np.load(verify(report['traces']),allow_pickle=False) as traces:
 for index,record in enumerate(report['records']):
  row=examples[record['id']];prefix='record-'+str(index)
  row.values.astype(np.float32).tofile(target/(prefix+'-input.f32'))
  traces['pytorch::'+row.id].tofile(target/(prefix+'-expected.f32'))
  row.times.astype(np.float64).tofile(target/(prefix+'-times.f64'))
  row.valid.astype(np.uint8).tofile(target/(prefix+'-valid.u8'))
  manifest['records'].append({'id':row.id,'ticks':len(row.times),'prefix':prefix,'segments':module.segments(row.valid),'durationSeconds':row.duration,'decodedIntervals':record['decodedIntervals']})
 first=examples[report['records'][0]['id']]
 for ticks in (1,7,63,127,252,411):
  values=first.values[None,:ticks].astype(np.float32)
  with torch.inference_mode():expected=wrapper(torch.from_numpy(values)).numpy()
  prefix='short-'+str(ticks);values.tofile(target/(prefix+'-input.f32'));expected.tofile(target/(prefix+'-expected.f32'))
  manifest['shortCases'].append({'ticks':ticks,'prefix':prefix})
manifest['files']={p.name:{'sha256':sha(p),'sizeBytes':p.stat().st_size} for p in target.iterdir()}
(target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\\n')
`, [wslPath(repository), source, wslPath(fixtures)]);
const manifest = JSON.parse(await readFile(join(fixtures, "manifest.json"), "utf8"));
const server = createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, "http://localhost").pathname);
    response.setHeader("Cache-Control", "no-store");
    response.setHeader("Cross-Origin-Opener-Policy", "same-origin");
    response.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
    if (pathname === "/") {
      response.setHeader("Content-Type", "text/html");
      response.end('<!doctype html><title>Trained neural qualification</title><script src="/runtime/ort.wasm.min.js"></script>');
      return;
    }
    const root = pathname.startsWith("/runtime/") ? runtimeRoot : pathname.startsWith("/fixtures/") ? fixtures : null;
    if (!root) throw new Error("Unknown route");
    const requested = resolve(root, pathname.split("/").slice(2).join("/"));
    const rel = relative(root, requested);
    if (!rel || rel.startsWith(`..${sep}`) || rel === "..") throw new Error("Path outside harness");
    const types = { ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm", ".json": "application/json" };
    response.setHeader("Content-Type", types[requested.slice(requested.lastIndexOf("."))] ?? "application/octet-stream");
    response.end(await readFile(requested));
  } catch { response.statusCode = 404; response.end("Not found"); }
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
let browser, result;
const errors = [];
try {
  browser = await chromium.launch({ executablePath: cli.get("--browser") ?? "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe", headless: true,
    args: ["--disable-gpu", "--no-first-run", "--no-default-browser-check"] });
  const page = await browser.newPage();
  page.on("pageerror", error => errors.push(String(error)));
  page.on("requestfailed", request => errors.push(`${request.url()}: ${request.failure()?.errorText}`));
  await page.goto(`http://127.0.0.1:${server.address().port}/`, { waitUntil: "load", timeout: 30000 });
  result = await page.evaluate(async manifest => {
    const ort = window.ort;
    ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = `${location.origin}/runtime/`;
    const load = async name => {
      const response = await fetch(`/fixtures/${name}`);
      if (!response.ok) throw new Error(`Fixture fetch failed: ${name}`);
      return new Float32Array(await response.arrayBuffer());
    };
    const start = performance.now();
    const session = await ort.InferenceSession.create('/fixtures/trained-tcn.onnx', { executionProviders: ['wasm'] });
    const sessionCreationMilliseconds = performance.now() - start;
    const run = async (values, ticks) => (await session.run({features:new ort.Tensor('float32',values,[1,ticks,104])})).probabilities.data;
    const compare = (actual, expected, description) => {
      if (actual.length !== expected.length) throw new Error('Length mismatch: '+description);
      let maximum=0;
      for(let i=0;i<actual.length;i++) {
        const delta=Math.abs(actual[i]-expected[i]);
        if(!Number.isFinite(actual[i]) || actual[i]<0 || actual[i]>1 || delta>1e-5+1e-4*Math.abs(expected[i])) throw new Error('Probability mismatch: '+description+' at '+i);
        maximum=Math.max(maximum,delta);
      }
      return {description,maximumAbsoluteProbabilityError:maximum,pass:true};
    };
    const checks=[],records=[];
    try {
      for(const entry of manifest.shortCases) {
        checks.push(compare(await run(await load(entry.prefix+'-input.f32'),entry.ticks),await load(entry.prefix+'-expected.f32'),'short input '+entry.ticks));
      }
      for(const entry of manifest.records) {
        const values=await load(entry.prefix+'-input.f32'),expected=await load(entry.prefix+'-expected.f32');
        const whole=new Float32Array(entry.ticks*3),chunks=new Float32Array(entry.ticks*3);
        const started=performance.now();
        for(const [first,last] of entry.segments) {
          whole.set(await run(values.subarray(first*104,last*104),last-first),first*3);
          for(let left=first;left<last;left+=128) {
            const right=Math.min(last,left+128),inputLeft=Math.max(first,left-62),inputRight=Math.min(last,right+62);
            const local=await run(values.subarray(inputLeft*104,inputRight*104),inputRight-inputLeft);
            chunks.set(local.subarray((left-inputLeft)*3,(right-inputLeft)*3),left*3);
          }
        }
        const fullPlusChunkDiagnosticMilliseconds=performance.now()-started;
        checks.push(compare(whole,expected,entry.id+': full WASM vs PyTorch'),compare(chunks,expected,entry.id+': chunks WASM vs PyTorch'),compare(chunks,whole,entry.id+': chunks vs whole WASM'));
        records.push({id:entry.id,ticks:entry.ticks,fullPlusChunkDiagnosticMilliseconds,whole:Array.from(whole),chunks:Array.from(chunks)});
      }
    } finally { await session.release(); }
    return {status:'pass',checks,records,sessionCreationMilliseconds,userAgent:navigator.userAgent,runtimeVersion:ort.env.versions.web};
  },manifest);
  result.browserVersion=browser.version();
} catch(error) { result={status:'fail',failure:{name:error.name,message:error.message}}; }
finally { if(browser) await browser.close(); await new Promise(resolve=>server.close(resolve)); }
const rawPath=join(working,'browser-traces.json');
await writeFile(rawPath,JSON.stringify(result),{flag:'wx'});
const identity=async path=>{const data=await readFile(path);return{file:basename(path),sha256:createHash('sha256').update(data).digest('hex'),sizeBytes:data.length};};
const report={...result,schemaVersion:1,createdAt:new Date().toISOString(),scope:'Predetermined trained TCN3407 outer0 in isolated desktop Chrome WASM CPU',
  records:result.records?.map(({whole,chunks,...record})=>record),executionProviders:['wasm'],wasmThreads:1,gpuUsed:false,
  physicalPhoneMeasured:false,productionAppStarted:false,cpu:os.cpus()[0]?.model,browserErrors:errors,
  script:await identity(fileURLToPath(import.meta.url)),packageLock:await identity(join(working,'package-lock.json')),
  fixtureManifest:await identity(join(fixtures,'manifest.json')),cpuPrerequisite:manifest.cpuReport,
  acceptance:{atol:1e-5,rtol:1e-4,decodedIntervalsExactlyEqual:true},
  limitations:['Desktop Chrome only; no physical phone benchmark.','Prepared AV inputs: no video decode, feature extraction or percentile preprocessing is measured.','Interval decoding is replayed through the canonical Python decoder using actual browser outputs; no browser decoder port is claimed.','Diagnostic wall times are single passes including full plus chunk inference, not steady-state latency benchmarks.']};
await writeFile(join(working,'report.json'),JSON.stringify(report,null,2)+'\n',{flag:'wx'});
const verified=pythonRun(`
import importlib.util,json,pathlib,shutil,sys,types
import numpy as np
repo,working,output=map(pathlib.Path,sys.argv[1:])
sys.path.insert(0,str(repo))
from analysis.neural_development import decode
raw=json.loads((working/'browser-traces.json').read_text());report=json.loads((working/'report.json').read_text());manifest=json.loads((working/'fixtures/manifest.json').read_text())
if raw['status']=='pass':
 try:
  arrays={}
  for record,entry in zip(raw['records'],manifest['records'],strict=True):
   if record['id']!=entry['id']:raise ValueError('Record mismatch')
   prefix=working/'fixtures'/entry['prefix']
   row=types.SimpleNamespace(times=np.fromfile(str(prefix)+'-times.f64',dtype=np.float64),valid=np.fromfile(str(prefix)+'-valid.u8',dtype=np.uint8).astype(bool),duration=entry['durationSeconds'])
   for key in ('whole','chunks'):
    probabilities=np.asarray(record[key],dtype=np.float32).reshape(entry['ticks'],3)
    actual=[interval.to_dict() for interval in decode(row,probabilities,manifest['decoder'])]
    if actual!=entry['decodedIntervals']:raise ValueError('Browser decoded interval mismatch: '+entry['id']+' '+key)
    arrays[key+'::'+entry['id']]=probabilities
  np.savez_compressed(working/'browser-probabilities.npz',**arrays)
  report['decodedIntervalsExactlyEqual']=True
  report['maximumAbsoluteProbabilityError']=max(c['maximumAbsoluteProbabilityError'] for c in raw['checks'])
 except Exception as error:report.update(status='fail',failure={'type':type(error).__name__,'message':str(error)})
(working/'report.json').write_text(json.dumps(report,indent=2)+'\\n')
output.mkdir(parents=True,exist_ok=False)
for name in ('report.json','package-lock.json','browser-probabilities.npz'):
 if (working/name).exists():shutil.copy2(working/name,output/name)
shutil.copytree(working/'fixtures',output/'fixtures')
print(json.dumps({'status':report['status'],'output':str(output),'maxError':report.get('maximumAbsoluteProbabilityError'),'intervalsEqual':report.get('decodedIntervalsExactlyEqual'),'failure':report.get('failure')}))
`,[wslPath(repository),wslPath(working),output]);
console.log(verified.trim());
if(JSON.parse(verified).status!=='pass')process.exitCode=1;
