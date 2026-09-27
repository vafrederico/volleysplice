#!/usr/bin/env node
import { privateValue } from "../lib/server/private-ledger.mjs";
// Actual image encoder qualification; no production app or GPU.
import {createServer} from 'node:http';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
import {join,basename,resolve} from 'node:path';
import os from 'node:os';

const root=privateValue("private-reference-0108");
const output=join(root,'browser-v1');
await mkdir(output,{recursive:false});
const harness=privateValue("private-reference-0109");
const require=createRequire(join(harness,'package.json'));
const {chromium}=require('playwright-core');
const runtime=join(harness,'node_modules','onnxruntime-web','dist');
const version=JSON.parse(await readFile(join(runtime,'..','package.json'),'utf8')).version;
if(version!=='1.22.0')throw new Error('Expected ORT Web 1.22.0');
const hash=b=>createHash('sha256').update(b).digest('hex');
const windowsPath=p=>p.replace(privateValue("private-reference-0060"), privateValue("private-reference-0204"));
const manifestPath=join(root,'browser-fixtures-v1','manifest.json');
const manifestBytes=await readFile(manifestPath);
const manifest=JSON.parse(manifestBytes);
const fixtures=new Map();
for(const model of manifest.models) for(const entry of [model.graph,...model.cases.flatMap(c=>[c.input,c.expected])]){
  const bytes=await readFile(windowsPath(entry.path));
  if(hash(bytes)!==entry.sha256 || bytes.length!==entry.sizeBytes)throw new Error('Fixture changed');
  fixtures.set(basename(entry.path),bytes);
}
const server=createServer(async(req,res)=>{
  const path=new URL(req.url,'http://localhost').pathname;
  res.setHeader('Cross-Origin-Opener-Policy','same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy','require-corp');
  if(path==='/'){res.setHeader('Content-Type','text/html');res.end('<script src="/runtime/ort.wasm.min.js"></script>');return;}
  const key=path.split('/').at(-1);
  if(path.startsWith('/runtime/') && key===basename(key)){
    try{res.setHeader('Content-Type',key.endsWith('.wasm')?'application/wasm':'text/javascript');res.end(await readFile(join(runtime,key)));}catch{res.statusCode=404;res.end();}return;
  }
  if(fixtures.has(key)){res.end(fixtures.get(key));return;}
  res.statusCode=404;res.end();
});
await new Promise(r=>server.listen(0,'127.0.0.1',r));
let browser,result;
try{
  browser=await chromium.launch({headless:true,executablePath:'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',args:['--disable-gpu','--no-first-run','--no-default-browser-check']});
  const page=await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  result=await page.evaluate(async manifest=>{
    const ort=window.ort;ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;ort.env.wasm.wasmPaths=location.origin+'/runtime/';
    const fetchBytes=async e=>{
      const bytes=await(await fetch('/fixture/'+e.path.split('/').at(-1))).arrayBuffer();
      const h=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
      if(h!==e.sha256)throw new Error('Browser fixture changed');return bytes;
    };
    const models=[];
    for(const model of manifest.models){
      const creation=performance.now();
      const session=await ort.InferenceSession.create(await fetchBytes(model.graph),{executionProviders:['wasm']});
      const creationMs=performance.now()-creation;
      const cases=[];let first;
      for(const c of model.cases){
        const x=new Float32Array(await fetchBytes(c.input));
        const expected=new Float32Array(await fetchBytes(c.expected));
        const inputs={image:new ort.Tensor('float32',x,c.shape)};first??=inputs;
        const actual=(await session.run(inputs)).tokens;
        let max=0,mae=0,passed=JSON.stringify(actual.dims)===JSON.stringify(c.outputShape);
        for(let i=0;i<expected.length;i++){const d=Math.abs(actual.data[i]-expected[i]);max=Math.max(max,d);mae+=d;passed&&=Number.isFinite(actual.data[i])&&d<=1e-4+1e-4*Math.abs(expected[i]);}
        cases.push({index:c.index,passed,maxAbsoluteError:max,meanAbsoluteError:mae/expected.length});
      }
      for(let i=0;i<2;i++)await session.run(first);
      const samples=[];for(let i=0;i<5;i++){const t=performance.now();await session.run(first);samples.push(performance.now()-t);}
      await session.release();
      models.push({name:model.name,creationMs,cases,latencySamplesMs:samples,medianMs:[...samples].sort((a,b)=>a-b)[2]});
    }
    return {passed:models.every(m=>m.cases.every(c=>c.passed)),models,userAgent:navigator.userAgent};
  },manifest);
  result.browserVersion=browser.version();
}catch(error){result={passed:false,error:String(error)};}
finally{if(browser)await browser.close();await new Promise(r=>server.close(r));}
const report={...result,runtimeVersion:version,wasmThreads:1,cpu:os.cpus()[0].model,
  fixtureManifest:{path:manifestPath,sha256:hash(manifestBytes)},scriptSha256:hash(await readFile(new URL(import.meta.url))),
  gpuUsed:false,physicalPhoneMeasured:false,productionAppStarted:false,
  scope:'Actual DINOv2-S/14 image encoder, prepared 336px image inputs, desktop Chrome WASM CPU.',
  limitations:['Desktop host shared with other experiments.','Excludes video decoding, image preprocessing, AV extraction, TCN and UI.','No phone thermal, battery, memory or hardware delegate measurement.']};
await writeFile(join(output,'report.json'),JSON.stringify(report,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify(report));
if(!report.passed)process.exitCode=1;
