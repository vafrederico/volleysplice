import { createServer, request } from 'node:http';
import { readFile, writeFile, rename } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { resolve, basename } from 'node:path';
import { privateValue } from '../lib/server/private-ledger.mjs';
import { SERVING_SIDE_MODEL_ID, SERVING_SIDE_MODEL_FINGERPRINT } from '../prod/src/lib/on-device/serving-side-cache.ts';
import { attachLabServing } from '../lib/production-editor-lab-serving.ts';
const args = new Map(process.argv.slice(2).reduce((rows,v,i,a)=>i%2?rows:[...rows,[v,a[i+1]]],[]));
if (!args.get('--work') || !args.get('--browser')) throw Error('External --work and --browser required');
const work=resolve(args.get('--work')), base=new URL(args.get('--url')??'http://localhost:3000');
// Keep disposable browser profiles beside the externally configured artifacts.
process.env.TMPDIR=work; process.env.TMP=work; process.env.TEMP=work;
const require=createRequire(resolve(args.get('--harness')??privateValue('private-reference-0109'),'package.json'));
const {chromium}=require('playwright-core');
const json=async url=>{const r=await fetch(url);if(!r.ok)throw Error(`HTTP ${r.status}`);return r.json();};
const catalog=await json(new URL('/api/editor-lab/tasks',base));
let jobs=JSON.parse(await readFile(resolve(work,'jobs.json'),'utf8'));
// Process the current editor example first, using its external ledger mapping.
const preferred=new URL(privateValue('private-reference-0115'),base).searchParams.get('task');
jobs.sort((a,b)=>Number(b===preferred)-Number(a===preferred));
if(args.get('--tail'))jobs=jobs.slice(-Number(args.get('--tail'))).reverse();
if(args.get('--retry-failed')){
  if(args.get('--tail'))throw Error('Choose only one job selector');
  const previous=JSON.parse(await readFile(resolve(work,'progress.json'),'utf8'));
  const failed=new Set(previous.filter(row=>row.error).map(row=>row.recordingId));
  jobs=jobs.filter(id=>failed.has(id));
}
const progressFile=resolve(work,args.get('--retry-failed')?'progress-retry.json':args.get('--tail')?`progress-tail-${Number(args.get('--tail'))}.json`:'progress.json');
const server=createServer(async(req,res)=>{
  if(req.url.startsWith('/lab/')){
    const upstream=request(new URL(req.url.slice(4),base),{method:req.method,headers:{...req.headers,host:base.host}},r=>{res.writeHead(r.statusCode,r.headers);r.pipe(res);});
    upstream.on('error',()=>{res.writeHead(502);res.end();});req.pipe(upstream);return;
  }
  try{
    if(req.url==='/'){res.setHeader('Content-Type','text/html');res.end('<!doctype html><script type="module" src="/runner.js"></script>');return;}
    const name=basename(req.url.split('?')[0]);
    if(!req.url.startsWith('/runtime/')&&name!=='runner.js'){res.writeHead(404);res.end();return;}
    res.setHeader('Content-Type',name.endsWith('.json')?'application/json':'application/javascript');
    res.end(await readFile(req.url.startsWith('/runtime/')?resolve('prod/public/runtime',name):resolve(work,name)));
  }catch{res.writeHead(500);res.end();}
});
await new Promise(r=>server.listen(0,'127.0.0.1',r));
const browser=await chromium.launch({executablePath:args.get('--browser'),headless:true});
let next=0;const results=[];
try{
  await Promise.all(Array.from({length:Number(args.get('--workers')??2)},async()=>{
    while(next<jobs.length){
      const id=jobs[next++];
      try{
        const input=JSON.parse(await readFile(resolve(work,`${id}.input.json`),'utf8'));
        const row=catalog.tasks.find(t=>t.id===id||t.recordingId===id);
        if(!row)throw Error('Recording absent from lab catalog');
        const task=await json(new URL(`/api/editor-lab/tasks/${encodeURIComponent(row.id)}?serving=omit`,base));
        const configurations=task.configurations.filter(c=>!c.humanReference);
        input.populations=Object.fromEntries(configurations.map(c=>[c.id,(c.draftEvents??c.events).map(({id,start,end,agreement})=>({id,start,end,...(agreement?{agreement}:{})}))]));
        const target=resolve(work,'..',`${id}.json`);
        let previous;try{previous=JSON.parse(await readFile(target,'utf8'));}catch{}
        let reusable=false;
        if(previous?.schemaVersion===1&&previous.labelsUsed===false&&previous.recordingId===id&&previous.duration===input.duration
          &&previous.modelId===SERVING_SIDE_MODEL_ID&&previous.modelFingerprint===SERVING_SIDE_MODEL_FINGERPRINT
          &&previous.contentSha256===input.contentSha256&&previous.evidenceSha256===input.evidenceSha256
          &&JSON.stringify(previous.roi)===JSON.stringify(input.roi)
          &&JSON.stringify(Object.fromEntries(Object.entries(previous.configurations??{}).map(([k,v])=>[k,v?.population])))===JSON.stringify(input.populations)){
          try{
            for(const config of configurations){
              const cached=previous.configurations[config.id];
              if(cached.output?.modelId!==SERVING_SIDE_MODEL_ID||cached.output.modelFingerprint!==SERVING_SIDE_MODEL_FINGERPRINT)throw Error('Stale serving model');
              attachLabServing(config,cached,input.duration);
            }
            reusable=true;
          }catch{}
        }
        if(reusable){
          results.push({recordingId:id,configurations:Object.keys(input.populations).length,cached:true});continue;
        }
        const page=await browser.newPage();page.setDefaultTimeout(3600000);
        page.on('console',m=>console.log(`${id}: ${m.text()}`));
        try{
          await page.goto(`http://127.0.0.1:${server.address().port}/`);
          await page.waitForFunction(()=>typeof window.runLabServing==='function');
          input.videoUrl=`/lab/api/labeling/tasks/${encodeURIComponent(row.id)}/video`;
          const receipt=await page.evaluate(input=>window.runLabServing(input),input);
          for(const config of configurations)attachLabServing(config,receipt.configurations[config.id],input.duration);
          await writeFile(`${target}.tmp`,JSON.stringify(receipt));await rename(`${target}.tmp`,target);
          results.push({recordingId:id,configurations:Object.keys(input.populations).length,anchors:receipt.uniqueAnchors});
          console.log(JSON.stringify({completed:results.length,total:jobs.length,recordingId:id}));
        }finally{await page.close();}
      }catch(error){results.push({recordingId:id,error:String(error)});console.error(`${id}: ${error}`);}
      await writeFile(progressFile,JSON.stringify(results,null,2));
    }
  }));
  await writeFile(progressFile,JSON.stringify(results,null,2));
  console.log(JSON.stringify({recordings:results.length,failed:results.filter(r=>r.error).length}));
  if(results.some(r=>r.error))process.exitCode=1;
}finally{await browser.close();server.close();}
