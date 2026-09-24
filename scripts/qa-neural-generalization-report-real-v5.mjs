#!/usr/bin/env node
import { privateValue } from "../lib/server/private-ledger.mjs";
// V5 QA derives from frozen V4 dac739f15234f48012e06c3a3e117e59439982cb8c0f6b47227bd571c25dd79b.
// Every V4 check is preserved; only V5 identities/version and visible deployment-note checks are added.
import {createRequire} from 'node:module';
import {readFile,writeFile,mkdir,stat} from 'node:fs/promises';
import {createReadStream} from 'node:fs';
import {join,resolve} from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {createGunzip} from 'node:zlib';
import assert from 'node:assert/strict';

export const DEPLOYMENT_NOTICE="These are offline accuracy experiments, not phone qualification. DINO FP16 used CUDA; mixed INT8 used native CPU and failed the tested desktop-browser numerical parity check, so its accuracy is not established in a browser. No physical-phone latency, peak-memory or thermal measurements were made; prior graph checks do not qualify every new checkpoint or the complete video pipeline.";
export const variants=['original-medium','expanded-medium','expanded-large','expanded-wide-validation','export-rally-training','export-rally-selection'];
const numeric=x=>typeof x==='number'&&Number.isFinite(x);
const fixed=r=>r.variant==='fixed-production';
const scenario=r=>String(r.scenario??r.variant??'baseline');
const draw=r=>String(r.draw??r.seed??'aggregate');
const dtype=r=>String(r.precision??'fp32');
const name=r=>fixed(r)?r.model+' / fixed shipped comparator':[r.model,r.variant,draw(r),dtype(r)].filter(x=>x!==null&&x!==undefined).join(' / ');
const complete=r=>['available','complete','feasible','passed'].includes(r.status)&&['precisionValue','recallValue','f1Value'].every(k=>numeric(r[k]));
const scope=r=>JSON.stringify([...(r.recordingIds||[])].sort());
const group=r=>JSON.stringify([r.model,r.variant,draw(r),dtype(r),r.aggregationScope??'unspecified']);
const pct=x=>numeric(x)?(100*x).toFixed(2)+'%':'—';
const sec=x=>numeric(x)?x.toFixed(1):'—';
export const filters=r=>({model:String(r.model),scenario:scenario(r),panel:String(r.panelId),policy:String(r.labelPolicy??'unspecified'),production:String(r.productionFilter??'all'),padding:String(r.paddingSeconds),draw:draw(r),dtype:dtype(r)});
const expanded=(r,scopes)=>r.recordingIds?r:{...r,recordingIds:scopes[r.scopeId]};

export function expectedRows(data,state){
 const selected=new Set(state.variants), match=(r,keys)=>keys.every(k=>state.filters[k]==='*'||filters(r)[k]===state.filters[k]);
 const keys=Object.keys(state.filters);
 const rows=data.series.filter(r=>state.mode==='variants'
  ?!fixed(r)&&selected.has(scenario(r))&&match(r,keys.filter(k=>k!=='scenario'))
  :match(r,keys.filter(k=>!(fixed(r)&&['model','draw','dtype'].includes(k))))).map(r=>expanded(r,data.scopes));
 if(state.production&&(state.mode==='variants'||state.filters.scenario!=='fixed-production')){
  const scopes=new Set(rows.map(scope));
  rows.push(...data.series.filter(r=>fixed(r)&&match(r,keys.filter(k=>!['scenario','draw','dtype','model'].includes(k))))
   .map(r=>expanded(r,data.scopes)).filter(r=>scopes.has(scope(r))));
 }
 return rows;
}

export function expectedTable(rows,floor='*'){
 return rows.filter(r=>floor==='*'||String(r.floorPercent)===floor)
  .sort((a,b)=>name(a).localeCompare(name(b))||a.floorPercent-b.floorPercent).slice(0,600).map(r=>[
   name(r),r.floorPercent+'%',r.status,...['precisionValue','recallValue','f1Value'].map(k=>complete(r)?pct(r[k]):'—'),
   ...['exportSeconds','humanExportSeconds','correctlyRemovedSeconds','incorrectExportSeconds','wantedExportOmittedSeconds','missedCoreSeconds'].map(k=>complete(r)?sec(r[k]):'—'),
   complete(r)&&numeric(r.completeRallyLosses)?String(r.completeRallyLosses):'—',complete(r)?pct(r.eventF1):'—',String(r.recordingIds.length)]);
}

export function expectedSegments(groups,key){
 const result=[],x=r=>43+(r.floorPercent-90)/10*402,y=r=>15+(1-r[key])*240;
 for(const rows of groups){let previous=null;for(const r of rows){if(!complete(r)){previous=null;continue}
  if(previous&&r.floorPercent-previous.floorPercent<=1.0000001&&scope(r)===scope(previous))result.push({x1:x(previous),y1:y(previous),x2:x(r),y2:y(r)});
  previous=r;
 }}return result;
}

export function chooseViews(data){
 // Membership-driven coverage only: never inspect precision, recall, F1 or status.
 const seen=new Map(['model','scenario','panel','policy','production','padding','draw','dtype'].map(k=>[k,new Set()])), views=[];
 for(const r of data.series){if(fixed(r))continue;const f=filters(r);
  if(Object.entries(f).some(([k,v])=>!seen.get(k).has(v))){views.push(f);for(const [k,v]of Object.entries(f))seen.get(k).add(v)}
 }
 return views;
}

const digest=b=>createHash('sha256').update(b).digest('hex');
const windows=p=>p.startsWith(privateValue("private-reference-0060"))?privateValue("private-reference-0203")+p.slice(privateValue("private-reference-0060").length).replaceAll('/','\\'):p.startsWith(privateValue("private-reference-0205"))?'C:\\'+p.slice(privateValue("private-reference-0205").length).replaceAll('/','\\'):p;
async function hashStream(stream){const h=createHash('sha256');let bytes=0;for await(const chunk of stream){h.update(chunk);bytes+=chunk.length}return {sha256:h.digest('hex'),bytes}}
async function identity(path){return {path:resolve(path),...await hashStream(createReadStream(path))}}
const nas=p=>new RegExp(privateValue("private-reference-0107"), "i").test(resolve(p));

export async function run(args){
 for(const key of ['--harness','--html','--source-json','--render-receipt','--output-root'])assert.ok(args.has(key),key+' required');
 const previousQA=await identity(fileURLToPath(new URL('./qa-neural-generalization-report-real-v4.mjs',import.meta.url)));
 assert.equal(previousQA.sha256,'dac739f15234f48012e06c3a3e117e59439982cb8c0f6b47227bd571c25dd79b');
 const out=resolve(args.get('--output-root'));assert.ok(nas(out),'NAS output required');
 const selfTest=args.get('--synthetic-self-test')==='true';
 await mkdir(out,{recursive:true});
 assert.equal(await stat(join(out,'qa.json')).then(()=>true,()=>false),false,'Refuse to overwrite QA receipt');
 process.env.TEMP=join(out,'temp');process.env.TMP=process.env.TEMP;await mkdir(process.env.TEMP,{recursive:true});
 const source=await identity(args.get('--source-json')),html=await identity(args.get('--html')),renderIdentity=await identity(args.get('--render-receipt'));
 const data=JSON.parse(await readFile(args.get('--source-json'),'utf8')),render=JSON.parse(await readFile(args.get('--render-receipt'),'utf8'));
 assert.equal(!!data.metadata.syntheticFixture,selfTest,'Synthetic fixtures require explicit self-test mode; real reports must be audited');
 assert.equal(render.kind,'deployment-disclosure-report-view-v5');assert.equal(render.runtimeQualificationNotice,DEPLOYMENT_NOTICE);assert.equal(render.sourceSha256,source.sha256);
 if(!selfTest||render.output)assert.equal(render.output.sha256,html.sha256);
 const renderer=await identity(fileURLToPath(new URL('./render-neural-generalization-report-v5.py',import.meta.url)));
 assert.equal(renderer.sha256,'1365d8a7b612fc9b624475814e4b0e150e4ad275a1e1565485bdf1d6e4693b38');assert.equal(render.rendererSha256,renderer.sha256);
 let audit=null;
 if(!selfTest){assert.equal(data.metadata.auditPassed,true);audit=await identity(windows(data.metadata.audit.path));assert.equal(audit.sha256,data.metadata.audit.sha256);
  const gate=JSON.parse(await readFile(audit.path,'utf8'));assert.equal(gate.passed,true);assert.equal(gate.reportContentSha256,render.reportContentSha256)}
 const require=createRequire(join(resolve(args.get('--harness')),'package.json')),{chromium}=require('playwright-core');
 const browser=await chromium.launch({executablePath:'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',headless:true,args:['--disable-gpu','--disk-cache-dir='+join(out,'cache')]});
 const context=await browser.newContext({viewport:{width:1440,height:1100},acceptDownloads:true}),page=await context.newPage();
 page.setDefaultTimeout(120000);
 const errors=[],requests=[],checks={views:[],memoryScope:'Chrome renderer JS heap snapshots and Node RSS; not whole browser process-tree peak RAM'};
 page.on('pageerror',e=>errors.push(String(e)));page.on('console',m=>{if(m.type()==='error')errors.push(m.text())});
 await context.route(/^https?:/,route=>{requests.push(route.request().url());return route.abort()});
 const cdp=await context.newCDPSession(page);await cdp.send('Performance.enable');
 const memory=async()=>{const m=(await cdp.send('Performance.getMetrics')).metrics;return {jsHeapUsedBytes:m.find(x=>x.name==='JSHeapUsedSize')?.value,jsHeapTotalBytes:m.find(x=>x.name==='JSHeapTotalSize')?.value,nodeRssBytes:process.memoryUsage().rss}};
 const state=()=>page.evaluate(()=>({mode:document.querySelector('#comparisonMode').value,variants:window.__reportQA.selectedVariants(),production:document.querySelector('#showProduction').checked,filters:Object.fromEntries([...document.querySelectorAll('#filters select')].map(e=>[e.id.slice(7),e.value]))}));
 const verify=async(label)=>{
  assert.equal(await page.locator('#runtimeQualification').textContent(),DEPLOYMENT_NOTICE,label+' deployment note');
  assert.equal(await page.locator('#runtimeQualification').isVisible(),true,label+' deployment note visible');
  const current=await state(),expected=expectedRows(data,current),actual=await page.evaluate(()=>window.__reportQA.activeRows());assert.deepEqual(actual,expected,label+' active rows');
  const floor=await page.locator('#tableFloor').inputValue(),table=expectedTable(expected,floor);
  const cells=await page.locator('#results tbody tr').evaluateAll(rows=>rows.map(r=>[...r.querySelectorAll('td')].map(c=>c.textContent)));
  assert.deepEqual(cells,table.length?table:[['No matching operating points']],label+' displayed table');
  const grouped=new Map();for(const r of expected){const key=group(r);if(!grouped.has(key))grouped.set(key,[]);grouped.get(key).push(r)}
  const ordered=[...grouped.values()].flatMap(rows=>rows.sort((a,b)=>a.floorPercent-b.floorPercent));
  for(const [id,key,title]of [['precisionChart','precisionValue','Precision'],['recallChart','recallValue','Recall'],['f1Chart','f1Value',await page.locator('#f1ChartTitle').textContent()]]){
   const circles=await page.locator('#'+id+' circle').evaluateAll(items=>items.map(e=>({x:Number(e.getAttribute('cx')),y:Number(e.getAttribute('cy')),label:e.getAttribute('aria-label')})));
   assert.deepEqual(circles,ordered.filter(complete).map(r=>({x:43+(r.floorPercent-90)/10*402,y:15+(1-r[key])*240,label:`${name(r)}, floor ${r.floorPercent}%, ${title} ${pct(r[key])}`})),label+' '+id);
   const segments=await page.locator('#'+id+' line:not(.gridline)').evaluateAll(items=>items.map(e=>Object.fromEntries(['x1','y1','x2','y2'].map(k=>[k,Number(e.getAttribute(k))]))));
   assert.deepEqual(segments,expectedSegments(grouped.values(),key),label+' no lines across gaps or changed scopes');
   assert.equal(await page.locator('#'+id+' text').filter({hasText:'×'}).count(),ordered.filter(r=>!complete(r)).length,label+' infeasible gaps');
  }
  const dimensions=await page.evaluate(()=>({scroll:document.documentElement.scrollWidth,viewport:innerWidth}));assert.ok(dimensions.scroll<=dimensions.viewport+1,label+' page overflow');
  const matching=data.tasks.map((t,i)=>({t,i})).filter(({t})=>{
   if(!(current.mode==='variants'?current.variants.includes(scenario(t)):scenario(t)===current.filters.scenario))return false;
   if(current.mode==='models'&&fixed(t))return true;
   return (current.filters.model==='*'||String(t.model)===current.filters.model)&&(current.filters.dtype==='*'||dtype(t)===current.filters.dtype)&&
    (['*','all registered draws'].includes(current.filters.draw)||draw(t)===current.filters.draw);
  });
  assert.deepEqual(await page.evaluate(()=>window.__reportQA.matchingTaskIndices()),matching.map(({i})=>i),label+' matching task population');
  const taskValue=await page.locator('#task').inputValue(),selected=taskValue==='*'?matching:matching.filter(({i})=>String(i)===taskValue);
  assert.deepEqual(await page.evaluate(()=>window.__reportQA.selectedTaskIndices()),selected.map(({i})=>i),label+' selected task population');
  assert.equal(await page.locator('#task').isDisabled(),!matching.length,label+' empty task state');
  const inventory=Array.isArray(data.inventory)?data.inventory:data.inventory.records;
  const roleRows=await page.locator('#inventory tbody tr').evaluateAll(rows=>rows.map(r=>({id:r.querySelector('td')?.textContent,tags:[...r.querySelectorAll('td:nth-child(4) .tag')].map(t=>t.textContent)})));
  for(const item of roleRows){if(!inventory.some(r=>r.id===item.id))continue;
   const expectedRoles=selected.flatMap(({t,i})=>Object.entries({fit:t.memberships?.fit??[],calibrate:t.memberships?.calibrate??[],evaluate:t.memberships?.evaluate??[],infer:t.memberships?.infer??[]}).filter(([,ids])=>ids.includes(item.id)).map(([role])=>role+(selected.length>1?' \u00b7 '+[t.id??`task${i+1}`,t.model,t.variant,t.draw??t.seed,t.precision].filter(x=>x!==undefined).join(' / '):'')));
   assert.deepEqual(item.tags,expectedRoles,label+' inventory roles '+item.id);
  }
  const result={label,state:current,rows:expected.length,complete:expected.filter(complete).length,tableRows:table.length,matchingTasks:matching.length,selectedTasks:selected.length,dimensions,memory:await memory()};checks.views.push(result);return result;
 };
 try{
  const started=performance.now();await page.goto(pathToFileURL(resolve(args.get('--html'))).href);await page.waitForFunction(()=>!!window.__reportQA);
  checks.startupMilliseconds=performance.now()-started;checks.initialMemory=await memory();
  assert.equal(await page.evaluate(()=>window.__reportQA.presentationVersion),5);
  assert.equal(await page.locator('#fixtureWarning').isVisible(),selfTest);
  const transport=await page.evaluate(()=>window.__reportQA.transport);assert.equal(transport.sourceSha256,source.sha256);assert.equal(transport.reportContentSha256,render.reportContentSha256);assert.equal(transport.originalBytes,source.bytes);
  assert.equal(await page.evaluate(()=>window.__reportQA.sourceRows),data.series.length);checks.transport=transport;
  const initial=await verify('initial-desktop');
  if(initial.matchingTasks){assert.equal(initial.selectedTasks,1,'Default must be one concrete matching task');const previous=await page.locator('#task').inputValue();await page.selectOption('#task','*');await verify('explicit-all-matching-tasks');await page.selectOption('#task',previous)}
  const views=chooseViews(data);checks.requestedViewCount=views.length;
  for(const [i,f]of views.entries()){
   const start=performance.now();await page.selectOption('#comparisonMode','models');
   // Set all controls together, then trigger the same user change handler once.
   await page.evaluate(f=>{for(const [k,v]of Object.entries(f)){const e=document.querySelector('#filter-'+k);if(![...e.options].some(o=>o.value===v))throw Error('Missing actual filter '+k+'='+v);e.value=v}document.querySelector('#filter-model').dispatchEvent(new Event('change'))},f);
   const filteredAt=performance.now(),result=await verify('saved-view-'+i);
   result.filterMilliseconds=filteredAt-start;result.verificationMilliseconds=performance.now()-filteredAt;
  }
  if(data.tasks.some(fixed)&&data.series.some(fixed)){await page.selectOption('#filter-scenario','fixed-production');await verify('fixed-production-membership-with-neural-filters')}
  const seed=data.series.find(r=>!fixed(r)&&variants.includes(scenario(r))&&r.panelId==='common-unseen')??data.series.find(r=>variants.includes(scenario(r)));
  if(seed){const f=filters(seed);await page.selectOption('#comparisonMode','variants');await page.evaluate(f=>{for(const [k,v]of Object.entries(f))if(k!=='scenario')document.querySelector('#filter-'+k).value=v;document.querySelector('#filter-model').dispatchEvent(new Event('change'))},f);
   await verify('matched-variants-desktop');await page.screenshot({path:join(out,'variants-desktop.png'),fullPage:true});
   await page.selectOption('#tableFloor','100');await verify('saved-100-percent-floor');await page.selectOption('#tableFloor','*');
   await page.uncheck('#showProduction');await verify('production-overlay-off');await page.check('#showProduction');
  }else checks.variantView='No variant rows in the saved report';
  await page.screenshot({path:join(out,'desktop.png'),fullPage:true});await page.setViewportSize({width:390,height:844});
  await verify('390px');await page.screenshot({path:join(out,'mobile-390.png'),fullPage:true});
  // Exhaustive transport parity in bounded batches; never materialize all rows in Chrome.
  const columnSet=new Set();for(const row of data.series)for(const key of Object.keys(row))columnSet.add(key);
  const columns=[...columnSet];let cells=0;
  for(let offset=0;offset<data.series.length;offset+=512){const rows=data.series.slice(offset,offset+512);
   const failed=await page.evaluate(({offset,rows,columns})=>{for(let i=0;i<rows.length;i++)for(const key of columns){const value=rows[i][key],actual=window.__reportQA.readCell(offset+i,key);const same=Object.is(actual,value)||(value!==null&&typeof value==='object'&&actual!==null&&typeof actual==='object'&&JSON.stringify(actual)===JSON.stringify(value));if(!same)return {row:offset+i,key}}return null},{offset,rows,columns});
   assert.equal(failed,null,'Packed cell differs: '+JSON.stringify(failed));cells+=rows.length*columns.length;
  }
  checks.packedParity={rows:data.series.length,columns:columns.length,cells,batchRows:512};
  const pending=page.waitForEvent('download');await page.click('#download');const download=await pending,saved=join(out,'exact-source.json.gz');await download.saveAs(saved);
  const restored=await hashStream(createReadStream(saved).pipe(createGunzip()));assert.equal(restored.sha256,source.sha256);assert.equal(restored.bytes,source.bytes);checks.downloadRestoresExactSource=restored;
  assert.deepEqual(errors,[]);assert.deepEqual(requests,[]);
  checks.finalMemory=await memory();checks.materializedAcrossInteractions=await page.evaluate(()=>window.__reportQA.materializedRows());
  // Close identity races, including the real audit and its source report/receipt.
  assert.deepEqual(await identity(source.path),source);assert.deepEqual(await identity(html.path),html);assert.deepEqual(await identity(renderIdentity.path),renderIdentity);assert.deepEqual(await identity(renderer.path),renderer);if(audit)assert.deepEqual(await identity(audit.path),audit);
  const result={kind:'source-derived-generalization-report-ui-qa-v5',previousQA,runtimeQualificationNotice:DEPLOYMENT_NOTICE,passed:true,syntheticSelfTest:selfTest,actualAuditedReport:!selfTest,source,html,renderReceipt:renderIdentity,renderer,audit,script:await identity(fileURLToPath(import.meta.url)),browser:await browser.version(),checks,errors,externalRequests:requests,profileAndTempRoot:process.env.TEMP};
  await writeFile(join(out,'qa.json'),JSON.stringify(result,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify({passed:true,syntheticSelfTest:selfTest,views:checks.views.length,rows:data.series.length,startupMilliseconds:checks.startupMilliseconds}));
 }finally{await context.close();await browser.close()}
}

if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 const args=new Map();for(let i=2;i<process.argv.length;i+=2)args.set(process.argv[i],process.argv[i+1]);await run(args);
}
