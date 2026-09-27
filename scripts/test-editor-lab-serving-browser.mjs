import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {mkdir,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {privateValue} from '../lib/server/private-ledger.mjs';
const args=new Map(process.argv.slice(2).reduce((r,v,i,a)=>i%2?r:[...r,[v,a[i+1]]],[]));
if(!args.get('--output')||!args.get('--browser'))throw Error('External --output and --browser required');
const require=createRequire(path.resolve(args.get('--harness')??privateValue('private-reference-0109'),'package.json'));
const {chromium}=require('playwright-core');
const base=args.get('--url')??'http://localhost:3000';
const get=async route=>{const r=await fetch(new URL(route,base));assert.equal(r.status,200);return r.json();};
const url=new URL(privateValue('private-reference-0115'),base);
const task=await get(`/api/editor-lab/tasks/${encodeURIComponent(url.searchParams.get('task'))}`);
const regeneration=await get(`/api/editor-lab/tasks/${encodeURIComponent(url.searchParams.get('task'))}?serving=omit`);
for(const config of task.configurations.filter(c=>c.id.startsWith('neural-'))){
  const raw=regeneration.configurations.find(c=>c.id===config.id);
  assert.equal(raw.servingPredictions.length,0);
  assert.deepEqual(raw.events.map(e=>[e.id,e.start,e.end]),config.events.map(e=>[e.id,e.start,e.end]));
}
const human=task.configurations.find(c=>c.humanReference);
assert.ok(human);
const expectedHuman=human.humanReference.scoreTracking.serveMarkers.filter(m=>!task.ignoredIntervals.some(r=>m.timestamp>=r.start&&m.timestamp<r.end));
assert.ok(expectedHuman.length>0);
const browser=await chromium.launch({executablePath:args.get('--browser'),headless:true});
const page=await browser.newPage({viewport:{width:1500,height:1000}});const errors=[];
page.on('pageerror',e=>errors.push(e.message));
try{
  for(const mode of ['neural-distilled-mobile-large-tcn-fp32','neural-distilled-mobile-large-tcn-fp32-high-recall']){
    const config=task.configurations.find(c=>c.id===mode);assert.ok(config.servingPredictions.length);
    url.searchParams.set('mode',mode);url.searchParams.set('suppression','none');
    await page.goto(url.href,{waitUntil:'networkidle',timeout:90000});
    const rail=page.getByRole('group',{name:'Human comparison rail',exact:true});await rail.waitFor();
    assert.equal(await rail.locator('[data-timeline-marker-id^="saved:"]').count(),expectedHuman.length);
    const marker=rail.locator('[data-timeline-marker-id^="saved:"]').first();
    assert.match(await marker.textContent(),/N|F|\?/);
    await marker.click({force:true});
    await page.waitForFunction(time=>Math.abs(document.querySelector('video').currentTime-time)<.25,expectedHuman[0].timestamp);
    await page.getByText('Serving-side results for this model’s rallies',{exact:true}).click();
    await page.getByText("this rally model's own starts",{exact:false}).waitFor();
    await page.getByText('Serving-side results for this model’s rallies',{exact:true}).click();
    await page.reload({waitUntil:'networkidle'});
    assert.equal(await rail.locator('[data-timeline-marker-id^="saved:"]').count(),expectedHuman.length);
    assert.ok(await rail.locator('[data-timeline-marker-id^="current:"]').count()>0);
    if(mode==='neural-distilled-mobile-large-tcn-fp32'){
      const previous=await page.evaluate(mode=>{
        const key=Object.keys(localStorage).find(k=>k.startsWith('volleycut:production-lab:trial:v1:')&&k.endsWith(':'+mode));
        const past=JSON.parse(localStorage.getItem(key));
        const manual={...past.scoreTracking.serveMarkers[0],origin:'manual',side:'review'};
        past.scoreTracking.serveMarkers=[manual];
        const present=structuredClone(past),cut=present.cuts.at(-1);
        cut.keepEnd-=.25;present.userTouchedCutIds.push(cut.id);
        const historyKey='volleycut:production-lab:edit-history:v1:'+encodeURIComponent(key);
        localStorage.setItem(key,JSON.stringify(present));
        localStorage.setItem(historyKey,JSON.stringify({version:1,past:[past],present,future:[]}));
        return {key,historyKey,manualId:manual.id,cutId:cut.id,end:cut.keepEnd,present,past};
      },mode);
      await page.reload({waitUntil:'networkidle'});
      const restored=await page.evaluate(({key,historyKey})=>({draft:JSON.parse(localStorage.getItem(key)),history:JSON.parse(localStorage.getItem(historyKey))}),previous);
      assert.equal(restored.draft.cuts.find(c=>c.id===previous.cutId).keepEnd,previous.end);
      assert.equal(restored.draft.scoreTracking.serveMarkers.find(m=>m.id===previous.manualId).side,'review');
      assert.ok(restored.draft.scoreTracking.serveMarkers.length>1);
      if(restored.history.past.length!==1){
        await mkdir(args.get('--output'),{recursive:true});
        await writeFile(path.join(args.get('--output'),'history-diagnostic.json'),JSON.stringify({previous,restored}));
      }
      assert.equal(restored.history.past.length,1,'New predictions must preserve prior undo entries');
      assert.ok(restored.history.past[0].scoreTracking.serveMarkers.length>1);
    }
  }
  await mkdir(args.get('--output'),{recursive:true});
  await page.screenshot({path:path.join(args.get('--output'),'serving-comparison.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:path.join(args.get('--output'),'serving-comparison-mobile.png'),fullPage:true});
  assert.deepEqual(errors,[]);
  const coverage=[];
  if(args.get('--coverage')!=='skip'){
    const catalog=await get('/api/editor-lab/tasks');
    for(const row of catalog.tasks){
      const item=await get(`/api/editor-lab/tasks/${encodeURIComponent(row.id)}`);
      for(const config of item.configurations.filter(c=>!c.humanReference)){
        const population=config.draftEvents??config.events;
        assert.equal(config.servingPredictions?.length,population.length,`Incomplete serving predictions: ${config.id}`);
        for(const event of population)assert.ok(config.servingPredictions.some(p=>p.id===event.id&&Math.abs(p.anchor-event.start)<1e-6));
      }
      coverage.push({index:coverage.length+1,configurations:item.configurations.filter(c=>!c.humanReference).length});
    }
  }
  const result={passed:true,recordings:coverage.length,configurations:coverage.reduce((s,r)=>s+r.configurations,0),humanMarkers:expectedHuman.length,
    checks:['Variant-specific anchors and complete predictions','Human N/F markers seek source video','Model markers survive reload','Manual edits and undo history survive newly prepared predictions','Desktop and mobile comparison rail'],errors};
  await writeFile(path.join(args.get('--output'),'serving-browser-checks.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
}finally{await browser.close();}
