import assert from 'node:assert/strict';
import test from 'node:test';
import { attachLabServing, servingPopulation } from '../../lib/production-editor-lab-serving.ts';
import { addMissingLabServeMarkers, initialLabDraft } from '../../lib/production-editor-lab-draft.ts';
import type { LabConfiguration, ProductionEditorLabTask } from '../../lib/production-editor-lab.ts';

const configuration: LabConfiguration = { id: 'variant', label: 'Variant', description: '', sourcePolicy: 'frozen',
  events: [10,20,30].map((start,i)=>({id:`r${i}`,parentId:`r${i}`,start,end:start+5})), proposals: [], removals: [], splits: [] };
const receipt = () => ({population:servingPopulation(configuration), output:{candidates:configuration.events.map((e,i)=>({
  id:e.id,anchor:e.start,side:i===1?'far':'near',verdict:['near','review','not-serve'][i],nearProbability:.75,
  serveDecisionSource:'serve-head',reviewReasons:[],
}))}});
const task = {schemaVersion:1,id:'recording-test',name:'Example',durationSeconds:60,mediaUrl:'/video',sourceRevision:'a'.repeat(64),
  ignoredIntervals:[],configurations:[configuration],signals:{times:[],live:[],serve:[],end:[],keep:[]},serving:[],provenance:{labelBlind:true,sourceHashes:{}}} as ProductionEditorLabTask;

test('each variant receives only its own complete serving population, retaining rejected predictions for inspection',()=>{
  const attached=attachLabServing(configuration,receipt(),60);
  assert.equal(attached.servingPredictions?.length,3);
  assert.deepEqual(attached.events.map(e=>e.serve?.side),['near','review',undefined]);
  assert.equal(configuration.events[0].serve,undefined);
  assert.throws(()=>attachLabServing({...configuration,events:configuration.events.map((e,i)=>i?e:{...e,start:11})},receipt(),60),/population/);
  const missing=receipt();missing.output.candidates.pop();
  assert.throws(()=>attachLabServing(configuration,missing,60),/anchors/);
  const shifted=receipt();shifted.output.candidates[0].anchor=11;
  assert.throws(()=>attachLabServing(configuration,shifted,60),/anchors/);
});

test('suppression retains serving results for the full original population so undo can restore markers',()=>{
  const suppressed={...configuration,draftEvents:configuration.events,events:configuration.events.slice(1)};
  const result=attachLabServing(suppressed,receipt(),60);
  assert.equal(result.draftEvents?.[0].serve?.side,'near');
  assert.equal(result.events.length,2);
});

test('new serving results fill old drafts without replacing manual edits, shifted starts, or deleted markers',()=>{
  const model=initialLabDraft(task,attachLabServing(configuration,receipt(),60));
  const saved=initialLabDraft(task,configuration);
  assert.equal(addMissingLabServeMarkers(saved,model).scoreTracking.serveMarkers.length,2);
  const manual=structuredClone(saved);
  manual.scoreTracking.serveMarkers=[{...model.scoreTracking.serveMarkers[0],side:'far',origin:'manual'}];
  manual.scoreTracking.removedModelMarkerIds.push(model.scoreTracking.serveMarkers[1].id);
  const merged=addMissingLabServeMarkers(manual,model);
  assert.deepEqual(merged.scoreTracking.serveMarkers,manual.scoreTracking.serveMarkers);
  saved.cuts[0].coreStart=11;
  saved.userTouchedCutIds.push('r1');
  assert.equal(addMissingLabServeMarkers(saved,model).scoreTracking.serveMarkers.length,0);
});

test('untouched model sides refresh and fresh serve-gate rejections remove only model markers',()=>{
  const fresh=initialLabDraft(task,attachLabServing(configuration,receipt(),60));
  const old=structuredClone(fresh);
  old.scoreTracking.serveMarkers[0].side='far';
  old.scoreTracking.serveMarkers[0].modelSide='far';
  old.scoreTracking.serveMarkers.push({...old.scoreTracking.serveMarkers[0],id:'lab-serve:r2',rallyId:'r2',timestamp:30});
  const merged=addMissingLabServeMarkers(old,fresh);
  assert.equal(merged.scoreTracking.serveMarkers[0].side,'near');
  assert.equal(merged.scoreTracking.serveMarkers.length,2);
  assert.deepEqual(merged.cuts,old.cuts);
  old.reviewedCutIds.push('r0');
  assert.equal(addMissingLabServeMarkers(old,fresh).scoreTracking.serveMarkers[0].side,'far');
  old.reviewedCutIds=[];
  old.scoreTracking.serveMarkers[0].side='near'; // Manual correction, immutable modelSide remains far.
  assert.equal(addMissingLabServeMarkers(old,fresh).scoreTracking.serveMarkers[0].modelSide,'far');
});
