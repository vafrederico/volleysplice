import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const source=readFileSync(new URL('./render-neural-generalization-report-v4.py',import.meta.url),'utf8');
const body=source.match(/TASK_SELECT = r'''([\s\S]*?)'''/)[1];
function runtime(){
 const values={model:'a',scenario:'original-medium',draw:'all registered draws',dtype:'fp32'};
 const elements=Object.fromEntries(Object.entries(values).map(([k,v])=>['filter-'+k,{value:v}]));
 elements.task={value:'',disabled:false,set innerHTML(value){this.html=value;this.options=[...value.matchAll(/value="([^"]*)"/g)].map(m=>({value:m[1]}));this.value=this.options[0]?.value??''}};
 elements.taskScope={textContent:''};
 const tasks=[{id:'a1',model:'a',variant:'original-medium',draw:1,precision:'fp32'},{id:'a2',model:'a',variant:'original-medium',draw:2,precision:'fp32'},{id:'a-half',model:'a',variant:'original-medium',draw:1,precision:'fp16'},{id:'other-model',model:'b',variant:'original-medium',draw:1,precision:'fp32'},{id:'other-scenario',model:'a',variant:'expanded-medium',draw:1,precision:'fp32'}];
 tasks.push({id:'production',model:'productionDefault',variant:'fixed-production',draw:'fixed',precision:'shipped'});
 const context=vm.createContext({TASKS:tasks,$:id=>elements[id],mode:false,variants:['original-medium','expanded-medium'],scenario:t=>String(t.variant),draw:t=>String(t.draw),dtype:t=>String(t.precision),fixed:t=>t.variant==='fixed-production',esc:x=>x,taskName:t=>t.id});
 vm.runInContext('const isVariantMode=()=>mode, selectedVariants=()=>variants;'+body,context);
 return {elements,context,update:()=>vm.runInContext('renderTaskSelect()',context),selected:()=>Array.from(vm.runInContext('selectedTasksForInventory().map(t=>t.id)',context))};
}
test('mean exposes actual draws and defaults one concrete task',()=>{const r=runtime();r.update();assert.deepEqual(r.elements.task.options.map(o=>o.value),['*','0','1']);assert.equal(r.elements.task.value,'0');assert.deepEqual(r.selected(),['a1'])});
test('explicit all uses matching tasks only and precision filters membership',()=>{const r=runtime();r.update();r.elements.task.value='*';assert.deepEqual(r.selected(),['a1','a2']);r.elements['filter-dtype'].value='fp16';r.update();assert.deepEqual(r.selected(),['a-half']);assert.equal(r.elements.task.options.length,2)});
test('empty historical context shows no unrelated roles',()=>{const r=runtime();r.elements['filter-scenario'].value='historical-nested';r.update();assert.equal(r.elements.task.disabled,true);assert.equal(r.elements.task.value,'');assert.deepEqual(r.selected(),[]);assert.match(r.elements.taskScope.textContent,/No saved task membership/)});
test('variant mode includes only selected variants with same model draw and precision',()=>{const r=runtime();r.context.mode=true;r.elements['filter-draw'].value='1';r.update();r.elements.task.value='*';assert.deepEqual(r.selected(),['a1','other-scenario']);r.context.variants=['expanded-medium'];r.update();assert.deepEqual(r.selected(),['other-scenario'])});
test('fixed production scenario ignores neural model draw and precision like its chart',()=>{const r=runtime();r.elements['filter-scenario'].value='fixed-production';r.elements['filter-draw'].value='2';r.update();assert.deepEqual(r.selected(),['production']);assert.equal(r.elements.task.disabled,false)});
