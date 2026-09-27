#!/usr/bin/env python3
"""Scoped task membership UI over the unchanged, lossless v3 report payload."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

V3_SHA = '87fafb74c793051b93cb1a0000cf2fd21464f88f7ec1ec196fcde9468af58dde'


def sha(value): return hashlib.sha256(value).hexdigest()


def load_v3():
    path = Path(__file__).with_name('render-neural-generalization-report-v3.py')
    if sha(path.read_bytes()) != V3_SHA: raise ValueError('Frozen v3 renderer changed')
    spec = importlib.util.spec_from_file_location('frozen_variant_report_v3', path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


TASK_SELECT = r'''
let matchingTaskOptions=[];
function selectedTasksForInventory(){
 const selected=$('task').value;
 return (selected==='*'?matchingTaskOptions:matchingTaskOptions.filter(({i})=>String(i)===selected)).map(({t})=>t);
}
function renderTaskSelect(){
 const previous=$('task').value,selectedDraw=$('filter-draw').value;
 matchingTaskOptions=TASKS.map((t,i)=>({t,i})).filter(({t})=>{
  if(!(isVariantMode()?selectedVariants().includes(scenario(t)):scenario(t)===$('filter-scenario').value))return false;
  if(!isVariantMode()&&fixed(t))return true;
  return ($('filter-model').value==='*'||String(t.model)===$('filter-model').value)&&
  ($('filter-dtype').value==='*'||dtype(t)===$('filter-dtype').value)&&
  (['*','all registered draws'].includes(selectedDraw)||draw(t)===selectedDraw);
 });
 const count=matchingTaskOptions.length;
 $('task').disabled=!count;
 $('task').innerHTML=count?`<option value="*">All ${count} matching saved tasks (combined roles)</option>`+
  matchingTaskOptions.map(({t,i})=>`<option value="${i}">${esc(taskName(t,i))}</option>`).join(''):
  '<option value="">No saved tasks match these filters</option>';
 if(count)$('task').value=previous==='*'||matchingTaskOptions.some(({i})=>String(i)===previous)?previous:String(matchingTaskOptions[0].i);
 $('taskScope').textContent=count?(!isVariantMode()&&$('filter-scenario').value==='fixed-production'?`${count} fixed production tasks match this scenario. Neural model, draw and embedding precision selectors do not change their saved memberships.`:
  `${count} saved tasks match the scenario / selected variants, model, draw and embedding precision. A complete-draw mean has no separate training task; choose a concrete draw to inspect its roles, or explicitly combine only these matching tasks.`):
  'No saved task membership matches this view. Video inventory and production exposure remain visible; no unrelated neural task roles are shown.';
}
'''


def render(raw, renderer_sha):
    v3 = load_v3(); html, receipt = v3.render(raw, renderer_sha)
    replace = v3.load_v2().replace_once
    start = html.index('function renderTaskSelect(){')
    end = html.index('\nfunction renderTaskExecution(){', start)
    html = html[:start]+TASK_SELECT+html[end:]
    html = replace(html, '<select id="task"></select></div>', '<select id="task"></select></div><p id="taskScope" class="subtle" role="status"></p>')
    html = replace(html, 'Durations below use seconds.</p>', 'Durations below use seconds.</p><p class="subtle">Event F1 uses one-to-one chronological matching at intersection-over-union (IoU) of at least 0.5, on original rallies untouched by ignored spans. Boundary errors describe matched, uncensored events only. The primary recall curve measures retained core time after padding and short-gap joining; it is not rally-count recall.</p>')
    html = replace(html, "const selected=$('task').value==='*'?TASKS:[TASKS[Number($('task').value)]].filter(Boolean);", 'const selected=selectedTasksForInventory();')
    html = replace(html, " const selected=$('task').value, task=selected==='*'?null:TASKS[Number(selected)], execution=task?.trainingExecution;",
        " if(!matchingTaskOptions.length){$('taskExecution').textContent='No matching task selected.';return}\n const selected=$('task').value, task=selected==='*'?null:selectedTasksForInventory()[0], execution=task?.trainingExecution;")
    html = replace(html, "if(selected==='*'){$('taskExecution').textContent=META.executionSharing||'Select a task to inspect its physical training owner and any checkpoint reuse.';return}",
        "if(selected==='*'){$('taskExecution').textContent=`Combined roles from only ${matchingTaskOptions.length} matching saved tasks. `+(META.executionSharing||'Select one matching task to inspect its physical training owner and checkpoint reuse.');return}")
    html = replace(html, 'presentationVersion:3,', 'presentationVersion:4,matchingTaskIndices:()=>matchingTaskOptions.map(({i})=>i),selectedTaskIndices:()=>matchingTaskOptions.filter(({t})=>selectedTasksForInventory().includes(t)).map(({i})=>i),')
    html = replace(html, 'Matched variant view v3 / lossless columnar transport v2', 'Scoped task view v4 / matched variants v3 / lossless columnar transport v2')
    receipt = {**receipt, 'kind': 'scoped-task-report-view-v4', 'htmlBytes': len(html.encode()), 'v3RendererSha256': V3_SHA,
        'taskMembershipPolicy': 'Mean draw exposes actual matching draws; default one concrete task; explicit all combines matching scenario/model/draw/precision only; no metric changes.'}
    return html, receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('input', 'output', 'receipt'): p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--allow-synthetic', action='store_true'); a = p.parse_args(); raw = a.input.read_bytes()
    if json.loads(raw).get('metadata', {}).get('syntheticFixture') and not a.allow_synthetic: raise ValueError('Synthetic fixture requires explicit flag')
    html, receipt = render(raw, sha(Path(__file__).read_bytes()))
    with a.output.open('x', encoding='utf-8', newline='\n') as stream: stream.write(html)
    receipt['output'] = {'path': str(a.output), 'sha256': sha(a.output.read_bytes())}
    with a.receipt.open('x') as stream: json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__': main()
