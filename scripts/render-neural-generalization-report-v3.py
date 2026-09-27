#!/usr/bin/env python3
"""Presentation-only matched variant view over the frozen lossless v2 transport."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

V2_SHA = '2fa77b4b3696740397da4ef46b375ae20687b58cf4c117ccc9044d1ee6e7cc43'


def sha(value): return hashlib.sha256(value).hexdigest()


def load_v2():
    path = Path(__file__).with_name('render-neural-generalization-report-v2.py')
    if sha(path.read_bytes()) != V2_SHA: raise ValueError('Frozen v2 renderer changed')
    spec = importlib.util.spec_from_file_location('frozen_columnar_report_v2', path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


CONTROLS = r'''
<section class="section" aria-label="Comparison mode">
 <div class="section-head"><h2>Choose a comparison</h2><select id="comparisonMode" aria-label="Comparison mode"><option value="models">Models within one scenario</option><option value="variants">Variants for one model</option></select></div>
 <div id="variantControls" hidden><p class="subtle">Use one model, draw or saved complete-draw mean, embedding precision, padding and label policy. Historical nested evaluation stays in the separate single-scenario view.</p><div id="variantChoices" class="legend" role="group" aria-label="Variants to compare"></div></div>
 <div id="comparisonWarning" class="notice" role="status" hidden></div>
</section>
'''

METHODS = r'''
const COMPARISON_VARIANTS=['original-medium','expanded-medium','expanded-large','expanded-wide-validation','export-rally-training','export-rally-selection'];
const isVariantMode=()=>$('comparisonMode').value==='variants';
const selectedVariants=()=>[...$('variantChoices').querySelectorAll('input:checked')].map(el=>el.value);
let comparisonModeEnabled=false,comparisonVariants=new Set();
const presentVariants=new Set(ROWS.map(scenario));
$('variantChoices').innerHTML=COMPARISON_VARIANTS.map(value=>`<label class="tag" style="padding:8px 10px"><input type="checkbox" value="${value}" ${presentVariants.has(value)?'checked':''}> ${value}${presentVariants.has(value)?'':' (no saved rows)'}</label>`).join('');
function prepareComparison(){
 const variants=isVariantMode();comparisonModeEnabled=variants;comparisonVariants=new Set(selectedVariants());$('variantControls').hidden=!variants;$('filter-scenario').disabled=variants;
 for(const id of ['model','draw','dtype']){const el=$('filter-'+id),all=[...el.options].find(o=>o.value==='*');if(all)all.disabled=variants;
  if(variants&&el.value==='*'){const preferred=id==='model'?'mobile-tcn':id==='draw'?'all registered draws':'fp32';el.value=[...el.options].some(o=>o.value===preferred)?preferred:[...el.options].find(o=>o.value!=='*')?.value||''}}
}
function rowMatches(r){
 if(comparisonModeEnabled&&(fixed(r)||!comparisonVariants.has(scenario(r))))return false;
 return fields.every(([id,,,getter])=>comparisonModeEnabled&&id==='scenario'||fixed(r)&&['draw','dtype','model'].includes(id)||$('filter-'+id).value==='*'||getter(r)===$('filter-'+id).value);
}
function renderComparisonNotice(){
 const enabled=isVariantMode();$('comparisonWarning').hidden=!enabled;if(!enabled)return;
 const neural=filtered.filter(r=>!fixed(r)),requested=selectedVariants(),present=new Set(neural.map(scenario));
 const missing=requested.filter(v=>!present.has(v)),scopes=new Set(neural.map(scopeKey));
 const mismatch=scopes.size>1,empty=!neural.length;
 $('comparisonWarning').className=mismatch||empty||missing.length?'notice error':'notice';
 $('comparisonWarning').textContent=(mismatch?'SOURCE SCOPE MISMATCH: these displayed variants are not a matched comparison.':empty?'No saved operating points for these variant/filter choices.':'Same recording scope for the displayed variants; values are copied from their saved evaluations.')+
  (missing.length?' No saved rows in this view for: '+missing.join(', ')+'.':'')+
  ' Infeasible or incomplete saved means remain gaps. The viewer never averages available draws or selects a favorable subset.';
 if(mismatch){$('scopeWarning').hidden=false;$('scopeWarning').textContent='Selected variants contain different recording scopes, including incomplete rows. Use common-unseen for the reserved shared panel; no matched-scope metric is synthesized.'}
}
$('comparisonMode').addEventListener('change',()=>{
 if(isVariantMode()){$('filter-panel').value=[...$('filter-panel').options].some(o=>o.value==='common-unseen')?'common-unseen':$('filter-panel').value;
  if($('filter-scenario').value==='historical-nested')$('filter-scenario').value=[...$('filter-scenario').options].find(o=>COMPARISON_VARIANTS.includes(o.value))?.value||$('filter-scenario').value}
 hiddenLines.clear();update();
});
$('variantChoices').addEventListener('change',()=>{hiddenLines.clear();update()});
'''


def render(raw, renderer_sha):
    v2 = load_v2(); html, receipt = v2.render(raw, renderer_sha)
    replace = v2.replace_once
    html = replace(html, '<div class="filters" id="filters" aria-label="Report filters">',
        CONTROLS+'<div class="filters" id="filters" aria-label="Report filters">')
    html = replace(html, 'const fields=[', METHODS+"\nconst fields=[['model','Model',unique(ROWS.map(r=>fixed(r)?null:r.model)),r=>String(r.model),META.defaultModel],")
    html = replace(html, "const optionAll=id==='draw'||id==='dtype';", "const optionAll=id==='model'||id==='draw'||id==='dtype';")
    html = replace(html, "function update(){filtered=ROWS.filter(r=>fields.every(([id,,,getter])=>(fixed(r)&&['draw','dtype'].includes(id))||$('filter-'+id).value==='*'||getter(r)===$('filter-'+id).value));",
        'function update(){prepareComparison();filtered=ROWS.filter(rowMatches);')
    html = replace(html, "if($('showProduction').checked&&$('filter-scenario').value!=='fixed-production')", "if($('showProduction').checked&&(isVariantMode()||$('filter-scenario').value!=='fixed-production'))")
    html = replace(html, "!['scenario','draw','dtype'].includes(id)", "!['scenario','draw','dtype','model'].includes(id)")
    html = replace(html, "const historical=$('filter-scenario').value==='historical-nested';", "const historical=!isVariantMode()&&$('filter-scenario').value==='historical-nested';")
    html = replace(html, "const activeScenario=$('filter-scenario').value;", "const activeScenario=isVariantMode()?'variant-comparison':$('filter-scenario').value;")
    html = replace(html, 'const panelDescriptions={', r'''
if(isVariantMode()&&selectedVariants().some(v=>v.startsWith('export-rally-'))){$('scenarioWarning').hidden=false;$('scenarioWarning').textContent='Export-rally-training uses approximate reviewed-export cores for training. Export-rally-selection also uses one whole export-source group for calibration alongside exact gold. Evaluation gold still follows the selected label tier. Fixed epochs do not make these label-only changes: supervision also changes sampling, scaling and optimizer-update counts.'}
const panelDescriptions={''')
    html = replace(html, 'renderCharts();renderTable();renderTaskSelect();renderInventory();',
        'renderComparisonNotice();renderCharts();renderTable();renderTaskSelect();renderInventory();')
    before = "(t.scenario===undefined&&t.variant===undefined||String(t.scenario??t.variant)===$('filter-scenario').value)&&($('filter-draw').value==='*'||t.draw===undefined&&t.seed===undefined||String(t.draw??t.seed)===$('filter-draw').value)"
    after = "(isVariantMode()?selectedVariants().includes(String(t.scenario??t.variant)):(t.scenario===undefined&&t.variant===undefined||String(t.scenario??t.variant)===$('filter-scenario').value))&&($('filter-model').value==='*'||String(t.model)===$('filter-model').value)&&($('filter-draw').value==='*'||t.draw===undefined&&t.seed===undefined||String(t.draw??t.seed)===$('filter-draw').value)"
    html = replace(html, before, after)
    html = replace(html, 'sourceRows:ROWS.length,syntheticFixture:!!META.syntheticFixture',
        'presentationVersion:3,comparisonMode:()=>$(\'comparisonMode\').value,selectedVariants,sourceRows:ROWS.length,syntheticFixture:!!META.syntheticFixture')
    html = replace(html, ' · Lossless columnar transport v2 · Audited payload SHA-256:',
        ' · Matched variant view v3 / lossless columnar transport v2 · Audited payload SHA-256:')
    receipt = {**receipt, 'kind': 'matched-variant-report-view-v3', 'transportKind': receipt['kind'],
        'htmlBytes': len(html.encode()), 'v2RendererSha256': V2_SHA,
        'presentationPolicy': 'Saved rows only; one-model matched-source variant comparison, no aggregation/reselection; historical separate; production same-scope only.'}
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
