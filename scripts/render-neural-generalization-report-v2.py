#!/usr/bin/env python3
"""Lossless columnar transport for the frozen report UI; no numerical changes.

Only active chart/table rows become browser objects. The exact input JSON bytes
are retained separately as gzip for download, with their SHA-256 identity.
"""
from __future__ import annotations

import argparse
from array import array
import base64
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

BASE_RENDERER_SHA = 'd825dac98ebece8b3fd3c96843452756afa2c720154329bf25d924f91d388198'
SCHEMA = 'lossless-columnar-report-v2'
MISSING = object()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def encoded_array(typecode, values):
    value = array(typecode, values)
    if sys.byteorder != 'little':
        value.byteswap()
    return base64.b64encode(value.tobytes()).decode('ascii')


def decoded_array(typecode, encoded):
    value = array(typecode)
    value.frombytes(base64.b64decode(encoded))
    if sys.byteorder != 'little':
        value.byteswap()
    return value


def pack_series(rows):
    names = list(dict.fromkeys(key for row in rows for key in row))
    columns = []
    for name in names:
        values = [row.get(name, MISSING) for row in rows]
        present = [v for v in values if v is not MISSING and v is not None]
        numeric = bool(present) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present)
        if numeric:
            if any(not math.isfinite(v) or isinstance(v, int) and abs(v) > 2**53-1 for v in present):
                raise ValueError('Numeric column cannot be represented exactly in browser: '+name)
            # Low-cardinality numbers (floors, padding, seeds) use dictionaries.
            first_values = set()
            for value in present:
                first_values.add(json.dumps(value, allow_nan=False))
                if len(first_values) > 64:
                    break
            numeric = len(first_values) > 64
        if numeric:
            states = bytes(0 if v is MISSING else 1 if v is None else 2 for v in values)
            column = {'name': name, 'kind': 'float64',
                      'values': encoded_array('d', (v if state == 2 else 0. for v, state in zip(values, states)))}
            if any(state != 2 for state in states):
                column['states'] = base64.b64encode(states).decode('ascii')
        else:
            dictionary, indexes, codes = [], {}, []
            for value in values:
                if value is MISSING:
                    codes.append(0)
                elif value is None:
                    codes.append(1)
                else:
                    key = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':'))
                    if key not in indexes:
                        indexes[key] = len(dictionary)+2
                        dictionary.append(value)
                    codes.append(indexes[key])
            column = {'name': name, 'kind': 'dictionary', 'dictionary': dictionary,
                      'codes': encoded_array('I', codes)}
        columns.append(column)
    return {'length': len(rows), 'columns': columns}


def unpack_series(packed):
    """Testing oracle only. The browser never reconstructs the full row array."""
    rows = [{} for _ in range(packed['length'])]
    for column in packed['columns']:
        if column['kind'] == 'float64':
            values = decoded_array('d', column['values'])
            states = base64.b64decode(column['states']) if 'states' in column else bytes([2])*len(rows)
            for row, state, value in zip(rows, states, values):
                if state:
                    row[column['name']] = None if state == 1 else value
        else:
            for row, code in zip(rows, decoded_array('I', column['codes'])):
                if code:
                    row[column['name']] = None if code == 1 else column['dictionary'][code-2]
    return rows


def load_base():
    path = Path(__file__).with_name('render-neural-generalization-report.py')
    if digest(path.read_bytes()) != BASE_RENDERER_SHA:
        raise ValueError('Frozen report UI source changed')
    spec = importlib.util.spec_from_file_location('frozen_report_ui', path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Frozen UI transport adapter mismatch: '+before[:80])
    return text.replace(before, after, 1)


BOOTSTRAP = r'''
function fromBase64(encoded){const raw=atob(encoded),bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);return bytes}
const LITTLE_ENDIAN=new Uint8Array(new Uint32Array([1]).buffer)[0]===1;
function typed(encoded,kind){const bytes=fromBase64(encoded),width=kind==='float64'?8:4;if(bytes.length%width)throw new Error('Invalid column buffer');if(LITTLE_ENDIAN)return kind==='float64'?new Float64Array(bytes.buffer):new Uint32Array(bytes.buffer);const output=kind==='float64'?new Float64Array(bytes.length/8):new Uint32Array(bytes.length/4),view=new DataView(bytes.buffer);for(let i=0;i<output.length;i++)output[i]=kind==='float64'?view.getFloat64(i*8,true):view.getUint32(i*4,true);return output}
class ColumnTable{
  constructor(packed,scopes){this.length=packed.length;this.scopes=scopes||{};this.columns=new Map();this.materializedRows=0;this.cursor=0;
    for(const c of packed.columns){let column;if(c.kind==='float64'){column={kind:c.kind,values:typed(c.values,c.kind),states:c.states?fromBase64(c.states):null};if(column.values.length!==this.length||column.states&&column.states.length!==this.length)throw new Error('Numeric column length differs');delete c.values;delete c.states}else if(c.kind==='dictionary'){column={kind:c.kind,codes:typed(c.codes,'uint32'),dictionary:c.dictionary};if(column.codes.length!==this.length)throw new Error('Categorical column length differs');delete c.codes}else throw new Error('Unknown column kind');this.columns.set(c.name,column)}
    const table=this;this.view=new Proxy(Object.create(null),{get(_target,key){return table.get(key,table.cursor)}})
  }
  get(name,index){const c=this.columns.get(name);let value;if(c){if(c.kind==='float64'){const state=c.states?c.states[index]:2;value=state===0?undefined:state===1?null:c.values[index]}else{const code=c.codes[index];value=code===0?undefined:code===1?null:c.dictionary[code-2]}}if(name==='recordingIds'&&value===undefined){const scope=this.get('scopeId',index);return scope===undefined?undefined:this.scopes[scope]}return value}
  row(index){const result={};for(const name of this.columns.keys()){const value=this.get(name,index);if(value!==undefined)result[name]=value}if(!result.recordingIds&&result.scopeId)result.recordingIds=this.scopes[result.scopeId];this.materializedRows++;return result}
  *map(callback){for(let i=0;i<this.length;i++){this.cursor=i;yield callback(this.view,i)}}
  filter(callback){const matched=[];for(let i=0;i<this.length;i++){this.cursor=i;if(callback(this.view,i))matched.push(this.row(i))}return matched}
}
async function loadPacked(){if(typeof DecompressionStream!=='function')throw new Error('This browser needs gzip DecompressionStream support to open this report. Download the JSON.gz file for another viewer.');const element=document.getElementById('packedReport'),bytes=fromBase64(element.textContent);element.textContent='';const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));const payload=await new Response(stream).json();if(payload.schema!=='lossless-columnar-report-v2')throw new Error('Unexpected report transport');const data=payload.header,rows=new ColumnTable(payload.series,data.scopes);return {data,rows,transport:payload.transport}}
const {data:DATA,rows:ROWS,transport:TRANSPORT}=await loadPacked();
const META=DATA.metadata||{},INVENTORY=Array.isArray(DATA.inventory)?DATA.inventory:(DATA.inventory?.records||[]),TASKS=DATA.tasks||[];
'''


def validate(data):
    metadata = data.get('metadata', {})
    if not metadata.get('syntheticFixture') and metadata.get('auditPassed') is not True:
        raise ValueError('Real report requires metadata.auditPassed=true')
    for row in data.get('series', []):
        if 'recordingIds' not in row and row.get('scopeId') and not isinstance(data.get('scopes', {}).get(row['scopeId']), list):
            raise ValueError('Unknown or invalid compact scopeId')
        floor = row.get('floorPercent')
        if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not math.isfinite(floor) or not 90 <= floor <= 100:
            raise ValueError('Recall floor must be within90..100')
        for key in ('precisionValue', 'recallValue', 'f1Value', 'eventF1'):
            value = row.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(key+' must be a fraction or null')


def audit_digest(data):
    content = {key: data[key] for key in ('inventory', 'scopes', 'series', 'tasks')}
    return digest(json.dumps(content, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def render(raw, renderer_sha):
    data = json.loads(raw)
    validate(data)
    source_sha = digest(raw)
    content_sha = audit_digest(data)
    metadata = data['metadata']
    if not metadata.get('syntheticFixture'):
        reference = metadata.get('audit')
        if not reference or digest(Path(reference['path']).read_bytes()) != reference['sha256']:
            raise ValueError('Real report audit identity absent or changed')
        gate = json.loads(Path(reference['path']).read_bytes())
        if gate.get('passed') is not True or gate.get('reportContentSha256') != content_sha:
            raise ValueError('Canonical report payload differs from audit')
    packed = {'schema': SCHEMA, 'header': {k: v for k, v in data.items() if k != 'series'},
              'series': pack_series(data['series']),
              'transport': {'sourceSha256': source_sha, 'reportContentSha256': content_sha,
                            'originalBytes': len(raw), 'baseRendererSha256': BASE_RENDERER_SHA}}
    packed_bytes = json.dumps(packed, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
    zipped_packed = gzip.compress(packed_bytes, compresslevel=6, mtime=0)
    zipped_original = gzip.compress(raw, compresslevel=6, mtime=0)
    template = load_base().TEMPLATE
    template = replace_once(template, '<script id="reportData" type="application/json">__DATA__</script><script>',
        '<script id="packedReport" type="application/octet-stream">__PACKED__</script>'
        '<script id="canonicalReport" type="application/gzip">__CANONICAL__</script><script>')
    before = "const DATA=JSON.parse(document.getElementById('reportData').textContent);\nconst META=DATA.metadata||{}, ROWS=DATA.series||[], INVENTORY=Array.isArray(DATA.inventory)?DATA.inventory:(DATA.inventory?.records||[]), TASKS=DATA.tasks||[];\n// Reuse each bound scope array; this is input normalization, never aggregation.\nfor(const row of ROWS){if(!row.recordingIds&&row.scopeId)row.recordingIds=DATA.scopes[row.scopeId]}"
    template = replace_once(template, before, BOOTSTRAP)
    template = replace_once(template, "'use strict';", "'use strict';\n(async function(){")
    template = replace_once(template, "['dtype','Model precision',", "['dtype','DINO embedding precision',")
    template = replace_once(template, '<select id="task"></select></div>',
        '<select id="task"></select></div><p id="taskExecution" class="subtle" style="overflow-wrap:anywhere" aria-live="polite"></p>')
    template = replace_once(template, 'function rolesFor(t,id)', r'''
function renderTaskExecution(){
 const selected=$('task').value, task=selected==='*'?null:TASKS[Number(selected)], execution=task?.trainingExecution;
 if(selected==='*'){$('taskExecution').textContent=META.executionSharing||'Select a task to inspect its physical training owner and any checkpoint reuse.';return}
 if(!execution){$('taskExecution').textContent='No neural training-execution receipt is attached to this saved task.';return}
 const owner=String(execution.physicalOwnerTaskId??'unspecified');
 $('taskExecution').textContent=(execution.trainingPerformedForThisTask?
  'Trained for this task. Physical training owner: '+owner+'.':
  'Reused training. Physical training owner: '+owner+'. All fixed checkpoint epochs ('+(execution.reusedAllFixedCheckpointEpochs||[]).join(', ')+') were inherited; this task did not run another optimizer fit.')+
  ' Calibration predictions and operating-point selection remain task-local. Draws sharing an owner are correlated experimental views, not independent training replications.';
}
function rolesFor(t,id)''')
    template = replace_once(template, 'function renderInventory(){', 'function renderInventory(){renderTaskExecution();')
    template = replace_once(template,
        'Exact-label scope: precision uses padded human export; recall measures retained human rally core. F1_padP_coreR is the primary ranking metric at the declared target padding.',
        'Time-based exact-label metrics: Recall is the fraction of human rally-core seconds retained after model padding and gap joining. Precision is the fraction of model export seconds matching padded human export. F1_padP_coreR combines them. These are not rally-count recall; event F1 and completely lost rally counts are separate table columns.')
    template = replace_once(template,
        'Reviewed draft scope: continuous-review metadata exists, but these recordings remain separate from completed semantic rally gold.',
        'Time-based reviewed-draft metrics: Recall is the fraction of reviewed live seconds retained after model padding and gap joining. Precision is the fraction of export seconds matching the padded reviewed-live reference. These recordings remain separate from completed semantic rally gold; this curve does not measure rally-count recall.')
    template = replace_once(template,
        'Coverage / proxy scope: these results do not certify precise rally separation. Saved export targets already include their export padding. Training on approximate individual export cores does not change the final reference tier.',
        'Export-coverage metrics: Recall is the fraction of the fixed human export seconds retained; Precision is the fraction of model export seconds matching that target. Human export targets already include their padding. These results do not certify precise rally separation or semantic rally recall; approximate-core training does not change the reference tier.')
    template = replace_once(template, '<div id="contextWarning" class="notice" role="note"></div>',
        '<div id="scenarioWarning" class="notice" role="note" hidden></div>'
        '<div id="panelWarning" class="notice" role="note"></div>'
        '<p class="subtle" id="precisionWarning">FP16 / INT8 change DINO embedding extraction only. Temporal heads stay FP32-trained; decoder settings and recall-floor eligibility use frozen FP32 calibration, with no recalibration. Recall on transferred embeddings or new footage can fall below the selected floor.</p>'
        '<div id="contextWarning" class="notice" role="note"></div>')
    template = replace_once(template, 'renderCharts();renderTable();renderTaskSelect();renderInventory();', r'''
const activeScenario=$('filter-scenario').value;
const proxyNotes={
 'export-rally-selection':'Calibration mixes exact rally labels with approximate individual cores from reviewed exports. Evaluation still uses the selected exact, draft or export-coverage reference.',
 'export-rally-training':'Reviewed-export individual cores supply approximate four-head training targets; calibration uses the designated original exact-label group. Evaluation still uses the selected exact, draft or export-coverage reference.'
};
$('scenarioWarning').hidden=!proxyNotes[activeScenario];$('scenarioWarning').textContent=proxyNotes[activeScenario]?proxyNotes[activeScenario]+' Fixed epochs do not make this a label-only experiment: four-head supervision also changes sampling, exact-tier scaling, optimizer-update counts and student tier order.':'';
const panelDescriptions={
 'all-labeled':'All labeled videos includes neural fitting and calibration footage. It is a descriptive all-video panel, not an unseen-video test.',
 'outside-original-sources':'These source groups were held out from the original corpus only. Expanded and export-proxy variants train on some of them; this panel is not uniformly unseen for those variants.',
 'common-unseen':'These source groups were reserved from fitting and calibration for every registered neural task. This is the common unseen panel for matched comparisons.',
 'task-unseen-sources':'This panel excludes both fitting and calibration source groups for the selected task. Recording populations can differ across draws; compare matched scopes only.',
 'historical-nested-exact':'Each historical outer fold held out its evaluated source group. This uses the original context/masking regime described below.'
};
$('panelWarning').textContent=(panelDescriptions[$('filter-panel').value]||'Use the saved task roles and recording scope to determine which sources were held out.')+' Production-exposure filters describe production lineage; they do not remove neural training or calibration sources.';
renderCharts();renderTable();renderTaskSelect();renderInventory();''')
    template = replace_once(template, "const unique=a=>[...new Set(a.filter(x=>x!==null&&x!==undefined).map(String))].sort((a,b)=>a.localeCompare(b,undefined,{numeric:true}));",
        "const unique=a=>{const values=new Set();for(const x of a)if(x!==null&&x!==undefined)values.add(String(x));return [...values].sort((a,b)=>a.localeCompare(b,undefined,{numeric:true}))};")
    start = template.index("$('download').onclick=")
    end = template.index('\nupdate();', start)
    template = template[:start]+r'''$('download').onclick=()=>{const bytes=fromBase64(document.getElementById('canonicalReport').textContent);const blob=new Blob([bytes],{type:'application/gzip'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=META.syntheticFixture?'SYNTHETIC-qa-fixture.json.gz':'neural-generalization-results.json.gz';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
$('download').textContent='Download exact source JSON.gz';
$('download').title=`Exact original JSON bytes after gzip decompression. SHA-256: ${TRANSPORT.sourceSha256}`;
$('provenance').textContent+=` · Lossless columnar transport v2 · Audited payload SHA-256: ${TRANSPORT.reportContentSha256} · Source download is gzip-compressed JSON (${TRANSPORT.originalBytes.toLocaleString()} original bytes).`;
''' + template[end:]
    template = replace_once(template, 'sourceRows:ROWS.length,syntheticFixture:!!META.syntheticFixture',
        'sourceRows:ROWS.length,syntheticFixture:!!META.syntheticFixture,columnar:true,materializedRows:()=>ROWS.materializedRows,columnCount:ROWS.columns.size,transport:TRANSPORT,activeRows:()=>filtered,readCell:(index,name)=>ROWS.get(name,index)')
    template = replace_once(template, '</script></body></html>', r'''
})().catch(error=>{const panel=document.createElement('div');panel.className='notice error';panel.textContent='Report could not open: '+String(error.message||error);document.querySelector('main').prepend(panel);window.__reportError=String(error);console.error(error)});
</script></body></html>''')
    replacements = {'__PACKED__': base64.b64encode(zipped_packed).decode(),
                    '__CANONICAL__': base64.b64encode(zipped_original).decode(),
                    '__SOURCE_SHA__': source_sha, '__RENDERER_SHA__': renderer_sha}
    for key, value in replacements.items():
        template = template.replace(key, value)
    receipt = {'kind': SCHEMA, 'sourceSha256': source_sha, 'reportContentSha256': content_sha,
               'rows': len(data['series']), 'columns': len(packed['series']['columns']),
               'canonicalJsonBytes': len(raw), 'canonicalGzipBytes': len(zipped_original),
               'packedJsonBytes': len(packed_bytes), 'packedGzipBytes': len(zipped_packed),
               'htmlBytes': len(template.encode()), 'baseRendererSha256': BASE_RENDERER_SHA,
               'rendererSha256': renderer_sha, 'syntheticFixture': bool(metadata.get('syntheticFixture'))}
    return template, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--allow-synthetic', action='store_true')
    args = parser.parse_args()
    raw = args.input.read_bytes()
    if json.loads(raw).get('metadata', {}).get('syntheticFixture') and not args.allow_synthetic:
        raise ValueError('Synthetic fixture requires --allow-synthetic')
    html, receipt = render(raw, digest(Path(__file__).read_bytes()))
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(html)
    receipt['output'] = {'path': str(args.output), 'sha256': digest(args.output.read_bytes())}
    if args.receipt:
        with args.receipt.open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
