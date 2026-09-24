"""Combine explicitly completed independent cases, preserving source receipts."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows={};specs={};receipts=[]
    for folder in a.input:
        path=folder/'result.json';raw=path.read_bytes();report=json.loads(raw)
        plan=json.loads((folder/'pipeline-plan.json').read_text());by_id={r['id']:r for r in plan['cases']}
        receipt={'result':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'runId':report['runId']}
        apk=folder/'apk-sha256.json'
        if apk.exists():receipt['apk']=json.loads(apk.read_text(encoding='utf-8-sig'))
        receipts.append(receipt)
        for row in report['results']:
            if row['status']!='complete':continue
            spec=by_id[row['id']];family=spec['family']
            if family not in ('production','mobile','dino') or spec['warmup']:continue
            if family in rows:raise ValueError('Duplicate successful family: '+family)
            if not row['servingSideReady'] or not row['sideSwitchReady']:raise ValueError('Incomplete score outputs')
            rows[family]={**row,'sourceResult':str(path)};specs[family]=spec
    if set(rows)!=set(('production','mobile','dino')):raise ValueError('Missing completed family')
    if len({s['seconds'] for s in specs.values()})!=1:raise ValueError('Video scopes differ')
    a.output.mkdir(parents=True,exist_ok=True)
    run_id='combined-'+hashlib.sha256(json.dumps(receipts,sort_keys=True).encode()).hexdigest()[:16]
    order=('production','mobile','dino')
    result={'runId':run_id,'status':'complete','scope':'One successful whole-recording pass per pipeline, collected across benchmark sessions. See source receipts; failed and interrupted cases are excluded.',
            'sourceReceipts':receipts,'results':[rows[k] for k in order]}
    (a.output/'result.json').write_text(json.dumps(result,indent=2))
    (a.output/'pipeline-plan.json').write_text(json.dumps({'runId':run_id,'cases':[specs[k] for k in order]},indent=2))
    print(json.dumps({'status':'complete','families':list(order),'sourceReceipts':receipts},indent=2))
if __name__=='__main__':main()
