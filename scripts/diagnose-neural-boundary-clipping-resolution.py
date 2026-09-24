#!/usr/bin/env python3
"""Post-hoc size of clipped-start offsets; frozen scores remain unchanged."""
from __future__ import annotations
from analysis.private_ledger import private_value
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
from statistics import mean
import numpy as np

ROOT=Path(private_value('private-reference-0079'))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return {'path':str(p),'sha256':sha(p),'sizeBytes':Path(p).stat().st_size}


def main():
    previous=read(ROOT/'observed-start-diagnosis.json')
    data=read(previous['input']['path']);metadata={};sources=[]
    for record in data['records']:
        reference=record['featureCache']
        if sha(reference['path'])!=reference['sha256']:raise ValueError('Feature metadata changed')
        with np.load(reference['path'],allow_pickle=False) as z:m=json.loads(str(z['metadata_json']))
        metadata[record['id']]={'fps':float(m['fps']),'frameDurationSeconds':1/float(m['fps'])}
        sources.append(reference)
    rows=[]
    for original in previous['policySeedRows']:
        starts=original['unobservedStarts']
        if not all(x['parentClipped'] for x in starts):raise ValueError('Other clipped start mechanism needs separate analysis')
        correct=[x for x in starts if x['withinOneSecondOfOwnGold']]
        if len(correct)!=original['correctMatchesExcludedByObservationFlag1s']:
            raise ValueError('Own-gold near matches do not exactly account for observed-filter loss')
        tiny=[x for x in correct if x['startErrorSeconds']<=.001+1e-12]
        within_frame=[x for x in correct if x['startErrorSeconds']<=metadata[x['recordingId']]['frameDurationSeconds']+1e-12]
        rows.append({'seed':original['seed'],'mode':original['mode'],
                     'unobservedStarts':len(starts),'correctMatchesRemoved1s':len(correct),
                     'offsetAtMost1ms':len(tiny),'offsetAtMostOneFrame':len(within_frame),
                     'offsetOver1msAtMostOneFrame':len(within_frame)-len(tiny),
                     'offsetOverOneFrameAtMost1s':len(correct)-len(within_frame),
                     'offsetOver1sNotCorrectAtTolerance':len(starts)-len(correct),
                     'examples':[{**x,**metadata[x['recordingId']]} for x in starts]})
    keys=[k for k,v in rows[0].items() if isinstance(v,(int,float)) and k!='seed']
    aggregates={mode:{k:{'mean':mean(r[k] for r in rows if r['mode']==mode),
                        'min':min(r[k] for r in rows if r['mode']==mode),
                        'max':max(r[k] for r in rows if r['mode']==mode)} for k in keys}
                for mode in ('proposal_confirmation','full_parent')}
    output={'kind':'post-hoc-clipped-start-resolution-diagnosis-v1','createdAt':datetime.now(timezone.utc).isoformat(),
            'postHoc':True,'frozenMetricsChanged':False,'outcomeRerun':False,
            'sourceDiagnosis':ref(ROOT/'observed-start-diagnosis.json'),'frameMetadata':metadata,'metadataSources':sources,
            'definition':'Offsets compare gold start to original parent-clipped boundary. One frame is 1/fps from hash-bound source feature metadata. Cumulative <=1ms and <=one-frame counts overlap; disjoint bins also supplied.',
            'verification':'Every unobserved start is parent clipped. For each of six seed/mode cells, number within1s of its own gold start equals the exact ordinary-minus-observed matching loss.',
            'limit':'This quantifies provenance/rounding sensitivity without changing observed-marker policy, running new configurations, or claiming a clipped marker is a validated serve contact.',
            'script':ref(Path(__file__)),'policySeedRows':rows,'aggregates':aggregates}
    path=ROOT/'clipped-start-resolution-diagnosis.json'
    with path.open('x',encoding='utf-8') as f:json.dump(output,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'passed':True,'output':ref(path),'frameMetadata':metadata,'aggregates':aggregates},indent=2))


if __name__=='__main__':main()
