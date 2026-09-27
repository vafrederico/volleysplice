#!/usr/bin/env python3
"""Gap-level blind spots of removal-only review; no model selection or edits."""
import importlib.util
import json
import statistics
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cohort_base',REPO/'scripts/evaluate-boundary-default-review-cohort.py')
base=importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
iv=base.iv


def main():
    inputs=base.read(base.PRIOR/'input.json')
    rows=[]
    for seed in base.SEEDS:
        result=base.read(base.PRIOR/'results'/f'{seed}.json')
        plans={p['id']:p['policies']['head_refined'] for p in result['plans']}
        details=[]
        for record in inputs['records']:
            plan=plans[record['id']]
            automatic=plan['events']
            ignored=record.get('ignoredIntervals',[])
            core_removed=iv.difference(iv.difference(record['productionEvents'],automatic),ignored)
            core_context=iv.export(core_removed,record,2)
            for flag in plan['proposals']:
                if flag['kind']!='additional_start':
                    continue
                siblings=sorted([e for e in automatic if e['parentId']==flag['parentId'] and e['componentId']==flag['componentId']],key=lambda e:(e['start'],e['end'],e['id']))
                i=next(i for i,e in enumerate(siblings) if e['id']==flag['candidateId'])
                assert i>0
                left,right=siblings[i-1],siblings[i]
                assert left['end']<=right['start']
                gap=iv.difference([(left['end'],right['start'])],ignored) if left['end']<right['start'] else ()
                padding=[]
                for pad in (0,1,2,3):
                    removed=iv.difference(iv.export(record['productionEvents'],record,pad),iv.export(automatic,record,pad))
                    context=iv.export(removed,record,2)
                    padding.append({'paddingSecondsEachSide':pad,'removedExportGapOverlapSeconds':iv.duration(iv.intersection(gap,removed)),
                        'exportReviewContextGapOverlapSeconds':iv.duration(iv.intersection(gap,context))})
                details.append({'recordingId':record['id'],'parentId':flag['parentId'],'flagId':flag['id'],
                    'gapStart':left['end'],'gapEnd':right['start'],'gapSeconds':iv.duration(gap),
                    'goldCoreInsideGapSeconds':iv.duration(iv.intersection(gap,record['rallies'])),
                    'coreQueueGapOverlapSeconds':iv.duration(iv.intersection(gap,core_removed)),
                    'coreQueueContextGapOverlapSeconds':iv.duration(iv.intersection(gap,core_context)),
                    'granularVetoMergesAcrossGap':bool(iv.intersection(gap,record['rallies'])),
                    'padding':padding})
        summary={'addedStartFlags':len(details),'zeroLengthGaps':sum(d['gapSeconds']==0 for d in details),
            'noRemovedCoreTriggerAtGap':sum(d['coreQueueGapOverlapSeconds']==0 for d in details),
            'granularVetoGapMerges':sum(d['granularVetoMergesAcrossGap'] for d in details),
            'padding':[{ 'paddingSecondsEachSide':pad,
                'gapsWithoutRemovedExportTrigger':sum(d['padding'][pad]['removedExportGapOverlapSeconds']==0 for d in details),
                'gapsOutsideExportQueueContext':sum(d['padding'][pad]['exportReviewContextGapOverlapSeconds']==0 for d in details)} for pad in (0,1,2,3)]}
        rows.append({'seed':seed,'summary':summary,'gaps':details})
    pooled={k:statistics.mean(r['summary'][k] for r in rows) for k in ('addedStartFlags','zeroLengthGaps','noRemovedCoreTriggerAtGap','granularVetoGapMerges')}
    pooled['padding']=[{'paddingSecondsEachSide':pad,**{key:statistics.mean(r['summary']['padding'][pad][key] for r in rows)
        for key in ('gapsWithoutRemovedExportTrigger','gapsOutsideExportQueueContext')}} for pad in (0,1,2,3)]
    output={'source':base.ref(Path(__file__)),'cohortResult':base.ref(base.OUTPUT/'results-v2.json'),
        'definition':'Each additional-start flag is paired with its preceding automatic event in the same original parent and valid component. Score the actual internal gap, not whether any part of its parent appeared in a review queue.',
        'targetExportPaddingEachSide':2,'reviewContextEachSide':2,'reviewContextJoinGapSeconds':3,
        'summarySeedMeans':pooled,'seeds':rows}
    path=base.OUTPUT/'split-gap-review-coverage-v1.json'
    with path.open('x') as handle:
        json.dump(output,handle,indent=2,allow_nan=False)
        handle.write('\n')
    print(json.dumps({'output':base.ref(path),'summarySeedMeans':pooled},indent=2))


if __name__=='__main__':
    main()
