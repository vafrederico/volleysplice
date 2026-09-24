#!/usr/bin/env python3
"""Post-hoc explanation of ordinary versus observed start recall; no rerun."""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO));sys.dont_write_bytecode=True
from analysis.private_ledger import private_value
from analysis import neural_production_combinations as iv
from analysis.neural_split_metrics import _prepared

ROOT=Path(private_value('private-reference-0079'))


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return {'path':str(p),'sha256':sha(p),'sizeBytes':Path(p).stat().st_size}
def require(ok,message):
    if not ok:raise ValueError(message)


def main():
    registration=read(ROOT/'registration.json');contract=registration['contract']
    data=read(contract['input']['path']);records={r['id']:r for r in data['records']}
    require(sha(contract['input']['path'])==contract['input']['sha256'],'Input changed')
    report=read(ROOT/'report.json');require(report['passed'],'Run incomplete')
    rows=[]
    for source in report['seeds']:
        require(sha(source['result']['path'])==source['result']['sha256'],'Result changed')
        cell=read(source['result']['path'])
        base=cell['baseline']['identityMetrics']['pooled']
        for mode in ('proposal_confirmation','full_parent'):
            arm=next(a for a in cell['reviewed'] if a['id']==f'head_refined--{mode}--evidence--budget-10')
            by_record={r['id']:r for r in arm['identityMetrics']['recordings']}
            unobserved=[]
            for edited in arm['perRecording']:
                record=records[edited['id']]
                _,_,eligible,_,excluded,_=_prepared(record)
                eligible_ids={g['index'] for g in eligible}
                parent_map={p['id']:p for p in record['productionEvents']}
                found=[]
                for event in edited['events']:
                    if event.get('startObserved',True) or not iv.difference([event],excluded):continue
                    gi=event.get('targetTruthIndex',event.get('humanGoldIndex'))
                    gold=record['rallies'][gi] if gi is not None else None
                    parent=parent_map[event.get('parentId',event['id'])]
                    parent_clipped=bool(gold and event['start']==parent['start'] and gold['start']<parent['start'])
                    ignored_clipped=bool(gold and gold['start']<event['start'] and any(abs(x['end']-event['start'])<1e-9 for x in record['ignoredIntervals']))
                    error=event['start']-gold['start'] if gold else None
                    found.append({'recordingId':record['id'],'eventId':event['id'],'parentId':parent['id'],
                                  'start':event['start'],'end':event['end'],'truthIndex':gi,
                                  'goldStart':gold['start'] if gold else None,'eligibleGold':gi in eligible_ids,
                                  'startErrorSeconds':error,'parentClipped':parent_clipped,'ignoredClipped':ignored_clipped,
                                  'withinOneSecondOfOwnGold':error is not None and abs(error)<=1,
                                  'startSource':event.get('startSource',event.get('startKind'))})
                require(len(found)==by_record[record['id']]['unobservedPredictionStarts'],'Unobserved count differs')
                unobserved.extend(found)
            identity=arm['identityMetrics']['pooled']
            ordinary=identity['startLocalization']['1'];observed=identity['observedStartLocalization']['1']
            row={'seed':cell['seed'],'mode':mode,'armId':arm['id'],'trueRallies':identity['trueRallies'],
                 'baselineOrdinaryMatched1s':base['startLocalization']['1']['matched'],
                 'baselineObservedMatched1s':base['observedStartLocalization']['1']['matched'],
                 'ordinaryMatched1s':ordinary['matched'],'observedMatched1s':observed['matched'],
                 'ordinaryRecall1s':ordinary['recall'],'observedRecall1s':observed['recall'],
                 'ordinaryMatchGain1s':ordinary['matched']-base['startLocalization']['1']['matched'],
                 'observedMatchChange1s':observed['matched']-base['observedStartLocalization']['1']['matched'],
                 'correctMatchesExcludedByObservationFlag1s':ordinary['matched']-observed['matched'],
                 'unobservedStartCount':len(unobserved),
                 'parentClippedStartCount':sum(x['parentClipped'] for x in unobserved),
                 'ignoredClippedStartCount':sum(x['ignoredClipped'] for x in unobserved),
                 'clippedWithin1sOfOwnGoldCount':sum(x['withinOneSecondOfOwnGold'] for x in unobserved),
                 'otherUnobservedCount':sum(not x['parentClipped'] and not x['ignoredClipped'] for x in unobserved),
                 'ordinaryMatchedQuarterSecond':identity['startLocalization']['0.25']['matched'],
                 'observedMatchedQuarterSecond':identity['observedStartLocalization']['0.25']['matched'],
                 'baselineMatchedQuarterSecond':base['startLocalization']['0.25']['matched'],
                 'unobservedStarts':unobserved}
            require(row['observedMatchChange1s']==row['ordinaryMatchGain1s']-row['correctMatchesExcludedByObservationFlag1s'],'Match-change decomposition failed')
            rows.append(row)
    numeric=[k for k,v in rows[0].items() if isinstance(v,(int,float)) and k!='seed']
    aggregates={m:{k:{'mean':mean(r[k] for r in rows if r['mode']==m),
                      'min':min(r[k] for r in rows if r['mode']==m),
                      'max':max(r[k] for r in rows if r['mode']==m)} for k in numeric}
                for m in ('proposal_confirmation','full_parent')}
    artifact={'kind':'post-hoc-observed-start-censoring-diagnosis-v1','createdAt':datetime.now(timezone.utc).isoformat(),
              'postHoc':True,'newConfigurations':0,'outcomeRerun':False,'contractSha256':registration['sha256'],
              'definitions':{'ordinary':'Timestamp within tolerance, including clipped/synthetic endpoints',
                             'observed':'Same matching after excluding endpoints explicitly marked unobserved',
                             'parentClipped':'Event start equals original production parent start, but linked gold serve-contact start is earlier',
                             'interpretation':'Numerically nearby permission boundaries are not observed serve contacts. Baseline model boundary guesses default observed; confirmed clipping removes that claim.',
                             'limit':'Parent permission is raw event coverage, not playback context or padded export. Those may contain earlier footage, but this registered editor cannot use it for endpoint corrections.'},
              'input':contract['input'],'resultReferences':[r['result'] for r in report['seeds']],
              'script':ref(Path(__file__)),'policySeedRows':rows,'aggregates':aggregates}
    output=ROOT/'observed-start-diagnosis.json'
    with output.open('x',encoding='utf-8') as f:json.dump(artifact,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'passed':True,'output':ref(output),'aggregates':aggregates},indent=2))


if __name__=='__main__':main()
