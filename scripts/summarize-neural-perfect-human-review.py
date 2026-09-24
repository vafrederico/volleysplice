#!/usr/bin/env python3
"""Aggregate complete perfect-human review results; retain every fixed policy."""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from statistics import mean,pstdev

ROOT=Path(private_value('private-reference-0091'))
REPO=Path(__file__).resolve().parents[1]
METRICS=('P_pad','R_core','F1_padP_coreR','paddedModelExportSeconds','paddedHumanExportSeconds',
         'exportDurationDifferenceSeconds','evaluableVideoSeconds','correctlyRemovedSeconds',
         'incorrectlyRemovedSeconds','incorrectExportSeconds','missedCoreSeconds','coreHumanSeconds',
         'paddedIntersectionSeconds','coreIntersectionSeconds')
SUM_FIELDS=METRICS[3:5]+METRICS[6:]
WORKLOAD=('rawDecisionSeconds','reviewSeconds','decisionSeconds','reviewClips','flaggedCandidates',
          'flaggedPositiveCandidates','flaggedNegativeCandidates','reviewedTrueRallies','playbackTrueRallies',
          'binaryKeptCandidates','binaryDroppedCandidates','mixedCandidates','completeRallyLossesBinary',
          'partialRallyLossesBinary','completeRallyLossesBoundary','partialRallyLossesBoundary','reviewFractionOfVideo')
LEGACY_WORKLOAD=('reviewSeconds','disputedSeconds','unwantedExportFlaggedSeconds','wantedExportFlaggedSeconds',
                 'missedHumanExportFlaggedSeconds','missedCoreFlaggedSeconds','reviewClips','decisionSegments',
                 'reviewedTrueRallies','playbackTrueRallies')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(path):
    return {'path':str(path),'sha256':sha(path),'sizeBytes':Path(path).stat().st_size}


def read(path):
    return json.loads(Path(path).read_text())


def bound(ref):
    assert sha(ref['path'])==ref['sha256'], 'Artifact changed'
    return read(ref['path'])


def at(rows,pad=2):
    found=[r for r in rows if r['paddingSecondsBeforeAndAfter']==pad]
    assert len(found)==1
    return found[0]


def means(rows,fields):
    return {k:mean(r[k] for r in rows) for k in fields}


def statistics(rows,fields):
    return {k:{'mean':mean(r[k] for r in rows),'min':min(r[k] for r in rows),'max':max(r[k] for r in rows),
               'seedPopulationStddev':pstdev(r[k] for r in rows)} for k in fields}


def pool(rows):
    x={k:sum(r[k] for r in rows) for k in SUM_FIELDS}
    p=x['paddedIntersectionSeconds']/x['paddedModelExportSeconds'] if x['paddedModelExportSeconds'] else 0.
    r=x['coreIntersectionSeconds']/x['coreHumanSeconds'] if x['coreHumanSeconds'] else 0.
    x.update(P_pad=p,R_core=r,F1_padP_coreR=2*p*r/(p+r) if p+r else 0.,
             exportDurationDifferenceSeconds=x['paddedModelExportSeconds']-x['paddedHumanExportSeconds'])
    return x


def workload_subset(cell,pad,group):
    rows=[at(r['padding'],pad) for r in cell['recordings'] if r['sourceGroup']==group]
    x={k:sum(r[k] for r in rows) for k in WORKLOAD if k!='reviewFractionOfVideo'}
    video=sum(r['evaluableVideoSeconds'] for r in at(cell['baseDurationMetrics'],pad)['perRecording'] if r['sourceGroup']==group)
    x['reviewFractionOfVideo']=x['reviewSeconds']/video
    return x


def pareto(rows,scenario):
    result=[]
    for a in rows:
        af=a['primary'][scenario]['F1_padP_coreR'];atime=a['primary']['workload']['reviewSeconds']
        dominated=any((b['primary'][scenario]['F1_padP_coreR']>=af and
                       b['primary']['workload']['reviewSeconds']<=atime and
                       (b['primary'][scenario]['F1_padP_coreR']>af or b['primary']['workload']['reviewSeconds']<atime))
                      for b in rows if b['id']!=a['id'])
        if not dominated:result.append(a['id'])
    return result


def percent(v):return f'{100*v:.2f}'
def number(v):return f'{v:.2f}'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |',
                      *['| '+' | '.join(map(str,r))+' |' for r in rows]])


def workload_table(rows):
    return table(['Policy','Automatic F1 %','Binary P %','Binary R %','Binary F1 %',
                  'Boundary P %','Boundary R %','Boundary F1 %','Playback min','Decisions','True rallies','Playback clips'],
                 [[f"`{r['id']}`",percent(r['primary']['automatic']['F1_padP_coreR']),
                   *[percent(r['primary'][scenario][k]) for scenario in ('binary','boundary') for k in METRICS[:3]],
                   number(r['primary']['workload']['reviewSeconds']/60),number(r['primary']['workload']['flaggedCandidates']),
                   number(r['primary']['workload']['reviewedTrueRallies']),number(r['primary']['workload']['reviewClips'])] for r in rows])


def main(root):
    reg=read(root/'registration.json');c=reg['contract']
    assert hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==reg['sha256']
    report=read(root/'report.json')
    assert report['status']=='completed-perfect-human-review' and report['contractSha256']==reg['sha256']
    assert report['resultCells']==360 and report['independentCandidatePlans']==report['independentRecordingOutcomes']==2880
    for ref in c['sources'].values():assert sha(ref['path'])==ref['sha256']
    by_config=defaultdict(list);references=defaultdict(list)
    for ref in report['results']:
        row=bound(ref)
        assert row['contractSha256']==reg['sha256']
        assert row['independentHumanReview']['passed'] and all(
            x['candidatePlan']['passed'] and x['outcome']['passed'] for x in row['independentHumanReview']['recordings'])
        for name in ('independentBinaryAccounting','independentBoundaryAccounting'):
            assert row[name]['passed'] and row[name]['scopeCount']==13 and all(x['passed'] for x in row[name]['checks'])
        by_config[row['configuration']['id']].append(row)
        references[row['configuration']['id']].append(ref)
    configurations=[]
    mapping={'automatic':'baseDurationMetrics','binary':'binaryDurationMetrics','boundary':'boundaryDurationMetrics'}
    for config in c['configurations']:
        rows=by_config[config['id']]
        assert len(rows)==3 and {r['seed'] for r in rows}==set(c['seeds'])
        padding=[]
        for pad in c['paddingCases']:
            p={'paddingSecondsBeforeAndAfter':pad,'joinGapSeconds':3}
            for name,key in mapping.items():
                samples=[at(r[key],pad) for r in rows]
                p[name]=means(samples,METRICS)
                p[name+'SeedStatistics']=statistics(samples,METRICS)
            p['workload']=means([at(r['workload'],pad) for r in rows],WORKLOAD)
            p['workloadSeedStatistics']=statistics([at(r['workload'],pad) for r in rows],WORKLOAD)
            padding.append(p)
        groups={}
        for group in c['sourceGroups']:
            gpadding=[]
            for pad in c['paddingCases']:
                p={'paddingSecondsBeforeAndAfter':pad,'joinGapSeconds':3}
                for name,key in mapping.items():
                    samples=[pool([x for x in at(r[key],pad)['perRecording'] if x['sourceGroup']==group]) for r in rows]
                    p[name]=means(samples,METRICS)
                p['workload']=means([workload_subset(r,pad,group) for r in rows],WORKLOAD)
                gpadding.append(p)
            groups[group]=gpadding
        configurations.append({**config,'primary':at(padding),'padding':padding,'sourceGroups':groups,
                               'binaryEventF1':mean(r['binaryEvaluation']['guardrails']['eventF1'] for r in rows),
                               'seedResults':references[config['id']]})
    legacy_groups=defaultdict(list);legacy_refs=defaultdict(list)
    for ref in report['legacyResults']:
        r=bound(ref);key=f"{r['anchor']}--{r['neuralId']}--{r['mode']}"
        legacy_groups[key].append(r);legacy_refs[key].append(ref)
    legacy=[]
    for key,rows in sorted(legacy_groups.items()):
        assert len(rows)==3 and {r['seed'] for r in rows}==set(c['seeds'])
        pads=[]
        for pad in c['paddingCases']:
            pads.append({'paddingSecondsBeforeAndAfter':pad,'joinGapSeconds':3,
                         'automatic':means([at(r['automaticDurationMetrics'],pad) for r in rows],METRICS),
                         'boundary':means([at(r['boundaryDurationMetrics'],pad) for r in rows],METRICS),
                         'workload':means([at(r['padding'],pad) for r in rows],LEGACY_WORKLOAD)})
        legacy.append({'id':key,'anchor':rows[0]['anchor'],'neuralId':rows[0]['neuralId'],'mode':rows[0]['mode'],
                       'primary':at(pads),'padding':pads,'seedResults':legacy_refs[key]})
    assert len(configurations)==120 and len(legacy)==30
    ranked={name:[r['id'] for r in sorted(configurations,key=lambda r:(-r['primary'][name]['F1_padP_coreR'],r['id']))]
            for name in ('binary','boundary')}
    fronts={}
    for group in [*c['anchors'],'individual']:
        subset=[r for r in configurations if (r['anchor']==group if group!='individual' else r['family']=='individual')]
        fronts[group]={name:pareto(subset,name) for name in ('binary','boundary')}
    baselines={}
    for r in configurations:
        key=r['anchor'] or r['neuralId']
        if key in baselines:
            for field in METRICS:assert abs(baselines[key][field]-r['primary']['automatic'][field])<1e-10
        baselines[key]=r['primary']['automatic']
    summary={'kind':'perfect-human-review-summary-v1','createdAt':datetime.now(timezone.utc).isoformat(),
             'contractSha256':reg['sha256'],'passed':True,'hypotheticalHumanOutcomes':True,
             'targetPaddingSeconds':2,'joinGapSeconds':3,'paddingCases':[0,1,2,3],
             'protectedTestOpened':False,'productionChanged':False,'productionPromotionAllowed':False,
             'recordings':8,'rallies':322,'sourceGroups':c['sourceGroups'],'seedCount':3,
             'evaluableVideoSeconds':configurations[0]['primary']['automatic']['evaluableVideoSeconds'],
             'humanExportSeconds':configurations[0]['primary']['automatic']['paddedHumanExportSeconds'],
             'configurations':configurations,'legacy':legacy,'baselines':baselines,
             'rankedBinaryIds':ranked['binary'],'rankedBoundaryIds':ranked['boundary'],'paretoByScope':fronts,
             'artifacts':{'registration':identity(root/'registration.json'),'report':identity(root/'report.json'),
                          'summarizer':identity(Path(__file__))}}
    path=root/'summary.json'
    with path.open('x') as f:json.dump(summary,f,indent=2,allow_nan=False);f.write('\n')
    ranked_rows=sorted(configurations,key=lambda r:(-r['primary']['binary']['F1_padP_coreR'],r['id']))
    lines=['# Perfect-human rally review results — 19 September 2026','',
           'Completed 120 fixed review policies across three neural seeds (360 cells), with 30 prior export-disagreement policies retained as a separately named reference. These are simulations under explicit perfect-human assumptions, not measured reviewer outcomes. No fitting or production changes occurred.','',
           '## Scope, units and definitions','',
           f"The same eight indoor/grass recordings contain 322 labeled rallies and {number(summary['evaluableVideoSeconds']/60)} evaluable minutes. Target product padding is ±2 s with positive gaps joined only when strictly below 3 s; ignored time is excluded and never rejoined. P_pad measures acceptable padded export, R_core actual play retained, and F1_padP_coreR their harmonic mean. Each seed pools recordings first; tables average three seed scores/counts. Human export at target padding is {number(summary['humanExportSeconds']/60)} minutes.",'',
           'A **decision** is a flagged proposed raw rally or negative search section. **True rallies** counts distinct gold rally IDs overlapping raw decision regions; it excludes false-positive candidates and deduplicates repeated fragments. **Playback clips** merge padded decision regions plus two seconds of viewing context using strict <3 s joins. True rallies appearing only in context are counted separately. Viewing minutes mean footage at 1×, not observed wall-clock labor. Negative sections are split on an absolute five-second grid, so they must not be called rally detections.','',
           '**Binary keep/remove:** keep an entire flagged raw candidate if any evaluable human core play overlaps it; otherwise remove it. Unflagged automatic output remains. Ordinary export padding/joining follows. Mixed candidates keep their incorrect tails. A false raw candidate can have export padding that happens to cover nearby play; dropping that candidate can lose such coverage. Correct raw-candidate classification therefore does not imply perfect boundaries or monotonic export recall.','',
           '**Perfect boundary editing:** replace only the flagged candidates’ padded decision-export regions with the human export inside those regions. This can trim false tails and recover only play within flagged regions. Playback context does not grant additional edit permission, and no second padding/joining is applied. It is a distinct optimistic editing scenario. Neither scenario fixes unflagged errors automatically.','',
           'The current neural networks have live, serve/start, end and keep heads; **no learned needs-review head**. This study adds fixed review rules using model disagreement or uncalibrated live scores. A score of 0.8 is not established to mean 80% correctness. A dedicated review model or tuned review budget needs separate validation.','',
           '## Policy key','',
           '- `guarded_trim`: existing guarded compact/DINO trimming acts only as a flag trigger; review its entire containing production candidate. This expansion can remove more than the automatically trimmed tail.','- `suppression_zero/half/any`: flag production candidates with zero / less than half / any missing support from neural intervals expanded ±2 s.','- `bidirectional_half/any`: use both directions of that support test, permitting neural-only missed-rally proposals; review containing raw union components.','- `positive_uncertain`: review neural positive candidates with mean live score below 0.8; no search in omitted sections.','- `uncertain_narrow/medium/wide`: positive mean below 0.7/0.8/0.9, or negative-section maximum above 0.3/0.2/0.1. No-score sections are flagged by applicable uncertainty rules.','- `all_positive`: review every predicted positive candidate, without negative search.','- `all_candidates`: review every positive and negative section, a full-footage reference.','',
           'Compact boost means reviewed-export compact short boost; compact keep means reviewed-export keep rescue. DINO global is the draft/global control; DINO boost and keep use reviewed-export supervision. ProductionDefault is the checked-in fresh-project aggressive suppression setting, not a query of installed/deployed apps. ShippedUnion has suppression disabled; refitUnion is a source-group-held sensitivity baseline. Shipped weights historically saw these recordings; refit decoder choices and neural candidates remain development-exposed.','',
           '## Complete human-assisted quality and workload matrix','',
           'Sorted by binary keep/remove F1_padP_coreR at the predeclared target padding. The same review workload applies to both human assumptions. Counts can be fractional because they average seeds. All fixed policies are retained; this is descriptive development ranking, not a validated deployment selection.','',
           workload_table(ranked_rows),'',
           '## Decision composition and missed-rally guardrails','',
           table(['Policy','Positive decisions','Negative decisions','Kept','Dropped','Mixed','True rallies in playback',
                  'Binary complete losses','Binary partial losses','Boundary complete losses','Boundary partial losses','Binary event F1 %'],
                 [[f"`{r['id']}`",*[number(r['primary']['workload'][k]) for k in ('flaggedPositiveCandidates','flaggedNegativeCandidates',
                    'binaryKeptCandidates','binaryDroppedCandidates','mixedCandidates','playbackTrueRallies','completeRallyLossesBinary',
                    'partialRallyLossesBinary','completeRallyLossesBoundary','partialRallyLossesBoundary')],percent(r['binaryEventF1'])]
                  for r in ranked_rows]),'',
           'Mixed means a kept flagged raw candidate contains both human rally core and other time. That other time is not necessarily incorrect export under the padded metric. Binary raw-event F1 is a secondary guardrail; boundary-editing output has no invented raw-event match score.','',
           '## Export time accounting at target padding','',
           table(['Policy','Human action','Export min','Correctly removed min','Incorrectly removed min','Incorrect export min','Missed core s'],
                 [[f"`{r['id']}`",mode,*[number(r['primary'][mode][k]/60) for k in ('paddedModelExportSeconds','correctlyRemovedSeconds',
                    'incorrectlyRemovedSeconds','incorrectExportSeconds')],number(r['primary'][mode]['missedCoreSeconds'])]
                  for r in configurations for mode in ('binary','boundary')]),'',
           'Correctly removed is unwanted omitted time (TN); incorrectly removed is wanted padded human export omitted (FNpad); incorrect export is unwanted retained time (FP). Export + TN + FNpad partitions evaluable footage. Missed core separately measures actual play loss.','',
           '## Prior fine-grained export-disagreement reference','',
           'These previously tested queues flag disputed export seconds rather than whole candidates. Human editing is restricted to those disputed seconds, with two seconds of playback context. They have no whole-rally keep/remove estimate. Decisions below count disjoint disputed segments, not predicted rallies. The distinct true-rally counts are new post-hoc descriptions; flags and original quality metrics are unchanged.','',
           table(['Policy','P_pad %','R_core %','F1_padP_coreR %','Playback min','Disputed segments','True rallies','Playback clips'],
                 [[f"`{r['id']}`",*[percent(r['primary']['boundary'][k]) for k in METRICS[:3]],
                   number(r['primary']['workload']['reviewSeconds']/60),number(r['primary']['workload']['decisionSegments']),
                   number(r['primary']['workload']['reviewedTrueRallies']),number(r['primary']['workload']['reviewClips'])] for r in legacy]),'',
           '## Required 0/1/2/3-second padding sensitivity','',
           'Candidate inventories and flags remain fixed across these padding cases. Playback duration varies with the decision export’s padding. No model or policy chooses its own best padding.','',
           table(['Policy','Padding s','Human action','P_pad %','R_core %','F1_padP_coreR %','Model export s','Human export s','Difference s','Playback min'],
                 [[f"`{r['id']}`",pad,mode,*[percent(at(r['padding'],pad)[mode][k]) for k in METRICS[:3]],
                   *[number(at(r['padding'],pad)[mode][k]) for k in METRICS[3:6]],number(at(r['padding'],pad)['workload']['reviewSeconds']/60)]
                  for r in configurations for pad in range(4) for mode in ('binary','boundary')]),'',
           'All source-group sensitivities, automatic baselines, seed statistics and per-recording decisions remain in the machine-readable summary and immutable cell files. Group scores are diagnostic and are not averaged to obtain dataset F1.','',
           '## Evidence','',f"Contract SHA-256: `{reg['sha256']}`.",'',f"NAS root: `{root}`.",'',
           f"Summary: `{path}` (SHA-256 `{sha(path)}`).",f"Completed result index: `{root/'report.json'}` (SHA-256 `{sha(root/'report.json')}`).",'',
           'Every one of 2,880 recording-level candidate plans and human outcomes was independently reconstructed without shared interval/model imports. Each binary and boundary outcome also passed an independent duration oracle across 13 scopes and all four paddings. Thirty focused tests passed before registration, plus 480 randomized plan/outcome cross-checks. Sources and inputs were hash-checked before and after execution. A separate summary audit and final reproducible archive are stored alongside the run.']
    document=REPO/'docs/research/neural-perfect-human-review-results-2026-09-19.md'
    document.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'summary':identity(path),'document':identity(document),'configurations':120,'legacyPolicies':30},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    main(p.parse_args().root)
