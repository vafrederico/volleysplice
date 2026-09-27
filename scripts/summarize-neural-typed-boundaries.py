#!/usr/bin/env python3
"""Aggregate every frozen typed-boundary arm; physical core losses stay visible."""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
ROOT=Path(private_value('private-reference-0079'))
DOC=REPO/'docs/research/neural-typed-boundary-results-2026-09-19.md'
UTILITY=REPO/'scripts/summarize-neural-split-advisor.py'
SPEC=importlib.util.spec_from_file_location('qualified_split_summary_utilities',UTILITY)
u=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(u)
WORKLOAD=('reviewSeconds','budgetSeconds','unusedBudgetSeconds','reviewJobs','jobsAvailable','reviewClips',
          'editRegions','boundaryFlagsReviewed','candidatesReviewed','reviewedTrueRallies','playbackTrueRallies',
          'eventTimelineRemovedSeconds','eventTimelineAddedSeconds')
ACTIONS=('reviewedCandidates','acceptedCount','rejectedCount','correctedTypeCount','editedParents')
COVERAGE=('coreHumanSeconds','baselineRawCoreCoveredSeconds','resultRawCoreCoveredSeconds',
          'baselineRawCoreRecall','resultRawCoreRecall','rawCoreSecondsLostFromBaseline',
          'rawCoreSecondsAddedToBaseline','rawSelectedSecondsLostFromBaseline','rawSelectedSecondsAddedToBaseline',
          'baselineCompleteMisses','resultCompleteMisses','additionalCompleteMisses','retainedCompleteMisses','recoveredCompleteMisses')


def scope_metric(evaluation,name,group=None,recording=None):
    metric=evaluation[name]
    if recording is not None:
        found=[r for r in metric['recordings'] if r['id']==recording]
        u.require(len(found)==1,'Missing/repeated recording metric')
        return u.numeric(found[0])
    return u.numeric(metric['sourceGroups'][group] if group is not None else metric['pooled'])


def summarize(evaluations,group=None,recording=None):
    out={'padding':[]}
    for pad in (0,1,2,3):
        samples=[]
        for e in evaluations:
            raw=[r for r in e['durationMetrics'] if r['paddingSecondsBeforeAndAfter']==pad]
            u.require(len(raw)==1 and raw[0]['joinGapSeconds']==3,'Padding contract differs')
            if group is None and recording is None:
                samples.append({k:raw[0][k] for k in u.DURATION_FIELDS})
            else:
                samples.append(u.pool_duration([r for r in raw[0]['perRecording']
                  if (group is None or r['sourceGroup']==group) and (recording is None or r['id']==recording)]))
        statistics=u.tree_stats(samples)
        out['padding'].append({'paddingSecondsBeforeAndAfter':pad,'joinGapSeconds':3,
                               'metrics':u.means(statistics),'seedStatistics':statistics})
    out['primary']=out['padding'][2]['metrics']
    for source,name in (('identityMetrics','identity'),('typedMetrics','typed'),
                        ('rawCoverageMetrics','legacyIdentityCoverage')):
        statistics=u.tree_stats([scope_metric(e,source,group,recording) for e in evaluations])
        out[name]=u.means(statistics);out[name+'SeedStatistics']=statistics
    out['coverage']={k:out['typed'][k] for k in COVERAGE}
    out['coverageSeedStatistics']={k:out['typedSeedStatistics'][k] for k in COVERAGE}
    work_samples=[];action_samples=[]
    for e in evaluations:
        selected=[r for r in e.get('perRecording',[]) if (group is None or r['sourceGroup']==group)
                  and (recording is None or r['id']==recording)]
        if 'workload' not in e:
            work_samples.append({k:0 for k in WORKLOAD});action_samples.append({k:None for k in ACTIONS})
        else:
            u.require(bool(selected),'Missing workload scope')
            work_samples.append({k:sum(r['workload'][k] for r in selected) for k in WORKLOAD})
            action_samples.append({k:sum(r[k] for r in selected) if all(k in r for r in selected) else None for k in ACTIONS})
    for name,samples in (('workload',work_samples),('humanActions',action_samples)):
        statistics=u.tree_stats(samples);out[name]=u.means(statistics);out[name+'SeedStatistics']=statistics
    return out


def expected_reviews(contract):
    return {f"{p}--{m}--{r}--budget-{round(100*b):02d}":
            {'policy':p,'humanMode':m,'ranker':r,'budgetFraction':b}
            for p in contract['policies'] for m in contract['humanModes']
            for r in contract['rankers'] for b in contract['budgetFractions']}


def check_evaluation(e):
    for name in ('fixedExportAudit','identityAudit','rawCoverageAudit','typedMetricAudit'):
        u.require(e[name]['passed'],'Failed audit: '+name)
    u.require([r['paddingSecondsBeforeAndAfter'] for r in e['durationMetrics']]==[0,1,2,3],'Incomplete padding scope')
    u.require(all(r['joinGapSeconds']==3 for r in e['durationMetrics']),'Join threshold differs')


def aggregate(contract,cells):
    seeds=contract['seeds'];u.require(len(set(seeds))==len(seeds)==3,'Exactly three distinct seeds required')
    u.require(sorted(c['seed'] for c in cells)==sorted(seeds),'Missing/duplicate seed')
    cells=sorted(cells,key=lambda c:seeds.index(c['seed']))
    expected=expected_reviews(contract);expected_auto={'automatic--'+p for p in contract['policies']}
    u.require(len(expected_auto)==4 and len(expected)==64,'Expected four automatic and 64 reviewed arms')
    first=cells[0]['baseline'];check_evaluation(first)
    automatic={};reviewed={}
    for cell in cells:
        u.require(cell['baseline']==first,'Production baseline differs across seeds')
        for name,wanted,storage in (('automatic',expected_auto,automatic),('reviewed',set(expected),reviewed)):
            u.require(len(cell[name])==len(wanted) and {r['id'] for r in cell[name]}==wanted,'Wrong '+name+' scope')
            for row in cell[name]:
                check_evaluation(row)
                u.require(row['durationMetrics']==first['durationMetrics'],'Export changed from production')
                if name=='automatic':
                    u.require(row['policy']==row['id'].removeprefix('automatic--'),'Automatic metadata differs')
                else:
                    u.require(all(row[k]==v for k,v in expected[row['id']].items()),'Review metadata differs')
                    for key in WORKLOAD:
                        u.require(abs(row['workload'][key]-sum(r['workload'][key] for r in row['perRecording']))<=1e-7,'Workload total differs')
                    if row['humanMode']=='full_parent':
                        c=row['typedMetrics']['pooled']
                        u.require(c['rawCoreSecondsLostFromBaseline']<=1e-8 and c['additionalCompleteMisses']==0,'Full-parent review lost core/rally')
                storage.setdefault(row['id'],[]).append(row)
    groups=sorted(first['identityMetrics']['sourceGroups'])
    recording_ids=sorted(r['id'] for r in first['identityMetrics']['recordings'])
    def arm(identity,rows,metadata):
        return {'id':identity,**metadata,**summarize(rows),
                'sourceGroups':{g:summarize(rows,group=g) for g in groups},
                'recordings':{r:summarize(rows,recording=r) for r in recording_ids}}
    return {'kind':'compact-typed-boundaries-summary-v1','seeds':seeds,'sourceGroups':groups,'recordings':recording_ids,
            'automaticOutcomes':sum(len(c['automatic']) for c in cells),'reviewedOutcomes':sum(len(c['reviewed']) for c in cells),
            'automaticArms':len(automatic),'reviewedArms':len(reviewed),'allArmsIncludingBaseline':1+len(automatic)+len(reviewed),
            'primaryRanking':{'metric':'F1_padP_coreR','targetPaddingSeconds':2,'result':'all arms tie exactly with production'},
            'verification':{'fixedExportAllFourPads':True,'allRegisteredArms':True,'fullParentCorePreserved':True,
                             'automaticAndProposalOnlyCoreLossAllowedAndReported':True},
            'typedMetricInterpretation':'Original model event candidates, all candidates for automatic or selected-parent candidates for reviewed; NOT corrected human output',
            'physicalCoverageInterpretation':'All gold core minus ignored time, including valid portions of ignored-touched gold; rates pool seconds within seed',
            'baseline':arm('production',[c['baseline'] for c in cells],{'kind':'baseline'}),
            'automatic':[arm(k,automatic[k],{'kind':'automatic','policy':automatic[k][0]['policy']}) for k in sorted(automatic)],
            'reviewed':[arm(k,reviewed[k],{'kind':'reviewed',**expected[k]}) for k in sorted(reviewed)]}


def quality_row(label,a,review=False):
    i=a['identitySeedStatistics'];c=a['coverageSeedStatistics'];w=a['workloadSeedStatistics'];f=u.ranged
    row=[label,*[f(i[k],percent=True) for k in ('eventPrecision','eventRecall','eventF1')],
         f(i['observedStartLocalization']['1']['recall'],percent=True),f(i['observedEndLocalization']['1']['recall'],percent=True),
         f(c['resultRawCoreRecall'],percent=True),f(c['rawCoreSecondsLostFromBaseline']),
         f(i['completeMisses']),f(c['additionalCompleteMisses'])]
    if review:row += [f(w['reviewSeconds'],seconds_to_minutes=True),f(w['reviewJobs']),f(w['reviewedTrueRallies'])]
    return row


QUALITY=['Arm','Rally P %','Rally R %','Rally F1 %','Observed start R @1s %','Observed end R @1s %',
         'Raw core R %','Raw core lost s','Complete misses','Additional misses']


def markdown(s):
    base=s['baseline'];f=u.fmt;rangef=u.ranged
    lines=['# Compact typed rally boundaries: results','',
        '<!-- EXECUTIVE FINDINGS START -->','<!-- EXECUTIVE FINDINGS END -->','',
        'This development experiment distinguishes initial rally-start correction, additional rallies, and separate end boundaries. Original production export footage remains fixed. Every arm ties on `F1_padP_coreR`; event-timeline quality and losses are reported separately. No production change, new model fit, or protected test.', '',
        f"Scope: {len(s['recordings'])} recordings, {len(s['sourceGroups'])} source groups, {base['identity']['trueRallies']:.0f} eligible gold rallies; {s['automaticOutcomes']} automatic outcomes and {s['reviewedOutcomes']} reviewed outcomes across three seeds. There are {s['allArmsIncludingBaseline']} aggregate arms including production. Values are seed means; brackets give seed minimum and maximum, not confidence intervals. Counts/rates pool recordings within each seed before averaging.", '',
        'Raw core recall measures the separate event timeline before export padding. A fixed export can preserve video while an event candidate loses a rally or its endpoints. Review minutes include whole-parent playback context at 1x and are not measured human labor.', '',
        '## Fixed export accounting', '',
        'This table applies exactly to every arm. Padding is symmetric; positive gaps strictly below three seconds join. Ignored intervals are removed without rejoining. Correct removed time is omitted footage outside wanted human export; incorrect removed time is wanted human export omitted. Duration columns are minutes.', '',
        u.table(['Pad each side s','P_pad %','R_core %','F1_padP_coreR %','Model export min','Human export min','Difference min','Correct removed min','Incorrect removed min','Incorrect export min'],
                [[p['paddingSecondsBeforeAndAfter'],*[f(p['metrics'][k],percent=True) for k in ('P_pad','R_core','F1_padP_coreR')],
                  *[f(p['metrics'][k],seconds_to_minutes=True) for k in ('paddedModelExportSeconds','paddedHumanExportSeconds','exportDurationDifferenceSeconds','correctlyRemovedSeconds','incorrectlyRemovedSeconds','incorrectExportSeconds')]] for p in base['padding']]),'',
        '## Automatic event timelines','',u.table(QUALITY,[quality_row(a['id'],a) for a in [base,*s['automatic']]]),'',
        '## Original candidate type and paired-boundary quality','',
        'These metrics score model candidates before any human correction. Targets are the first and subsequent materially overlapping gold rallies within each original production parent and valid component. Inaccessible gold endpoints remain in denominators. An observed proposed start must be within tolerance and the proposed interval must materially overlap the same gold rally. Wrong-type counts use a separate untyped matching. Joint start/end quality uses the same typed start-matched identity and requires an observed end; no unrelated end may be substituted.', '',
        u.table(['Policy','Type','Tolerance s','True targets','Candidates with observed start','Matched','Typed P %','Typed R %','Typed F1 %'],
                [[a['policy'],kind,tol,*[rangef(a['typedSeedStatistics']['byType'][kind][tol][k]) for k in ('true','predicted','matched')],
                  *[rangef(a['typedSeedStatistics']['byType'][kind][tol][k],percent=True) for k in ('precision','recall','f1')]]
                 for a in s['automatic'] for kind in ('initial_start','additional_start') for tol in ('0.5','1','2')]),'',
        u.table(['Policy','Wrong type @1s','Initial proposed for additional','Additional proposed for initial','Joint start/end F1 @1s %','Unobserved starts','Unobserved ends'],
                [[a['policy'],*[rangef(a['typedSeedStatistics']['typeDiagnostics']['1'][k]) for k in ('wrongType','initialProposedForAdditional','additionalProposedForInitial')],
                  rangef(a['typedSeedStatistics']['samePairBoundaries']['1']['f1'],percent=True),rangef(a['typedSeedStatistics']['unobservedCandidateStarts']),rangef(a['typedSeedStatistics']['unobservedCandidateEnds'])] for a in s['automatic']]),'',
        f"The production parent scope contains {base['typed']['targetCount']:.0f} parent-specific candidate targets, with {base['typed']['inaccessibleTargetStarts']:.0f} inaccessible starts and {base['typed']['inaccessibleTargetEnds']:.0f} inaccessible ends. Target counts can differ from unique gold rallies because targets are parent-specific and require material overlap.",'']
    for mode,title,explanation in [
        ('proposal_confirmation','Restricted proposal confirmation','The human reviews every model event candidate in selected parents, accepts one-to-one observed starts within one second plus material interval overlap, and may correct a wrong type. Starts-only arms keep inherited/partitioned ends; paired arms correct accepted end boundaries inside parent permission. Unproposed rallies cannot be invented. Paired corrections may remove an unproposed rally from the event timeline, so raw-core loss and additional misses are reported rather than assumed zero.'),
        ('full_parent','Full-parent human correction ceiling','The human replaces selected parent portions with every gold-core intersection, including rallies absent from model proposals, and corrects both endpoints wherever visible. This is an optimistic full annotation operation, not proposal-only confirmation. It preserves all production-covered core within selected parents but cannot add video outside production. Unselected parents remain original.')]:
        arms=[a for a in s['reviewed'] if a['humanMode']==mode]
        lines += ['## '+title,'',explanation,'',u.table(QUALITY+['Review min','Parent jobs','Real rallies reviewed'],[quality_row(a['id'],a,True) for a in arms]),'']
    lines += ['## Review workload and candidate diagnostics','',
        'Typed metrics in these reviewed rows still describe original selected candidates, not gold-corrected output. They expose what the review queue selected. Full-parent output may contain additional gold rallies beyond these candidates. Null acceptance counts mean that the full-parent mode does not classify individual candidate acceptance.', '',
        u.table(['Arm','Flags reviewed','Candidates reviewed','Accepted candidates','Rejected candidates','Corrected types','Original wrong types @1s','Original joint F1 @1s %','Unused budget min'],
                [[a['id'],rangef(a['workloadSeedStatistics']['boundaryFlagsReviewed']),rangef(a['workloadSeedStatistics']['candidatesReviewed']),
                  *[rangef(a['humanActionsSeedStatistics'][k]) for k in ('acceptedCount','rejectedCount','correctedTypeCount')],
                  rangef(a['typedSeedStatistics']['typeDiagnostics']['1']['wrongType']),rangef(a['typedSeedStatistics']['samePairBoundaries']['1']['f1'],percent=True),
                  rangef(a['workloadSeedStatistics']['unusedBudgetSeconds'],seconds_to_minutes=True)] for a in s['reviewed']]),'',
        '## Source-group sensitivity','',
        'Below: production, all automatic arms, and 10% evidence-order review arms. Every arm and every numeric metric is retained for all source groups and individual recordings in summary JSON, including all padding and tolerance cases. These slices are not independent repetitions of the dataset.','']
    for group in s['sourceGroups']:
        arms=[base,*s['automatic'],*[a for a in s['reviewed'] if a['ranker']=='evidence' and a['budgetFraction']==.1]]
        lines += ['### '+group,'',u.table(QUALITY+['Review min','Parent jobs','Real rallies reviewed'],
                   [quality_row(a['id'],a['sourceGroups'][group],True) for a in arms]),'']
    lines += ['## Limits and reproducibility','',
        'Observed start/end localization is a rally-boundary proxy, not serve-side, point-winner or reconstructed-score accuracy. Both human modes assume perfect permitted actions. Results are development estimates; previous model and decoder choices already used this footage. Fresh footage and measured human review are still needed before product promotion.', '',
        'The registered result files bind all source/input hashes and independent plan, queue, human, workload, duration, identity and typed-metric audits. The summary verifies the full 12 automatic/192 reviewed outcome matrix, exact four-padding export invariance, workload totals, and zero full-parent core loss. It intentionally permits and exposes automatic/proposal-only event-timeline losses. `summary.json` contains 69 aggregate arms over pooled, source-group and recording scopes, with numeric mean/min/max and available-seed counts.', '']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--doc',type=Path,default=DOC)
    args=p.parse_args();reg=u.read(args.root/'registration.json');c=reg['contract']
    u.require(u.canonical(c)==reg['sha256'],'Registration changed')
    for r in [*c['sources'].values(),*c['sourceCopies'].values(),c['input'],c['probabilities'],c['protocol'],c['qualification']]:
        u.require(u.sha(r['path'])==r['sha256'],'Frozen dependency changed')
    report=u.read(args.root/'report.json');u.require(report['passed'] and report['contractSha256']==reg['sha256'],'Run incomplete')
    cells=[u.bound(x['result']) for x in report['seeds']]
    u.require(all(x['contractSha256']==reg['sha256'] for x in cells),'Wrong cell contract')
    result=aggregate(c,cells)
    u.require(result['automaticOutcomes']==c['automaticOutcomes'] and result['reviewedOutcomes']==c['reviewedOutcomes'],'Outcome total mismatch')
    result.update(createdAt=datetime.now(timezone.utc).isoformat(),contractSha256=reg['sha256'],
                  registration=u.reference(args.root/'registration.json'),report=u.reference(args.root/'report.json'),
                  inputs=[x['result'] for x in report['seeds']],summarizer=u.reference(Path(__file__)),utilitySource=u.reference(UTILITY))
    with (args.root/'summary.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    args.doc.parent.mkdir(parents=True,exist_ok=True)
    with args.doc.open('x',encoding='utf-8') as stream:stream.write(markdown(result))
    print(json.dumps({'passed':True,'summary':u.reference(args.root/'summary.json'),'document':u.reference(args.doc),
                      'automaticArms':result['automaticArms'],'reviewedArms':result['reviewedArms']}))


if __name__=='__main__':main()
