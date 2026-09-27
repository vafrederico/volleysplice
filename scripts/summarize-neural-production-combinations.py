#!/usr/bin/env python3
"""Summarize a completed, independently audited fixed combination study.

This post-processing script performs no interval editing, tuning or selection.
All 107 fixed automatic configurations and 30 review policies remain visible.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(private_value('private-reference-0083'))
REPO = Path(__file__).resolve().parents[1]
METRICS = ('P_pad', 'R_core', 'F1_padP_coreR', 'paddedModelExportSeconds',
           'paddedHumanExportSeconds', 'exportDurationDifferenceSeconds',
           'evaluableVideoSeconds', 'correctlyRemovedSeconds', 'incorrectlyRemovedSeconds',
           'incorrectExportSeconds', 'missedCoreSeconds', 'coreHumanSeconds',
           'paddedIntersectionSeconds', 'coreIntersectionSeconds')
SUM_FIELDS = METRICS[3:5] + METRICS[6:]
NAMES = {
    'productionDefault': 'Production default (aggressive suppression)',
    'productionBalanced': 'Production balanced suppression',
    'productionConservative': 'Production conservative suppression',
    'shippedPrevious': 'Shipped previous head', 'shippedV2': 'Shipped v2 head',
    'shippedUnion': 'Production unsuppressed union',
    'refitPrevious': 'Source-held refit previous head', 'refitV2': 'Source-held refit v2 head',
    'refitUnion': 'Source-held refit union',
    'compact_boost': 'Compact TCN, short boost', 'compact_keep': 'Compact TCN, keep rescue',
    'compact_baseline': 'Compact TCN, baseline',
    'dino_global': 'DINO+TCN, global control', 'dino_boost': 'DINO+TCN, short boost',
    'dino_keep': 'DINO+TCN, keep rescue', 'dino_baseline': 'DINO+TCN, baseline',
    'best_f1_pair': 'Compact boost + DINO global', 'recovery_pair': 'Compact keep + DINO boost',
    'union': 'union', 'intersection': 'strict intersection',
    'tolerant_intersection': '2 s tolerant intersection',
    'guarded_trim': 'protected-component trimming',
    'guarded_component_rejection': 'protected-component rejection',
    'nn_union': 'NN union', 'nn_intersection': 'NN intersection',
    'three_union': 'three-model union', 'majority': 'two-of-three majority',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(path):
    return {'path': str(path), 'sha256': sha(path), 'sizeBytes': Path(path).stat().st_size}


def read(path):
    return json.loads(Path(path).read_text())


def bound(ref):
    assert sha(ref['path']) == ref['sha256'], 'Bound artifact changed'
    return read(ref['path'])


def at(rows, pad=2):
    found = [r for r in rows if r['paddingSecondsBeforeAndAfter'] == pad]
    assert len(found) == 1
    return found[0]


def stats(rows, fields):
    return {key: {'mean': mean(r[key] for r in rows), 'min': min(r[key] for r in rows),
                  'max': max(r[key] for r in rows), 'seedPopulationStddev': pstdev(r[key] for r in rows)}
            for key in fields}


def means(rows, fields):
    return {k: mean(r[k] for r in rows) for k in fields}


def pooled(rows):
    total = {k: sum(r[k] for r in rows) for k in SUM_FIELDS}
    p = total['paddedIntersectionSeconds']/total['paddedModelExportSeconds'] if total['paddedModelExportSeconds'] else 0.
    r = total['coreIntersectionSeconds']/total['coreHumanSeconds'] if total['coreHumanSeconds'] else 0.
    total.update(P_pad=p, R_core=r, F1_padP_coreR=2*p*r/(p+r) if p+r else 0.,
                 exportDurationDifferenceSeconds=total['paddedModelExportSeconds']-total['paddedHumanExportSeconds'])
    return total


def label(key):
    return ' / '.join(NAMES.get(part, part) for part in key.split('--'))


def number(value):
    return f'{value:.2f}'


def percent(value):
    return f'{100*value:.2f}'


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join('---' for _ in headers)+' |',
                      *['| '+' | '.join(map(str, row))+' |' for row in rows]])


def duration_table(rows):
    return table(['Configuration', 'P_pad %', 'R_core %', 'F1_padP_coreR %', 'Export min',
                  'Correctly removed min', 'Incorrectly removed min', 'Incorrect export min', 'Missed core s'],
                 [[f"`{r['id']}`", *[percent(r['primary'][k]) for k in METRICS[:3]],
                   *[number(r['primary'][k]/60) for k in ('paddedModelExportSeconds', 'correctlyRemovedSeconds',
                       'incorrectlyRemovedSeconds', 'incorrectExportSeconds')], number(r['primary']['missedCoreSeconds'])]
                  for r in rows])


def main(output, document):
    reg = read(output/'registration.json')
    c = reg['contract']
    assert hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest() == reg['sha256']
    report = read(output/'report.json')
    audit = read(output/'recipe-audit-v1.json')
    assert report['status'] == 'completed-fixed-combinations'
    assert audit['passed'] and audit['contractSha256'] == report['contractSha256'] == reg['sha256']
    assert audit['report'] == {'path': str(output/'report.json'), 'sha256': sha(output/'report.json')}
    for ref in c['code'].values():
        assert sha(ref['path']) == ref['sha256']
    neural = bound(c['neuralInput'])
    production = bound(c['productionInput'])
    groups = sorted({r['sourceGroup'] for r in neural['records']})
    all_auto = [bound(ref) for ref in report['automaticResults']]
    all_review = [bound(ref) for ref in report['reviewResults']]
    grouped = defaultdict(list)
    for r in all_auto:
        assert r['contractSha256'] == reg['sha256']
        assert r['independentAccounting']['scopeCount'] == 13
        assert all(x['passed'] for x in r['independentAccounting']['checks'])
        grouped[r['configuration']['id']].append(r)
    automatic = []
    for config in c['configurations']:
        rows = grouped[config['id']]
        expected = {None} if config['family'] == 'production' else set(c['seeds'])
        assert {r['seed'] for r in rows} == expected and len(rows) == len(expected)
        pads = []
        for pad in (0,1,2,3):
            samples = [at(r['durationMetrics'], pad) for r in rows]
            pads.append({'paddingSecondsBeforeAndAfter': pad, 'joinGapSeconds': 3,
                         **means(samples, METRICS), 'seedStatistics': stats(samples, METRICS)})
        group_rows = {}
        for group in groups:
            gpads = []
            for pad in (0,1,2,3):
                samples = [pooled([v for v in at(r['durationMetrics'],pad)['perRecording'] if v['sourceGroup']==group]) for r in rows]
                gpads.append({'paddingSecondsBeforeAndAfter': pad, 'joinGapSeconds':3,
                              **means(samples,METRICS), 'seedStatistics': stats(samples,METRICS)})
            group_rows[group] = gpads
        guards = means([r['guardrails'] for r in rows], rows[0]['guardrails'])
        row = {**config, 'label': label(config['id']), 'seedCount':len(rows),
               'primary':at(pads), 'padding':pads, 'sourceGroups':group_rows, 'guardrailsMean':guards,
               'seedGuardrails': [{ 'seed':r['seed'], **r['guardrails'],
                                    'primary':{k:at(r['durationMetrics'])[k] for k in METRICS},
                                    'pairedAgainst':r.get('pairedAgainst'),
                                    'pairedGuardrails':r.get('pairedGuardrails')} for r in rows]}
        if config['family'] != 'production':
            paired = [r['pairedGuardrails'] for r in rows]
            row['pairedAgainst'] = rows[0]['pairedAgainst']
            row['pairedMean'] = means(paired, ('newCompleteLosses','newPartialLosses','worsenedRallies',
                        'additionalLostCoreSeconds','recoveredCoreSeconds','newShortCompleteLosses'))
            row['pairedMaximum'] = {k:max(r[k] for r in paired) for k in ('newCompleteLosses','worsenedRallies')}
            row['guardrailDeltaMean'] = means([r['guardrailDelta'] for r in paired], paired[0]['guardrailDelta'])
        automatic.append(row)
    by_id = {r['id']:r for r in automatic}
    for row in automatic:
        if 'pairedAgainst' not in row:
            continue
        base = by_id[row['pairedAgainst']]['primary']
        delta = {k:row['primary'][k]-base[k] for k in METRICS}
        row['primaryDeltaAgainstAnchor'] = delta
        screen = c['conservativeScreen']
        conditions = {
            'meanF1Gain': delta['F1_padP_coreR'] >= screen['meanF1MinimumGain']-1e-12,
            'meanRecall': delta['R_core'] >= screen['meanRecallMinimumDelta']-1e-12,
            'meanLongRecall': row['guardrailDeltaMean']['longR_core'] >= screen['meanLongRecallMinimumDelta']-1e-12,
            'meanEventF1': row['guardrailDeltaMean']['eventF1'] >= screen['meanEventF1MinimumDelta']-1e-12,
            'noNewCompleteLossesAnySeed': row['pairedMaximum']['newCompleteLosses']==0,
            'noWorsenedRalliesAnySeed': row['pairedMaximum']['worsenedRallies']==0,
        }
        row['conservativeScreen']={'passed':all(conditions.values()),'conditions':conditions}
    ranked = sorted(automatic,key=lambda r:(-r['primary']['F1_padP_coreR'],r['id']))
    for i,r in enumerate(ranked,1):
        r['developmentRank']=i
    review_groups=defaultdict(list)
    for row in all_review:
        assert row['contractSha256']==reg['sha256']
        assert row['independentAccounting']['scopeCount']==13
        assert all(x['passed'] for x in row['independentAccounting']['checks'])
        review_groups[(row['anchor'],row['neuralId'],row['mode'])].append(row)
    reviews=[]
    queue_fields=('reviewSeconds','disputedSeconds','unwantedExportFlaggedSeconds','wantedExportFlaggedSeconds',
                  'missedHumanExportFlaggedSeconds','missedCoreFlaggedSeconds','reviewClips')
    for (anchor,nn,mode),rows in sorted(review_groups.items()):
        assert len(rows)==3 and {r['seed'] for r in rows}==set(c['seeds'])
        pads=[]
        for pad in (0,1,2,3):
            q = means([at(r['queue'],pad) for r in rows],queue_fields)
            baseline=at(by_id[anchor]['padding'],pad)
            q['unwantedExportCaptureFraction']=q['unwantedExportFlaggedSeconds']/baseline['incorrectExportSeconds'] if baseline['incorrectExportSeconds'] else 0.
            q['missedCoreCaptureFraction']=q['missedCoreFlaggedSeconds']/baseline['missedCoreSeconds'] if baseline['missedCoreSeconds'] else 0.
            pads.append({'paddingSecondsBeforeAndAfter':pad,'joinGapSeconds':3,**q,
                         'automatic':{k:baseline[k] for k in METRICS},
                         'oracle':means([at(r['oracleDurationMetrics'],pad) for r in rows],METRICS)})
        reviews.append({'id':f'{anchor}--{nn}--{mode}','anchor':anchor,'neuralId':nn,'mode':mode,
                        'primary':at(pads),'padding':pads,'seedCount':3})
    assert len(automatic)==107 and len(reviews)==30
    summary = {'kind':'audited-production-combinations-summary-v1','createdAt':datetime.now(timezone.utc).isoformat(),
               'passed':True,'contractSha256':reg['sha256'], 'primaryMetric':'F1_padP_coreR',
               'targetPaddingSeconds':2,'joinGapSeconds':3,'paddingCases':[0,1,2,3],
               'protectedTestOpened':False,'productionChanged':False,'productionPromotionAllowed':False,
               'seedAggregation':'Pool duration counts over recordings within each seed; average the three seed metrics and summed seconds. Production is fixed once.',
               'population':{'recordings':8,'sourceGroups':groups,'rallies':322,
                             'evaluableVideoSeconds':automatic[0]['primary']['evaluableVideoSeconds'],
                             'humanExportSeconds':automatic[0]['primary']['paddedHumanExportSeconds'],
                             'humanCoreSeconds':automatic[0]['primary']['coreHumanSeconds']},
               'artifacts':{'registration':identity(output/'registration.json'),'report':identity(output/'report.json'),
                            'recipeAudit':identity(output/'recipe-audit-v1.json'),'summarizer':identity(Path(__file__))},
               'automatic':automatic,'rankedAutomaticIds':[r['id'] for r in ranked], 'reviews':reviews,
               'conservativeScreenPassingIds':[r['id'] for r in automatic if r.get('conservativeScreen',{}).get('passed')],
               'actualAppFidelity':report['actualAppFidelity']}
    target=output/'summary.json'
    with target.open('x') as f:
        json.dump(summary,f,indent=2,allow_nan=False); f.write('\n')
    lines = ['# Production and neural combination results — 19 September 2026', '',
             'Completed 107 fixed automatic configurations (303 seed/configuration cells) and 30 review policies (90 cells), with independent interval construction and duration-accounting checks. No new training, production change, or protected-test evaluation occurred.', '',
             '## Reading the tables', '',
             '**Target: symmetric ±2 seconds; positive gaps are joined only when strictly below 3 seconds.** All tables use the same 8 indoor/grass recordings, 4 source groups and 322 labeled rallies. Ignored time is removed from all interval unions and never rejoined. Neural values average three seed results; each seed pools duration counts across recordings before computing ratios. Seconds are totals over the eight recordings, averaged across seeds, not three concatenated copies of the dataset.', '',
             f"Evaluable footage: **{number(summary['population']['evaluableVideoSeconds']/60)} min**. Human export at ±2 s: **{number(summary['population']['humanExportSeconds']/60)} min**. Actual rally core: **{number(summary['population']['humanCoreSeconds']/60)} min**.", '',
             '- **P_pad**: share of exported time within the equally padded human export.',
             '- **R_core**: share of actual rally core retained in the model export.',
             '- **F1_padP_coreR**: harmonic mean of those two values; the fixed primary ranking metric.',
             '- **Correctly removed**: unwanted footage excluded from export (TN).',
             '- **Incorrectly removed**: wanted padded human export omitted (FNpad), including desired context.',
             '- **Incorrect export**: unwanted footage retained (FP).',
             '- **Missed core**: actual play omitted; this is the loss reflected by R_core.', '',
             'Export + correctly removed + incorrectly removed partitions evaluable video time. Incorrect export is a subset of export. Neither export duration alone nor model-minus-human duration measures error.', '',
             '## Production identity and scope limits', '',
             '“Production default” means the current checked-in browser/Android fresh-project default: aggressive whole-rally suppression. Saved projects can retain different settings. This study did not query the deployed website or an installed APK. The unsuppressed production union, balanced and conservative suppression options are listed separately.', '',
             'The adapter replayed the actual checked-in browser TypeScript runtime against audited feature caches; both shipped raw heads matched cached endpoints exactly. Browser and Android core weight assets match. This does not revalidate native feature extraction on devices. The separate app-fidelity table preserves materialized exports and suppression barriers, while primary ranking uses the repository’s canonical export contract.', '',
             'Shipped production weights have historical exposure to these videos. Source-group-held refit heads are a separate sensitivity comparison; their historical decoder/settings remain development-exposed and no refitted suppression head was introduced. Neural candidates and recipes are adaptive development choices. Rankings are descriptive on this scope, not an untouched estimate of generalization or permission to deploy.', '',
             '## Standalone models and production', '',
             duration_table([r for r in automatic if r['family'] in ('production','neural')]), '',
             '## Configuration key', '',
             table(['Key','Meaning'],[[f'`{k}`',v] for k,v in NAMES.items()]), '',
             'All combinations operate on raw core intervals before final padding/joining. Two-second tolerant support is dilation only, without short-gap joining. Protected-component policies preserve every original raw interval in a previous/v2 agreement component; agreement uses ±2 s and strictly <0.5 s joins. Source-held refit recipes use refit source tags. Matched neural seeds are paired, never all nine seed combinations.', '',
             '## Complete automatic matrix at target padding', '',
             'Sorted by mean development F1_padP_coreR at the predeclared ±2 s. All fixed candidates are retained, including aggressive veto controls and candidates failing coverage guardrails. The source-held refit rows and shipped-exposure rows are distinct evidence regimes.', '',
             duration_table(ranked), '',
             '## Coverage and event guardrails', '',
             'Loss counts are out of 322 rallies and can be fractional because of the seed average. Long means rally core longer than 3 s. Event F1 uses the evaluator’s raw-event IoU matching and is a secondary guardrail. “Worsened max” counts any lower retained rally-core duration than that candidate’s paired anchor in the worst seed, including partially cut rallies. Standalone neural/pair rows use production default as their paired anchor.', '',
             table(['Configuration','Event F1 %','Long R %','Complete losses','Partial losses','Short complete losses',
                    'Paired anchor','New complete max','Worsened max','Conservative screen'],
                   [[f"`{r['id']}`",percent(r['guardrailsMean']['eventF1']),percent(r['guardrailsMean']['longR_core']),
                     number(r['guardrailsMean']['completeLosses']),number(r['guardrailsMean']['partialLosses']),
                     number(r['guardrailsMean']['shortCompleteLosses']),r.get('pairedAgainst','—'),
                     r.get('pairedMaximum',{}).get('newCompleteLosses','—'),r.get('pairedMaximum',{}).get('worsenedRallies','—'),
                     ('PASS' if r['conservativeScreen']['passed'] else 'fail') if 'conservativeScreen' in r else '—'] for r in ranked]), '',
             'The screen requires at least +2 percentage points mean F1, recall/long-recall losses at most 0.5 points, event-F1 loss at most 1 point, and zero new complete or worsened rally coverage in every seed. It is stricter than F1 ranking and does not authorize deployment.', '',
             '## Review queues: unchanged automatic output', '',
             'Suppression review flags production export absent from the neural export. Bidirectional review also flags neural export missing from production. Automatic P/R/F1 and export durations remain those of the production anchor until a person edits the result. Queue playback includes ±2 s context around disputed spans and <3 s joining; context counts toward workload but not the hypothetical correction area. Clip counts are seed means. These are label-blind disagreement flags, not calibrated confidence probabilities.', '',
             table(['Policy','Review min','Clips','Disputed min','Incorrect export flagged min','FP capture %',
                    'Wanted export flagged min','Omitted human export flagged min','Missed core flagged s','Missed core capture %'],
                   [[f"`{r['id']}`",number(r['primary']['reviewSeconds']/60),number(r['primary']['reviewClips']),
                     number(r['primary']['disputedSeconds']/60),number(r['primary']['unwantedExportFlaggedSeconds']/60),
                     percent(r['primary']['unwantedExportCaptureFraction']),number(r['primary']['wantedExportFlaggedSeconds']/60),
                     number(r['primary']['missedHumanExportFlaggedSeconds']/60),number(r['primary']['missedCoreFlaggedSeconds']),
                     percent(r['primary']['missedCoreCaptureFraction'])] for r in reviews]), '',
             '## Perfect disputed-region review: optimistic upper bound only', '',
             'The following uses ground truth to correct only disputed export seconds. It is not an automatic model, not a measured human result, and is excluded from the automatic ranking. It cannot fix errors shared by both models. No padding or joining is applied again after the hypothetical edits.', '',
             duration_table([{'id':r['id'],'primary':r['primary']['oracle']} for r in reviews]), '',
             '## Exact app-export fidelity (separate from primary ranking)', '',
             table(['Production variant','Pad s','P_pad %','R_core %','F1_padP_coreR %','Export s','Human export s',
                    'Export minus human s','App-only s','Canonical-only s'],
                   [[f"`{r['variant']}`",pad,*[percent(at(r['actualAppDurationMetrics'],pad)[k]) for k in METRICS[:3]],
                     *[number(at(r['actualAppDurationMetrics'],pad)[k]) for k in METRICS[3:6]],
                     number(r['exportDifferences'][pad]['appOnlySeconds']),number(r['exportDifferences'][pad]['canonicalOnlySeconds'])]
                    for r in report['actualAppFidelity'] for pad in range(4)]), '',
             '## Required automatic padding sensitivity', '',
             'The remaining padding cases are sensitivities; no candidate chooses its own best padding. Duration difference is model export minus human export. JSON retains every other duration field, per-seed statistics, source-group results, paired loss identities, and all review/oracle padding cases.', '',
             table(['Configuration','Pad s','P_pad %','R_core %','F1_padP_coreR %','Model export s','Human export s','Difference s'],
                   [[f"`{r['id']}`",pad,*[percent(at(r['padding'],pad)[k]) for k in METRICS[:3]],
                     *[number(at(r['padding'],pad)[k]) for k in METRICS[3:6]]] for r in automatic for pad in range(4)]), '',
             '## Per-source-group sensitivity at target padding', '',
             'These group scores are diagnostics, not an average used to calculate the dataset ranking.', '',
             table(['Configuration','Source group','P_pad %','R_core %','F1_padP_coreR %','Export s','Incorrect export s','Missed core s'],
                   [[f"`{r['id']}`",g,*[percent(at(r['sourceGroups'][g])[k]) for k in METRICS[:3]],
                     *[number(at(r['sourceGroups'][g])[k]) for k in ('paddedModelExportSeconds','incorrectExportSeconds','missedCoreSeconds')]]
                    for r in automatic for g in groups]), '',
             '## Evidence', '',
             f"Contract SHA-256: `{reg['sha256']}`.", '',
             f"NAS run root: `{output}`.", '',
             f"Full machine-readable summary: `{target}` (SHA-256 `{sha(target)}`).",
             f"Full immutable result index: `{output/'report.json'}` (SHA-256 `{sha(output/'report.json')}`).", '',
             'Every result includes an independent endpoint-sweep duration check on pooled, four group and eight individual-recording scopes, at all four paddings. The separate recipe auditor reconstructs all 2,424 automatic recording outputs, 2,880 review recording/padding rows and 128 app-export overrides without shared interval/model/application imports. Qualification passed 44 focused tests. Registered source/input hashes were checked before and after execution. The final source archive and manifest are stored alongside the run.']
    document.parent.mkdir(parents=True,exist_ok=True)
    document.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'summary':identity(target),'document':identity(document),
                      'screenPassingIds':summary['conservativeScreenPassingIds'],
                      'topAutomaticIds':summary['rankedAutomaticIds'][:10]},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT)
    parser.add_argument('--document',type=Path,default=REPO/'docs/research/neural-production-combinations-results-2026-09-19.md')
    args=parser.parse_args()
    main(args.output,args.document)
